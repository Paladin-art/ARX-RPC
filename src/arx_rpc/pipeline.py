"""Fit on past labels, select on validation labels, freeze before evaluation."""

from dataclasses import asdict
import itertools
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from .data import FeatureBank, ModelConfig
from .metrics import aggregate
from .models import ConservativeGate, ResidualTCN, weighted_mae


class ARXRPC:
    def __init__(self, config=None):
        self.config = config or ModelConfig()
        self.config.validate()

    def _reference(self, matrix):
        x = self.preprocessor.transform(matrix)
        return self.anchor.predict(x), self.tree.predict(x)

    def _history(self, bank):
        origins = np.arange(self.config.warmup, bank.n)
        anchor, tree = self._reference(bank.matrix(origins, 1))
        residual = np.full(bank.n, np.nan)
        residual[origins] = bank.y[origins] - anchor - tree
        return np.column_stack([residual, bank.exog])

    def _sequences(self, history, origins):
        scaled = self.history_preprocessor.transform(history)
        return np.asarray([scaled[o - self.config.lookback:o] for o in origins], dtype=np.float32)

    def _components(self, bank, history, origins):
        cfg = self.config
        origins = np.asarray(origins, dtype=int)
        if not len(origins):
            raise ValueError("No valid forecast origins.")
        if origins.min() < cfg.warmup + cfg.lookback or origins.max() > bank.n:
            raise ValueError("Insufficient history or forecast origin beyond the available grid.")
        for origin in origins:
            known_prices = bank.y[origin - cfg.lookback - cfg.warmup:origin]
            if not np.isfinite(known_prices).all():
                raise ValueError("Observed price history before each origin must be finite.")
        anchor, tree = self._reference(bank.matrix(origins, cfg.horizon))
        self.temporal.eval()
        tensors = torch.from_numpy(self._sequences(history, origins))
        chunks = []
        with torch.no_grad():
            for chunk in tensors.split(cfg.batch_size):
                chunks.append(self.temporal(chunk).numpy())
        temporal = np.concatenate(chunks).ravel() * self.residual_std + self.residual_mean
        target_indices = (origins[:, None] + np.arange(cfg.horizon)).ravel()
        return pd.DataFrame({"origin_time": np.repeat(bank.times(origins), cfg.horizon),
                             "timestamp": bank.times(target_indices),
                             "horizon_step": np.tile(np.arange(1, cfg.horizon + 1), len(origins)),
                             "arx": anchor, "tree_residual": tree, "tcn_residual": temporal})

    def fit(self, frame, train_end, validation_end):
        cfg = self.config
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(1)
        bank = FeatureBank(frame, cfg)
        train_stop = int(bank.time.searchsorted(pd.Timestamp(train_end)))
        val_stop = int(bank.time.searchsorted(pd.Timestamp(validation_end)))
        if not 0 < train_stop < val_stop <= bank.n:
            raise ValueError("Require chronological train_end < validation_end within the data grid.")
        if not np.isfinite(bank.y[:val_stop]).all():
            raise ValueError("Training and validation prices must be observed and finite.")
        if len(cfg.features) and np.isnan(bank.exog[:train_stop]).all(axis=0).any():
            raise ValueError("A forecast covariate is entirely missing during training.")
        train_origins = bank.phase_origins(0, train_stop, cfg.train_stride)
        val_origins = bank.phase_origins(train_stop, val_stop, cfg.predict_stride)
        if len(train_origins) < 2 or len(val_origins) < 1:
            raise ValueError("Insufficient complete training or validation forecast paths.")
        x = bank.matrix(train_origins, cfg.horizon)
        y = bank.targets(train_origins).ravel()
        self.preprocessor = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
        xs = self.preprocessor.fit_transform(x)
        self.anchor = Ridge(alpha=cfg.ridge_alpha).fit(xs, y)
        self.tree = HistGradientBoostingRegressor(max_iter=cfg.tree_iterations, max_leaf_nodes=15,
                                                 learning_rate=0.05, l2_regularization=1.0,
                                                 early_stopping=False, random_state=cfg.seed)
        self.tree.fit(xs, y - self.anchor.predict(xs))
        history = self._history(bank)
        training_history = history[cfg.warmup:train_stop]
        self.history_preprocessor = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
        self.history_preprocessor.fit(training_history)
        self.residual_mean = float(training_history[:, 0].mean())
        self.residual_std = max(float(training_history[:, 0].std()), 1e-8)
        self.temporal = ResidualTCN(history.shape[1], cfg.channels, cfg.horizon, cfg.dilations)
        optimizer = torch.optim.Adam(self.temporal.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

        def tensors(origins):
            anchor, tree = self._reference(bank.matrix(origins, cfg.horizon))
            actual = bank.targets(origins)
            target = (actual - (anchor + tree).reshape(actual.shape) - self.residual_mean) / self.residual_std
            tail = actual >= np.quantile(bank.y[:train_stop], cfg.tail_quantile)
            weight = np.where(tail, cfg.tail_weight, 1.0)
            return tuple(torch.as_tensor(a, dtype=torch.float32) for a in
                         (self._sequences(history, origins), target, weight))

        train_tensors, validation_tensors = tensors(train_origins), tensors(val_origins)
        loader = DataLoader(TensorDataset(*train_tensors), batch_size=cfg.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(cfg.seed))
        best_loss, best_state = float("inf"), None
        training_log = []

        def loss(predicted, target, weight):
            point_loss = torch.nn.functional.smooth_l1_loss(predicted, target, reduction="none")
            return (point_loss * weight).sum() / weight.sum()

        for epoch in range(cfg.epochs):
            self.temporal.train()
            for xb, yb, wb in loader:
                optimizer.zero_grad()
                objective = loss(self.temporal(xb), yb, wb)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(self.temporal.parameters(), 1.0)
                optimizer.step()
            self.temporal.eval()
            with torch.no_grad():
                xv, yv, wv = validation_tensors
                value = float(loss(self.temporal(xv), yv, wv))
            training_log.append({"epoch": epoch + 1, "validation_loss": value})
            if value < best_loss:
                best_loss = value
                best_state = {k: v.detach().clone() for k, v in self.temporal.state_dict().items()}
        if best_state is None:
            raise RuntimeError("TCN optimization did not produce a finite validation loss.")
        self.temporal.load_state_dict(best_state)
        components = self._components(bank, history, val_origins)
        components["rpc"] = components.arx
        validation = aggregate(components)
        truth = frame.set_index("timestamp").price.reindex(validation.timestamp).to_numpy(float)
        best = None
        for alpha_tree, alpha_tcn in itertools.product([0.0, 0.5, 1.0, 1.5], [0.0, 0.5, 1.0]):
            prediction = validation.arx.to_numpy() + alpha_tree * validation.tree_residual.to_numpy() + alpha_tcn * validation.tcn_residual.to_numpy()
            score = weighted_mae(truth, prediction, cfg.tail_quantile, cfg.tail_weight)
            if best is None or score < best:
                best = score
                self.alpha_tree, self.alpha_tcn = alpha_tree, alpha_tcn
        validation["rpc"] = validation.arx + self.alpha_tree * validation.tree_residual + self.alpha_tcn * validation.tcn_residual
        self.gate = ConservativeGate().fit(truth, validation.arx.to_numpy(), validation.rpc.to_numpy(), cfg.tail_quantile)
        self.audit = {"config": asdict(cfg), "train_end_exclusive": str(pd.Timestamp(train_end)),
                      "validation_end_exclusive": str(pd.Timestamp(validation_end)),
                      "training_origins": len(train_origins), "validation_origins": len(val_origins),
                      "training_last_target": str(bank.times([train_origins[-1] + cfg.horizon - 1])[0]),
                      "validation_last_target": str(validation.timestamp.max()),
                      "alpha_tree": self.alpha_tree, "alpha_tcn": self.alpha_tcn,
                      "gate": vars(self.gate).copy(), "training_log": training_log}
        return self

    def predict(self, frame, origins):
        """Return aggregated and origin-specific paths; no fitting or target access."""
        bank = FeatureBank(frame, self.config)
        history = self._history(bank)
        long = self._components(bank, history, origins)
        long["rpc"] = long.arx + self.alpha_tree * long.tree_residual + self.alpha_tcn * long.tcn_residual
        prediction = aggregate(long)
        prediction["cs_rpc"], prediction["gate_lambda"] = self.gate.predict(prediction.arx.to_numpy(), prediction.rpc.to_numpy())
        return prediction, long

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError("Not an ARX-RPC model checkpoint.")
        return model
