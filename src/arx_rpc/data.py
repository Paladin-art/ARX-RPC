"""Chronological data access. An origin is the first unobserved timestamp."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class ModelConfig:
    horizon: int = 96
    lookback: int = 288
    lags: list[int] = field(default_factory=lambda: [1, 4, 16, 96, 192, 288])
    rolling_windows: list[int] = field(default_factory=lambda: [16, 96, 288])
    features: list[str] = field(default_factory=lambda: ["load_forecast", "wind_forecast", "temperature_forecast"])
    ridge_alpha: float = 10.0
    tree_iterations: int = 60
    channels: int = 8
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    train_stride: int = 16
    predict_stride: int = 4
    tail_quantile: float = 0.9
    tail_weight: float = 3.0
    seed: int = 42

    @property
    def warmup(self):
        return max(self.lags + self.rolling_windows)

    def validate(self):
        sizes = [self.horizon, self.lookback, self.epochs, self.batch_size,
                 self.train_stride, self.predict_stride, self.tree_iterations,
                 self.channels, *self.lags, *self.rolling_windows, *self.dilations]
        if not self.lags or not self.rolling_windows or not self.dilations or any(x < 1 for x in sizes):
            raise ValueError("Window sizes, lags, training counts and strides must be positive.")
        if not 0 < self.tail_quantile < 1 or self.tail_weight < 1:
            raise ValueError("Require 0 < tail_quantile < 1 and tail_weight >= 1.")
        if self.predict_stride > self.horizon:
            raise ValueError("predict_stride must not exceed horizon, otherwise coverage has gaps.")
        if self.ridge_alpha < 0 or self.learning_rate <= 0 or len(set(self.features)) != len(self.features):
            raise ValueError("Invalid regularization, learning rate or duplicate feature names.")


def load_frame(path, features):
    frame = pd.read_csv(path)
    required = ["timestamp", "price", *features]
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    frame = frame[required].copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, errors="raise")
    if frame.timestamp.dt.tz is not None:
        raise ValueError("Use timestamps in one local market timezone, without timezone offsets.")
    if not frame.timestamp.is_monotonic_increasing or frame.timestamp.duplicated().any():
        raise ValueError("Timestamps must be strictly increasing and unique.")
    if len(frame) < 3 or not frame.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Expected a complete, regular 15-minute grid.")
    for col in required[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if np.isinf(frame[col].to_numpy(float)).any():
            raise ValueError(f"Infinite values in {col}.")
    return frame


class FeatureBank:
    def __init__(self, frame, config):
        self.frame, self.config = frame, config
        self.time = pd.DatetimeIndex(frame.timestamp)
        self.y = frame.price.to_numpy(float)
        self.exog = frame[config.features].to_numpy(float)
        self.n = len(frame)
        self.base = np.full((self.n + 1, len(config.lags) + 2 * len(config.rolling_windows) + len(config.features)), np.nan)
        for origin in range(config.warmup, self.n + 1):
            values = [self.y[origin - lag] for lag in config.lags]
            for window in config.rolling_windows:
                history = self.y[origin - window:origin]
                values.extend([history.mean(), history.std()])
            values.extend(self.exog[origin - 1])
            self.base[origin] = values

    def times(self, indices):
        return self.time[0] + pd.to_timedelta(np.asarray(indices) * 15, unit="min")

    def matrix(self, origins, horizon):
        origins = np.asarray(origins, dtype=int)
        targets = origins[:, None] + np.arange(horizon)[None, :]
        dates = self.times(targets.ravel())
        minute = dates.hour * 60 + dates.minute
        calendar = np.column_stack([np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440),
                                    np.sin(2 * np.pi * dates.dayofweek / 7), np.cos(2 * np.pi * dates.dayofweek / 7),
                                    np.tile(np.arange(horizon) / self.config.horizon, len(origins))])
        return np.column_stack([np.repeat(self.base[origins], horizon, axis=0), calendar])

    def phase_origins(self, start, end, stride):
        # Every training/validation label must belong to its own phase.
        start = max(start, self.config.warmup + self.config.lookback)
        return np.arange(start, end - self.config.horizon + 1, stride, dtype=int)

    def targets(self, origins):
        indices = np.asarray(origins)[:, None] + np.arange(self.config.horizon)
        return self.y[indices]


def demo_frame(days=42, seed=42):
    """Generate artificial prices and forecast covariates, never market records."""
    if days < 8:
        raise ValueError("The demo needs at least eight days.")
    rng = np.random.default_rng(seed)
    time = pd.date_range("2025-01-01", periods=days * 96, freq="15min")
    step = np.arange(len(time))
    daily = 2 * np.pi * step / 96
    load = 90 + 18 * np.sin(daily - 1) + 5 * np.sin(daily / 7) + rng.normal(0, 2, len(time))
    wind = 35 + 13 * np.cos(daily / 3) + rng.normal(0, 2, len(time))
    temperature = 10 + 8 * np.sin(daily - 2)
    tension = np.maximum(load - wind - 65, 0)
    driver = 80 + 1.6 * load - 0.9 * wind + 0.15 * tension ** 2
    volatility = 4 + 12 * (step >= int(0.75 * len(time)))
    residual = np.zeros(len(time))
    for t in range(1, len(time)):
        residual[t] = 0.82 * residual[t - 1] + rng.normal(0, volatility[t])
    return pd.DataFrame({"timestamp": time, "price": driver + residual,
                         "load_forecast": load, "wind_forecast": wind, "temperature_forecast": temperature})
