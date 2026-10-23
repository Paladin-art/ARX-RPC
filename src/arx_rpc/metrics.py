"""Evaluation and analytical diagnostics, separate from prediction."""

import numpy as np
import pandas as pd


def metrics(actual, predicted, quantile=0.9):
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if y.shape != p.shape or not len(y) or not np.isfinite(y).all():
        raise ValueError("Metrics require aligned, nonempty and finite observed prices.")
    valid = np.isfinite(p)
    if not valid.any():
        raise ValueError("No finite predictions.")
    tail = (y >= np.quantile(y, quantile)) & valid
    peak_bias = float(p[valid].max() - y.max())
    return {"MAE": float(np.abs(y[valid] - p[valid]).mean()),
            "RMSE": float(np.sqrt(np.square(y[valid] - p[valid]).mean())),
            "ExtremeMAE": float(np.abs(y[tail] - p[tail]).mean()) if tail.any() else None,
            "PeakBias": peak_bias, "AbsPeakBias": abs(peak_bias), "Coverage": float(valid.mean())}


def aggregate(long_predictions):
    cols = ["arx", "tree_residual", "tcn_residual", "rpc"]
    result = long_predictions.groupby("timestamp", as_index=False)[cols].mean()
    counts = long_predictions.groupby("timestamp").size()
    result["forecast_count"] = result.timestamp.map(counts)
    return result


def evaluate_windows(predictions, windows, observed, quantile=0.9):
    frame = predictions.set_index("timestamp")
    truth = observed.set_index("timestamp").price
    rows = []
    for window in windows:
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        if start >= end:
            raise ValueError("Window start must precede its exclusive end.")
        expected = pd.date_range(start, end, freq="15min", inclusive="left")
        actual = truth.reindex(expected).to_numpy(float)
        for variant in ("arx", "rpc", "cs_rpc"):
            pred = frame[variant].reindex(expected).to_numpy(float)
            rows.append({"window": window["name"], "model": variant,
                         "n_targets": len(expected), **metrics(actual, pred, quantile)})
    return pd.DataFrame(rows)


def error_analysis(predictions, observed):
    frame = predictions.merge(observed[["timestamp", "price"]], on="timestamp", how="left")
    for name in ("rpc", "cs_rpc"):
        residual = frame.price - frame.arx
        correction = frame[name] - frame.arx
        frame[f"{name}_correction"] = correction
        frame[f"{name}_relative_magnitude"] = correction.abs() / residual.abs().where(residual.abs() > 1e-12)
        frame[f"{name}_aligned"] = residual * correction > 0
        frame[f"{name}_squared_error_change"] = (frame.price - frame[name]) ** 2 - residual ** 2
        frame[f"{name}_analytic_change"] = correction ** 2 - 2 * residual * correction
    return frame
