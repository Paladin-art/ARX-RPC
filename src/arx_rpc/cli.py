"""Small command-line interface for demonstration, training and forecasting."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import ModelConfig, demo_frame, load_frame
from .metrics import error_analysis, evaluate_windows
from .pipeline import ARXRPC


def main():
    parser = argparse.ArgumentParser(description="ARX-anchored residual-path correction")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate reproducible synthetic 15-minute data")
    demo.add_argument("--output", default="data/demo.csv")
    demo.add_argument("--days", type=int, default=42)
    demo.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train", help="Fit, select parameters, and evaluate fixed windows")
    train.add_argument("--config", default="config/demo.json")
    predict = sub.add_parser("predict", help="Forecast from a saved model without refitting")
    predict.add_argument("--model", required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--origin", required=True, help="First unobserved 15-minute timestamp")
    predict.add_argument("--output", default="outputs/forecast.csv")
    args = parser.parse_args()
    if args.command == "demo":
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        demo_frame(args.days, args.seed).to_csv(path, index=False)
        print(f"Synthetic demonstration data: {path}")
        return
    if args.command == "predict":
        model = ARXRPC.load(args.model)
        frame = load_frame(args.data, model.config.features)
        timestamp = pd.Timestamp(args.origin)
        origin = int(pd.DatetimeIndex(frame.timestamp).searchsorted(timestamp))
        expected = frame.timestamp.iloc[0] + pd.Timedelta(minutes=15 * origin)
        if timestamp != expected:
            raise ValueError("Origin must lie on the data grid or immediately after its end.")
        prediction, _ = model.predict(frame, [origin])
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_csv(output, index=False)
        print(f"Forecast saved: {output}")
        return
    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = ModelConfig(**raw.get("model", {}))
    frame = load_frame(raw["data"], cfg.features)
    validation_end = pd.Timestamp(raw["validation_end"])
    windows = raw["windows"]
    if not windows or any(pd.Timestamp(w["start"]) < validation_end for w in windows):
        raise ValueError("Every predefined evaluation window must follow validation.")
    end = max(pd.Timestamp(w["end"]) for w in windows)
    if end > frame.timestamp.iloc[-1] + pd.Timedelta(minutes=15):
        raise ValueError("Evaluation window extends beyond the observed data.")
    print("Fitting ARX, nonlinear residual learner, and TCN ...", flush=True)
    model = ARXRPC(cfg).fit(frame, raw["train_end"], raw["validation_end"])
    start_index = int(frame.timestamp.searchsorted(validation_end))
    stop_index = int(frame.timestamp.searchsorted(end))
    prediction, paths = model.predict(frame, np.arange(start_index, stop_index, cfg.predict_stride))
    prediction = prediction.loc[prediction.timestamp.lt(end)].reset_index(drop=True)
    report = evaluate_windows(prediction, windows, frame, cfg.tail_quantile)
    output = Path(raw.get("output", "outputs/demo"))
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.joblib")
    prediction.to_csv(output / "predictions.csv", index=False)
    paths.to_csv(output / "origin_predictions.csv", index=False)
    report.to_csv(output / "metrics.csv", index=False)
    error_analysis(prediction, frame).to_csv(output / "error_analysis.csv", index=False)
    (output / "selection.json").write_text(json.dumps(model.audit, indent=2, allow_nan=False), encoding="utf-8")
    print(report.to_string(index=False))
    print(f"Outputs saved: {output}")


if __name__ == "__main__":
    main()
