from __future__ import annotations

import pandas as pd


def summarize_feature_checks(feature_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, features in feature_frames.items():
        missing_values = int(features.isna().sum().sum())
        zero_variance = int((features.nunique() <= 1).sum())
        rows.extend(
            [
                {
                    "model": model_name,
                    "metric": "feature_missing_values",
                    "value": missing_values,
                    "status": "warn" if missing_values else "ok",
                    "detail": "",
                },
                {
                    "model": model_name,
                    "metric": "zero_variance_features",
                    "value": zero_variance,
                    "status": "warn" if zero_variance else "ok",
                    "detail": "",
                },
            ]
        )
    return pd.DataFrame(rows)


def summarize_prediction_checks(predictions: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, preds in predictions.items():
        min_pred = float(preds.min())
        max_pred = float(preds.max())
        std_pred = float(preds.std())
        unique_count = int(preds.nunique())
        range_status = "ok"
        if min_pred < 0 or max_pred > 1:
            range_status = "fail"
        elif std_pred < 0.01 or unique_count < 10:
            range_status = "warn"
        rows.append(
            {
                "model": model_name,
                "metric": "prediction_range",
                "value": f"{min_pred:.4f}-{max_pred:.4f}",
                "status": range_status,
                "detail": f"std={std_pred:.4f}, unique={unique_count}",
            }
        )
    return pd.DataFrame(rows)
