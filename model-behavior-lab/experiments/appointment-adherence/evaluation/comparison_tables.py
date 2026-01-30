import pandas as pd


def build_model_summary(predictions: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for model_name, preds in predictions.items():
        rows.append(
            {
                "model": model_name,
                "mean_predicted_risk": preds.mean(),
                "p90_predicted_risk": preds.quantile(0.9),
            }
        )
    return pd.DataFrame(rows)
