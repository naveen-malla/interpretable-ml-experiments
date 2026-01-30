from __future__ import annotations

import pandas as pd


def summarize_logic_checks(
    df: pd.DataFrame,
    predictions: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if "risk_tier" in df and "model_a" in predictions:
        tier_order = ["Low", "Medium", "High"]
        tier_means = (
            pd.DataFrame({"risk_tier": df["risk_tier"], "pred": predictions["model_a"]})
            .groupby("risk_tier")
            .mean(numeric_only=True)
            .reindex(tier_order)
        )
        monotonic = tier_means["pred"].is_monotonic_increasing
        rows.append(
            {
                "check": "risk_tier_monotonicity_model_a",
                "value": tier_means["pred"].round(4).to_dict(),
                "status": "ok" if monotonic else "warn",
                "detail": "",
            }
        )

    if "num_past_iits" in df:
        for model_name, preds in predictions.items():
            corr = float(pd.Series(df["num_past_iits"]).corr(preds, method="spearman"))
            status = "ok" if corr >= 0 else "warn"
            rows.append(
                {
                    "check": "past_no_show_spearman",
                    "model": model_name,
                    "value": round(corr, 4),
                    "status": status,
                    "detail": "expected_non_negative",
                }
            )

    return pd.DataFrame(rows)
