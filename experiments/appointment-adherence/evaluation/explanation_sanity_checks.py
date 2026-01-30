import pandas as pd


def paradox_rate(df: pd.DataFrame, shap_df: pd.DataFrame) -> float:
    mask = (df["appointment_success_rate"] >= 0.95) & (df["n_appointments"] <= 3)
    if mask.sum() == 0:
        return 0.0
    paradox = shap_df.loc[mask, "appointment_success_rate"] > 0
    return paradox.mean()


def summarize_sanity_checks(df: pd.DataFrame, shap_a: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["paradox_rate_high_success"],
            "value": [paradox_rate(df, shap_a)],
        }
    )
