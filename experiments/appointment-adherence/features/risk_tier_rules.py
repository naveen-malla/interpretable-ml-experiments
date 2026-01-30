from typing import Literal

import pandas as pd

RiskTier = Literal["Low", "Medium", "High"]


def assign_risk_tier(num_past_iits: int, pct_late_arrivals: float) -> RiskTier:
    if (num_past_iits >= 3) or (pct_late_arrivals >= 30):
        return "High"
    if (num_past_iits >= 1) or (pct_late_arrivals >= 15):
        return "Medium"
    return "Low"


def add_risk_tier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_tier"] = df.apply(
        lambda row: assign_risk_tier(row["num_past_iits"], row["pct_late_arrivals"]), axis=1
    )
    return df
