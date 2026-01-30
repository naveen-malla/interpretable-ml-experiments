import pandas as pd

from .risk_tier_rules import add_risk_tier


def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_risk_tier(df)
