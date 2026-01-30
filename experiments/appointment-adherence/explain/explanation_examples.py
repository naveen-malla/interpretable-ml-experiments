import pandas as pd


def find_counter_intuitive_cases(df: pd.DataFrame) -> pd.DataFrame:
    candidates = df[
        (df["appointment_success_rate"] >= 0.95)
        & (df["n_appointments"] <= 3)
        & (df["pred_model_a"] >= 0.7)
    ]
    return candidates.sort_values("pred_model_a", ascending=False).head(10)


def find_disagreement_cases(df: pd.DataFrame) -> pd.DataFrame:
    disagreements = df[
        (df["pred_model_a"] >= 0.7)
        & (df["pred_model_b"] <= 0.3)
    ]
    return disagreements.sort_values("pred_model_a", ascending=False).head(10)
