import pandas as pd

RAW_FEATURES = [
    "num_past_iits",
    "n_appointments",
    "n_attended",
    "appointment_success_rate",
    "prev_iit_status",
    "second_last_iit_status",
    "pct_late_arrivals",
    "age_group",
]


def build_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[RAW_FEATURES].copy()
