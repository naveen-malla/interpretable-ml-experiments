import pandas as pd
from xgboost import XGBClassifier

FEATURES = ["appointment_success_rate", "risk_tier"]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_tier"] = df["risk_tier"].map({"Low": 0, "Medium": 1, "High": 2})
    return df[FEATURES]


def train_model(df: pd.DataFrame, target: str) -> XGBClassifier:
    X = prepare_features(df)
    y = df[target]
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=7,
    )
    model.fit(X, y)
    return model
