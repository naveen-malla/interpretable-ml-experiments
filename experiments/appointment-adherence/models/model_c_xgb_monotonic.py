import pandas as pd
from xgboost import XGBClassifier

FEATURES = [
    "num_past_iits",
    "n_appointments",
    "n_attended",
    "prev_iit_status",
    "second_last_iit_status",
    "age_group",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=["age_group"], drop_first=False)
    return df[[col for col in df.columns if col in FEATURES or col.startswith("age_group_")]]


def _monotone_constraints(feature_names: list[str]) -> tuple[int, ...]:
    constraints = []
    for name in feature_names:
        if name in {"num_past_iits", "prev_iit_status", "second_last_iit_status"}:
            constraints.append(1)
        elif name == "n_attended":
            constraints.append(-1)
        else:
            constraints.append(0)
    return tuple(constraints)


def train_model(df: pd.DataFrame, target: str) -> XGBClassifier:
    X = prepare_features(df)
    y = df[target]
    model = XGBClassifier(
        n_estimators=240,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=7,
        monotone_constraints=_monotone_constraints(list(X.columns)),
    )
    model.fit(X, y)
    return model
