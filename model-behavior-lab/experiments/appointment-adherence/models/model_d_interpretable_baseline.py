import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

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


def train_model(df: pd.DataFrame, target: str) -> ExplainableBoostingClassifier:
    X = prepare_features(df)
    y = df[target]
    model = ExplainableBoostingClassifier(random_state=7)
    model.fit(X, y)
    return model
