import pandas as pd
import shap


def compute_tree_shap(model, X: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return pd.DataFrame(shap_values, columns=X.columns)


def compute_ebm_explanations(model, X: pd.DataFrame) -> pd.DataFrame:
    explanation = model.explain_local(X, model.predict_proba(X)[:, 1])
    contributions = []
    for i in range(len(X)):
        row_contrib = dict(zip(explanation.data(i)["names"], explanation.data(i)["scores"]))
        contributions.append(row_contrib)
    return pd.DataFrame(contributions).fillna(0.0)
