from pathlib import Path

import pandas as pd

from data.generate_synthetic import generate_synthetic_dataset
from evaluation.comparison_tables import build_model_summary
from evaluation.data_checks import summarize_data_checks
from evaluation.explanation_sanity_checks import summarize_sanity_checks
from evaluation.logic_checks import summarize_logic_checks
from evaluation.model_checks import summarize_feature_checks, summarize_prediction_checks
from explain.explanation_examples import find_counter_intuitive_cases, find_disagreement_cases
from explain.narrative_generation import build_narratives
from explain.shap_analysis import compute_ebm_explanations, compute_tree_shap
from models.model_a_xgb_ratio_risktier import prepare_features as prep_a
from models.model_a_xgb_ratio_risktier import train_model as train_a
from models.model_b_xgb_counts_only import prepare_features as prep_b
from models.model_b_xgb_counts_only import train_model as train_b
from models.model_c_xgb_monotonic import prepare_features as prep_c
from models.model_c_xgb_monotonic import train_model as train_c
from models.model_d_interpretable_baseline import prepare_features as prep_d
from models.model_d_interpretable_baseline import train_model as train_d


def _output_dir() -> Path:
    output = Path("experiments/appointment-adherence/outputs")
    output.mkdir(parents=True, exist_ok=True)
    return output


def main() -> None:
    data_path = Path("experiments/appointment-adherence/data/synthetic_appointments.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = generate_synthetic_dataset()
        df.to_csv(data_path, index=False)

    output_dir = _output_dir()
    summarize_data_checks(df).to_csv(output_dir / "data_checks.csv", index=False)

    model_a = train_a(df, "follow_up_required")
    model_b = train_b(df, "follow_up_required")
    model_c = train_c(df, "follow_up_required")
    model_d = train_d(df, "follow_up_required")

    X_a = prep_a(df)
    X_b = prep_b(df)
    X_c = prep_c(df)
    X_d = prep_d(df)

    preds = {
        "model_a": pd.Series(model_a.predict_proba(X_a)[:, 1], index=df.index),
        "model_b": pd.Series(model_b.predict_proba(X_b)[:, 1], index=df.index),
        "model_c": pd.Series(model_c.predict_proba(X_c)[:, 1], index=df.index),
        "model_d": pd.Series(model_d.predict_proba(X_d)[:, 1], index=df.index),
    }

    df_predictions = df.copy()
    for key, series in preds.items():
        df_predictions[f"pred_{key}"] = series

    shap_a = compute_tree_shap(model_a, X_a)
    shap_b = compute_tree_shap(model_b, X_b)
    shap_c = compute_tree_shap(model_c, X_c)
    shap_d = compute_ebm_explanations(model_d, X_d)

    counter_intuitive = find_counter_intuitive_cases(df_predictions)
    disagreement = find_disagreement_cases(df_predictions)

    narratives_a = build_narratives(counter_intuitive, shap_a.loc[counter_intuitive.index], "model_a")
    narratives_b = build_narratives(disagreement, shap_b.loc[disagreement.index], "model_b")

    build_model_summary(preds).to_csv(output_dir / "model_summary.csv", index=False)
    summarize_feature_checks(
        {"model_a": X_a, "model_b": X_b, "model_c": X_c, "model_d": X_d}
    ).to_csv(output_dir / "feature_checks.csv", index=False)
    summarize_prediction_checks(preds).to_csv(output_dir / "prediction_checks.csv", index=False)
    summarize_logic_checks(df, preds).to_csv(output_dir / "logic_checks.csv", index=False)
    summarize_sanity_checks(df, shap_a).to_csv(output_dir / "sanity_checks.csv", index=False)
    counter_intuitive.to_csv(output_dir / "counter_intuitive_cases.csv", index=False)
    disagreement.to_csv(output_dir / "model_disagreements.csv", index=False)
    pd.concat([narratives_a, narratives_b], ignore_index=True).to_csv(
        output_dir / "narratives.csv", index=False
    )

    print("Experiment complete. Outputs saved to", output_dir)


if __name__ == "__main__":
    main()
