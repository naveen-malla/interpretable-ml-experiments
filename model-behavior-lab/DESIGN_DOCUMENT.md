# Design Document

## Purpose

The Model Behavior Lab is designed to make interpretability failure modes reproducible and inspectable. Each experiment is a self-contained artifact that can be re-run end-to-end with a single command. The primary goal is to study explanation stability, clinician plausibility, and the risk of misleading narratives when feature engineering choices encode hidden interventions.

## Design Principles

1. **Isolation by experiment**
   - Each experiment lives in its own folder under `experiments/`.
   - Experiments are not coupled; there is no shared pipeline logic.

2. **Reproducibility**
   - Every experiment has a `run_experiment.py` script that generates data, trains models, and writes outputs.
   - Synthetic data generation is deterministic under a fixed seed.

3. **Interpretability-first evaluation**
   - Models are compared based on explanation behavior and narrative plausibility rather than headline accuracy.
   - Outputs include explanation tables, counter-intuitive examples, and qualitative notes.

4. **Clarity over cleverness**
   - Code is modular and readable.
   - Features are explicitly defined and named to match the research questions.

## Data Generation Strategy

The synthetic data pipeline for each experiment intentionally encodes real-world confounding patterns such as implicit intervention effects. This allows the experiment to demonstrate how models can learn counter-intuitive relationships when ratio features or proxy rules are used without their denominators.

## Output Expectations

Experiments should emit:
- Comparison tables of explanation behavior.
- Example narratives for clinician review.
- Written notes on feature design risks and safer modeling choices.
