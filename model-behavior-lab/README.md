# Model Behavior Lab

This repository contains controlled experiments that study how feature design and modeling choices shape interpretability, explanation stability, and clinician trust in decision-support settings. The experiments are intentionally scoped to research workflows and are **not** intended for production use.

## Repository Structure

```
model-behavior-lab/
├── README.md
├── experiments/
│   └── appointment-adherence/
│       ├── README.md
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── explain/
│       ├── evaluation/
│       └── run_experiment.py
├── notes/
└── requirements.txt
```

## Experiments

- **appointment-adherence**: A synthetic study of appointment adherence that demonstrates how ratio features, derived risk tiers, and implicit interventions can create misleading explanations.

## Documentation

- `DESIGN_DOCUMENT.md` documents the design principles behind the experiments and data generation choices.
- `DECISIONS.md` records major research and engineering decisions for reproducibility.
- `notes/` contains deep-dive writeups on explanation failure modes, feature design principles, and clinician trust guidance.

## Getting Started

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the experiment:
   ```bash
   python experiments/appointment-adherence/run_experiment.py
   ```

## License

This is a research repository intended for internal experimentation.
