# Appointment Adherence Experiment

This experiment demonstrates how feature design and modeling choices affect interpretability in appointment adherence predictions. It intentionally creates explanation paradoxes, such as cases where a perfect appointment record appears to increase no-show risk, and then shows how these disappear under safer feature designs.

## Key Concepts

- **Ratio features without denominators** can create misleading signals.
- **Rule-based risk tiers** reused as model inputs can dominate explanations.
- **Implicit interventions** (extra reminders for risky patients) can flip the sign of risk indicators.

## Running the Experiment

```bash
python run_experiment.py
```

Outputs are written to the `outputs/` directory, including explanation tables and narrative examples.
