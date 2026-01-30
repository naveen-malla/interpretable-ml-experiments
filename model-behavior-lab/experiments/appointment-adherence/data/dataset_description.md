# Synthetic Appointment Dataset

This dataset models patient appointment history with engineered confounding to expose interpretability pitfalls.

## Target

- `follow_up_required`: 1 if the next appointment is likely to be missed, 0 otherwise.

## Raw Features

- `num_past_iits`: Number of prior no-shows.
- `n_appointments`: Number of prior appointments.
- `n_attended`: Number of prior attended appointments.
- `appointment_success_rate`: Ratio of attended appointments to total history.
- `prev_iit_status`: Whether the most recent appointment was missed.
- `second_last_iit_status`: Whether the second-most recent appointment was missed.
- `pct_late_arrivals`: Percent of attended visits that were late.
- `age_group`: Categorical age group.

## Derived Feature

- `risk_tier`:
  - High: `num_past_iits >= 3` OR `pct_late_arrivals >= 30`
  - Medium: `num_past_iits >= 1` OR `pct_late_arrivals >= 15`
  - Low: otherwise

## Confounding Pattern

Patients flagged as Medium or High risk receive implicit interventions (extra reminders). This improves adherence outcomes and can create paradoxical explanations when ratio features or the risk tier are used directly.
