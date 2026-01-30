import numpy as np
import pandas as pd


AGE_GROUPS = ["18-34", "35-49", "50-64", "65+"]


def _age_risk(age_group: str) -> float:
    return {
        "18-34": 0.15,
        "35-49": 0.1,
        "50-64": 0.05,
        "65+": 0.08,
    }[age_group]


def generate_synthetic_dataset(
    n_patients: int = 2500,
    min_appointments: int = 3,
    max_appointments: int = 8,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate synthetic appointment histories with intervention effects."""
    rng = np.random.default_rng(seed)
    rows = []

    for patient_id in range(n_patients):
        n_visits = rng.integers(min_appointments, max_appointments + 1)
        age_group = rng.choice(AGE_GROUPS, p=[0.32, 0.28, 0.22, 0.18])
        base_risk = _age_risk(age_group)

        past_no_shows = 0
        past_attended = 0
        past_late = 0
        past_outcomes = []

        for visit_idx in range(n_visits):
            if visit_idx == 0:
                outcome_no_show = rng.random() < (base_risk + 0.05)
                if not outcome_no_show:
                    late = rng.random() < 0.18
                    past_late += int(late)
                    past_attended += 1
                else:
                    past_no_shows += 1
                past_outcomes.append(int(outcome_no_show))
                continue

            n_appointments = visit_idx
            appointment_success_rate = past_attended / n_appointments
            pct_late_arrivals = 100 * (past_late / max(past_attended, 1))
            prev_iit_status = past_outcomes[-1]
            second_last_iit_status = past_outcomes[-2] if len(past_outcomes) > 1 else 0

            rule_high = (past_no_shows >= 3) or (pct_late_arrivals >= 30)
            rule_medium = (past_no_shows >= 1) or (pct_late_arrivals >= 15)
            risk_tier = "High" if rule_high else "Medium" if rule_medium else "Low"

            intervention_effect = -0.08 if risk_tier in {"High", "Medium"} else 0.0
            ratio_paradox = 0.12 if (appointment_success_rate > 0.95 and n_appointments <= 3) else 0.0

            no_show_logit = (
                base_risk
                + 0.08 * past_no_shows
                + 0.04 * prev_iit_status
                + 0.02 * second_last_iit_status
                - 0.05 * past_attended
                + 0.02 * (pct_late_arrivals / 10)
                + ratio_paradox
                + intervention_effect
            )
            no_show_prob = 1 / (1 + np.exp(-no_show_logit))
            outcome_no_show = rng.random() < no_show_prob

            if not outcome_no_show:
                late = rng.random() < (0.15 + 0.02 * past_no_shows)
                past_late += int(late)
                past_attended += 1
            else:
                past_no_shows += 1
            past_outcomes.append(int(outcome_no_show))

            rows.append(
                {
                    "patient_id": patient_id,
                    "appointment_index": visit_idx,
                    "num_past_iits": past_no_shows,
                    "n_appointments": n_appointments,
                    "n_attended": past_attended,
                    "appointment_success_rate": appointment_success_rate,
                    "prev_iit_status": prev_iit_status,
                    "second_last_iit_status": second_last_iit_status,
                    "pct_late_arrivals": pct_late_arrivals,
                    "age_group": age_group,
                    "risk_tier": risk_tier,
                    "follow_up_required": int(outcome_no_show),
                }
            )

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    df = generate_synthetic_dataset()
    output_path = "experiments/appointment-adherence/data/synthetic_appointments.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
