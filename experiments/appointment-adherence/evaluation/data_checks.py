from __future__ import annotations

from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "appointment_index",
    "num_past_iits",
    "n_appointments",
    "n_attended",
    "appointment_success_rate",
    "prev_iit_status",
    "second_last_iit_status",
    "pct_late_arrivals",
    "age_group",
    "risk_tier",
    "follow_up_required",
)

ALLOWED_RISK_TIERS = {"Low", "Medium", "High"}
ALLOWED_AGE_GROUPS = {"18-34", "35-49", "50-64", "65+"}


def _format_list(values: Iterable[str]) -> str:
    return ", ".join(sorted(values)) if values else ""


def summarize_data_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    checks.append(
        {
            "metric": "row_count",
            "value": len(df),
            "status": "ok" if len(df) > 0 else "fail",
            "detail": "",
        }
    )
    checks.append(
        {
            "metric": "unique_patients",
            "value": df["patient_id"].nunique() if "patient_id" in df else 0,
            "status": "ok" if "patient_id" in df else "fail",
            "detail": "",
        }
    )
    if "follow_up_required" in df:
        checks.append(
            {
                "metric": "no_show_rate",
                "value": df["follow_up_required"].mean(),
                "status": "ok",
                "detail": "",
            }
        )

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    checks.append(
        {
            "metric": "missing_required_columns",
            "value": len(missing_columns),
            "status": "fail" if missing_columns else "ok",
            "detail": _format_list(missing_columns),
        }
    )

    missing_values = df.isna().sum()
    missing_total = int(missing_values.sum())
    missing_columns_with_values = missing_values[missing_values > 0]
    checks.append(
        {
            "metric": "missing_values_total",
            "value": missing_total,
            "status": "warn" if missing_total else "ok",
            "detail": _format_list(missing_columns_with_values.index.tolist()),
        }
    )

    if {"patient_id", "appointment_index"}.issubset(df.columns):
        duplicate_count = int(df.duplicated(subset=["patient_id", "appointment_index"]).sum())
        checks.append(
            {
                "metric": "duplicate_patient_visits",
                "value": duplicate_count,
                "status": "warn" if duplicate_count else "ok",
                "detail": "",
            }
        )

    if "appointment_success_rate" in df:
        out_of_range = int((df["appointment_success_rate"].lt(0) | df["appointment_success_rate"].gt(1)).sum())
        checks.append(
            {
                "metric": "appointment_success_rate_out_of_range",
                "value": out_of_range,
                "status": "warn" if out_of_range else "ok",
                "detail": "",
            }
        )

    if "pct_late_arrivals" in df:
        out_of_range = int((df["pct_late_arrivals"].lt(0) | df["pct_late_arrivals"].gt(100)).sum())
        checks.append(
            {
                "metric": "pct_late_arrivals_out_of_range",
                "value": out_of_range,
                "status": "warn" if out_of_range else "ok",
                "detail": "",
            }
        )

    if {"n_attended", "n_appointments"}.issubset(df.columns):
        invalid = int((df["n_attended"] > df["n_appointments"]).sum())
        checks.append(
            {
                "metric": "attended_exceeds_appointments",
                "value": invalid,
                "status": "warn" if invalid else "ok",
                "detail": "",
            }
        )

    if {"num_past_iits", "n_appointments"}.issubset(df.columns):
        invalid = int((df["num_past_iits"] > df["n_appointments"]).sum())
        checks.append(
            {
                "metric": "past_no_shows_exceed_appointments",
                "value": invalid,
                "status": "warn" if invalid else "ok",
                "detail": "",
            }
        )

    if "appointment_index" in df:
        invalid = int((df["appointment_index"] < 1).sum())
        checks.append(
            {
                "metric": "appointment_index_below_one",
                "value": invalid,
                "status": "warn" if invalid else "ok",
                "detail": "",
            }
        )

    if "follow_up_required" in df:
        invalid = int((~df["follow_up_required"].isin([0, 1])).sum())
        checks.append(
            {
                "metric": "follow_up_required_invalid",
                "value": invalid,
                "status": "warn" if invalid else "ok",
                "detail": "",
            }
        )

    if "risk_tier" in df:
        invalid = int((~df["risk_tier"].isin(ALLOWED_RISK_TIERS)).sum())
        checks.append(
            {
                "metric": "risk_tier_invalid",
                "value": invalid,
                "status": "warn" if invalid else "ok",
                "detail": _format_list(set(df["risk_tier"]) - ALLOWED_RISK_TIERS),
            }
        )

    if "age_group" in df:
        invalid = int((~df["age_group"].isin(ALLOWED_AGE_GROUPS)).sum())
        checks.append(
            {
                "metric": "age_group_invalid",
                "value": invalid,
                "status": "warn" if invalid else "ok",
                "detail": _format_list(set(df["age_group"]) - ALLOWED_AGE_GROUPS),
            }
        )

    return pd.DataFrame(checks)
