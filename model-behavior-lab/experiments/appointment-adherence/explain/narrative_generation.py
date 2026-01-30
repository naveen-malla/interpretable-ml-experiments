import pandas as pd


def narrative_from_contributions(row: pd.Series, top_n: int = 3) -> str:
    contributions = row.dropna().sort_values(key=abs, ascending=False)
    top = contributions.head(top_n)
    parts = []
    for feature, value in top.items():
        direction = "increased" if value > 0 else "decreased"
        parts.append(f"{feature} {direction} risk")
    return "; ".join(parts)


def build_narratives(df: pd.DataFrame, contribs: pd.DataFrame, label: str) -> pd.DataFrame:
    narratives = []
    for idx in df.index:
        narrative = narrative_from_contributions(contribs.loc[idx])
        narratives.append({"row_id": idx, "model": label, "narrative": narrative})
    return pd.DataFrame(narratives)
