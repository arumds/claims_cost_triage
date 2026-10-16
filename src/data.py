"""Loading, cleaning and data-quality checks."""
import numpy as np
import pandas as pd

from . import config as C


def load_raw(path=C.DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per check -> easy to paste into the 'data limitations' slide."""
    ident = (df.insurance_company_cost + df.policyholder_cost - df.contractor_cost).abs()
    rev = df[df.manual_review]
    checks = {
        "rows": len(df),
        "claims": df.claim_id.nunique(),
        "quotes": df.quote_id.nunique(),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "quotes_with_multiple_rows": int((df.quote_id.value_counts() > 1).sum()),
        "claims_with_prelim_and_final": int(
            df.groupby("claim_id").quote_event.nunique().eq(2).sum()),
        "share_manual_review": round(df.manual_review.mean(), 3),
        "rejections_total": int((df.decision_on_quote == "rejected").sum()),
        "rejections_outside_manual_review": int(
            ((df.decision_on_quote == "rejected") & ~df.manual_review).sum()),
        "rejection_rate_in_review": round((rev.decision_on_quote == "rejected").mean(), 3),
        "median_insurer_cost_reviewed": round(rev.insurance_company_cost.median()),
        "median_insurer_cost_not_reviewed": round(df[~df.manual_review].insurance_company_cost.median()),
        "share_rows_cost_identity_holds": round((ident < 1).mean(), 3),
        "estimate_missing_share": round(df.estimated_repair_cost.isna().mean(), 3),
        "estimate_zero_count": int((df.estimated_repair_cost == 0).sum()),
        "estimate_round_10k_share": round((df.estimated_repair_cost.dropna() % 10_000 == 0).mean(), 3),
        "insurer_cost_at_cap": int((df.insurance_company_cost >= C.COST_CAP).sum()),
        "contractor_cost_at_cap": int((df.contractor_cost >= C.COST_CAP).sum()),
        "building_area_missing_share": round(df.building_area.isna().mean(), 3),
        "construction_year_missing_share": round(df.construction_year.isna().mean(), 3),
        "inspection_org_missing_share": round(df.inspection_org.isna().mean(), 3),
        "inspection_org_levels": df.inspection_org.nunique(),
        "corr_estimate_vs_contractor": round(
            df[["estimated_repair_cost", "contractor_cost"]].corr().iloc[0, 1], 3),
    }
    return pd.DataFrame({"check": checks.keys(), "value": checks.values()})


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df["est_missing"] = df.estimated_repair_cost.isna().astype(int)
    df["est_zero"] = (df.estimated_repair_cost == 0).astype(int)
    df["area_missing"] = df.building_area.isna().astype(int)
    df["year_missing"] = df.construction_year.isna().astype(int)
    df["at_cost_cap"] = (df.contractor_cost >= C.COST_CAP).astype(int)
    df["identity_breaks"] = (
        (df.insurance_company_cost + df.policyholder_cost - df.contractor_cost).abs() >= 1
    ).astype(int)
    for col in C.CATEGORICAL:
        df[col] = df[col].fillna("MISSING")
    return df


# Categorical columns that can disagree between rows of the same quote
VOTE_COLS = ["inspection_type", "insurance_type", "building_type",
             "building_geographical_location", "inspection_org"]


def _majority(s: pd.Series):
    """Most frequent value; ties go to the value that appears first (file order)."""
    counts = s.value_counts()
    top = set(counts[counts == counts.max()].index)
    return next(v for v in s if v in top)


def to_quote_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per quote (the decision unit).

    Rows sharing a quote_id carry the same contractor cost, estimate, images, building size/year,
    excess, decision and review flag, but ~10% of quotes have rows with conflicting categories
    (e.g. the same building labelled House in one row and Commercial in another). Some of this
    looks like multiple coverages, some like entry errors. We keep one row per quote:
    - categories resolved by majority vote within the quote (ties -> first row in file order)
    - n_coverage_rows: number of rows the quote had
    - conflicting_rows: 1 if any category disagreed between the rows
    - insurance_company_cost: max over rows (only used for reporting, never as a model input)
    Alternatives tested (multi-hot flags of all labels present) gave no accuracy gain.
    """
    df = df.sort_values(["claim_id", "quote_id"], kind="stable")
    agg = df.groupby("quote_id").agg(
        n_coverage_rows=("claim_id", "size"),
        insurance_company_cost=("insurance_company_cost", "max"),
    )
    first = df.drop_duplicates("quote_id").set_index("quote_id").drop(columns=["insurance_company_cost"])

    multi_ids = agg.index[agg.n_coverage_rows > 1]
    multi = df[df.quote_id.isin(multi_ids)]
    if len(multi):
        voted = multi.groupby("quote_id")[VOTE_COLS].agg(_majority)
        first.loc[voted.index, VOTE_COLS] = voted
        conflict = multi.groupby("quote_id")[VOTE_COLS].nunique().gt(1).any(axis=1)
    else:
        conflict = pd.Series(dtype=bool)
    agg["conflicting_rows"] = conflict.reindex(agg.index, fill_value=False).astype(int)

    q = first.join(agg).reset_index()
    q["quote_seq"] = q.groupby("claim_id").cumcount() + 1   # order of the quote within its claim
    q["rejected"] = (q.decision_on_quote == "rejected").astype(int)

    # group rare inspection orgs so the categorical stays learnable
    counts = q.inspection_org.value_counts()
    rare = counts[counts < C.RARE_ORG_MIN_COUNT].index
    q.loc[q.inspection_org.isin(rare), "inspection_org"] = "OTHER"
    return q
