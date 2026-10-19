"""Claims Cost Triage – interactive demo.

Run from the project folder (venv active):
    streamlit run app.py
"""
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.data import VOTE_COLS, clean, data_quality_report, load_raw, to_claim_level, to_quote_level
from src.features import add_quote_features
from src.models import (fair_cost_importance, fair_cost_oof, fit_fair_cost_models, fit_final_rejection_model,
                        predict_fair, rejection_oof, score_all)
from src.triage import add_rule, head_to_head, queue_compare, queue_overlap, rule_table

st.set_page_config(page_title="Claims Cost Triage", page_icon="🏠", layout="wide")
ORANGE, BLUE, GREY, GREEN = "#dd6b20", "#2b6cb0", "#a0aec0", "#38a169"
REVIEW_TOP_SHARE = 0.20      # top 20% by expected € at risk go to review

# EDA helpers (as in notebook 01, section 4)
EDA_VARS = {
    "inspection_type": "Damage type", "inspection_org": "Inspection org (≥30 claims)",
    "estimate_bin": "Inspector estimate", "images_bin": "Inspection images",
    "construction_year_bin": "Construction year", "building_area_bin": "Building area",
    "building_geographical_location": "Location", "quote_event": "Quote stage",
    "decision_on_quote": "Decision", "review": "Review",
}
# Streamlit sizes charts with autosize "fit", so a pixel height includes title and axes and can squeeze
# the bands until Vega hides labels. A fixed step per category plus labelOverlap=False keeps every label.
CAT_AXIS = alt.Axis(labelOverlap=False, labelLimit=320)
EDA_ORDER = {
    "construction_year_bin": ["<1950", "1950–69", "1970–89", "1990–2009", "2010+", "missing"],
    "building_area_bin": ["<80 m²", "80–120", "120–200", "200–400", "400+ m²", "missing"],
    "estimate_bin": ["0", "10–20k", "30–50k", "60–80k", "90k+", "missing"],
    "images_bin": ["0–10", "11–20", "21–40", "41+"],
}


def eda_frame(claims: pd.DataFrame) -> pd.DataFrame:
    """Claim table with the notebook's buckets and the bill ÷ estimate ratio."""
    e = claims.copy()
    bins = {
        "construction_year_bin": ("construction_year", [0, 1950, 1970, 1990, 2010, 2030]),
        "building_area_bin": ("building_area", [0, 80, 120, 200, 400, 1e6]),
        "estimate_bin": ("estimated_repair_cost", [-1, 0, 20_000, 50_000, 80_000, 1e7]),
        "images_bin": ("inspection_images", [-1, 10, 20, 40, 1e4]),
    }
    for name, (col, edges) in bins.items():
        labels = [k for k in EDA_ORDER[name] if k != "missing"]
        e[name] = pd.cut(e[col], edges, labels=labels).astype(object).where(e[col].notna(), "missing")
    e["review"] = np.where(e.manual_review, "manual review", "auto-processed")
    e["cost_to_estimate"] = e.contractor_cost / e.estimated_repair_cost.where(
        (e.estimated_repair_cost > 0) & (e.contractor_cost >= 100))
    return e


def box_by(e: pd.DataFrame, value: str, var: str, title: str, ref: float | None = None, domain: list | None = None,
           ticks: list | None = None, tick_format: str = "~s"):
    """Box plot of a positive value per bucket (log scale, outliers hidden) + the numbers as a table."""
    d = e[[var, value]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d[value] > 0].astype({var: str})
    stats = d.groupby(var)[value].agg(claims="size", median="median",
                                      p25=lambda s: s.quantile(.25), p75=lambda s: s.quantile(.75))
    stats = stats[stats.claims >= (30 if var == "inspection_org" else 10)]
    order = ([k for k in EDA_ORDER[var] if k in stats.index] if var in EDA_ORDER
             else list(stats.sort_values("median", ascending=False).index))
    d = d[d[var].isin(order)]
    chart = alt.Chart(d).mark_boxplot(extent=1.5, outliers=False, size=16, color=BLUE,
                                      median={"color": ORANGE, "strokeWidth": 2}).encode(
        y=alt.Y(f"{var}:N", sort=order, title=None, axis=CAT_AXIS),
        x=alt.X(f"{value}:Q", title=title,
                scale=alt.Scale(type="log", domain=domain, clamp=True) if domain else alt.Scale(type="log"),
                axis=alt.Axis(values=ticks, format=tick_format) if ticks else alt.Axis()))
    if ref is not None:
        chart += alt.Chart(pd.DataFrame({"x": [ref]})).mark_rule(color="black", strokeDash=[4, 4]).encode(x="x:Q")
    return chart.properties(height=alt.Step(34)), stats.reindex(order)


# Data + models (cached)
def _data_path() -> Path:
    return Path(os.environ.get("CLAIMS_DATA", C.DATA_PATH))


@st.cache_data(show_spinner="Loading and cleaning data…")
def load(path: str):
    raw = load_raw(path)
    cleaned = clean(raw)
    q = add_quote_features(to_quote_level(cleaned))
    q["rejected"] = q.decision_on_quote.eq("rejected").astype(int)
    claims = to_claim_level(cleaned)                 # one quote per claim, as in notebook 01
    return raw, q, claims, data_quality_report(raw)


@st.cache_resource(show_spinner="Training models (≈1 min, only once)…")
def train(path: str):
    raw, q, claims, _ = load(path)
    q = q.copy()
    fair_q, fair_claims, m_a = fair_cost_oof(claims, q)
    q = add_rule(q.join(fair_q))
    fair_claims["rejected"] = claims.decision_on_quote.eq("rejected").astype(int)   # € views: one quote per claim
    r = q[q.manual_review].copy()
    r["p_reject"], m_b = rejection_oof(r)
    rej_model, coefs = fit_final_rejection_model(r)
    q["p_reject"] = score_all(rej_model, q)
    q.loc[r.index, "p_reject"] = r.p_reject
    q["priority_eur"] = q.p_reject * q.contractor_cost
    threshold = q.priority_eur.quantile(1 - REVIEW_TOP_SHARE)
    q["suggested_action"] = np.where((q.priority_eur >= threshold) | (q.above_p90 == 1),
                                     "manual review", "auto-approve")
    r["priority"] = r.p_reject * r.contractor_cost
    fair_final = fit_fair_cost_models(claims)
    fair_imp = fair_cost_importance(claims)
    h2h, h2h_curves = head_to_head(r)
    return dict(q=q, r=r, m_a=m_a, m_b=m_b, coefs=coefs, rej_model=rej_model,
                fair=fair_final, threshold=threshold, fair_imp=fair_imp, fair_claims=fair_claims,
                rule_tbl=rule_table(q), h2h=h2h, h2h_curves=h2h_curves, overlap=queue_overlap(q))


path = _data_path()
if not path.exists():
    st.title("Claims Cost Triage")
    up = st.file_uploader("Data file not found – upload tech_data_2026.xlsx", type="xlsx")
    if up is None:
        st.stop()
    path = Path("uploaded_data.xlsx")
    path.write_bytes(up.getvalue())

raw, q0, claims, dq = load(str(path))
M = train(str(path))
q, r = M["q"], M["r"]
eur = lambda x: f"€{x:,.0f}"

# Navigation
st.sidebar.title("🏠 Claims Cost Triage")
page = st.sidebar.radio("Go to", [
    "1 · Problem & approach", "2 · Data & limitations", "3 · Explore the data", "4 · Hypotheses",
    "5 · Model results", "6 · Quote checker", "7 · Review queue", "8 · Verdict & next steps"])
st.sidebar.caption(f"{len(raw):,} rows · {q.claim_id.nunique():,} claims · {len(q):,} quotes")


# 1. Problem
if page.startswith("1"):
    st.title("Can AI/ML help reduce claims costs?")
    st.markdown("**Short answer: partly yes.** A fair-cost model works with today's data; "
                "a reliable *overpricing* detector needs better labels (a random audit).")
    st.caption("With the number of reviews fixed, this is a ranking problem under a capacity constraint: order "
               "the queue so the reviews catch as much unnecessary cost as possible, and measure success in euro "
               "caught rather than accuracy.")
    c = st.columns(4)
    c[0].metric("Quotes", f"{len(q):,}")
    c[1].metric("Manually reviewed", f"{q.manual_review.mean():.0%}")
    c[2].metric("Rejected quotes", f"{q.rejected.sum()}",
                delta=f"{q.loc[q.manual_review, 'rejected'].mean():.1%} of manually reviewed quotes", delta_color="off",
                help="All of them in manually reviewed cases; auto-processed quotes are never rejected")
    c[3].metric("Paid to contractors", f"€{claims.contractor_cost.sum() / 1e6:,.0f}M",
                help="Sum of contractor bills, one quote per claim (revised quotes are not added up twice)")

    st.subheader("Business problem → data science problem")
    st.markdown("""
| Moment | What we know | Role |
|---|---|---|
| **Inspection** (before repair) | damage & insurance type, building, images, inspection org, **inspector estimate**, excess | model **inputs** |
| **Contractor quote** (after repair) | **contractor cost**, insurer/policyholder split, review flag, decision | **targets / outcomes** |
""")
    st.info("**Data science problem:** from inspection data, predict what the repair *should* cost. "
            "When the quote arrives, compare it with that prediction and send the quotes with the "
            "highest expected € at risk to manual review. Everything else is approved automatically.")
    st.markdown("##### The workflow: today and proposed")
    rev_share = claims.manual_review.mean()
    rej_share = claims.loc[claims.manual_review, "decision_on_quote"].eq("rejected").mean()
    node = 'node [shape=box, style="rounded,filled", fillcolor="#f7fafc", color="#a0aec0", fontname="Helvetica", fontsize=11];'
    edge = 'edge [color="#718096", fontname="Helvetica", fontsize=10];'
    today = f"""digraph {{ rankdir=TB; {node} {edge}
        insp [label="Inspection\\nestimate, photos, damage type"];
        quote [label="Contractor quote arrives"];
        sel [label="Picked for review?\\n(goes with insurer cost)", shape=diamond, fillcolor="#ffffff"];
        rev [label="Manual review\\n{rev_share:.0%} of claims"];
        auto [label="Auto-approved\\n{1 - rev_share:.0%} of claims, never checked", fillcolor="#fde8d7", color="{ORANGE}"];
        dec [label="Approve or reject\\n{rej_share:.0%} of reviewed rejected"];
        insp -> quote -> sel; sel -> rev [label=" yes"]; sel -> auto [label=" no"]; rev -> dec;
    }}"""
    proposed = f"""digraph {{ rankdir=TB; {node} {edge}
        insp [label="Inspection\\nestimate, photos, damage type"];
        fair [label="Fair-cost model\\nfair price p50 and upper band p90", fillcolor="#e6effa", color="{BLUE}"];
        quote [label="Contractor quote arrives"];
        rule [label="Primary rule\\nbill > {C.RULE_K}× estimate?", shape=diamond, fillcolor="#fdf2e9", color="{ORANGE}"];
        cmp [label="Bill vs fair price?", shape=diamond, fillcolor="#ffffff"];
        fast [label="Fast approval\\nclose to fair (0.8–1.25×)", fillcolor="#e6f4ea", color="{GREEN}"];
        rev [label="Manual review\\nrule flag, above p90 or largest bills"];
        audit [label="Random audit (~5%)\\nmeasures the real saving", style="rounded,dashed"];
        out [label="Approve, negotiate or reject\\nrecord the approved amount"];
        insp -> fair; insp -> quote; quote -> rule;
        rule -> rev [label=" yes"]; rule -> cmp [label=" no"]; fair -> cmp;
        cmp -> fast [label=" close"]; cmp -> rev [label=" too high"]; rev -> out;
        fast -> audit [style=dashed]; out -> fair [style=dashed, label=" retrain"];
    }}"""
    g1, g2 = st.columns(2)
    g1.markdown("**Today:** review is picked by insurer cost, not by the bill")
    g1.graphviz_chart(today, width="stretch")
    g2.markdown("**Proposed:** the rule catches large overruns, then every other bill is compared with a fair price")
    g2.graphviz_chart(proposed, width="stretch")
    st.caption("Counts are per claim (one quote per claim). The audit and the recorded approved amount are what turn "
               "the fair-cost gap into a measured saving.")

    p_, a, b = st.columns(3)
    p_.markdown(f"**Primary · Rule**  \nReview if the bill is more than **{C.RULE_K}× the inspector estimate**. "
                "Transparent, runs tomorrow, no model needed.")
    a.markdown("**Secondary · Model A · Fair cost**  \nQuantile gradient boosting → typical cost (p50) and "
               "upper normal bound (p90) from inspection data only.")
    b.markdown("**Secondary · Model B · Rejection risk**  \nRegularised logistic regression trained on reviewed "
               "quotes → probability a reviewer rejects the quote.")
    st.markdown("**Priority** = P(reject) × quoted amount. Quotes above the fair-cost p90 are also flagged.  \n"
                "**Success metric:** share of rejected € caught for a given review effort, rule vs model head to head.")


# 2. Data & limitations
elif page.startswith("2"):
    st.title("Data & limitations")
    no_est = ~claims.estimated_repair_cost.gt(0)
    c = st.columns(4)
    c[0].metric("Raw rows → claims", f"{len(raw):,} → {len(claims):,}",
                help="Duplicate rows, coverage lines and revised quotes collapse to one quote per claim (see below)")
    c[1].metric("Rejection labels", f"{q.rejected.sum()} quotes",
                delta=f"{int(((q.decision_on_quote == 'rejected') & ~q.manual_review).sum())} outside manual review",
                delta_color="off", help="Rejections exist only where someone reviewed, so labels are scarce and biased")
    c[2].metric("Claims with no usable estimate", f"{no_est.mean():.0%}",
                help=f"{no_est.sum():,} claims have a zero or missing estimate, so bill ÷ estimate cannot be computed")
    c[3].metric("Building area missing", f"{claims.building_area.isna().mean():.0%}",
                help=f"Construction year missing for {claims.construction_year.isna().mean():.0%} of claims")

    st.subheader("How the data was prepared")
    dedup = raw.drop_duplicates()
    multi = dedup.quote_id.value_counts()
    st.markdown("**Examples:** what the raw data looks like and what the cleaning keeps")
    ex1, ex2, ex3 = st.tabs(["Exact duplicate rows", "One quote, several rows", "One claim, several quotes"])
    with ex1:
        dup_ids = raw.loc[raw.duplicated(), "quote_id"].unique()
        st.markdown(f"**{int(raw.duplicated().sum())} rows** are exact copies of another row, identical in every "
                    f"column ({len(dup_ids)} quotes affected). The copies are dropped.")
        pick = st.selectbox("Quote with an exact duplicate", dup_ids, key="ex_dup")
        rows_ = raw[raw.quote_id == pick]
        st.dataframe(rows_.assign(dropped_as_copy=rows_.duplicated()), hide_index=True, width="stretch",
                     column_config={"dropped_as_copy": st.column_config.CheckboxColumn("dropped as copy")})
    with ex2:
        multi_ids = multi[multi > 1].index
        default = int(np.flatnonzero(multi_ids == 1)[0]) if (multi_ids == 1).any() else 0
        st.markdown(f"**{len(multi_ids):,} quotes** span several rows (coverage lines). The contractor cost, estimate, "
                    "decision and review flag are the same on every row; the insurer/policyholder split and some "
                    "categories can differ. The quote-level table keeps one row: categories by majority vote, "
                    "insurer cost = the maximum.")
        pick = st.selectbox("Quote", multi_ids, index=default, key="ex_quote")
        rows_ = dedup[dedup.quote_id == pick]
        varying = [c for c in rows_.columns if rows_[c].nunique(dropna=False) > 1]
        st.dataframe(rows_, hide_index=True, width="stretch")
        st.caption("Columns that differ between these rows: " + (", ".join(varying) if varying else "none"))
        st.markdown("**Kept in the quote-level table:**")
        st.dataframe(q.loc[q.quote_id == pick, raw.columns], hide_index=True, width="stretch")
    with ex3:
        per_claim = q.groupby("claim_id").quote_id.nunique()
        multi_claims = per_claim[per_claim > 1].index
        mixed = q.groupby("claim_id").quote_event.nunique().pipe(lambda s_: s_[s_ > 1].index)
        default = int(np.flatnonzero(multi_claims == mixed[0])[0]) if len(mixed) else 0
        st.markdown(f"**{len(multi_claims):,} claims** have more than one quote ({len(mixed):,} with both a preliminary "
                    "and a final quote). Quotes are adjusted over time, so they are revisions of the same repair. "
                    "The claim-level table keeps the final quote if there is one, otherwise the latest preliminary quote.")
        pick = st.selectbox("Claim", multi_claims, index=default, key="ex_claim")
        kept = claims.loc[claims.claim_id == pick, "quote_id"]
        rows_ = q.loc[q.claim_id == pick, list(raw.columns)].sort_values("quote_id")
        st.dataframe(rows_.assign(kept_for_claim=rows_.quote_id.isin(kept)), hide_index=True, width="stretch",
                     column_config={"kept_for_claim": st.column_config.CheckboxColumn("kept (one quote per claim)"),
                                    "contractor_cost": st.column_config.NumberColumn("contractor cost", format="€%.0f"),
                                    "estimated_repair_cost": st.column_config.NumberColumn("estimate", format="€%.0f")})
        st.caption(f"Summing every quote would count the same repair several times: "
                   f"{eur(q.contractor_cost.sum())} across all quotes vs {eur(claims.contractor_cost.sum())} "
                   "with one quote per claim.")

    st.markdown("**The numbers:** from raw rows to one quote per claim")
    f = st.columns(4)
    f[0].metric("Raw rows", f"{len(raw):,}")
    f[1].metric("After removing exact duplicates", f"{len(dedup):,}", delta=f"−{len(raw) - len(dedup)} rows",
                delta_color="off", help="Rows identical in every column")
    f[2].metric("Quotes (one row each)", f"{len(q):,}",
                delta=f"{(multi > 1).sum()} quotes spanned {int(multi[multi > 1].sum()):,} rows", delta_color="off",
                help="Rows sharing a quote_id are coverage lines with the same contractor cost, decision and review "
                     "flag, so each quote is kept once")
    f[3].metric("Claims (one quote each)", f"{len(claims):,}", delta=f"−{len(q) - len(claims):,} earlier quotes",
                delta_color="off", help="Final quote kept if there is one, else the latest preliminary quote. "
                                        "Train/test splits are made by claim, so no claim is on both sides")
    st.markdown(f"""
- **Conflicting rows within a quote:** {int(raw.drop_duplicates().groupby("quote_id")[VOTE_COLS].nunique().gt(1).any(axis=1).sum()):,} quotes have rows with different labels (e.g. the same building as House in one row and Commercial in another). The quote-level table (rejection model, review queue) resolves each category by majority vote (ties → first row). The claim-level table (fair-cost model, Explore page) follows the notebook cleaning and keeps the quote's last row as it is; for {int((claims.merge(q, on="quote_id", suffixes=("", "_vote")).pipe(lambda m_: pd.concat([m_[c_].ne(m_[f"{c_}_vote"]) for c_ in VOTE_COLS if c_ != "inspection_org"], axis=1)).any(axis=1)).sum()):,} claims that row differs from the majority, mostly in damage type.
- **Model inputs are known when the quote arrives:** damage, insurance and building type, location, quote stage, inspection org, estimate, photos, area, year and excess. The rejection model adds the quoted amount and its gap to the fair price.
- **Missing values are left as missing, not filled in:** the fair-cost model (gradient boosting) handles them natively; the rejection model uses the median.
- **One quote per claim for the fair-cost model:** a claim's quotes are revisions of the same repair, so the final quote is kept (else the latest preliminary one) and no repair is counted twice ({len(claims):,} claims).
- **Rare inspection organisations grouped:** orgs with fewer than {C.RARE_ORG_MIN_COUNT} quotes become "OTHER" ({(q.inspection_org == "OTHER").sum():,} quotes).
- **Outcome columns excluded from model inputs:** insurer / policyholder cost, decision and review flag are only known after the decision.
""")

    st.subheader("Key limitations")
    st.markdown("""
- **Selection bias:** rejections exist only where a human looked, and humans mostly look at big claims.
- **No savings label:** we know approved/rejected, not what the right price was. Only 70 rejected quotes.
- **Coarse estimate:** always a multiple of €10k; 240 zeros, 3.5% missing. `building_area` 50% missing.
- **Messy structure:** several rows per quote and several quotes per claim (handled above); costs capped at €300k (kept, but caps limit what the model can learn at the top end).
- **Leakage risk:** outcome columns would make the model look better than it is; they are excluded (see above).
- **No dates:** drift over time can't be checked.
""")
    with st.expander("Full data-quality report"):
        st.dataframe(dq, hide_index=True, width="stretch")


# 3. Explore
elif page.startswith("3"):
    st.title("Explore the data")
    st.caption(f"The exploration from `notebooks/01_data_exploration.ipynb`, on the cleaned data: one quote per claim "
               f"({len(claims):,} claims, the final quote where there is one).")
    e = eda_frame(claims)
    e["rejected"] = e.decision_on_quote.eq("rejected").astype(int)
    tabs = st.tabs(["Overview", "Contractor cost by feature", "Bill ÷ estimate by feature", "Feature distributions",
                    "Inspection organisations", "Review & decisions"])

    with tabs[0]:
        a, b = st.columns(2)
        miss = raw.isna().mean().loc[lambda s_: s_ > 0].sort_values(ascending=False)
        miss = miss.rename_axis("column").reset_index(name="share")
        a.altair_chart(alt.Chart(miss).mark_bar(color=BLUE).encode(
            y=alt.Y("column:N", sort="-x", title=None, axis=CAT_AXIS),
            x=alt.X("share:Q", axis=alt.Axis(format="%"), title="Share of raw rows missing"),
            tooltip=["column", alt.Tooltip("share", format=".1%")]).properties(height=alt.Step(28), title="Missing values"),
            width="stretch")
        mix = (raw.drop_duplicates().groupby("claim_id").quote_event.apply(lambda s_: " + ".join(sorted(set(s_))))
               .value_counts().rename_axis("quote stages in the claim").reset_index(name="claims"))
        b.markdown("**Quote stages per claim**")
        b.dataframe(mix, hide_index=True, width="stretch")
        rows_per_quote = (raw.drop_duplicates().quote_id.value_counts().value_counts().sort_index()
                          .rename_axis("rows per quote").reset_index(name="quotes"))
        quotes_per_claim = (q.groupby("claim_id").quote_id.nunique().value_counts().sort_index()
                            .rename_axis("quotes per claim").reset_index(name="claims"))
        c1, c2 = st.columns(2)
        c1.altair_chart(alt.Chart(rows_per_quote).mark_bar(color=BLUE).encode(
            x=alt.X("rows per quote:O", axis=alt.Axis(labelAngle=0)), y="quotes:Q",
            tooltip=["rows per quote", "quotes"]).properties(height=220, title="Rows per quote"), width="stretch")
        c2.altair_chart(alt.Chart(quotes_per_claim).mark_bar(color=BLUE).encode(
            x=alt.X("quotes per claim:O", axis=alt.Axis(labelAngle=0)), y="claims:Q",
            tooltip=["quotes per claim", "claims"]).properties(height=220, title="Quotes per claim"), width="stretch")
        st.info("**Takeaway:** building area is missing for about half the rows and construction year for about 11%, "
                "so they are left as missing rather than filled in. A quote can span several rows and a claim several "
                "quotes; the analysis uses one quote per claim. Only a minority of claims have both a preliminary and "
                "a final quote.")

    def box_grid(data, value, xtitle, fmt, ref=None, domain=None, ticks=None, tick_format="~s"):
        """One box plot per feature, two per row on a shared x scale, plus the numbers in one table."""
        tables, items = {}, list(EDA_VARS.items())
        for k_ in range(0, len(items), 2):
            for col_, (var, label) in zip(st.columns(2), items[k_:k_ + 2]):
                ch, stats = box_by(data, value, var, xtitle, ref=ref, domain=domain, ticks=ticks, tick_format=tick_format)
                col_.altair_chart(ch.properties(title=label), width="stretch")
                tables[label] = stats
        with st.expander("The numbers behind the charts"):
            st.dataframe(pd.concat(tables, names=["feature", "group"]), width="stretch",
                         column_config={c_: st.column_config.NumberColumn(format=fmt) for c_ in ["median", "p25", "p75"]})

    with tabs[1]:
        st.caption("Contractor cost per group, one quote per claim. Orange line = median, box = middle 50% of claims, "
                   "whiskers = typical range (outliers hidden). Log scale, same axis in every panel; bills are capped at €300k.")
        box_grid(e[e.contractor_cost >= 100], "contractor_cost", "Contractor cost, € (log scale)", "€%.0f",
                 domain=[1_000, 400_000], ticks=[1_000, 3_000, 10_000, 30_000, 100_000, 300_000])
        st.info("**Takeaway:** damage type drives the bill most (fire about 3.4× water). Bigger estimates and more "
                "photos go with bigger bills. Preliminary quotes are larger than final ones, rejected claims are large, "
                "and reviewed claims are *smaller* than auto-processed ones. Building age, area and location matter little.")

    with tabs[2]:
        st.caption("Contractor cost ÷ inspector estimate per group (above 1 = bill above the estimate; dashed line = "
                   "bill equals estimate). Log scale, same axis in every panel. Claims with a zero or missing estimate "
                   "are left out.")
        box_grid(e, "cost_to_estimate", "Bill ÷ estimate (log scale)", "%.2f×", ref=1, domain=[0.1, 10],
                 ticks=[0.1, 0.25, 0.5, 1, 2, 4, 10], tick_format="~g")
        st.info("**Takeaway:** the estimate's error has a direction that depends on the claim. Fire and building damage "
                "come in above the estimate, water and 'other' below it; small estimates are too low and large ones too "
                "high. Reviewed claims sit *below* their estimate, so review does not target overruns. A flat rule like "
                "'review if bill > 1.5× estimate' would flag the wrong claims.")

    with tabs[3]:
        cats = ["inspection_type", "insurance_type", "building_type", "building_geographical_location",
                "quote_event", "decision_on_quote", "review"]
        cols_ = st.columns(2)
        for i, c_ in enumerate(cats):
            vc = e[c_].fillna("MISSING").value_counts().rename_axis(c_).reset_index(name="claims")
            cols_[i % 2].altair_chart(alt.Chart(vc).mark_bar(color=BLUE).encode(
                y=alt.Y(f"{c_}:N", sort="-x", title=None, axis=CAT_AXIS), x="claims:Q",
                tooltip=[c_, "claims"]).properties(height=alt.Step(28), title=c_), width="stretch")
        nums = ["building_area", "construction_year", "inspection_images", "insurance_excess_amount"]
        for col_, c_ in zip(st.columns(4), nums):
            col_.altair_chart(alt.Chart(e[[c_]].dropna()).mark_bar(color=BLUE).encode(
                x=alt.X(f"{c_}:Q", bin=alt.Bin(maxbins=40), title=c_), y=alt.Y("count():Q", title="claims")).properties(
                height=200), width="stretch")
        num_all = ["contractor_cost", "estimated_repair_cost", "insurance_company_cost", "policyholder_cost",
                   "building_area", "construction_year", "inspection_images", "insurance_excess_amount"]
        corr = (e[num_all].corr(method="spearman").reset_index(names="x")
                .melt(id_vars="x", var_name="y", value_name="rho"))
        base = alt.Chart(corr).encode(x=alt.X("x:N", sort=num_all, title=None), y=alt.Y("y:N", sort=num_all, title=None))
        heat = base.mark_rect().encode(color=alt.Color("rho:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
                                                       title="Spearman ρ"), tooltip=["x", "y", alt.Tooltip("rho", format=".2f")])
        txt = base.mark_text(fontSize=11).encode(text=alt.Text("rho:Q", format=".2f"))
        st.altair_chart((heat + txt).properties(height=380, title="Spearman correlation"), width="stretch")
        st.info("**Takeaway:** water damage is the largest segment. Building attributes add almost no signal on their own. "
                "The estimate is the strongest pre-decision predictor of the bill, but far from sufficient. Insurer and "
                "policyholder cost correlate strongly, but they are outcomes of processing, so they leak and stay out "
                "of the models.")

    with tabs[4]:
        k = st.columns(3)
        k[0].metric("Orgs with 10+ claims", f"{e.inspection_org[~e.inspection_org.isin(['OTHER', 'MISSING'])].nunique():,}")
        k[1].metric("Claims from rare orgs (pooled as OTHER)", f"{e.inspection_org.eq('OTHER').mean():.0%}")
        k[2].metric("Claims with no org", f"{e.inspection_org.eq('MISSING').mean():.0%}")
        orgt = (e.groupby("inspection_org").agg(claims=("claim_id", "size"), median_ratio=("cost_to_estimate", "median"),
                                                 median_bill=("contractor_cost", "median"),
                                                 review_rate=("manual_review", "mean"))
                .query("claims >= 30").drop(index=["OTHER"], errors="ignore").sort_values("median_ratio").reset_index())
        bars = alt.Chart(orgt).mark_bar().encode(
            y=alt.Y("inspection_org:N", sort="-x", title=None, axis=CAT_AXIS),
            x=alt.X("median_ratio:Q", title="Median bill ÷ estimate"),
            color=alt.condition(alt.datum.median_ratio > 1.3, alt.value(ORANGE), alt.value(BLUE)),
            tooltip=["inspection_org", "claims", alt.Tooltip("median_ratio", format=".2f"),
                     alt.Tooltip("median_bill", format=",.0f"), alt.Tooltip("review_rate", format=".0%")])
        rule = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color="black").encode(x="x:Q")
        st.altair_chart((bars + rule).properties(height=alt.Step(26),
                                                 title="Inspection orgs with 30+ claims"), width="stretch")
        st.info("**Takeaway:** organisations estimate differently: some see bills below their estimate, others well above. "
                "That is useful for partner feedback, and a feature with many levels, so rare orgs are pooled.")

    with tabs[5]:
        a, b = st.columns(2)
        a.markdown("**Review × decision (claims)**")
        a.dataframe(pd.crosstab(e.review, e.decision_on_quote, margins=True), width="stretch")
        summ = e.groupby("review").agg(claims=("claim_id", "size"), median_bill=("contractor_cost", "median"),
                                       median_insurer_cost=("insurance_company_cost", "median"),
                                       bill_to_estimate=("cost_to_estimate", "median"),
                                       rejection_rate=("rejected", "mean"))
        b.markdown("**Reviewed vs auto-processed**")
        b.dataframe(summ, width="stretch", column_config={
            "median_bill": st.column_config.NumberColumn("median bill", format="€%.0f"),
            "median_insurer_cost": st.column_config.NumberColumn("median insurer cost", format="€%.0f"),
            "bill_to_estimate": st.column_config.NumberColumn("bill ÷ estimate", format="%.2f×"),
            "rejection_rate": st.column_config.NumberColumn("rejection rate", format="percent")})
        gap = e[e.cost_to_estimate.notna()].assign(gap=lambda d_: pd.cut(
            d_.cost_to_estimate, [0, .5, .8, 1.25, 2, 1e9],
            labels=["<0.5×", "0.5–0.8×", "0.8–1.25× (close)", "1.25–2×", ">2×"]))
        gt = gap.groupby("gap", observed=True).agg(claims=("claim_id", "size"), review_rate=("manual_review", "mean"))
        gt["rejected_if_reviewed"] = gap[gap.manual_review].groupby("gap", observed=True).rejected.mean()
        gt = gt.reset_index().assign(gap=lambda d_: d_.gap.astype(str))
        st.altair_chart(alt.Chart(gt).mark_bar(color=ORANGE).encode(
            x=alt.X("gap:N", sort=None, title="Contractor bill ÷ inspector estimate",
                    axis=alt.Axis(labelAngle=0, labelOverlap=False)),
            y=alt.Y("review_rate:Q", axis=alt.Axis(format="%"), title="Share manually reviewed"),
            tooltip=["gap", "claims", alt.Tooltip("review_rate", format=".0%"),
                     alt.Tooltip("rejected_if_reviewed", format=".0%")]).properties(
            height=260, title="Bills far above the estimate are reviewed the least"), width="stretch")
        top_gap, low_gap = gt.iloc[-1], gt.iloc[:2]
        st.caption(f"Claims billed over 2× their estimate are reviewed {top_gap.review_rate:.0%} of the time, vs "
                   f"{low_gap.review_rate.min():.0%}–{low_gap.review_rate.max():.0%} for bills below the estimate. "
                   f"When they are reviewed, {top_gap.rejected_if_reviewed:.0%} are rejected, the highest rate of any "
                   f"group, but from only {int(gap[(gap.gap == '>2×') & gap.manual_review].shape[0])} reviewed claims.")
        c1, c2 = st.columns(2)
        ch, _ = box_by(e[e.insurance_company_cost >= 100], "insurance_company_cost", "review", "Insurer cost, € (log scale)")
        c1.altair_chart(ch.properties(title="Reviewed claims have a much higher insurer cost…"), width="stretch")
        ch, _ = box_by(e[e.contractor_cost >= 100], "contractor_cost", "review", "Contractor cost, € (log scale)")
        c2.altair_chart(ch.properties(title="…but a smaller bill"), width="stretch")
        rv = e[e.manual_review]
        ch, _ = box_by(rv[rv.contractor_cost >= 100], "contractor_cost", "decision_on_quote", "Contractor cost, € (log scale)")
        st.altair_chart(ch.properties(title="Within review: rejected claims are larger"), width="stretch")
        st.info("**Takeaway:** all rejections happen in reviewed claims; auto-processed claims are never rejected because "
                "nobody checks them. Review goes with high insurer cost, not with a large bill, so rejection labels come "
                "from a biased sample. Rejected claims skew towards the largest bills, which is why 'largest bill first' "
                "is a strong baseline.")

# 4. Hypotheses
elif page.startswith("4"):
    st.title("Hypotheses: what we set out to show")
    st.markdown("**Goal:** pay less for repairs without reviewing more claims. The exploration raised these "
                "hypotheses; the models test them.")
    e = eda_frame(claims)
    by_type = e.groupby("inspection_type").cost_to_estimate.median()
    by_size = e.groupby("estimate_bin").cost_to_estimate.median().dropna()
    lo, hi = min(by_type.min(), by_size.min()), max(by_type.max(), by_size.max())
    by_rev = e.groupby("review").agg(ratio=("cost_to_estimate", "median"), insurer=("insurance_company_cost", "median"),
                                     bill=("contractor_cost", "median"))
    ma, mb = M["m_a"], M["m_b"]
    est_err, fair_err = ma.median_abs_pct_err.iloc[0], ma.median_abs_pct_err.iloc[1]
    auc_gap = float(mb.loc[mb.score == "baseline: quote vs fair-cost model", "ROC_AUC"].iloc[0])
    h2h = M["h2h"].set_index("approach")
    rule_eur, model_eur = (h2h.loc[a, "rejected_eur_caught@20%"] for a in
                           ["primary: rule  bill ÷ estimate", "secondary: fair-cost + P(reject) × bill"])
    bill_eur = h2h.loc["reference: largest bill first", "rejected_eur_caught@20%"]
    hyps = [
        dict(key="H1", name="The inspector estimate is a biased benchmark",
             where="3 · Explore the data → Bill ÷ estimate by feature",
             result=f"Bill ÷ estimate ranges from {lo:.2f}× to {hi:.2f}× across damage types and estimate sizes.",
             ok=hi / lo > 1.5),
        dict(key="H2", name="Review is not aimed at bills that run over",
             where="3 · Explore the data → Review & decisions",
             result=(f"Auto-processed bills run {by_rev.ratio['auto-processed']:.2f}× their estimate vs "
                     f"{by_rev.ratio['manual review']:.2f}× for reviewed; reviewed claims have "
                     f"{by_rev.insurer['manual review'] / by_rev.insurer['auto-processed']:.0f}× the insurer cost but a smaller bill."),
             ok=by_rev.ratio["auto-processed"] > by_rev.ratio["manual review"]),
        dict(key="H3", name="Inspection data can price a repair much better than the estimate",
             where="5 · Model results",
             result=f"Typical error ±{fair_err:.0%} vs ±{est_err:.0%} for the estimate "
                    f"(MAE {eur(ma.MAE_eur.iloc[1])} vs {eur(ma.MAE_eur.iloc[0])}).",
             ok=fair_err < 0.75 * est_err),
        dict(key="H4", name="Bills above fair cost are the ones reviewers reject",
             where="5 · Model results",
             result=f"ROC-AUC {auc_gap:.2f}: the fair-cost gap does not predict today's rejections, "
                    "so savings cannot be proven with current labels; a random audit is needed."
                    if auc_gap < 0.6 else f"ROC-AUC {auc_gap:.2f}: the fair-cost gap predicts rejections.",
             ok=auc_gap >= 0.6),
        dict(key="H5", name="The models beat the simple rule at the same review effort",
             where="5 · Model results → Head to head",
             result=f"{model_eur:.0%} vs {rule_eur:.0%} of rejected €. But 'largest bill first' alone gets "
                    f"{bill_eur:.0%}: the win comes from bill size, which is consistent with H4.",
             ok=model_eur > rule_eur + 0.1),
    ]
    rows = "\n".join(f"| **{h['key']}** | {h['name']} | {'✅ Supported' if h['ok'] else '❌ Not supported'} | "
                     f"{h['result']} | {h['where']} |" for h in hyps)
    st.markdown("| | Hypothesis | Verdict | Evidence | Where to look |\n|---|---|---|---|---|\n" + rows)
    n_ok = sum(bool(h["ok"]) for h in hyps)
    st.info(f"**{n_ok} of {len(hyps)} hold.** The estimate is a poor yardstick and review misses overruns (H1, H2), "
            "a fair-cost model prices claims far better (H3), and the models beat the simple rule (H5). But the price "
            "gap does not explain today's rejections (H4), so the savings stay a candidate until a random audit measures them.")
    st.caption("Results are computed live from the data and models.")


# 5. Model results
elif page.startswith("5"):
    st.title("Model results")
    st.caption("All numbers are out-of-fold: every quote is scored by a model that never saw its claim.")
    PRIMARY, SECONDARY = "primary: rule  bill ÷ estimate", "secondary: fair-cost + P(reject) × bill"
    LABEL = {PRIMARY: "Primary · rule", SECONDARY: "Secondary · fair-cost + rejection model"}
    h2h = M["h2h"].set_index("approach").loc[[PRIMARY, SECONDARY]]
    at20 = h2h["rejected_eur_caught@20%"]

    # The answer first
    st.markdown(f"""
**Two ways to pick which quotes get reviewed, judged on the same {h2h.n_quotes.iloc[0]} reviewed quotes:**
- **Primary · rule:** review if the bill is more than **{C.RULE_K}× the inspector estimate**. No model.
- **Secondary · models:** **A** predicts a fair price from inspection data, **B** predicts how likely a reviewer
  rejects the quote. Review the quotes with the most € at risk: **P(reject) × bill**.
""")
    k = st.columns(3)
    k[0].metric("Rejected € caught · primary rule", f"{at20[PRIMARY]:.0%}", help="Reviewing the top 20% of quotes")
    k[1].metric("Rejected € caught · secondary", f"{at20[SECONDARY]:.0%}",
                delta=f"+{(at20[SECONDARY] - at20[PRIMARY]) * 100:.0f} pts vs rule", help="Reviewing the top 20% of quotes")
    k[2].metric("Fair price error (model A)", eur(M["m_a"].MAE_eur.iloc[1]),
                delta=f"vs {eur(M['m_a'].MAE_eur.iloc[0])} for the raw estimate", delta_color="off")

    t1, t2, t3, t4 = st.tabs(["① Primary rule", "② Secondary models", "③ Head to head: how the secondary wins",
                              "④ Unchecked overpricing (audit candidates)"])

    # 1. Primary rule
    with t1:
        st.markdown(f"Compare the contractor's bill with the inspector's estimate and review when it is more than "
                    f"**{C.RULE_K}×** higher. Transparent and ready to run: the floor the models must beat.")
        rk = M["rule_tbl"].set_index("trigger_k").loc[C.RULE_K]
        k = st.columns(4)
        k[0].metric("Quotes flagged", f"{rk.flagged_share:.0%}", delta=f"{int(rk.quotes_flagged):,} quotes", delta_color="off")
        k[1].metric("Severe overruns caught", f"{rk.severe_overruns_caught:.0%}",
                    help=f"Bill more than €{C.SEVERE_OVERRUN_EUR:,} above the estimate. Defined by the same gap, so this flatters the rule.")
        k[2].metric("Rejections caught", f"{rk.rejected_caught_if_reviewed:.0%}", help="Of the quotes reviewers rejected")
        k[3].metric("Quotes it can't score", f"{M['overlap'].value.iloc[-1]:.0%}", help="Estimate missing or 0")
        st.warning(f"**Strength:** it finds almost every bill far above its estimate. **Weakness:** it flags "
                   f"{rk.flagged_share:.0%} of all quotes yet catches only {rk.rejected_caught_if_reviewed:.0%} of what "
                   "reviewers rejected, and it ignores how big the bill is: a €5k bill at 3× ranks above a €200k bill at 1.2×.")
        with st.expander("Other thresholds k"):
            rt = M["rule_tbl"]
            st.dataframe(pd.DataFrame({
                "rule: bill > k × estimate": rt.trigger_k.map("k = {:.2f}".format),
                "quotes flagged": rt.quotes_flagged, "share flagged": rt.flagged_share.map("{:.0%}".format),
                "severe overruns caught": rt.severe_overruns_caught.map("{:.0%}".format),
                "overrun € in flagged set": rt.overrun_eur_flagged.map(lambda x: f"€{x / 1e6:.0f}M"),
                "rejections caught": rt.rejected_caught_if_reviewed.map("{:.0%}".format)}),
                hide_index=True, width="stretch")

    # 2. Secondary models
    with t2:
        a, b = st.columns(2)
        a.subheader("A · Fair-cost model")
        a.markdown("What *should* this repair cost? Quantile gradient boosting on inspection data, **including the "
                   "inspector's estimate but never the contractor's bill**. Gives a typical price (p50) and an upper "
                   "normal bound (p90).")
        ma = M["m_a"]
        a.dataframe(pd.DataFrame({"price guess": ["inspector estimate", "fair-cost model (p50)"],
                                  "MAE": ma.MAE_eur.map(eur),
                                  "median % error": ma.median_abs_pct_err.map("{:.0%}".format),
                                  "p90 coverage": ma.p90_coverage.map(lambda x: "–" if pd.isna(x) else f"{x:.0%}")}),
                    hide_index=True, width="stretch")
        a.caption("Halves the error of the inspector estimate.")
        imp = M["fair_imp"].head(10)
        a.altair_chart(alt.Chart(imp).mark_bar(color=BLUE).encode(
            y=alt.Y("feature:N", sort="-x", title=None, axis=CAT_AXIS),
            x=alt.X("share_pct:Q", title="Share of importance (%)"),
            tooltip=["feature", alt.Tooltip("share_pct", format=".1f", title="share %"),
                     alt.Tooltip("typical_error_increase_pct", format=".1f",
                                 title="error increase if shuffled (%)")]).properties(
            height=alt.Step(26), title="What drives the fair price (permutation importance)"), width="stretch")

        b.subheader("B · Rejection model")
        b.markdown("How likely is a reviewer to reject this quote? Regularised logistic regression on reviewed quotes, "
                   "using the bill, the estimate, the gap to the fair price (from A) and inspection details.")
        mb = M["m_b"].set_index("score")
        b.dataframe(pd.DataFrame({"score": ["rejection model", "largest bill first", "bill ÷ estimate"],
                                  "ROC AUC": mb.ROC_AUC.loc[["model p_reject", "baseline: largest quote",
                                                             "baseline: quote/estimate ratio"]].values}),
                    hide_index=True, width="stretch")
        b.caption(f"Trained on {mb.n_rejected.iloc[0]} rejections (base rate {mb.base_rate.iloc[0]:.1%}), so little "
                  "signal is available. On its own it adds little; multiplying by the bill is what makes the queue strong.")
        with b.expander("Coefficients"):
            st.caption("Odds ratio per 1 SD (numeric) or vs. other categories. Above 1 = more likely rejected. Read as directions.")
            st.dataframe(M["coefs"], hide_index=True, width="stretch")
        st.info("**Together:** priority = **P(reject) from B × bill**, so a quote ranks high when it is both risky and "
                "expensive. Quotes above A's p90 are also sent to review.")

    # 3. Head to head
    with t3:
        budget = st.slider("Review budget (share of reviewed quotes)", 0.05, 1.0, 0.20, 0.05)
        qc, pts = queue_compare(r, budget)
        qc = qc.set_index("approach")
        p, s_ = qc.loc[PRIMARY], qc.loc[SECONDARY]
        n_rej, n_all = int(p.rejections_total), len(pts)
        rej_total = pts.contractor_cost[pts.rejected == 1].sum()
        st.markdown(f"""**How to read it.** Reviewers rejected **{n_rej} of {n_all}** reviewed quotes, worth
**{eur(rej_total)}** in total. Each approach ranks all {n_all} quotes and reviews only its top **{budget:.0%}**
({int(p.quotes_reviewed)} quotes). *Rejected € caught* = the share of those {eur(rej_total)} that lands in its queue.""")

        st.markdown(f"##### 1 · Result at a {budget:.0%} budget")
        st.dataframe(pd.DataFrame({
            "": [LABEL[PRIMARY], LABEL[SECONDARY]],
            "ranks by": ["bill ÷ estimate", "P(reject) × bill"],
            "rejected € caught": [f"{p.rejected_eur_caught:.0%}", f"{s_.rejected_eur_caught:.0%}"],
            "rejected quotes caught": [f"{int(p.rejections_caught)} of {n_rej}", f"{int(s_.rejections_caught)} of {n_rej}"],
            "median bill in queue": [eur(p.median_bill), eur(s_.median_bill)],
            "median bill ÷ estimate in queue": [f"{p.median_bill_to_estimate:.2f}×", f"{s_.median_bill_to_estimate:.2f}×"]}),
            hide_index=True, width="stretch")
        st.caption(f"The € share is higher than the count share for the secondary ({int(s_.rejections_caught)} of {n_rej} "
                   f"quotes but {s_.rejected_eur_caught:.0%} of €) because the rejections it catches are the largest bills.")

        st.markdown("##### 2 · The gap holds at every budget")
        curves = M["h2h_curves"][M["h2h_curves"].approach.isin([PRIMARY, SECONDARY])].assign(
            approach=lambda d: d.approach.map(LABEL))
        base = alt.Chart(curves).mark_line(strokeWidth=2.5).encode(
            x=alt.X("share_reviewed:Q", axis=alt.Axis(format="%"), title="Share of quotes reviewed"),
            y=alt.Y("share_value_caught:Q", axis=alt.Axis(format="%"), title="Share of rejected € caught"),
            color=alt.Color("approach:N", scale=alt.Scale(domain=list(LABEL.values()), range=[ORANGE, BLUE]),
                            legend=alt.Legend(orient="bottom", title=None, labelLimit=320)))
        vline = alt.Chart(pd.DataFrame({"x": [budget]})).mark_rule(color="black", strokeDash=[4, 4]).encode(x="x")
        st.altair_chart((base + vline).properties(height=320), width="stretch")

        st.markdown("##### 3 · Why: the two queues pick different quotes")
        plot = pts.melt(id_vars=["quote_id", "contractor_cost", "rule_ratio", "rejected"],
                        var_name="approach", value_name="in_queue")
        plot["approach"] = plot.approach.str.replace("in_queue::", "").map(LABEL)
        plot["status"] = np.select(
            [plot.in_queue & plot.rejected.eq(1), ~plot.in_queue & plot.rejected.eq(1), plot.in_queue],
            ["rejected · caught", "rejected · missed", "approved · in queue"], "approved · not in queue")
        status = ["rejected · caught", "rejected · missed", "approved · in queue", "approved · not in queue"]
        st.altair_chart(alt.Chart(plot).mark_circle(stroke="white", strokeWidth=.5).encode(
            x=alt.X("contractor_cost:Q", scale=alt.Scale(type="log", domain=[3000, 320000], clamp=True), title="Bill (€, log)",
                    axis=alt.Axis(values=[3e3, 1e4, 3e4, 1e5, 3e5], format="~s")),
            y=alt.Y("rule_ratio:Q", scale=alt.Scale(type="log", domain=[.1, 20], clamp=True),
                    title="Bill ÷ estimate (log)", axis=alt.Axis(values=[.1, .5, 1, 2, 5, 10])),
            color=alt.Color("status:N", scale=alt.Scale(domain=status, range=["#c53030", "#feb2b2", BLUE, "#e2e8f0"]),
                            legend=alt.Legend(orient="bottom", title=None)),
            size=alt.condition(alt.datum.rejected == 1, alt.value(70), alt.value(22)),
            order=alt.Order("rejected:Q"),
            tooltip=["quote_id", alt.Tooltip("contractor_cost", format=",.0f", title="bill €"),
                     alt.Tooltip("rule_ratio", format=".2f", title="bill ÷ estimate"), "status"])
            .properties(height=320, width=380)
            .facet(column=alt.Column("approach:N", title=None, sort=list(LABEL.values()),
                                     header=alt.Header(labelFontSize=13, labelFontWeight="bold"))))
        rv = r[r.rule_ratio.notna()]
        rej, ok = rv[rv.rejected == 1], rv[rv.rejected == 0]
        st.success(f"""- **Rejected quotes are big bills:** median {eur(rej.contractor_cost.median())} vs {eur(ok.contractor_cost.median())} for approved ones.
- **But they are not unusually far above the estimate:** median bill ÷ estimate {rej.rule_ratio.median():.2f}× vs {ok.rule_ratio.median():.2f}×.
- So the **rule looks up** (high ratio) and fills its queue with mid-size overruns, while the **secondary looks right**
  (high € at risk) and picks the big bills, where the dark-red rejections sit.""")
        st.caption("Caveat: most of the gain comes from the bill-size term. Reviewers' rejections don't follow overpricing "
                   "in this data, so an audit is needed to test whether the price gap finds overpricing nobody reviewed.")
        with st.expander("Same budget on all quotes: how much do the two queues overlap?"):
            ov = M["overlap"].set_index("metric").value
            k = st.columns(3)
            k[0].metric(f"Review budget (rule k = {C.RULE_K})", f"{ov.iloc[0]:.0%}")
            k[1].metric("Flagged by both", f"{ov['flagged by both']:.0%}",
                        delta=f"agreement {ov['agreement (Jaccard)']:.0%}", delta_color="off")
            k[2].metric("€ above fair p90 caught (rule vs secondary)",
                        f"{ov['€ above fair p90 caught: rule']:.0%} vs {ov['€ above fair p90 caught: secondary']:.0%}",
                        help="This yardstick comes from the fair-cost model, so it favours the secondary.")

    # 4. Unchecked overpricing
    with t4:
        st.markdown("The head-to-head can only judge quotes reviewers saw. This tab shows the bills the fair-cost model "
                    "would question that nobody looked at: each claim's bill ÷ its fair price (p50), one quote per claim.")
        fc = M["fair_claims"]
        g = fc.assign(ratio=fc.contractor_cost / fc.fair_p50)
        g["gap"] = pd.cut(g.ratio, [0, .5, .8, 1.25, 2, 1e9],
                          labels=["<0.5×", "0.5–0.8×", "0.8–1.25× (close)", "1.25–2×", ">2×"])
        gt = g.groupby("gap", observed=True).agg(claims=("ratio", "size"), review_rate=("manual_review", "mean"))
        gt["share"] = gt.claims / len(g)
        gt["rejection_rate_if_reviewed"] = g[g.manual_review].groupby("gap", observed=True).rejected.mean()
        gt["excess_m"] = (g.assign(ex=(g.contractor_cost - g.fair_p50).clip(lower=0))
                           .groupby("gap", observed=True).ex.sum() / 1e6)
        gt = gt.reset_index().assign(gap=lambda d: d.gap.astype(str))
        far = g[g.ratio > 2]
        far_auto = far[~far.manual_review]
        k = st.columns(3)
        k[0].metric("Claims > 2× fair price, approved without review", f"{len(far_auto)} of {len(far)}")
        k[1].metric("Their amount above fair price", f"€{(far_auto.contractor_cost - far_auto.fair_p50).sum() / 1e6:.0f}M",
                    help="Candidate saving, not proven – needs an audit")
        k[2].metric("Claims close to fair price (0.8–1.25×)", f"{(g.ratio.between(.8, 1.25)).mean():.0%}",
                    help="Candidates for fast approval")
        st.warning("**A candidate saving, not a proven one.** Nobody checked these bills, so we don't know how much of the "
                   "gap is real overpricing and how much is damage the inspection data doesn't show. A random audit of "
                   "these claims is what turns this number into a measured saving.")
        gap_colors = alt.Scale(domain=list(gt.gap), range=[GREY, GREY, GREEN, ORANGE, "#c05621"])
        c1, c2 = st.columns(2)
        for col, y, ttl, fmt in [(c1, "share", "Share of claims", "%"), (c2, "excess_m", "Amount above fair price (€M)", "")]:
            col.altair_chart(alt.Chart(gt).mark_bar().encode(
                x=alt.X("gap:N", sort=None, title="Bill ÷ fair price (p50)", axis=alt.Axis(labelAngle=0)),
                y=alt.Y(f"{y}:Q", axis=alt.Axis(format=fmt) if fmt else alt.Axis(), title=None),
                color=alt.Color("gap:N", scale=gap_colors, legend=None),
                tooltip=["gap", "claims", alt.Tooltip("share", format=".0%"),
                         alt.Tooltip("excess_m", format=".1f", title="€M above fair price"),
                         alt.Tooltip("review_rate", format=".0%", title="reviewed today"),
                         alt.Tooltip("rejection_rate_if_reviewed", format=".1%")]).properties(
                height=300, title=ttl), width="stretch")
        st.caption("Today claims over 2× the fair price are reviewed only slightly more often than close ones (≈28% vs 24%). "
                   "Rejection rate when reviewed: ≈5% for close claims vs ≈17% for those over 2×, based on only 35 rejected claims.")


# 6. Quote checker
elif page.startswith("6"):
    st.title("Quote checker")
    st.caption("Enter what the inspection found and the contractor's quote. The models say what the repair "
               "typically costs and whether the quote should be reviewed.")

    presets = {"Custom": None, "Claim 2 · fire, €133.6k quote": dict(
        inspection_type="Fire & natural damage", insurance_type="Natural & Environmental", building_type="House",
        building_geographical_location="urban", inspection_org="C657E129A4E", quote_event="preliminary quote",
        estimated_repair_cost=50000, contractor_cost=133563.77, inspection_images=22, building_area=111.3,
        construction_year=1995, insurance_excess_amount=6000)}
    pick = st.selectbox("Example", list(presets))
    p = presets[pick] or {}
    opts = lambda col: sorted(q[col].unique())
    idx = lambda col, default: opts(col).index(p.get(col, default)) if p.get(col, default) in opts(col) else 0

    with st.form("checker"):
        c1, c2, c3 = st.columns(3)
        inspection_type = c1.selectbox("Damage type", opts("inspection_type"), index=idx("inspection_type", "Water damage"))
        insurance_type = c1.selectbox("Insurance type", opts("insurance_type"), index=idx("insurance_type", "Water"))
        building_type = c1.selectbox("Building type", opts("building_type"), index=idx("building_type", "House"))
        location = c2.selectbox("Location", opts("building_geographical_location"), index=idx("building_geographical_location", "urban"))
        org = c2.selectbox("Inspection org", opts("inspection_org"), index=idx("inspection_org", "OTHER"))
        quote_event = c2.selectbox("Quote event", opts("quote_event"), index=idx("quote_event", "preliminary quote"))
        est = c3.number_input("Inspector estimate (€)", 0, 400_000, int(p.get("estimated_repair_cost", 40000)), 10_000)
        quote = c3.number_input("Contractor quote (€)", 0.0, 300_000.0, float(p.get("contractor_cost", 55000)), 1000.0)
        excess = c3.number_input("Insurance excess (€)", 0, 50_000, int(p.get("insurance_excess_amount", 5000)), 1000)
        with st.expander("Building details (optional)"):
            d1, d2, d3 = st.columns(3)
            images = d1.number_input("Inspection images", 0, 300, int(p.get("inspection_images", 30)))
            area = d2.number_input("Building area (m², 0 = unknown)", 0.0, 1000.0, float(p.get("building_area", 0)))
            year = d3.number_input("Construction year (0 = unknown)", 0, 2026, int(p.get("construction_year", 1985)))
        go = st.form_submit_button("Check quote", type="primary")

    if go:
        row = pd.DataFrame([dict(
            inspection_org=org, inspection_type=inspection_type, insurance_type=insurance_type,
            building_geographical_location=location, building_type=building_type, quote_event=quote_event,
            estimated_repair_cost=est or np.nan, inspection_images=images,
            building_area=area or np.nan, construction_year=year or np.nan,
            insurance_excess_amount=excess,
            contractor_cost=quote)])
        row = add_quote_features(row).join(predict_fair(M["fair"], row))
        p_rej = float(score_all(M["rej_model"], row).iloc[0])
        p50, p90 = row.fair_p50.iloc[0], row.fair_p90.iloc[0]
        priority = p_rej * quote
        review = (priority >= M["threshold"]) or (quote > p90)

        m = st.columns(4)
        m[0].metric("Typical cost (p50)", eur(p50))
        m[1].metric("Upper normal (p90)", eur(p90))
        m[2].metric("Quote vs typical", f"{quote / p50:.2f}×", delta=eur(quote - p50), delta_color="inverse")
        m[3].metric("P(reject)", f"{p_rej:.0%}")

        rng = pd.DataFrame({"label": ["fair range"], "lo": [p50], "hi": [p90]})
        bar = alt.Chart(rng).mark_bar(height=26, color="#c6dbef").encode(
            x=alt.X("lo:Q", title="€", scale=alt.Scale(domain=[0, max(quote, p90, est) * 1.15])), x2="hi:Q", y=alt.Y("label:N", title=None))
        pts = alt.Chart(pd.DataFrame({"what": ["typical (p50)", "inspector estimate", "contractor quote"],
                                      "x": [p50, est, quote], "label": ["fair range"] * 3})).mark_point(
            size=220, filled=True).encode(x="x:Q", y="label:N",
                                          color=alt.Color("what:N", scale=alt.Scale(domain=["typical (p50)", "inspector estimate", "contractor quote"],
                                                                     range=[BLUE, GREY, ORANGE]), legend=alt.Legend(orient="bottom", title=None)),
                                          shape=alt.Shape("what:N"), tooltip=["what", alt.Tooltip("x", format=",.0f")])
        st.altair_chart((bar + pts).properties(height=110), width="stretch")

        reasons = []
        if quote > p90:
            reasons.append(f"quote is above the upper normal cost ({eur(p90)}) for this kind of claim")
        if priority >= M["threshold"]:
            reasons.append(f"expected € at risk ({eur(priority)}) is in the top {REVIEW_TOP_SHARE:.0%} of all quotes")
        if review:
            st.error("**Recommendation: MANUAL REVIEW** – " + "; ".join(reasons) + ".")
        else:
            st.success(f"**Recommendation: AUTO-APPROVE** – quote is within the normal range and expected € at risk "
                       f"({eur(priority)}) is below the review threshold ({eur(M['threshold'])}).")
        st.caption("Fair cost is learned from approved historical quotes; P(reject) is trained on reviewed quotes only "
                   "(biased toward large claims), so treat it as a ranking signal, not a calibrated probability.")


# 7. Review queue
elif page.startswith("7"):
    st.title("Review queue")
    today = int(q.manual_review.sum())
    n = st.slider("How many quotes can the team review?", 0, len(q) // 5 * 5, today - today % 5, 5,
                  help=f"Today the insurer reviewed {today:,} quotes, {today / len(q):.1%} of the {len(q):,} "
                       f"in the file, which is the default here. The deck reports the same {today:,}.")
    rank_by = st.radio("Rank by", ["Primary rule: bill ÷ estimate", "Secondary: P(reject) × quote (€ at risk)"],
                       horizontal=True, help="Quotes without a usable estimate go last under the rule")
    key = "priority_eur" if rank_by.startswith("Secondary") else "rule_ratio"
    who = "model" if key == "priority_eur" else "rule"
    st.caption("All historical quotes, ranked by " + (
        "expected € at risk: P(reject) × quote." if who == "model" else
        f"how far the bill is above the inspector estimate (flagged above {C.RULE_K}×). Quotes without an estimate go last."))
    ranked = q.sort_values(key, ascending=False, na_position="last").reset_index(drop=True)
    top = ranked.head(n)
    c = st.columns(4)
    c[0].metric("Quotes to review", f"{n:,}", help=f"{n / len(q):.1%} of the {len(q):,} quotes in the file.")
    c[1].metric("Of which currently auto-approved", f"{(~top.manual_review).sum():,}")
    c[2].metric("Quoted € in queue", eur(top.contractor_cost.sum()))
    rej_eur = q.loc[q.rejected == 1, "contractor_cost"].sum()
    c[3].metric("Known rejections in queue", f"{top.rejected.sum()} / {q.rejected.sum()}",
                delta=f"{top.loc[top.rejected == 1, 'contractor_cost'].sum() / rej_eur:.0%} of rejected €",
                delta_color="off",
                help="Rejections only exist where someone reviewed, so today's process 'catches' 100% by definition. "
                     "Compare by € and look at the four groups below.")

    st.subheader(f"{who.capitalize()} queue vs today's reviews")
    in_q = q.quote_id.isin(top.quote_id)
    groups = [("A · In queue, reviewed today", in_q & q.manual_review, "both agree: review"),
              ("B · In queue, auto-approved today", in_q & ~q.manual_review, f"{who} ADDS these"),
              ("C · Not in queue, reviewed today", ~in_q & q.manual_review, f"{who} DROPS these"),
              ("D · Not in queue, auto-approved today", ~in_q & ~q.manual_review, "both agree: auto-approve")]
    rows = []
    for name, mask, meaning in groups:
        d = q[mask]
        reviewed = bool(d.manual_review.all()) if len(d) else False
        rows.append({"group": name, "meaning": meaning, "quotes": len(d),
                     "median quote": eur(d.contractor_cost.median()) if len(d) else "–",
                     "above fair p90": f"{(d.contractor_cost > d.fair_p90).mean():.0%}" if len(d) else "–",
                     "rejected": f"{int(d.rejected.sum())} ({d.rejected.mean():.0%})" if reviewed else "unknown",
                     "rejected €": eur(d.loc[d.rejected == 1, "contractor_cost"].sum()) if reviewed else "unknown – never checked"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    a_, b_, c_ = (q[g[1]] for g in groups[:3])
    caught_a = a_.loc[a_.rejected == 1, "contractor_cost"].sum() / rej_eur
    lost_c = eur(c_.loc[c_.rejected == 1, "contractor_cost"].sum())
    if who == "model":
        st.info(f"**Reading it:** the overlap (A, {len(a_)} quotes) holds **{caught_a:.0%} of all rejected €** "
                f"(rejection rate {a_.rejected.mean():.0%} vs {q[q.manual_review].rejected.mean():.0%} overall). "
                f"To free capacity, the model stops reviewing {len(c_)} smaller, mostly normally priced quotes (C) that held only "
                f"{lost_c} of rejections, and instead reviews {len(b_)} larger "
                f"quotes (B, median {eur(b_.contractor_cost.median()) if len(b_) else '–'}) that nobody checked. "
                "What B would yield is unknown – a pilot or random audit would measure it.")
    else:
        st.info(f"**Reading it:** the rule keeps only {len(a_)} of today's reviews (A), holding **{caught_a:.0%} of all "
                f"rejected €**. It drops {len(c_)} reviewed quotes (C, median {eur(c_.contractor_cost.median())}, median "
                f"bill ÷ estimate {c_.rule_ratio.median():.2f}×) that held {lost_c} of rejections, because most "
                f"rejected quotes are not far above their estimate. Instead it adds {len(b_)} auto-approved quotes (B, median "
                f"{eur(b_.contractor_cost.median()) if len(b_) else '–'}, median bill ÷ estimate "
                f"{b_.rule_ratio.median():.2f}×) that overran their estimate. "
                f"{int(q.rule_ratio.isna().sum())} quotes have no usable estimate and can never enter its queue. "
                "Whether B holds real overpricing is unknown – a pilot or random audit would measure it.")

    st.subheader("The queue")

    only_auto = st.checkbox("Show only quotes that were auto-approved in reality")
    show = top[~top.manual_review] if only_auto else top
    show = show.assign(above_p90=show.contractor_cost > show.fair_p90)
    cols = ["claim_id", "quote_id", "quote_event", "inspection_type", "estimated_repair_cost", "contractor_cost",
            "rule_ratio", "rule_flag", "fair_p50", "fair_p90", "above_p90", "p_reject", "priority_eur",
            "manual_review", "decision_on_quote"]
    st.dataframe(show[cols].round(2), hide_index=True, width="stretch", height=480,
                 column_config={"p_reject": st.column_config.ProgressColumn("P(reject)", min_value=0, max_value=1, format="%.2f"),
                                "priority_eur": st.column_config.NumberColumn("€ at risk", format="€%.0f"),
                                "estimated_repair_cost": st.column_config.NumberColumn("estimate", format="€%.0f"),
                                "contractor_cost": st.column_config.NumberColumn("quote", format="€%.0f"),
                                "rule_ratio": st.column_config.NumberColumn("bill ÷ estimate", format="%.2f"),
                                "rule_flag": st.column_config.CheckboxColumn(f"rule flag (> {C.RULE_K}×)"),
                                "fair_p50": st.column_config.NumberColumn("fair p50", format="€%.0f"),
                                "fair_p90": st.column_config.NumberColumn("fair p90", format="€%.0f"),
                                "above_p90": st.column_config.CheckboxColumn("above fair p90")})
    st.download_button("Download queue (CSV)", show[cols].to_csv(index=False), "review_queue.csv", "text/csv")


# 8. Verdict & next steps
elif page.startswith("8"):
    st.title("Verdict & next steps")
    h2h = M["h2h"].set_index("approach")["rejected_eur_caught@20%"]
    rule_eur, model_eur = h2h["primary: rule  bill ÷ estimate"], h2h["secondary: fair-cost + P(reject) × bill"]
    bill_eur = h2h["reference: largest bill first"]
    fc = M["fair_claims"]
    ratio = fc.contractor_cost / fc.fair_p50
    far = fc[ratio > 2]
    far_auto = far[~far.manual_review]
    far_eur = (far_auto.contractor_cost - far_auto.fair_p50).sum() / 1e6
    ma = M["m_a"]

    st.subheader("Does ML help reduce claims costs? Partly.")
    c = st.columns(3)
    c[0].metric("Rejected € caught, top 20% reviewed", f"{model_eur:.0%}",
                delta=f"{(model_eur - rule_eur) * 100:+.0f} pts vs the rule", delta_color="normal")
    c[1].metric("Fair-cost model error (MAE)", eur(ma.MAE_eur.iloc[1]),
                delta=f"vs {eur(ma.MAE_eur.iloc[0])} for the estimate", delta_color="off")
    c[2].metric("Unchecked bills over 2× fair price", f"{len(far_auto)} claims",
                delta=f"≈€{far_eur:.0f}M above fair price", delta_color="off")

    a, b = st.columns(2)
    a.success(f"""**Where it helps**
- **Better review queue:** {model_eur:.0%} vs {rule_eur:.0%} of rejected € for the same effort. Most of the gain is bill
  size: "largest bill first" alone gets {bill_eur:.0%}.
- **Better price benchmark:** half the error of the inspector estimate, and it also prices the
  {M['overlap'].value.iloc[-1]:.0%} of quotes without an estimate.
- **Finds bills nobody checked:** {len(far_auto)} of {len(far)} claims over 2× their fair price were never reviewed.""")
    b.warning(f"""**Where it doesn't (yet)**
- **Overpricing doesn't predict rejection:** reviewers' decisions don't follow the price gap in this data.
- **Savings are not proven:** there is no "correct price" label and auto-approved quotes were never checked.
- **Few, biased labels:** only {M['m_b'].n_rejected.iloc[0]} rejections, all from reviews aimed at large claims.""")

    st.info("**Recommendation:** (1) run the **rule** now as a transparent floor; (2) rank the review queue with the "
            "**model**; (3) start a **random audit** of auto-approved claims above the fair price and record the "
            "approved amount after review. Better data, not a better algorithm, is what proves the saving.")

    st.subheader("With more time")
    st.markdown(f"""
1. **Random audit** of ~5% of auto-approved quotes: unbiased labels and a measured saving.
2. **Record the approved amount and the rejection reason**, so savings become a target and price is separated from coverage.
3. **Add dates and itemised quote lines**, to validate on later claims and flag specific inflated items.
4. **Tune the rule per damage type and in €**, since one k = {C.RULE_K} over-flags some segments and misses others.
5. **Shadow-run before switching:** score new quotes with rule and model side by side, then pilot and compare € saved per review hour.""")
    with st.expander("More ideas"):
        st.markdown(f"""
- Calibrated price ranges with conformal prediction (p90 covers {ma.p90_coverage.iloc[1]:.0%}, target 90%).
- Image models on inspection photos, and text from inspection reports and quotes.
- Feedback to inspection organisations on estimate accuracy; monitor contractors whose bills sit above fair cost.
- Negotiate preliminary quotes against the fair price; watch preliminary → final quote changes.
- Explainable flags for reviewers, and monitoring for drift, calibration and contractor behaviour.""")
