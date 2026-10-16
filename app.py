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
from src.data import clean, data_quality_report, load_raw, to_quote_level
from src.evaluate import budget_table, capture_curve
from src.features import add_quote_features
from src.models import (fair_cost_importance, fair_cost_oof, fit_fair_cost_models, fit_final_rejection_model,
                        predict_fair, rejection_oof, score_all)

st.set_page_config(page_title="Claims Cost Triage", page_icon="🏠", layout="wide")
ORANGE, BLUE, GREY, GREEN = "#dd6b20", "#2b6cb0", "#a0aec0", "#38a169"
REVIEW_TOP_SHARE = 0.20      # top 20% by expected € at risk go to review


# ------------------------------------------------------------------ data + models (cached)
def _data_path() -> Path:
    return Path(os.environ.get("CLAIMS_DATA", C.DATA_PATH))


@st.cache_data(show_spinner="Loading and cleaning data…")
def load(path: str):
    raw = load_raw(path)
    q = add_quote_features(to_quote_level(clean(raw)))
    return raw, q, data_quality_report(raw)


@st.cache_resource(show_spinner="Training models (≈1 min, only once)…")
def train(path: str):
    raw, q, _ = load(path)
    q = q.copy()
    fair_oof, m_a = fair_cost_oof(q)
    q = q.join(fair_oof[["fair_p50", "fair_p90", "excess_vs_fair", "above_p90"]])
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
    fair_final = fit_fair_cost_models(q)
    fair_imp = fair_cost_importance(q)
    return dict(q=q, r=r, m_a=m_a, m_b=m_b, coefs=coefs, rej_model=rej_model,
                fair=fair_final, threshold=threshold, fair_imp=fair_imp)


path = _data_path()
if not path.exists():
    st.title("Claims Cost Triage")
    up = st.file_uploader("Data file not found – upload tech_data_2026.xlsx", type="xlsx")
    if up is None:
        st.stop()
    path = Path("uploaded_data.xlsx")
    path.write_bytes(up.getvalue())

raw, q0, dq = load(str(path))
M = train(str(path))
q, r = M["q"], M["r"]
eur = lambda x: f"€{x:,.0f}"

# ------------------------------------------------------------------ navigation
st.sidebar.title("🏠 Claims Cost Triage")
page = st.sidebar.radio("Go to", [
    "1 · Problem & approach", "2 · Data & limitations", "3 · Explore the data",
    "4 · Model results", "5 · Quote checker", "6 · Review queue", "7 · Verdict & next steps"])
st.sidebar.caption(f"{len(raw):,} rows · {q.claim_id.nunique():,} claims · {len(q):,} quotes")


# ------------------------------------------------------------------ 1. problem
if page.startswith("1"):
    st.title("Can AI/ML help reduce claims costs?")
    st.markdown("**Short answer: partly yes.** A fair-cost model works with today's data; "
                "a reliable *overpricing* detector needs better labels (a random audit).")
    c = st.columns(4)
    c[0].metric("Quotes", f"{len(q):,}")
    c[1].metric("Manually reviewed", f"{q.manual_review.mean():.0%}")
    c[2].metric("Rejected quotes", f"{q.rejected.sum()}", help="All of them in manually reviewed cases")
    c[3].metric("Fair-cost model error", eur(M["m_a"].MAE_eur.iloc[1]),
                delta=f"vs {eur(M['m_a'].MAE_eur.iloc[0])} for raw estimate", delta_color="off")

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
    a, b = st.columns(2)
    a.markdown("**Model A · Fair cost**  \nQuantile gradient boosting → typical cost (p50) and "
               "upper normal bound (p90) from inspection data only.")
    b.markdown("**Model B · Rejection risk**  \nRegularised logistic regression trained on reviewed "
               "quotes → probability a reviewer rejects the quote.")
    st.markdown("**Priority** = P(reject) × quoted amount. Quotes above the fair-cost p90 are also flagged.  \n"
                "**Success metric:** share of rejected € caught for a given review effort.")


# ------------------------------------------------------------------ 2. data & limitations
elif page.startswith("2"):
    st.title("Data & limitations")
    c = st.columns(3)
    c[0].metric("Rejections outside manual review", "0", help="Unreviewed quotes are never rejected")
    c[1].metric("Median insurer cost – reviewed", eur(q[q.manual_review].insurance_company_cost.median()))
    c[2].metric("Median insurer cost – auto", eur(q[~q.manual_review].insurance_company_cost.median()))

    st.subheader("Quotes far above the estimate are reviewed the *least*")
    v = q[q.estimated_repair_cost > 0].copy()
    v["ratio"] = v.contractor_cost / v.estimated_repair_cost
    v["gap"] = pd.cut(v.ratio, [0, .5, .8, 1.25, 2, 1e9],
                      labels=["<0.5×", "0.5–0.8×", "0.8–1.25× (close)", "1.25–2×", ">2×"])
    t = v.groupby("gap", observed=True).agg(quotes=("ratio", "size"), review_rate=("manual_review", "mean"))
    t["rejection_rate_if_reviewed"] = v[v.manual_review].groupby("gap", observed=True).rejected.mean()
    chart = alt.Chart(t.reset_index()).mark_bar(color=ORANGE).encode(
        x=alt.X("gap:N", sort=None, title="Contractor quote ÷ inspector estimate"),
        y=alt.Y("review_rate:Q", axis=alt.Axis(format="%"), title="Share manually reviewed"),
        tooltip=["gap", "quotes", alt.Tooltip("review_rate", format=".0%"),
                 alt.Tooltip("rejection_rate_if_reviewed", format=".0%")])
    st.altair_chart(chart, width="stretch")
    st.caption("Review follows claim size, not the gap between estimate and bill. "
               "Rejection rates are similar (6–11%) across all gap groups.")

    st.subheader("How the data was prepared")
    dedup = raw.drop_duplicates()
    multi = dedup.quote_id.value_counts()
    f = st.columns(4)
    f[0].metric("Raw rows", f"{len(raw):,}")
    f[1].metric("After removing exact duplicates", f"{len(dedup):,}", delta=f"−{len(raw) - len(dedup)} rows",
                delta_color="off", help="Rows identical in every column")
    f[2].metric("Quotes (one row each)", f"{len(q):,}",
                delta=f"{(multi > 1).sum()} quotes spanned {int(multi[multi > 1].sum()):,} rows", delta_color="off",
                help="Rows sharing a quote_id are coverage lines with the same contractor cost, decision and review "
                     "flag, so each quote is kept once")
    f[3].metric("Claims", f"{q.claim_id.nunique():,}", help="Train/test splits are made by claim, so no claim is on both sides")
    st.markdown(f"""
- **Conflicting rows within a quote:** {int(q.conflicting_rows.sum()):,} quotes have rows with different labels (e.g. the same building as House in one row and Commercial in another). Each category is resolved by majority vote (ties → first row), and `n_coverage_rows` / `conflicting_rows` are kept as model features. Keeping all labels as flags gave no accuracy gain.
- **Missing values and zeros are flagged, not filled in:** missing or zero estimate, missing building area / construction year.
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


# ------------------------------------------------------------------ 3. explore
elif page.startswith("3"):
    st.title("Explore the data")
    f1, f2, f3 = st.columns(3)
    dmg = f1.multiselect("Damage type", sorted(q.inspection_type.unique()), default=sorted(q.inspection_type.unique()))
    bld = f2.multiselect("Building type", sorted(q.building_type.unique()), default=sorted(q.building_type.unique()))
    rev = f3.selectbox("Review status", ["all", "manual review", "auto-processed"])
    d = q[q.inspection_type.isin(dmg) & q.building_type.isin(bld) & (q.estimated_repair_cost > 0)].copy()
    if rev != "all":
        d = d[d.manual_review == (rev == "manual review")]
    d["status"] = np.where(d.rejected == 1, "rejected", np.where(d.manual_review, "reviewed – approved", "auto-approved"))
    st.caption(f"{len(d):,} quotes selected")

    sc = alt.Chart(d).mark_circle(size=25, opacity=.5).encode(
        x=alt.X("estimated_repair_cost:Q", scale=alt.Scale(type="log"), title="Inspector estimate (€)"),
        y=alt.Y("contractor_cost:Q", scale=alt.Scale(type="log", domainMin=1000), title="Contractor quote (€)"),
        color=alt.Color("status:N", scale=alt.Scale(domain=["auto-approved", "reviewed – approved", "rejected"],
                                                    range=[GREY, BLUE, ORANGE])),
        tooltip=["claim_id", "quote_id", "inspection_type", "estimated_repair_cost", "contractor_cost", "status"])
    line = alt.Chart(pd.DataFrame({"x": [1e4, 4e5]})).mark_line(color="black", strokeDash=[4, 4]).encode(x="x", y="x")
    st.altair_chart((sc + line).interactive(), width="stretch")
    st.caption("Dashed line = bill equals estimate. Zoom and hover for details.")

    st.subheader("Inspection organisations (≥30 quotes)")
    d["ratio"] = d.contractor_cost / d.estimated_repair_cost
    org = (d[d.inspection_org != "OTHER"].groupby("inspection_org")
           .agg(quotes=("ratio", "size"), median_ratio=("ratio", "median"), review_rate=("manual_review", "mean"))
           .query("quotes >= 30").reset_index())
    if len(org):
        st.altair_chart(alt.Chart(org).mark_bar().encode(
            y=alt.Y("inspection_org:N", sort="-x", title=None),
            x=alt.X("median_ratio:Q", title="Median quote ÷ estimate"),
            color=alt.condition(alt.datum.median_ratio > 1.3, alt.value(ORANGE), alt.value(BLUE)),
            tooltip=["inspection_org", "quotes", alt.Tooltip("median_ratio", format=".2f"),
                     alt.Tooltip("review_rate", format=".0%")]), width="stretch")
    else:
        st.info("No organisation has 30+ quotes in this selection.")


# ------------------------------------------------------------------ 4. model results
elif page.startswith("4"):
    st.title("Model results (out-of-fold, grouped by claim)")
    a, b = st.columns(2)
    a.subheader("A · Fair-cost model")
    ma = M["m_a"]
    a.dataframe(pd.DataFrame({"model": ma.model, "MAE": ma.MAE_eur.map(eur),
                              "median % error": ma.median_abs_pct_err.map("{:.0%}".format),
                              "p90 coverage": ma.p90_coverage.map(lambda x: "–" if pd.isna(x) else f"{x:.0%}")}),
                hide_index=True, width="stretch")
    a.caption("Halves the error of using the inspector estimate as the expected cost.")
    imp = M["fair_imp"].head(10)
    a.altair_chart(alt.Chart(imp).mark_bar(color=BLUE).encode(
        y=alt.Y("feature:N", sort="-x", title=None),
        x=alt.X("share_pct:Q", title="Share of importance (%)"),
        tooltip=["feature", alt.Tooltip("share_pct", format=".1f", title="share %"),
                 alt.Tooltip("typical_error_increase_pct", format=".1f",
                             title="error increase if shuffled (%)")]).properties(
        height=340, title="What drives the fair cost (permutation importance)"), width="stretch")
    b.subheader("B · Rejection model")
    b.dataframe(M["m_b"][["score", "ROC_AUC", "PR_AUC"]].rename(columns={"ROC_AUC": "ROC AUC", "PR_AUC": "PR AUC"}),
                hide_index=True, width="stretch")
    b.caption(f"Base rate {M['m_b'].base_rate.iloc[0]:.1%} · {M['m_b'].n_rejected.iloc[0]} rejected quotes.")
    b.caption("Only slightly better than 'largest quote first'. With 70 positives, little signal is available.")

    st.subheader("Quote vs fair price: the gap is the signal")
    st.markdown("The fair-cost model only sees **inspection data, never the quote itself**, so the gap between "
                "the quote and the predicted price is informative. A more accurate model means a narrower normal "
                "range: real outliers stand out and fewer normal quotes are flagged by mistake.")
    g = q.assign(ratio=q.contractor_cost / q.fair_p50)
    g["gap"] = pd.cut(g.ratio, [0, .5, .8, 1.25, 2, 1e9],
                      labels=["<0.5×", "0.5–0.8×", "0.8–1.25× (close)", "1.25–2×", ">2×"])
    gt = g.groupby("gap", observed=True).agg(quotes=("ratio", "size"), review_rate=("manual_review", "mean"))
    gt["share"] = gt.quotes / len(g)
    gt["rejection_rate_if_reviewed"] = g[g.manual_review].groupby("gap", observed=True).rejected.mean()
    gt["excess_m"] = (g.assign(ex=(g.contractor_cost - g.fair_p50).clip(lower=0))
                       .groupby("gap", observed=True).ex.sum() / 1e6)
    gt = gt.reset_index()
    far = g[g.ratio > 2]
    far_auto = far[~far.manual_review]
    k = st.columns(3)
    k[0].metric("Quotes close to fair price (0.8–1.25×)", f"{(g.ratio.between(.8, 1.25)).mean():.0%}",
                help="Candidates for automatic approval")
    k[1].metric("Quotes > 2× fair price, approved without review", f"{len(far_auto)} of {len(far)}")
    k[2].metric("Their amount above fair price", f"€{(far_auto.contractor_cost - far_auto.fair_p50).sum() / 1e6:.0f}M",
                help="Candidate saving, not proven – needs an audit")
    gap_colors = alt.Scale(domain=list(gt.gap.astype(str)), range=[GREY, GREY, GREEN, ORANGE, "#c05621"])
    c1, c2 = st.columns(2)
    c1.altair_chart(alt.Chart(gt.assign(gap=gt.gap.astype(str))).mark_bar().encode(
        x=alt.X("gap:N", sort=None, title="Quote ÷ fair price (p50)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("share:Q", axis=alt.Axis(format="%"), title="Share of quotes"),
        color=alt.Color("gap:N", scale=gap_colors, legend=None),
        tooltip=["gap", "quotes", alt.Tooltip("share", format=".0%"), alt.Tooltip("review_rate", format=".0%"),
                 alt.Tooltip("rejection_rate_if_reviewed", format=".1%")]).properties(
        height=300, title="Share of quotes"), width="stretch")
    c2.altair_chart(alt.Chart(gt.assign(gap=gt.gap.astype(str))).mark_bar().encode(
        x=alt.X("gap:N", sort=None, title="Quote ÷ fair price (p50)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("excess_m:Q", title="€M"),
        color=alt.Color("gap:N", scale=gap_colors, legend=None),
        tooltip=["gap", alt.Tooltip("excess_m", format=".1f", title="€M above fair price"),
                 alt.Tooltip("review_rate", format=".0%", title="reviewed today")]).properties(
        height=300, title="Amount above fair price (€M)"), width="stretch")
    st.caption("Close quotes → auto-approve. Far above → review. Today quotes far above the fair price are reviewed "
               "no more often than normal ones (≈26% vs 21%). Rejection rate when reviewed: ≈8% for close quotes vs "
               "≈13% far above – a real but weak signal.")

    st.subheader("Review effort vs rejected € caught")
    scores = {"model: P(reject) × quote": r.priority,
              "baseline: largest quote": r.contractor_cost,
              "baseline: quote ÷ estimate": r.log_quote_to_estimate.fillna(0)}
    curves = pd.concat([capture_curve(r.contractor_cost, r.rejected, s).assign(score=n) for n, s in scores.items()])
    budget = st.slider("Review budget (share of reviewed quotes)", 0.05, 1.0, 0.20, 0.05)
    base = alt.Chart(curves).mark_line(strokeWidth=2.5).encode(
        x=alt.X("share_reviewed:Q", axis=alt.Axis(format="%"), title="Share of quotes reviewed"),
        y=alt.Y("share_value_caught:Q", axis=alt.Axis(format="%"), title="Share of rejected € caught"),
        color=alt.Color("score:N", scale=alt.Scale(domain=list(scores), range=[ORANGE, BLUE, GREEN]),
                        legend=alt.Legend(orient="bottom", title=None)))
    rule = alt.Chart(pd.DataFrame({"x": [budget]})).mark_rule(color="black", strokeDash=[4, 4]).encode(x="x")
    st.altair_chart(base + rule, width="stretch")
    caught = {n: c.iloc[int(np.ceil(budget * len(c))) - 1].share_value_caught
              for n, c in curves.groupby("score")}
    cols = st.columns(len(caught))
    for col, (n, val) in zip(cols, caught.items()):
        col.metric(n, f"{val:.0%}")

    with st.expander("Rejection model coefficients (feature importance for model B)"):
        st.caption("Standardised logistic-regression coefficients: odds ratio per 1 SD (numeric) or vs. other "
                   "categories. Above 1 = more likely to be rejected. Based on only 70 rejections – read as directions.")
        st.dataframe(M["coefs"], hide_index=True, width="stretch")


# ------------------------------------------------------------------ 5. quote checker
elif page.startswith("5"):
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
            insurance_excess_amount=excess, n_coverage_rows=1,
            quote_seq=1 if quote_event == "preliminary quote" else 2,
            est_missing=int(est == 0), est_zero=0, area_missing=int(area == 0), year_missing=int(year == 0),
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


# ------------------------------------------------------------------ 6. review queue
elif page.startswith("6"):
    st.title("Review queue")
    st.caption("All historical quotes scored by the models, ranked by expected € at risk.")
    share = st.slider("Review capacity (share of all quotes)", 0.05, 0.6, 0.23, 0.01)
    ranked = q.sort_values("priority_eur", ascending=False).reset_index(drop=True)
    n = int(share * len(ranked))
    top = ranked.head(n)
    c = st.columns(4)
    c[0].metric("Quotes to review", f"{n:,}")
    c[1].metric("Of which currently auto-approved", f"{(~top.manual_review).sum():,}")
    c[2].metric("Quoted € in queue", eur(top.contractor_cost.sum()))
    rej_eur = q.loc[q.rejected == 1, "contractor_cost"].sum()
    c[3].metric("Known rejections in queue", f"{top.rejected.sum()} / {q.rejected.sum()}",
                delta=f"{top.loc[top.rejected == 1, 'contractor_cost'].sum() / rej_eur:.0%} of rejected €",
                delta_color="off",
                help="Rejections only exist where someone reviewed, so today's process 'catches' 100% by definition. "
                     "Compare by € and look at the four groups below.")

    st.subheader("Model queue vs today's reviews")
    in_q = q.quote_id.isin(top.quote_id)
    groups = [("A · In queue, reviewed today", in_q & q.manual_review, "both agree: review"),
              ("B · In queue, auto-approved today", in_q & ~q.manual_review, "model ADDS these"),
              ("C · Not in queue, reviewed today", ~in_q & q.manual_review, "model DROPS these"),
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
    st.info(f"**Reading it:** the overlap (A, {len(a_)} quotes) holds "
            f"**{a_.loc[a_.rejected == 1, 'contractor_cost'].sum() / rej_eur:.0%} of all rejected €** "
            f"(rejection rate {a_.rejected.mean():.0%} vs {q[q.manual_review].rejected.mean():.0%} overall). "
            f"To free capacity, the model stops reviewing {len(c_)} smaller, mostly normally priced quotes (C) that held only "
            f"{eur(c_.loc[c_.rejected == 1, 'contractor_cost'].sum())} of rejections, and instead reviews {len(b_)} larger "
            f"quotes (B, median {eur(b_.contractor_cost.median()) if len(b_) else '–'}) that nobody checked. "
            "What B would yield is unknown – a pilot or random audit would measure it.")

    st.subheader("The queue")

    only_auto = st.checkbox("Show only quotes that were auto-approved in reality")
    show = top[~top.manual_review] if only_auto else top
    show = show.assign(above_p90=show.contractor_cost > show.fair_p90)
    cols = ["claim_id", "quote_id", "quote_event", "inspection_type", "estimated_repair_cost", "contractor_cost",
            "fair_p50", "fair_p90", "above_p90", "p_reject", "priority_eur", "manual_review", "decision_on_quote"]
    st.dataframe(show[cols].round(2), hide_index=True, width="stretch", height=480,
                 column_config={"p_reject": st.column_config.ProgressColumn("P(reject)", min_value=0, max_value=1, format="%.2f"),
                                "priority_eur": st.column_config.NumberColumn("€ at risk", format="€%.0f"),
                                "estimated_repair_cost": st.column_config.NumberColumn("estimate", format="€%.0f"),
                                "contractor_cost": st.column_config.NumberColumn("quote", format="€%.0f"),
                                "fair_p50": st.column_config.NumberColumn("fair p50", format="€%.0f"),
                                "fair_p90": st.column_config.NumberColumn("fair p90", format="€%.0f"),
                                "above_p90": st.column_config.CheckboxColumn("above fair p90")})
    st.download_button("Download queue (CSV)", show[cols].to_csv(index=False), "review_queue.csv", "text/csv")


# ------------------------------------------------------------------ 7. verdict & next steps
elif page.startswith("7"):
    st.title("Verdict & next steps")

    st.subheader("Does ML help reduce future claims costs? Partly.")
    a, b = st.columns(2)
    a.success("""**Where it helps (supported by the data)**
- **Better price benchmark:** the fair-cost model halves the error of the inspector estimate (≈€22k vs €41k).
- **Smarter review effort:** about half of quotes are close to the fair price and could be auto-approved, while 220 of 297 quotes priced over 2× their fair cost went through without review (≈€17M above the fair price).
- **Partner insight:** inspection orgs' estimates range from ≈0.5× to 1.8× the final bill, which is a lever even without ML.""")
    b.warning("""**Where it doesn't (yet)**
- **Rejection model ≈ simple rule:** it catches 79% vs 77% of rejected € at a 20% review budget.
- **Savings not proven:** there's no "correct price" label and auto-approved quotes were never checked.
- **Biased labels:** reviews mostly targeted large claims.
- **Algorithm isn't the bottleneck:** LightGBM tested as a robustness check, no significant gain over the current model.""")
    st.info("**Conclusion:** ML gives a better cost benchmark and steers review effort, but today its edge over simple "
            "rules is small and savings can't be proven. The fastest way to unlock value is **better data**: a random "
            "audit of auto-approved quotes, and recording the approved amount after review.")

    st.subheader("Approaches I would explore with more time")
    t1, t2, t3, t4 = st.tabs(["📋 Better data & labels", "🧠 Better models", "🤝 Beyond single claims", "🚀 Putting it into practice"])
    t1.markdown("""*Highest impact.*
- **Random audit** of ~5% of auto-approved quotes, to get unbiased labels and measure true savings.
- **Record the approved amount after review**, so savings become a regression target.
- **Add dates**, to check price changes over time and validate on later claims instead of random splits.
- **Itemised quote lines** (materials, labour, hours), to flag specific inflated items, not just a high total.""")
    t2.markdown("""
- **Image models** on inspection photos to estimate damage severity, instead of only counting photos.
- **Text from inspection reports and quotes** (NLP / LLMs) to extract the scope of work.
- **Contractor anomaly detection:** contractors whose prices regularly sit above the fair cost.
- **Cost-sensitive review rule:** review only when expected saving > cost of a review.
- **Calibrated price ranges** with conformal prediction (p90 currently covers 83%, target 90%).""")
    t3.markdown("""
- **Feedback to inspection organisations** on how accurate their estimates are.
- **Steer claims to preferred contractors** at inspection time when the predicted cost is high.
- **Watch preliminary → final quote changes**, to catch costs that creep upward.""")
    t4.markdown("""
- **Pilot / A/B test:** route half of new quotes with the model and half with today's process; compare € saved.
- **Monitoring:** data drift, fairness across contractors and regions, and behaviour changes once contractors know the rules.""")

