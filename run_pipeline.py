"""End-to-end pipeline:  python run_pipeline.py [path/to/tech_data_2026.xlsx]   (default: data/tech_data_2026.xlsx)

1. data-quality report          -> outputs/data_quality.csv
2. cleaning + quote-level table (rejection model) + claim-level table (fair-cost model)
3. EDA figures and tables       -> outputs/figures/, outputs/eda_*.csv
4. Model A  fair-cost model (one quote per claim) -> outputs/fair_cost_metrics.csv, fair_cost_importance.csv,
                                   figures/06_fair_cost_importance.png
5. Model B  rejection model      -> outputs/rejection_metrics.csv, rejection_coefficients.csv
6. Business evaluation          -> outputs/review_budget_table.csv, figures/05_capture_curve.png
7. Score every quote            -> outputs/scored_quotes.csv (review priority list)
8. Primary rule vs fair-cost    -> outputs/rule_table.csv, rule_vs_fair_cost.csv, queue_overlap.csv,
                                   figures/07_rule_vs_fair_cost.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data import clean, data_quality_report, load_raw, to_claim_level, to_quote_level
from src.eda import run_eda
from src.evaluate import budget_table, plot_capture, plot_head_to_head, plot_importance
from src.features import add_quote_features
from src.models import (fair_cost_importance, fair_cost_oof, fit_final_rejection_model,
                        rejection_oof, score_all)
from src.triage import add_rule, head_to_head, queue_overlap, rule_table

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def main(path=C.DATA_PATH):
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw(path)
    dq = data_quality_report(raw)
    dq.to_csv(C.OUTPUT_DIR / "data_quality.csv", index=False)
    print("\n== Data quality ==\n", dq.to_string(index=False))

    cleaned = clean(raw)
    q = add_quote_features(to_quote_level(cleaned))
    q["rejected"] = q.decision_on_quote.eq("rejected").astype(int)
    claims = to_claim_level(cleaned)
    print(f"\nQuote-level table: {len(q)} quotes, {q.claim_id.nunique()} claims | claim-level table: {len(claims)} claims")

    for name, t in run_eda(q).items():
        t.to_csv(C.OUTPUT_DIR / f"eda_{name}.csv")
        print(f"\n== EDA: {name} ==\n", t.round(3).to_string())

    # Model A - fair cost, trained on one quote per claim; every quote is scored out-of-fold
    fair_q, fair_claims, m_a = fair_cost_oof(claims, q)
    q = add_rule(q.join(fair_q))
    m_a.to_csv(C.OUTPUT_DIR / "fair_cost_metrics.csv", index=False)
    print("\n== Model A: fair-cost model (out-of-fold) ==\n", m_a.to_string(index=False))

    imp_a = fair_cost_importance(claims)
    imp_a.to_csv(C.OUTPUT_DIR / "fair_cost_importance.csv", index=False)
    plot_importance(imp_a)
    print("\n== Model A feature importance (out-of-fold permutation) ==\n",
          imp_a[["feature", "share_pct", "typical_error_increase_pct"]].head(10).to_string(index=False))

    # Model B - rejection on reviewed quotes
    r = q[q.manual_review].copy()
    r["p_reject"], m_b = rejection_oof(r)
    m_b.to_csv(C.OUTPUT_DIR / "rejection_metrics.csv", index=False)
    print("\n== Model B: rejection model on reviewed quotes (out-of-fold) ==\n", m_b.to_string(index=False))

    # Business view: rank by expected rejected value
    r["priority"] = r.p_reject * r.contractor_cost
    scores = {
        "model: p_reject x quote": r.priority,
        "baseline: largest quote": r.contractor_cost,
        "baseline: quote/estimate": r.log_quote_to_estimate.fillna(0),
    }
    bt = budget_table(r, scores)
    bt.to_csv(C.OUTPUT_DIR / "review_budget_table.csv")
    plot_capture(r, scores)
    print("\n== Share of rejected € caught at a given review budget ==\n", bt.round(3).to_string())

    model, imp = fit_final_rejection_model(r)
    imp.to_csv(C.OUTPUT_DIR / "rejection_coefficients.csv", index=False)
    print("\n== Rejection model coefficients (standardised; odds ratio per 1 SD / category) ==\n", imp.head(10).to_string(index=False))

    # Score every quote -> triage list. Scores on auto-processed quotes are an extrapolation
    # (model trained on the reviewed population only) and must be validated with a random audit.
    q["p_reject"] = score_all(model, q)
    q.loc[r.index, "p_reject"] = r.p_reject          # keep honest OOF scores where available
    q["priority_eur"] = q.p_reject * q.contractor_cost
    q["suggested_action"] = np.where(
        (q.priority_eur >= q.priority_eur.quantile(.8)) | (q.above_p90 == 1), "manual review", "auto-approve")
    cols = ["claim_id", "quote_id", "quote_event", "inspection_type", "inspection_org",
            "estimated_repair_cost", "contractor_cost", "fair_p50", "fair_p90", "excess_vs_fair",
            "p_reject", "priority_eur", "suggested_action", "manual_review", "decision_on_quote"]
    q.sort_values("priority_eur", ascending=False)[cols].round(2).to_csv(
        C.OUTPUT_DIR / "scored_quotes.csv", index=False)

    auto = q[~q.manual_review]
    print(f"\nAuto-processed quotes flagged for review: {(auto.suggested_action == 'manual review').mean():.1%}")
    # € above fair cost per claim (latest quote only), so revised quotes of one repair are not counted twice
    ac = fair_claims[~fair_claims.manual_review]
    print(f"Auto-processed claims (one quote each) above fair-cost p90: {ac.above_p90.mean():.1%}"
          f"  | sum of excess over fair p50 on those: €{ac.loc[ac.above_p90 == 1, 'excess_vs_fair'].sum():,.0f}")

    # Primary rule vs secondary fair-cost model, head to head
    rt = rule_table(q)
    rt.to_csv(C.OUTPUT_DIR / "rule_table.csv", index=False)
    print(f"\n== Primary rule: review if bill > k × estimate ==\n", rt.to_string(index=False))
    h2h, curves = head_to_head(r)
    h2h.to_csv(C.OUTPUT_DIR / "rule_vs_fair_cost.csv", index=False)
    plot_head_to_head(curves)
    print("\n== Head-to-head on reviewed quotes with an estimate (out-of-fold) ==\n", h2h.to_string(index=False))
    ov = queue_overlap(q)
    ov.to_csv(C.OUTPUT_DIR / "queue_overlap.csv", index=False)
    print(f"\n== Rule (k={C.RULE_K}) vs secondary at the same budget, all quotes ==\n", ov.to_string(index=False))
    print(f"\nDone. Outputs in {C.OUTPUT_DIR}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else C.DATA_PATH)
