"""Primary triage: a transparent rule on the bill vs the inspector estimate.
Secondary: the fair-cost model (Model A), alone and combined with the rejection model (Model B).
Head-to-head: both rank the same reviewed quotes; which catches more rejected €?"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .evaluate import capture_curve


def add_rule(q: pd.DataFrame, k: float = C.RULE_K) -> pd.DataFrame:
    """Rule: review if bill > k × estimate. Quotes without a usable estimate (missing or 0) can't be scored."""
    q = q.copy()
    est = q.estimated_repair_cost.where(q.estimated_repair_cost > 0)
    q["rule_ratio"] = q.contractor_cost / est
    q["overrun"] = q.contractor_cost - est
    q["severe_overrun"] = (q.overrun > C.SEVERE_OVERRUN_EUR).astype(int)
    q["rule_flag"] = (q.rule_ratio > k).astype(int)
    return q


def rule_table(q: pd.DataFrame, triggers=C.RULE_TRIGGERS) -> pd.DataFrame:
    """Per trigger k: how many quotes the rule sends to review and what it captures (quotes with an estimate)."""
    d = q[q.rule_ratio.notna()]
    rows = []
    for k in triggers:
        f = d.rule_ratio > k
        rows.append({"trigger_k": k, "quotes_flagged": int(f.sum()), "flagged_share": f.mean(),
                     "severe_overruns_caught": d.severe_overrun[f].sum() / d.severe_overrun.sum(),
                     "overrun_eur_flagged": d.overrun[f].clip(lower=0).sum(),
                     "rejected_caught_if_reviewed": d.rejected[f & d.manual_review].sum()
                                                    / d.rejected[d.manual_review].sum()})
    return pd.DataFrame(rows).round(3)


def h2h_scores(r: pd.DataFrame) -> dict[str, pd.Series]:
    """The scores compared head to head, all on the same rows (higher = review first)."""
    return {"primary: rule  bill ÷ estimate": r.rule_ratio,
            "secondary: fair-cost + P(reject) × bill": r.p_reject * r.contractor_cost,
            "reference: largest bill first": r.contractor_cost}


def head_to_head(r: pd.DataFrame, budgets=(.1, .2, .3, .5)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the scores on reviewed quotes with a usable estimate (the rule can't score the rest).

    Returns (summary table with ROC/PR-AUC and rejected € caught per budget, capture curves for plotting)."""
    r = r[r.rule_ratio.notna()]
    rows, curves = [], []
    for name, s in h2h_scores(r).items():
        cc = capture_curve(r.contractor_cost, r.rejected, s)
        row = {"approach": name, "ROC_AUC": roc_auc_score(r.rejected, s),
               "PR_AUC": average_precision_score(r.rejected, s)}
        for b in budgets:
            row[f"rejected_eur_caught@{b:.0%}"] = cc.share_value_caught.iloc[int(np.ceil(b * len(cc))) - 1]
        rows.append(row)
        curves.append(cc.assign(approach=name))
    out = pd.DataFrame(rows).round(3)
    out["n_quotes"], out["n_rejected"] = len(r), int(r.rejected.sum())
    return out, pd.concat(curves, ignore_index=True)


def queue_overlap(q: pd.DataFrame, k: float = C.RULE_K) -> pd.DataFrame:
    """Budget-matched: the rule flags a share s of quotes; how much does the secondary top-s agree with it?"""
    d = q[q.rule_ratio.notna()]
    s = d.rule_flag.mean()
    sec = d.priority_eur >= d.priority_eur.quantile(1 - s)
    rule = d.rule_flag.eq(1)
    return pd.DataFrame({
        "metric": ["review budget (share of quotes with estimate)", "flagged by both", "rule only",
                   "secondary only", "agreement (Jaccard)",
                   "€ above fair p90 caught: rule", "€ above fair p90 caught: secondary",
                   "quotes without estimate (rule can't score)"],
        "value": [s, (rule & sec).mean(), (rule & ~sec).mean(), (~rule & sec).mean(),
                  (rule & sec).sum() / (rule | sec).sum(),
                  _p90_excess(d, rule), _p90_excess(d, sec),
                  q.rule_ratio.isna().mean()]}).round(3)


def _p90_excess(d: pd.DataFrame, flag: pd.Series) -> float:
    ex = (d.contractor_cost - d.fair_p50).clip(lower=0) * d.above_p90
    return ex[flag].sum() / ex.sum()


def queue_compare(r: pd.DataFrame, budget: float,
                  include_reference: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What each approach's queue holds when both review the same share of reviewed quotes.

    Returns (one row per approach, the quotes with an in-queue flag per approach for plotting)."""
    r = r[r.rule_ratio.notna()].copy()
    n = int(np.ceil(budget * len(r)))
    scores = {k: v for k, v in h2h_scores(r).items() if include_reference or not k.startswith("reference")}
    rows, flags = [], {}
    rej_eur = r.contractor_cost[r.rejected == 1].sum()
    for name, s in scores.items():
        inq = r.index.isin(s.sort_values(ascending=False, kind="mergesort").index[:n])
        t = r[inq]
        flags[name] = inq
        rows.append({"approach": name, "quotes_reviewed": n, "median_bill": t.contractor_cost.median(),
                     "median_bill_to_estimate": t.rule_ratio.median(),
                     "rejections_caught": int(t.rejected.sum()), "rejections_total": int(r.rejected.sum()),
                     "rejected_eur_caught": t.contractor_cost[t.rejected == 1].sum() / rej_eur})
    pts = r[["quote_id", "contractor_cost", "rule_ratio", "rejected"]].assign(**{f"in_queue::{k}": v for k, v in flags.items()})
    return pd.DataFrame(rows), pts
