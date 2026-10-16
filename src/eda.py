"""Exploratory plots and tables used for the presentation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#a0aec0"


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=150)
    plt.close(fig)


def run_eda(q: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    v = q[(q.estimated_repair_cost > 0)].copy()
    v["ratio"] = v.contractor_cost / v.estimated_repair_cost

    # 1. Inspector estimate vs contractor quote
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(v.estimated_repair_cost, v.contractor_cost, s=6, alpha=.35, color=BLUE)
    lim = [1e3, 5e5]
    ax.plot(lim, lim, color=ORANGE, lw=1.5, label="quote = estimate")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel="Inspector estimate (€)", ylabel="Contractor quote (€)",
           title="Estimates are coarse and only loosely predict quotes")
    ax.legend()
    _save(fig, "01_estimate_vs_quote.png")

    # 2. Quote/estimate ratio by damage type
    order = v.groupby("inspection_type").ratio.median().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([v.loc[v.inspection_type == t, "ratio"].clip(upper=5) for t in order],
               tick_labels=order, vert=False, showfliers=False)
    ax.axvline(1, color=ORANGE, lw=1)
    ax.set(xlabel="Contractor quote / inspector estimate", title="Estimate accuracy by damage type")
    _save(fig, "02_ratio_by_damage_type.png")

    # 3. Inspection organisations differ systematically
    org = (v[v.inspection_org != "OTHER"].groupby("inspection_org")
           .agg(n=("ratio", "size"), median_ratio=("ratio", "median"))
           .query("n >= 30").sort_values("median_ratio"))
    tables["inspection_org_ratio"] = org
    fig, ax = plt.subplots(figsize=(7, max(3, .25 * len(org))))
    ax.barh(org.index, org.median_ratio, color=[ORANGE if r > 1.3 else BLUE for r in org.median_ratio])
    ax.axvline(1, color="black", lw=.8)
    ax.set(xlabel="Median quote / estimate", title="Inspection orgs (≥30 quotes)")
    _save(fig, "03_inspection_org_ratio.png")

    # 4. Review is not random: big claims are reviewed
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.logspace(3, np.log10(3.2e5), 40)
    for flag, col, lab in [(False, GREY, "auto"), (True, ORANGE, "manual review")]:
        ax.hist(q.loc[q.manual_review == flag, "insurance_company_cost"].clip(lower=1e3),
                bins=bins, alpha=.6, color=col, label=lab, density=True)
    ax.set(xscale="log", xlabel="Insurer cost (€)", ylabel="density",
           title="Manual review targets large claims (selection bias)")
    ax.legend()
    _save(fig, "04_review_selection.png")

    # 5. Rejection rate (reviewed quotes only) by segment
    r = q[q.manual_review]
    seg = (r.groupby("inspection_type").agg(n=("rejected", "size"), rejection_rate=("rejected", "mean"))
           .sort_values("rejection_rate"))
    tables["rejection_by_damage_type"] = seg
    tables["review_vs_auto"] = q.groupby("manual_review").agg(
        quotes=("quote_id", "size"),
        median_quote=("contractor_cost", "median"),
        median_insurer_cost=("insurance_company_cost", "median"),
        rejection_rate=("rejected", "mean"))
    return tables
