"""Business evaluation: how much rejected value is caught for a given review budget."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C


def save_fig(fig, name: str):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=150)
    plt.close(fig)


def capture_curve(value: pd.Series, flag: pd.Series, score: pd.Series) -> pd.DataFrame:
    order = score.sort_values(ascending=False, kind="mergesort").index
    v = (value * flag).loc[order]
    return pd.DataFrame({"share_reviewed": np.arange(1, len(v) + 1) / len(v),
                         "share_value_caught": v.cumsum().values / v.sum(),
                         "share_cases_caught": flag.loc[order].cumsum().values / flag.sum()})


def budget_table(r: pd.DataFrame, scores: dict[str, pd.Series], budgets=(.1, .2, .3, .5)):
    rows = []
    for name, s in scores.items():
        cc = capture_curve(r.contractor_cost, r.rejected, s)
        for b in budgets:
            row = cc.iloc[int(np.ceil(b * len(cc))) - 1]
            rows.append({"score": name, "review_budget": b,
                         "rejected_value_caught": round(row.share_value_caught, 3),
                         "rejected_cases_caught": round(row.share_cases_caught, 3)})
    return pd.DataFrame(rows).pivot_table(index=["review_budget"], columns="score",
                                          values="rejected_value_caught")


def plot_capture(r: pd.DataFrame, scores: dict[str, pd.Series]):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for (name, s), col in zip(scores.items(), ["#dd6b20", "#2b6cb0", "#38a169", "#805ad5"]):
        cc = capture_curve(r.contractor_cost, r.rejected, s)
        ax.plot(cc.share_reviewed, cc.share_value_caught, label=name, color=col, lw=2)
    ax.plot([0, 1], [0, 1], "--", color="#a0aec0", label="random")
    ax.set(xlabel="Share of quotes reviewed (ranked by score)",
           ylabel="Share of rejected € caught",
           title="Review triage: value caught vs review effort\n(manually reviewed quotes, out-of-fold)")
    ax.legend(loc="lower right")
    save_fig(fig, "05_capture_curve.png")


def plot_importance(imp: pd.DataFrame, top: int = 12):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(d.feature, d.share_pct, color="#2b6cb0")
    for y, v in enumerate(d.share_pct):
        ax.text(v + .5, y, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, d.share_pct.max() * 1.12)
    ax.set(xlabel="Share of importance (%)",
           title="Fair-cost model: what drives the expected repair cost\n(out-of-fold permutation importance)")
    save_fig(fig, "06_fair_cost_importance.png")


def plot_head_to_head(curves: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for (name, cc), col in zip(curves.groupby("approach", sort=False), ["#dd6b20", "#2b6cb0", "#718096"]):
        ax.plot(cc.share_reviewed, cc.share_value_caught, label=name, color=col, lw=2,
                ls=":" if name.startswith("reference") else "-")
    ax.plot([0, 1], [0, 1], "--", color="#a0aec0", label="random")
    ax.set(xlabel="Share of quotes reviewed (ranked by score)", ylabel="Share of rejected € caught",
           title="Primary rule vs secondary model\n(reviewed quotes with an estimate, out-of-fold)")
    ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, "07_rule_vs_fair_cost.png")
