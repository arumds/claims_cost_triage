"""Model A: 'fair cost' quantile gradient boosting.  Model B: regularised logistic rejection model."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from . import config as C
from .features import feature_matrix


def _regressor(quantile: float):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, max_iter=400, learning_rate=0.05,
        max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
        categorical_features="from_dtype", random_state=C.RANDOM_STATE)



# ---------------------------------------------------------------- Model A
def fair_cost_oof(q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold predictions of the median (p50) and upper (p90) 'fair' quote.

    Uses only inspection-time features. Trained on approved quotes, folds grouped by claim
    so no claim leaks between train and test.
    """
    X, _ = feature_matrix(q, include_quote=False)
    y = np.log1p(q.contractor_cost.values)
    out = pd.DataFrame(index=q.index, columns=["fair_p50", "fair_p90"], dtype=float)
    for tr, te in GroupKFold(C.N_FOLDS).split(X, y, groups=q.claim_id):
        tr = tr[q.rejected.values[tr] == 0]
        for qt, col in [(.5, "fair_p50"), (.9, "fair_p90")]:
            m = _regressor(qt).fit(X.iloc[tr], y[tr])
            out.iloc[te, out.columns.get_loc(col)] = np.expm1(m.predict(X.iloc[te]))

    res = q[["contractor_cost", "estimated_repair_cost"]].join(out)
    res["excess_vs_fair"] = res.contractor_cost - res.fair_p50
    res["above_p90"] = (res.contractor_cost > res.fair_p90).astype(int)

    # metrics vs naive baseline "quote = inspector estimate"
    ok = res.estimated_repair_cost.notna()
    metrics = pd.DataFrame({
        "model": ["baseline: quote = estimate", "fair-cost model (p50)"],
        "MAE_eur": [mean_absolute_error(res.contractor_cost[ok], res.estimated_repair_cost[ok]),
                    mean_absolute_error(res.contractor_cost[ok], res.fair_p50[ok])],
        "median_abs_pct_err": [
            np.median(np.abs(res.estimated_repair_cost[ok] / res.contractor_cost[ok].clip(lower=1) - 1)),
            np.median(np.abs(res.fair_p50[ok] / res.contractor_cost[ok].clip(lower=1) - 1))],
        "p90_coverage": [np.nan, 1 - res.above_p90.mean()],   # should be ~0.90
    }).round(3)
    return res, metrics


# ---------------------------------------------------------------- Model B
# Only ~70 rejected quotes exist (all in the manually reviewed set), so a strongly regularised,
# low-variance model generalises better than boosting. inspection_org (≈60 levels) is left out
# because with so few positives it overfits (tested: gradient boosting with org scored below the
# 'largest quote first' baseline out-of-fold).
REJ_CATS = ["inspection_type", "insurance_type", "building_type", "quote_event",
            "building_geographical_location"]
REJ_NUMS = ["log_quote", "log_estimate", "log_quote_to_fair", "inspection_images",
            "insurance_excess_amount", "quote_seq", "n_coverage_rows", "est_missing", "est_zero"]


def _rejection_frame(r: pd.DataFrame) -> pd.DataFrame:
    X = r[REJ_CATS].copy()
    X["log_quote"] = np.log1p(r.contractor_cost)
    X["log_estimate"] = np.log1p(r.estimated_repair_cost.fillna(0))
    X["log_quote_to_fair"] = np.log1p(r.contractor_cost) - np.log1p(r.fair_p50)
    for c in ["inspection_images", "insurance_excess_amount", "quote_seq", "n_coverage_rows",
              "est_missing", "est_zero"]:
        X[c] = r[c]
    return X


def _rejection_model(C_reg: float = 0.1):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), REJ_CATS),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), REJ_NUMS),
    ])
    return make_pipeline(pre, LogisticRegression(C=C_reg, class_weight="balanced", max_iter=2000))


def rejection_oof(r: pd.DataFrame, n_repeats: int = 5) -> tuple[pd.Series, pd.DataFrame]:
    """OOF rejection probabilities on manually reviewed quotes (the only labelled set),
    averaged over repeated grouped CV to reduce variance from the tiny positive class."""
    X, y = _rejection_frame(r), r.rejected.values
    p = np.zeros(len(r))
    for rep in range(n_repeats):
        cv = StratifiedGroupKFold(C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE + rep)
        for tr, te in cv.split(X, y, groups=r.claim_id):
            p[te] += _rejection_model().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1] / n_repeats
    p = pd.Series(p, index=r.index, name="p_reject")

    def _m(name, s):
        s = pd.Series(s, index=r.index).fillna(0)
        return {"score": name, "ROC_AUC": roc_auc_score(y, s), "PR_AUC": average_precision_score(y, s)}

    metrics = pd.DataFrame([
        _m("model p_reject", p),
        _m("baseline: largest quote", r.contractor_cost),
        _m("baseline: quote/estimate ratio", r.log_quote_to_estimate),
        _m("baseline: quote vs fair-cost model", X.log_quote_to_fair),
    ]).round(3)
    metrics["base_rate"] = round(y.mean(), 3)
    metrics["n_rejected"] = int(y.sum())
    return p, metrics


def fit_final_rejection_model(r: pd.DataFrame):
    X = _rejection_frame(r)
    model = _rejection_model().fit(X, r.rejected)
    names = model[0].get_feature_names_out()
    coefs = model[-1].coef_[0]
    importance = (pd.DataFrame({"feature": names, "coef_std_units": coefs,
                                "odds_ratio": np.exp(coefs)})
                  .reindex(np.argsort(-np.abs(coefs))).round(3))
    return model, importance


def score_all(model, q: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(_rejection_frame(q))[:, 1], index=q.index, name="p_reject")


# ---------------------------------------------------------------- final models for the app
def fit_fair_cost_models(q: pd.DataFrame) -> dict:
    """Fit p50/p90 fair-cost models on all approved quotes (for scoring new quotes)."""
    X, _ = feature_matrix(q, include_quote=False)
    keep = q.rejected.values == 0
    y = np.log1p(q.contractor_cost.values)
    models = {qt: _regressor(qt).fit(X[keep], y[keep]) for qt in (.5, .9)}
    categories = {c: X[c].cat.categories for c in C.CATEGORICAL}
    return {"models": models, "categories": categories}


def predict_fair(fair: dict, new: pd.DataFrame) -> pd.DataFrame:
    """Predict fair_p50 / fair_p90 for new quotes (categories aligned to training)."""
    X, _ = feature_matrix(new, include_quote=False)
    for c, cats in fair["categories"].items():
        X[c] = pd.Categorical(new[c].where(new[c].isin(cats), "OTHER" if "OTHER" in cats else cats[0]),
                              categories=cats)
    return pd.DataFrame({"fair_p50": np.expm1(fair["models"][.5].predict(X)),
                         "fair_p90": np.expm1(fair["models"][.9].predict(X))}, index=new.index)


# ---------------------------------------------------------------- feature importance (Model A)
def fair_cost_importance(q: pd.DataFrame, n_repeats: int = 3) -> pd.DataFrame:
    """Out-of-fold permutation importance for the p50 fair-cost model.

    For each claim-grouped fold: fit on the training part (approved quotes), then shuffle one
    feature at a time in the held-out part and measure how much the error (MAE of log cost)
    increases. Averaged over folds. Bigger increase = the model relies more on that feature.
    """
    from sklearn.inspection import permutation_importance
    X, _ = feature_matrix(q, include_quote=False)
    y = np.log1p(q.contractor_cost.values)
    imps = []
    for tr, te in GroupKFold(C.N_FOLDS).split(X, y, groups=q.claim_id):
        tr = tr[q.rejected.values[tr] == 0]
        m = _regressor(.5).fit(X.iloc[tr], y[tr])
        res = permutation_importance(m, X.iloc[te], y[te], scoring="neg_mean_absolute_error",
                                     n_repeats=n_repeats, random_state=C.RANDOM_STATE)
        imps.append(res.importances_mean)
    imp = pd.Series(np.mean(imps, axis=0), index=X.columns).clip(lower=0)
    out = pd.DataFrame({"feature": imp.index,
                        "mae_increase_log": imp.values,
                        # an increase of d in log-MAE ≈ typical error grows by (e^d - 1)
                        "typical_error_increase_pct": (np.expm1(imp.values) * 100),
                        "share_pct": imp.values / imp.sum() * 100})
    return out.sort_values("share_pct", ascending=False).round(3).reset_index(drop=True)
