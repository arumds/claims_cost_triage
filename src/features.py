"""Feature construction (only information available when the quote arrives)."""
import numpy as np
import pandas as pd

from . import config as C


def add_quote_features(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    est = q.estimated_repair_cost.where(q.estimated_repair_cost > 0)
    q["log_quote_to_estimate"] = np.log1p(q.contractor_cost) - np.log1p(est)
    q["quote_minus_estimate"] = q.contractor_cost - est
    return q


def feature_matrix(q: pd.DataFrame, include_quote: bool) -> tuple[pd.DataFrame, list[bool]]:
    cols = C.CATEGORICAL + C.NUMERIC_PRE_QUOTE + (C.QUOTE_FEATURES if include_quote else [])
    X = q[cols].copy()
    for c in C.CATEGORICAL:
        X[c] = X[c].astype("category")
    is_cat = [c in C.CATEGORICAL for c in cols]
    return X, is_cat
