"""Feature construction (only information available when the quote arrives)."""
import numpy as np
import pandas as pd

from . import config as C


def add_quote_features(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    est = q.estimated_repair_cost.where(q.estimated_repair_cost > 0)
    q["log_quote_to_estimate"] = np.log1p(q.contractor_cost) - np.log1p(est)
    return q


def feature_matrix(q: pd.DataFrame) -> pd.DataFrame:
    """Inspection-time features for the fair-cost model, categoricals as pandas categories."""
    X = q[C.CATEGORICAL + C.NUMERIC_PRE_QUOTE].copy()
    for c in C.CATEGORICAL:
        X[c] = X[c].astype("category")
    return X
