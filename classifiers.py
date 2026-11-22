"""Classifier construction, training at the fixed threshold and metrics
(Sections 3.7.5, 3.8)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def _xgb_device_kwargs(device):
    import xgboost
    major = int(xgboost.__version__.split(".")[0])
    if major >= 2:
        return {"tree_method": "hist", "device": "cuda" if device == "cuda" else "cpu"}
    return {"tree_method": "gpu_hist" if device == "cuda" else "hist"}


def make_estimator(name, params, seed, device="cpu", n_jobs=-1):
    """Final estimator for one training set. LR gets its own scaler, fitted on
    that training set only (the Pipeline guarantees this)."""
    params = dict(params)
    if name == "lr":
        clf = LogisticRegression(solver="liblinear", max_iter=10000, random_state=seed, **params)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **params)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, eval_metric="logloss",
                             n_jobs=n_jobs, **_xgb_device_kwargs(device), **params)
    raise ValueError(name)


def predict_proba(model, X):
    X = np.asarray(X, dtype=np.float32 if hasattr(model, "get_booster") else np.float64)
    return model.predict_proba(X)[:, 1]


def fit_predict(name, params, seed, X_train, y_train, X_test, device="cpu", n_jobs=-1):
    model = make_estimator(name, params, seed, device=device, n_jobs=n_jobs)
    Xtr = np.asarray(X_train, dtype=np.float32 if name == "xgb" else np.float64)
    model.fit(Xtr, np.asarray(y_train).astype(int))
    return predict_proba(model, X_test)


def metrics_from_proba(y_true, proba, threshold=config.THRESHOLD):
    """Positive class = 1. Precision is set to 0 (and flagged) when no positive
    prediction is made, so F1 is 0 rather than undefined."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    no_pos_pred = (tp + fp) == 0
    precision = 0.0 if no_pos_pred else tp / (tp + fp)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "f1": f1, "accuracy": (tp + tn) / len(y_true), "precision": precision,
        "recall": recall, "specificity": spec, "fpr": 1 - spec if (tn + fp) else 0.0,
        "no_positive_prediction": bool(no_pos_pred),
    }
