"""Hyperparameter selection rerun (Section 3.7.2 - 3.7.4).

Imputation and (for LR) scaling are fitted inside each CV training fold, as the
report now describes. Run once before the formal experiment; the selected
configurations are then held fixed.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import config
from classifiers import _xgb_device_kwargs


def _imputer(spec, train_raw):
    if not train_raw[spec.predictors].isna().any().any():
        return "passthrough"
    if spec.discrete:
        # Cleveland: missing only in discrete ca / thal -> most frequent
        return SimpleImputer(strategy="most_frequent")
    return SimpleImputer(strategy="median")    # Pima


def tune_one(prep, name, device="cpu", n_jobs=-1, smoke=False):
    spec = prep.spec
    X = prep.train_raw[spec.predictors].to_numpy(dtype=float)
    y = prep.train_raw[spec.target].to_numpy().astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.CV_SEED)
    imp = _imputer(spec, prep.train_raw)
    space = {f"clf__{k}": v for k, v in config.SEARCH_SPACES[name].items()}

    if name == "lr":
        clf = LogisticRegression(solver="liblinear", max_iter=10000, random_state=config.SEARCH_CLF_SEED)
        pipe = Pipeline([("imp", imp), ("scaler", StandardScaler()), ("clf", clf)])
        if smoke:
            space = {"clf__C": [0.1, 1], "clf__penalty": ["l2"], "clf__class_weight": [None, "balanced"]}
        search = GridSearchCV(pipe, space, scoring="f1", cv=cv, n_jobs=n_jobs, refit=False)
    elif name == "rf":
        clf = RandomForestClassifier(random_state=config.SEARCH_CLF_SEED, n_jobs=1)
        pipe = Pipeline([("imp", imp), ("clf", clf)])
        n_iter = config.SMOKE["n_iter"]["rf"] if smoke else config.N_ITER["rf"]
        search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="f1", cv=cv,
                                    n_jobs=n_jobs, random_state=config.SEARCH_SEED, refit=False)
    elif name == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(random_state=config.SEARCH_CLF_SEED, eval_metric="logloss",
                            n_jobs=1 if device == "cuda" else n_jobs, **_xgb_device_kwargs(device))
        pipe = Pipeline([("imp", imp), ("clf", clf)])
        n_iter = config.SMOKE["n_iter"]["xgb"] if smoke else config.N_ITER["xgb"]
        # one process when the GPU is used, to avoid GPU contention
        search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="f1", cv=cv,
                                    n_jobs=1 if device == "cuda" else n_jobs,
                                    random_state=config.SEARCH_SEED, refit=False)
    else:
        raise ValueError(name)

    t0 = time.time()
    search.fit(X, y)
    secs = time.time() - t0
    res = pd.DataFrame(search.cv_results_)
    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    return best, float(search.best_score_), res, secs


def run_tuning(preps, classifiers, out_dir, device="cpu", n_jobs=-1, smoke=False):
    os.makedirs(out_dir, exist_ok=True)
    best_all, log = {}, []
    for dname, prep in preps.items():
        best_all[dname] = {}
        for c in classifiers:
            print(f"[tune] {dname} / {config.CLASSIFIER_NAMES[c]} ...", flush=True)
            best, score, res, secs = tune_one(prep, c, device=device, n_jobs=n_jobs, smoke=smoke)
            best_all[dname][c] = best
            res.to_csv(os.path.join(out_dir, f"cv_results_{dname}_{c}.csv"), index=False)
            old = config.THESIS_TABLE_3_8.get(dname, {}).get(c, {})
            changed = {k: (old.get(k), v) for k, v in best.items() if old.get(k) != v}
            log.append({"dataset": dname, "classifier": c, "best_cv_f1": score,
                        "n_candidates": len(res), "seconds": round(secs, 1),
                        "same_as_table_3_8": not changed,
                        "changed_vs_table_3_8": json.dumps(changed, default=str),
                        "best_params": json.dumps(best, default=str)})
            print(f"        best CV F1 = {score:.4f}; "
                  f"{'same as Table 3.8' if not changed else 'differs from Table 3.8: ' + str(changed)}")
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(best_all, f, indent=2, default=str)
    pd.DataFrame(log).to_csv(os.path.join(out_dir, "tuning_summary.csv"), index=False)
    return best_all
