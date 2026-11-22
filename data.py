"""Dataset acquisition, recoding, the fixed split and training-only imputation
(Section 3.3).

All three datasets end up with the same layout: predictor columns + a binary
column named 'target' where 1 = presence of the condition.
"""
import json
import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

UCI_CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

PIMA_RENAME = {
    "preg": "Pregnancies", "plas": "Glucose", "pres": "BloodPressure",
    "skin": "SkinThickness", "insu": "Insulin", "mass": "BMI",
    "pedi": "DiabetesPedigreeFunction", "age": "Age", "class": "target",
}
PIMA_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

CLEVELAND_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                  "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
CLEVELAND_CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CLEVELAND_DISCRETE = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Expected stratified split (Table 3.3): (train_neg, train_pos, test_neg, test_pos)
EXPECTED_SPLIT = {
    "pima": (400, 214, 100, 54),
    "cleveland": (131, 111, 33, 28),
    "wdbc": (285, 170, 72, 42),
}


@dataclass
class DatasetSpec:
    name: str
    continuous: list
    discrete: list
    target: str = "target"
    target_display: str = "target"
    imputation: dict = field(default_factory=dict)   # column -> constant

    @property
    def predictors(self):
        # keep the original column order: continuous and discrete interleaved
        return self.column_order

    column_order: list = field(default_factory=list)


@dataclass
class PreparedData:
    spec: DatasetSpec
    train_raw: pd.DataFrame      # with NaN, used only by the hyperparameter search
    test_raw: pd.DataFrame
    train: pd.DataFrame          # imputed with training-set constants
    test: pd.DataFrame

    @property
    def X_train(self):
        return self.train[self.spec.predictors]

    @property
    def y_train(self):
        return self.train[self.spec.target].to_numpy()

    @property
    def X_test(self):
        return self.test[self.spec.predictors]

    @property
    def y_test(self):
        return self.test[self.spec.target].to_numpy()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_pima(data_dir):
    arff_path = os.path.join(data_dir, "pima.arff")
    csv_path = os.path.join(data_dir, "pima.csv")
    if os.path.exists(arff_path):
        from scipy.io import arff
        raw, _ = arff.loadarff(arff_path)
        df = pd.DataFrame(raw)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].str.decode("utf-8")
        source = arff_path
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        source = csv_path
    else:
        from sklearn.datasets import fetch_openml
        bunch = fetch_openml(data_id=37, as_frame=True, parser="auto")
        df = bunch.frame.copy()
        source = "OpenML data_id=37 (downloaded)"
    df = df.rename(columns=PIMA_RENAME)
    if not pd.api.types.is_numeric_dtype(df["target"]):
        df["target"] = (df["target"].astype(str).str.strip() == "tested_positive").astype(int)
    df["target"] = df["target"].astype(int)
    predictors = [c for c in PIMA_RENAME.values() if c != "target"]
    for c in predictors:
        df[c] = pd.to_numeric(df[c])
    for c in PIMA_ZERO_AS_MISSING:
        df.loc[df[c] == 0, c] = np.nan
    spec = DatasetSpec(name="pima", continuous=predictors, discrete=[],
                       target_display="Outcome", column_order=predictors)
    return df[predictors + ["target"]], spec, source


def _load_cleveland(data_dir):
    path = os.path.join(data_dir, "processed.cleveland.data")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(UCI_CLEVELAND_URL, path)
        source = UCI_CLEVELAND_URL + " (downloaded)"
    else:
        source = path
    df = pd.read_csv(path, header=None, names=CLEVELAND_COLS, na_values="?")
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    predictors = CLEVELAND_COLS[:-1]
    spec = DatasetSpec(name="cleveland", continuous=CLEVELAND_CONTINUOUS,
                       discrete=CLEVELAND_DISCRETE, target_display="target",
                       column_order=predictors)
    return df[predictors + ["target"]], spec, source


def _load_wdbc(_data_dir):
    from sklearn.datasets import load_breast_cancer
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    # scikit-learn codes 0 = malignant; recode so that malignant = 1
    df["target"] = (df["target"] == 0).astype(int)
    predictors = [c for c in df.columns if c != "target"]
    spec = DatasetSpec(name="wdbc", continuous=predictors, discrete=[],
                       target_display="diagnosis", column_order=predictors)
    return df[predictors + ["target"]], spec, "sklearn.datasets.load_breast_cancer"


LOADERS = {"pima": _load_pima, "cleveland": _load_cleveland, "wdbc": _load_wdbc}


# ---------------------------------------------------------------------------
# Split + imputation
# ---------------------------------------------------------------------------
def _fit_imputation(train_raw, spec):
    consts = {}
    for c in spec.predictors:
        if train_raw[c].isna().any():
            if c in spec.discrete:
                # most frequent category; ties broken by the smallest code
                consts[c] = float(train_raw[c].mode(dropna=True).sort_values().iloc[0])
            else:
                consts[c] = float(train_raw[c].median(skipna=True))
    return consts


def _apply_imputation(df, spec, consts):
    out = df.copy()
    for c, v in consts.items():
        out[c] = out[c].fillna(v)
    for c in spec.discrete:
        out[c] = out[c].round().astype(int)
    out[spec.target] = out[spec.target].astype(int)
    return out


def prepare(name, data_dir="data", out_dir=None, verbose=True):
    df, spec, source = LOADERS[name](data_dir)
    df = df.reset_index(drop=True)
    train_raw, test_raw = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df["target"],
        random_state=config.SPLIT_SEED, shuffle=True,
    )
    train_raw = train_raw.reset_index().rename(columns={"index": "orig_row"})
    test_raw = test_raw.reset_index().rename(columns={"index": "orig_row"})

    consts = _fit_imputation(train_raw, spec)
    spec.imputation = consts
    train = _apply_imputation(train_raw, spec, consts)
    test = _apply_imputation(test_raw, spec, consts)

    counts = (
        int((train["target"] == 0).sum()), int((train["target"] == 1).sum()),
        int((test["target"] == 0).sum()), int((test["target"] == 1).sum()),
    )
    ok = counts == EXPECTED_SPLIT[name]
    if verbose:
        print(f"[{name}] source: {source}")
        print(f"[{name}] split train neg/pos = {counts[0]}/{counts[1]}, "
              f"test neg/pos = {counts[2]}/{counts[3]}  "
              f"({'matches Table 3.3' if ok else 'DOES NOT MATCH Table 3.3 ' + str(EXPECTED_SPLIT[name])})")
        if consts:
            print(f"[{name}] imputation constants (training set): {consts}")

    if out_dir:
        d = os.path.join(out_dir, "data", name)
        os.makedirs(d, exist_ok=True)
        train_raw.to_csv(os.path.join(d, "train_raw.csv"), index=False)
        test_raw.to_csv(os.path.join(d, "test_raw.csv"), index=False)
        train.to_csv(os.path.join(d, "train_imputed.csv"), index=False)
        test.to_csv(os.path.join(d, "test_imputed.csv"), index=False)
        with open(os.path.join(d, "preparation.json"), "w") as f:
            json.dump({
                "source": source, "split_counts_train_neg_pos_test_neg_pos": counts,
                "matches_table_3_3": ok, "imputation_constants": consts,
                "continuous": spec.continuous, "discrete": spec.discrete,
            }, f, indent=2)

    return PreparedData(spec=spec, train_raw=train_raw, test_raw=test_raw,
                        train=train, test=test)
