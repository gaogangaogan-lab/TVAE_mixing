"""Fixed experimental settings (Chapter 3 / Table 3.4, 3.6, 3.7, 3.8, 4.2).

Everything that the report fixes in advance lives here, so that a change to the
design is a change to one file.
"""

# ---------------------------------------------------------------------------
# Split, CV and repetition seeds (Sections 3.3.1, 3.7.2, 3.9.3)
# ---------------------------------------------------------------------------
SPLIT_SEED = 42            # 80:20 stratified split
CV_SEED = 42               # StratifiedKFold shuffling in the hyperparameter search
SEARCH_SEED = 2026         # RandomizedSearchCV candidate sampling (recorded, separate from repetitions)
SEARCH_CLF_SEED = 42       # random_state of stochastic classifiers inside CV
ROOT_SEEDS = [100, 101, 102, 103, 104]
DISCOVERY_SEED = 100              # RQ1: the highest-F1 proportion is found in this run
REPLICATION_SEEDS = [101, 102, 103, 104]   # RQ2: is that proportion still above 0% here?
TEST_SIZE = 0.2
N_LEVELS = 11              # 0%, 10%, ..., 100%
LEVELS = list(range(N_LEVELS))  # level k means nominal proportion k/10

# ---------------------------------------------------------------------------
# TVAE (Table 3.4) - SDV defaults
# ---------------------------------------------------------------------------
TVAE_CONFIG = {
    "epochs": 300,
    "batch_size": 500,
    "embedding_dim": 128,
    "compress_dims": (128, 128),
    "decompress_dims": (128, 128),
}
POOL_MULTIPLIER = 10       # pool = 10 x N per batch
MAX_POOL_BATCHES = 5       # Section 3.6.2

# ---------------------------------------------------------------------------
# Classification (Section 3.8)
# ---------------------------------------------------------------------------
THRESHOLD = 0.5
CLASSIFIERS = ["lr", "rf", "xgb"]
CLASSIFIER_NAMES = {"lr": "LR", "rf": "RF", "xgb": "XGBoost"}

# ---------------------------------------------------------------------------
# Hyperparameter search (Tables 3.6, 3.7)
# ---------------------------------------------------------------------------
N_ITER = {"rf": 200, "xgb": 250}       # LR uses the full grid (36)

SEARCH_SPACES = {
    "lr": {
        "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        "penalty": ["l1", "l2"],
        "class_weight": [None, "balanced"],
    },
    "rf": {
        "n_estimators": [100, 200, 300, 500, 800],
        "max_depth": [None, 3, 5, 8, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 6, 10],
        "max_features": ["sqrt", "log2", 0.5, 0.75, None],
        "criterion": ["gini", "entropy"],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "bootstrap": [True, False],
    },
    "xgb": {
        "n_estimators": [100, 200, 300, 500, 800],
        "max_depth": [2, 3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2],
        "min_child_weight": [1, 2, 3, 5, 8, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.05, 0.1, 0.25, 0.5, 1],
        "reg_alpha": [0, 0.001, 0.01, 0.1, 0.5, 1],
        "reg_lambda": [0.5, 1, 2, 5, 10, 20],
    },
}

# Table 3.8 (selected before the rerun). Used only with --use-thesis-params,
# or as the comparison column when the rerun finishes.
THESIS_TABLE_3_8 = {
    "pima": {
        "lr": {"C": 0.1, "class_weight": "balanced", "penalty": "l2"},
        "rf": {"n_estimators": 500, "max_depth": 10, "min_samples_split": 5,
               "min_samples_leaf": 6, "max_features": 0.5, "criterion": "gini",
               "class_weight": "balanced", "bootstrap": True},
        "xgb": {"n_estimators": 100, "max_depth": 8, "learning_rate": 0.075,
                "min_child_weight": 2, "subsample": 0.8, "colsample_bytree": 0.7,
                "gamma": 1.0, "reg_alpha": 0, "reg_lambda": 5},
    },
    "cleveland": {
        "lr": {"C": 10, "class_weight": "balanced", "penalty": "l1"},
        "rf": {"n_estimators": 100, "max_depth": 8, "min_samples_split": 15,
               "min_samples_leaf": 10, "max_features": 0.5, "criterion": "gini",
               "class_weight": "balanced_subsample", "bootstrap": True},
        "xgb": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.15,
                "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 1.0,
                "gamma": 1.0, "reg_alpha": 0.1, "reg_lambda": 20},
    },
    "wdbc": {
        "lr": {"C": 0.1, "class_weight": "balanced", "penalty": "l2"},
        "rf": {"n_estimators": 200, "max_depth": None, "min_samples_split": 10,
               "min_samples_leaf": 4, "max_features": "sqrt", "criterion": "entropy",
               "class_weight": "balanced", "bootstrap": False},
        "xgb": {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.2,
                "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 1.0,
                "gamma": 0, "reg_alpha": 0.001, "reg_lambda": 2},
    },
}

# ---------------------------------------------------------------------------
# Settings (Table 3.1)
# ---------------------------------------------------------------------------
SETTINGS = ["main", "B", "C"]
# Display names used in tables and figures. The internal keys "B" and "C" are kept
# so that stored results stay readable: "B" is Ablation A in the report
# (unstratified replacement) and "C" is Ablation B (independent draws).
SETTING_NAMES = {
    "main": "Main (fixed-budget mixing)",
    "B": "Ablation A (unstratified)",
    "C": "Ablation B (independent draws)",
}
SETTING_SHORT = {"main": "Main", "B": "Abl. A", "C": "Abl. B"}

# Smoke-test overrides (pipeline check only, NOT for reported results)
SMOKE = {"epochs": None,  # full 300 epochs: TVAE fits in seconds, and 5 epochs left Pima's pool almost without positives
          "n_iter": {"rf": 4, "xgb": 4}, "lr_grid_small": True}
