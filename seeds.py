"""Derived random streams (Section 3.9.3).

Each root seed produces separate deterministic streams. A derived seed depends only
on the root and fixed integer codes, never on Python's process-dependent hash(),
so the same inputs give the same seed on every machine and every run.
"""
import random

import numpy as np

DATASET_CODE = {"pima": 1, "cleveland": 2, "wdbc": 3}
PURPOSE_CODE = {
    "tvae_fit": 1,
    "pool_sample": 2,
    "real_order": 3,
    "syn_order": 4,
    "ablation_c_real": 5,
    "ablation_c_syn": 6,
    "row_shuffle": 7,
    "classifier": 8,
}
CLASSIFIER_CODE = {"none": 0, "lr": 1, "rf": 2, "xgb": 3}
CLASS_CODE = {"none": 0, "neg": 1, "pos": 2}


def derive_seed(root, dataset, purpose, classifier="none", level=None, cls="none", batch=0):
    """Return a 32-bit seed for one component.

    level: None when not applicable, otherwise 0..10 (stored as level+1 so that
    'not applicable' and 'level 0' never collide).
    batch: pool batch index (0 for everything else).
    """
    level_code = 0 if level is None else int(level) + 1
    key = (
        DATASET_CODE[dataset],
        PURPOSE_CODE[purpose],
        CLASSIFIER_CODE[classifier],
        level_code,
        CLASS_CODE[cls],
        int(batch),
    )
    ss = np.random.SeedSequence(entropy=int(root), spawn_key=key)
    return int(ss.generate_state(1, dtype=np.uint32)[0])


class SeedLog:
    """Records every derived seed so the mapping can be archived."""

    def __init__(self):
        self.rows = []

    def get(self, root, dataset, purpose, classifier="none", level=None, cls="none", batch=0):
        s = derive_seed(root, dataset, purpose, classifier, level, cls, batch)
        self.rows.append({
            "root": root, "dataset": dataset, "purpose": purpose,
            "classifier": classifier, "level": level, "class": cls,
            "batch": batch, "seed": s,
        })
        return s


def set_global_seeds(seed):
    """Seed Python, NumPy and (if present) torch before a TVAE fit or sample."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
