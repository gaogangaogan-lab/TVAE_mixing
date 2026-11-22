"""TVAE fitting and synthetic pool generation (Sections 3.4, 3.6.2).

The SDV API has changed across 1.x releases (SingleTableMetadata -> Metadata,
cuda -> enable_gpu), so the calls below check what the installed version offers.
"""
import inspect
import json
import os

import numpy as np
import pandas as pd

import config
from seeds import set_global_seeds

TABLE = "data"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def _update(md, column, **kw):
    try:
        md.update_column(table_name=TABLE, column_name=column, **kw)
    except TypeError:
        md.update_column(column_name=column, **kw)


def build_metadata(df, spec):
    try:
        from sdv.metadata import Metadata
        md = Metadata.detect_from_dataframe(data=df, table_name=TABLE)
    except ImportError:
        from sdv.metadata import SingleTableMetadata
        md = SingleTableMetadata()
        md.detect_from_dataframe(df)
    # Never let auto-detection create a primary key or id column
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        if hasattr(md, "remove_primary_key"):
            try:
                md.remove_primary_key(table_name=TABLE)
            except TypeError:
                try:
                    md.remove_primary_key()
                except Exception:
                    pass
            except Exception:
                pass
    for c in spec.continuous:
        _update(md, c, sdtype="numerical")
    for c in spec.discrete:
        _update(md, c, sdtype="categorical")
    _update(md, spec.target, sdtype="categorical")
    md.validate()
    return md


def metadata_dict(md):
    try:
        return md.to_dict()
    except Exception:
        return str(md)


# ---------------------------------------------------------------------------
# TVAE
# ---------------------------------------------------------------------------
class TVAEGenerator:
    """Thin wrapper so the experiment code does not depend on SDV details."""

    def __init__(self, spec, device="cuda", epochs=None):
        self.spec = spec
        self.device = device
        self.epochs = epochs or config.TVAE_CONFIG["epochs"]
        self.synth = None
        self.metadata = None

    def fit(self, train_df, seed):
        from sdv.single_table import TVAESynthesizer
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        cols = self.spec.predictors + [self.spec.target]
        data = train_df[cols].copy()
        self.metadata = build_metadata(data, self.spec)
        params = inspect.signature(TVAESynthesizer.__init__).parameters
        kw = dict(
            epochs=self.epochs,
            batch_size=config.TVAE_CONFIG["batch_size"],
            embedding_dim=config.TVAE_CONFIG["embedding_dim"],
            compress_dims=config.TVAE_CONFIG["compress_dims"],
            decompress_dims=config.TVAE_CONFIG["decompress_dims"],
            enforce_min_max_values=True,
            enforce_rounding=True,
        )
        use_gpu = self.device == "cuda"
        if "enable_gpu" in params:
            kw["enable_gpu"] = use_gpu
        elif "cuda" in params:
            kw["cuda"] = use_gpu
        if "verbose" in params:
            kw["verbose"] = False
        self.synth = TVAESynthesizer(self.metadata, **kw)
        set_global_seeds(seed)
        self.synth.fit(data)
        return self

    def loss_values(self):
        try:
            return self.synth.get_loss_values()
        except Exception:
            return None

    def batch_info(self, n_train):
        """Effective batches per epoch; ctgan's TVAE DataLoader keeps the last
        partial batch (drop_last=False). Checked against the installed code."""
        bs = config.TVAE_CONFIG["batch_size"]
        full, rest = divmod(n_train, bs)
        sizes = [bs] * full + ([rest] if rest else [])
        keeps_last = None
        try:
            import ctgan.synthesizers.tvae as tv
            src = inspect.getsource(tv).replace(" ", "")
            if "drop_last=True" in src:
                keeps_last = False
            else:
                # explicit drop_last=False, or torch's DataLoader default (False)
                keeps_last = True
        except Exception:
            pass
        return {"batch_sizes_per_epoch": sizes, "ctgan_keeps_last_batch": keeps_last}

    def sample(self, n, seed):
        set_global_seeds(seed)
        out = self.synth.sample(num_rows=int(n))
        return out


class MockGenerator:
    """FOR PIPELINE TESTING ONLY - never use for reported results.

    Resamples real rows per class and adds Gaussian noise to continuous columns,
    so the downstream code can be exercised without SDV / torch / a GPU.
    """

    def __init__(self, spec, device="cpu", epochs=None):
        self.spec = spec
        self.train = None

    def fit(self, train_df, seed):
        self.train = train_df[self.spec.predictors + [self.spec.target]].reset_index(drop=True)
        return self

    def loss_values(self):
        return None

    def batch_info(self, n_train):
        return {"mock": True}

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(self.train), size=int(n))
        out = self.train.iloc[idx].reset_index(drop=True).copy()
        for c in self.spec.continuous:
            sd = self.train[c].std()
            out[c] = out[c] + rng.normal(0, 0.15 * sd, size=len(out))
        return out


def make_generator(kind, spec, device, epochs=None):
    if kind == "tvae":
        return TVAEGenerator(spec, device=device, epochs=epochs)
    if kind == "mock":
        return MockGenerator(spec, device=device, epochs=epochs)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Pool (Section 3.6.2)
# ---------------------------------------------------------------------------
def _clean_pool(df, spec):
    out = df.copy()
    for c in spec.discrete:
        out[c] = pd.to_numeric(out[c]).round().astype(int)
    out[spec.target] = pd.to_numeric(out[spec.target]).round().astype(int)
    return out


def generate_pool(gen, spec, n_train, n_neg, n_pos, seed_fn):
    """Draw batches of 10 x N until each class has at least the counts the
    100%-mixing set needs, up to MAX_POOL_BATCHES. seed_fn(batch) -> seed."""
    batch_size = config.POOL_MULTIPLIER * n_train
    batches = []
    info = {"batch_size": batch_size, "batches_drawn": 0, "success": False}
    for b in range(config.MAX_POOL_BATCHES):
        part = _clean_pool(gen.sample(batch_size, seed_fn(b)), spec)
        batches.append(part)
        pool = pd.concat(batches, ignore_index=True)
        info["batches_drawn"] = b + 1
        n_neg_pool = int((pool[spec.target] == 0).sum())
        n_pos_pool = int((pool[spec.target] == 1).sum())
        if n_neg_pool >= n_neg and n_pos_pool >= n_pos:
            info["success"] = True
            break
    pool = pd.concat(batches, ignore_index=True)
    pool.insert(0, "pool_id", np.arange(len(pool)))
    feat_cols = spec.predictors + [spec.target]
    info.update({
        "pool_size": int(len(pool)),
        "pool_neg": int((pool[spec.target] == 0).sum()),
        "pool_pos": int((pool[spec.target] == 1).sum()),
        "pool_pos_share": float((pool[spec.target] == 1).mean()),
        "other_target_values": int((~pool[spec.target].isin([0, 1])).sum()),
        "duplicate_rows_within_pool": int(pool.duplicated(subset=feat_cols).sum()),
    })
    return pool, info


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
