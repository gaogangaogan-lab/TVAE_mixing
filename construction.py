"""Training-set construction: allocation rule, nested main settings and the
two ablation variants (Sections 3.6.1 - 3.6.3, 4.2).

Record identifiers:
  real records  -> positions 0..N-1 in the imputed real training set
  synthetic     -> pool_id of the synthetic pool
"""
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

import config


# ---------------------------------------------------------------------------
# Allocation (Section 3.6.1) - Python round() = round-half-to-even
# ---------------------------------------------------------------------------
def allocation(level, n_neg, n_pos):
    """Return dict with M, Mneg, Mpos, Rneg, Rpos for level k (p = k/10).

    Fractions keep the arithmetic exact, so 0.1 x 455 is exactly 45.5 and the
    half-to-even rule is applied to the true value, not a float approximation.
    """
    n = n_neg + n_pos
    m = round(Fraction(level * n, 10))
    m_neg = round(Fraction(m * n_neg, n))
    m_pos = m - m_neg
    return {
        "level": level, "p": level / 10, "N": n, "M": int(m),
        "Mneg": int(m_neg), "Mpos": int(m_pos),
        "Rneg": int(n_neg - m_neg), "Rpos": int(n_pos - m_pos),
        "realised_share": m / n,
    }


def allocation_table(n_neg, n_pos):
    return [allocation(k, n_neg, n_pos) for k in config.LEVELS]


# ---------------------------------------------------------------------------
# Training-set specification
# ---------------------------------------------------------------------------
@dataclass
class SetSpec:
    setting: str
    level: int
    real_ids: np.ndarray
    syn_ids: np.ndarray
    shared_with: tuple = None          # (setting, level) whose predictions are reused
    notes: dict = field(default_factory=dict)

    @property
    def key(self):
        return (self.setting, self.level)


def _prefix(order, k):
    return np.asarray(order[:k], dtype=np.int64)


def build_all_sets(real_y, pool_y, real_seed, syn_seed, c_seed_fn):
    """Build the main settings and Ablations B and C for one repetition.

    real_y   : target array of the real training set (index = real id)
    pool_y   : target array of the synthetic pool (index = pool_id)
    real_seed, syn_seed : seeds of the two priority orderings of this repetition
    c_seed_fn(kind, level, cls) -> seed for Ablation C fresh orderings
    """
    real_y = np.asarray(real_y)
    pool_y = np.asarray(pool_y)
    n_neg = int((real_y == 0).sum())
    n_pos = int((real_y == 1).sum())
    n = n_neg + n_pos

    # One random priority ordering over all real records and one over the pool.
    real_order = np.random.default_rng(real_seed).permutation(len(real_y))
    syn_order = np.random.default_rng(syn_seed).permutation(len(pool_y))
    # Class-specific orderings = the same orderings filtered by class.
    real_neg_order = real_order[real_y[real_order] == 0]
    real_pos_order = real_order[real_y[real_order] == 1]
    syn_neg_order = syn_order[pool_y[syn_order] == 0]
    syn_pos_order = syn_order[pool_y[syn_order] == 1]

    sets = {}
    for k in config.LEVELS:
        a = allocation(k, n_neg, n_pos)

        # ---- Main: class-specific, fixed size, nested -------------------
        main_real = np.concatenate([_prefix(real_neg_order, a["Rneg"]),
                                    _prefix(real_pos_order, a["Rpos"])])
        main_syn = np.concatenate([_prefix(syn_neg_order, a["Mneg"]),
                                   _prefix(syn_pos_order, a["Mpos"])])
        sets[("main", k)] = SetSpec("main", k, main_real, main_syn)

        # ---- Ablation B: unfiltered orderings, fixed size, no class split -
        b_real = _prefix(real_order, n - a["M"])
        b_syn = _prefix(syn_order, a["M"])
        sets[("B", k)] = SetSpec("B", k, b_real, b_syn,
                                 shared_with=("main", 0) if k == 0 else None)

        # ---- Ablation C: fresh class-specific draws at 10%..90% ----------
        if k in (0, config.N_LEVELS - 1):
            sets[("C", k)] = SetSpec("C", k, main_real.copy(), main_syn.copy(),
                                     shared_with=("main", k))
        else:
            rn = np.random.default_rng(c_seed_fn("real", k, "neg")).permutation(real_neg_order)
            rp = np.random.default_rng(c_seed_fn("real", k, "pos")).permutation(real_pos_order)
            sn = np.random.default_rng(c_seed_fn("syn", k, "neg")).permutation(syn_neg_order)
            sp = np.random.default_rng(c_seed_fn("syn", k, "pos")).permutation(syn_pos_order)
            c_real = np.concatenate([_prefix(rn, a["Rneg"]), _prefix(rp, a["Rpos"])])
            c_syn = np.concatenate([_prefix(sn, a["Mneg"]), _prefix(sp, a["Mpos"])])
            sets[("C", k)] = SetSpec("C", k, c_real, c_syn)

    return sets, {"real_order": real_order, "syn_order": syn_order}


def describe_set(s, real_y, pool_y):
    """Counts and positive-class shares recorded with every training set."""
    ry = np.asarray(real_y)[s.real_ids]
    sy = np.asarray(pool_y)[s.syn_ids]
    n_real, n_syn = len(ry), len(sy)
    n_total = n_real + n_syn
    pos_real, pos_syn = int((ry == 1).sum()), int((sy == 1).sum())
    return {
        "n_total": n_total, "n_real": n_real, "n_syn": n_syn,
        "n_real_pos": pos_real, "n_syn_pos": pos_syn,
        "n_real_neg": n_real - pos_real, "n_syn_neg": n_syn - pos_syn,
        "realised_syn_share": n_syn / n_total if n_total else np.nan,
        "pos_share_real": pos_real / n_real if n_real else np.nan,
        "pos_share_syn": pos_syn / n_syn if n_syn else np.nan,
        "pos_share_total": (pos_real + pos_syn) / n_total if n_total else np.nan,
    }


# ---------------------------------------------------------------------------
# Property checks (Section 4.2) - run on every repetition, fail loudly
# ---------------------------------------------------------------------------
def check_properties(sets, real_y, pool_y):
    real_y = np.asarray(real_y)
    n_neg = int((real_y == 0).sum())
    n_pos = int((real_y == 1).sum())
    n = n_neg + n_pos
    problems = []
    for k in config.LEVELS:
        a = allocation(k, n_neg, n_pos)
        for st in ("main", "C"):
            d = describe_set(sets[(st, k)], real_y, pool_y)
            if d["n_total"] != n:
                problems.append(f"{st} L{k}: size {d['n_total']} != {n}")
            if (d["n_real_neg"] + d["n_syn_neg"], d["n_real_pos"] + d["n_syn_pos"]) != (n_neg, n_pos):
                problems.append(f"{st} L{k}: class totals changed")
            if (d["n_syn_neg"], d["n_syn_pos"]) != (a["Mneg"], a["Mpos"]):
                problems.append(f"{st} L{k}: synthetic class counts differ from allocation")
        dB = describe_set(sets[("B", k)], real_y, pool_y)
        if dB["n_total"] != n:
            problems.append(f"B L{k}: size != N")
        for st in config.SETTINGS:
            s = sets[(st, k)]
            if len(np.unique(s.real_ids)) != len(s.real_ids) or len(np.unique(s.syn_ids)) != len(s.syn_ids):
                problems.append(f"{st} L{k}: repeated record ids")
    # nesting of main / A / B components between adjacent levels
    for st in ("main", "B"):
        for k in config.LEVELS[:-1]:
            s0, s1 = sets[(st, k)], sets[(st, k + 1)]
            if not set(s1.real_ids).issubset(set(s0.real_ids)):
                problems.append(f"{st} L{k}->L{k+1}: real component not nested")
            if not set(s0.syn_ids).issubset(set(s1.syn_ids)):
                problems.append(f"{st} L{k}->L{k+1}: synthetic component not nested")
    return problems
