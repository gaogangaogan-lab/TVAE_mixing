"""Chapter 4 tables and figures under the discovery / replication framing.

RQ1  The highest-F1 mixing proportion is identified in ONE run, the discovery run
     (config.DISCOVERY_SEED = 100). If 0% ties with mixed proportions, a mixed
     proportion is taken (the largest synthetic share among the tied ones).
RQ2  That proportion is checked in the replication runs (seeds 101-104): in each
     run, F1 at that proportion minus F1 at 0% in the same run. A gain counts as
     replicated only if the difference is positive in every replication run.
Abl. Ablations B and C are read with exactly the same rule; the summary counts
     how many of the nine combinations show a replicated gain per setting.
RQ3  100% mixing only, all five runs: predictor-based fidelity vs F1.

Everything is descriptive; no significance tests. Means/SDs over the five runs
are kept only as context on run-to-run variability.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import config

DS_ORDER = ["pima", "cleveland", "wdbc"]
DS_NAMES = {"pima": "Pima", "cleveland": "Cleveland", "wdbc": "WDBC"}
TOL = 1e-12
DISC = config.DISCOVERY_SEED
REPL = config.REPLICATION_SEEDS

# Reference categorical palette (dataviz skill, light mode) in fixed order, plus
# distinct markers so identity never depends on colour alone.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
MUTED = "#6b6b6b"


# ---------------------------------------------------------------------------
# Loading and helpers
# ---------------------------------------------------------------------------
def load(out_dir):
    raw = os.path.join(out_dir, "raw")
    metrics, fid_vars, fid_sum, status = [], [], [], []
    for done in sorted(glob.glob(os.path.join(raw, "*", "rep_*", "done.json"))):
        rep_dir = os.path.dirname(done)
        d = os.path.basename(os.path.dirname(rep_dir))
        rep = int(os.path.basename(rep_dir).split("_")[1])
        with open(done) as f:
            st = json.load(f)
        status.append({"dataset": d, "rep": rep, "status": st.get("status")})
        if st.get("status") != "ok":
            continue
        metrics.append(pd.read_csv(os.path.join(rep_dir, "metrics.csv")))
        fid_vars.append(pd.read_csv(os.path.join(rep_dir, "fidelity_variables.csv")))
        with open(os.path.join(rep_dir, "fidelity_summary.json")) as f:
            fid_sum.append(json.load(f))
    if not metrics:
        raise SystemExit(f"No finished repetitions found under {raw}")
    df = pd.concat(metrics, ignore_index=True)
    df = df[df.setting.isin(config.SETTINGS)]       # older runs may still hold Ablation A rows
    return df, pd.concat(fid_vars, ignore_index=True), pd.DataFrame(fid_sum), pd.DataFrame(status)


def _sd(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan


def _fmt(m, s, nd=3):
    if pd.isna(m):
        return ""
    return f"{m:.{nd}f} ({s:.{nd}f})" if not pd.isna(s) else f"{m:.{nd}f}"


def _pct(k):
    return f"{int(k) * 10}%"


def _sign_counts(diffs):
    d = np.asarray(diffs, float)
    return int((d > TOL).sum()), int((np.abs(d) <= TOL).sum()), int((d < -TOL).sum())


def _combos():
    for d in DS_ORDER:
        for c in config.CLASSIFIERS:
            yield d, c


def _pivot(df, d, c, setting):
    s = df[(df.dataset == d) & (df.classifier == c) & (df.setting == setting)]
    return s.pivot_table(index="rep", columns="level", values="f1")


def discovery_best(row):
    """Highest F1 in one run. Ties with 0% resolve to a mixed proportion
    (the largest synthetic share among the tied mixed proportions)."""
    mx = row.max()
    tied = [int(k) for k in row.index if abs(row[k] - mx) <= TOL]
    mixed = [k for k in tied if k != 0]
    return (max(mixed) if mixed else 0), tied


# ---------------------------------------------------------------------------
# RQ1 - discovery run
# ---------------------------------------------------------------------------
def rq1_tables(df):
    best_rows, traj_rows, shape_rows = [], [], []
    for d, c in _combos():
        piv = _pivot(df, d, c, "main")
        if DISC not in piv.index:
            continue
        r = piv.loc[DISC]
        best, tied = discovery_best(r)
        gain = r[best] - r[0]
        best_rows.append({
            "Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c],
            "Highest-F1 Proportion": _pct(best),
            "Tied Proportions": ", ".join(_pct(k) for k in tied) if len(tied) > 1 else "",
            "F1 at Highest": round(r[best], 4), "F1 at 0%": round(r[0], 4),
            "Difference from 0%": round(gain, 4),
            "Reading": ("0% highest" if best == 0 else
                        "mixed equals 0% (tie)" if abs(gain) <= TOL else "mixed above 0%"),
        })
        traj = {"Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c]}
        traj.update({_pct(k): round(r[k], 4) for k in r.index})
        traj_rows.append(traj)
        steps = np.diff(r.to_numpy())
        shape_rows.append({
            "Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c],
            "F1 at 0%": round(r[0], 4), "F1 at 100%": round(r[10], 4),
            "100% minus 0%": round(r[10] - r[0], 4),
            "Levels at or above 0%": ", ".join(_pct(k) for k in r.index if k > 0 and r[k] >= r[0] - TOL),
            "Adjacent increases / decreases / unchanged (of 10)":
                f"{int((steps > TOL).sum())} / {int((steps < -TOL).sum())} / {int((np.abs(steps) <= TOL).sum())}",
            "Range (max - min)": round(r.max() - r.min(), 4),
        })
    return pd.DataFrame(best_rows), pd.DataFrame(traj_rows), pd.DataFrame(shape_rows)


# ---------------------------------------------------------------------------
# RQ2 - replication runs (and the same rule for the ablations)
# ---------------------------------------------------------------------------
def replication_table(df, setting):
    rows = []
    for d, c in _combos():
        piv = _pivot(df, d, c, setting)
        if DISC not in piv.index:
            continue
        best, tied = discovery_best(piv.loc[DISC])
        row = {"Setting": setting, "Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c],
               "Discovery Proportion": _pct(best),
               "Discovery Gain": round(piv.loc[DISC, best] - piv.loc[DISC, 0], 4)}
        reps = [r for r in REPL if r in piv.index]
        if best == 0:
            for r in reps:
                row[f"Seed {r}"] = ""
            row.update({"Above / Equal / Below 0%": "", "Mean Difference (SD)": "",
                        "Replicated in All Runs": "no gain to replicate (0% highest)"})
        else:
            diffs = [piv.loc[r, best] - piv.loc[r, 0] for r in reps]
            for r, x in zip(reps, diffs):
                row[f"Seed {r}"] = round(x, 4)
            p, z, n = _sign_counts(diffs)
            row.update({"Above / Equal / Below 0%": f"{p} / {z} / {n}",
                        "Mean Difference (SD)": _fmt(np.mean(diffs), _sd(diffs), 4),
                        "Replicated in All Runs": "yes" if p == len(reps) else "no"})
        rows.append(row)
    return pd.DataFrame(rows)


def per_level_above_baseline(df, setting="main"):
    """For every mixed level: in how many replication runs is it above 0%?"""
    s = df[(df.setting == setting) & (df.rep.isin(REPL))]
    rows = []
    for k in range(1, config.N_LEVELS):
        row = {"Proportion": _pct(k)}
        tot_p = tot = 0
        for d, c in _combos():
            g = s[(s.dataset == d) & (s.classifier == c)].pivot_table(index="rep", columns="level", values="f1")
            diffs = (g[k] - g[0]).to_numpy()
            p, z, n = _sign_counts(diffs)
            row[f"{DS_NAMES[d]} {config.CLASSIFIER_NAMES[c]}"] = f"{p}/{z}/{n}"
            tot_p += p
            tot += len(diffs)
        row["All Combinations: Runs Above 0%"] = f"{tot_p} / {tot}"
        rows.append(row)
    return pd.DataFrame(rows)


def best_mixed_vs_baseline(df):
    """Per replication run: best of the ten mixed levels vs 0%. Descriptive only;
    the max of ten noisy values is expected to beat one value most of the time."""
    s = df[(df.setting == "main") & (df.rep.isin(REPL))]
    rows = []
    for d, c in _combos():
        g = s[(s.dataset == d) & (s.classifier == c)].pivot_table(index="rep", columns="level", values="f1")
        diffs = (g.drop(columns=0).max(axis=1) - g[0]).to_numpy()
        p, z, n = _sign_counts(diffs)
        rows.append({"Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c],
                     "Best Mixed Above / Equal / Below 0%": f"{p} / {z} / {n}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ablations B and C
# ---------------------------------------------------------------------------
def ablation_count_summary(rep_tables):
    rows = []
    for st, t in rep_tables.items():
        yes = t[t["Replicated in All Runs"] == "yes"]
        rows.append({
            "Setting": config.SETTING_NAMES[st],
            "Combinations with 0% Highest in Discovery": int((t["Discovery Proportion"] == "0%").sum()),
            "Combinations with a Replicated Gain (of 9)": int(len(yes)),
            "Which": "; ".join(f"{r.Dataset} {r.Classifier} at {r['Discovery Proportion']}" for _, r in yes.iterrows()),
        })
    return pd.DataFrame(rows)


def ablation_per_level_vs_main(df):
    """Paired difference (ablated minus main) at every level, within each run;
    summarised over all five runs as context."""
    main = df[df.setting == "main"].set_index(["dataset", "classifier", "level", "rep"])
    rows = []
    for st in [s for s in config.SETTINGS if s != "main"]:
        ab = df[df.setting == st].set_index(["dataset", "classifier", "level", "rep"])
        j = ab.join(main[["f1"]], rsuffix="_main")
        j["diff"] = j["f1"] - j["f1_main"]
        for (d, c, k), g in j.groupby(level=[0, 1, 2]):
            p, z, n = _sign_counts(g["diff"])
            rows.append({
                "Ablation": st, "dataset": d, "classifier": c, "level": k,
                "main_mean": g["f1_main"].mean(), "main_sd": _sd(g["f1_main"]),
                "ablated_mean": g["f1"].mean(), "ablated_sd": _sd(g["f1"]),
                "paired_diff_mean": g["diff"].mean(), "paired_diff_sd": _sd(g["diff"]),
                "runs_above/equal/below_main": f"{p}/{z}/{n}",
            })
    out = pd.DataFrame(rows)
    out["dataset"] = pd.Categorical(out["dataset"], DS_ORDER)
    return out.sort_values(["Ablation", "dataset", "classifier", "level"]).reset_index(drop=True)


def ablation_b_class_shares(df):
    b = df[(df.setting == "B") & (df.classifier == config.CLASSIFIERS[0])]
    g = b.groupby(["dataset", "level"])[["pos_share_real", "pos_share_syn", "pos_share_total"]].mean()
    out = g.reset_index()
    out["dataset"] = out["dataset"].map(DS_NAMES)
    out["level"] = out["level"].map(_pct)
    return out.round(4)


def ablation_b_confusion(df, rep_tables):
    """Confusion counts (mean over replication runs) at the discovery proportion,
    main vs Ablation B, to see how class drift changes the predictions."""
    rows = []
    rev = {v: k for k, v in DS_NAMES.items()}
    revc = {v: k for k, v in config.CLASSIFIER_NAMES.items()}
    for _, r in rep_tables["B"].iterrows():
        if r["Discovery Proportion"] == "0%":
            continue
        d, c, k = rev[r.Dataset], revc[r.Classifier], int(r["Discovery Proportion"][:-1]) // 10
        for st in ["main", "B"]:
            g = df[(df.setting == st) & (df.dataset == d) & (df.classifier == c)
                   & (df.level == k) & (df.rep.isin(REPL))]
            rows.append({"Dataset": r.Dataset, "Classifier": r.Classifier, "Proportion": _pct(k),
                         "Setting": st, **g[["TP", "FP", "FN", "TN", "f1", "pos_share_total"]].mean().round(3).to_dict()})
        g0 = df[(df.setting == "main") & (df.dataset == d) & (df.classifier == c)
                & (df.level == 0) & (df.rep.isin(REPL))]
        rows.append({"Dataset": r.Dataset, "Classifier": r.Classifier, "Proportion": "0%",
                     "Setting": "baseline", **g0[["TP", "FP", "FN", "TN", "f1", "pos_share_total"]].mean().round(3).to_dict()})
    return pd.DataFrame(rows)


def ablation_c_smoothness(df):
    per_pair, summary = [], []
    for d, c in _combos():
        res = {}
        for st in ["main", "C"]:
            piv = _pivot(df, d, c, st)
            level_sd = piv.std(ddof=1)
            delta = piv.diff(axis=1).iloc[:, 1:]
            for k in range(config.N_LEVELS - 1):
                dk = delta.iloc[:, k].to_numpy()
                per_pair.append({"dataset": d, "classifier": c,
                                 "version": "nested (main)" if st == "main" else "independent (C)",
                                 "pair": f"{_pct(k)}-{_pct(k + 1)}",
                                 "mean_abs_change": float(np.mean(np.abs(dk))), "sd_of_change": _sd(dk)})
            res[st] = (float(level_sd.loc[1:9].mean()), float(np.mean(np.abs(delta.to_numpy()))),
                       float(np.mean([_sd(delta.iloc[:, k]) for k in range(delta.shape[1])])))
        summary.append({
            "Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c],
            "Mean |Adjacent Change| (Nested)": round(res["main"][1], 4),
            "Mean |Adjacent Change| (Independent)": round(res["C"][1], 4),
            "SD of Adjacent Change (Nested)": round(res["main"][2], 4),
            "SD of Adjacent Change (Independent)": round(res["C"][2], 4),
            "Per-Level SD 10-90% (Nested)": round(res["main"][0], 4),
            "Per-Level SD 10-90% (Independent)": round(res["C"][0], 4),
        })
    return pd.DataFrame(summary), pd.DataFrame(per_pair)


# ---------------------------------------------------------------------------
# RQ3 - fidelity at 100% (all five runs)
# ---------------------------------------------------------------------------
def rq3_fidelity_table(fid_vars, fid_sum):
    rows = []
    for d in DS_ORDER:
        fv = fid_vars[fid_vars.dataset == d]
        if fv.empty:
            continue
        for (var, typ, meas), g in fv.groupby(["variable", "type", "measure"], sort=False):
            rows.append({"Dataset": DS_NAMES[d], "Variable": var, "Type": typ, "Measure": meas,
                         "Mean over Runs": round(g["score"].mean(), 4),
                         "SD over Runs": round(_sd(g["score"]), 4), "Runs": len(g)})
        fs = fid_sum[fid_sum.dataset == d]
        for col, label in [("continuous_mean", "Continuous predictors (mean)"),
                           ("discrete_mean", "Discrete predictors (mean)"),
                           ("combined_predictors", "Combined score (predictors only)"),
                           ("pool_pos_share", "Unfiltered pool positive share")]:
            vals = fs[col].astype(float)
            rows.append({"Dataset": DS_NAMES[d], "Variable": label, "Type": "summary",
                         "Measure": "class frequency" if col == "pool_pos_share" else "",
                         "Mean over Runs": "not applicable" if vals.isna().all() else round(vals.mean(), 4),
                         "SD over Runs": "" if vals.isna().all() else round(_sd(vals), 4), "Runs": len(fs)})
    return pd.DataFrame(rows)


def rq3_pairs(df, fid_sum):
    m = df[(df.setting == "main") & (df.level == 10)][["dataset", "classifier", "rep", "f1"]]
    j = m.merge(fid_sum[["dataset", "rep", "combined_predictors"]], on=["dataset", "rep"])
    j["dataset"] = pd.Categorical(j["dataset"], DS_ORDER)
    return j.sort_values(["dataset", "classifier", "combined_predictors"]).rename(
        columns={"combined_predictors": "fidelity", "rep": "seed"}).reset_index(drop=True)


def rq3_trend_table(pairs):
    """Runs ordered by fidelity (low -> high) with the three classifiers' F1."""
    w = pairs.pivot_table(index=["dataset", "seed", "fidelity"], columns="classifier",
                          values="f1", observed=True).reset_index()
    w = w.sort_values(["dataset", "fidelity"])
    w["dataset"] = w["dataset"].map(DS_NAMES)
    return w.rename(columns={c: f"F1 {config.CLASSIFIER_NAMES[c]}" for c in config.CLASSIFIERS}).round(4)


def rq3_descriptive(pairs):
    rows = []
    for (d, c), g in pairs.groupby(["dataset", "classifier"], observed=True):
        g = g.sort_values("fidelity")
        f, y = g["fidelity"].to_numpy(), g["f1"].to_numpy()
        same = opp = 0
        for i in range(len(f)):
            for k in range(i + 1, len(f)):
                s = np.sign(f[k] - f[i]) * np.sign(y[k] - y[i])
                same += s > 0
                opp += s < 0
        rows.append({"Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c],
                     "Fidelity Range": round(f.max() - f.min(), 4), "F1 Range": round(y.max() - y.min(), 4),
                     "F1 at Highest-Fidelity Run": round(y[-1], 4), "F1 at Lowest-Fidelity Run": round(y[0], 4),
                     "Pairs Same Direction / Opposite (of 10)": f"{int(same)} / {int(opp)}"})
    return pd.DataFrame(rows)


def supporting_metrics(df):
    cols = ["f1", "precision", "recall", "specificity", "fpr", "accuracy", "TP", "FP", "TN", "FN"]
    g = df.groupby(["dataset", "classifier", "setting", "level"])[cols]
    out = pd.concat([g.mean().add_suffix("_mean"), g.std(ddof=1).add_suffix("_sd")], axis=1).reset_index()
    return out[["dataset", "classifier", "setting", "level"] + [x for c in cols for x in (c + "_mean", c + "_sd")]]


def mean_sd_context(df):
    """Mean (SD) over all five runs at each level - context on variability only."""
    rows = []
    for st in config.SETTINGS:
        for d, c in _combos():
            piv = _pivot(df, d, c, st)
            row = {"Setting": st, "Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c]}
            row.update({_pct(k): _fmt(piv[k].mean(), _sd(piv[k])) for k in piv.columns})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                         "axes.axisbelow": True})
    return plt


def _legend_first(axes, **kw):
    for ax in np.ravel(axes):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, **kw)
            return


def fig_rq1(df, path):
    """Discovery-run trajectories; the identified proportion is ringed."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    x = np.array(config.LEVELS) * 10
    for ax, d in zip(axes, DS_ORDER):
        for i, c in enumerate(config.CLASSIFIERS):
            piv = _pivot(df, d, c, "main")
            if DISC not in piv.index:
                continue
            r = piv.loc[DISC]
            best, _ = discovery_best(r)
            ax.plot(x, r.to_numpy(), color=SERIES_COLORS[i], marker=MARKERS[i], ms=5, lw=2,
                    label=config.CLASSIFIER_NAMES[c])
            ax.scatter([best * 10], [r[best]], s=170, facecolors="none", edgecolors=SERIES_COLORS[i],
                       linewidths=1.8, zorder=4)
        ax.set_title(f"{DS_NAMES[d]} (seed {DISC})")
        ax.set_xticks(x)
        ax.set_xlabel("Mixing proportion (% synthetic)")
    axes[0].set_ylabel("F1-score (discovery run)")
    _legend_first(axes, loc="lower left")
    fig.text(0.5, -0.02, "Ring = highest-F1 proportion (ties with 0% resolved to the mixed proportion)",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_replication(rep_tables, settings, path, title_suffix=""):
    """Dot plot: F1 at the discovery proportion minus F1 at 0%, one dot per run."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, d in zip(axes, DS_ORDER):
        ax.axhline(0, color="#333333", lw=1)
        labels = []
        for ci, c in enumerate(config.CLASSIFIERS):
            for si, st in enumerate(settings):
                t = rep_tables[st]
                r = t[(t.Dataset == DS_NAMES[d]) & (t.Classifier == config.CLASSIFIER_NAMES[c])]
                if r.empty:
                    continue
                r = r.iloc[0]
                xpos = ci + (si - (len(settings) - 1) / 2) * 0.26
                if r["Discovery Proportion"] == "0%":
                    ax.text(xpos, 0.002, "0%\nhighest", ha="center", va="bottom", fontsize=6.5, color=MUTED)
                    continue
                vals = [r[f"Seed {s}"] for s in REPL if f"Seed {s}" in r and r[f"Seed {s}"] != ""]
                step = 0.045 if len(settings) == 1 else 0.035
                offs = (np.arange(len(vals)) - (len(vals) - 1) / 2) * step
                ax.scatter(xpos + offs, vals, s=34, color=SERIES_COLORS[si], marker=MARKERS[si],
                           edgecolor="white", linewidth=1.2, zorder=3,
                           label=config.SETTING_NAMES[st] if (d == DS_ORDER[0] and ci == 0) else None)
                ax.scatter([xpos], [r["Discovery Gain"]], s=60, facecolors="none",
                           edgecolors=SERIES_COLORS[si], marker=MARKERS[si], linewidths=1.4, zorder=3)
                ax.text(xpos, -0.079, r["Discovery Proportion"],
                        ha="center", fontsize=6.5, color=MUTED)
            labels.append(config.CLASSIFIER_NAMES[c])
        ax.set_xticks(range(len(config.CLASSIFIERS)))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.6, len(config.CLASSIFIERS) - 0.4)
        ax.set_title(DS_NAMES[d] + title_suffix)
    axes[0].set_ylabel("F1 at discovery proportion minus F1 at 0%")
    axes[0].set_ylim(-0.085, 0.06)
    if len(settings) > 1:
        _legend_first(axes, loc="upper left", fontsize=7.5)
    fig.text(0.5, -0.03, f"Filled = replication runs (seeds {REPL[0]}-{REPL[-1]}); hollow = discovery run "
             f"(seed {DISC}); grey label = proportion tested", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_ablation_grid(rep_tables, path):
    """3 x 3 grid (dataset x classifier); x = setting. Same reading as the RQ2 figure."""
    plt = _plt()
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.6), sharey=True)
    for i, d in enumerate(DS_ORDER):
        for j, c in enumerate(config.CLASSIFIERS):
            ax = axes[i, j]
            ax.axhline(0, color="#333333", lw=1)
            ticks = []
            for si, st in enumerate(config.SETTINGS):
                t = rep_tables[st]
                r = t[(t.Dataset == DS_NAMES[d]) & (t.Classifier == config.CLASSIFIER_NAMES[c])]
                short = {"main": "Main", "B": "B", "C": "C"}[st]
                if r.empty:
                    ticks.append(short)
                    continue
                r = r.iloc[0]
                ticks.append(f"{short}\n{r['Discovery Proportion']}")
                if r["Discovery Proportion"] == "0%":
                    ax.text(si, 0.004, "0%\nhighest", ha="center", va="bottom", fontsize=7, color=MUTED)
                    continue
                vals = [r[f"Seed {s}"] for s in REPL if r[f"Seed {s}"] != ""]
                offs = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.09
                ax.scatter(si + offs, vals, s=30, color=SERIES_COLORS[si], marker=MARKERS[si],
                           edgecolor="white", linewidth=1.0, zorder=3)
                ax.scatter([si], [r["Discovery Gain"]], s=75, facecolors="none",
                           edgecolors=SERIES_COLORS[si], marker=MARKERS[si], linewidths=1.5, zorder=4)
                if r["Replicated in All Runs"] == "yes":
                    ax.text(si, 0.052, "all 4\nabove", ha="center", fontsize=7, color="#333333")
            ax.set_xticks(range(len(config.SETTINGS)))
            ax.set_xticklabels(ticks, fontsize=7.5)
            ax.set_xlim(-0.6, len(config.SETTINGS) - 0.4)
            ax.set_title(f"{DS_NAMES[d]} - {config.CLASSIFIER_NAMES[c]}", fontsize=9)
            if j == 0:
                ax.set_ylabel("F1 minus F1 at 0%")
    axes[0, 0].set_ylim(-0.08, 0.068)
    fig.text(0.5, -0.01, f"Filled = replication runs (seeds {REPL[0]}-{REPL[-1]}); hollow = discovery run (seed {DISC}); "
             "tick label = setting and the proportion tested", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_per_level(df, path):
    plt = _plt()
    s = df[(df.setting == "main") & (df.rep.isin(REPL))]
    counts, total = [], 0
    for k in range(1, config.N_LEVELS):
        n = tot = 0
        for d, c in _combos():
            g = s[(s.dataset == d) & (s.classifier == c)].pivot_table(index="rep", columns="level", values="f1")
            dd = (g[k] - g[0]).to_numpy()
            n += int((dd > TOL).sum())
            tot += len(dd)
        counts.append(n)
        total = tot
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(1, config.N_LEVELS) * 10
    ax.bar(x, counts, width=6, color=SERIES_COLORS[0])
    ax.axhline(total / 2, color=MUTED, lw=1, ls="--")
    ax.text(x[-1] + 4, total / 2, "half of runs", va="center", fontsize=8, color=MUTED)
    for xi, n in zip(x, counts):
        ax.text(xi, n + 0.4, str(n), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xlabel("Mixing proportion (% synthetic)")
    ax.set_ylabel(f"Runs above 0% (of {total})")
    ax.set_ylim(0, total * 0.6)
    ax.set_title("Replication runs in which a fixed proportion beats the 0% baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_variability(df, d, path):
    """Mean +/- SD over five runs, main vs ablations (context on variability)."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    x = np.array(config.LEVELS) * 10
    for ax, c in zip(axes, config.CLASSIFIERS):
        for i, st in enumerate(config.SETTINGS):
            piv = _pivot(df, d, c, st)
            if piv.empty:
                continue
            ax.errorbar(x + (i - 1) * 0.8, piv.mean().to_numpy(), yerr=piv.std(ddof=1).to_numpy(),
                        color=SERIES_COLORS[i], marker=MARKERS[i], ms=4.5, lw=1.8, capsize=2,
                        elinewidth=0.9, label=config.SETTING_NAMES[st])
        ax.set_title(f"{DS_NAMES[d]} - {config.CLASSIFIER_NAMES[c]}")
        ax.set_xticks(x)
        ax.set_xlabel("Mixing proportion (% synthetic)")
    axes[0].set_ylabel("F1-score (mean ± SD, five runs)")
    handles, labels = next((a.get_legend_handles_labels() for a in axes if a.get_legend_handles_labels()[0]), ([], []))
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_rq3_trend(pairs, path):
    """Top: fidelity per run (ordered low -> high). Bottom: F1 of the three
    classifiers in the same order. Two panels, no dual axis."""
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), gridspec_kw={"height_ratios": [1, 1.5]})
    for j, d in enumerate(DS_ORDER):
        g = pairs[pairs.dataset == d]
        if g.empty:
            continue
        order = g.drop_duplicates("seed").sort_values("fidelity")
        seeds = order["seed"].tolist()
        xs = np.arange(len(seeds))
        top, bot = axes[0, j], axes[1, j]
        top.plot(xs, order["fidelity"], color=MUTED, marker="o", ms=6, lw=1.2)
        lo, hi = order["fidelity"].min(), order["fidelity"].max()
        pad = max((hi - lo) * 0.6, 0.005)
        top.set_ylim(lo - pad, hi + pad)
        for xi, v in zip(xs, order["fidelity"]):
            top.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
        top.set_title(DS_NAMES[d])
        top.set_xticks(xs)
        top.set_xticklabels([])
        for i, c in enumerate(config.CLASSIFIERS):
            gc = g[g.classifier == c].set_index("seed").loc[seeds]
            bot.plot(xs, gc["f1"], color=SERIES_COLORS[i], marker=MARKERS[i], ms=5, lw=2,
                     label=config.CLASSIFIER_NAMES[c])
        bot.set_xticks(xs)
        bot.set_xticklabels([f"seed {s}" for s in seeds], fontsize=7.5)
        bot.set_xlabel("Runs ordered by fidelity (low → high)")
    axes[0, 0].set_ylabel("Combined fidelity\n(axis zoomed)")
    axes[1, 0].set_ylabel("F1 at 100% mixing")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_rq3_scatter(pairs, path):
    plt = _plt()
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 8.5))
    for i, d in enumerate(DS_ORDER):
        for j, c in enumerate(config.CLASSIFIERS):
            ax = axes[i, j]
            g = pairs[(pairs.dataset == d) & (pairs.classifier == c)]
            if g.empty:
                ax.set_visible(False)
                continue
            ax.scatter(g["fidelity"], g["f1"], s=40, color=SERIES_COLORS[j], marker=MARKERS[j],
                       edgecolor="white", linewidth=1.5, zorder=3)
            for _, r in g.iterrows():
                ax.annotate(str(r["seed"]), (r["fidelity"], r["f1"]), textcoords="offset points",
                            xytext=(4, 3), fontsize=7, color=MUTED)
            ax.set_title(f"{DS_NAMES[d]} - {config.CLASSIFIER_NAMES[c]}", fontsize=9)
            if i == 2:
                ax.set_xlabel("Combined fidelity (predictors)")
            if j == 0:
                ax.set_ylabel("F1 at 100% mixing")
            ax.ticklabel_format(useOffset=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_rq3_variables(fid_vars, path):
    plt = _plt()
    sizes = [fid_vars[fid_vars.dataset == d]["variable"].nunique() for d in DS_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(12, max(4, 0.22 * max(sizes) + 1.2)))
    for ax, d in zip(axes, DS_ORDER):
        g = fid_vars[(fid_vars.dataset == d) & (~fid_vars["type"].str.startswith("target"))]
        if g.empty:
            ax.set_visible(False)
            continue
        s = g.groupby(["variable", "type"])["score"].agg(["mean", "std"]).reset_index().sort_values("mean")
        ypos = np.arange(len(s))
        for yi, (_, row) in zip(ypos, s.iterrows()):
            k = 1 if row["type"] == "discrete" else 0
            ax.errorbar(row["mean"], yi, xerr=row["std"], fmt=MARKERS[k], color=SERIES_COLORS[k],
                        ms=5.5, elinewidth=0.9, capsize=2)
        ax.set_yticks(ypos)
        ax.set_yticklabels(s["variable"])
        ax.set_xlim(0.5, 1.0)
        ax.set_title(DS_NAMES[d])
        ax.set_xlabel("Fidelity score (mean ± SD, five runs)")
        ax.tick_params(axis="y", labelsize=7)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=SERIES_COLORS[0], marker=MARKERS[0], ls=""),
                Line2D([], [], color=SERIES_COLORS[1], marker=MARKERS[1], ls="")],
               ["Continuous (KSComplement)", "Discrete (TVComplement)"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_rq3_ranges(desc, path):
    """Spread over five runs: fidelity vs F1 (both on the 0-1 score scale)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    groups = [(d, c) for d in DS_ORDER for c in config.CLASSIFIERS]
    x = np.arange(len(groups))
    fr, yr = [], []
    for d, c in groups:
        r = desc[(desc.Dataset == DS_NAMES[d]) & (desc.Classifier == config.CLASSIFIER_NAMES[c])]
        fr.append(float(r["Fidelity Range"].iloc[0]) if len(r) else np.nan)
        yr.append(float(r["F1 Range"].iloc[0]) if len(r) else np.nan)
    ax.bar(x - 0.2, fr, width=0.38, color=MUTED, label="Fidelity range")
    ax.bar(x + 0.2, yr, width=0.38, color=SERIES_COLORS[0], label="F1 range (100% mixing)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{DS_NAMES[d]}\n{config.CLASSIFIER_NAMES[c]}" for d, c in groups], fontsize=7.5)
    ax.set_ylabel("Max minus min over five runs")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_analysis(out_dir):
    df, fid_vars, fid_sum, status = load(out_dir)
    tdir, fdir = os.path.join(out_dir, "tables"), os.path.join(out_dir, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    status.to_csv(os.path.join(tdir, "repetition_status.csv"), index=False)

    out = {}
    out["rq1_discovery_best"], out["rq1_discovery_trajectories"], out["rq1_discovery_shape"] = rq1_tables(df)
    rep_tables = {st: replication_table(df, st) for st in config.SETTINGS}
    out["rq2_replication_main"] = rep_tables["main"]
    out["rq2_per_level_above_baseline"] = per_level_above_baseline(df)
    out["rq2_best_mixed_vs_baseline"] = best_mixed_vs_baseline(df)
    out["ablation_replication_all_settings"] = pd.concat(rep_tables.values(), ignore_index=True)
    out["ablation_count_summary"] = ablation_count_summary(rep_tables)
    out["ablation_per_level_vs_main"] = ablation_per_level_vs_main(df)
    out["ablation_b_class_shares"] = ablation_b_class_shares(df)
    out["ablation_b_confusion"] = ablation_b_confusion(df, rep_tables)
    out["ablation_c_smoothness"], out["ablation_c_adjacent_pairs"] = ablation_c_smoothness(df)
    out["rq3_fidelity_by_variable"] = rq3_fidelity_table(fid_vars, fid_sum)
    pairs = rq3_pairs(df, fid_sum)
    out["rq3_fidelity_f1_pairs"] = pairs
    out["rq3_trend_by_fidelity"] = rq3_trend_table(pairs)
    out["rq3_descriptive"] = rq3_descriptive(pairs)
    out["context_mean_sd_all_runs"] = mean_sd_context(df)
    out["supporting_metrics_all_settings"] = supporting_metrics(df)
    flags = df[df.no_positive_prediction.astype(bool)]
    out["no_positive_prediction_flags"] = flags[["dataset", "classifier", "setting", "level", "rep"]]

    # remove tables from the earlier (five-run-mean) analysis so they are not mistaken for current ones
    for old in glob.glob(os.path.join(tdir, "table_4_*.csv")) + [
            os.path.join(tdir, n) for n in ["ablation_best_proportion_summary.csv", "rq1_shape_descriptors.csv",
                                            "paired_diff_vs_baseline_all_levels.csv", "f1_aggregated_all_settings.csv",
                                            "rq3_descriptive.csv"]]:
        if os.path.exists(old):
            os.remove(old)
    for name, t in out.items():
        t.to_csv(os.path.join(tdir, name + ".csv"), index=False)
    try:
        with pd.ExcelWriter(os.path.join(tdir, "all_tables.xlsx")) as xw:
            for name, t in out.items():
                t.to_excel(xw, sheet_name=name[:31], index=False)
    except Exception as e:
        print(f"(Excel export skipped: {e})")

    for old in glob.glob(os.path.join(fdir, "fig_*.png")):
        os.remove(old)
    fig_rq1(df, os.path.join(fdir, "fig_rq1_discovery_trajectories.png"))
    fig_replication(rep_tables, ["main"], os.path.join(fdir, "fig_rq2_replication.png"))
    fig_per_level(df, os.path.join(fdir, "fig_rq2_per_level_above_baseline.png"))
    fig_ablation_grid(rep_tables, os.path.join(fdir, "fig_ablation_replication.png"))
    for d in DS_ORDER:
        if (df.dataset == d).any():
            fig_variability(df, d, os.path.join(fdir, f"fig_variability_{d}.png"))
    fig_rq3_trend(pairs, os.path.join(fdir, "fig_rq3_trend_by_fidelity.png"))
    fig_rq3_scatter(pairs, os.path.join(fdir, "fig_rq3_scatter.png"))
    fig_rq3_variables(fid_vars, os.path.join(fdir, "fig_rq3_variable_fidelity.png"))
    fig_rq3_ranges(out["rq3_descriptive"], os.path.join(fdir, "fig_rq3_ranges.png"))

    pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 70)
    print("\nValid repetitions per dataset:", status[status.status == "ok"].groupby("dataset").size().to_dict())
    print(f"\n=== RQ1: highest-F1 proportion in the discovery run (seed {DISC}) ===")
    print(out["rq1_discovery_best"].to_string(index=False))
    print(f"\n=== RQ2: replication in seeds {REPL} ===")
    print(out["rq2_replication_main"].to_string(index=False))
    print("\n=== Ablations: replicated gains per setting ===")
    print(out["ablation_count_summary"].to_string(index=False))
    print(f"\nTables -> {tdir}\nFigures -> {fdir}")
