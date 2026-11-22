"""Aggregation, Chapter 4 tables and figures (Sections 3.8.4, 3.9.2, 3.10, 4.3).

All summaries are descriptive: means and sample SDs (ddof = 1) over the valid
repetitions, paired differences computed within repetitions first.
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

# Reference categorical palette (dataviz skill, light mode), fixed order, plus
# distinct markers so identity never depends on colour alone.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]


# ---------------------------------------------------------------------------
# Loading
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
    return (pd.concat(metrics, ignore_index=True), pd.concat(fid_vars, ignore_index=True),
            pd.DataFrame(fid_sum), pd.DataFrame(status))


def _sd(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan


def _fmt(m, s, nd=3):
    if pd.isna(m):
        return ""
    return f"{m:.{nd}f} ({s:.{nd}f})" if not pd.isna(s) else f"{m:.{nd}f}"


def _pct(k):
    return f"{k * 10}%"


def _combos(df):
    for d in DS_ORDER:
        for c in config.CLASSIFIERS:
            sub = df[(df.dataset == d) & (df.classifier == c)]
            if len(sub):
                yield d, c, sub


# ---------------------------------------------------------------------------
# Trajectory summaries
# ---------------------------------------------------------------------------
def aggregate(df, metric="f1"):
    g = df.groupby(["dataset", "classifier", "setting", "level"])[metric]
    out = g.agg(mean="mean", sd=lambda x: _sd(x), n_reps="count").reset_index()
    return out


def best_summary(sub, setting):
    """Highest mean F1, near-maximum band, per-run best, paired diff vs 0%."""
    s = sub[sub.setting == setting]
    piv = s.pivot_table(index="rep", columns="level", values="f1")
    means = piv.mean(axis=0)
    sds = piv.std(axis=0, ddof=1)
    mx = means.max()
    tied = [int(k) for k in means.index if abs(means[k] - mx) <= TOL]
    ref = tied[0]                                    # lowest tied level supplies the SD
    band = [int(k) for k in means.index if means[k] >= mx - sds[ref] - TOL]
    per_run = {}
    for rep, row in piv.iterrows():
        rmx = row.max()
        per_run[int(rep)] = [int(k) for k in row.index if abs(row[k] - rmx) <= TOL]
    diffs = (piv[ref] - piv[0]).to_numpy()
    return {
        "best_levels": tied, "best_level": ref, "f1_best_mean": means[ref], "f1_best_sd": sds[ref],
        "f1_0_mean": means[0], "f1_0_sd": sds[0], "band": band, "per_run_best": per_run,
        "per_run_f1_best": piv[ref].to_dict(), "per_run_f1_0": piv[0].to_dict(),
        "diffs": diffs, "diff_mean": float(np.mean(diffs)), "diff_sd": _sd(diffs),
        "n_pos": int((diffs > TOL).sum()), "n_zero": int((np.abs(diffs) <= TOL).sum()),
        "n_neg": int((diffs < -TOL).sum()), "n_reps": int(len(piv)),
        "means": means, "sds": sds,
    }


def table_4_4(agg):
    m = agg[agg.setting == "main"]
    rows = []
    for d in DS_ORDER:
        for c in config.CLASSIFIERS:
            r = m[(m.dataset == d) & (m.classifier == c)].set_index("level")
            if r.empty:
                continue
            row = {"Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c]}
            for k in config.LEVELS:
                row[_pct(k)] = _fmt(r.loc[k, "mean"], r.loc[k, "sd"]) if k in r.index else ""
            row["Reps"] = int(r["n_reps"].min())
            rows.append(row)
    return pd.DataFrame(rows)


def tables_4_5_4_6(df):
    t5, t6 = [], []
    for d, c, sub in _combos(df):
        b = best_summary(sub, "main")
        reps = sorted(b["per_run_best"])
        t5.append({
            "Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c],
            "Best Proportion (Mean F1)": ", ".join(_pct(k) for k in b["best_levels"]),
            "F1 at Best": _fmt(b["f1_best_mean"], b["f1_best_sd"]),
            "F1 at 0%": _fmt(b["f1_0_mean"], b["f1_0_sd"]),
            "Near-Maximum Band": ", ".join(_pct(k) for k in b["band"]),
            "Best Proportion in Runs": "; ".join(
                f"r{r}: " + "/".join(_pct(k) for k in b["per_run_best"][r]) for r in reps),
        })
        row = {"Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c],
               "Best Proportion": _pct(b["best_level"])}
        for i, r in enumerate(reps, 1):
            row[f"Run {i} at best"] = round(b["per_run_f1_best"][r], 4)
        for i, r in enumerate(reps, 1):
            row[f"Run {i} at 0%"] = round(b["per_run_f1_0"][r], 4)
        row["Paired Differences (Best minus 0%)"] = ", ".join(f"{x:+.4f}" for x in b["diffs"])
        row["Mean Difference (SD)"] = _fmt(b["diff_mean"], b["diff_sd"], 4)
        row["Runs Positive / Zero / Negative"] = f"{b['n_pos']} / {b['n_zero']} / {b['n_neg']}"
        row["Note"] = ("best is the 0% baseline: no evidence of a gain from mixing"
                       if b["best_level"] == 0 else "")
        t6.append(row)
    return pd.DataFrame(t5), pd.DataFrame(t6)


def shape_descriptors(df):
    """Predefined descriptive shape measures for RQ1 (no tests)."""
    rows = []
    for d, c, sub in _combos(df):
        s = sub[sub.setting == "main"]
        piv = s.pivot_table(index="rep", columns="level", values="f1")
        means, sds = piv.mean(), piv.std(ddof=1)
        lo = means[0] - sds[0]
        below = [int(k) for k in means.index if k > 0 and means[k] < lo]
        steps = np.diff(means.to_numpy())
        d100 = (piv[10] - piv[0]).to_numpy()
        rows.append({
            "Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c],
            "F1 at 0%": round(means[0], 4), "F1 at 100%": round(means[10], 4),
            "100% minus 0% (paired, mean (SD))": _fmt(np.mean(d100), _sd(d100), 4),
            "First level below 0% mean - SD": _pct(below[0]) if below else "none",
            "Levels below 0% mean - SD": ", ".join(_pct(k) for k in below),
            "Adjacent increases / decreases (of 10)": f"{int((steps > 0).sum())} / {int((steps < 0).sum())}",
            "Largest single drop (between)": (f"{_pct(int(np.argmin(steps)))}-{_pct(int(np.argmin(steps)) + 1)}"
                                              f" ({steps.min():+.4f})"),
            "Range of means (max - min)": round(means.max() - means.min(), 4),
        })
    return pd.DataFrame(rows)


def supporting_metrics(df):
    cols = ["f1", "precision", "recall", "specificity", "fpr", "accuracy", "TP", "FP", "TN", "FN"]
    g = df.groupby(["dataset", "classifier", "setting", "level"])[cols]
    mean = g.mean().add_suffix("_mean")
    sd = g.std(ddof=1).add_suffix("_sd")
    out = pd.concat([mean, sd], axis=1).reset_index()
    order = ["dataset", "classifier", "setting", "level"] + [x for c in cols for x in (c + "_mean", c + "_sd")]
    return out[order]


def paired_vs_baseline_all_levels(df):
    rows = []
    for d, c, sub in _combos(df):
        s = sub[sub.setting == "main"].pivot_table(index="rep", columns="level", values="f1")
        for k in config.LEVELS:
            dd = (s[k] - s[0]).to_numpy()
            rows.append({"dataset": d, "classifier": c, "level": k, "diff_mean": np.mean(dd),
                         "diff_sd": _sd(dd), "n_pos": int((dd > TOL).sum()),
                         "n_zero": int((np.abs(dd) <= TOL).sum()), "n_neg": int((dd < -TOL).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------
def ablation_per_level(df):
    """Table 4.7, long form (297 rows). Paired difference = ablated - main."""
    main = df[df.setting == "main"].set_index(["dataset", "classifier", "level", "rep"])
    rows = []
    for st in ["A", "B", "C"]:
        ab = df[df.setting == st].set_index(["dataset", "classifier", "level", "rep"])
        j = ab.join(main[["f1", "pos_share_total"]], rsuffix="_main")
        j["diff"] = j["f1"] - j["f1_main"]
        for (d, c, k), g in j.groupby(level=[0, 1, 2]):
            rows.append({
                "Ablation": st, "dataset": d, "classifier": c, "level": k,
                "main_mean": g["f1_main"].mean(), "main_sd": _sd(g["f1_main"]),
                "ablated_mean": g["f1"].mean(), "ablated_sd": _sd(g["f1"]),
                "paired_diff_mean": g["diff"].mean(), "paired_diff_sd": _sd(g["diff"]),
                "diff_runs_pos/zero/neg": f"{int((g['diff'] > TOL).sum())}/"
                                          f"{int((g['diff'].abs() <= TOL).sum())}/{int((g['diff'] < -TOL).sum())}",
                "|diff| vs main SD": (abs(g["diff"].mean()) / _sd(g["f1_main"])
                                      if _sd(g["f1_main"]) and _sd(g["f1_main"]) > 0 else np.nan),
                "n_total_mean": g["n_total"].mean(),
                "realised_syn_share_mean": g["realised_syn_share"].mean(),
                "pos_share_real_mean": g["pos_share_real"].mean() if st == "B" else np.nan,
                "pos_share_syn_mean": g["pos_share_syn"].mean() if st == "B" else np.nan,
                "pos_share_total_mean": g["pos_share_total"].mean() if st == "B" else np.nan,
                "pos_share_total_sd": _sd(g["pos_share_total"]) if st == "B" else np.nan,
            })
    out = pd.DataFrame(rows)
    out["dataset"] = pd.Categorical(out["dataset"], DS_ORDER)
    return out.sort_values(["Ablation", "dataset", "classifier", "level"]).reset_index(drop=True)


def ablation_best_summary(df):
    """Where the highest-F1 level sits under each setting (the core ablation output)."""
    rows = []
    for d, c, sub in _combos(df):
        for st in config.SETTINGS:
            if (sub.setting == st).sum() == 0:
                continue
            b = best_summary(sub, st)
            reps = sorted(b["per_run_best"])
            rows.append({
                "Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c], "Setting": st,
                "Best Level (Mean F1)": ", ".join(_pct(k) for k in b["best_levels"]),
                "F1 at Best": _fmt(b["f1_best_mean"], b["f1_best_sd"]),
                "Near-Maximum Band": ", ".join(_pct(k) for k in b["band"]),
                "Best Level in Runs": "; ".join("/".join(_pct(k) for k in b["per_run_best"][r]) for r in reps),
                "Best minus 0% (mean (SD))": _fmt(b["diff_mean"], b["diff_sd"], 4),
                "Runs + / 0 / -": f"{b['n_pos']} / {b['n_zero']} / {b['n_neg']}",
            })
    return pd.DataFrame(rows)


def ablation_c_smoothness(df):
    """Nested (main) vs independent (C): per-level SD, mean |adjacent change|,
    SD of paired adjacent changes - adjacent changes computed within each run."""
    per_pair, summary = [], []
    for d, c, sub in _combos(df):
        res = {}
        for st in ["main", "C"]:
            piv = sub[sub.setting == st].pivot_table(index="rep", columns="level", values="f1")
            level_sd = piv.std(ddof=1)
            delta = piv.diff(axis=1).iloc[:, 1:]            # within-run adjacent changes
            for k in range(config.N_LEVELS - 1):
                dk = delta.iloc[:, k].to_numpy()
                per_pair.append({"dataset": d, "classifier": c,
                                 "version": "nested (main)" if st == "main" else "independent (C)",
                                 "pair": f"{_pct(k)}-{_pct(k + 1)}",
                                 "mean_abs_change": float(np.mean(np.abs(dk))),
                                 "sd_of_change": _sd(dk)})
            res[st] = {
                "mean_level_sd_10_90": float(level_sd.loc[1:9].mean()),
                "mean_abs_adjacent_change": float(np.mean(np.abs(delta.to_numpy()))),
                "mean_sd_of_adjacent_change": float(np.mean([_sd(delta.iloc[:, k]) for k in range(delta.shape[1])])),
            }
        summary.append({
            "Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c],
            "Per-level SD 10-90% (nested)": round(res["main"]["mean_level_sd_10_90"], 4),
            "Per-level SD 10-90% (independent)": round(res["C"]["mean_level_sd_10_90"], 4),
            "Mean |adjacent change| (nested)": round(res["main"]["mean_abs_adjacent_change"], 4),
            "Mean |adjacent change| (independent)": round(res["C"]["mean_abs_adjacent_change"], 4),
            "SD of adjacent change (nested)": round(res["main"]["mean_sd_of_adjacent_change"], 4),
            "SD of adjacent change (independent)": round(res["C"]["mean_sd_of_adjacent_change"], 4),
        })
    return pd.DataFrame(summary), pd.DataFrame(per_pair)


# ---------------------------------------------------------------------------
# Fidelity (Table 4.3) and RQ3 pairs (Table 4.8)
# ---------------------------------------------------------------------------
def table_4_3(fid_vars, fid_sum):
    rows = []
    for d in DS_ORDER:
        fv = fid_vars[fid_vars.dataset == d]
        if fv.empty:
            continue
        for (var, typ, meas), g in fv.groupby(["variable", "type", "measure"], sort=False):
            rows.append({"Dataset": DS_NAMES[d], "Variable": var, "Type": typ, "Measure": meas,
                         "Mean over runs": round(g["score"].mean(), 4),
                         "SD over runs": round(_sd(g["score"]), 4), "Runs": len(g)})
        fs = fid_sum[fid_sum.dataset == d]
        for col, label in [("continuous_mean", "Continuous predictors (mean)"),
                           ("discrete_mean", "Discrete predictors (mean)"),
                           ("combined_predictors", "Combined score (predictors only)"),
                           ("pool_pos_share", "Unfiltered pool positive share")]:
            vals = fs[col].astype(float)
            rows.append({"Dataset": DS_NAMES[d], "Variable": label, "Type": "summary",
                         "Measure": "" if col != "pool_pos_share" else "class frequency",
                         "Mean over runs": "not applicable" if vals.isna().all() else round(vals.mean(), 4),
                         "SD over runs": "" if vals.isna().all() else round(_sd(vals), 4),
                         "Runs": len(fs)})
    return pd.DataFrame(rows)


def table_4_8(df, fid_sum):
    m = df[(df.setting == "main") & (df.level == 10)][["dataset", "classifier", "rep", "f1"]]
    j = m.merge(fid_sum[["dataset", "rep", "combined_predictors"]], on=["dataset", "rep"])
    j["dataset"] = pd.Categorical(j["dataset"], DS_ORDER)
    j = j.sort_values(["dataset", "classifier", "rep"])
    return j.rename(columns={"combined_predictors": "Predictor-Based Combined Fidelity",
                             "f1": "F1 at 100%", "rep": "Seed"})


def rq3_descriptive(t48):
    """Descriptive reading of the 5 pairs per combination, prespecified rule:
    compare F1 at the highest- and lowest-fidelity runs, and report the spread
    of fidelity relative to the spread of F1 (range restriction check)."""
    rows = []
    for (d, c), g in t48.groupby(["dataset", "classifier"], observed=True):
        g = g.sort_values("Predictor-Based Combined Fidelity")
        f = g["Predictor-Based Combined Fidelity"].to_numpy()
        y = g["F1 at 100%"].to_numpy()
        conc = disc = 0
        for i in range(len(f)):
            for k in range(i + 1, len(f)):
                s = np.sign(f[k] - f[i]) * np.sign(y[k] - y[i])
                conc += s > 0
                disc += s < 0
        rows.append({
            "Dataset": DS_NAMES[d], "Classifier": config.CLASSIFIER_NAMES[c],
            "Fidelity range": round(f.max() - f.min(), 4),
            "F1 range": round(y.max() - y.min(), 4),
            "F1 at highest-fidelity run": round(y[-1], 4),
            "F1 at lowest-fidelity run": round(y[0], 4),
            "Pairs same direction / opposite (of 10)": f"{int(conc)} / {int(disc)}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
    return plt


def fig_main(agg, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    x = np.array(config.LEVELS) * 10
    for ax, d in zip(axes, DS_ORDER):
        for i, c in enumerate(config.CLASSIFIERS):
            r = agg[(agg.setting == "main") & (agg.dataset == d) & (agg.classifier == c)].sort_values("level")
            if r.empty:
                continue
            ax.errorbar(x[: len(r)] + (i - 1) * 0.8, r["mean"], yerr=r["sd"], color=SERIES_COLORS[i],
                        marker=MARKERS[i], ms=5, lw=2, capsize=2, elinewidth=1,
                        label=config.CLASSIFIER_NAMES[c])
        ax.set_title(DS_NAMES[d])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}" for v in x])
        ax.set_xlabel("Mixing proportion (% synthetic)")
    axes[0].set_ylabel("F1-score (mean ± SD over runs)")
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, loc="lower left")
            break
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_ablation(agg, d, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    x = np.array(config.LEVELS) * 10
    for ax, c in zip(axes, config.CLASSIFIERS):
        for i, st in enumerate(config.SETTINGS):
            r = agg[(agg.setting == st) & (agg.dataset == d) & (agg.classifier == c)].sort_values("level")
            if r.empty:
                continue
            ax.errorbar(x[: len(r)] + (i - 1.5) * 0.7, r["mean"], yerr=r["sd"], color=SERIES_COLORS[i],
                        marker=MARKERS[i], ms=4.5, lw=1.8, capsize=2, elinewidth=0.9,
                        label=config.SETTING_NAMES[st])
        ax.set_title(f"{DS_NAMES[d]} - {config.CLASSIFIER_NAMES[c]}")
        ax.set_xticks(x)
        ax.set_xlabel("Matched mixing level (% synthetic)")
    axes[0].set_ylabel("F1-score (mean ± SD)")
    handles, labels = next((a.get_legend_handles_labels() for a in axes
                            if a.get_legend_handles_labels()[0]), ([], []))
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_rq3(t48, path):
    plt = _plt()
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 8.5))
    for i, d in enumerate(DS_ORDER):
        for j, c in enumerate(config.CLASSIFIERS):
            ax = axes[i, j]
            g = t48[(t48.dataset == d) & (t48.classifier == c)]
            if g.empty:
                ax.set_visible(False)
                continue
            ax.scatter(g["Predictor-Based Combined Fidelity"], g["F1 at 100%"], s=36,
                       color=SERIES_COLORS[0], edgecolor="white", linewidth=1.5, zorder=3)
            for _, r in g.iterrows():
                ax.annotate(str(r["Seed"]), (r["Predictor-Based Combined Fidelity"], r["F1 at 100%"]),
                            textcoords="offset points", xytext=(4, 3), fontsize=7, color="#555555")
            ax.set_title(f"{DS_NAMES[d]} - {config.CLASSIFIER_NAMES[c]}", fontsize=9)
            if i == 2:
                ax.set_xlabel("Combined fidelity (predictors)")
            if j == 0:
                ax.set_ylabel("F1 at 100% mixing")
            ax.ticklabel_format(useOffset=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_analysis(out_dir):
    df, fid_vars, fid_sum, status = load(out_dir)
    tdir = os.path.join(out_dir, "tables")
    fdir = os.path.join(out_dir, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    agg = aggregate(df)
    agg.to_csv(os.path.join(tdir, "f1_aggregated_all_settings.csv"), index=False)
    status.to_csv(os.path.join(tdir, "repetition_status.csv"), index=False)

    out = {}
    out["table_4_4_main_f1"] = table_4_4(agg)
    out["table_4_5_best_proportion"], out["table_4_6_consistency"] = tables_4_5_4_6(df)
    out["rq1_shape_descriptors"] = shape_descriptors(df)
    out["paired_diff_vs_baseline_all_levels"] = paired_vs_baseline_all_levels(df)
    out["supporting_metrics_all_settings"] = supporting_metrics(df)
    out["table_4_7_ablation_per_level"] = ablation_per_level(df)
    out["ablation_best_proportion_summary"] = ablation_best_summary(df)
    out["ablation_c_smoothness"], out["ablation_c_adjacent_pairs"] = ablation_c_smoothness(df)
    out["table_4_3_fidelity"] = table_4_3(fid_vars, fid_sum)
    t48 = table_4_8(df, fid_sum)
    out["table_4_8_fidelity_f1_pairs"] = t48
    out["rq3_descriptive"] = rq3_descriptive(t48)
    flags = df[df.no_positive_prediction.astype(bool)]
    out["no_positive_prediction_flags"] = flags[["dataset", "classifier", "setting", "level", "rep"]]

    for name, t in out.items():
        t.to_csv(os.path.join(tdir, name + ".csv"), index=False)
    try:
        with pd.ExcelWriter(os.path.join(tdir, "all_tables.xlsx")) as xw:
            for name, t in out.items():
                t.to_excel(xw, sheet_name=name[:31], index=False)
    except Exception as e:      # openpyxl missing is fine - CSVs are the record
        print(f"(Excel export skipped: {e})")

    fig_main(agg, os.path.join(fdir, "fig_main_trajectories.png"))
    for d in DS_ORDER:
        if (agg.dataset == d).any():
            fig_ablation(agg, d, os.path.join(fdir, f"fig_ablation_{d}.png"))
    fig_rq3(t48, os.path.join(fdir, "fig_rq3_fidelity_vs_f1.png"))

    n_ok = status[status.status == "ok"].groupby("dataset").size().to_dict()
    print("\nValid repetitions per dataset:", n_ok)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60)
    print("\n=== Table 4.5 (highest-F1 proportion) ===")
    print(out["table_4_5_best_proportion"].to_string(index=False))
    print("\n=== Ablation: highest-F1 level by setting ===")
    print(out["ablation_best_proportion_summary"].to_string(index=False))
    print(f"\nTables -> {tdir}\nFigures -> {fdir}")
