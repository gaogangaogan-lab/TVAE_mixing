"""Command-line entry point.

  python run.py check                      # environment + data + split check
  python run.py tune                       # hyperparameter rerun (Section 3.7)
  python run.py experiment                 # main + ablations, 5 repetitions
  python run.py analyze                    # Chapter 4 tables and figures
  python run.py all                        # tune -> experiment -> analyze

  add --smoke for a quick end-to-end check (full TVAE, tiny search,
  one repetition, separate output folder). Smoke results are NOT for the report.
"""
import os

# Must be set before torch initialises CUDA, for deterministic cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse
import json
import platform
import sys
import time
import warnings

import config

# sklearn >= 1.8 deprecates 'penalty' (still valid until 1.10); keep the log readable
warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*", category=FutureWarning)
# Harmless: XGBoost trains on the GPU and predicts on CPU arrays (results identical)
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
# SDV advice; the metadata is already archived in generator_info.json
warnings.filterwarnings("ignore", message=".*saving the metadata using 'save_to_json'.*")


def detect_device(requested):
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def record_environment(out_dir, device):
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "device": device}
    for mod in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "sdv", "sdmetrics", "ctgan",
                "rdt", "torch", "matplotlib"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except ImportError:
            info[mod] = "not installed"
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def load_params(args, out_dir):
    if args.use_thesis_params:
        print("Using the Table 3.8 configurations (before the rerun).")
        return config.THESIS_TABLE_3_8
    path = args.params or os.path.join(out_dir, "tuning", "best_params.json")
    if not os.path.exists(path):
        raise SystemExit(f"No tuned parameters at {path}. Run 'python run.py tune' first, "
                         f"or pass --use-thesis-params to use Table 3.8.")
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="TVAE fixed-budget mixing experiment")
    ap.add_argument("stage", choices=["check", "tune", "experiment", "analyze", "all"])
    ap.add_argument("--datasets", nargs="+", default=["pima", "cleveland", "wdbc"],
                    choices=["pima", "cleveland", "wdbc"])
    ap.add_argument("--classifiers", nargs="+", default=config.CLASSIFIERS, choices=config.CLASSIFIERS)
    ap.add_argument("--reps", nargs="+", type=int, default=config.ROOT_SEEDS)
    ap.add_argument("--data-dir", default="data", help="folder with pima.arff / processed.cleveland.data")
    ap.add_argument("--out", default=None, help="output folder (default: results or results_smoke)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--params", default=None, help="JSON of selected hyperparameters")
    ap.add_argument("--use-thesis-params", action="store_true", help="use Table 3.8 instead of the rerun")
    ap.add_argument("--smoke", action="store_true", help="quick pipeline check, not for reporting")
    ap.add_argument("--generator", default="tvae", choices=["tvae", "mock"],
                    help="'mock' is for testing the code without SDV/GPU only")
    ap.add_argument("--force", action="store_true", help="redo repetitions already marked done")
    args = ap.parse_args()

    out_dir = args.out or ("results_smoke" if args.smoke else "results")
    if args.smoke and args.reps == config.ROOT_SEEDS:
        args.reps = [config.ROOT_SEEDS[0]]
    if args.generator == "mock":
        warnings.warn("MOCK generator selected - output is for code testing only.")
    device = detect_device(args.device)
    env = record_environment(out_dir, device)
    print(f"Output folder: {os.path.abspath(out_dir)}   device: {device}")
    if args.stage in ("check",):
        print(json.dumps(env, indent=2))

    import data as data_mod
    preps = {}
    if args.stage != "analyze":
        for d in args.datasets:
            preps[d] = data_mod.prepare(d, data_dir=args.data_dir, out_dir=out_dir)
    if args.stage == "check":
        import construction
        for d, p in preps.items():
            y = p.y_train
            tab = construction.allocation_table(int((y == 0).sum()), int((y == 1).sum()))
            print(f"[{d}] allocation (Rneg/Rpos | Mneg/Mpos): " +
                  "; ".join(f"{r['level']*10}%: {r['Rneg']}/{r['Rpos']}|{r['Mneg']}/{r['Mpos']}" for r in tab))
        return

    if args.stage in ("tune", "all"):
        from tuning import run_tuning
        run_tuning(preps, args.classifiers, os.path.join(out_dir, "tuning"),
                   device=device, n_jobs=args.n_jobs, smoke=args.smoke)

    if args.stage in ("experiment", "all"):
        from experiment import run_repetition
        params = load_params(args, out_dir)
        epochs = config.SMOKE["epochs"] if args.smoke else None
        t0 = time.time()
        for d in args.datasets:
            for r in args.reps:
                run_repetition(preps[d], r, params, args.classifiers, out_dir, device=device,
                               generator=args.generator, epochs=epochs, n_jobs=args.n_jobs,
                               force=args.force)
        print(f"Experiment stage finished in {(time.time() - t0) / 60:.1f} min")

    if args.stage in ("analyze", "all"):
        from analysis import run_analysis
        run_analysis(out_dir)


if __name__ == "__main__":
    main()
