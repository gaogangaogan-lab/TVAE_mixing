"""One repetition of the full design for one dataset (Section 3.9.1):
fit TVAE -> pool -> orderings -> main + ablation sets -> fidelity -> classifiers.

Every repetition writes to its own folder and a 'done.json' marker, so an
interrupted run can be resumed without redoing finished repetitions.
"""
import json
import os
import time

import numpy as np
import pandas as pd

import config
import construction as cons
import fidelity
from classifiers import fit_predict, metrics_from_proba
from generator import generate_pool, make_generator, save_json
from seeds import SeedLog


def run_repetition(prep, root, params, classifiers, out_dir, device="cpu",
                   generator="tvae", epochs=None, n_jobs=-1, force=False):
    spec = prep.spec
    dname = spec.name
    rep_dir = os.path.join(out_dir, "raw", dname, f"rep_{root}")
    done_path = os.path.join(rep_dir, "done.json")
    if os.path.exists(done_path) and not force:
        print(f"[{dname} rep {root}] already done - skipped")
        return
    os.makedirs(rep_dir, exist_ok=True)
    seeds = SeedLog()
    t_start = time.time()

    train = prep.train.reset_index(drop=True)
    test = prep.test.reset_index(drop=True)
    real_y = train[spec.target].to_numpy()
    n_neg, n_pos = int((real_y == 0).sum()), int((real_y == 1).sum())
    n = n_neg + n_pos

    # ---- 1. TVAE fit -------------------------------------------------------
    t0 = time.time()
    gen = make_generator(generator, spec, device, epochs=epochs)
    gen.fit(train, seeds.get(root, dname, "tvae_fit"))
    fit_secs = time.time() - t0
    loss = gen.loss_values()
    if isinstance(loss, pd.DataFrame):
        loss.to_csv(os.path.join(rep_dir, "tvae_loss.csv"), index=False)
    gen_info = {"generator": generator, "fit_seconds": round(fit_secs, 1),
                "epochs": epochs or config.TVAE_CONFIG["epochs"], **gen.batch_info(n)}
    if hasattr(gen, "metadata") and gen.metadata is not None:
        from generator import metadata_dict
        gen_info["metadata"] = metadata_dict(gen.metadata)
    print(f"[{dname} rep {root}] TVAE fitted in {fit_secs:.0f}s", flush=True)

    # ---- 2. Pool -----------------------------------------------------------
    pool, pool_info = generate_pool(
        gen, spec, n, n_neg, n_pos,
        seed_fn=lambda b: seeds.get(root, dname, "pool_sample", batch=b))
    pool.to_csv(os.path.join(rep_dir, "pool.csv"), index=False)
    save_json({**gen_info, "pool": pool_info}, os.path.join(rep_dir, "generator_info.json"))
    if not pool_info["success"]:
        save_json({"status": "generation_failure", **pool_info}, done_path)
        pd.DataFrame(seeds.rows).to_csv(os.path.join(rep_dir, "seeds.csv"), index=False)
        print(f"[{dname} rep {root}] GENERATION FAILURE - recorded, not replaced")
        return
    pool_y = pool[spec.target].to_numpy()

    # ---- 3. Training sets --------------------------------------------------
    sets, orders = cons.build_all_sets(
        real_y, pool_y,
        real_seed=seeds.get(root, dname, "real_order"),
        syn_seed=seeds.get(root, dname, "syn_order"),
        c_seed_fn=lambda kind, k, cls: seeds.get(
            root, dname, "ablation_c_real" if kind == "real" else "ablation_c_syn",
            level=k, cls=cls))
    problems = cons.check_properties(sets, real_y, pool_y)
    if problems:
        raise RuntimeError("Construction property check failed:\n" + "\n".join(problems))
    np.savez_compressed(os.path.join(rep_dir, "orders.npz"), **orders)
    index_rows = []
    for (st, k), s in sets.items():
        index_rows.append({"setting": st, "level": k,
                           "shared_with": None if s.shared_with is None else f"{s.shared_with[0]}:{s.shared_with[1]}",
                           "real_ids": " ".join(map(str, s.real_ids.tolist())),
                           "syn_pool_ids": " ".join(map(str, s.syn_ids.tolist()))})
    pd.DataFrame(index_rows).to_csv(os.path.join(rep_dir, "set_indices.csv"), index=False)

    # ---- 4. Fidelity of the 100%-mixing main set ---------------------------
    s10 = sets[("main", config.N_LEVELS - 1)]
    syn_full = pool.set_index("pool_id").loc[s10.syn_ids].reset_index()
    fid_rows, fid_sum = fidelity.evaluate(train, syn_full, spec)
    fid_df = pd.DataFrame(fid_rows)
    fid_df.insert(0, "rep", root)
    fid_df.insert(0, "dataset", dname)
    fid_df.to_csv(os.path.join(rep_dir, "fidelity_variables.csv"), index=False)
    save_json({"dataset": dname, "rep": root, **fid_sum,
               "pool_neg": pool_info["pool_neg"], "pool_pos": pool_info["pool_pos"],
               "pool_pos_share": pool_info["pool_pos_share"]},
              os.path.join(rep_dir, "fidelity_summary.json"))

    # ---- 5. Classifiers ----------------------------------------------------
    X_real = train[spec.predictors].to_numpy(dtype=float)
    X_pool = pool[spec.predictors].to_numpy(dtype=float)
    X_test = test[spec.predictors].to_numpy(dtype=float)
    y_test = test[spec.target].to_numpy()

    # Order of evaluation: main first, so shared sets can reuse its predictions.
    keys = [(st, k) for st in config.SETTINGS for k in config.LEVELS]
    metric_rows, preds = [], {}
    for c in classifiers:
        t0 = time.time()
        for (st, k) in keys:
            s = sets[(st, k)]
            info = cons.describe_set(s, real_y, pool_y)
            clf_seed = seeds.get(root, dname, "classifier", classifier=c, level=k)
            if s.shared_with is not None:
                proba = preds[(c,) + s.shared_with]
                shared = f"{s.shared_with[0]}:{s.shared_with[1]}"
            else:
                X = np.vstack([X_real[s.real_ids], X_pool[s.syn_ids]])
                y = np.concatenate([real_y[s.real_ids], pool_y[s.syn_ids]])
                perm = np.random.default_rng(
                    seeds.get(root, dname, "row_shuffle", level=k)).permutation(len(y))
                proba = fit_predict(c, params[dname][c], clf_seed, X[perm], y[perm], X_test,
                                    device=device, n_jobs=n_jobs)
                shared = ""
            preds[(c, st, k)] = proba
            m = metrics_from_proba(y_test, proba)
            metric_rows.append({"dataset": dname, "classifier": c, "setting": st,
                                "level": k, "p_nominal": k / 10, "rep": root,
                                "classifier_seed": clf_seed, "shared_with": shared,
                                **info, **m})
        print(f"[{dname} rep {root}] {config.CLASSIFIER_NAMES[c]}: {len(keys)} conditions "
              f"in {time.time() - t0:.0f}s", flush=True)

    pd.DataFrame(metric_rows).to_csv(os.path.join(rep_dir, "metrics.csv"), index=False)
    np.savez_compressed(os.path.join(rep_dir, "test_probabilities.npz"),
                        **{f"{c}__{st}__{k}": v for (c, st, k), v in preds.items()})
    pd.DataFrame(seeds.rows).drop_duplicates().to_csv(os.path.join(rep_dir, "seeds.csv"), index=False)
    save_json({"status": "ok", "seconds": round(time.time() - t_start, 1),
               "n_conditions": len(metric_rows)}, done_path)
    print(f"[{dname} rep {root}] finished in {time.time() - t_start:.0f}s", flush=True)
