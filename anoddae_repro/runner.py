"""Resumable, multi-process experiment runner.

One job = (experiment, dataset file, setting, seed).  Results are appended as
JSON lines to results/<exp><tag>.jsonl; per-sample test scores are saved to
results/scores/<exp>/<protocol>_cap<max_size>/tag<tag>/<setting>/<file>_s<seed>.npz
(see score_path) for offline ensembling, so runs that differ in protocol, size
cap or tag never share a score file.  The historical runs in raw/ predate this
layout and keep their scores under raw/scores/<exp>/<setting>/.

Usage (see scripts/run_all.sh):
    python -m anoddae_repro.runner --exp ddae_semi_T50 --setting semi --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

# ----------------------------------------------------------------------------
# Experiment registry: name -> model kwargs (+ runner options)
# ----------------------------------------------------------------------------
BASE = dict(hidden=(512, 512), act="lrelu", beta_start=1e-4, beta_end=0.02,
            t_dim=4, epochs=100, lr=1e-3)
EXPS = {
    # ---- main reproduction (paper Table 1) --------------------------------
    "ddae_semi_T50":    dict(BASE, T=50, schedule="linear"),
    "ddae_unsup_T1000": dict(BASE, T=1000, schedule="linear", score_steps=100),
    "ddaec_semi_T50":   dict(BASE, T=50, schedule="linear", contrastive_alpha=0.5),
    "ddaec_unsup_T1000": dict(BASE, T=1000, schedule="linear", score_steps=100,
                              contrastive_alpha=0.5),
    # full (un-strided) scoring check for the T=1000 approximation
    "ddae_unsup_T1000_full": dict(BASE, T=1000, schedule="linear"),
    # ---- DAE baselines (fixed-magnitude noise, no timestep) ----------------
    "dae_s0.1": dict(BASE, T=2, dae_sigma=0.1),
    "dae_s0.5": dict(BASE, T=2, dae_sigma=0.5),
}
# diffusion-step sweep (paper Fig. 5) and scheduler comparison (Fig. 7)
for _T in (5, 10, 20, 50, 100, 300, 500, 1000):
    EXPS[f"sweepT_{_T}"] = dict(BASE, T=_T, schedule="linear",
                                score_steps=100 if _T > 100 else None)
for _s in ("linear", "quadratic", "sigmoid", "exponential"):
    EXPS[f"sched_{_s}_T100"] = dict(BASE, T=100, schedule=_s)
EXPS["sched_cosine-official_T100"] = dict(BASE, T=100, schedule="cosine", cosine_impl="official")
EXPS["sched_cosine-fixed_T100"] = dict(BASE, T=100, schedule="cosine", cosine_impl="fixed")

EXPS["ddae_unsup_T1000_fullcheck"] = dict(BASE, T=1000, schedule="linear", score_steps=100)
# DDAE-C sensitivity to the (unreported) loss weight alpha
for _a in (0.1, 0.01):
    EXPS[f"ddaec{_a}_semi_T50"] = dict(BASE, T=50, schedule="linear", contrastive_alpha=_a)
# ---- improvements -----------------------------------------------------------
# contamination-robust training (drop the top-q training rows every 10 epochs)
for _q in (0.05, 0.1):
    EXPS[f"impr_trim{_q}_T1000"] = dict(BASE, T=1000, schedule="linear", score_steps=100, trim_frac=_q)
    EXPS[f"impr_trim{_q}_T50"] = dict(BASE, T=50, schedule="linear", trim_frac=_q)
# control for the trim: drop the same share of rows at random (no scoring)
EXPS["impr_randtrim0.1_T50"] = dict(BASE, T=50, schedule="linear", trim_frac=0.1, trim_mode="random")
# oracle control: drop the true anomalies (by label) first, same row count as the scored trim
EXPS["impr_oracletrim0.1_T50"] = dict(BASE, T=50, schedule="linear", trim_frac=0.1, trim_mode="oracle")
# T=1000 controls (random and label-based trim at the paper's unsupervised T)
EXPS["impr_randtrim0.1_T1000"] = dict(BASE, T=1000, schedule="linear", score_steps=100, trim_frac=0.1, trim_mode="random")
EXPS["impr_oracletrim0.1_T1000"] = dict(BASE, T=1000, schedule="linear", score_steps=100, trim_frac=0.1, trim_mode="oracle")
# trim-share sensitivity
EXPS["impr_trim0.2_T50"] = dict(BASE, T=50, schedule="linear", trim_frac=0.2)

RUN_OPTS = {  # runner options that are not model kwargs
    "per_step": {"ddae_unsup_T1000", "ddae_semi_T50", "sweepT_1000"},
    # also score with all T-1 steps from the same trained model (approximation check)
    "fullcheck": {"ddae_unsup_T1000_fullcheck"},
}


def _metrics(y, s):
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
    s = np.nan_to_num(np.asarray(s, dtype=np.float64), nan=0.0, posinf=1e30, neginf=-1e30)
    p, r, _ = precision_recall_curve(y, s)
    return {"roc": 100 * roc_auc_score(y, s), "ap": 100 * average_precision_score(y, s),
            "prauc": 100 * auc(r, p)}


def score_path(res, exp, protocol, max_size, tag, setting, file, seed):
    """Per-sample score file of one job; unique per protocol, size cap and tag.

    Cap and tag are separate path components ("tag" + tag, so the empty tag is "tag"),
    which keeps e.g. cap 180 + tag "0_x" apart from cap 1800 + tag "_x".
    """
    return os.path.join(res, "scores", exp, f"{protocol}_cap{max_size}", f"tag{tag}", setting, f"{file}_s{seed}.npz")


def score_file_ok(path, n_test):
    """True if a score file opens and has n_test 0/1 labels and n_test finite official scores."""
    try:
        with np.load(path) as z:
            y, s = z["y"], z["official"]
            return (y.shape == (n_test,) and s.shape == (n_test,) and bool(np.isin(y, (0, 1)).all())
                    and bool(np.isfinite(s).all()))
    except Exception:
        return False


def check_tag(exps, tag):
    """Refuse a tag that makes <exp><tag>.jsonl the file name of another experiment's run.

    E.g. sweepT_5 + "0_win" and sweepT_50 + "_win" both give sweepT_50_win.jsonl.  Any such
    clash puts a longer registered name at the start of <exp><tag>, so that is what is checked
    (case-insensitively, as Windows file names are).
    """
    for exp in exps:
        name = (exp + tag).lower()
        for other in EXPS:
            if len(other) > len(exp) and name.startswith(other.lower()):
                sys.exit(f"--tag {tag!r}: {exp}{tag}.jsonl could also be a results file of {other}; pick another tag")


def read_records(path):
    """Records of a results file.  A half-written last line (the run was killed while
    appending) is dropped after saving the file as <path>.bak; a bad line elsewhere is an error."""
    with open(path) as fh:
        text = fh.read()
    lines = text.splitlines()
    records = []
    for i, line in enumerate(lines, 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) and not text.endswith("\n"):
                shutil.copyfile(path, path + ".bak")
                with open(path, "w") as fh:
                    fh.writelines(l + "\n" for l in lines[:-1])
                print(f"{path}: dropped an incomplete last line (backup in {path}.bak)", flush=True)
                break
            sys.exit(f"{path}: line {i} is not valid JSON")
    return records


def run_job(job):
    exp, file, setting, seed, protocol, max_size, *rest = job
    tag = rest[0] if rest else ""
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, ROOT)
    from anoddae_repro.data import make_split, batch_size_rule
    from anoddae_repro.models import DDAE
    t_start = time.time()
    try:
        cfg = dict(EXPS[exp])
        split = make_split(file, setting, seed, protocol=protocol, max_size=max_size)
        np.random.seed(seed)
        torch.manual_seed(seed)
        bs = batch_size_rule(split.n_total)
        model = DDAE(split.x_train.shape[1], batch_size=bs, seed=seed, **cfg)
        model.fit(split.x_train, split.y_train)
        rng = np.random.default_rng(seed)
        calib_idx = rng.choice(len(split.x_train), size=min(4096, len(split.x_train)), replace=False)
        t1 = time.time()
        per_step = exp in RUN_OPTS["per_step"]
        sc = model.score(split.x_test, x_calib=split.x_train[calib_idx], per_step=per_step)
        score_time = time.time() - t1
        y = split.y_test
        out = {k: _metrics(y, v) for k, v in sc.items() if not k.startswith("_")}
        if exp in RUN_OPTS["fullcheck"]:
            model.score_steps = None
            t2 = time.time()
            scf = model.score(split.x_test)
            out["official_full"] = _metrics(y, scf["official"])
            out["paper_full"] = _metrics(y, scf["paper"])
            out["full_score_time"] = {"roc": time.time() - t2, "ap": 0.0, "prauc": 0.0}
        # trivial reference detector: squared distance to the training mean
        cen = ((split.x_test - split.x_train.mean(0)) ** 2).sum(1)
        out["centroid"] = _metrics(y, cen)
        _rk = lambda a: np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
        official_cen_rho = float(np.corrcoef(_rk(sc["official"]), _rk(cen))[0, 1])  # Spearman
        rec = dict(exp=exp, file=file, setting=setting, seed=seed, protocol=protocol,
                   max_size=max_size, n_train=len(split.x_train), n_test=len(split.x_test), d=split.x_train.shape[1],
                   anomaly_rate=float(y.mean()), train_anomaly_rate=float(np.mean(split.y_train)), batch_size=bs, final_loss=model.history[-1],
                   train_time=model.train_time, score_time=score_time, metrics=out)
        rec["device"] = os.environ.get("ANODDAE_DEVICE", "cpu")
        rec["official_cen_rho"] = official_cen_rho
        if per_step:
            ts, sq = sc["_per_step"]
            cum = np.cumsum(sq, axis=0)
            # Spearman correlation of each step's error with the centroid score
            rank = lambda a: np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
            rc = rank(cen)
            rec["per_step_cen_rho"] = [float(np.corrcoef(rank(sq[k]), rc)[0, 1]) for k in range(len(ts))]
            rec["per_step"] = dict(t=ts.tolist(),
                                   step=[_metrics(y, sq[k])["ap"] for k in range(len(ts))],
                                   step_roc=[_metrics(y, sq[k])["roc"] for k in range(len(ts))],
                                   cum=[_metrics(y, cum[k])["ap"] for k in range(len(ts))],
                                   cum_roc=[_metrics(y, cum[k])["roc"] for k in range(len(ts))])
        spath = score_path(RES, exp, protocol, max_size, tag, setting, file, seed)
        os.makedirs(os.path.dirname(spath), exist_ok=True)
        tmp = spath[:-4] + f".tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, y=y.astype(np.int8),
                            **{k: v.astype(np.float32) for k, v in sc.items() if not k.startswith("_")})
        os.replace(tmp, spath)  # a crash never leaves a half-written score file
        if getattr(model, "trim_log", None):
            # which training rows each trim round dropped, with the training labels to judge them
            tpath = os.path.join(RES, "trimlog", exp, f"{protocol}_cap{max_size}", f"tag{tag}", setting, f"{file}_s{seed}.npz")
            os.makedirs(os.path.dirname(tpath), exist_ok=True)
            ttmp = tpath[:-4] + f".tmp{os.getpid()}.npz"
            np.savez_compressed(ttmp, y_train=np.asarray(split.y_train).astype(np.int8),
                                epochs=np.array([e for e, _ in model.trim_log], dtype=np.int32),
                                dropped=np.stack([d for _, d in model.trim_log]))
            os.replace(ttmp, tpath)
        rec["wall"] = time.time() - t_start
        return rec
    except Exception:
        return dict(exp=exp, file=file, setting=setting, seed=seed, error=traceback.format_exc())
    finally:
        # give cached GPU memory back after every job; workers otherwise fill the card together
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _cost(file, setting, exp, sizes, max_size):
    n, d = sizes[file]
    n = min(max(n, 1000), max_size)
    cfg = EXPS[exp]
    per = d * 1024 + 262144
    ntr = n * (0.47 if setting == "semi" else 1.0)
    K = cfg.get("score_steps") or (cfg["T"] - 1)
    c = 3 * cfg["epochs"] * ntr * per * (2.2 if cfg.get("contrastive_alpha") else 1.0)
    return c + n * K * per


def dataset_sizes():
    from anoddae_repro.data import benchmark_datasets, family_members, DATA_ROOT
    cache = os.path.join(RES, "dataset_sizes.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    sizes = {}
    for ds in benchmark_datasets():
        for f in family_members(ds):
            for sub in ("Classical", "CV_by_ResNet18", "NLP_by_BERT"):
                p = os.path.join(DATA_ROOT, sub, f + ".npz")
                if os.path.exists(p):
                    z = np.load(p, allow_pickle=True)
                    sizes[f] = [int(z["y"].shape[0]), int(z["X"].shape[1])]
    os.makedirs(RES, exist_ok=True)
    json.dump(sizes, open(cache, "w"))
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--setting", nargs="+", default=["semi"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--protocol", default="dte")
    ap.add_argument("--max_size", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--datasets", nargs="*", default=None, help="restrict to these benchmark names")
    ap.add_argument("--lite", action="store_true",
                    help="cheap protocol for sweeps: first file per CV/NLP family, max_size 10000")
    ap.add_argument("--tag", default="", help="suffix for the results file (tags that differ only in "
                    "letter case share files on Windows)")
    args = ap.parse_args()
    unknown = [e for e in args.exp if e not in EXPS]
    if unknown:
        sys.exit(f"unknown --exp {unknown}")
    check_tag(args.exp, args.tag)
    sys.path.insert(0, ROOT)
    from anoddae_repro.data import benchmark_datasets, family_members
    sizes = dataset_sizes()
    max_size = 10000 if args.lite else args.max_size
    names = args.datasets or benchmark_datasets()
    files = []
    for ds in names:
        fm = family_members(ds)
        files += fm[:1] if args.lite else fm
    jobs = []
    for exp in args.exp:
        out = os.path.join(RES, f"{exp}{args.tag}.jsonl")
        done = set()
        if os.path.exists(out):
            for r in read_records(out):
                # a finished job only counts for the same protocol and size cap, and only
                # if its score file is intact; records without max_size (written before it
                # was stored) never count, so those jobs run again
                same_data = r.get("protocol") == args.protocol and r.get("max_size") == max_size
                if "error" not in r and r.get("exp") == exp and same_data and score_file_ok(score_path(
                        RES, exp, args.protocol, max_size, args.tag, r["setting"], r["file"], r["seed"]), r["n_test"]):
                    done.add((r["file"], r["setting"], r["seed"]))
        for st in args.setting:
            for sd in args.seeds:
                for f in files:
                    if (f, st, sd) not in done:
                        jobs.append((exp, f, st, sd, args.protocol, max_size, args.tag))
    jobs.sort(key=lambda j: -_cost(j[1], j[2], j[0], sizes, j[5]))
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    # fork is unavailable on native Windows; spawn re-imports this module instead
    ctx = get_context("spawn" if sys.platform == "win32" else "fork")
    failed = 0
    with ctx.Pool(args.workers, maxtasksperchild=20) as pool:
        for k, rec in enumerate(pool.imap_unordered(run_job, jobs), 1):
            out = os.path.join(RES, f"{rec['exp']}{args.tag}.jsonl")
            with open(out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if "error" in rec:
                failed += 1
                print("ERROR", rec["file"], rec["error"][-500:], flush=True)
            else:
                m = rec["metrics"]
                print(f"[{k}/{len(jobs)} {time.time()-t0:7.0f}s] {rec['exp']} {rec['setting']} s{rec['seed']} "
                      f"{rec['file']:<22} AP off {m['official']['ap']:6.2f} paper {m['paper']['ap']:6.2f} "
                      f"feat {m.get('featnorm', {}).get('ap', float('nan')):6.2f} cen {m['centroid']['ap']:6.2f} "
                      f"| ROC {m['official']['roc']:6.2f} | {rec['wall']:.0f}s", flush=True)
    print(f"done: {len(jobs) - failed} succeeded, {failed} failed", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
