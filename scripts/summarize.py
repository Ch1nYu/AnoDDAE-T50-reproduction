"""Build all summary tables (CSV + JSON) from results/*.jsonl.

Outputs results/summary/*.csv and results/summary/summary.json, which the
figure script and the report read.  Safe to run while experiments are still
running: incomplete experiments are reported with their seed/dataset counts.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anoddae_repro.analyze import (PAPER_TABLE1, PAPER_STD, load, per_dataset, summary,  # noqa: E402
                                   dte_table, avg_rank, family_of, RES)

OUT = os.path.join(RES, "summary")
os.makedirs(OUT, exist_ok=True)
S = {}


def mstd(recs, variant, metric, setting, require=57):
    r = summary(recs, variant, metric, setting, require)
    return r


# --------------------------------------------------------------------------- #
# 1) Table 1 reproduction
# --------------------------------------------------------------------------- #
main = {
    "DDAE": ({"semi": "ddae_semi_T50", "unsup": "ddae_unsup_T1000"}, "official"),
    "DDAE (Eq.6 squared score)": ({"semi": "ddae_semi_T50", "unsup": "ddae_unsup_T1000"}, "paper"),
    "DDAE-C": ({"semi": "ddaec_semi_T50", "unsup": "ddaec_unsup_T1000"}, "official"),
    "DAE (sigma=0.1)": ({"semi": "dae_s0.1", "unsup": "dae_s0.1"}, "official"),
    "DAE (sigma=0.5)": ({"semi": "dae_s0.5", "unsup": "dae_s0.5"}, "official"),
    "Centroid distance ||x-mean||^2": ({"semi": "ddae_semi_T50", "unsup": "ddae_unsup_T1000"}, "centroid"),
    "kNN (our run)": ({"semi": "knn", "unsup": "knn"}, "knn"),
}
rows = []
cache = {}
for name, (exps, variant) in main.items():
    row = {"method": name}
    for st in ("unsup", "semi"):
        exp = exps[st]
        if exp not in cache:
            cache[exp] = load(exp)
        recs = [r for r in cache[exp] if r["setting"] == st]
        for metric, lab in (("ap", "PR"), ("roc", "ROC")):
            r = mstd(recs, variant, metric, st)
            if r is None:
                row[f"{st}_{lab}"] = np.nan
                continue
            row[f"{st}_{lab}"] = r["mean"]
            row[f"{st}_{lab}_std"] = r["std"]
            row[f"{st}_seeds"] = r["n_seeds"]
    rows.append(row)
for name, v in PAPER_TABLE1.items():
    sd = PAPER_STD[name]
    rows.append({"method": f"[paper] {name}", "unsup_PR": v[0], "unsup_PR_std": sd[0], "unsup_ROC": v[1],
                 "unsup_ROC_std": sd[1], "semi_PR": v[2], "semi_PR_std": sd[2], "semi_ROC": v[3],
                 "semi_ROC_std": sd[3]})
t1 = pd.DataFrame(rows).set_index("method")
t1.to_csv(os.path.join(OUT, "table1_reproduction.csv"))
S["table1"] = json.loads(t1.round(3).to_json(orient="index"))

# --------------------------------------------------------------------------- #
# 2) Per-dataset tables + ranks among the 26 DTE baselines
# --------------------------------------------------------------------------- #
for st, exp in (("semi", "ddae_semi_T50"), ("unsup", "ddae_unsup_T1000")):
    recs = [r for r in cache.get(exp, load(exp)) if r["setting"] == st]
    if not recs:
        continue
    for metric, lab in (("ap", "PR"), ("roc", "ROC")):
        ours = per_dataset(recs, "official", metric, st).mean(axis=1)
        cen = per_dataset(recs, "centroid", metric, st).mean(axis=1)
        ref = dte_table(st, metric)
        ref.index = [i.lower() if i not in ("MNIST-C", "FashionMNIST", "CIFAR10", "SVHN", "MVTec-AD") else i
                     for i in ref.index]
        df = ref.copy()
        idx = {k.lower() if k not in ("MNIST-C", "FashionMNIST", "CIFAR10", "SVHN", "MVTec-AD") else k: v
               for k, v in ours.items()}
        df["DDAE (ours)"] = pd.Series(idx)
        df["Centroid"] = pd.Series({k.lower() if k not in ("MNIST-C", "FashionMNIST", "CIFAR10", "SVHN", "MVTec-AD")
                                    else k: v for k, v in cen.items()})
        df.to_csv(os.path.join(OUT, f"per_dataset_{st}_{lab}.csv"))
        complete = df.dropna()
        if len(complete) >= 50:
            rk = avg_rank(complete)
            S[f"rank_{st}_{lab}"] = {"n_datasets": len(complete), "ranks": rk.round(2).to_dict(),
                                     "means": complete.mean().round(2).to_dict()}
            # paired tests: DDAE vs strongest baselines
            tests = {}
            for b in ("kNN", "DTE-NP", "DDPM", "DTE-C", "Centroid", "LOF", "IForest", "ICL"):
                if b in complete:
                    diff = complete["DDAE (ours)"] - complete[b]
                    p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
                    tests[b] = {"mean_diff": float(diff.mean()), "wins": int((diff > 0).sum()),
                                "losses": int((diff < 0).sum()), "p_wilcoxon": float(p)}
            S[f"tests_{st}_{lab}"] = tests

# --------------------------------------------------------------------------- #
# 3) Score-rule variants and kNN ensembles (no re-training needed)
# --------------------------------------------------------------------------- #
def rank01(a):
    a = np.asarray(a, dtype=np.float64)
    return pd.Series(a).rank(pct=True).values


def ensemble_table(exp, st, other="knn", other_key="knn"):
    """Rank-average of DDAE 'official' score and kNN score per (file, seed)."""
    from anoddae_repro.runner import _metrics
    base = os.path.join(RES, "scores", exp, st)
    obase = os.path.join(RES, "scores", other, st)
    if not (os.path.isdir(base) and os.path.isdir(obase)):
        return None
    recs = []
    for fn in sorted(os.listdir(base)):
        of = os.path.join(obase, fn)
        if not os.path.exists(of):
            continue
        a, b = np.load(os.path.join(base, fn)), np.load(of)
        if len(a["y"]) != len(b["y"]) or (a["y"] != b["y"]).any():
            continue
        file, seed = fn[:-4].rsplit("_s", 1)
        y = a["y"]
        m = {"ens_rank": _metrics(y, rank01(a["official"]) + rank01(b[other_key])),
             "ens_feat_rank": _metrics(y, rank01(a["featnorm"]) + rank01(b[other_key])) if "featnorm" in a else None}
        recs.append({"file": file, "seed": int(seed), "setting": st, "metrics": {k: v for k, v in m.items() if v}})
    return recs


variants = {}
for st, exp in (("semi", "ddae_semi_T50"), ("unsup", "ddae_unsup_T1000")):
    recs = [r for r in cache.get(exp, []) if r["setting"] == st]
    if not recs:
        continue
    out = {}
    for v in ("official", "paper", "stepnorm", "featnorm", "centroid"):
        for metric in ("ap", "roc"):
            r = mstd(recs, v, metric, st)
            if r:
                out[f"{v}_{metric}"] = r
    ens = ensemble_table(exp, st)
    if ens:
        for v in ("ens_rank", "ens_feat_rank"):
            for metric in ("ap", "roc"):
                r = mstd(ens, v, metric, st)
                if r:
                    out[f"{v}_{metric}"] = r
    knn_recs = [r for r in cache.get("knn", load("knn")) if r["setting"] == st]
    for metric in ("ap", "roc"):
        r = mstd(knn_recs, "knn", metric, st)
        if r:
            out[f"knn_{metric}"] = r
    variants[st] = out
S["variants"] = variants

# --------------------------------------------------------------------------- #
# 4) Noise analysis (lite protocol): T sweep, schedulers, strided-score check
# --------------------------------------------------------------------------- #
def lite_mean(exp, st, variant="official", metric="ap"):
    recs = [r for r in load(exp, "_lite") if r["setting"] == st]
    if not recs:
        return None
    df = per_dataset(recs, variant, metric, st)
    if df.empty:
        return None
    col = df.iloc[:, 0]
    return {"mean": float(col.mean()), "n": int(col.notna().sum())}


sweep = {}
for T in (5, 10, 20, 50, 100, 300, 500, 1000):
    for st in ("semi", "unsup"):
        for v in ("official", "paper", "stepnorm", "featnorm", "centroid"):
            for metric in ("ap", "roc"):
                r = lite_mean(f"sweepT_{T}", st, v, metric)
                if r:
                    sweep.setdefault(st, {}).setdefault(v, {}).setdefault(metric, {})[T] = r
S["sweepT_lite"] = sweep
sched = {}
for s_ in ("linear", "quadratic", "sigmoid", "exponential", "cosine-official", "cosine-fixed"):
    for st in ("semi", "unsup"):
        for metric in ("ap", "roc"):
            for v in ("official", "centroid"):
                r = lite_mean(f"sched_{s_}_T100", st, v, metric)
                if r:
                    sched.setdefault(st, {}).setdefault(s_, {})[f"{v}_{metric}"] = r
S["schedulers_lite"] = sched
fc = [r for r in load("ddae_unsup_T1000_fullcheck", "_lite")]
if fc:
    a = np.array([[r["metrics"]["official"]["ap"], r["metrics"]["official_full"]["ap"],
                   r["metrics"]["official"]["roc"], r["metrics"]["official_full"]["roc"],
                   r["score_time"], r["metrics"]["full_score_time"]["roc"]] for r in fc])
    S["strided_check"] = {"n_files": len(a), "ap_strided": float(a[:, 0].mean()), "ap_full": float(a[:, 1].mean()),
                          "roc_strided": float(a[:, 2].mean()), "roc_full": float(a[:, 3].mean()),
                          "max_abs_ap_diff": float(np.abs(a[:, 0] - a[:, 1]).max()),
                          "mean_abs_ap_diff": float(np.abs(a[:, 0] - a[:, 1]).mean()),
                          "max_abs_roc_diff": float(np.abs(a[:, 2] - a[:, 3]).max()),
                          "speedup": float(a[:, 5].sum() / max(a[:, 4].sum(), 1e-9))}

# --------------------------------------------------------------------------- #
# 5) Per-step curves (paper Fig. 6 analogue)
# --------------------------------------------------------------------------- #
for st, exp in (("semi", "ddae_semi_T50"), ("unsup", "ddae_unsup_T1000")):
    recs = [r for r in cache.get(exp, []) if r["setting"] == st and "per_step" in r and r["seed"] == 0]
    if not recs:
        continue
    fam = {}
    for r in recs:
        fam.setdefault(family_of(r["file"]), []).append(r["per_step"])
    t = recs[0]["per_step"]["t"]
    curves = {}
    for key in ("step", "cum", "step_roc", "cum_roc"):
        per_ds = [np.mean([np.array(p[key]) for p in ps], axis=0) for ps in fam.values()]
        curves[key] = np.mean(per_ds, axis=0).round(3).tolist()
    S[f"per_step_{st}"] = {"t": t, "n_datasets": len(fam), **curves}

# --------------------------------------------------------------------------- #
# 6) Protocol effect (DTE pipeline vs released-code pipeline) and trimming
# --------------------------------------------------------------------------- #
def seed_mean(recs, variant, metric, st, seeds=None):
    rr = [r for r in recs if r["setting"] == st and (seeds is None or r["seed"] in seeds)]
    df = per_dataset(rr, variant, metric, st)
    if df.empty:
        return None
    full = df.dropna()
    return {"mean": float(full.mean(axis=1).mean()), "n_datasets": int(len(full)), "n_seeds": int(df.shape[1])}


prot = {}
knn_dte = load("knn")
knn_repo = [dict(r, exp="knn_repo") for r in load("knn_repo_protocol")]
for st in ("semi", "unsup"):
    for metric in ("ap", "roc"):
        a = seed_mean(knn_dte, "knn", metric, st, seeds={0, 1, 2})
        b = seed_mean(knn_repo, "knn", metric, st, seeds={0, 1, 2})
        if a and b:
            prot[f"knn_{st}_{metric}"] = {"dte": a, "repo": b}
        for exp in ("ddae_semi_T50", "ddae_unsup_T1000"):
            if (st == "semi") != (exp == "ddae_semi_T50"):
                continue
            a = seed_mean(load(exp), "official", metric, st, seeds={0})
            b = seed_mean(load(exp, "_repo"), "official", metric, st, seeds={0})
            if a and b:
                prot[f"ddae_{st}_{metric}"] = {"dte": a, "repo": b}
S["protocol_effect"] = prot

trim = {}
for q in ("0.1", "0.05"):
    for T in ("1000", "50"):
        base = load(f"sweepT_{T}", "_lite")
        imp = load(f"impr_trim{q}_T{T}", "_lite")
        if not imp:
            continue
        for v in ("official", "featnorm"):
            for metric in ("ap", "roc"):
                a = seed_mean(base, v, metric, "unsup")
                b = seed_mean(imp, v, metric, "unsup")
                if a and b:
                    trim[f"q{q}_T{T}_{v}_{metric}"] = {"base": a, "trim": b}
S["trimming_lite"] = trim


# seed ensemble (semi, 5 models): only datasets whose test set is identical across seeds
def seed_ensemble(exp="ddae_semi_T50", st="semi", seeds=range(5)):
    from anoddae_repro.data import benchmark_datasets, family_members
    from anoddae_repro.runner import _metrics
    base = os.path.join(RES, "scores", exp, st)
    rows = []
    for ds in benchmark_datasets():
        for f in family_members(ds):
            ps = [os.path.join(base, f"{f}_s{s_}.npz") for s_ in seeds]
            if not all(os.path.exists(p_) for p_ in ps):
                return None
            zs = [np.load(p_) for p_ in ps]
            if not all(len(z["y"]) == len(zs[0]["y"]) and (z["y"] == zs[0]["y"]).all() for z in zs):
                rows.append({"dataset": family_of(f), "single": np.nan, "ens": np.nan})
                continue
            single = np.mean([_metrics(z["y"], z["official"])["ap"] for z in zs])
            ens = _metrics(zs[0]["y"], np.mean([pd.Series(z["official"]).rank(pct=True).values for z in zs], axis=0))["ap"]
            rows.append({"dataset": family_of(f), "single": single, "ens": ens})
    g = pd.DataFrame(rows).groupby("dataset").mean().dropna()
    return {"n_datasets": int(len(g)), "single": float(g.single.mean()), "ens": float(g.ens.mean()),
            "wins": int((g.ens > g.single).sum()), "losses": int((g.ens < g.single).sum())}


se = seed_ensemble()
if se:
    S["seed_ensemble_semi"] = se
mt = os.path.join(OUT, "multiT_ensemble.json")
if os.path.exists(mt):
    S["multiT"] = json.load(open(mt))

json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
pd.set_option("display.width", 220)
print(t1.round(2).to_string())
for k in ("rank_semi_PR", "rank_unsup_PR", "rank_semi_ROC", "rank_unsup_ROC"):
    if k in S:
        r = S[k]["ranks"]
        print(k, "n=", S[k]["n_datasets"], "DDAE rank", r.get("DDAE (ours)"), "| top5:", list(r.items())[:5])
for st, v in variants.items():
    print(st, {k: round(x["mean"], 2) for k, x in v.items()})
if "strided_check" in S:
    print("strided check", S["strided_check"])
