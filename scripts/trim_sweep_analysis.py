"""Summarise the trimming-ratio sweep (results/trimq*_lite.jsonl)."""
import json, os, glob, collections
import numpy as np
from scipy.stats import wilcoxon
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("ANODDAE_RESULTS", os.path.join(ROOT, "results"))
sys_path = os.path.dirname(RES)
import sys; sys.path.insert(0, sys_path)
from anoddae_repro.data import benchmark_datasets, family_members

fam = {}
for ds in benchmark_datasets():
    for f in family_members(ds):
        fam[f] = ds

def load(exp, setting):
    rows = collections.defaultdict(dict)   # dataset -> seed -> (ap, roc, contamination)
    for l in open(os.path.join(RES, f"{exp}_lite.jsonl")):
        r = json.loads(l)
        if r["setting"] != setting or "error" in r: continue
        m = r["metrics"]["official"]
        rows[fam[r["file"]]][r["seed"]] = (m["ap"], m["roc"], r.get("train_contamination", r["anomaly_rate"]))
    return rows

def per_ds(rows):  # mean over seeds
    return {d: np.mean([v[:2] for v in s.values()], axis=0) for d, s in rows.items()}

out = {}
for setting, exps in [("unsup", ["trimq0.0_T50", "trimq0.02_T50", "trimq0.05_T50", "trimq0.1_T50", "trimq0.15_T50",
                                 "trimq0.2_T50", "trimq0.3_T50", "trimq_oracle_T50", "trimq0.1_every5_T50", "trimq0.1_every20_T50"]),
                      ("semi", ["trimq0.0_T50", "trimq0.05_T50", "trimq0.1_T50", "trimq0.2_T50"])]:
    base_rows = load("trimq0.0_T50", setting)
    base = per_ds(base_rows)
    cont = {d: np.mean([v[2] for v in s.values()]) for d, s in base_rows.items()}
    res = {}
    for e in exps:
        pdict = per_ds(load(e, setting))
        ds = sorted(set(pdict) & set(base))
        ap = np.array([pdict[d][0] for d in ds]); ap0 = np.array([base[d][0] for d in ds])
        roc = np.array([pdict[d][1] for d in ds])
        # seed std of the 57-dataset mean
        seeds = collections.defaultdict(list)
        for d, s in load(e, setting).items():
            for sd, v in s.items(): seeds[sd].append(v[0])
        seed_means = [np.mean(v) for v in seeds.values() if len(v) == len(ds)]
        diff = ap - ap0
        p = wilcoxon(diff).pvalue if e != "trimq0.0_T50" else None
        groups = {}
        for name, lo, hi in [("low (<5%)", 0, .05), ("mid (5-15%)", .05, .15), ("high (>15%)", .15, 1.1)]:
            idx = [i for i, d in enumerate(ds) if lo <= cont[d] < hi]
            groups[name] = dict(n=len(idx), ap=float(ap[idx].mean()), gain=float(diff[idx].mean()))
        res[e] = dict(n=len(ds), ap=float(ap.mean()), roc=float(roc.mean()),
                      seed_std=float(np.std(seed_means)) if len(seed_means) > 1 else None,
                      gain=float(diff.mean()), wins=int((diff > 0).sum()), losses=int((diff < 0).sum()),
                      p=p, groups=groups)
        print(f"{setting:5} {e:22} AP {ap.mean():6.2f} ROC {roc.mean():6.2f} gain {diff.mean():+6.2f} "
              f"W/L {(diff>0).sum()}/{(diff<0).sum()} p={p if p is None else round(p,4)}  " +
              "  ".join(f"{k} {v['ap']:.1f}({v['gain']:+.1f},n={v['n']})" for k, v in groups.items()))
    out[setting] = res
os.makedirs(os.path.join(RES, "summary"), exist_ok=True)
json.dump(out, open(os.path.join(RES, "summary", "trim_sweep.json"), "w"), indent=1)
