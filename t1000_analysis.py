"""T=1000 experiment (lab, GPU): does the scored trim still help at the paper's unsupervised T?

Reads raw_gpu/<exp>_gpu.jsonl (last successful record per file/seed), averages members and seeds
within each of the 57 groups (as recompute.py), and writes tables/t1000_datasets.csv,
t1000_mean57.csv, t1000_stats.csv and t1000_summary.json.
All methods ran on the same GPU with the same code, so they are compared with each other only.
Usage: python t1000_analysis.py [--partial]   (--partial: allow missing jobs, for progress checks)
"""
from pathlib import Path
import csv, json, math, sys
import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from anoddae_repro.data import CLASSICAL, benchmark_datasets
from paired_stats import compare, fmt

partial = "--partial" in sys.argv
res, tables = root / "raw_gpu", root / "tables"
METHODS = {"T1000": "sweepT_1000", "T1000_trim": "impr_trim0.1_T1000", "T1000_rand": "impr_randtrim0.1_T1000",
           "T1000_oracle": "impr_oracletrim0.1_T1000", "T50": "sweepT_50", "T50_trim": "impr_trim0.1_T50"}
BLOCKS = {
    "T1000": [("T1000_trim", "T1000"), ("T1000_trim", "T1000_rand"), ("T1000_rand", "T1000"),
              ("T1000_oracle", "T1000"), ("T1000_oracle", "T1000_trim"), ("T1000", "centroid"), ("T1000_trim", "centroid")],
    "T50_vs_T1000": [("T1000", "T50"), ("T50_trim", "T1000"), ("T1000_trim", "T50_trim"), ("T50_trim", "T50"), ("T50_trim", "centroid")],
}
sizes = json.loads((res / "dataset_sizes.json").read_text())
expected = {(f, s) for f in sizes for s in range(6)}
runs = {}
for m, exp in METHODS.items():
    p = res / f"{exp}_gpu.jsonl"
    last = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                assert (r["setting"], r["protocol"], r["max_size"], r.get("device")) == ("unsup", "dte2", 50000, "cuda"), r["file"]
                last[(r["file"], r["seed"])] = r
    missing = expected - set(last)
    if missing and not partial:
        sys.exit(f"{exp}: {len(missing)} of 726 file/seed jobs missing")
    runs[m] = last
if partial:  # progress check: keep only methods that have results
    runs = {m: v for m, v in runs.items() if v}
    BLOCKS = {b: [p for p in ps if all(x == "centroid" or x in runs for x in p)] for b, ps in BLOCKS.items()}
    BLOCKS = {b: ps for b, ps in BLOCKS.items() if ps}
keys = set.intersection(*(set(v) for v in runs.values())) if partial else expected
for k in keys:  # same data and same centroid in every method
    c = {json.dumps(runs[m][k]["metrics"]["centroid"]) for m in runs}
    assert len(c) == 1, f"{k}: centroid differs between methods"
grp = lambda f: f if f in CLASSICAL else f.split("_")[0]
G = {}
for m, rs in runs.items():
    for k in keys:
        r = rs[k]; g = G.setdefault(grp(k[0]), {})
        mm = r["metrics"]
        g.setdefault(f"{m}_AP", []).append(mm["official"]["ap"]); g.setdefault(f"{m}_ROC_AUC", []).append(mm["official"]["roc"])
        g.setdefault(f"{m}_rho", []).append(r["official_cen_rho"])
        if m == "T1000" or (partial and "T1000" not in runs and m == next(iter(runs))):
            g.setdefault("centroid_AP", []).append(mm["centroid"]["ap"]); g.setdefault("centroid_ROC_AUC", []).append(mm["centroid"]["roc"])
            g.setdefault("_steps", []).append(r["per_step_cen_rho"]); g.setdefault("_step_ap", []).append(r["per_step"]["step"])
            t_axis = r["per_step"]["t"]
names = [n for n in benchmark_datasets() if n in G]
units = []
for n in names:
    u = {"dataset": n}
    for k, v in G[n].items():
        if not k.startswith("_"):
            u[k] = float(np.mean(v))
    u["_steps"] = np.mean(np.array(G[n]["_steps"]), axis=0); u["_step_ap"] = np.mean(np.array(G[n]["_step_ap"]), axis=0)
    units.append(u)
tables.mkdir(exist_ok=True)
cols = [c for c in units[0] if not c.startswith("_")]
with open(tables / "t1000_datasets.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for u in units: w.writerow({c: (f"{u[c]:.8f}" if isinstance(u[c], float) else u[c]) for c in cols})
mean57 = {c: float(np.mean([u[c] for u in units])) for c in cols if c != "dataset"}
with open(tables / "t1000_mean57.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["dataset"] + list(mean57)); w.writerow(["mean57"] + [f"{v:.8f}" for v in mean57.values()])
rows = []
for block, pairs in BLOCKS.items():
    for r in compare(units, pairs):
        r["block"] = block; rows.append(r)
with open(tables / "t1000_stats.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["block"] + [k for k in rows[0] if k != "block"]); w.writeheader()
    for r in rows: w.writerow(fmt(r))
steps = np.mean([u["_steps"] for u in units], axis=0); step_ap = np.mean([u["_step_ap"] for u in units], axis=0)
pick = [i for i, t in enumerate(t_axis) if i in (0, 4, 9, 19, 29, 49, 74, 99)]
gap = (mean57["T1000_trim_AP"] - mean57["T1000_AP"]) / (mean57["T1000_oracle_AP"] - mean57["T1000_AP"])
summary = {"groups": len(units), "jobs_per_method": len(keys), "partial": partial, "mean57": mean57,
           "oracle_gap_closed_AP": gap,
           "per_step": [{"t": t_axis[i], "rho_with_centroid": float(steps[i]), "step_AP": float(step_ap[i])} for i in pick]}
(tables / "t1000_summary.json").write_text(json.dumps(summary, indent=1))
print(f"{len(units)} groups, {len(keys)} jobs per method{' (partial)' if partial else ''}")
for c in mean57: print(f"  {c:22s} {mean57[c]:7.2f}" if not c.endswith("rho") else f"  {c:22s} {mean57[c]:7.3f}")
print(f"  oracle gap closed by scored trim (AP): {gap:.1%}")
for r in rows:
    print(f"  [{r['block']}] {r['metric']:7s} {r['comparison']:28s} diff {r['mean_diff']:+6.2f}  {r['wins']}-{r['losses']}  p_holm {r['p_holm']:.3g}  rb {r['rank_biserial']:+.2f}")
for s in summary["per_step"]: print(f"  t={s['t']:4d}  rho(step error, centroid)={s['rho_with_centroid']:.3f}  step AP={s['step_AP']:.2f}")
