"""Random-trim control: is the gain from *which* rows are dropped, or from dropping rows at all?

The three methods (no trim, scored trim, random trim) are rerun together on one Windows PC
(see raw_win/README.md), so they share a platform; they are
compared with each other only, never with the historical Linux run in raw/.
Reads raw_win/<exp>_win.jsonl (per-run metrics; for a job that was retried, the last
successful record counts),
averages members and seeds within each of the 57 groups like recompute.py, and writes
tables/random_trim_datasets.csv, random_trim_mean57.csv and random_trim_stats.csv.

Statistics: two-sided Wilcoxon signed-rank over groups, five comparisons x two metrics,
Holm-corrected within each block; block units=57 is the main analysis, units=56 merges
30_satellite and 31_satimage-2 (they share almost all rows), as in stats_tests.py.
"""
from pathlib import Path
import csv
import json
import math
import os
from statistics import mean

from anoddae_repro.data import CLASSICAL, benchmark_datasets
from paired_stats import compare, fmt

root = Path(__file__).resolve().parent
raw = root / "raw_win"
tables = root / "tables"
METHODS = {"original": "sweepT_50", "trim10": "impr_trim0.1_T50", "randtrim10": "impr_randtrim0.1_T50"}
PAIRS = [("trim10", "randtrim10"), ("randtrim10", "original"), ("trim10", "original"),
         ("randtrim10", "centroid"), ("trim10", "centroid")]
MERGE = {"30_satellite", "31_satimage-2"}
SAME_DATA = ("n_train", "n_test", "d", "anomaly_rate", "batch_size", "max_size")


class InputError(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise InputError(msg)


sizes = json.loads((raw / "dataset_sizes.json").read_text())
expected = {(name, seed) for name in sizes for seed in range(6)}
check(len(expected) == 726, f"expected 726 file/seed pairs, got {len(expected)}")

runs = {}
for method, exp in METHODS.items():
    rows = [json.loads(line) for line in (raw / f"{exp}_win.jsonl").read_text().splitlines()]
    # a retried job leaves several records; the last successful one counts
    last = {}
    for r in rows:
        if "error" not in r:
            last[(r["file"], r["seed"])] = r
    check(set(last) == expected, f"{exp}: {len(last)} file/seed pairs with a success, {len(expected - set(last))} missing")
    rows = list(last.values())
    for r in rows:
        check((r["exp"], r["setting"], r["protocol"], r.get("max_size")) == (exp, "unsup", "dte2", 50000),
              f"{exp} {r['file']} s{r['seed']}: unexpected run settings")
        m = r["metrics"]
        check(all(math.isfinite(m[k][x]) and 0 <= m[k][x] <= 100 for k in ("official", "centroid") for x in ("ap", "roc")),
              f"{exp} {r['file']} s{r['seed']}: metric not a finite number in 0-100")
    runs[method] = {(r["file"], r["seed"]): r for r in rows}

# all three methods must have seen exactly the same data on every file/seed
for key, base in runs["original"].items():
    for method in ("trim10", "randtrim10"):
        other = runs[method][key]
        check(all(base[f] == other[f] for f in SAME_DATA), f"{key}: data differs between original and {method}")
        check(base["metrics"]["centroid"] == other["metrics"]["centroid"], f"{key}: centroid differs for {method}")

groups = {name: {} for name in benchmark_datasets()}
for method, rs in runs.items():
    for (name, _), r in rs.items():
        group = groups[name if name in CLASSICAL else name.split("_")[0]]
        m = r["metrics"]
        group.setdefault(method, []).append((m["official"]["ap"], m["official"]["roc"]))
        if method == "original":
            group.setdefault("centroid", []).append((m["centroid"]["ap"], m["centroid"]["roc"]))

cols = [*METHODS, "centroid"]
metric_cols = [f"{m}_{k}" for m in cols for k in ("AP", "ROC_AUC")]
values = {ds: {f"{m}_{k}": mean(v[i] for v in g[m]) for m in cols for i, k in enumerate(("AP", "ROC_AUC"))}
          for ds, g in groups.items()}
check(len(values) == 57, "expected 57 groups")


units57 = list(values.values())
units56 = [v for ds, v in values.items() if ds not in MERGE]
units56.append({k: mean(values[ds][k] for ds in MERGE) for k in metric_cols})
results = compare(units57, PAIRS) + compare(units56, PAIRS)

# write only after every check and computation has passed: temporary files first, then swap
tables.mkdir(exist_ok=True)
outputs = {
    "random_trim_datasets.csv": [["dataset", *metric_cols]]
    + [[ds, *(f"{v[h]:.8f}" for h in metric_cols)] for ds, v in values.items()],
    "random_trim_mean57.csv": [["dataset", *metric_cols],
                               ["mean57", *(f"{mean(v[h] for v in units57):.8f}" for h in metric_cols)]],
    "random_trim_stats.csv": [list(results[0])] + [list(fmt(r).values()) for r in results],
}
for name, table in outputs.items():
    with (tables / f"{name}.tmp").open("w", newline="") as f:
        csv.writer(f).writerows(table)
for name in outputs:
    os.replace(tables / f"{name}.tmp", tables / name)

print("mean57 (x100):", {h: round(mean(v[h] for v in units57), 2) for h in metric_cols})
for r in results:
    print(f"n={r['units']} {r['metric']:8} {r['comparison']:24} mean {r['mean_diff']:+7.2f}  "
          f"W/L {r['wins']:2}/{r['losses']:2}  p_holm={r['p_holm']:.3g}  r_rb={r['rank_biserial']:+.3f}")
