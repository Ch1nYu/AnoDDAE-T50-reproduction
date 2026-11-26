"""Rebuild tables/datasets.csv and tables/mean57.csv from per-sample scores.

Default: the historical Linux run in raw/ (tag _full, legacy score layout).
For a new run: python recompute.py --raw results --tag _rerun --out results/tables
"""
from pathlib import Path
import argparse
import csv
import json
from statistics import mean

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from anoddae_repro.data import CLASSICAL, benchmark_datasets
from anoddae_repro.runner import score_path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Rebuild tables from a full 726-job run of both methods.")
parser.add_argument("--raw", default="raw", help="folder with <exp><tag>.jsonl and scores/ (default: raw, the historical run)")
parser.add_argument("--tag", default="_full", help="runner --tag of that run (default: _full)")
parser.add_argument("--max-size", type=int, default=None, help="runner --max_size of that run (default: 50000)")
parser.add_argument("--out", default="tables", help="output folder (default: tables)")
args = parser.parse_args()
raw = root / args.raw
# The historical run predates the per-run score layout and the max_size field.
legacy = args.raw == "raw" and args.tag == "_full"
if legacy and args.max_size not in (None, 50000):
    parser.error("the historical run in raw/ used --max_size 50000")
if args.max_size is not None and args.max_size <= 0:
    parser.error("--max-size must be a positive integer")
max_size = 50000 if args.max_size is None else args.max_size


class InputError(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise InputError(msg)


sizes = json.loads((raw / "dataset_sizes.json").read_text())
expected = {(name, seed) for name in sizes for seed in range(6)}
check(len(expected) == 726, f"expected 726 file/seed pairs, got {len(expected)}")
methods = {"original": "sweepT_50", "trim10": "impr_trim0.1_T50"}
groups = {name: {} for name in benchmark_datasets()}
labels, centroids = {}, {}
print(f"reading {'historical run' if legacy else 'run'} in {raw}, tag {args.tag}, max_size {max_size}")
for method, exp in methods.items():
    rows = [json.loads(line) for line in (raw / f"{exp}{args.tag}.jsonl").read_text().splitlines()]
    rows = [r for r in rows if "error" not in r and r.get("protocol") == "dte2" and r.get("setting") == "unsup"
            and (legacy or r.get("max_size") == max_size)]
    if legacy:
        keys = {(r["file"], r["seed"]) for r in rows}
        check(len(rows) == len(keys), f"{exp}: {len(rows) - len(keys)} duplicated rows in the historical run")
    else:
        # a retried job leaves several records; the last successful one is the one whose
        # score file the runner kept
        last = {}
        for r in rows:
            last[(r["file"], r["seed"])] = r
        rows = list(last.values())
    keys = {(r["file"], r["seed"]) for r in rows}
    check(keys == expected, f"{exp}: {len(keys)} file/seed pairs, {len(expected - keys)} missing")
    for r in rows:
        name, seed = r["file"], r["seed"]
        check(r["exp"] == exp, f"{exp}{args.tag}.jsonl holds a record of {r['exp']} ({name} seed {seed})")
        cen = (r["metrics"]["centroid"]["ap"], r["metrics"]["centroid"]["roc"])
        check(all(np.isfinite(cen)) and all(0 <= c <= 100 for c in cen), f"bad centroid metrics for {name} seed {seed}")
        if (name, seed) in centroids:
            check(centroids[(name, seed)] == cen, f"centroid metrics differ between methods for {name} seed {seed}")
        centroids[(name, seed)] = cen
        path = (raw / "scores" / exp / "unsup" / f"{name}_s{seed}.npz" if legacy else
                Path(score_path(str(raw), exp, "dte2", max_size, args.tag, "unsup", name, seed)))
        check(path.exists(), f"missing score file {path}")
        group = groups[name if name in CLASSICAL else name.split("_")[0]]
        with np.load(path) as data:
            y = data["y"]
            score = data["official"].astype(np.float64)
        check(len(y) == r["n_test"] and np.isfinite(score).all(), f"bad scores in {path}")
        if (name, seed) in labels:
            check(np.array_equal(labels[(name, seed)], y), f"labels differ between methods for {name} seed {seed}")
        labels[(name, seed)] = y
        pair = (100 * average_precision_score(y, score), 100 * roc_auc_score(y, score))
        logged = (r["metrics"]["official"]["ap"], r["metrics"]["official"]["roc"])
        check(np.allclose(pair, logged, rtol=0, atol=1e-6), f"{path}: scores give {pair}, run log says {logged}")
        group.setdefault(method, []).append(pair)
        if method == "original":
            baseline = r["metrics"]["centroid"]
            group.setdefault("centroid", []).append((baseline["ap"], baseline["roc"]))

out = root / args.out
out.mkdir(parents=True, exist_ok=True)
columns = ["dataset"] + [f"{m}_{metric}" for m in [*methods, "centroid"] for metric in ("AP", "ROC_AUC")]
values = []
for dataset, group in groups.items():
    values.append([dataset] + [mean(v) for m in [*methods, "centroid"] for v in zip(*group[m])])
for filename, rows in [("datasets", values), ("mean57", [["mean57", *map(mean, zip(*(v[1:] for v in values)))]])]:
    with (out / f"{filename}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for label, *metrics in rows:
            writer.writerow([label, *(f"{x:.8f}" for x in metrics)])
