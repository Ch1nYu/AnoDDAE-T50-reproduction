"""Paired significance tests on the 57 dataset groups in tables/datasets.csv.

Run recompute.py first. Writes stats.csv next to datasets.csv
(default tables/; use --tables results/tables after a full rerun).

Two analyses, each with its own Holm-Bonferroni correction over six tests:
- units=57: every benchmark group is one paired unit (main result).
- units=56: sensitivity check. 30_satellite and 31_satimage-2 share almost all
  feature rows (5,801 of satimage-2's 5,803 rows also occur in satellite), so they
  are averaged into one unit first.
"""
from pathlib import Path
import argparse
import csv

import numpy as np

from anoddae_repro.data import benchmark_datasets
from paired_stats import compare, fmt

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--tables", default="tables", help="folder with datasets.csv (default: tables)")
tables = root / parser.parse_args().tables
with (tables / "datasets.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
names = [r["dataset"] for r in rows]
if sorted(names) != sorted(benchmark_datasets()):
    raise SystemExit(f"{tables / 'datasets.csv'}: expected each of the 57 benchmark groups once, "
                     f"got {len(names)} rows with {len(set(names))} distinct names")
for r in rows:
    for k, v in r.items():
        if k != "dataset" and not 0 <= float(v) <= 100:
            raise SystemExit(f"{r['dataset']} {k}={v}: not a number in 0-100")

PAIRS = [("trim10", "original"), ("trim10", "centroid"), ("original", "centroid")]
MERGE = {"30_satellite", "31_satimage-2"}


def units(merge):
    data = [{k: float(r[k]) for k in r if k != "dataset"} for r in rows if not (merge and r["dataset"] in MERGE)]
    if merge:
        pair = [r for r in rows if r["dataset"] in MERGE]
        assert len(pair) == 2
        data.append({k: float(np.mean([float(r[k]) for r in pair])) for k in pair[0] if k != "dataset"})
    return data


results = compare(units(False), PAIRS) + compare(units(True), PAIRS)

with (tables / "stats.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    for r in results:
        writer.writerow(fmt(r))

for r in results:
    print(f"n={r['units']} {r['metric']:8} {r['comparison']:22} mean {r['mean_diff']:+7.2f}  "
          f"W/L {r['wins']:2}/{r['losses']:2}  p={r['p_value']:.3g}  p_holm={r['p_holm']:.3g}  "
          f"r_rb={r['rank_biserial']:+.3f}")
