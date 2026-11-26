"""Three supporting experiments (raw_more/, run 2026-11-25 on one Windows PC).

A  oracle trim (impr_oracletrim0.1_T50, dte2, seeds 0-5) against raw_win/: no trim, scored trim, random trim
B  contamination (protocol clean<c>, c = 0, 0.05, 0.1, 0.2; --lite: one file per group, max 10000 rows;
   seeds 0-2): no trim, scored trim, oracle trim, centroid; test rows are disjoint from training rows
C  trim share (impr_trim0.05_T50, impr_trim0.2_T50, dte2, seeds 0-2) with 0 and 0.1 from raw_win/ (seeds 0-2)

Members and seeds are averaged within each of the 57 groups; tests are two-sided Wilcoxon over groups,
Holm-corrected within each block (paired_stats.compare). Writes tables/extra_*.csv and tables/extra_summary.json.
"""
import csv, json
from pathlib import Path
from statistics import mean
import numpy as np
from anoddae_repro.data import CLASSICAL, benchmark_datasets
from paired_stats import compare, fmt

ROOT = Path(__file__).resolve().parent
WIN, MORE, TABLES = ROOT / "raw_win", ROOT / "raw_more", ROOT / "tables"
GROUPS = benchmark_datasets()
group_of = lambda f: f if f in CLASSICAL else f.split("_")[0]


def records(path, protocol, seeds):
    """Last successful record per (file, seed) for one protocol."""
    last = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "error" not in r and r.get("protocol") == protocol and r["seed"] in seeds:
            last[(r["file"], r["seed"])] = r
    return last


def by_group(recs, key):
    g = {}
    for (f, _), r in recs.items():
        g.setdefault(group_of(f), []).append(key(r))
    assert set(g) == set(GROUPS), f"missing groups: {sorted(set(GROUPS) - set(g))}"
    return {k: mean(v) for k, v in g.items()}


AP = lambda r: r["metrics"]["official"]["ap"]
ROC = lambda r: r["metrics"]["official"]["roc"]
CAP = lambda r: r["metrics"]["centroid"]["ap"]
CROC = lambda r: r["metrics"]["centroid"]["roc"]


def units(methods):
    """methods: name -> recs. Returns per-group dicts with <name>_AP / <name>_ROC_AUC (+ centroid from the first)."""
    first = next(iter(methods.values()))
    cols = {f"{m}_{k}": by_group(rs, fn) for m, rs in methods.items() for k, fn in (("AP", AP), ("ROC_AUC", ROC))}
    cols["centroid_AP"], cols["centroid_ROC_AUC"] = by_group(first, CAP), by_group(first, CROC)
    return [{"dataset": g, **{c: v[g] for c, v in cols.items()}} for g in GROUPS]


def mean57(us):
    return {k: mean(u[k] for u in us) for k in us[0] if k != "dataset"}


def write(name, rows):
    with (TABLES / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(rows[0].keys())
        w.writerows([[f"{v:.8f}" if isinstance(v, float) else v for v in r.values()] for r in rows])


summary, stats = {}, []
S6, S3 = set(range(6)), {0, 1, 2}

# ---- A: oracle trim --------------------------------------------------------
A = {"original": records(WIN / "sweepT_50_win.jsonl", "dte2", S6),
     "trim10": records(WIN / "impr_trim0.1_T50_win.jsonl", "dte2", S6),
     "randtrim10": records(WIN / "impr_randtrim0.1_T50_win.jsonl", "dte2", S6),
     "oracle10": records(MORE / "impr_oracletrim0.1_T50_more.jsonl", "dte2", S6)}
assert all(len(v) == 726 for v in A.values()), {k: len(v) for k, v in A.items()}
ua = units(A)
write("extra_oracle_datasets.csv", ua)
summary["A_mean57"] = mean57(ua)
m = summary["A_mean57"]
summary["A_gap_closed_AP"] = (m["trim10_AP"] - m["original_AP"]) / (m["oracle10_AP"] - m["original_AP"])
for r in compare(ua, [("oracle10", "trim10"), ("oracle10", "randtrim10"), ("oracle10", "original"), ("oracle10", "centroid")]):
    stats.append({"block": "A_oracle", **r})

# ---- B: contamination ------------------------------------------------------
summary["B"] = {}
for c in ("0", "0.05", "0.1", "0.2"):
    p = f"clean{c}"
    B = {"original": records(MORE / "sweepT_50_more.jsonl", p, S3),
         "trim10": records(MORE / "impr_trim0.1_T50_more.jsonl", p, S3),
         "oracle10": records(MORE / "impr_oracletrim0.1_T50_more.jsonl", p, S3)}
    assert all(len(v) == 171 for v in B.values()), (p, {k: len(v) for k, v in B.items()})
    ub = units(B)
    tr_rate = by_group(B["original"], lambda r: r["train_anomaly_rate"])
    te_rate = by_group(B["original"], lambda r: r["anomaly_rate"])
    for u in ub:
        u["train_anomaly_rate"], u["test_anomaly_rate"] = tr_rate[u["dataset"]], te_rate[u["dataset"]]
    write(f"extra_contam{c}_datasets.csv", ub)
    summary["B"][c] = mean57(ub)
    summary["B"][c]["groups_reaching_target"] = sum(abs(u["train_anomaly_rate"] - float(c)) < 0.005 for u in ub)
    for r in compare(ub, [("trim10", "original"), ("trim10", "centroid"), ("oracle10", "original"), ("original", "centroid")]):
        stats.append({"block": f"B_clean{c}", **r})

# ---- C: trim share ---------------------------------------------------------
C = {"trim0": records(WIN / "sweepT_50_win.jsonl", "dte2", S3),
     "trim5": records(MORE / "impr_trim0.05_T50_more.jsonl", "dte2", S3),
     "trim10": records(WIN / "impr_trim0.1_T50_win.jsonl", "dte2", S3),
     "trim20": records(MORE / "impr_trim0.2_T50_more.jsonl", "dte2", S3)}
assert all(len(v) == 363 for v in C.values()), {k: len(v) for k, v in C.items()}
uc = units(C)
write("extra_trimshare_datasets.csv", uc)
summary["C_mean57"] = mean57(uc)
for r in compare(uc, [("trim5", "trim0"), ("trim10", "trim0"), ("trim20", "trim0"), ("trim20", "trim10"), ("trim10", "trim5"), ("trim20", "centroid")]):
    stats.append({"block": "C_trimshare", **r})

write("extra_stats.csv", [fmt(r) for r in stats])
(TABLES / "extra_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
for k, v in summary.items():
    print(k, json.dumps(v, default=lambda x: round(x, 2))[:600])
for r in stats:
    print(f"{r['block']:12} {r['metric']:8} {r['comparison']:24} mean {r['mean_diff']:+7.2f} W/L {r['wins']:2}/{r['losses']:2} p_holm={r['p_holm']:.3g}")
