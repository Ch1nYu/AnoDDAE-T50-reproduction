"""Are the rows that trimming drops really anomalies?

Reads raw_trimlog/npz/<exp>/<file>_s<seed>.npz (written by the runner under results/trimlog/:
y_train, and the training-row indices each trim round dropped) and raw_trimlog/<exp>_trimlog.jsonl.
Per job: base = anomaly share of the training rows; drop = anomaly share of the dropped rows, averaged over
the 9 trim rounds; recall = share of training anomalies dropped in the last round, against its ceiling
min(1, dropped/anomalies). Members and seeds are averaged within each of the 57 groups, as in recompute.py.
Also checks that every AP/ROC matches raw_win/ (the logging must not change training).
Writes tables/trimlog_datasets.csv and tables/trimlog_summary.txt.
Usage: python trimlog_analysis.py
"""
import csv, json, os
from pathlib import Path
from statistics import mean, median
import numpy as np
from scipy.stats import wilcoxon, spearmanr
from anoddae_repro.data import CLASSICAL, benchmark_datasets

ROOT = Path(__file__).resolve().parent
RAW_WIN = ROOT / "raw_win"
RAW = ROOT / "raw_trimlog"
EXPS = {"score": "impr_trim0.1_T50", "random": "impr_randtrim0.1_T50"}

def last_ok(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "error" not in r:
            out[(r["file"], r["seed"])] = r
    return out

groups = {g: {} for g in benchmark_datasets()}
mismatch = 0
for m, exp in EXPS.items():
    new, old = last_ok(RAW / f"{exp}_trimlog.jsonl"), last_ok(os.path.join(RAW_WIN, f"{exp}_win.jsonl"))
    assert len(new) == len(old) == 726, (exp, len(new), len(old))
    for k, r in new.items():
        a, b = r["metrics"]["official"], old[k]["metrics"]["official"]
        mismatch += (a["ap"], a["roc"]) != (b["ap"], b["roc"])
        z = np.load(RAW / "npz" / exp / f"{k[0]}_s{k[1]}.npz")
        y, d = z["y_train"].astype(float), z["dropped"]
        assert d.shape[0] == 9, k
        n_anom = y.sum()
        rec = y[d[-1]].sum() / n_anom if n_anom else np.nan
        ceil = min(1.0, d.shape[1] / n_anom) if n_anom else np.nan
        g = groups[k[0] if k[0] in CLASSICAL else k[0].split("_")[0]].setdefault(m, [])
        g.append(dict(base=y.mean(), drop=y[d].mean(), recall=rec, ceil=ceil, ap=a["ap"]))

rows = []
for g, v in groups.items():
    row = {"dataset": g}
    for m in EXPS:
        for f in ("base", "drop", "recall", "ceil", "ap"):
            row[f"{m}_{f}"] = float(np.nanmean([x[f] for x in v[m]]))
    rows.append(row)
assert len(rows) == 57
with open(ROOT / "tables" / "trimlog_datasets.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

out = []
def say(s):
    print(s); out.append(s)

col = lambda c: np.array([r[c] for r in rows])
base, sd, rd = col("score_base") * 100, col("score_drop") * 100, col("random_drop") * 100
lift = sd / np.where(base > 0, base, np.nan)
say(f"AP/ROC mismatches vs raw_win: {mismatch}")
say(f"mean57 anomaly share %: training {base.mean():.2f}  scored-dropped {sd.mean():.2f}  random-dropped {rd.mean():.2f}")
say(f"scored drop > base in {int((sd > base).sum())}/57 groups, < base in {int((sd < base).sum())}; median lift {np.nanmedian(lift):.2f}x")
say(f"Wilcoxon scored vs random dropped share: p={wilcoxon(sd, rd).pvalue:.3g}")
say(f"Wilcoxon scored dropped vs training share: p={wilcoxon(sd, base).pvalue:.3g}")
say(f"last-round recall of training anomalies: scored {np.nanmean(col('score_recall')):.3f} (ceiling {np.nanmean(col('score_ceil')):.3f}), random {np.nanmean(col('random_recall')):.3f}")
gain = col("score_ap") - col("random_ap")
rho = spearmanr(sd - rd, gain)
say(f"Spearman (scored - random dropped share) vs (scored - random AP): rho={rho.statistic:.3f} p={rho.pvalue:.3g}")
say("groups where scored drop < training share: " + ", ".join(r["dataset"] for r in rows if r["score_drop"] < r["score_base"]))
assert mismatch == 0, "logging changed training: AP/ROC differ from raw_win"
(ROOT / "tables" / "trimlog_summary.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
