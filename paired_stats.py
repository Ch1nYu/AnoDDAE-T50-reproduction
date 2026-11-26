"""Paired Wilcoxon tests with Holm correction, shared by stats_tests.py and random_trim_control.py."""
import math

import numpy as np
from scipy.stats import rankdata, wilcoxon


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation (R+ - R-) / (R+ + R-), zero differences dropped."""
    d = d[d != 0]
    if len(d) == 0:
        return float("nan")
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def compare(units, pairs, metrics=("AP", "ROC_AUC")):
    """One block of tests: every pair on every metric, Holm-corrected within the block.

    units: list of dicts with keys like "trim10_AP". A comparison whose differences are all
    zero has nothing to rank; its W, p, effect size and Holm p are NaN and it is left out
    of the correction.
    """
    rows = []
    for metric in metrics:
        for a, b in pairs:
            x = np.array([u[f"{a}_{metric}"] for u in units], dtype=float)
            y = np.array([u[f"{b}_{metric}"] for u in units], dtype=float)
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                raise ValueError(f"non-finite values in {a} vs {b} ({metric})")
            d = x - y
            if d.any():
                test = wilcoxon(x, y, alternative="two-sided")
                w, p = float(test.statistic), float(test.pvalue)
                if not 0.0 <= p <= 1.0:
                    raise ValueError(f"p-value out of range for {a} vs {b} ({metric}): {p}")
            else:
                w = p = float("nan")
            rows.append({"units": len(units), "metric": metric, "comparison": f"{a} vs {b}",
                         "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                         "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                         "wilcoxon_W": w, "p_value": p, "rank_biserial": rank_biserial(d),
                         "p_holm": float("nan")})
    tested = sorted((i for i, r in enumerate(rows) if not math.isnan(r["p_value"])), key=lambda i: rows[i]["p_value"])
    running = 0.0
    for rank, i in enumerate(tested):
        running = max(running, min(1.0, (len(tested) - rank) * rows[i]["p_value"]))
        rows[i]["p_holm"] = running
    return rows


def fmt(row):
    return {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}
