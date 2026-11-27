"""Sample-level agreement between DDAE scores and simple detectors.

For every file (seed 0) compute the Spearman rank correlation between the saved
DDAE test scores and (a) the training-free centroid score ||x - mean||^2,
(b) the kNN score.  A correlation near 1 means DDAE ranks test points almost
exactly like that detector.  Output: results/summary/score_correlation.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anoddae_repro.data import benchmark_datasets, family_members, make_split  # noqa: E402
from anoddae_repro.analyze import family_of, RES  # noqa: E402

rows = []
for st, exp, proto in (("unsup", "ddae_unsup_T1000", "dte2"), ("semi", "ddae_semi_T50", "dte")):
    for ds in benchmark_datasets():
        for f in family_members(ds):
            p = os.path.join(RES, "scores", exp, st, f"{f}_s0.npz")
            k = os.path.join(RES, "scores", "knn", st, f"{f}_s0.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            s = make_split(f, st, 0, protocol=proto)
            assert (s.y_test == z["y"]).all()
            cen = ((s.x_test - s.x_train.mean(0)) ** 2).sum(1)
            r = {"setting": st, "file": f, "dataset": family_of(f),
                 "rho_centroid": spearmanr(z["official"], cen).statistic}
            if os.path.exists(k):
                r["rho_knn"] = spearmanr(z["official"], np.load(k)["knn"]).statistic
            rows.append(r)
df = pd.DataFrame(rows)
agg = df.groupby(["setting", "dataset"]).mean(numeric_only=True).reset_index()
agg.to_csv(os.path.join(RES, "summary", "score_correlation.csv"), index=False)
print(agg.groupby("setting")[["rho_centroid", "rho_knn"]].describe().round(3).T)
