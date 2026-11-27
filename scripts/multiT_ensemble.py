"""Offline improvement test: rank-average DDAE scores trained with different T
(lite protocol, seed 0) -- e.g. a 'local' model (T=50) plus a 'global' model
(T=1000).  Uses the per-sample scores saved by the sweep runs; no training.
Output: results/summary/multiT_ensemble.json
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anoddae_repro.analyze import family_of, RES  # noqa: E402
from anoddae_repro.data import benchmark_datasets, family_members  # noqa: E402
from anoddae_repro.runner import _metrics  # noqa: E402

TS = [5, 20, 50, 100, 300, 1000]


def load_scores(T, st, f, key="official"):
    p = os.path.join(RES, "scores", f"sweepT_{T}", st, f"{f}_s0.npz")
    if not os.path.exists(p):
        return None, None
    z = np.load(p)
    return z["y"], z[key]


def rank(a):
    return pd.Series(a).rank(pct=True).values


out = {}
for st in ("semi", "unsup"):
    files = [family_members(ds)[0] for ds in benchmark_datasets()]
    combos = [(T,) for T in TS] + list(itertools.combinations(TS, 2)) + [(20, 50, 100), (50, 300, 1000), tuple(TS)]
    res = {}
    for combo in combos:
        aps, rocs = [], []
        ok = True
        for f in files:
            ys, ss = [], []
            for T in combo:
                y, s = load_scores(T, st, f)
                if y is None:
                    ok = False
                    break
                ys.append(y)
                ss.append(rank(s))
            if not ok:
                break
            m = _metrics(ys[0], np.mean(ss, axis=0))
            aps.append(m["ap"])
            rocs.append(m["roc"])
        if ok:
            res["+".join(map(str, combo))] = {"ap": float(np.mean(aps)), "roc": float(np.mean(rocs)), "n": len(aps)}
    out[st] = res
    if res:
        best = sorted(res.items(), key=lambda kv: -kv[1]["ap"])[:6]
        print(st, "single:", {k: round(v["ap"], 2) for k, v in res.items() if "+" not in k})
        print(st, "best combos:", [(k, round(v["ap"], 2), round(v["roc"], 2)) for k, v in best])
json.dump(out, open(os.path.join(RES, "summary", "multiT_ensemble.json"), "w"), indent=1)
