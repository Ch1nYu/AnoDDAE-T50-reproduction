"""Training-free reference: score(x) = ||x - mean(x_train)||^2 in the standardised space.
This is what a DDAE trained on pure-noise inputs converges to (x0_hat -> E[x0]),
i.e. the released cosine schedule (alpha_bar_t = 0 for all t)."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anoddae_repro.data import benchmark_datasets, family_members, make_split
from anoddae_repro.runner import _metrics, RES
out = open(os.path.join(RES, "centroid.jsonl"), "w")
for st, proto in (("semi", "dte"), ("unsup", "dte2")):
    for seed in range(5):
        for ds in benchmark_datasets():
            for f in family_members(ds):
                s = make_split(f, st, seed, protocol=proto)
                sc = ((s.x_test - s.x_train.mean(0)) ** 2).sum(1)
                out.write(json.dumps(dict(exp="centroid", file=f, setting=st, seed=seed,
                                          metrics={"centroid": _metrics(s.y_test, sc)})) + "\n")
out.close()
