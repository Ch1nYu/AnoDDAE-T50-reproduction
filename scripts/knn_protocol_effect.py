"""How much does the data protocol alone move a deterministic baseline?
kNN (k=5) under the DTE protocol vs the released AnoDDAE code protocol ('repo')."""
import os, sys, json
import numpy as np
from sklearn.neighbors import NearestNeighbors
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anoddae_repro.data import benchmark_datasets, family_members, make_split
from anoddae_repro.runner import _metrics, RES
out = open(os.path.join(RES, "knn_repo_protocol.jsonl"), "w")
for st in ("semi", "unsup"):
    for seed in (0, 1, 2):
        for ds in benchmark_datasets():
            for f in family_members(ds):
                s = make_split(f, st, seed, protocol="repo", max_size=50000)
                d, _ = NearestNeighbors(n_neighbors=5).fit(s.x_train).kneighbors(s.x_test)
                out.write(json.dumps(dict(exp="knn_repo", file=f, setting=st, seed=seed,
                                          metrics={"knn": _metrics(s.y_test, d[:, -1])})) + "\n")
out.close()
