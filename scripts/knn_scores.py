"""kNN reference detectors under the DTE protocol, saving per-sample scores.

knn      : distance to the 5th nearest training neighbour (PyOD KNN default,
           identical to the 'kNN' column of DTE / AnoDDAE Fig. 4)
knn_mean : mean distance to the 5 nearest neighbours (rank-equivalent to DTE-NP)
"""
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anoddae_repro.data import benchmark_datasets, family_members, make_split  # noqa: E402
from anoddae_repro.runner import _metrics, RES  # noqa: E402


def job(f, setting, seed):
    s = make_split(f, setting, seed, protocol="dte" if setting == "semi" else "dte2")
    nn = NearestNeighbors(n_neighbors=5, n_jobs=1).fit(s.x_train)
    d, _ = nn.kneighbors(s.x_test)
    sc = {"knn": d[:, -1], "knn_mean": d.mean(1)}
    sdir = os.path.join(RES, "scores", "knn", setting)
    os.makedirs(sdir, exist_ok=True)
    np.savez_compressed(os.path.join(sdir, f"{f}_s{seed}.npz"), y=s.y_test.astype(np.int8),
                        **{k: v.astype(np.float32) for k, v in sc.items()})
    return dict(exp="knn", file=f, setting=setting, seed=seed,
                metrics={k: _metrics(s.y_test, v) for k, v in sc.items()})


if __name__ == "__main__":
    settings = sys.argv[1].split(",")
    seeds = [int(x) for x in sys.argv[2].split(",")]
    files = [f for ds in benchmark_datasets() for f in family_members(ds)]
    jobs = [(f, st, sd) for st in settings for sd in seeds for f in files]
    out = os.path.join(RES, "knn.jsonl")
    res = Parallel(n_jobs=4, verbose=0)(delayed(job)(*j) for j in jobs)
    with open(out, "a") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    print("wrote", len(res), "records to", out)
