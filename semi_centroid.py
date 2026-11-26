"""Semi-supervised squared-distance baseline (no training), for comparison with the paper's semi-supervised DDAE.

Protocol dte (DTE public code; the paper's semi-supervised setting), seeds 0-4, 57 groups (members of
CV/NLP families averaged first, as recompute.py). Score = squared distance to the training mean.
Writes tables/semi_centroid.json: {group: [AP, ROC-AUC]} (x100). Needs ADBENCH_ROOT (full ADBench data).
Usage: python semi_centroid.py
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from anoddae_repro.data import make_split, benchmark_datasets, family_members, CV_FAMILIES, NLP_FAMILIES

out = {}
for fam in benchmark_datasets():
    files = family_members(fam) if fam in CV_FAMILIES + NLP_FAMILIES else [fam]
    aps, rocs = [], []
    for f in files:
        for seed in range(5):
            s = make_split(f, "semi", seed, protocol="dte")
            sc = ((s.x_test - s.x_train.mean(0)) ** 2).sum(1)
            aps.append(100 * average_precision_score(s.y_test, sc)); rocs.append(100 * roc_auc_score(s.y_test, sc))
    out[fam] = [float(np.mean(aps)), float(np.mean(rocs))]
(root / "tables").mkdir(exist_ok=True)
(root / "tables" / "semi_centroid.json").write_text(json.dumps(out, indent=1))
v = list(out.values())
print(f"{len(v)} groups  mean AP {np.mean([a for a, _ in v]):.2f}  ROC-AUC {np.mean([r for _, r in v]):.2f}")
