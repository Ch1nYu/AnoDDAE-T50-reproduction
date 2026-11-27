"""Aggregate JSONL results into the paper's reporting format.

Per (setting, seed): family-average over files -> 57 dataset values ->
57-dataset mean.  Reported: mean over seeds of the 57-dataset mean, +- the
mean over datasets of the per-dataset std across seeds (verified to be the
convention of AnoDDAE Table 1 / DTE: it reproduces their +- values exactly);
`std_of_mean` keeps the std over seeds of the 57-dataset mean.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ANODDAE_RESULTS points the analysis at a results folder, e.g. raw_full/ (see FULL_REPRO.md)
RES = os.environ.get("ANODDAE_RESULTS", os.path.join(ROOT, "results"))
REF = os.path.join(ROOT, "reference")

# AnoDDAE paper, Table 1 (57 datasets, 5 seeds); PR-AUC == average precision
PAPER_TABLE1 = {
    #            unsup PR, unsup ROC, semi PR, semi ROC
    "DAE":    (19.85, 63.73, 59.60, 81.02),
    "DDPM":   (30.09, 70.01, 52.41, 77.92),
    "DTE-NP": (29.47, 72.15, 57.13, 81.49),
    "DTE-IG": (21.62, 65.95, 54.49, 77.85),
    "DTE-C":  (28.79, 72.76, 52.02, 80.37),
    "DDAE-C": (31.97, 73.72, 59.48, 81.54),
    "DDAE":   (32.77, 74.08, 61.36, 83.17),
}
PAPER_STD = {
    "DAE": (2.51, 3.44, 3.18, 1.73), "DDPM": (2.84, 2.29, 2.26, 1.37),
    "DTE-NP": (1.75, 1.19, 2.67, 1.29), "DTE-IG": (4.63, 7.53, 6.17, 4.65),
    "DTE-C": (3.28, 3.15, 2.77, 1.43), "DDAE-C": (0.87, 0.79, 2.55, 1.11),
    "DDAE": (0.69, 0.51, 2.23, 1.0),
}


def family_of(file):
    from anoddae_repro.data import CV_FAMILIES, NLP_FAMILIES, short_name
    head = file.split("_")[0]
    if head in CV_FAMILIES or head in NLP_FAMILIES:
        return head
    return short_name(file)


def load(exp, tag=""):
    path = os.path.join(RES, f"{exp}{tag}.jsonl")
    recs = []
    if not os.path.exists(path):
        return recs
    seen = {}
    for line in open(path):
        r = json.loads(line)
        if "error" in r:
            continue
        seen[(r["file"], r["setting"], r["seed"])] = r  # last one wins
    return list(seen.values())


def per_dataset(recs, variant="official", metric="ap", setting=None):
    """-> DataFrame index=dataset(57), columns=seed (family-averaged)."""
    acc = defaultdict(list)
    for r in recs:
        if setting and r["setting"] != setting:
            continue
        if variant not in r["metrics"]:
            continue
        acc[(family_of(r["file"]), r["seed"])].append(r["metrics"][variant][metric])
    if not acc:
        return pd.DataFrame()
    s = pd.Series({k: float(np.mean(v)) for k, v in acc.items()})
    df = s.unstack()
    return df


def summary(recs, variant="official", metric="ap", setting="semi", require=57):
    df = per_dataset(recs, variant, metric, setting)
    if df.empty:
        return None
    full = df.dropna(axis=1, thresh=require) if require else df
    complete = full.columns[full.notna().sum() >= require] if require else full.columns
    if len(complete) == 0:
        return dict(n_datasets=int(df.notna().sum().max()), mean=float(df.mean().mean()), std=float("nan"), n_seeds=0)
    means = full[complete].mean()
    sub = full[complete]
    # paper / DTE convention: mean over datasets of the per-dataset std across seeds
    pd_std = float(sub.std(axis=1, ddof=0).mean()) if len(complete) > 1 else 0.0
    return dict(n_datasets=int(full[complete].notna().sum().min()), mean=float(means.mean()),
                std=pd_std, std_of_mean=float(means.std(ddof=1)) if len(means) > 1 else 0.0,
                n_seeds=len(complete))


def dte_table(setting, metric):
    name = {"ap": "PR", "roc": "ROC"}[metric]
    return pd.read_csv(os.path.join(REF, f"dte_{setting}_{name}_mean.csv"), index_col=0)


def avg_rank(df):
    """df: index=dataset, columns=methods (higher better) -> mean rank (1 = best)."""
    return df.rank(axis=1, ascending=False).mean().sort_values()
