"""ADBench data loading under two protocols.

* ``dte``  — the protocol of Livernoche et al. (DTE, ICLR 2024), which the
  AnoDDAE paper says it adopts and from which all of its baseline numbers are
  taken.  Re-implemented line-by-line from ``vicliv/DTE/data_generator.py`` and
  ``vicliv/DTE/run.py``:
    - datasets with < 1000 rows are up-sampled with replacement to 1000 rows,
    - datasets with > 50 000 rows are sub-sampled to 50 000 rows,
    - semi-supervised: the first 50 % of normal rows (file order) form the
      training set; the other normals + all anomalies form the test set,
    - unsupervised: train = bootstrap sample of the full data, test = full data,
    - semi-supervised: StandardScaler fitted on the training rows only;
      unsupervised: fitted on the full data before the bootstrap, as DTE's run.py
      does; NaNs -> column mean.
  The ten CV/NLP datasets are *families* of files (e.g. CIFAR10_0..9); the DTE
  tables report the family average, so ``family_members`` lists all files.

* ``repo`` — the protocol of the released AnoDDAE code (``src/data.py`` +
  ``src/utils.py``): StandardScaler on the full data (incl. test anomalies),
  random 50 % of normals for training, unsupervised train = test = X, no
  up/down-sampling.  ``max_size`` can still be given to cap very large sets.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

# Default matches the clone location given in README.md (relative to the repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get(
    "ADBENCH_ROOT",
    os.path.join(_REPO_ROOT, "data_src", "adbench_repo", "adbench", "datasets"))

CLASSICAL = [
    "1_ALOI", "2_annthyroid", "3_backdoor", "4_breastw", "5_campaign", "6_cardio",
    "7_Cardiotocography", "8_celeba", "9_census", "10_cover", "11_donors", "12_fault",
    "13_fraud", "14_glass", "15_Hepatitis", "16_http", "17_InternetAds", "18_Ionosphere",
    "19_landsat", "20_letter", "21_Lymphography", "22_magic.gamma", "23_mammography",
    "24_mnist", "25_musk", "26_optdigits", "27_PageBlocks", "28_pendigits", "29_Pima",
    "30_satellite", "31_satimage-2", "32_shuttle", "33_skin", "34_smtp", "35_SpamBase",
    "36_speech", "37_Stamps", "38_thyroid", "39_vertebral", "40_vowels", "41_Waveform",
    "42_WBC", "43_WDBC", "44_Wilt", "45_wine", "46_WPBC", "47_yeast",
]
CV_FAMILIES = ["CIFAR10", "FashionMNIST", "MNIST-C", "MVTec-AD", "SVHN"]
NLP_FAMILIES = ["20news", "agnews", "amazon", "imdb", "yelp"]


def _listdir(sub):
    p = os.path.join(DATA_ROOT, sub)
    return sorted(f[:-4] for f in os.listdir(p) if f.endswith(".npz"))


def family_members(family: str) -> list[str]:
    """Files that make up one of the 57 benchmark datasets."""
    if family in CV_FAMILIES:
        return [f for f in _listdir("CV_by_ResNet18") if f.split("_")[0] == family]
    if family in NLP_FAMILIES:
        return [f for f in _listdir("NLP_by_BERT") if f.split("_")[0] == family]
    return [family]


def benchmark_datasets() -> list[str]:
    """The 57 ADBench datasets used by DTE / AnoDDAE (47 tabular + 5 CV + 5 NLP)."""
    return CLASSICAL + CV_FAMILIES + NLP_FAMILIES


def short_name(name: str) -> str:
    """'17_InternetAds' -> 'internetads' (DTE table naming); families unchanged."""
    if name in CV_FAMILIES or name in NLP_FAMILIES:
        return name
    return name.split("_", 1)[1].lower()


def load_raw(file_name: str):
    for sub in ("Classical", "CV_by_ResNet18", "NLP_by_BERT"):
        p = os.path.join(DATA_ROOT, sub, file_name + ".npz")
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            return d["X"].astype(np.float64), d["y"].astype(np.int64)
    raise FileNotFoundError(file_name)


def _set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class Split:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_total: int  # rows after up/down-sampling (used for the batch-size rule)


def _fill_nan(a):
    col_mean = np.nanmean(a, axis=0)
    col_mean[np.isnan(col_mean)] = 0
    idx = np.where(np.isnan(a))
    a[idx] = np.take(col_mean, idx[1])
    return a


def make_split(file_name: str, setting: str, seed: int, protocol: str = "dte",
               max_size: int = 50000, min_size: int = 1000) -> Split:
    """protocol: 'dte'  = released DTE code (small sets up-sampled in both settings)
                 'dte2' = same, but no up-sampling in the unsupervised setting
                          (a protocol defined by this project, not DTE's released pipeline)
                 'repo' = released AnoDDAE code path."""
    assert setting in ("semi", "unsup"), setting
    X, y = load_raw(file_name)
    if protocol in ("dte", "dte2"):
        _set_seed(seed)
        upsample = not (protocol == "dte2" and setting == "unsup")
        if upsample and len(y) < min_size:
            _set_seed(seed)
            idx = np.random.choice(np.arange(len(y)), min_size, replace=True)
            X, y = X[idx], y[idx]
        if len(y) > max_size:
            _set_seed(seed)
            idx = np.random.choice(np.arange(len(y)), max_size, replace=False)
            X, y = X[idx], y[idx]
        n_total = len(y)
        if setting == "semi":
            ind = np.arange(len(X))
            normal, anomal = ind[y == 0], ind[y == 1]
            n_tr = round(0.5 * normal.size)
            tr, te = normal[:n_tr], np.append(normal[n_tr:], anomal)
            x_tr, y_tr, x_te, y_te = X[tr], y[tr], X[te], y[te]
        else:
            x_tr, y_tr, x_te, y_te = X.copy(), y.copy(), X.copy(), y.copy()
        scaler = StandardScaler().fit(x_tr)
        x_tr, x_te = _fill_nan(scaler.transform(x_tr)), _fill_nan(scaler.transform(x_te))
        if setting == "unsup":  # bootstrap training sample (DTE run.py)
            sub = np.random.choice(np.arange(len(x_te)), size=len(x_te), replace=True)
            x_tr, y_tr = x_te[sub], y_te[sub]
        return Split(x_tr.astype(np.float32), y_tr, x_te.astype(np.float32), y_te, n_total)

    if protocol == "repo":  # released AnoDDAE code path
        if max_size and len(y) > max_size:
            _set_seed(seed)
            idx = np.random.choice(np.arange(len(y)), max_size, replace=False)
            X, y = X[idx], y[idx]
        n_total = len(y)
        X = StandardScaler().fit_transform(X)
        X = _fill_nan(X)
        if setting == "unsup":
            return Split(X.astype(np.float32), y, X.astype(np.float32), y, n_total)
        np.random.seed(seed)
        anomaly_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]
        np.random.shuffle(normal_idx)
        half = len(normal_idx) // 2
        tr, te = normal_idx[:half], np.concatenate([normal_idx[half:], anomaly_idx])
        return Split(X[tr].astype(np.float32), y[tr], X[te].astype(np.float32), y[te], n_total)
    if protocol.startswith("clean") and setting == "unsup":
        # inductive split with controlled contamination, e.g. 'clean0.05':
        # test = 30% of normals + 30% of anomalies (stratified, disjoint from training);
        # train = the other 70% of normals + anomalies drawn from the other 70% of anomalies
        # so that the training anomaly share is c (capped by what is available).
        # The scaler is fit on the training rows only.
        c = float(protocol[len("clean"):])
        if len(y) > max_size:
            _set_seed(seed)
            idx = np.random.choice(np.arange(len(y)), max_size, replace=False)
            X, y = X[idx], y[idx]
        n_total = len(y)
        rng = np.random.default_rng(seed)
        normal, anomal = rng.permutation(np.where(y == 0)[0]), rng.permutation(np.where(y == 1)[0])
        n_te_n, n_te_a = round(0.3 * normal.size), max(1, round(0.3 * anomal.size))
        te = np.concatenate([normal[:n_te_n], anomal[:n_te_a]])
        tr_n, pool_a = normal[n_te_n:], anomal[n_te_a:]
        want = round(c / (1 - c) * tr_n.size) if c < 1 else pool_a.size
        tr = np.concatenate([tr_n, pool_a[:min(want, pool_a.size)]])
        tr = tr[rng.permutation(tr.size)]
        scaler = StandardScaler().fit(X[tr])
        x_tr, x_te = _fill_nan(scaler.transform(X[tr])), _fill_nan(scaler.transform(X[te]))
        return Split(x_tr.astype(np.float32), y[tr], x_te.astype(np.float32), y[te], n_total)
    raise ValueError(protocol)


def batch_size_rule(n: int) -> int:
    """Closest power of two in [2^3, 2^13] to n/10 (src/utils.get_batch_size)."""
    sizes = [2 ** i for i in range(3, 14)]
    return min(sizes, key=lambda b: abs(b - n // 10))
