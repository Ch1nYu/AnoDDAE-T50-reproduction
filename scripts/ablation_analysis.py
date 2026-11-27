"""Ablation table: one component changed at a time, same lite protocol, 3 seeds."""
import json, os, sys, collections
import numpy as np
from scipy.stats import wilcoxon
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("ANODDAE_RESULTS", os.path.join(ROOT, "results")); sys.path.insert(0, ROOT)
from anoddae_repro.data import benchmark_datasets, family_members
fam = {f: ds for ds in benchmark_datasets() for f in family_members(ds)}

def per_ds(exp, setting, metric="official"):
    p = os.path.join(RES, f"{exp}_lite.jsonl")
    if not os.path.exists(p): return None
    acc = collections.defaultdict(list)
    for l in open(p):
        r = json.loads(l)
        if r.get("setting") != setting or "error" in r or r["seed"] not in (0, 1, 2): continue
        acc[fam[r["file"]]].append(r["metrics"][metric]["ap"])
    return {d: np.mean(v) for d, v in acc.items()} if acc else None

ROWS = [  # label, {column: (exp, metric)}
    ("DDAE (baseline)", {"u50": ("trimq0.0_T50", "official"), "u1k": ("abl1k_base", "official"), "s50": ("trimq0.0_T50", "official")}),
    ("w/o timestep embedding", {"u50": ("abl_temb0_T50", "official"), "u1k": ("abl1k_temb0", "official"), "s50": ("abl_temb0_T50", "official")}),
    ("timestep embedding dim 32", {"u50": ("abl_temb32_T50", "official"), "u1k": ("abl1k_temb32", "official"), "s50": ("abl_temb32_T50", "official")}),
    ("cosine schedule (fixed)", {"u50": ("abl_cosfix_T50", "official"), "u1k": ("abl1k_cosfix", "official"), "s50": ("abl_cosfix_T50", "official")}),
    ("squared score (Eq. 6)", {"u50": ("trimq0.0_T50", "paper"), "u1k": ("abl1k_base", "paper"), "s50": ("trimq0.0_T50", "paper")}),
    ("fixed noise, DAE σ = 0.1", {"u50": ("dae_s0.1", "official"), "u1k": ("dae_s0.1", "official"), "s50": ("dae_s0.1", "official")}),
    ("fixed noise, DAE σ = 0.5", {"u50": ("dae_s0.5", "official"), "u1k": ("dae_s0.5", "official"), "s50": ("dae_s0.5", "official")}),
    ("+ contrastive, α = 0.1", {"u50": ("abl_ddaec0.1_T50", "official"), "u1k": ("abl1k_ddaec0.1", "official"), "s50": ("abl_ddaec0.1_T50", "official")}),
    ("+ contrastive, α = 0.5", {"u50": ("abl_ddaec0.5_T50", "official"), "u1k": ("abl1k_ddaec0.5", "official"), "s50": ("abl_ddaec0.5_T50", "official")}),
    ("+ trimming, q = 10% (ours)", {"u50": ("trimq0.1_T50", "official"), "u1k": ("abl1k_trim0.1", "official"), "s50": ("trimq0.1_T50", "official")}),
]
COLS = {"u50": "unsup", "u1k": "unsup", "s50": "semi"}
base = {c: per_ds(*ROWS[0][1][c][:1], COLS[c], ROWS[0][1][c][1]) for c in COLS}
out = []
for label, spec in ROWS:
    row = {"label": label}
    for c, (exp, met) in spec.items():
        v = per_ds(exp, COLS[c], met)
        if v is None or base[c] is None:
            row[c] = None; continue
        ds = sorted(set(v) & set(base[c]))
        a = np.array([v[d] for d in ds]); b = np.array([base[c][d] for d in ds])
        diff = a - b
        p = None if np.allclose(diff, 0) else float(wilcoxon(diff).pvalue)
        row[c] = dict(ap=float(a.mean()), delta=float(diff.mean()), p=p, n=len(ds))
    out.append(row)
    print(f"{label:30}" + "".join(
        f"  {c}: " + ("    n/a        " if row[c] is None else f"{row[c]['ap']:5.1f} ({row[c]['delta']:+5.1f}{'' if row[c]['p'] is None else ', p=%.3f' % row[c]['p']}, n={row[c]['n']})")
        for c in COLS))
json.dump(out, open(os.path.join(RES, "summary", "ablation.json"), "w"), indent=1)
