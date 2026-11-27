"""Assemble report data (from results/summary/*) and inline it, the CSS and the
chart kit into report/AnoDDAE_report.html (self-contained single file)."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anoddae_repro.analyze import load, per_dataset, family_of  # noqa: E402

SUM = os.path.join(ROOT, "results", "summary")
REP = os.path.join(ROOT, "report")
S = json.load(open(os.path.join(SUM, "summary.json")))
D = {"summary": S, "meta": {"date": "2026-11-05"}}

NICE = {"DDAE (ours)": "DDAE（本復現）", "Centroid": "到均值距離 ‖x−μ‖²（零訓練）"}


def board(st, lab):
    """Leaderboard rows: 26 DTE baselines + ours + paper DDAE + centroid."""
    key = f"rank_{st}_{lab}"
    if key not in S:
        return []
    means, ranks = S[key]["means"], S[key]["ranks"]
    rows = []
    for m, v in means.items():
        kind = "ours" if m == "DDAE (ours)" else "ref" if m == "Centroid" else "base"
        rows.append({"label": NICE.get(m, m), "value": v, "rank": ranks.get(m), "kind": kind})
    paper = {"semi_PR": 61.36, "semi_ROC": 83.17, "unsup_PR": 32.77, "unsup_ROC": 74.08}[f"{st}_{lab}"]
    rows.append({"label": "DDAE（論文 Table 1）", "value": paper, "kind": "paper", "note": "論文數值，非逐資料集"})
    rows.sort(key=lambda r: -r["value"])
    return rows


D["board"] = {f"{st}_{lab}": board(st, lab) for st in ("semi", "unsup") for lab in ("PR", "ROC")}

# per-dataset scatter data
sc = {}
for st, lab in (("unsup", "PR"), ("semi", "PR"), ("unsup", "ROC"), ("semi", "ROC")):
    p = os.path.join(SUM, f"per_dataset_{st}_{lab}.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, index_col=0)
        sc[f"{st}_{lab}"] = [{"label": i, "ddae": r["DDAE (ours)"], "centroid": r["Centroid"], "knn": r["kNN"],
                             "dtenp": r["DTE-NP"]} for i, r in df.iterrows() if not math.isnan(r["DDAE (ours)"])]
D["scatter"] = sc

# alpha_bar curves for the schedule-bug figure (computed exactly as in the code)
from anoddae_repro.models import NoiseSchedule  # noqa: E402
ab = {}
for name, T, kind, impl in (("linear, T=50", 50, "linear", "official"), ("linear, T=1000", 1000, "linear", "official"),
                            ("cosine 修正版, T=100", 100, "cosine", "fixed"), ("cosine 官方版, T=100", 100, "cosine", "official")):
    a = NoiseSchedule(T, kind, cosine_impl=impl).alpha_bar.numpy()
    idx = np.unique(np.linspace(0, T - 1, min(T, 120)).astype(int))
    ab[name] = [[round(float(i / (T - 1)), 4), round(float(a[i]), 4)] for i in idx]
D["alpha_bar"] = ab
lin1000 = NoiseSchedule(1000, "linear").alpha_bar.numpy()
D["pure_noise_from_t"] = int(np.argmax(lin1000 < 0.01))

# kNN pipeline validation
val = {}
for st in ("semi", "unsup"):
    fs = sorted(f for f in os.listdir(os.environ.get("ANODDAE_RESULTS", os.path.join(ROOT, "results"))) if f.startswith(f"knn_validation_{st}_seed") and f.endswith(".csv") and "dte2" not in f)
    if fs:
        dfs = [pd.read_csv(os.path.join(ROOT, "results", f), index_col=0) for f in fs]
        roc = pd.concat([d["roc"] for d in dfs], axis=1).mean(axis=1)
        pr = pd.concat([d["pr"] for d in dfs], axis=1).mean(axis=1)
        val[st] = {"ours_roc": float(roc.mean()), "ours_ap": float(pr.mean()), "dte_roc": float(dfs[0]["dte_roc"].mean()),
                   "dte_ap": float(dfs[0]["dte_pr"].mean()),
                   "median_abs_diff_roc": float((roc - dfs[0]["dte_roc"]).abs().median()),
                   "exact_2dp": int(((roc - dfs[0]["dte_roc"]).abs() < 0.005).sum()), "n": int(len(roc))}
D["knn_validation"] = val
p = os.path.join(ROOT, "results", "check_vs_official_internetads.csv")
if os.path.exists(p):
    a = np.loadtxt(p, delimiter=",", skiprows=1)
    D["official_check"] = {"off_ap": [float(a[:, 1].mean()), float(a[:, 1].std())], "off_roc": [float(a[:, 2].mean()), float(a[:, 2].std())],
                           "re_ap": [float(a[:, 4].mean()), float(a[:, 4].std())], "re_roc": [float(a[:, 5].mean()), float(a[:, 5].std())],
                           "n": int(len(a))}
pc = os.path.join(SUM, "score_correlation.csv")
if os.path.exists(pc):
    cc = pd.read_csv(pc)
    D["corr"] = {f"{st}_{k}": float(cc[cc.setting == st][f"rho_{k}"].mean()) for st in ("semi", "unsup") for k in ("centroid", "knn")}
    D["corr"].update({f"{st}_{k}_min": float(cc[cc.setting == st][f"rho_{k}"].min()) for st in ("semi", "unsup") for k in ("centroid", "knn")})
# flattened improvement numbers (keys without dots so the page can address them)
tr = S.get("trimming_lite", {})
D["trim"] = {}
for T in ("50", "1000"):
    for v in ("official", "featnorm"):
        for m in ("ap", "roc"):
            k = f"q0.1_T{T}_{v}_{m}"
            if k in tr:
                D["trim"][f"T{T}_{v}_{m}_base"] = tr[k]["base"]["mean"]
                D["trim"][f"T{T}_{v}_{m}_trim"] = tr[k]["trim"]["mean"]
                D["trim"][f"T{T}_{v}_{m}_gain"] = tr[k]["trim"]["mean"] - tr[k]["base"]["mean"]
mt = S.get("multiT", {})
D["multiT"] = {st: {k.replace("+", "_"): v for k, v in d.items()} for st, d in mt.items()}
from anoddae_repro.analyze import summary as _summ  # noqa: E402
tf = [r for r in load("impr_trim0.1_T50") if r["setting"] == "unsup"]
if tf:
    a = _summ(tf, "official", "ap", "unsup")
    b0 = per_dataset([r for r in load("ddae_unsup_T1000") if r["setting"] == "unsup" and r["seed"] == 0], "official", "ap", "unsup").iloc[:, 0].mean()
    c0 = per_dataset([r for r in load("ddae_unsup_T1000") if r["setting"] == "unsup" and r["seed"] == 0], "centroid", "ap", "unsup").iloc[:, 0].mean()
    D["trim_full"] = {"ap": a["mean"], "base": float(b0), "gain": a["mean"] - float(b0), "centroid": float(c0),
                      "note": f"同 seed 比較：與 T=1000 原版相比 {a['mean'] - b0:+.2f}，與到均值距離（{c0:.2f}）相比 {a['mean'] - c0:+.2f}，皆不顯著（Wilcoxon p > 0.6）。T=50 不需改 T 即可在無監督追平"}
else:
    D["trim_full"] = {"note": "執行中，完成後更新"}
pu = os.path.join(SUM, "upsampling_effect.json")
if os.path.exists(pu):
    u = json.load(open(pu))
    D["upsampling"] = {"n_up": len(u["upsampled"]), "n_other": 57 - len(u["upsampled"]),
                       "ddae_other": u["means_other"]["DDAE (ours)"], "dtenp_other": u["means_other"]["DTE-NP"],
                       "knn_other": u["means_other"]["kNN"], "ddae_up": u["means_up"]["DDAE (ours)"],
                       "dtenp_up": u["means_up"]["DTE-NP"], "knn_up": u["means_up"]["kNN"], "icl_up": u["means_up"]["ICL"],
                       "p_dtenp": u["tests_other"]["DTE-NP"]["p"], "p_knn": u["tests_other"]["kNN"]["p"],
                       "p_ddpm": u["tests_other"]["DDPM"]["p"], "d_dtenp": u["tests_other"]["DTE-NP"]["mean_diff"],
                       "d_knn": u["tests_other"]["kNN"]["mean_diff"], "d_ddpm": u["tests_other"]["DDPM"]["mean_diff"],
                       "rank_ddae": u["rank_other"]["DDAE (ours)"], "rank_dtenp": u["rank_other"]["DTE-NP"]}
def _seed0(exp, st, v="official", m="ap"):
    rr = [r for r in load(exp) if r["setting"] == st and r["seed"] == 0]
    df = per_dataset(rr, v, m, st)
    if df.empty or df.iloc[:, 0].notna().sum() < 57:
        return None
    return float(df.iloc[:, 0].mean())


D["ablate"] = {}
for key, (se, ue) in {"ddae": ("ddae_semi_T50", "ddae_unsup_T1000"), "ddaec": ("ddaec_semi_T50", "ddaec_unsup_T1000"),
                      "dae01": ("dae_s0.1", "dae_s0.1"), "dae05": ("dae_s0.5", "dae_s0.5")}.items():
    D["ablate"][key] = {"semi_ap": _seed0(se, "semi"), "semi_roc": _seed0(se, "semi", m="roc"),
                        "unsup_ap": _seed0(ue, "unsup"), "unsup_roc": _seed0(ue, "unsup", m="roc")}
if D["ablate"]["ddae"]["semi_ap"] and D["ablate"]["dae05"]["semi_ap"] and D["ablate"]["dae01"]["semi_ap"]:
    D["ablate"]["ddae_minus_dae_semi"] = D["ablate"]["ddae"]["semi_ap"] - max(D["ablate"]["dae05"]["semi_ap"], D["ablate"]["dae01"]["semi_ap"])
def _lite(exp, st, v="official", m="ap"):
    rr = [r for r in load(exp, "_lite") if r["setting"] == st]
    df = per_dataset(rr, v, m, st)
    return None if df.empty else float(df.iloc[:, 0].mean())


_b, _t = _lite("sweepT_50", "semi"), _lite("impr_trim0.1_T50", "semi")
if _b is not None and _t is not None:
    D["trim_semi"] = {"base": _b, "trim": _t, "gain": _t - _b}
D["ddaec_alpha"] = {}
for key, exp in (("ddae", "sweepT_50"), ("a05", "ddaec_semi_T50"), ("a01", "ddaec0.1_semi_T50"), ("a001", "ddaec0.01_semi_T50")):
    for m in ("ap", "roc"):
        rr = [r for r in load(exp, "_lite") if r["setting"] == "semi"]
        df = per_dataset(rr, "official", m, "semi")
        if not df.empty and df.iloc[:, 0].notna().sum() >= 57:
            D["ddaec_alpha"][f"{key}_{m}"] = float(df.iloc[:, 0].mean())
cen = load("centroid")
if cen:
    from anoddae_repro.analyze import summary
    D["centroid_all"] = {f"{st}_{m}": summary([r for r in cen if r["setting"] == st], "centroid", m, st)
                         for st in ("semi", "unsup") for m in ("ap", "roc")}

TABLES = {}


def f1(v):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def table1_html():
    t = S.get("table1", {})
    rows = [("DDAE", "DDAE", "ours"), ("DDAE (Eq.6 squared score)", "DDAE（論文 Eq.6 平方評分）", "ours2"),
            ("DDAE-C", "DDAE-C", "ours"), ("DAE (sigma=0.1)", "DAE σ=0.1", "ours2"), ("DAE (sigma=0.5)", "DAE σ=0.5", "ours2"),
            ("Centroid distance ||x-mean||^2", "到均值距離（零訓練）", "ours2"), ("kNN (our run)", "kNN k=5（本次重跑）", "ours2")]
    paper = {"DDAE": "[paper] DDAE", "DDAE-C": "[paper] DDAE-C", "DAE (sigma=0.5)": "[paper] DAE"}
    out = ['<table><thead><tr><th>方法</th><th>來源</th><th class="num">無監督 AP</th><th class="num">無監督 ROC</th>'
           '<th class="num">半監督 AP</th><th class="num">半監督 ROC</th><th class="num">seeds</th></tr></thead><tbody>']
    for key, name, kind in rows:
        r = t.get(key)
        if not r:
            continue
        def cell(st, m):
            v, sd = r.get(f"{st}_{m}"), r.get(f"{st}_{m}_std")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return '<td class="num muted">未完成</td>'
            one_seed = int(r.get(f"{st}_seeds", 0) or 0) <= 1
            if sd is None or (isinstance(sd, float) and math.isnan(sd)) or one_seed:
                return f'<td class="num">{v:.2f}</td>'
            return f'<td class="num">{v:.2f}<span class="muted"> ± {sd:.2f}</span></td>'
        seeds = f"{int(r.get('unsup_seeds', 0) or 0)} / {int(r.get('semi_seeds', 0) or 0)}"
        out.append(f'<tr class="ours"><td>{name}</td><td>本復現</td>{cell("unsup","PR")}{cell("unsup","ROC")}{cell("semi","PR")}{cell("semi","ROC")}<td class="num">{seeds}</td></tr>')
        if key in paper and paper[key] in t:
            p = t[paper[key]]
            out.append(f'<tr class="paper"><td>{"DAE" if key.startswith("DAE") else name}</td><td>論文 Table 1</td>'
                       + "".join(f'<td class="num">{p[f"{st}_{m}"]:.2f}<span class="muted"> ± {p[f"{st}_{m}_std"]:.2f}</span></td>'
                                 for st, m in (("unsup", "PR"), ("unsup", "ROC"), ("semi", "PR"), ("semi", "ROC")))
                       + '<td class="num">5 / 5</td></tr>')
    for k in ("[paper] DTE-NP", "[paper] DDPM", "[paper] DTE-C"):
        if k in t:
            p = t[k]
            out.append(f'<tr class="paper"><td>{k[8:]}</td><td>論文（取自 DTE）</td>'
                       + "".join(f'<td class="num">{p[f"{st}_{m}"]:.2f}<span class="muted"> ± {p[f"{st}_{m}_std"]:.2f}</span></td>'
                                 for st, m in (("unsup", "PR"), ("unsup", "ROC"), ("semi", "PR"), ("semi", "ROC")))
                       + '<td class="num">5 / 5</td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def tests_html():
    out = ['<table><thead><tr><th>設定</th><th>對手</th><th class="num">平均差（AP）</th><th class="num">勝 / 負</th>'
           '<th class="num">Wilcoxon p</th></tr></thead><tbody>']
    any_ = False
    for st, lab in (("semi", "半監督"), ("unsup", "無監督")):
        tt = S.get(f"tests_{st}_PR", {})
        for b, r in tt.items():
            any_ = True
            cls = "pos" if r["mean_diff"] > 0 else "neg"
            out.append(f'<tr><td>{lab}</td><td>{"到均值距離" if b == "Centroid" else b}</td><td class="num {cls}">{r["mean_diff"]:+.2f}</td>'
                       f'<td class="num">{r["wins"]} / {r["losses"]}</td><td class="num">{r["p_wilcoxon"]:.3g}</td></tr>')
    out.append("</tbody></table>")
    return "".join(out) if any_ else "<p class='muted'>尚未完成</p>"


TABLES["TABLE1"] = table1_html()
TABLES["TESTS"] = tests_html()
html = open(os.path.join(REP, "template.html"), encoding="utf-8").read()
body = open(os.path.join(REP, "body.html"), encoding="utf-8").read() if os.path.exists(os.path.join(REP, "body.html")) else ""
for key, frag in TABLES.items():
    body = body.replace(f"<!--__{key}__-->", frag)
html = html.replace("<!--__BODY__-->", body)
html = html.replace("/*__RENDER__*/", open(os.path.join(REP, "render.js"), encoding="utf-8").read())
html = html.replace("/*__CSS__*/", open(os.path.join(REP, "style.css"), encoding="utf-8").read())
html = html.replace("/*__CHARTS__*/", open(os.path.join(REP, "charts.js"), encoding="utf-8").read())
html = html.replace("/*__DATA__*/null", json.dumps(D, ensure_ascii=False, default=float))
out = os.path.join(REP, "AnoDDAE_report.html")
open(out, "w", encoding="utf-8").write(html)
print("wrote", out, f"{len(html) / 1024:.0f} KB")
