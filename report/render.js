(function () {
  const get = (p) => p.split(".").reduce((o, k) => (o == null ? undefined : o[k]), D);
  document.querySelectorAll("[data-k]").forEach((el) => {
    const v = get(el.dataset.k);
    if (v == null) { if (!el.textContent.trim()) el.textContent = "–"; return; }
    const f = el.dataset.f || "1";
    el.textContent = typeof v === "number" ? v.toFixed(+f) : v;
  });
  const $ = (id) => document.getElementById(id);
  const has = (id) => !!$(id);

  // 1) leaderboards
  if (has("lb-semi") && D.board.semi_PR.length) Viz.hbar($("lb-semi"), D.board.semi_PR, { title: "半監督：57 資料集平均 AP", metric: "AP" });
  if (has("lb-unsup") && D.board.unsup_PR.length) Viz.hbar($("lb-unsup"), D.board.unsup_PR, { title: "無監督：57 資料集平均 AP", metric: "AP" });

  // 2) per-dataset scatters
  if (has("sc-unsup") && D.scatter.unsup_PR)
    Viz.scatter($("sc-unsup"), D.scatter.unsup_PR.map((r) => ({ label: r.label, x: r.centroid, y: r.ddae })),
      { title: "無監督：每個資料集的 AP", xlabel: "到均值距離 ‖x−μ‖²（零訓練）", ylabel: "DDAE（T=1000）", xname: "到均值距離", yname: "DDAE" });
  if (has("sc-semi") && D.scatter.semi_PR)
    Viz.scatter($("sc-semi"), D.scatter.semi_PR.map((r) => ({ label: r.label, x: r.knn, y: r.ddae })),
      { title: "半監督：每個資料集的 AP", xlabel: "kNN（k=5）", ylabel: "DDAE（T=50）", xname: "kNN", yname: "DDAE" });

  // 3) alpha_bar schedules (bug illustration)
  if (has("ab-chart") && D.alpha_bar) {
    const cols = ["--series-1", "--series-3", "--bar-strong", "--series-2"];
    const names = Object.keys(D.alpha_bar);
    Viz.lines($("ab-chart"), names.map((n, i) => ({ name: n, color: cols[i], points: D.alpha_bar[n], markers: false,
      dash: i === 3 ? "" : "" })), { title: "訊號保留比例 ᾱₜ", xlabel: "t / T", ymin: -0.02, ymax: 1.02, xname: "t/T", yname: "ᾱₜ",
      xticks: [0, 0.25, 0.5, 0.75, 1] });
  }

  // 4) per-step curves
  function perStep(id, ps, title, shadeFrom, refY, refLab) {
    if (!has(id) || !ps) return;
    const t = ps.t;
    const series = [
      { name: "單一步驟 t 的 AP", color: "--series-1", points: t.map((x, i) => [x, ps.step[i]]), markers: false },
      { name: "累積 Σ₁..ₜ 的 AP", color: "--series-3", points: t.map((x, i) => [x, ps.cum[i]]), markers: false },
    ];
    const o = { title, xlabel: "擴散步驟 t", ylabel: "AP（%）", xname: "t", yname: "AP" };
    if (shadeFrom != null) o.band = [shadeFrom, t[t.length - 1], "ᾱₜ < 0.01：輸入近乎純雜訊"];
    if (refY != null) o.refs = [{ y: refY, label: refLab, color: "--series-2" }];
    Viz.lines($(id), series, o);
  }
  perStep("ps-unsup", D.summary.per_step_unsup, "無監督（T=1000，seed 0）", D.pure_noise_from_t,
    D.centroid_all ? D.centroid_all.unsup_ap.mean : null, "到均值距離");
  perStep("ps-semi", D.summary.per_step_semi, "半監督（T=50，seed 0）", null,
    D.centroid_all ? D.centroid_all.semi_ap.mean : null, "到均值距離");

  // 5) schedulers (lite protocol)
  function sched(id, st, title) {
    const sd = D.summary.schedulers_lite && D.summary.schedulers_lite[st];
    if (!has(id) || !sd) return;
    const order = [["linear", "linear"], ["quadratic", "quad."], ["sigmoid", "sigm."], ["exponential", "exp."],
      ["cosine-official", "cosine", "官方版"], ["cosine-fixed", "cosine", "修正版"]];
    const cats = order.filter((o) => sd[o[0]]).map((o) => ({ label: o[1], sub: o[2] || "", value: sd[o[0]].official_ap.mean,
      kind: o[0] === "cosine-official" ? "bug" : o[0] === "cosine-fixed" ? "fix" : "base",
      note: o[0] === "cosine-official" ? "ᾱₜ ≡ 0，輸入全為雜訊" : "" }));
    const cen = sd.linear && sd.linear.centroid_ap ? sd.linear.centroid_ap.mean : null;
    Viz.vbars($(id), cats, { title, metric: "AP", refs: cen ? [{ y: cen, label: "到均值距離", color: "--ink-3" }] : [] });
  }
  sched("sch-semi", "semi", "半監督（T=100）");
  sched("sch-unsup", "unsup", "無監督（T=100）");

  // 6) T sweep (lite protocol)
  function sweep(id, st, title) {
    const sw = D.summary.sweepT_lite && D.summary.sweepT_lite[st];
    if (!has(id) || !sw || !sw.official) return;
    const Ts = Object.keys(sw.official.ap).map(Number).sort((a, b) => a - b);
    const ser = [{ name: "官方評分 Σ‖e‖", color: "--series-1", points: Ts.map((T) => [T, sw.official.ap[T].mean]) }];
    if (sw.featnorm) ser.push({ name: "特徵正規化評分", color: "--series-3", points: Ts.filter((T) => sw.featnorm.ap[T]).map((T) => [T, sw.featnorm.ap[T].mean]) });
    const cen = sw.centroid ? sw.centroid.ap[Ts[0]].mean : null;
    Viz.lines($(id), ser, { title, logx: true, xticks: [5, 10, 20, 50, 100, 300, 1000], xlabel: "擴散步數 T（對數軸）",
      ylabel: "AP（%）", xname: "T", yname: "AP", refs: cen ? [{ y: cen, label: "到均值距離", color: "--ink-3" }] : [] });
  }
  sweep("sw-semi", "semi", "半監督");
  sweep("sw-unsup", "unsup", "無監督");

  // 7) improvement variants
  function variants(id, st, title, paperRef) {
    const v = D.summary.variants && D.summary.variants[st];
    if (!has(id) || !v) return;
    const spec = [["official_ap", "DDAE 官方評分 Σ‖e‖", "ours"], ["paper_ap", "論文 Eq.6 Σ‖e‖²", "base"],
      ["stepnorm_ap", "逐步驟正規化", "base"], ["featnorm_ap", "逐特徵正規化", "base"],
      ["knn_ap", "kNN（k=5）", "base"], ["ens_rank_ap", "DDAE + kNN 排名集成", "ens"], ["ens_feat_rank_ap", "特徵正規化 + kNN 集成", "ens"],
      ["centroid_ap", "到均值距離（零訓練）", "base"]];
    const rows = spec.filter((s) => v[s[0]]).map((s) => ({ label: s[1], mean: v[s[0]].mean, std: v[s[0]].std, kind: s[2],
      note: `${v[s[0]].n_seeds} 個 seed × ${v[s[0]].n_datasets} 個資料集` }));
    Viz.dots($(id), rows, { title, metric: "AP", refs: [{ x: paperRef, label: "論文 DDAE", color: "--series-2" }] });
  }
  variants("var-semi", "semi", "半監督（57 資料集平均 AP，±1 SD over seeds）", 61.36);
  variants("var-unsup", "unsup", "無監督（57 資料集平均 AP，±1 SD over seeds）", 32.77);
})();
