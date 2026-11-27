/* Minimal SVG chart kit for the AnoDDAE reproduction report.
   Every color comes from CSS tokens so both themes work; every mark has a tooltip. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const tok = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmt1 = (v) => (v == null || isNaN(v) ? "–" : v.toFixed(1));
  const fmt2 = (v) => (v == null || isNaN(v) ? "–" : v.toFixed(2));

  function s(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  // ---------- tooltip ----------
  let tip;
  function tooltip() {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "viz-tip";
      tip.setAttribute("role", "status");
      tip.hidden = true;
      document.body.appendChild(tip);
    }
    return tip;
  }
  function showTip(html, evt) {
    const t = tooltip();
    t.innerHTML = html;
    t.hidden = false;
    const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
    t.style.left = Math.max(8, x) + "px";
    t.style.top = Math.max(8, y) + "px";
  }
  function hideTip() { if (tip) tip.hidden = true; }
  function bindTip(node, html) {
    node.addEventListener("pointerenter", (e) => showTip(html, e));
    node.addEventListener("pointermove", (e) => showTip(html, e));
    node.addEventListener("pointerleave", hideTip);
    node.setAttribute("tabindex", "0");
    node.addEventListener("focus", () => {
      const r = node.getBoundingClientRect();
      showTip(html, { clientX: r.right, clientY: r.top });
    });
    node.addEventListener("blur", hideTip);
  }

  function legend(el, items) {
    const d = document.createElement("div");
    d.className = "legend";
    items.forEach((it) => {
      const sp = document.createElement("span");
      const sw = document.createElement("i");
      sw.className = "sw" + (it.dash ? " dash" : "");
      sw.style.background = it.dash ? "transparent" : `var(${it.color})`;
      if (it.dash) sw.style.borderTop = `2px dashed var(${it.color})`;
      sp.appendChild(sw);
      sp.appendChild(document.createTextNode(it.name));
      d.appendChild(sp);
    });
    el.appendChild(d);
  }
  function frame(el, width, height) {
    el.innerHTML = "";
    const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img",
      preserveAspectRatio: "xMinYMin meet", class: "viz" });
    el.appendChild(svg);
    return svg;
  }
  function niceTicks(lo, hi, n) {
    const span = hi - lo, step0 = span / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const err = step0 / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  // ---------- horizontal ranked bars ----------
  // rows: [{label, value, kind: 'ours'|'paper'|'ref'|'base', note}]
  function hbar(el, rows, o) {
    o = o || {};
    const W = 560, rowH = 17, top = 26, left = 170, right = 46;
    const H = top + rows.length * rowH + 30;
    const svg = frame(el, W, H);
    const vmax = o.max || Math.max(...rows.map((r) => r.value)) * 1.05;
    const vmin = o.min || 0;
    const x = (v) => left + ((v - vmin) / (vmax - vmin)) * (W - left - right);
    const ticks = niceTicks(vmin, vmax, 5);
    ticks.forEach((t) => {
      s("line", { x1: x(t), x2: x(t), y1: top - 6, y2: H - 26, class: "grid" }, svg);
      s("text", { x: x(t), y: H - 10, class: "tick", "text-anchor": "middle" }, svg).textContent = t;
    });
    if (o.title) s("text", { x: 0, y: 12, class: "ctitle" }, svg).textContent = o.title;
    const colors = { ours: "--series-1", paper: "--series-2", ref: "--series-3", base: "--bar-neutral" };
    rows.forEach((r, i) => {
      const y = top + i * rowH;
      const g = s("g", { class: "mark" }, svg);
      const lab = s("text", { x: left - 8, y: y + rowH * 0.68, class: "rlabel" + (r.kind !== "base" ? " strong" : ""),
        "text-anchor": "end" }, g);
      lab.textContent = r.label;
      const w = Math.max(1, x(r.value) - x(vmin));
      s("rect", { x: x(vmin), y: y + 3, width: w, height: rowH - 6, rx: 2,
        fill: `var(${colors[r.kind] || "--bar-neutral"})` }, g);
      s("text", { x: x(r.value) + 5, y: y + rowH * 0.68, class: "vlabel" + (r.kind !== "base" ? " strong" : "") },
        g).textContent = fmt1(r.value);
      s("rect", { x: 0, y: y, width: W, height: rowH, fill: "transparent", class: "hit" }, g);
      bindTip(g, `<b>${r.label}</b><br>${o.metric || "AP"}：${fmt2(r.value)}` + (r.rank ? `<br>平均排名：${fmt2(r.rank)}` : "") +
        (r.note ? `<br><span class="muted">${r.note}</span>` : ""));
    });
  }

  // ---------- line chart (optional log-x, shaded band, reference lines) ----------
  // series: [{name, color: '--series-1', points: [[x,y],...], dash}], refs: [{y, label, color}]
  function lines(el, series, o) {
    o = o || {};
    const W = o.width || 560, H = o.height || 300, L = 48, R = 16, T = 30, B = 40;
    const svg = frame(el, W, H);
    if (o.title) {
      const h = document.createElement("div");
      h.className = "ctitle-h";
      h.textContent = o.title;
      el.insertBefore(h, svg);
    }
    if (o.legend !== false && (series.length > 1 || (o.refs || []).length)) {
      legend(el, series.map((sr) => ({ name: sr.name, color: sr.color })).concat(
        (o.refs || []).map((r) => ({ name: r.label, color: r.color || "--ink-3", dash: true }))));
      el.appendChild(svg);
    }
    const xs = series.flatMap((sr) => sr.points.map((p) => p[0]));
    const ys = series.flatMap((sr) => sr.points.map((p) => p[1])).concat((o.refs || []).map((r) => r.y));
    let x0 = o.xmin ?? Math.min(...xs), x1 = o.xmax ?? Math.max(...xs);
    let y0 = o.ymin ?? Math.min(...ys), y1 = o.ymax ?? Math.max(...ys);
    const padY = (y1 - y0) * 0.08 || 1; if (o.ymin == null) y0 -= padY; if (o.ymax == null) y1 += padY;
    const lx = (v) => (o.logx ? Math.log10(v) : v);
    const X = (v) => L + ((lx(v) - lx(x0)) / (lx(x1) - lx(x0))) * (W - L - R);
    const Y = (v) => H - B - ((v - y0) / (y1 - y0)) * (H - T - B);
    if (o.band) {
      const [b0, b1, lab] = o.band;
      s("rect", { x: X(b0), y: T, width: X(b1) - X(b0), height: H - T - B, class: "band" }, svg);
      s("text", { x: X(b0) + 6, y: H - B - 8, class: "bandlabel" }, svg).textContent = lab;
    }
    niceTicks(y0, y1, 5).forEach((t) => {
      s("line", { x1: L, x2: W - R, y1: Y(t), y2: Y(t), class: "grid" }, svg);
      s("text", { x: L - 6, y: Y(t) + 4, class: "tick", "text-anchor": "end" }, svg).textContent = t;
    });
    const xt = o.xticks || niceTicks(x0, x1, 6);
    xt.forEach((t) => {
      s("text", { x: X(t), y: H - B + 16, class: "tick", "text-anchor": "middle" }, svg).textContent = t;
    });
    s("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axis" }, svg);
    if (o.xlabel) s("text", { x: (L + W - R) / 2, y: H - 6, class: "alabel", "text-anchor": "middle" }, svg).textContent = o.xlabel;
    if (o.ylabel) s("text", { x: 0, y: T - 14, class: "alabel" }, svg).textContent = o.ylabel;
    (o.refs || []).forEach((r) => {
      s("line", { x1: L, x2: W - R, y1: Y(r.y), y2: Y(r.y), class: "ref", stroke: `var(${r.color || "--ink-3"})` }, svg);
    });
    series.forEach((sr) => {
      const d = sr.points.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join("");
      s("path", { d, fill: "none", stroke: `var(${sr.color})`, "stroke-width": 2,
        "stroke-dasharray": sr.dash || "", "stroke-linejoin": "round" }, svg);
      if (sr.markers !== false && sr.points.length <= 20) {
        sr.points.forEach((p) => {
          const c = s("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: `var(${sr.color})`, class: "dot" }, svg);
          bindTip(c, `<b>${sr.name}</b><br>${o.xname || "x"} = ${p[0]}<br>${o.yname || "y"}：${fmt2(p[1])}`);
        });
      }
      if (sr.label !== false) {
        const p = sr.points[sr.points.length - 1];
        if (o.endLabels) s("text", { x: X(p[0]) - 4, y: Y(p[1]) - 8, class: "dlabel", "text-anchor": "end",
          fill: `var(--ink-2)` }, svg).textContent = sr.name;
      }
    });
    // crosshair for dense series
    const dense = series.filter((sr) => sr.points.length > 20);
    if (dense.length) {
      const hair = s("line", { y1: T, y2: H - B, class: "hair", visibility: "hidden" }, svg);
      const hit = s("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (e) => {
        const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
        const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
        const xv = o.logx ? Math.pow(10, lx(x0) + ((loc.x - L) / (W - L - R)) * (lx(x1) - lx(x0)))
          : x0 + ((loc.x - L) / (W - L - R)) * (x1 - x0);
        let html = "";
        dense.forEach((sr) => {
          let best = sr.points[0];
          sr.points.forEach((p) => { if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p; });
          html = html || `<b>${o.xname || "x"} = ${best[0]}</b>`;
          html += `<br><span class="sw" style="background:var(${sr.color})"></span>${sr.name}：${fmt2(best[1])}`;
          hair.setAttribute("x1", X(best[0])); hair.setAttribute("x2", X(best[0]));
        });
        hair.setAttribute("visibility", "visible");
        showTip(html, e);
      });
      hit.addEventListener("pointerleave", () => { hair.setAttribute("visibility", "hidden"); hideTip(); });
    }
  }

  // ---------- scatter with y = x diagonal ----------
  // pts: [{label, x, y}]
  function scatter(el, pts, o) {
    o = o || {};
    const W = 420, H = 396, L = 46, R = 14, T = o.title ? 48 : 26, B = 40;
    const svg = frame(el, W, H);
    const lo = o.min ?? 0, hi = o.max ?? 100;
    const X = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
    const Y = (v) => H - B - ((v - lo) / (hi - lo)) * (H - T - B);
    if (o.title) s("text", { x: 0, y: 14, class: "ctitle" }, svg).textContent = o.title;
    niceTicks(lo, hi, 5).forEach((t) => {
      s("line", { x1: L, x2: W - R, y1: Y(t), y2: Y(t), class: "grid" }, svg);
      s("line", { x1: X(t), x2: X(t), y1: T, y2: H - B, class: "grid" }, svg);
      s("text", { x: L - 6, y: Y(t) + 4, class: "tick", "text-anchor": "end" }, svg).textContent = t;
      s("text", { x: X(t), y: H - B + 16, class: "tick", "text-anchor": "middle" }, svg).textContent = t;
    });
    s("line", { x1: X(lo), y1: Y(lo), x2: X(hi), y2: Y(hi), class: "diag" }, svg);
    if (o.xlabel) s("text", { x: (L + W - R) / 2, y: H - 6, class: "alabel", "text-anchor": "middle" }, svg).textContent = o.xlabel;
    if (o.ylabel) s("text", { x: 0, y: T - 14, class: "alabel" }, svg).textContent = o.ylabel;
    pts.forEach((p) => {
      const c = s("circle", { cx: X(p.x), cy: Y(p.y), r: 4.5, class: "sdot", fill: "var(--series-1)" }, svg);
      bindTip(c, `<b>${p.label}</b><br>${o.xname || "x"}：${fmt2(p.x)}<br>${o.yname || "y"}：${fmt2(p.y)}`);
    });
  }

  // ---------- dot plot with error bars (variants) ----------
  // rows: [{label, mean, std, kind}], refs: [{x, label, color}]
  function dots(el, rows, o) {
    o = o || {};
    const W = 560, rowH = 26, T = 34, L = 176, R = 58;
    const H = T + rows.length * rowH + 30;
    const svg = frame(el, W, H);
    const vals = rows.flatMap((r) => [r.mean - (r.std || 0), r.mean + (r.std || 0)]).concat((o.refs || []).map((r) => r.x));
    const lo = o.min ?? Math.floor(Math.min(...vals) - 1), hi = o.max ?? Math.ceil(Math.max(...vals) + 1);
    const X = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
    if (o.title) s("text", { x: 0, y: 14, class: "ctitle" }, svg).textContent = o.title;
    niceTicks(lo, hi, 5).forEach((t) => {
      s("line", { x1: X(t), x2: X(t), y1: T - 6, y2: H - 26, class: "grid" }, svg);
      s("text", { x: X(t), y: H - 10, class: "tick", "text-anchor": "middle" }, svg).textContent = t;
    });
    (o.refs || []).forEach((r) => {
      s("line", { x1: X(r.x), x2: X(r.x), y1: T - 10, y2: H - 26, class: "ref", stroke: `var(${r.color || "--series-2"})` }, svg);
      s("text", { x: X(r.x), y: T - 16, class: "reflabel", "text-anchor": "middle" }, svg).textContent = r.label;
    });
    rows.forEach((r, i) => {
      const y = T + i * rowH + rowH / 2;
      const g = s("g", { class: "mark" }, svg);
      s("text", { x: L - 10, y: y + 4, class: "rlabel" + (r.kind === "ours" ? " strong" : ""), "text-anchor": "end" }, g).textContent = r.label;
      if (r.std) s("line", { x1: X(r.mean - r.std), x2: X(r.mean + r.std), y1: y, y2: y, class: "err" }, g);
      s("circle", { cx: X(r.mean), cy: y, r: 5, fill: `var(${r.kind === "ours" ? "--series-1" : r.kind === "ens" ? "--series-3" : "--bar-strong"})`, class: "sdot" }, g);
      s("text", { x: W - R + 10, y: y + 4, class: "vlabel" + (r.kind === "ours" ? " strong" : "") }, g).textContent = fmt1(r.mean);
      s("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: "transparent" }, g);
      bindTip(g, `<b>${r.label}</b><br>${o.metric || "AP"}：${fmt2(r.mean)}` + (r.std ? ` ± ${fmt2(r.std)}` : "") +
        (r.note ? `<br><span class="muted">${r.note}</span>` : ""));
    });
  }

  // ---------- grouped vertical bars (schedulers) ----------
  // cats: [{label, value, kind}], refs [{y,label}]
  function vbars(el, cats, o) {
    o = o || {};
    const W = 420, H = 260, L = 44, R = 12, T = 18, B = 52;
    const svg = frame(el, W, H);
    if (o.title) {
      const h = document.createElement("div");
      h.className = "ctitle-h";
      h.textContent = o.title;
      el.insertBefore(h, svg);
    }
    const vals = cats.map((c) => c.value).concat((o.refs || []).map((r) => r.y));
    const lo = o.min ?? 0, hi = o.max ?? Math.ceil(Math.max(...vals) * 1.1);
    const Y = (v) => H - B - ((v - lo) / (hi - lo)) * (H - T - B);
    const bw = (W - L - R) / cats.length;
    niceTicks(lo, hi, 5).forEach((t) => {
      s("line", { x1: L, x2: W - R, y1: Y(t), y2: Y(t), class: "grid" }, svg);
      s("text", { x: L - 6, y: Y(t) + 4, class: "tick", "text-anchor": "end" }, svg).textContent = t;
    });
    cats.forEach((c, i) => {
      const x = L + i * bw;
      const g = s("g", { class: "mark" }, svg);
      const col = c.kind === "bug" ? "--series-2" : c.kind === "fix" ? "--series-3" : "--series-1";
      s("rect", { x: x + bw * 0.18, y: Y(c.value), width: bw * 0.64, height: Y(lo) - Y(c.value), rx: 3, fill: `var(${col})` }, g);
      s("text", { x: x + bw / 2, y: Y(c.value) - 5, class: "vlabel", "text-anchor": "middle" }, g).textContent = fmt1(c.value);
      const lab = s("text", { x: x + bw / 2, y: H - B + 16, class: "tick", "text-anchor": "middle" }, g);
      lab.textContent = c.label;
      if (c.sub) s("text", { x: x + bw / 2, y: H - B + 30, class: "tick", "text-anchor": "middle" }, g).textContent = c.sub;
      s("rect", { x, y: T, width: bw, height: H - T - B, fill: "transparent" }, g);
      bindTip(g, `<b>${c.label}${c.sub ? " " + c.sub : ""}</b><br>${o.metric || "AP"}：${fmt2(c.value)}` + (c.note ? `<br><span class="muted">${c.note}</span>` : ""));
    });
    (o.refs || []).forEach((r) => {
      s("line", { x1: L, x2: W - R, y1: Y(r.y), y2: Y(r.y), class: "ref", stroke: `var(${r.color || "--ink-3"})` }, svg);
    });
    if ((o.refs || []).length) {
      legend(el, o.refs.map((r) => ({ name: `${r.label}（${fmt1(r.y)}）`, color: r.color || "--ink-3", dash: true })));
      el.appendChild(svg);
    }
    s("line", { x1: L, x2: W - R, y1: Y(lo), y2: Y(lo), class: "axis" }, svg);
  }

  window.Viz = { hbar, lines, scatter, dots, vbars, fmt1, fmt2 };
})();
