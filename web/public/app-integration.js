/* Connect the supplied console design to the real FastAPI runtime. */
(() => {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pretty = (s) => String(s ?? "—").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
  const stamp = (ms) => ms ? new Date(ms).toLocaleTimeString() : "—";
  let socket;
  let zoom = 1;
  const confidencePercent = (d) => {
    const raw = Number(d.confidence ?? d.conf ?? 0);
    return Math.min(100, Math.max(0, raw <= 1 ? raw * 100 : raw));
  };
  function setZoom(next) {
    zoom = Math.min(2.5, Math.max(1, Math.round(next * 4) / 4));
    const stream = $("#stream");
    if (stream) stream.style.transform = `scale(${zoom})`;
    const level = $("#zoomLevel");
    if (level) level.textContent = `${Math.round(zoom * 100)}%`;
    const out = $("#zoomOut"), inside = $("#zoomIn");
    if (out) out.disabled = zoom <= 1;
    if (inside) inside.disabled = zoom >= 2.5;
  }
  document.addEventListener("click", (event) => {
    if (event.target.closest("#zoomOut")) setZoom(zoom - 0.25);
    if (event.target.closest("#zoomIn")) setZoom(zoom + 0.25);
  });
  setZoom(1);

  function renderDetectionChart(classes) {
    const rows = Object.entries(classes || {}).map(([name, raw]) => [name, Number(raw) || 0]).filter(([, count]) => count > 0);
    if (!rows.length) return '<p>No detection counts recorded yet.</p>';
    rows.sort((a, b) => b[1] - a[1]);
    const total = rows.reduce((sum, [, count]) => sum + count, 0);
    const max = Math.max(...rows.map(([, count]) => count), 1);
    const exactShares = rows.map(([, count]) => count / total * 100);
    const displayedShares = exactShares.map(Math.floor);
    const remainingPoints = 100 - displayedShares.reduce((sum, share) => sum + share, 0);
    const remainderOrder = exactShares.map((share, index) => ({ index, fraction: share - Math.floor(share), count: rows[index][1] }))
      .sort((a, b) => b.fraction - a.fraction || b.count - a.count);
    for (let i = 0; i < remainingPoints; i++) displayedShares[remainderOrder[i].index]++;
    const bars = rows.map(([name, count], index) => {
      const label = pretty(name);
      const height = Math.max(5, Math.round(count / max * 78) + 18);
      const share = displayedShares[index];
      return `<div class="detection-bar" data-class="${esc(name)}" data-count="${count}" tabindex="0" style="--bar-height:${height}%"><div class="bar-tip"><b>${esc(label)}</b><span>${count} detections</span><small>${share}% of total</small></div><i style="height:${height}%"></i><small>${esc(label)}</small></div>`;
    }).join("");
    return `<div class="detection-bars">${bars}</div><div class="chart-foot"><span>${total} detections across ${rows.length} classes</span><span>Hover or focus a bar for details</span></div>`;
  }

  window.applyInspectionFrame = (frame) => {
    if (!frame) return;
    document.body.classList.add("api-has-frame");
    const verdict = String(frame.verdict || "review").toLowerCase();
    const badge = $("#verdictBadge");
    if (badge) { badge.textContent = `● ${verdict.toUpperCase()}`; badge.className = `tag ${verdict}`; }
    const decision = document.querySelector(".decision");
    if (decision) decision.className = `decision ${verdict}`;
    const title = $("#decisionTitle");
    const detail = $("#decisionDetail");
    const icon = $("#decisionIcon");
    if (title) title.textContent = verdict === "pass" ? "Within specification" : verdict === "fail" ? "Fail inspection" : "Route to operator review";
    if (detail) detail.textContent = verdict === "pass" ? "No actionable defects found." : verdict === "fail" ? "Defect threshold exceeded; route for disposition." : "Potential defect requires confirmation.";
    if (icon) icon.textContent = verdict === "pass" ? "✓" : verdict === "fail" ? "!" : "△";

    const detections = Array.isArray(frame.detections) ? frame.detections : [];
    const product = pretty(frame.profile);
    const resultProduct = $("#resultProduct");
    const resultDefects = $("#resultDefects");
    const resultConfidence = $("#resultConfidence");
    if (resultProduct) resultProduct.textContent = product;
    if (resultDefects) resultDefects.textContent = String(detections.length);
    if (resultConfidence) resultConfidence.textContent = detections.length
      ? `${Math.max(...detections.map(confidencePercent)).toFixed(1)}%` : "—";
    const anomaly = frame.anomaly || {};
    const anomalyRegions = Array.isArray(anomaly.regions) ? anomaly.regions : [];
    const flaggedAnomaly = !detections.length && Boolean(anomaly.is_anomaly);
    const visibleCount = detections.length || (flaggedAnomaly ? Math.max(anomalyRegions.length, 1) : 0);
    const countLabel = detections.length ? "Defects found" : flaggedAnomaly ? "Anomaly regions" : "Defects found";
    const count = $("#detectionCount");
    if (count) count.textContent = `${countLabel}: ${visibleCount}`;
    const stageLabel = $("#stageDefectLabel");
    const stageCount = $("#stageDefectCount");
    if (stageLabel) stageLabel.textContent = countLabel;
    if (stageCount) stageCount.textContent = String(visibleCount);

    const rows = $("#detectionRows");
    if (rows) {
      rows.innerHTML = detections.map((d) => {
        const name = d.class || d.class_name || d.name || d.label || "Unknown defect";
        const confidence = confidencePercent(d);
        const pct = confidence.toFixed(1);
        return `<div class="detection"><i></i><span><b>${esc(product)} · ${esc(pretty(name))}</b><small>YOLO confidence · ${pct}%</small></span><strong>${pct}%<u><i style="width:${confidence}%"></i></u></strong></div>`;
      }).join("") || (flaggedAnomaly
        ? `<div class="detection"><i></i><span><b>Unclassified anomaly</b><small>Profile anomaly score · z = ${Number(anomaly.z || 0).toFixed(2)}</small></span></div>`
        : '<div class="detection"><i></i><span><b>No defects detected</b><small>Current processed frame</small></span></div>');
    }

    const frameLabel = document.querySelector(".frame");
    if (frameLabel && frame.frame_index != null) frameLabel.textContent = `FRAME ${String(frame.frame_index).padStart(6, "0")}`;
    const inspection = document.querySelector(".verdict .panel-head small");
    if (inspection && frame.inspection_id) inspection.textContent = `Inspection ${frame.inspection_id}`;
  };

  function setLive(frame) {
    if (!frame) return;
    window.applyInspectionFrame?.(frame);
    document.body.classList.add("api-has-frame");
    const source = $("#source");
    if (source) source.textContent = frame.source === "dataset" ? "Dataset replay" : pretty(frame.source);
    const threshold = $("#decisionThreshold");
    if (threshold) threshold.textContent = frame.threshold_profile || "Active profile";
    const part = $("#partSequence");
    if (part) part.textContent = frame.part_state?.item_id || frame.stream_id || "—";
    const time = $("#vtime");
    if (time) time.textContent = stamp(frame.ts_ms);
    const recentTime = $("#recentTime");
    if (recentTime) recentTime.textContent = stamp(frame.ts_ms);
    const recentFps = $("#recentFps");
    if (recentFps) recentFps.textContent = `${Number(frame.fps || 0).toFixed(1)} FPS`;
    const live = $("#liveTag");
    if (live) live.textContent = frame.source === "dataset" ? "● REPLAY" : "● LIVE INPUT";
    const frameLabel = document.querySelector(".frame");
    if (frameLabel) frameLabel.textContent = `FRAME ${String(frame.frame_index ?? "—").padStart(6, "0")}`;
    const fps = $("#statFps");
    if (fps) fps.innerHTML = `${Number(frame.fps || 0).toFixed(1)} <em>FPS</em>`;
    const latency = $("#statLatency");
    if (latency) latency.innerHTML = `${Number(frame.latency_ms?.total || 0).toFixed(1)} <em>ms</em>`;
    const engine = $("#statEngine");
    if (engine) engine.textContent = frame.engine || "YOLO";
    const waitingTitle = $("#awaitingResults b");
    if (waitingTitle) waitingTitle.textContent = "Latest frame processed";
    const waitingCopy = $("#awaitingResults small");
    if (waitingCopy) waitingCopy.textContent = `${pretty(frame.source)} · ${frame.detections?.length || 0} detections · ${frame.verdict?.toUpperCase() || "—"}`;
    document.body.classList.add("api-connected");
  }

  async function refreshHealth() {
    try {
      const h = await (await fetch("/api/health", { cache: "no-store" })).json();
      const status = $("#modelStatus");
      if (status) status.textContent = h.detector_engine?.startsWith("yolo") ? "REAL YOLO" : h.detector_engine === "detector-off" ? "PROFILE ANOMALY GATE" : pretty(h.detector_engine || "OFFLINE");
      const model = $("#modelName");
      if (model) model.innerHTML = `${esc(h.detector_engine === "detector-off" ? "Anomaly gate" : h.detector_engine || "Detector")} <small>· ${esc(h.profile || "—")}</small>`;
      const reviews = $("#reviewCount");
      if (reviews) reviews.textContent = h.db?.open_reviews ?? "0";
      const inspected = $("#statInspections");
      if (inspected) inspected.textContent = h.db?.inspections ?? "—";
      const pending = $("#statReviews");
      if (pending) pending.textContent = h.db?.open_reviews ?? "—";
      const fps = $("#statFps");
      if (fps && !document.body.classList.contains("api-has-frame")) fps.innerHTML = `${Number(h.fps || 0).toFixed(1)} <em>FPS</em>`;
      document.body.classList.toggle("api-connected", h.status === "ok");
      if (h.status === "ok") openSocket();
      const warn = (h.warnings || []).join(", ");
      const statusText = $("#status");
      if (statusText) statusText.textContent = `● ${h.detector_engine?.startsWith("yolo") ? "REAL YOLO" : "PROFILE ANOMALY GATE"} · ${h.source} · ${h.device}${warn ? " · " + warn : ""}`;
      refreshRecent();
    } catch {
      document.body.classList.remove("api-connected");
      const status = $("#modelStatus");
      if (status) status.textContent = "OFFLINE";
    }
  }

  async function refreshRecent() {
    try {
      const payload = await (await fetch("/api/inspections?limit=10&status=all", {cache:"no-store"})).json();
      const rows = (payload.items || []).map((r) => {
        const det = r.detections?.[0];
        const verdict = String(r.final_verdict || r.verdict || "review").toUpperCase();
        return [stamp(r.ts_ms), r.id, pretty(det?.class || det?.class_name || "—"), det ? `${Math.round((det.conf || 0)*100)}%` : "—", verdict, verdict === "PASS" ? "green" : verdict === "FAIL" ? "red" : "amber"];
      });
      data.splice(0, data.length, ...rows);
      window.renderRows?.();
    } catch { /* leave table empty if unavailable */ }
  }

  function openSocket() {
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${protocol}//${location.host}/ws/live`);
    socket.onmessage = (event) => {
      try { const msg = JSON.parse(event.data); if (msg.type === "frame") setLive(msg.frame); } catch { /* ignore malformed frames */ }
    };
    socket.onclose = () => setTimeout(openSocket, 2000);
  }

  async function hydrate(id) {
    const host = $("#subPage");
    if (!host) return;
    host.insertAdjacentHTML("beforeend", '<p class="api-page-status">Loading live project data…</p>');
    const status = host.querySelector(".api-page-status");
    try {
      if (id === "review") {
        const q = await (await fetch("/api/review/queue?limit=100")).json();
        host.innerHTML = `<div class="mini-grid"><div class="panel mini"><small>OPEN REVIEWS</small><b>${q.open_reviews ?? q.count ?? 0}</b><span>From SQLite inspection queue</span></div></div><section class="panel card"><h2>Flagged inspections</h2><p>Choose a disposition to record an operator review.</p><div class="table-scroll"><table><thead><tr><th>INSPECTION</th><th>TIME</th><th>PROFILE</th><th>DETECTION</th><th>RESULT</th><th>ACTION</th></tr></thead><tbody>${(q.items || []).map((r) => `<tr><td class="id">${esc(r.id)}</td><td>${esc(stamp(r.ts_ms))}</td><td>${esc(pretty(r.profile))}</td><td>${esc(pretty(r.detections?.[0]?.class || "—"))}</td><td>${esc(String(r.verdict || "review").toUpperCase())}</td><td><button class="button" data-review="${esc(r.id)}">Review →</button></td></tr>`).join("") || '<tr><td colspan="6">No inspections are waiting for review.</td></tr>'}</tbody></table></div></section>`;
      } else if (id === "metrics") {
        const m = await (await fetch("/api/metrics")).json();
        host.innerHTML = `<div class="mini-grid"><div class="panel mini"><small>INSPECTIONS</small><b>${m.total ?? 0}</b><span>Stored records</span></div><div class="panel mini"><small>PASS RATE</small><b>${((m.pass_rate || 0) * 100).toFixed(1)}%</b><span>Current database</span></div><div class="panel mini"><small>OPEN REVIEWS</small><b>${m.counts?.open_reviews ?? 0}</b><span>Awaiting disposition</span></div><div class="panel mini"><small>LATENCY P95</small><b>${Number(m.latency?.p95 || 0).toFixed(1)} ms</b><span>Measured pipeline</span></div></div><section class="panel card"><h2>Detections by class</h2><p>Counts returned by the inspection metrics API</p>${renderDetectionChart(m.classes)}</section>`;
      } else if (id === "datasets") {
        const d = await (await fetch("/api/datasets/verify")).json();
        host.innerHTML = `<div class="mini-grid"><div class="panel mini"><small>NEU-DET IMAGES</small><b>${d.neu?.total ?? "—"}</b><span>Dataset report</span></div><div class="panel mini"><small>PROFILES</small><b>${d.profiles?.length ?? 0}</b><span>Configured profiles</span></div></div><section class="panel card"><h2>Dataset verification</h2><pre class="api-json">${esc(JSON.stringify({neu:d.neu,mvtec:d.mvtec,profiles:d.profiles,warnings:d.warnings},null,2))}</pre></section>`;
      } else if (id === "audit") {
        const a = await (await fetch("/api/audit?limit=100")).json();
        host.innerHTML = `<section class="panel card"><h2>Recent activity</h2><p>Events from the backend audit log</p>${(a.items || []).map((r) => `<div class="audit-row"><time>${esc(stamp(r.ts_ms))}</time><b>${esc(r.action)}<small>${esc(r.entity)} · ${esc(r.entity_id || "—")}</small></b><span>${esc(r.actor)}</span></div>`).join("") || "No audit events recorded."}</section>`;
      } else if (id === "settings") {
        const [t,c] = await Promise.all([fetch("/api/thresholds").then(r=>r.json()),fetch("/api/control/state").then(r=>r.json())]);
        host.innerHTML = `<section class="panel card"><h2>Decision thresholds</h2><p>Changes are saved to the active runtime profile.</p><div class="setting"><span><b>Fusion confidence floor</b><small>Minimum score accepted by the fusion stage.</small></span><label><input id="threshold" type="range" min=".05" max=".95" step=".01" value="${t.conf_min}"> <b id="thresholdVal">${Number(t.conf_min).toFixed(2)}</b></label></div><div class="setting"><b>Active profile</b><span>${esc(c.profile_name || c.profile)}</span></div><div class="setting"><b>Detector engine</b><span>${esc(c.detector?.engine)}</span></div><div class="setting"><b>Device</b><span>${esc(c.detector?.device)}</span></div><div class="setting"><b>Steel novelty gate</b><span>Untrusted · informational only</span></div><button class="button" id="saveThreshold">Save confidence floor</button></section>`;
        $("#threshold")?.addEventListener("input",e=>$("#thresholdVal").textContent=Number(e.target.value).toFixed(2));
      }
      host.querySelector(".api-page-status")?.remove();
    } catch (err) {
      if (status) status.textContent = `Could not load this page from the API: ${err.message}`;
    }
  }

  const originalPage = window.page;
  if (typeof originalPage === "function") window.page = function(id, keepMenuOpen) {
    originalPage(id, keepMenuOpen);
    if (["review","metrics","datasets","audit","settings"].includes(id)) hydrate(id);
  };

  document.addEventListener("click", async (event) => {
    const review = event.target.closest("[data-review]");
    if (review) {
      event.preventDefault(); event.stopImmediatePropagation();
      const disposition = prompt("Disposition: true_defect, false_alarm, or reclassify", "true_defect");
      if (!disposition || !["true_defect","false_alarm","reclassify"].includes(disposition)) return;
      try {
        const r = await fetch(`/api/inspections/${encodeURIComponent(review.dataset.review)}/review`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({disposition})});
        if (!r.ok) throw new Error((await r.json()).detail || "Review failed");
        window.toast?.("Disposition recorded"); hydrate("review");
      } catch (err) { window.toast?.(err.message); }
    }
    if (event.target.closest("#saveThreshold")) {
      event.preventDefault();
      try {
        const r = await fetch("/api/thresholds", {method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify({conf_min:Number($("#threshold").value)})});
        if (!r.ok) throw new Error((await r.json()).detail || "Could not save threshold");
        window.toast?.("Confidence floor saved");
      } catch (err) { window.toast?.(err.message); }
    }
    if (event.target.closest("#export")) location.href="/api/export/inspections.csv";
    if (event.target.closest("#pause")) {
      const paused = event.target.closest("#pause").dataset.paused === "1";
      fetch("/api/control",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action:paused?"resume":"pause"})}).catch(()=>{});
    }
  }, true);

  const originalConnect = window.connect;
  refreshHealth(); setInterval(refreshHealth, 5000);
  document.addEventListener("DOMContentLoaded", () => {
    document.body.classList.remove("auth-locked");
    $("#overlayBoxes")?.classList.add("hidden");
    const stats = document.querySelectorAll("#livePage .stats .mini, #livePage .stats>div");
    if (stats.length) refreshHealth();
  });
})();
