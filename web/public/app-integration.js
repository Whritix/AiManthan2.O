/* Connect the supplied console design to the real FastAPI runtime. */
(() => {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pretty = (s) => String(s ?? "—").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
  const stamp = (ms) => ms ? new Date(ms).toLocaleTimeString() : "—";
  let socket;
  let latestSpeed = null;
  let vlmReady = false;
  let vlmReportFilename = null;
  let initialReportRoutingDone = false;
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

  function renderDatasetOverview(verification, unified) {
    const neu = verification.neu || {}, mvtec = verification.mvtec || {};
    if (!unified.available) return `<section class="panel card"><h2>Dataset summary unavailable</h2><p>The prepared combined dataset manifest is missing. NEU-DET: ${Number(neu.total || 0).toLocaleString()} source images; MVTec: ${Number(mvtec.category_count || 0)} categories.</p></section>`;
    const families = ["steel", "metal_nut", "screw", "transistor", "bottle"];
    const classes = Array.isArray(unified.classes) ? unified.classes : [];
    const profiles = unified.by_profile || {};
    const totals = unified.images || {};
    const metric = (name) => unified.test_overall?.[name] == null ? "—" : `${(Number(unified.test_overall[name]) * 100).toFixed(1)}%`;
    const familyRows = families.map((family) => {
      const counts = profiles[family] || {};
      const names = classes.filter((name) => name.startsWith(`${family}_`)).map((name) => pretty(name.slice(family.length + 1)));
      return `<tr><th scope="row">${esc(pretty(family))}</th><td>${esc(names.join(" · "))}</td><td>${Number(counts.train || 0).toLocaleString()}</td><td>${Number(counts.val || 0).toLocaleString()}</td><td>${Number(counts.test || 0).toLocaleString()}</td></tr>`;
    }).join("");
    const sourceMissing = Object.values(neu.splits || {}).reduce((sum, split) => sum + Number(split.missing || 0), 0)
      + (verification.profiles || []).reduce((sum, profile) => sum + Number(profile.missing || 0), 0);
    return `<div class="dataset-overview">
      <section class="panel card dataset-head"><div><small>PREPARED DEFECT DATASET</small><h2>Five products · ${Number(unified.class_count || classes.length)} defect classes</h2><p>Combined detection dataset used to train the unified YOLO11s candidate.</p></div><span class="dataset-status ${unified.active ? "active" : ""}">${unified.active ? "● ACTIVE DETECTOR" : "○ PREPARED, NOT ACTIVE"}</span></section>
      <div class="dataset-stats"><div class="panel mini"><small>TRAIN IMAGES</small><b>${Number(totals.train || 0).toLocaleString()}</b></div><div class="panel mini"><small>VALIDATION IMAGES</small><b>${Number(totals.val || 0).toLocaleString()}</b></div><div class="panel mini"><small>TEST IMAGES</small><b>${Number(totals.test || 0).toLocaleString()}</b></div><div class="panel mini"><small>SOURCE FILES MISSING</small><b>${sourceMissing}</b></div></div>
      <section class="panel card"><h2>Training images by product</h2><p>Counts include clean images. The steel training split also contains 420 synthetic crazing variations of existing images.</p><div class="table-scroll"><table class="dataset-table"><thead><tr><th>PRODUCT</th><th>NAMED DEFECTS</th><th>TRAIN</th><th>VAL</th><th>TEST</th></tr></thead><tbody>${familyRows}</tbody></table></div></section>
      <section class="panel card"><h2>Internal test result</h2><div class="dataset-metrics"><span><small>PRECISION</small><b>${metric("metrics/precision(B)")}</b></span><span><small>RECALL</small><b>${metric("metrics/recall(B)")}</b></span><span><small>mAP50</small><b>${metric("metrics/mAP50(B)")}</b></span><span><small>mAP50–95</small><b>${metric("metrics/mAP50-95(B)")}</b></span></div><p>Internal ${Number(unified.test_images || totals.test || 0)}-image test split. Some non-steel classes have very few test examples; this is not production-camera accuracy.</p></section>
      <section class="panel card"><h2>Source datasets</h2><div class="dataset-sources"><div><b>NEU-DET</b><p>${Number(neu.total || 0).toLocaleString()} steel images. Five classes were trained; ${Number(neu.splits?.heldout?.count || 0)} rolled-in-scale images were held out.</p></div><div><b>MVTec AD</b><p>${Number(mvtec.category_count || 0)} categories available; this detector uses metal nut, screw, transistor, and bottle. ${Number(mvtec.total_train_good || 0).toLocaleString()} good training images across the source archive.</p></div></div><p class="dataset-caveat">${esc(unified.split_note || "")}</p><details><summary>Why the previous JSON showed “detector: false”</summary><p>Those fields describe older anomaly-gate profiles. They do not describe the current unified YOLO detector. The prepared 21-class checkpoint is marked above when active.</p></details></section>
    </div>`;
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
    if (detail) detail.textContent = (frame.warnings || []).some((warning) => String(warning).includes("differs from the defect model"))
      ? "Product appearance is outside the defect model's training examples; review the image manually."
      : verdict === "pass" ? "No actionable defects found." : verdict === "fail" ? "Defect threshold exceeded; route for disposition." : "Potential defect requires confirmation.";
    if (icon) icon.textContent = verdict === "pass" ? "✓" : verdict === "fail" ? "!" : "△";

    const detections = Array.isArray(frame.detections) ? frame.detections : [];
    const product = pretty(frame.profile);
    const resultProduct = $("#resultProduct");
    const resultDefects = $("#resultDefects");
    const resultConfidence = $("#resultConfidence");
    const resultSeverity = $("#resultSeverity");
    if (resultProduct) resultProduct.textContent = product;
    if (resultDefects) resultDefects.textContent = String(detections.length);
    if (resultConfidence) resultConfidence.textContent = detections.length
      ? `${Math.max(...detections.map(confidencePercent)).toFixed(1)}%` : "—";
    const severity = Math.max(1, Math.min(4, Number(frame.severity) || 1));
    const severityNames = ["Low", "Moderate", "Major", "Critical"];
    if (resultSeverity) resultSeverity.textContent = `${severityNames[severity - 1]} · ${severity}/4`;
    const risk = frame.risk;
    const riskPanel = $("#riskPanel");
    if (riskPanel) riskPanel.classList.toggle("hidden", !risk);
    if (risk) {
      $("#riskTitle").textContent = `${pretty(risk.product)} · ${pretty(risk.defect)} · ${String(risk.decision).toUpperCase()}`;
      $("#riskConcern").textContent = risk.immediate_concern;
      $("#riskIfUsed").textContent = risk.if_used;
      $("#riskWhen").textContent = risk.time_to_failure;
      $("#riskNeeded").textContent = risk.needed_for_estimate;
      $("#riskNote").textContent = risk.note;
    }
    const plc = frame.simulated_plc;
    const plcAction = $("#plcAction"), plcTarget = $("#plcTarget"), plcReason = $("#plcReason");
    if (plcAction) plcAction.textContent = plc?.action || "—";
    if (plcTarget) plcTarget.textContent = plc?.target || "No simulated action available";
    if (plcReason) plcReason.textContent = `${plc?.reason || "No routing result"} · simulation only; no equipment command sent`;
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
        return `<div class="detection"><i></i><span><b>${esc(product)} · ${esc(pretty(name))}</b><small>YOLO confidence · ${pct}% · Severity ${Math.max(1, Math.min(4, Number(d.severity) || 1))}/4</small></span><strong>${pct}%<u><i style="width:${confidence}%"></i></u></strong></div>`;
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
    if (source) source.textContent = frame.source === "dataset"
      ? `Dataset replay · ${pretty(frame.profile)} · 10 per product` : `${pretty(frame.source)} · ${pretty(frame.profile)}`;
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
    const timeSource = $("#timeSource");
    if (timeSource) timeSource.textContent = frame.source === "dataset" ? "DATASET REPLAY" : "LIVE INPUT";
    const frameLabel = document.querySelector(".frame");
    if (frameLabel) frameLabel.textContent = `FRAME ${String(frame.frame_index ?? "—").padStart(6, "0")}`;
    const fps = $("#statFps");
    if (fps) fps.innerHTML = `${Number(frame.fps || 0).toFixed(1)} <em>FPS</em>`;
    const latency = $("#statLatency");
    if (latency) latency.innerHTML = `${Number(frame.latency_ms?.total || 0).toFixed(1)} <em>ms</em>`;
    latestSpeed = `${Number(frame.latency_ms?.total || 0).toFixed(1)} ms/frame · ${Number(frame.fps || 0).toFixed(1)} FPS measured`;
    const demoSpeed = $("#demoSpeed");
    if (demoSpeed) demoSpeed.textContent = latestSpeed;
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
      const demoModel = $("#demoModelStatus");
      if (demoModel) demoModel.textContent = `${h.detector_engine?.startsWith("yolo") ? "● REAL YOLO READY" : "● DETECTOR UNAVAILABLE"} · ${h.detector_engine || "offline"}`;
      const model = $("#modelName");
      if (model) model.innerHTML = `${esc(h.detector_engine === "detector-off" ? "Anomaly gate" : h.detector_engine || "Detector")} <small>· ${esc(h.profile || "—")}</small>`;
      const reviews = $("#reviewCount");
      if (reviews) reviews.textContent = h.db?.current_open_reviews ?? h.db?.open_reviews ?? "0";
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
      const demoModel = $("#demoModelStatus");
      if (demoModel) demoModel.textContent = "● LOCAL API OFFLINE";
    }
  }

  async function refreshReport() {
    try {
      const response = await fetch("/api/report", { cache: "no-store" });
      if (!response.ok) return;
      const report = await response.json();
      if (!initialReportRoutingDone) {
        initialReportRoutingDone = true;
        if (report.profile === "unrecognized" && (report.decisions?.review || 0) > 0 &&
            !window.processingInput && (!window.currentDashboard || window.currentDashboard === "live")) {
          window.page?.("review");
        }
      }
      if (vlmReportFilename !== report.filename) {
        vlmReportFilename = report.filename;
        $("#vlmResult")?.classList.add("hidden");
      }
      const status = $("#reportStatus"), summaryNode = $("#reportSummary");
      if (!status || !summaryNode) return;
      $("#reportCsv")?.classList.remove("hidden");
      $("#reportZip")?.classList.remove("hidden");
      status.textContent = `${report.filename} · ${pretty(report.profile)} · ${pretty(report.source)}`;
      const counts = Object.entries(report.counts_by_class || {}).map(([name, count]) => `${pretty(name)} ${count}`).join(" · ") || "no named defects";
      const decisions = report.decisions || {};
      summaryNode.textContent = `${report.inspections} stored inspections · ${report.detections} detections · ${counts} · PASS ${decisions.pass || 0} / REVIEW ${decisions.review || 0} / FAIL ${decisions.fail || 0} · ${report.avg_processing_ms} ms/inspection${report.truncated ? " · report limited to 2,000 inspections" : ""}.${report.domain_warning ? " " + report.domain_warning : ""}`;
    } catch { /* keep last report during reconnect */ }
  }

  async function refreshVlmStatus() {
    const statusNode = $("#vlmStatus"), button = $("#runVlm");
    try {
      const response = await fetch("/api/vlm/status", {cache:"no-store"});
      if (!response.ok) throw new Error("Status unavailable");
      const status = await response.json();
      vlmReady = Boolean(status.ready);
      if (statusNode) statusNode.textContent = `${status.ready ? "● Local VLM ready" : "● Local VLM unavailable"} · ${status.model} · ${status.message}`;
      if (button && button.dataset.running !== "1") button.disabled = !vlmReady;
    } catch {
      vlmReady = false;
      if (statusNode) statusNode.textContent = "● Local VLM unavailable · check the local model service";
      if (button) button.disabled = true;
    }
  }

  async function runVlmAdvice() {
    const button = $("#runVlm"), status = $("#vlmStatus");
    if (!button || !vlmReady || button.dataset.running === "1") return;
    button.dataset.running = "1";
    button.disabled = true;
    button.textContent = "Analyzing image…";
    if (status) status.textContent = "● Local VLM is reviewing the completed inspection image";
    try {
      const response = await fetch("/api/vlm/advice", {method:"POST"});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "VLM analysis failed");
      const routeName = (name) => pretty(String(name || "—").replaceAll("_", " "));
      $("#vlmRecommendation").textContent = routeName(result.recommendation);
      $("#vlmSuggestion").textContent = routeName(result.vlm_suggestion);
      $("#vlmCondition").textContent = result.visible_condition || "—";
      $("#vlmEvidence").textContent = (result.evidence || []).join(" · ") || "No visual evidence stated";
      $("#vlmRule").textContent = result.rule_reason || "—";
      $("#vlmUncertainty").textContent = result.uncertainty || "Image-only advice; verify before reuse.";
      $("#vlmMeta").textContent = `${result.model} · ${Number(result.latency_ms || 0).toFixed(0)} ms · ${result.disagreement ? "Model disagreement: human review needed" : "Advisory only"}`;
      $("#vlmResult").classList.remove("hidden");
      if (status) status.textContent = result.disagreement ? "● VLM disagrees with the rule · human review needed" : "● Local VLM analysis complete";
    } catch (error) {
      if (status) status.textContent = `● ${error.message}`;
    } finally {
      button.dataset.running = "0";
      button.disabled = !vlmReady;
      button.textContent = "Analyze completed inspection";
    }
  }

  async function hydrateDemo() {
    const host = $("#subPage");
    if (!host || window.currentDashboard !== "input") return;
    const note = host.querySelector(".input-note.caution small");
    if (note) note.textContent = "The unified YOLO detector names supported defects across all five product families. The anomaly gate is shown separately.";
    const scope = host.querySelector(".input-guidance > p");
    if (scope) scope.textContent = "Only steel, screws, metal nuts, transistors, and bottles can be inspected. Unrecognized objects go to manual review.";
    try {
      const response = await fetch("/api/demo/catalog", { cache: "no-store" });
      if (!response.ok) return;
      const catalog = await response.json();
      if (window.currentDashboard !== "input" || $("#guidedDemo")) return;
      host.insertAdjacentHTML("beforeend", `<section class="panel guided-demo" id="guidedDemo"><div class="demo-head"><div><small>GUIDED PRESENTATION</small><h2>Five-product demo</h2><p>Try a recommended defect, then a clean example for each product.</p></div><div class="demo-runtime"><b id="demoModelStatus">● Checking model…</b><span id="demoSpeed">Processing speed appears after the first frame</span></div></div><div class="demo-grid">${(catalog.items || []).map((item, index) => `<article class="demo-card"><img src="${esc(item.defect_url)}" alt="${esc(pretty(item.family))} example"><div><small>0${index + 1} · ${esc(pretty(item.family))}</small><b>${esc(pretty(item.defect_label))}</b><p>Compare a named defect with a clean sample.</p><div><button type="button" class="button primary" data-demo-family="${esc(item.family)}" data-demo-kind="defect" data-demo-name="${esc(item.defect_filename)}">Run defect</button><button type="button" class="button" data-demo-family="${esc(item.family)}" data-demo-kind="clean" data-demo-name="${esc(item.clean_filename)}">Run clean</button></div></div></article>`).join("")}</div><p class="demo-note">${esc(catalog.note)} Results depend on the running model; this is a guided demonstration.</p></section>`);
      if (latestSpeed) $("#demoSpeed").textContent = latestSpeed;
      else {
        fetch("/api/frames/latest", {cache:"no-store"}).then((response) => response.json()).then((payload) => {
          const frame = payload.frame;
          if (frame && $("#demoSpeed")) $("#demoSpeed").textContent = `${Number(frame.latency_ms?.total || 0).toFixed(1)} ms/frame · ${Number(payload.fps || frame.fps || 0).toFixed(1)} FPS measured`;
        }).catch(() => {});
      }
      refreshHealth();
    } catch { /* upload workflow remains available */ }
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

  async function hydrate(id, reviewScope = "current") {
    const host = $("#subPage");
    if (!host) return;
    host.insertAdjacentHTML("beforeend", '<p class="api-page-status">Loading live project data…</p>');
    const status = host.querySelector(".api-page-status");
    try {
      if (id === "review") {
        const q = await (await fetch(`/api/review/queue?limit=100&scope=${reviewScope}`)).json();
        host.innerHTML = `<div class="mini-grid"><div class="panel mini"><small>CURRENT UPLOAD REVIEWS</small><b>${q.open_reviews ?? q.count ?? 0}</b><span>Older records retained: ${q.historical_open_reviews ?? 0}</span></div></div><section class="panel card"><h2>Flagged inspections</h2><p>Choose a disposition to record an operator review for the current upload.</p><div class="table-scroll"><table><thead><tr><th>INSPECTION</th><th>TIME</th><th>PROFILE</th><th>DETECTION / REASON</th><th>RESULT</th><th>EVIDENCE</th><th>ACTION</th></tr></thead><tbody>${(q.items || []).map((r) => `<tr><td class="id">${esc(r.id)}</td><td>${esc(stamp(r.ts_ms))}</td><td>${esc(r.profile === "unrecognized" ? "Unrecognized product" : pretty(r.profile))}</td><td>${esc(r.reason === "product_not_recognized" ? "Product identity uncertain · no defect inference" : pretty(r.detections?.[0]?.class || "—"))}</td><td>${esc(String(r.verdict || "review").toUpperCase())}</td><td>${r.image_url ? `<a class="review-evidence" href="${esc(r.image_url)}" target="_blank" rel="noopener"><img src="${esc(r.image_url)}" alt="Uploaded inspection evidence">View image</a>` : "—"}</td><td><button class="button" data-review="${esc(r.id)}" data-review-kind="${r.profile === "unrecognized" ? "product" : "defect"}" data-review-evidence="${esc(r.image_url || "")}">Review →</button></td></tr>`).join("") || '<tr><td colspan="7">No inspections are waiting for review for this upload.</td></tr>'}</tbody></table></div></section>`;
        host.querySelector(".mini small").textContent = reviewScope === "all" ? "ALL OPEN REVIEWS" : "CURRENT UPLOAD REVIEWS";
        host.querySelector(".panel.card p").textContent = reviewScope === "all" ? "Historical pending reviews, newest first." : "Choose a disposition for the current upload.";
        host.querySelector(".panel.card p").insertAdjacentHTML("afterend", `<button type="button" class="button review-scope" data-review-scope="${reviewScope === "all" ? "current" : "all"}">${reviewScope === "all" ? "Show current upload" : "Show historical queue"}</button>`);
        host.insertAdjacentHTML("beforeend", `<dialog class="review-dialog" id="reviewDialog" aria-labelledby="reviewDialogTitle"><h2 id="reviewDialogTitle">Review inspection</h2><p id="reviewInspection"></p><img id="reviewEvidence" class="review-dialog-image hidden" alt="Uploaded inspection evidence"><label>Disposition<select id="reviewDisposition"><option value="true_defect">True defect</option><option value="false_alarm">False alarm</option><option value="reclassify">Reclassify</option></select></label><label id="reviewLabelWrap" class="hidden"><span id="reviewLabelTitle">Correct defect label</span><input id="reviewLabel" maxlength="80" placeholder="e.g. scratch"></label><label>Note (optional)<textarea id="reviewNote" rows="3" maxlength="500" placeholder="What did you observe?"></textarea></label><p class="review-error hidden" id="reviewError" role="alert"></p><div class="review-dialog-actions"><button type="button" class="button" data-review-cancel>Cancel</button><button type="button" class="button primary" data-review-save>Save review</button></div></dialog>`);
        $("#reviewDialog").dataset.scope = reviewScope;
      } else if (id === "metrics") {
        const m = await (await fetch("/api/metrics")).json();
        host.innerHTML = `<div class="mini-grid"><div class="panel mini"><small>INSPECTIONS</small><b>${m.total ?? 0}</b><span>Stored records</span></div><div class="panel mini"><small>PASS RATE</small><b>${((m.pass_rate || 0) * 100).toFixed(1)}%</b><span>Current database</span></div><div class="panel mini"><small>OPEN REVIEWS</small><b>${m.counts?.open_reviews ?? 0}</b><span>Awaiting disposition</span></div><div class="panel mini"><small>LATENCY P95</small><b>${Number(m.latency?.p95 || 0).toFixed(1)} ms</b><span>Measured pipeline</span></div></div><section class="panel card"><h2>Detections by class</h2><p>Counts returned by the inspection metrics API</p>${renderDetectionChart(m.classes)}</section>`;
      } else if (id === "datasets") {
        const [verification, unified] = await Promise.all([
          fetch("/api/datasets/verify").then((response) => response.json()),
          fetch("/api/datasets/unified").then((response) => response.json()),
        ]);
        host.innerHTML = renderDatasetOverview(verification, unified);
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
    if (id === "processing") $("#vlmResult")?.classList.add("hidden");
    if (["review","metrics","datasets","audit","settings"].includes(id)) hydrate(id);
    if (id === "input") hydrateDemo();
  };

  document.addEventListener("click", async (event) => {
    if (event.target.closest("#runVlm")) {
      event.preventDefault(); event.stopImmediatePropagation();
      runVlmAdvice();
      return;
    }
    const demoButton = event.target.closest("[data-demo-family][data-demo-kind]");
    if (demoButton) {
      event.preventDefault(); event.stopImmediatePropagation();
      demoButton.disabled = true;
      try {
        const response = await fetch(`/api/demo/image/${encodeURIComponent(demoButton.dataset.demoFamily)}/${encodeURIComponent(demoButton.dataset.demoKind)}`);
        if (!response.ok) throw new Error("Demo image unavailable");
        const file = new File([await response.blob()], demoButton.dataset.demoName || "demo.jpg", {type:"image/jpeg"});
        updateInputSelection(file, false);
        processSelectedInput();
      } catch (error) { window.toast?.(error.message); demoButton.disabled = false; }
      return;
    }
    const review = event.target.closest("[data-review]");
    if (review) {
      event.preventDefault(); event.stopImmediatePropagation();
      const dialog = $("#reviewDialog");
      if (!dialog) return;
      dialog.dataset.inspectionId = review.dataset.review;
      $("#reviewInspection").textContent = `Inspection ${review.dataset.review}`;
      const evidence = $("#reviewEvidence");
      if (evidence) {
        evidence.classList.toggle("hidden", !review.dataset.reviewEvidence);
        if (review.dataset.reviewEvidence) evidence.src = review.dataset.reviewEvidence;
        else evidence.removeAttribute("src");
      }
      const productReview = review.dataset.reviewKind === "product";
      $("#reviewDisposition").value = productReview ? "reclassify" : "true_defect";
      $("#reviewLabel").value = "";
      $("#reviewLabel").placeholder = productReview ? "Product family or unsupported" : "e.g. scratch";
      $("#reviewLabelTitle").textContent = productReview ? "Identified product" : "Correct defect label";
      $("#reviewNote").value = "";
      $("#reviewLabelWrap").classList.toggle("hidden", !productReview);
      $("#reviewDialogTitle").textContent = productReview ? "Review product identity" : "Review inspection";
      $("#reviewError").classList.add("hidden");
      dialog.showModal();
      return;
    }
    const reviewScope = event.target.closest("[data-review-scope]");
    if (reviewScope) {
      event.preventDefault(); event.stopImmediatePropagation();
      hydrate("review", reviewScope.dataset.reviewScope);
      return;
    }
    if (event.target.closest("[data-review-cancel]")) {
      event.preventDefault(); event.stopImmediatePropagation();
      $("#reviewDialog")?.close();
      return;
    }
    if (event.target.closest("[data-review-save]")) {
      event.preventDefault(); event.stopImmediatePropagation();
      const dialog = $("#reviewDialog"), save = event.target.closest("[data-review-save]");
      const disposition = $("#reviewDisposition")?.value;
      const label = $("#reviewLabel")?.value.trim();
      const error = $("#reviewError");
      if (disposition === "reclassify" && !label) {
        error.textContent = "Enter the correct defect label.";
        error.classList.remove("hidden");
        return;
      }
      save.disabled = true;
      try {
        const response = await fetch(`/api/inspections/${encodeURIComponent(dialog.dataset.inspectionId)}/review`, {
          method:"POST", headers:{"Content-Type":"application/json"},
          body:JSON.stringify({disposition, label:disposition === "reclassify" ? label : null, note:$("#reviewNote")?.value.trim() || null}),
        });
        if (!response.ok) throw new Error((await response.json()).detail || "Review failed");
        dialog.close();
        window.toast?.("Review saved");
        hydrate("review", dialog.dataset.scope || "current");
      } catch (err) {
        error.textContent = err.message;
        error.classList.remove("hidden");
      } finally { save.disabled = false; }
      return;
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
    if (event.target.closest("#replay")) {
      event.preventDefault();
      event.stopPropagation();
      (async () => {
        const send = async (action, value) => {
          const response = await fetch("/api/control", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({action, value})});
          if (!response.ok) throw new Error("Could not start dataset replay");
        };
        try {
          await send("resume");
          await send("source", "dataset");
          const pause = $("#pause");
          if (pause) { pause.dataset.paused = "0"; pause.textContent = "Ⅱ Pause stream"; }
          window.toast?.("Ten samples per product replay started");
        } catch (error) { window.toast?.(error.message); }
      })();
      return;
    }
    if (event.target.closest("#pause")) {
      const paused = event.target.closest("#pause").dataset.paused === "1";
      fetch("/api/control",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action:paused?"resume":"pause"})}).catch(()=>{});
    }
  }, true);

  document.addEventListener("change", (event) => {
    if (event.target.id === "reviewDisposition") {
      $("#reviewLabelWrap")?.classList.toggle("hidden", event.target.value !== "reclassify");
      $("#reviewError")?.classList.add("hidden");
    }
  });

  const originalConnect = window.connect;
  refreshHealth(); refreshReport(); refreshVlmStatus(); hydrateDemo(); setInterval(refreshHealth, 5000); setInterval(refreshReport, 5000); setInterval(refreshVlmStatus, 15000);
  document.addEventListener("DOMContentLoaded", () => {
    document.body.classList.remove("auth-locked");
    $("#overlayBoxes")?.classList.add("hidden");
    const stats = document.querySelectorAll("#livePage .stats .mini, #livePage .stats>div");
    if (stats.length) refreshHealth();
  });
})();
