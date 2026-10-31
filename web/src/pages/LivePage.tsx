import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Clock,
  Cpu,
  Eye,
  Gauge,
  Layers,
  Pause,
  Play,
  Radio,
  RefreshCw,
  ScanLine,
  ShieldAlert,
  ShieldCheck,
  SkipForward,
  SlidersHorizontal,
  WifiOff,
  XCircle,
  Zap,
} from "lucide-react";
import { api } from "../lib/api";
import { fmtMs, fmtPct, fmtRelative, fmtTime, pctWidth, shortId, titleCase } from "../lib/format";
import type { Detection, FrameResult, Inspection, ProfileInfo } from "../lib/types";
import { useLive } from "../hooks/useLive";
import { useNow, usePolling } from "../hooks/usePolling";
import {
  Badge,
  Empty,
  ErrorNote,
  InfoNote,
  Panel,
  ProgressBar,
  SeverityMeter,
  Spinner,
  VerdictBadge,
  VerdictChip,
  WarningList,
} from "../components/ui";

const CLASS_COLOR: Record<string, string> = {
  crazing: "#7dd3fc",
  inclusion: "#f472b6",
  patches: "#fbbf24",
  pitted_surface: "#a78bfa",
  scratches: "#fb7185",
  "rolled-in_scale": "#f97316",
  unknown: "#f59e0b",
};

function DetectionRow({ detection }: { detection: Detection }) {
  return (
    <li className="det-row">
      <span className="det-swatch" style={{ background: CLASS_COLOR[detection.class] ?? "#94a3b8" }} />
      <div className="det-main">
        <div className="det-title">
          <strong>{titleCase(detection.class)}</strong>
          {detection.source === "fixture" && <Badge tone="warn">fixture</Badge>}
        </div>
        <div className="det-meta">
          conf {(detection.conf * 100).toFixed(1)}% · area {fmtPct(detection.area_frac)} · votes {detection.votes}
        </div>
      </div>
      <SeverityMeter value={detection.severity} compact />
    </li>
  );
}

function LatencyRow({ label, value, max, accent }: { label: string; value: number; max: number; accent?: boolean }) {
  return (
    <div className={`lat-row ${accent ? "lat-accent" : ""}`}>
      <span className="lat-label">{label}</span>
      <div className="lat-track">
        <span style={{ width: pctWidth(max > 0 ? value / max : 0) }} />
      </div>
      <span className="lat-value">{fmtMs(value)}</span>
    </div>
  );
}

function RecentCard({ inspection }: { inspection: Inspection }) {
  const image = inspection.overlay_url || inspection.image_url;
  const top = inspection.detections?.[0];
  return (
    <Link to={`/review?id=${inspection.id}`} className="recent-card">
      <div className="recent-thumb">
        {image ? <img src={image} alt={`Inspection ${inspection.id}`} loading="lazy" /> : <ScanLine size={22} />}
        <VerdictChip verdict={inspection.final_verdict || inspection.verdict} />
      </div>
      <div className="recent-meta">
        <span className="recent-time">{fmtTime(inspection.ts_ms)}</span>
        <span className="recent-class">{top ? titleCase(top.class) : titleCase(inspection.reason)}</span>
        <span className="recent-sev">
          <SeverityMeter value={inspection.severity} compact />
          {inspection.operator_disposition ? <Badge tone="info">reviewed</Badge> : null}
        </span>
      </div>
    </Link>
  );
}

function GateCard({ frame }: { frame: FrameResult }) {
  const anomaly = frame.anomaly;
  const trusted = anomaly.gate_trusted;
  return (
    <Panel
      title="Novelty Gate"
      subtitle="PatchCore-lite · per-profile memory bank"
      actions={<Badge tone={trusted ? "good" : "warn"} dot>{trusted ? "TRUSTED" : "UNTRUSTED"}</Badge>}
    >
      <div className="gate-score-row">
        <div className={`gate-score ${anomaly.is_anomaly ? "gate-score-hot" : ""}`}>
          <span className="gate-z">{anomaly.z.toFixed(2)}</span>
          <span className="gate-z-label">z-score</span>
        </div>
        <div className="gate-facts">
          <div>
            <span>Raw distance</span>
            <strong>{anomaly.raw_score.toFixed(4)}</strong>
          </div>
          <div>
            <span>Threshold</span>
            <strong>{anomaly.threshold.toFixed(2)}</strong>
          </div>
          <div>
            <span>Quality</span>
            <strong className={trusted ? "text-good" : "text-warn"}>{titleCase(anomaly.gate_quality)}</strong>
          </div>
          <div>
            <span>Separation</span>
            <strong>{anomaly.separation === null ? "—" : anomaly.separation.toFixed(2)}</strong>
          </div>
        </div>
      </div>
      <div className="gate-bar">
        <span
          className={anomaly.is_anomaly ? "hot" : ""}
          style={{ width: `${Math.max(2, Math.min(100, (anomaly.z / Math.max(1.2, anomaly.threshold * 2)) * 100))}%` }}
        />
        <i style={{ left: `${Math.min(98, (anomaly.threshold / Math.max(1.2, anomaly.threshold * 2)) * 100)}%` }} />
      </div>
      <div className="gate-bar-labels">
        <span>0.0</span>
        <span>threshold {anomaly.threshold.toFixed(2)}</span>
        <span>{(Math.max(1.2, anomaly.threshold * 2)).toFixed(1)}+</span>
      </div>
      {!trusted && (
        <InfoNote tone="warn">
          Gate separation failed validation on held-out unknowns. It is computed for transparency but does not drive
          this profile's verdict.
        </InfoNote>
      )}
      <div className="gate-links">
        <span className="mono">{anomaly.model}</span>
        {anomaly.cached && <Badge tone="muted">cached</Badge>}
        {anomaly.heatmap_url && (
          <a href={anomaly.heatmap_url} target="_blank" rel="noreferrer" className="text-link">
            heatmap ↗
          </a>
        )}
      </div>
    </Panel>
  );
}

export default function LivePage() {
  const { frame, fps, health, control, connected, error, refreshControl } = useLive();
  const now = useNow(1000);
  const profiles = usePolling<ProfileInfo[]>(api.profiles, 30000);
  const recent = usePolling(() => api.inspections({ limit: 10 }), 5000);
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [streamError, setStreamError] = useState(false);
  const [streamRetry, setStreamRetry] = useState(0);
  const [showRecent, setShowRecent] = useState(true);
  const [inputName, setInputName] = useState<string | null>(null);

  useEffect(() => {
    setStreamError(false);
  }, [control?.profile]);

  const command = async (action: string, value?: string) => {
    setBusy(action);
    setActionError(null);
    try {
      await api.control(action, value);
      await refreshControl();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const selectInput = async (file?: File) => {
    if (!file) return;
    setBusy("upload");
    setActionError(null);
    try {
      await api.uploadInput(file);
      setInputName(file.name);
      await refreshControl();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const paused = control?.paused ?? false;
  const detectorModes = control?.detector_modes ?? ["auto", "real", "fixture", "off"];
  const sequenceModes = control?.sequence_modes ?? ["mixed", "good", "defect", "unknown"];
  const source = control?.source ?? frame?.source ?? "dataset";
  const provisional = frame ? frame.part_state.frame_in_part < frame.part_state.frames_in_part : false;
  const fixture =
    frame?.engine?.includes("fixture") === true ||
    health?.detector_mode === "fixture" ||
    control?.detector?.warning?.includes("fixture") === true;
  const route = frame?.routing?.destination ?? (frame?.verdict === "pass" ? "release" : "pending");
  const reason = frame?.routing?.reason ?? (frame ? "no_known_defect" : "waiting");
  const detections = frame?.detections ?? [];
  const maxSev = detections.reduce((max, det) => Math.max(max, det.severity), 1);
  const latency = frame?.latency_ms;
  const maxLatency = latency ? Math.max(latency.infer_detector, latency.infer_gate, latency.post, 1) : 1;
  const pipelineReady = !!frame;
  const currentProfile = profiles.data?.find((p) => p.id === (control?.profile ?? health?.profile));
  const recentItems = recent.data?.items ?? [];

  const stageTags = useMemo(
    () => ({
      simulated: source === "dataset",
      fixture,
    }),
    [source, fixture],
  );

  return (
    <div className="live-page">
      <div className="control-strip">
        <div className="control-group">
          <button
            className={`btn ${paused ? "btn-primary" : "btn-ghost"}`}
            onClick={() => void command(paused ? "resume" : "pause")}
            disabled={busy !== null}
          >
            {paused ? <Play size={15} /> : <Pause size={15} />}
            {paused ? "Resume" : "Pause"}
          </button>
          <button
            className="btn btn-ghost"
            onClick={() => void command("skip")}
            disabled={busy !== null || source !== "dataset"}
            title="Abandon the current part and advance the replay"
          >
            <SkipForward size={15} /> Skip part
          </button>
          <button className="btn btn-ghost" onClick={() => void refreshControl()} disabled={busy !== null}>
            <RefreshCw size={15} className={busy === "refresh" ? "spin" : ""} /> Sync
          </button>
        </div>

        <div className="control-divider" />

        <div className="control-group control-selects">
          <label>
            <span>Profile</span>
            <select
              value={control?.profile ?? ""}
              disabled={busy !== null || !profiles.data}
              onChange={(event) => void command("profile", event.target.value)}
            >
              {profiles.data?.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.display_name}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Sequence</span>
            <select
              value={control?.sequence ?? "mixed"}
              disabled={busy !== null || source !== "dataset"}
              onChange={(event) => void command("sequence", event.target.value)}
            >
              {sequenceModes.map((mode) => (
                <option key={mode} value={mode}>
                  {titleCase(mode)}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Detector</span>
            <select
              value={health?.detector_mode ?? "auto"}
              disabled={busy !== null}
              onChange={(event) => void command("detector", event.target.value)}
            >
              {detectorModes.map((mode) => (
                <option key={mode} value={mode}>
                  {titleCase(mode)}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Input</span>
            <select
              value={source}
              disabled={busy !== null}
              onChange={(event) => void command("source", event.target.value)}
            >
              <option value="dataset">Dataset replay</option>
              <option value="video">Video</option>
              <option value="image">Image</option>
              <option value="camera">Camera</option>
            </select>
          </label>
          <label className="video-picker">
            <span>{busy === "upload" ? "Loading file…" : inputName || "Choose image or video"}</span>
            <input
              type="file"
              accept="image/jpeg,image/png,image/bmp,image/webp,video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,video/webm,.m4v"
              disabled={busy !== null}
              onChange={(event) => {
                void selectInput(event.currentTarget.files?.[0]);
                event.currentTarget.value = "";
              }}
            />
          </label>
        </div>

        <div className="control-status">
          <span className={`status-dot ${connected ? "on" : "off"}`} />
          <span>{connected ? "stream linked" : "polling snapshot"}</span>
          {control?.detector?.warning && <Badge tone="warn">{humanEngine(control.detector.engine)}</Badge>}
        </div>
      </div>

      {actionError && <ErrorNote>{actionError}</ErrorNote>}
      {error && !actionError && <ErrorNote>{error}</ErrorNote>}

      <div className="live-grid">
        <div className="live-main">
          <div className={`stage stage-${frame?.verdict ?? "idle"}`}>
            <div className="stage-frame">
              {!streamError ? (
                <img
                  key={`${control?.profile}-${streamRetry}`}
                  className="stage-video"
                  src="/live.mjpg"
                  alt="Live inspection stream with detector and novelty overlays"
                  onError={() => setStreamError(true)}
                  onLoad={() => setStreamError(false)}
                />
              ) : frame?.overlay_url ? (
                <img className="stage-video" src={frame.overlay_url} alt="Latest committed inspection overlay" />
              ) : (
                <div className="stage-placeholder">
                  <WifiOff size={28} />
                  <strong>Stream unavailable</strong>
                  <span>The API is still reachable; waiting for the next rendered frame.</span>
                  <button
                    className="btn btn-ghost"
                    onClick={() => {
                      setStreamRetry((value) => value + 1);
                      setStreamError(false);
                    }}
                  >
                    <RefreshCw size={14} /> Retry MJPEG
                  </button>
                </div>
              )}

              <div className="stage-grid" aria-hidden="true" />
              <div className="stage-sweep" aria-hidden="true" />
              <span className="corner corner-tl" aria-hidden="true" />
              <span className="corner corner-tr" aria-hidden="true" />
              <span className="corner corner-bl" aria-hidden="true" />
              <span className="corner corner-br" aria-hidden="true" />

              <div className="stage-top">
                <div className="stage-cam">
                  <span className={`rec-dot ${paused ? "paused" : ""}`} />
                  <span>{source === "dataset" ? "REPLAY · CAM 01" : source === "video" ? "VIDEO · DEMO CLIP" : source === "image" ? "IMAGE · INSPECTION" : "CAM 01"}</span>
                  <span className="stage-divider" />
                  <span className="mono">{frame?.stream_id ?? "cam-1"}</span>
                </div>
                <div className="stage-clock">
                  <Clock size={13} />
                  {frame ? fmtTime(frame.ts_ms) : fmtTime(now)}
                </div>
              </div>

              <div className="stage-bottom">
                <div className="stage-part">
                  <span className="stage-eyebrow">
                    {frame ? `PART ${String(frame.part_state.index).padStart(4, "0")}` : "PART ----"}
                    {frame && frame.part_state.total > 0 && <em> / {frame.part_state.total}</em>}
                  </span>
                  <strong>{frame ? titleCase(frame.part_state.kind) : "Awaiting replay"}</strong>
                  <div className="stage-progress">
                    <span
                      style={{
                        width: pctWidth(
                          frame && frame.part_state.frames_in_part
                            ? frame.part_state.frame_in_part / frame.part_state.frames_in_part
                            : 0,
                        ),
                      }}
                    />
                  </div>
                  <span className="stage-caption">
                    {frame
                      ? `${frame.part_state.frame_in_part}/${frame.part_state.frames_in_part} frames · ${frame.threshold_profile}`
                      : "connecting to pipeline"}
                  </span>
                </div>
                <div className="stage-verdict-mini">
                  <span className={`mini-label ${frame?.verdict ?? "idle"}`}>{frame ? frame.verdict.toUpperCase() : "IDLE"}</span>
                  <span className="mono">z {frame ? frame.anomaly.z.toFixed(2) : "—"}</span>
                  <span className="mono">{frame ? fmtMs(frame.latency_ms.total) : "—"}</span>
                </div>
              </div>

              <div className="stage-tags">
                {stageTags.simulated && (
                  <span className="stage-tag stage-tag-sim">
                    <Radio size={12} /> SIMULATED FEED
                  </span>
                )}
                {stageTags.fixture && (
                  <span className="stage-tag stage-tag-fixture">
                    <AlertTriangle size={12} /> FIXTURE GT BOXES
                  </span>
                )}
                {frame?.warnings?.includes("gate_not_validated") && (
                  <span className="stage-tag stage-tag-gate">
                    <ShieldAlert size={12} /> GATE UNTRUSTED
                  </span>
                )}
              </div>

              {!pipelineReady && (
                <div className="stage-wait">
                  <Spinner label="Waiting for the first inspection frame" />
                </div>
              )}
            </div>

            <div className="stage-footer">
              <div className="stage-foot-item">
                <Activity size={14} />
                <span>{fps.toFixed(1)} fps</span>
              </div>
              <div className="stage-foot-item">
                <Cpu size={14} />
                <span>{frame?.device ?? health?.device ?? "—"}</span>
              </div>
              <div className="stage-foot-item stage-foot-wide">
                <Layers size={14} />
                <span>{frame?.engine ?? health?.detector_engine ?? "awaiting engine"}</span>
              </div>
              <div className="stage-foot-item">
                <Eye size={14} />
                <span>frame {frame?.frame_index ?? 0}</span>
              </div>
              <div className="stage-foot-item">
                <CircleDot size={14} />
                <span>{frame?.part_state.ground_truth_class ? `GT ${titleCase(frame.part_state.ground_truth_class)}` : "GT hidden"}</span>
              </div>
            </div>
          </div>

          <div className="live-under">
            <Panel
              title="Fusion decision"
              subtitle="Deterministic rules over detector evidence and gate novelty"
              actions={<Badge tone="info">{frame?.threshold_profile ?? "threshold_v1"}</Badge>}
            >
              <div className="factor-grid">
                <div className="factor">
                  <span className="factor-icon"><ScanLine size={16} /></span>
                  <div>
                    <span>Detector evidence</span>
                    <strong>
                      {detections.length
                        ? `${detections.length} class${detections.length > 1 ? "es" : ""} · severity ${maxSev}`
                        : "No supported defect class"}
                    </strong>
                  </div>
                </div>
                <div className="factor">
                  <span className="factor-icon"><Gauge size={16} /></span>
                  <div>
                    <span>Novelty gate</span>
                    <strong>
                      z {frame?.anomaly.z.toFixed(2) ?? "—"} · {frame?.anomaly.gate_trusted ? "trusted" : "untrusted"}
                    </strong>
                  </div>
                </div>
                <div className="factor">
                  <span className="factor-icon"><SlidersHorizontal size={16} /></span>
                  <div>
                    <span>Route</span>
                    <strong>{titleCase(route)}</strong>
                  </div>
                </div>
              </div>
              {frame && (
                <div className="decision-summary">
                  <span className="decision-arrow">→</span>
                  <p>
                    <strong>{titleCase(reason)}.</strong>{" "}
                    {frame.verdict === "pass"
                      ? "No supported defect class and no trusted novelty escalation. Part is released."
                      : frame.verdict === "review"
                        ? "Ambiguous evidence is escalated rather than silently released."
                        : "A supported defect class crossed the operating confidence floor. Part is held."}
                  </p>
                </div>
              )}
            </Panel>

            <Panel
              title="Detections"
              subtitle={`${detections.length} supported track${detections.length === 1 ? "" : "s"} on this part`}
              actions={<WarningList warnings={frame?.warnings ?? []} />}
            >
              {detections.length ? (
                <ul className="det-list">
                  {detections.map((detection) => (
                    <DetectionRow key={detection.id} detection={detection} />
                  ))}
                </ul>
              ) : (
                <Empty
                  icon={<CheckCircle2 size={22} />}
                  title="No supported detections"
                  hint="The detector must see a class across enough frames and above the confidence floor before it is reported."
                />
              )}
              {frame?.warnings?.includes("low_confidence_detection") && (
                <InfoNote tone="warn">At least one track fell below the confidence floor and was not reported as a defect.</InfoNote>
              )}
              {frame?.warnings?.includes("transient_detection") && (
                <InfoNote tone="warn">A track lacked support across frames. Single-frame flicker is suppressed.</InfoNote>
              )}
            </Panel>
          </div>
        </div>

        <aside className="live-rail">
          <div className={`verdict-hero ${frame?.verdict ?? "idle"}`}>
            <div className="verdict-hero-head">
              <span>Part verdict</span>
              <Badge tone={provisional ? "warn" : "neutral"} dot>
                {provisional ? "ACCUMULATING" : "FINAL"}
              </Badge>
            </div>
            <VerdictBadge verdict={frame?.verdict} size="hero" provisional={provisional} />
            <div className="verdict-hero-grid">
              <div>
                <span>Routing</span>
                <strong>{titleCase(route)}</strong>
              </div>
              <div>
                <span>Reason</span>
                <strong>{titleCase(reason)}</strong>
              </div>
              <div>
                <span>Severity</span>
                <SeverityMeter value={frame?.severity ?? 1} />
              </div>
              <div>
                <span>Frames</span>
                <strong>
                  {frame ? `${frame.part_state.frame_in_part}/${frame.part_state.frames_in_part}` : "—"}
                </strong>
              </div>
            </div>
            {frame && (
              <div className="verdict-foot">
                <span className="mono">{shortId(frame.inspection_id, 18)}</span>
                <span>{fmtRelative(frame.ts_ms)}</span>
              </div>
            )}
          </div>

          <GateCard frame={frame ?? emptyFrame()} />

          <Panel
            title="Part record"
            subtitle="Replay labels are shown for honesty; the operator does not see them in production."
          >
            <dl className="kv">
              <div>
                <dt>Part</dt>
                <dd>{frame ? `${frame.part_state.index} / ${frame.part_state.total || "—"}` : "—"}</dd>
              </div>
              <div>
                <dt>Inspection</dt>
                <dd className="mono">{frame ? shortId(frame.inspection_id, 16) : "—"}</dd>
              </div>
              <div>
                <dt>Replay kind</dt>
                <dd>{titleCase(frame?.part_state.kind)}</dd>
              </div>
              <div>
                <dt>Ground truth</dt>
                <dd className={frame?.part_state.ground_truth_class ? "text-warn" : ""}>
                  {frame?.part_state.ground_truth_class ? titleCase(frame.part_state.ground_truth_class) : "not provided"}
                </dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd className="mono break-all">{frame?.model_version ?? "—"}</dd>
              </div>
              <div>
                <dt>Profile</dt>
                <dd>
                  {currentProfile?.display_name ?? titleCase(control?.profile ?? health?.profile)}{" "}
                  {currentProfile && !currentProfile.detector && <Badge tone="info">gate-only</Badge>}
                </dd>
              </div>
            </dl>
            {currentProfile?.heldout_class && (
              <InfoNote tone="warn">
                Held-out class <strong>{titleCase(currentProfile.heldout_class)}</strong> is excluded from detector
                training and scored only through the novelty gate.
              </InfoNote>
            )}
            <div className="record-links">
              <Link to="/datasets" className="text-link">
                dataset provenance <ArrowRight size={13} />
              </Link>
              <Link to="/settings" className="text-link">
                thresholds <ArrowRight size={13} />
              </Link>
            </div>
          </Panel>

          <Panel
            title="Latency"
            subtitle="Measured per pipeline stage on this frame"
            actions={
              <Badge tone={(latency?.total ?? 0) < 120 ? "good" : (latency?.total ?? 0) < 250 ? "warn" : "bad"}>
                {fmtMs(latency?.total)}
              </Badge>
            }
          >
            <div className="lat-list">
              <LatencyRow label="Detector" value={latency?.infer_detector ?? 0} max={maxLatency} accent />
              <LatencyRow label="Gate" value={latency?.infer_gate ?? 0} max={maxLatency} />
              <LatencyRow label="Post / encode" value={(latency?.post ?? 0) + (latency?.encode ?? 0)} max={maxLatency} />
              <LatencyRow label="Total" value={latency?.total ?? 0} max={Math.max(maxLatency, latency?.total ?? 0)} />
            </div>
            <div className="lat-foot">
              <span><Zap size={13} /> {fps.toFixed(1)} fps sustained</span>
              <span><Clock size={13} /> gate every {frame?.anomaly.cached ? "reused" : "frame"}</span>
            </div>
          </Panel>
        </aside>
      </div>

      <section className="recent-section">
        <div className="section-head">
          <div>
            <h2>Recent parts</h2>
            <p>Committed inspections with evidence stills. Open a card to review or inspect.</p>
          </div>
          <div className="section-actions">
            <button className="btn btn-ghost" onClick={() => setShowRecent((value) => !value)}>
              {showRecent ? "Collapse" : "Expand"}
            </button>
            <Link className="btn btn-ghost" to="/review">
              Queue <ChevronRight size={14} />
            </Link>
          </div>
        </div>
        {showRecent &&
          (recent.loading ? (
            <Spinner label="Loading recent inspections" />
          ) : recent.error ? (
            <ErrorNote>{recent.error}</ErrorNote>
          ) : recentItems.length ? (
            <div className="recent-strip">
              {recentItems.map((inspection) => (
                <RecentCard key={inspection.id} inspection={inspection} />
              ))}
            </div>
          ) : (
            <Empty icon={<ScanLine size={22} />} title="No completed parts yet" hint="The pipeline commits a part after its replay frames are fused." />
          ))}
      </section>
    </div>
  );
}

function humanEngine(engine: string): string {
  if (engine.includes("fixture")) return "fixture detector";
  return engine;
}

function emptyFrame(): FrameResult {
  return {
    schema_version: "1.0",
    frame_id: "empty",
    inspection_id: "",
    ts_ms: 0,
    stream_id: "cam-1",
    profile: "steel",
    source: "dataset",
    device: "—",
    engine: "—",
    model_version: "—",
    threshold_profile: "threshold_v1",
    latency_ms: { capture: 0, preprocess: 0, infer_detector: 0, infer_gate: 0, post: 0, encode: 0, total: 0 },
    detections: [],
    anomaly: {
      raw_score: 0,
      z: 0,
      threshold: 0.75,
      is_anomaly: false,
      model: "patchcore-lite/none",
      gate_trusted: false,
      gate_quality: "unknown",
      separation: null,
      regions: [],
      heatmap_url: null,
      cached: false,
    },
    verdict: "pass",
    routing: null,
    severity: 1,
    part_state: {
      index: 0,
      total: 0,
      item_id: "",
      kind: "waiting",
      ground_truth_class: null,
      ground_truth_kind: null,
      frame_in_part: 0,
      frames_in_part: 0,
    },
    image_url: null,
    overlay_url: null,
    crop_url: null,
    fps: 0,
    frame_index: 0,
    warnings: [],
  };
}
