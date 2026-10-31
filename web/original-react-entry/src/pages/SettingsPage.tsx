import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  Cpu,
  Database,
  Download,
  FileJson,
  Fingerprint,
  Gauge,
  HardDrive,
  Info,
  Layers,
  RefreshCw,
  RotateCcw,
  Save,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  Terminal,
} from "lucide-react";
import { api, exportCsvUrl } from "../lib/api";
import { fmtBytes, fmtDateTime, fmtNum, fmtRelative, titleCase } from "../lib/format";
import type { GateSummary, ModelInfo, Threshold } from "../lib/types";
import { usePolling } from "../hooks/usePolling";
import { Badge, Empty, ErrorNote, InfoNote, Panel, Spinner } from "../components/ui";

interface ThresholdForm {
  name: string;
  pass_max: string;
  review_max: string;
  unknown_z: string;
  conf_min: string;
}

const DEFAULTS: ThresholdForm = {
  name: "Production v1",
  pass_max: "0.55",
  review_max: "0.75",
  unknown_z: "0.75",
  conf_min: "0.35",
};

const ENV_ROWS: { key: string; label: string; hint: string }[] = [
  { key: "profile", label: "INDORE_PROFILE", hint: "Active replay profile" },
  { key: "source", label: "INDORE_SOURCE", hint: "dataset / camera / video / image" },
  { key: "detector_mode", label: "INDORE_DETECTOR", hint: "auto / real / fixture / off" },
  { key: "device", label: "INDORE_DEVICE", hint: "auto / cpu / cuda / 0" },
  { key: "imgsz", label: "INDORE_IMGSZ", hint: "Detector input size" },
  { key: "conf", label: "INDORE_CONF", hint: "Raw detector floor (hard pre-filter)" },
  { key: "iou", label: "INDORE_IOU", hint: "NMS IoU" },
  { key: "part_frames", label: "INDORE_PART_FRAMES", hint: "Replay frames fused per part" },
  { key: "target_fps", label: "INDORE_TARGET_FPS", hint: "Loop rate cap" },
  { key: "stream_size", label: "INDORE_STREAM_SIZE", hint: "MJPEG render size" },
  { key: "jpeg_quality", label: "INDORE_JPEG_QUALITY", hint: "MJPEG JPEG quality" },
  { key: "gate_every", label: "INDORE_GATE_EVERY", hint: "Gate interval; 0 = adaptive" },
  { key: "actor", label: "INDORE_ACTOR", hint: "Audit actor identity" },
  { key: "db_path", label: "INDORE_DB", hint: "SQLite path" },
  { key: "mes_configured", label: "INDORE_MES_URL", hint: "Signed webhook target" },
];

export default function SettingsPage() {
  const thresholds = usePolling<Threshold>(api.thresholds, 0);
  const models = usePolling<ModelInfo>(api.models, 30000);
  const gate = usePolling<GateSummary>(api.gate, 30000);
  const config = usePolling<Record<string, any>>(api.config, 0);
  const [form, setForm] = useState<ThresholdForm>(DEFAULTS);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);

  useEffect(() => {
    if (!thresholds.data || dirty) return;
    setForm({
      name: thresholds.data.name,
      pass_max: String(thresholds.data.pass_max),
      review_max: String(thresholds.data.review_max),
      unknown_z: String(thresholds.data.unknown_z),
      conf_min: String(thresholds.data.conf_min),
    });
  }, [thresholds.data, dirty]);

  const update = (key: keyof ThresholdForm, value: string) => {
    setForm((current) => ({ ...current, [key]: value }));
    setDirty(true);
    setFlash(null);
  };

  const validation = useMemo(() => {
    const pass = Number(form.pass_max);
    const review = Number(form.review_max);
    const unknown = Number(form.unknown_z);
    const conf = Number(form.conf_min);
    if ([pass, review, unknown, conf].some((value) => Number.isNaN(value) || value < 0 || value > 3)) {
      return "All thresholds must be numbers between 0 and 3.";
    }
    if (!(pass > 0 && pass < review)) return "pass_max must be greater than 0 and below review_max.";
    if (review > unknown) return "review_max must be less than or equal to unknown_z.";
    return null;
  }, [form]);

  const save = async () => {
    if (validation) {
      setError(validation);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const next = await api.updateThresholds({
        name: form.name,
        pass_max: Number(form.pass_max),
        review_max: Number(form.review_max),
        unknown_z: Number(form.unknown_z),
        conf_min: Number(form.conf_min),
      });
      thresholds.setData(next);
      setDirty(false);
      setFlash("Thresholds saved. Fusion reads them at the next part finalize.");
      await models.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const restore = () => {
    setForm(DEFAULTS);
    setDirty(true);
    setError(null);
    setFlash("Defaults staged locally. Save to apply and audit.");
  };

  const modelData = models.data;
  const gateData = gate.data;
  const cfg = config.data;

  return (
    <div className="settings-page">
      <div className="page-actions">
        <div className="control-group">
          <button className="btn btn-ghost" onClick={() => { void thresholds.refresh(); void models.refresh(); void gate.refresh(); void config.refresh(); }}>
            <RefreshCw size={15} /> Refresh
          </button>
          <a className="btn btn-ghost" href={exportCsvUrl()}>
            <Download size={15} /> Export inspections
          </a>
        </div>
        <a className="text-link" href="/docs" target="_blank" rel="noreferrer">
          OpenAPI / Swagger <ArrowUpRight size={13} />
        </a>
      </div>

      {error && <ErrorNote>{error}</ErrorNote>}
      {flash && (
        <div className="flash flash-good">
          <CheckCircle2 size={16} /> {flash}
        </div>
      )}

      <div className="settings-grid">
        <Panel
          title="Fusion thresholds"
          subtitle="Live values read from SQLite at every part finalize"
          actions={
            thresholds.data ? (
              <Badge tone="info">{thresholds.data.id} · updated {fmtRelative(thresholds.data.updated_at_ms)}</Badge>
            ) : (
              <Spinner />
            )
          }
        >
          {!thresholds.data ? (
            <Spinner label="Loading thresholds" />
          ) : (
            <>
              <div className="threshold-form">
                <label className="field">
                  <span>Profile name</span>
                  <input value={form.name} maxLength={80} onChange={(event) => update("name", event.target.value)} />
                </label>
                <div className="threshold-row">
                  <label className="field">
                    <span>pass_max <em>elevated novelty z</em></span>
                    <input
                      type="number"
                      min={0}
                      max={3}
                      step={0.01}
                      value={form.pass_max}
                      onChange={(event) => update("pass_max", event.target.value)}
                    />
                    <small>Trusted detector profiles escalate at or above this gate z.</small>
                  </label>
                  <label className="field">
                    <span>unknown_z <em>unknown anomaly z</em></span>
                    <input
                      type="number"
                      min={0}
                      max={3}
                      step={0.01}
                      value={form.unknown_z}
                      onChange={(event) => update("unknown_z", event.target.value)}
                    />
                    <small>Gate-only profiles escalate at or above this z.</small>
                  </label>
                </div>
                <div className="threshold-row">
                  <label className="field">
                    <span>review_max <em>reserved</em></span>
                    <input
                      type="number"
                      min={0}
                      max={3}
                      step={0.01}
                      value={form.review_max}
                      onChange={(event) => update("review_max", event.target.value)}
                    />
                    <small>Kept for contract compatibility; v1 fusion does not consume it.</small>
                  </label>
                  <label className="field">
                    <span>conf_min <em>part-level floor</em></span>
                    <input
                      type="number"
                      min={0}
                      max={3}
                      step={0.01}
                      value={form.conf_min}
                      onChange={(event) => update("conf_min", event.target.value)}
                    />
                    <small>Suppresses tracks whose median confidence is below this value.</small>
                  </label>
                </div>
              </div>

              {validation && <InfoNote tone="warn">{validation}</InfoNote>}
              {Number(form.conf_min) !== Number(cfg?.conf ?? 0.35) && (
                <InfoNote tone="warn">
                  The raw detector pre-filter is <strong>{fmtNum(cfg?.conf ?? 0.35, 2)}</strong>. Setting conf_min below
                  that cannot recover tracks the detector already discarded. Change <span className="mono">INDORE_CONF</span> at
                  process start to move the hard floor.
                </InfoNote>
              )}

              <div className="form-actions">
                <button className="btn btn-primary" onClick={() => void save()} disabled={saving || !!validation || !dirty}>
                  <Save size={15} /> {saving ? "Saving…" : "Save thresholds"}
                </button>
                <button className="btn btn-ghost" onClick={restore}>
                  <RotateCcw size={15} /> Stage defaults
                </button>
                {dirty && <span className="dirty-tag">unsaved changes</span>}
              </div>

              <InfoNote tone="warn">
                Threshold changes are audited and apply to the next finalized part. They do not persist into environment
                variables or restart-time configuration.
              </InfoNote>
            </>
          )}
        </Panel>

        <Panel
          title="Operating point"
          subtitle="Why the cell runs at 0.35 confidence"
          actions={<Badge tone="good" dot>selected</Badge>}
        >
          <div className="operating-point">
            <div className="operating-big">
              <span>Detector floor</span>
              <strong>{fmtNum(cfg?.conf ?? 0.35, 2)}</strong>
              <em>raw YOLO11s conf</em>
            </div>
            <div className="operating-facts">
              <div><span>Known test image hit</span><strong>95.1%</strong></div>
              <div><span>Held-out fire rate</span><strong>24.3%</strong></div>
              <div><span>Synthetic good FP</span><strong>18.8%</strong></div>
              <div><span>At 0.55 good FP</span><strong>7.8%</strong></div>
            </div>
          </div>
          <InfoNote tone="warn">
            Measured on the 640 deployment path with the promoted model. At 0.35 the detector fires on 24.3% of held-out
            images and 18.8% of clean crops; at 0.55 those fall to 10.3% and 7.8%. The steel gate is untrusted, so
            held-out fire is not an UNKNOWN escalation.
          </InfoNote>
          <ul className="guidance">
            <li>0.35 is selected for recall: 95.1% of test images get a detection, versus 88.4% at 0.55.</li>
            <li>0.55 is the higher-precision fallback (7.8% clean-crop FP); 0.65 cuts FP to 4.6% but test coverage drops to 80.0%.</li>
            <li>The novelty gate cannot fix detector hallucinations when the profile gate is untrusted.</li>
            <li>Any change here must be re-evaluated on the locked test split and the held-out class.</li>
          </ul>
        </Panel>
      </div>

      <div className="settings-lower">
        <Panel
          title="Runtime contract"
          subtitle="Read-only values resolved at process start"
          actions={cfg ? <Badge tone="info">{cfg.version}</Badge> : null}
        >
          {!cfg ? (
            <Spinner label="Loading runtime" />
          ) : (
            <div className="table-wrap">
              <table className="data-table compact env-table">
                <thead>
                  <tr>
                    <th>Environment key</th>
                    <th>Value</th>
                    <th>Meaning</th>
                  </tr>
                </thead>
                <tbody>
                  {ENV_ROWS.map((row) => (
                    <tr key={row.key}>
                      <td className="mono">{row.label}</td>
                      <td className="mono">
                        {row.key === "mes_configured" ? (cfg.mes_configured ? "configured" : "not set") : String(cfg[row.key] ?? "—")}
                      </td>
                      <td>{row.hint}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="config-paths">
            <span><Terminal size={13} /> project {cfg?.project_root ?? "—"}</span>
            <span><Database size={13} /> db {cfg?.db_path ?? "—"}</span>
          </div>
        </Panel>

        <Panel
          title="Models & artifacts"
          subtitle="Promoted runtime weights only"
          actions={modelData ? <Badge tone="good" dot>{modelData.versions.length} version(s)</Badge> : null}
        >
          {!modelData ? (
            <Spinner label="Loading models" />
          ) : (
            <>
              <div className="model-runtime">
                <div>
                  <span>Detector</span>
                  <strong>{modelData.runtime?.detector?.engine ?? "—"}</strong>
                  <em className="mono">{modelData.runtime?.detector?.model_version ?? "—"}</em>
                </div>
                <div>
                  <span>Device</span>
                  <strong>{modelData.runtime?.device ?? "—"}</strong>
                  <em>{modelData.runtime?.detector?.warning || "no detector warning"}</em>
                </div>
                <div>
                  <span>Gate</span>
                  <strong>{modelData.runtime?.gate?.profile ?? "—"}</strong>
                  <em>{titleCase(modelData.runtime?.gate?.gate_quality)}</em>
                </div>
              </div>

              <div className="table-wrap">
                <table className="data-table compact">
                  <thead>
                    <tr>
                      <th>Artifact</th>
                      <th>Kind</th>
                      <th>Size</th>
                      <th>Modified</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelData.artifacts.map((artifact) => (
                      <tr key={artifact.name}>
                        <td className="mono">{artifact.name}</td>
                        <td><Badge tone={artifact.kind === "gate" ? "info" : artifact.kind === "detector" ? "good" : "muted"}>{artifact.kind}</Badge></td>
                        <td className="mono">{fmtBytes(artifact.size)}</td>
                        <td>{fmtDateTime(artifact.modified_ms)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {modelData.versions.map((version) => (
                <div key={version.id} className="model-version">
                  <div className="model-version-head">
                    <Fingerprint size={14} />
                    <strong>{version.name}</strong>
                    <Badge tone="good">active</Badge>
                    <span>{fmtDateTime(version.created_at_ms)}</span>
                  </div>
                  <div className="model-version-meta">
                    <span className="mono">sha {version.sha256?.slice(0, 20) ?? "—"}…</span>
                    <span className="mono break-all">{version.path || "fixture (no weights)"}</span>
                  </div>
                </div>
              ))}

              {!modelData.versions.length && <Empty icon={<HardDrive size={22} />} title="No model versions registered" />}

              <InfoNote tone="warn">
                Runtime loads <strong>artifacts/models/detector_best.pt</strong> when present. Training checkpoints are
                never promoted automatically.
              </InfoNote>
            </>
          )}
        </Panel>
      </div>

      <div className="settings-lower">
        <Panel title="Gate matrix" subtitle="Quality and trust by profile" actions={<Layers size={15} />}>
          {!gateData ? (
            <Spinner label="Loading gate summary" />
          ) : (
            <div className="table-wrap">
              <table className="data-table compact">
                <thead>
                  <tr>
                    <th>Profile</th>
                    <th>Quality</th>
                    <th>Sep.</th>
                    <th>Trusted</th>
                    <th>Model</th>
                  </tr>
                </thead>
                <tbody>
                  {gateData.profiles.map((profile) => (
                    <tr key={profile.id}>
                      <td>{titleCase(profile.id)}</td>
                      <td>
                        <Badge tone={["strong", "good"].includes(profile.quality) ? "good" : ["weak", "marginal"].includes(profile.quality) ? "warn" : "bad"}>
                          {titleCase(profile.quality)}
                        </Badge>
                      </td>
                      <td className={`mono ${(profile.separation ?? 0) < 0 ? "text-bad" : ""}`}>
                        {profile.separation === null ? "—" : profile.separation.toFixed(2)}
                      </td>
                      <td>{profile.trusted ? <Badge tone="good">yes</Badge> : <Badge tone="bad">no</Badge>}</td>
                      <td className="mono small">{profile.model ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {gateData?.active && (
            <div className="active-gate">
              <span>Active gate</span>
              <strong>{titleCase(gateData.active.profile)}</strong>
              <Badge tone={gateData.active.ready ? "good" : "bad"}>{gateData.active.ready ? "ready" : gateData.active.error ?? "not ready"}</Badge>
              <span className="mono">{gateData.active.device}</span>
            </div>
          )}
          <InfoNote tone="warn">
            A profile with an untrusted gate is detector-first. Novelty numbers remain visible but are not allowed to
            create or suppress a defect verdict.
          </InfoNote>
        </Panel>

        <Panel title="Operational notes" subtitle="Limits of this demo cell">
          <ul className="guidance">
            <li><strong>Simulated feed:</strong> dataset replay is labelled on every screen and in the burned-in overlay.</li>
            <li><strong>Fixture inference:</strong> when no promoted weights exist, boxes come from ground-truth labels and are explicitly marked.</li>
            <li><strong>Not safety-rated:</strong> this is a research QA cell, not a certified machine-safety system.</li>
            <li><strong>SQLite:</strong> WAL mode, single-node. Not a clustered production historian.</li>
            <li><strong>MVTec:</strong> CC BY-NC-SA 4.0; non-commercial research use only.</li>
          </ul>
          <div className="settings-icons">
            <span><Cpu size={14} /> CPU fallback available</span>
            <span><ShieldAlert size={14} /> Gate caveats disclosed</span>
            <span><ShieldCheck size={14} /> Audited decisions</span>
            <span><Gauge size={14} /> Measured latency</span>
          </div>
        </Panel>
      </div>

      <div className="settings-foot">
        <FileJson size={14} /> Configuration is environment-first. Runtime UI changes are intentionally limited to
        audited fusion thresholds and replay controls.
      </div>
    </div>
  );
}
