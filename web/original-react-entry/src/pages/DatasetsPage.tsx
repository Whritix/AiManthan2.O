import { useState } from "react";
import {
  AlertTriangle,
  ArrowUpRight,
  Boxes,
  CheckCircle2,
  Database,
  FileCheck2,
  Fingerprint,
  Image as ImageIcon,
  Layers,
  LockKeyhole,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  XCircle,
} from "lucide-react";
import { api } from "../lib/api";
import { fmtInt, fmtNum, fmtRelative, titleCase } from "../lib/format";
import type { DatasetProfile, DatasetReport } from "../lib/types";
import { usePolling } from "../hooks/usePolling";
import { Badge, Empty, ErrorNote, InfoNote, Panel, ProgressBar, Spinner, warningTone, humanWarning } from "../components/ui";

const SPLIT_ORDER = ["train", "val", "test", "heldout"] as const;

function profileSample(profileId: string, kind: string, index = 0) {
  return `/api/datasets/image?profile=${encodeURIComponent(profileId)}&kind=${kind}&index=${index}`;
}

function GateBadge({ quality }: { quality: string }) {
  const tone =
    quality === "strong" || quality === "good"
      ? "good"
      : quality === "weak"
        ? "warn"
        : quality === "marginal"
          ? "warn"
          : "bad";
  return <Badge tone={tone}>{titleCase(quality)}</Badge>;
}

function ProfileCard({ profile }: { profile: DatasetProfile }) {
  const good = profile.counts.good ?? 0;
  const defect = profile.counts.known_defect ?? profile.counts.defect ?? 0;
  const unknown = profile.counts.unknown_defect ?? 0;
  const check = profile.gate_check ?? {};
  const rows = [
    { key: "good", label: "Good", data: check.good },
    { key: "defect", label: "Known defect", data: check.defect },
    { key: "unknown", label: "Held-out / unknown", data: check.unknown },
  ].filter((row) => row.data);

  return (
    <article className={`profile-card ${profile.gate_trusted ? "" : "profile-card-warn"}`}>
      <header className="profile-card-head">
        <div>
          <h3>{profile.display_name}</h3>
          <p>
            {profile.detector ? "Detector + gate" : "Gate-only onboarding"} · {profile.license}
          </p>
        </div>
        <GateBadge quality={profile.gate_quality} />
      </header>

      <div className="profile-samples">
        <a href={profileSample(profile.id, "good")} target="_blank" rel="noreferrer" className="sample">
          <img src={profileSample(profile.id, "good")} alt={`${profile.display_name} good sample`} loading="lazy" />
          <span>good</span>
          <em>{good}</em>
        </a>
        {(defect > 0 || profile.detector) && (
          <a href={profileSample(profile.id, "defect")} target="_blank" rel="noreferrer" className="sample">
            <img src={profileSample(profile.id, "defect")} alt={`${profile.display_name} defect sample`} loading="lazy" />
            <span>known defect</span>
            <em>{defect}</em>
          </a>
        )}
        <a href={profileSample(profile.id, "unknown")} target="_blank" rel="noreferrer" className="sample">
          <img src={profileSample(profile.id, "unknown")} alt={`${profile.display_name} unknown sample`} loading="lazy" />
          <span>unknown</span>
          <em>{unknown}</em>
        </a>
      </div>

      <dl className="profile-facts">
        <div>
          <dt>Detector</dt>
          <dd>{profile.detector ? <Badge tone="good">enabled</Badge> : <Badge tone="muted">off</Badge>}</dd>
        </div>
        <div>
          <dt>Gate trusted</dt>
          <dd>{profile.gate_trusted ? <Badge tone="good">yes</Badge> : <Badge tone="bad">no</Badge>}</dd>
        </div>
        <div>
          <dt>Separation</dt>
          <dd className="mono">{profile.gate_separation === null ? "—" : profile.gate_separation.toFixed(2)}</dd>
        </div>
        <div>
          <dt>Missing frames</dt>
          <dd className={profile.missing ? "text-bad" : "text-good"}>{profile.missing}</dd>
        </div>
      </dl>

      <div className="profile-classes">
        {profile.heldout_class && (
          <span className="heldout-chip">
            <LockKeyhole size={12} /> held out: {titleCase(profile.heldout_class)}
          </span>
        )}
        {profile.counts.bank_crops ? <span>{fmtInt(profile.counts.bank_crops)} bank crops</span> : null}
        {profile.counts.bank_images ? <span>{fmtInt(profile.counts.bank_images)} bank images</span> : null}
        {profile.counts.calib_frames ? <span>{fmtInt(profile.counts.calib_frames)} calibration frames</span> : null}
      </div>

      {!profile.gate_trusted && (
        <InfoNote tone="warn">
          {profile.id === "steel"
            ? "Negative separation on held-out unknowns. This gate is displayed for transparency and never drives the verdict."
            : "Gate quality is too low to trust. It is computed but does not gate this profile."}
        </InfoNote>
      )}

      {rows.length > 0 && (
        <details className="gate-details">
          <summary>Gate calibration detail</summary>
          <div className="table-wrap">
            <table className="data-table compact">
              <thead>
                <tr>
                  <th>Set</th>
                  <th>n</th>
                  <th>z p50</th>
                  <th>z p95</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.key}>
                    <td>{row.label}</td>
                    <td className="mono">{row.data?.n ?? "—"}</td>
                    <td className="mono">{fmtNum(row.data?.z_p50, 3)}</td>
                    <td className="mono">{fmtNum(row.data?.z_p95, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </article>
  );
}

export default function DatasetsPage() {
  const report = usePolling<DatasetReport>(() => api.datasets(false), 0);
  const [verifying, setVerifying] = useState(false);
  const [verifyError, setVerifyError] = useState<string | null>(null);

  const verify = async () => {
    setVerifying(true);
    setVerifyError(null);
    try {
      const next = await api.datasets(true);
      report.setData(next);
    } catch (err) {
      setVerifyError(err instanceof Error ? err.message : String(err));
    } finally {
      setVerifying(false);
    }
  };

  const data = report.data;
  const neu = data?.neu ?? {};
  const mvtec = data?.mvtec ?? {};
  const splits = (neu.splits ?? {}) as Record<string, { count: number; per_class: Record<string, number>; missing: number; sha256: string | null }>;
  const trained = (neu.trained_classes ?? []) as string[];
  const classes = (neu.classes ?? []) as string[];
  const mvtecCategories = (mvtec.categories ?? []) as string[];
  const profileWarnings = data?.profiles?.filter((profile) => !profile.gate_trusted) ?? [];

  return (
    <div className="datasets-page">
      <div className="page-actions">
        <div className="control-group">
          <span className={`verify-state ${data ? "ok" : ""}`}>
            {data ? <CheckCircle2 size={15} /> : <RefreshCw size={15} className="spin" />}
            {data ? `verified ${fmtRelative(Date.parse(data.verified_at))}` : "verifying"}
          </span>
          <button className="btn btn-primary" onClick={() => void verify()} disabled={verifying}>
            <RefreshCw size={15} className={verifying ? "spin" : ""} />
            {verifying ? "Re-verifying…" : "Run verification"}
          </button>
        </div>
        <span className="muted-line">Checks disk counts, split hashes, held-out isolation, frame presence, and gate quality.</span>
      </div>

      {report.error && <ErrorNote>{report.error}</ErrorNote>}
      {verifyError && <ErrorNote>{verifyError}</ErrorNote>}

      {!data ? (
        <Panel>
          <Spinner label="Verifying datasets" />
        </Panel>
      ) : (
        <>
          <div className="stats-grid dataset-stats">
            <div className="stat">
              <div className="stat-top"><span className="stat-label">NEU-DET</span><Database size={16} className="stat-icon" /></div>
              <div className="stat-value">{fmtInt(neu.total)}<span className="stat-unit">images</span></div>
              <div className="stat-hint">{classes.length} classes · {trained.length} trained · 1 held out</div>
            </div>
            <div className="stat">
              <div className="stat-top"><span className="stat-label">Held-out isolation</span><Fingerprint size={16} className="stat-icon" /></div>
              <div className="stat-value">{neu.heldout_leak_free ? "CLEAN" : "LEAK"}</div>
              <div className="stat-hint">train/val/test hash comparison</div>
            </div>
            <div className="stat">
              <div className="stat-top"><span className="stat-label">MVTec AD</span><Boxes size={16} className="stat-icon" /></div>
              <div className="stat-value">{fmtInt(mvtec.category_count)}<span className="stat-unit">categories</span></div>
              <div className="stat-hint">{fmtInt(mvtec.total_train_good)} train-good · {fmtInt(mvtec.total_test)} test · {fmtInt(mvtec.total_masks)} masks</div>
            </div>
            <div className="stat">
              <div className="stat-top"><span className="stat-label">Profiles ready</span><Layers size={16} className="stat-icon" /></div>
              <div className="stat-value">{data.profiles.filter((profile) => profile.missing === 0).length}<span className="stat-unit">/ {data.profiles.length}</span></div>
              <div className="stat-hint">{profileWarnings.length} with untrusted gate</div>
            </div>
            <div className="stat">
              <div className="stat-top"><span className="stat-label">Verification warnings</span><AlertTriangle size={16} className="stat-icon" /></div>
              <div className="stat-value">{data.warnings.length}</div>
              <div className="stat-hint">see disclosure panel below</div>
            </div>
          </div>

          <div className="dataset-grid">
            <Panel
              title="NEU-DET split"
              subtitle="Six-class source · stratified split seed 42 · held-out class excluded from training"
              actions={<Badge tone="info">CC BY 4.0</Badge>}
            >
              <div className="split-grid">
                {SPLIT_ORDER.map((key) => {
                  const split = splits[key];
                  if (!split) return null;
                  const isHeldout = key === "heldout";
                  return (
                    <div key={key} className={`split-card ${isHeldout ? "split-heldout" : ""}`}>
                      <div className="split-head">
                        <span>{titleCase(key)}</span>
                        {isHeldout ? <LockKeyhole size={13} /> : <Badge tone="muted">{Object.keys(split.per_class ?? {}).length} cls</Badge>}
                      </div>
                      <strong>{fmtInt(split.count)}</strong>
                      <div className="split-bar">
                        <ProgressBar value={split.count / Math.max(1, neu.total ?? 1800)} tone={isHeldout ? "warn" : "accent"} />
                      </div>
                      <div className="split-classes">
                        {Object.entries(split.per_class ?? {}).map(([name, count]) => (
                          <span key={name}>{titleCase(name)} <em>{count}</em></span>
                        ))}
                      </div>
                      <div className="split-foot">
                        <span className={split.missing ? "text-bad" : "text-good"}>
                          {split.missing ? `${split.missing} missing` : "all files present"}
                        </span>
                        {split.sha256 && <span className="mono">sha {split.sha256.slice(0, 8)}</span>}
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="class-strip">
                <span className="class-strip-label">Trained classes</span>
                {trained.map((name) => (
                  <span key={name} className="class-pill">{titleCase(name)}</span>
                ))}
                <span className="class-strip-label">Held out</span>
                <span className="class-pill class-pill-heldout">
                  <LockKeyhole size={12} /> {titleCase(neu.heldout_class)}
                </span>
              </div>

              <div className="dataset-notes">
                <InfoNote tone="good">
                  Leak check: {neu.heldout_leak_free ? "held-out files never appear in train, validation, or test." : "LEAK DETECTED — inspect immediately."}
                </InfoNote>
                <InfoNote tone="warn">
                  The source is not perfectly clean: 123 images contain mixed-class boxes and one crazing XML was
                  misfiled. The converter repairs both per box; the split manifest is hash-locked.
                </InfoNote>
              </div>

              <dl className="dataset-meta">
                <div><dt>Source</dt><dd className="mono break-all">{neu.source ?? "—"}</dd></div>
                <div><dt>Seed</dt><dd>{neu.seed ?? "—"}</dd></div>
                <div><dt>Class order</dt><dd>{classes.map(titleCase).join(" · ")}</dd></div>
              </dl>
            </Panel>

            <Panel
              title="MVTec AD source"
              subtitle="Non-commercial benchmark used for gate-only onboarding profiles"
              actions={<Badge tone="warn">CC BY-NC-SA 4.0</Badge>}
            >
              <div className="mvtec-metrics">
                <div><span>Categories</span><strong>{fmtInt(mvtec.category_count)}</strong></div>
                <div><span>Train good</span><strong>{fmtInt(mvtec.total_train_good)}</strong></div>
                <div><span>Test images</span><strong>{fmtInt(mvtec.total_test)}</strong></div>
                <div><span>Ground-truth masks</span><strong>{fmtInt(mvtec.total_masks)}</strong></div>
              </div>
              <div className="category-cloud">
                {mvtecCategories.map((category) => (
                  <span key={category} className={`category-pill ${["metal_nut", "screw", "transistor", "bottle"].includes(category) ? "category-used" : ""}`}>
                    {titleCase(category)}
                  </span>
                ))}
              </div>
              <InfoNote tone="warn">
                MVTec is CC BY-NC-SA 4.0: research and non-commercial use only. The four highlighted categories power
                gate-only onboarding demos; the rest remain source evidence.
              </InfoNote>
              <dl className="dataset-meta">
                <div><dt>Root</dt><dd className="mono break-all">{mvtec.root ?? "—"}</dd></div>
                <div><dt>License</dt><dd>{mvtec.license ?? "—"}</dd></div>
              </dl>
            </Panel>
          </div>

          <Panel
            title="Replay profiles"
            subtitle="Every replay frame is labelled at build time; nothing is inferred from a directory name"
            actions={<Badge tone="info">{data.profiles.length} profiles</Badge>}
          >
            <div className="profile-grid">
              {data.profiles.map((profile) => (
                <ProfileCard key={profile.id} profile={profile} />
              ))}
            </div>
          </Panel>

          <div className="dataset-lower">
            <Panel title="Disclosures" subtitle="What this verification does and does not prove">
              {data.warnings.length ? (
                <ul className="warning-rows">
                  {data.warnings.map((warning) => (
                    <li key={warning}>
                      <Badge tone={warningTone(warning)}>
                        {warningTone(warning) === "bad" ? <ShieldAlert size={12} /> : <AlertTriangle size={12} />}
                        {humanWarning(warning.split(":")[0])}
                      </Badge>
                      <span>{warning}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <InfoNote tone="good">No verification warnings. All expected files and gates are present.</InfoNote>
              )}
              <div className="disclosure-grid">
                <div>
                  <h4><CheckCircle2 size={14} /> Verified here</h4>
                  <ul>
                    <li>Image and mask counts on disk.</li>
                    <li>Split file lists and SHA-256 fingerprints.</li>
                    <li>Held-out class never appears in trained splits.</li>
                    <li>Every replay item resolves to an existing file.</li>
                    <li>Gate bank presence, calibration z, and separation.</li>
                  </ul>
                </div>
                <div>
                  <h4><XCircle size={14} /> Not proven here</h4>
                  <ul>
                    <li>Annotation correctness beyond parseable boxes.</li>
                    <li>Coverage of every possible production defect.</li>
                    <li>Safety certification or line qualification.</li>
                    <li>Non-commercial MVTec use in a shipped product.</li>
                  </ul>
                </div>
              </div>
            </Panel>

            <Panel title="Gate quality matrix" subtitle="Measured separation of good vs held-out anomaly">
              <div className="table-wrap">
                <table className="data-table compact">
                  <thead>
                    <tr>
                      <th>Profile</th>
                      <th>Quality</th>
                      <th>Sep.</th>
                      <th>Trusted</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.profiles.map((profile) => (
                      <tr key={profile.id}>
                        <td>{profile.display_name}</td>
                        <td><GateBadge quality={profile.gate_quality} /></td>
                        <td className={`mono ${(profile.gate_separation ?? 0) < 0 ? "text-bad" : ""}`}>
                          {profile.gate_separation === null ? "—" : profile.gate_separation.toFixed(2)}
                        </td>
                        <td>{profile.gate_trusted ? <Badge tone="good">yes</Badge> : <Badge tone="bad">no</Badge>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="quality-legend">
                <span><ShieldCheck size={13} /> ≥1.5 strong</span>
                <span>≥0.75 good</span>
                <span>≥0.30 weak</span>
                <span><ShieldAlert size={13} /> &lt;0 not validated</span>
              </div>
              <a className="text-link" href="/docs#/default" target="_blank" rel="noreferrer">
                API documentation <ArrowUpRight size={13} />
              </a>
            </Panel>
          </div>

          <div className="datasets-footnote">
            <Sparkles size={14} /> Verification is deterministic and read-only: it never rewrites the dataset or the
            split manifest.
          </div>
        </>
      )}
    </div>
  );
}
