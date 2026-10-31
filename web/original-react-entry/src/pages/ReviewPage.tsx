import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  ClipboardCheck,
  Clock,
  Download,
  Eye,
  FileCheck2,
  Filter,
  Image as ImageIcon,
  RefreshCw,
  ScanLine,
  Search,
  ShieldAlert,
  SlidersHorizontal,
  Tag,
  XCircle,
} from "lucide-react";
import { api } from "../lib/api";
import { fmtDateTime, fmtMs, fmtPct, fmtRelative, fmtTime, shortHash, shortId, titleCase } from "../lib/format";
import type { Detection, Disposition, Inspection } from "../lib/types";
import { useNow, usePolling } from "../hooks/usePolling";
import {
  Badge,
  Empty,
  ErrorNote,
  InfoNote,
  Panel,
  SeverityMeter,
  Spinner,
  VerdictChip,
  WarningList,
} from "../components/ui";

type EvidenceTab = "overlay" | "source" | "heatmap" | "crop";

const CLASS_COLOR: Record<string, string> = {
  crazing: "#7dd3fc",
  inclusion: "#f472b6",
  patches: "#fbbf24",
  pitted_surface: "#a78bfa",
  scratches: "#fb7185",
  unknown: "#f59e0b",
};

const DISPOSITIONS: {
  value: Disposition;
  label: string;
  hint: string;
  icon: typeof CheckCircle2;
  tone: string;
}[] = [
  { value: "true_defect", label: "Confirm defect", hint: "The part is genuinely defective.", icon: AlertTriangle, tone: "bad" },
  { value: "false_alarm", label: "False alarm", hint: "The part is good; release it.", icon: CheckCircle2, tone: "good" },
  { value: "reclassify", label: "Reclassify", hint: "Defect is real but the label is wrong.", icon: Tag, tone: "info" },
];

function QueueItem({
  item,
  active,
  onSelect,
}: {
  item: Inspection;
  active: boolean;
  onSelect: () => void;
}) {
  const top = item.detections?.[0];
  const image = item.overlay_url || item.image_url;
  return (
    <button className={`queue-item ${active ? "active" : ""}`} onClick={onSelect}>
      <div className="queue-thumb">
        {image ? <img src={image} alt="" loading="lazy" /> : <ScanLine size={18} />}
        <VerdictChip verdict={item.final_verdict || item.verdict} />
      </div>
      <div className="queue-copy">
        <div className="queue-topline">
          <span className="mono">{shortId(item.id, 12)}</span>
          <span className="queue-age">{fmtRelative(item.ts_ms)}</span>
        </div>
        <strong>{top ? titleCase(top.class) : titleCase(item.reason)}</strong>
        <div className="queue-meta">
          <span>{titleCase(item.profile)}</span>
          <span>z {item.anomaly_z.toFixed(2)}</span>
          <SeverityMeter value={item.severity} compact />
        </div>
      </div>
      <ChevronRight size={15} className="queue-chevron" />
    </button>
  );
}

function DetectionTable({ detections }: { detections: Detection[] }) {
  if (!detections.length) {
    return <p className="muted-line">No supported detector tracks were committed for this part.</p>;
  }
  return (
    <div className="table-wrap">
      <table className="data-table compact">
        <thead>
          <tr>
            <th>Class</th>
            <th>Conf</th>
            <th>Area</th>
            <th>Votes</th>
            <th>Severity</th>
            <th>BBox x,y,w,h</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {detections.map((detection) => (
            <tr key={detection.id}>
              <td>
                <span className="class-cell">
                  <i style={{ background: CLASS_COLOR[detection.class] ?? "#94a3b8" }} />
                  {titleCase(detection.class)}
                </span>
              </td>
              <td className="mono">{fmtPct(detection.conf)}</td>
              <td className="mono">{fmtPct(detection.area_frac)}</td>
              <td className="mono">{detection.votes}</td>
              <td><SeverityMeter value={detection.severity} compact /></td>
              <td className="mono">{(detection.bbox ?? []).join(", ")}</td>
              <td>
                <Badge tone={detection.source === "fixture" ? "warn" : "neutral"}>{detection.source}</Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ReviewPage() {
  const [params, setParams] = useSearchParams();
  const now = useNow(5000);
  const queue = usePolling(() => api.reviewQueue(), 5000);
  const history = usePolling(() => api.inspections({ limit: 24 }), 12000);
  const [selectedId, setSelectedId] = useState<string | null>(params.get("id"));
  const [detail, setDetail] = useState<Inspection | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [filter, setFilter] = useState<"all" | "review" | "fail">("all");
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<EvidenceTab>("overlay");
  const [disposition, setDisposition] = useState<Disposition | null>(null);
  const [label, setLabel] = useState("");
  const [note, setNote] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);

  const items = queue.data?.items ?? [];
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return items.filter((item) => {
      const verdict = item.final_verdict || item.verdict;
      if (filter !== "all" && verdict !== filter) return false;
      if (!needle) return true;
      return (
        item.id.toLowerCase().includes(needle) ||
        item.profile.toLowerCase().includes(needle) ||
        (item.detections ?? []).some((detection) => detection.class.toLowerCase().includes(needle)) ||
        (item.reason ?? "").toLowerCase().includes(needle)
      );
    });
  }, [items, filter, query]);

  // Keep a valid selection, preferring the URL parameter when present.
  useEffect(() => {
    const requested = params.get("id");
    if (requested) {
      if (requested !== selectedId) setSelectedId(requested);
      return;
    }
    if (selectedId && (queue.loading || items.some((item) => item.id === selectedId))) return;
    setSelectedId(items[0]?.id ?? null);
  }, [items, params, queue.loading, selectedId]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let disposed = false;
    setDetailLoading(true);
    api
      .inspection(selectedId)
      .then((next) => {
        if (!disposed) {
          setDetail(next);
          setDetailLoading(false);
        }
      })
      .catch(() => {
        if (!disposed) {
          setDetail(null);
          setDetailLoading(false);
        }
      });
    return () => {
      disposed = true;
    };
  }, [selectedId, queue.data]);

  useEffect(() => {
    setDisposition(null);
    setLabel("");
    setNote("");
    setFormError(null);
    setTab("overlay");
  }, [selectedId]);

  const selected = detail ?? items.find((item) => item.id === selectedId) ?? null;
  const reviewed = !!selected?.operator_disposition;
  const openFail = items.filter((item) => (item.final_verdict || item.verdict) === "fail").length;
  const openReview = items.filter((item) => (item.final_verdict || item.verdict) === "review").length;
  const oldest = items.reduce<number | null>((min, item) => (min === null || item.ts_ms < min ? item.ts_ms : min), null);
  const historyItems = (history.data?.items ?? []).filter((item) => item.operator_disposition);

  const chooseDisposition = (value: Disposition) => {
    setDisposition(value);
    if (value === "false_alarm") setLabel("good");
    else if (value === "true_defect") {
      const top = selected?.detections?.[0]?.class ?? selected?.ground_truth_class ?? "defect";
      setLabel(top === "good" ? "defect" : top);
    } else {
      setLabel(selected?.ground_truth_class ?? "");
    }
    setFormError(null);
  };

  const submit = async () => {
    if (!selected || !disposition) return;
    if (!label.trim()) {
      setFormError("A label is required so the disposition is auditable.");
      return;
    }
    setSubmitting(true);
    setFormError(null);
    try {
      await api.review(selected.id, disposition, label.trim(), note.trim() || undefined);
      setFlash(`${shortId(selected.id, 14)} dispositioned as ${titleCase(disposition)}.`);
      setSelectedId(null);
      setParams({}, { replace: true });
      setDisposition(null);
      setLabel("");
      setNote("");
      await queue.refresh();
      await history.refresh();
    } catch (err) {
      setFormError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const nextPart = () => {
    const index = items.findIndex((item) => item.id === selectedId);
    const next = items[index + 1] ?? items[0];
    if (next) {
      setSelectedId(next.id);
      setParams({ id: next.id }, { replace: true });
    }
  };

  const imageMap: Record<EvidenceTab, string | null | undefined> = {
    overlay: selected?.overlay_url,
    source: selected?.image_url,
    heatmap: selected?.heatmap_url,
    crop: selected?.crop_url,
  };

  return (
    <div className="review-page">
      <div className="review-stats">
        <div className="review-stat">
          <span>Open queue</span>
          <strong>{queue.data?.open_reviews ?? items.length}</strong>
          <em>unreviewed escalations</em>
        </div>
        <div className="review-stat review-stat-fail">
          <span>Fail · held</span>
          <strong>{openFail}</strong>
          <em>quarantine / rework</em>
        </div>
        <div className="review-stat review-stat-warn">
          <span>Review · ambiguous</span>
          <strong>{openReview}</strong>
          <em>novelty or weak evidence</em>
        </div>
        <div className="review-stat">
          <span>Oldest item</span>
          <strong>{oldest ? fmtRelative(oldest) : "—"}</strong>
          <em>{now ? "queue age" : ""}</em>
        </div>
        <div className="review-stat review-stat-quiet">
          <span>Dispositioned</span>
          <strong>{historyItems.length}</strong>
          <em>latest 24 inspections</em>
        </div>
      </div>

      {flash && (
        <div className="flash flash-good">
          <CheckCircle2 size={16} />
          {flash}
          <button onClick={() => setFlash(null)}>dismiss</button>
        </div>
      )}

      <div className="review-grid">
        <aside className="queue-panel">
          <div className="queue-head">
            <div>
              <h2>Escalation queue</h2>
              <p>Newest first · auto-refresh 5s</p>
            </div>
            <button
              className="icon-btn"
              onClick={() => void queue.refresh()}
              title="Refresh queue"
              aria-label="Refresh queue"
            >
              <RefreshCw size={15} className={queue.loading ? "spin" : ""} />
            </button>
          </div>

          <div className="queue-controls">
            <label className="search-box">
              <Search size={14} />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search id, class, reason" />
            </label>
            <div className="segmented">
              {(["all", "review", "fail"] as const).map((value) => (
                <button
                  key={value}
                  className={filter === value ? "on" : ""}
                  onClick={() => setFilter(value)}
                >
                  {titleCase(value)}
                  <em>{value === "all" ? items.length : value === "review" ? openReview : openFail}</em>
                </button>
              ))}
            </div>
          </div>

          <div className="queue-list">
            {queue.loading && !items.length ? (
              <Spinner label="Loading queue" />
            ) : queue.error ? (
              <ErrorNote>{queue.error}</ErrorNote>
            ) : filtered.length ? (
              filtered.map((item) => (
                <QueueItem
                  key={item.id}
                  item={item}
                  active={item.id === selectedId}
                  onSelect={() => {
                    setSelectedId(item.id);
                    setParams({ id: item.id }, { replace: true });
                  }}
                />
              ))
            ) : (
              <Empty
                icon={<Filter size={22} />}
                title={items.length ? "No items match this filter" : "Queue is clear"}
                hint={
                  items.length
                    ? "Clear the search or switch the verdict filter."
                    : "No unreviewed REVIEW or FAIL parts. The cell is holding steady."
                }
              />
            )}
          </div>

          {historyItems.length > 0 && (
            <div className="queue-history">
              <div className="queue-history-head">
                <ClipboardCheck size={14} />
                <span>Recent decisions</span>
              </div>
              {historyItems.slice(0, 5).map((item) => (
                <button
                  key={item.id}
                  className="history-row"
                  onClick={() => {
                    setSelectedId(item.id);
                    setParams({ id: item.id }, { replace: true });
                  }}
                >
                  <span className="mono">{shortId(item.id, 10)}</span>
                  <Badge tone={item.operator_disposition === "false_alarm" ? "good" : item.operator_disposition === "reclassify" ? "info" : "bad"}>
                    {titleCase(item.operator_disposition)}
                  </Badge>
                  <span>{fmtRelative(item.ts_ms)}</span>
                </button>
              ))}
            </div>
          )}
        </aside>

        <section className="review-detail">
          {!selected ? (
            <Panel title="Inspection detail" subtitle="Select an escalation from the queue">
              <Empty
                icon={<ClipboardCheck size={26} />}
                title="Nothing selected"
                hint="Pick a REVIEW or FAIL part from the queue. Evidence, detections, and the disposition form will load here."
              />
            </Panel>
          ) : detailLoading && !detail ? (
            <Panel>
              <Spinner label="Loading inspection evidence" />
            </Panel>
          ) : (
            <>
              <Panel
                flush
                className="review-head-panel"
                bodyClassName="review-head-body"
              >
                <div className="review-head">
                  <div className="review-head-left">
                    <VerdictChip verdict={selected.final_verdict || selected.verdict} />
                    <div>
                      <h2>{titleCase(selected.detections?.[0]?.class ?? selected.reason)}</h2>
                      <p>
                        <span className="mono">{selected.id}</span> · {titleCase(selected.profile)} ·{" "}
                        {fmtDateTime(selected.ts_ms)}
                      </p>
                    </div>
                  </div>
                  <div className="review-head-right">
                    <Badge tone={selected.source === "dataset" ? "warn" : "info"} dot>
                      {selected.source === "dataset" ? "replay evidence" : `${selected.source} evidence`}
                    </Badge>
                    <SeverityMeter value={selected.severity} />
                    <button className="btn btn-ghost" onClick={nextPart}>
                      Next <ArrowRight size={14} />
                    </button>
                  </div>
                </div>
                {selected.warnings?.length > 0 && <WarningList warnings={selected.warnings} />}
              </Panel>

              <div className="review-columns">
                <div className="review-left">
                  <Panel
                    title="Evidence"
                    subtitle="Immutable stills hashed at commit time"
                    actions={
                      <a className="text-link" href={imageMap[tab] ?? "#"} target="_blank" rel="noreferrer">
                        <Download size={14} /> open full
                      </a>
                    }
                    flush
                  >
                    <div className="evidence-tabs">
                      {(["overlay", "source", "heatmap", "crop"] as const).map((value) => (
                        <button key={value} className={tab === value ? "on" : ""} onClick={() => setTab(value)}>
                          {value === "overlay" ? "Overlay" : value === "source" ? "Source" : value === "heatmap" ? "Heatmap" : "Crop"}
                          {!imageMap[value] && <span className="tab-off">—</span>}
                        </button>
                      ))}
                    </div>
                    <div className="evidence-view">
                      {imageMap[tab] ? (
                        <img src={imageMap[tab] as string} alt={`${tab} evidence for ${selected.id}`} />
                      ) : (
                        <Empty
                          icon={<ImageIcon size={24} />}
                          title={`No ${tab} image`}
                          hint="This artifact was not produced for the inspection."
                        />
                      )}
                      {selected.source === "dataset" && (
                        <span className="evidence-tag">REPLAY EVIDENCE</span>
                      )}
                    </div>
                    <div className="evidence-foot">
                      <span><Eye size={13} /> overlay is detector boxes + gate regions</span>
                      <span className="mono">sha {shortHash(selected.image_sha ?? selected.overlay_sha)}</span>
                    </div>
                  </Panel>

                  <Panel title="Detector evidence" subtitle="Tracks surviving multi-frame support and the confidence floor">
                    <DetectionTable detections={selected.detections ?? []} />
                    {selected.ground_truth_class && (
                      <InfoNote tone="warn">
                        Replay ground truth: <strong>{titleCase(selected.ground_truth_class)}</strong> (
                        {titleCase(selected.ground_truth_kind)}). This label is fixture/replay metadata and is never
                        available to production inference.
                      </InfoNote>
                    )}
                  </Panel>

                  <Panel title="Fusion trace" subtitle="Why this part landed in the queue">
                    <div className="trace-grid">
                      <div>
                        <span>Route</span>
                        <strong>{titleCase(selected.route)}</strong>
                      </div>
                      <div>
                        <span>Reason</span>
                        <strong>{titleCase(selected.reason)}</strong>
                      </div>
                      <div>
                        <span>Gate z</span>
                        <strong>{selected.anomaly_z.toFixed(2)} (raw {selected.anomaly_raw.toFixed(4)})</strong>
                      </div>
                      <div>
                        <span>Latency</span>
                        <strong>{fmtMs(selected.latency_total_ms)}</strong>
                      </div>
                      <div>
                        <span>Frames fused</span>
                        <strong>{selected.frame_count}</strong>
                      </div>
                      <div>
                        <span>Threshold profile</span>
                        <strong>{selected.threshold_profile ?? "—"}</strong>
                      </div>
                      <div className="trace-wide">
                        <span>Model version</span>
                        <strong className="mono break-all">{selected.model_version ?? "—"}</strong>
                      </div>
                      <div className="trace-wide">
                        <span>Evidence hash</span>
                        <strong className="mono break-all">{selected.image_sha ?? "—"}</strong>
                      </div>
                    </div>
                  </Panel>
                </div>

                <div className="review-right">
                  <Panel
                    title={reviewed ? "Operator disposition" : "Disposition"}
                    subtitle={
                      reviewed
                        ? `Recorded ${fmtDateTime(selected.reviewed_at_ms)}`
                        : "Every decision is audited and queued for MES delivery."
                    }
                    actions={reviewed ? <Badge tone="good" dot>CLOSED</Badge> : <Badge tone="warn" dot>OPEN</Badge>}
                  >
                    {reviewed ? (
                      <div className="decision-record">
                        <div className="decision-chip">
                          <FileCheck2 size={18} />
                          <div>
                            <strong>{titleCase(selected.operator_disposition)}</strong>
                            <span>label: {selected.operator_label ?? "—"}</span>
                          </div>
                        </div>
                        {selected.operator_note && <p className="decision-note">{selected.operator_note}</p>}
                        <p className="muted-line">
                          This inspection is already dispositioned. The queue item is closed and the event is in the
                          audit trail.
                        </p>
                        <Link to="/audit" className="btn btn-ghost">
                          Open audit trail <ArrowRight size={14} />
                        </Link>
                      </div>
                    ) : (
                      <>
                        <div className="disposition-grid">
                          {DISPOSITIONS.map(({ value, label: text, hint, icon: Icon, tone }) => (
                            <button
                              key={value}
                              className={`disposition disposition-${tone} ${disposition === value ? "on" : ""}`}
                              onClick={() => chooseDisposition(value)}
                            >
                              <Icon size={18} />
                              <strong>{text}</strong>
                              <span>{hint}</span>
                            </button>
                          ))}
                        </div>

                        <label className="field">
                          <span>Label <em>required</em></span>
                          <input
                            value={label}
                            onChange={(event) => setLabel(event.target.value)}
                            placeholder="defect class or good"
                          />
                        </label>
                        <label className="field">
                          <span>Operator note <em>optional</em></span>
                          <textarea
                            rows={3}
                            value={note}
                            onChange={(event) => setNote(event.target.value)}
                            placeholder="What did you see? Context for the next reviewer."
                          />
                        </label>

                        {formError && <ErrorNote>{formError}</ErrorNote>}
                        <button className="btn btn-primary btn-wide" disabled={!disposition || submitting} onClick={() => void submit()}>
                          {submitting ? <RefreshCw size={15} className="spin" /> : <CheckCircle2 size={15} />}
                          {submitting ? "Committing…" : "Commit disposition"}
                        </button>
                        <p className="form-foot">
                          <ShieldAlert size={13} /> Disposition writes an audit event and a signed outbox record.
                        </p>
                      </>
                    )}
                  </Panel>

                  <Panel title="Gate context" subtitle="Novelty never classifies; it only escalates">
                    <dl className="kv">
                      <div>
                        <dt>Gate z</dt>
                        <dd className="mono">{selected.anomaly_z.toFixed(3)}</dd>
                      </div>
                      <div>
                        <dt>Raw distance</dt>
                        <dd className="mono">{selected.anomaly_raw.toFixed(5)}</dd>
                      </div>
                      <div>
                        <dt>Gate trusted</dt>
                        <dd>{selected.gate_trusted ? <Badge tone="good">yes</Badge> : <Badge tone="warn">no</Badge>}</dd>
                      </div>
                      <div>
                        <dt>Reason</dt>
                        <dd>{titleCase(selected.reason)}</dd>
                      </div>
                    </dl>
                    {!selected.gate_trusted && (
                      <InfoNote tone="warn">
                        This profile's gate is not validated. The detector alone drove the verdict; z is retained for
                        transparency only.
                      </InfoNote>
                    )}
                  </Panel>

                  <Panel title="Operator guidance" subtitle="Keep disposition decisions consistent">
                    <ul className="guidance">
                      <li><strong>Confirm defect</strong> when the part should not ship, regardless of whether the class label is perfect.</li>
                      <li><strong>False alarm</strong> when the part is good. This feeds the false-positive metric.</li>
                      <li><strong>Reclassify</strong> when the defect is real but the detector's class is wrong.</li>
                      <li>Treat replay ground truth as a hint only. Decide from the image and the evidence trace.</li>
                    </ul>
                  </Panel>
                </div>
              </div>
            </>
          )}
        </section>
      </div>

      <div className="review-footer-links">
        <Link to="/metrics" className="text-link">
          <SlidersHorizontal size={14} /> See how dispositions move the metrics <ArrowRight size={13} />
        </Link>
        <Link to="/audit" className="text-link">
          <XCircle size={14} /> Audit trail & outbox <ArrowRight size={13} />
        </Link>
      </div>
    </div>
  );
}
