import { useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CloudUpload,
  Database,
  FileClock,
  Filter,
  Inbox,
  Link2,
  RefreshCw,
  RotateCcw,
  Search,
  Send,
  ServerCog,
  ShieldCheck,
  TerminalSquare,
  XCircle,
} from "lucide-react";
import { api } from "../lib/api";
import { fmtDateTime, fmtInt, fmtRelative, fmtTime, titleCase } from "../lib/format";
import type { AuditEvent, OutboxEvent, OutboxResponse } from "../lib/types";
import { usePolling } from "../hooks/usePolling";
import { Badge, Empty, ErrorNote, InfoNote, JsonBlock, Panel, Spinner } from "../components/ui";

type Tab = "audit" | "outbox" | "inbound";

const ACTION_TONE = (action: string): "good" | "warn" | "bad" | "info" | "neutral" => {
  if (action.startsWith("review.")) return action.includes("false_alarm") ? "warn" : action.includes("true_defect") ? "bad" : "info";
  if (action.startsWith("thresholds")) return "warn";
  if (action.startsWith("profile") || action.startsWith("sequence") || action.startsWith("pipeline")) return "info";
  if (action.includes("abandon") || action.includes("fail")) return "bad";
  if (action.includes("flush") || action.includes("delivered")) return "good";
  return "neutral";
};

const OUTBOX_TONE = (status: string): "good" | "warn" | "bad" | "muted" => {
  if (status === "delivered") return "good";
  if (status === "pending") return "warn";
  if (status === "dead_letter") return "bad";
  return "muted";
};

function AuditRow({ event }: { event: AuditEvent }) {
  const [open, setOpen] = useState(false);
  const hasDetail = event.detail_json && Object.keys(event.detail_json).length > 0;
  return (
    <div className={`audit-row ${open ? "open" : ""}`}>
      <button className="audit-main" onClick={() => setOpen((value) => !value)} disabled={!hasDetail}>
        <span className="audit-time">
          <strong>{fmtTime(event.ts_ms)}</strong>
          <em>{fmtRelative(event.ts_ms)}</em>
        </span>
        <span className="audit-action">
          <Badge tone={ACTION_TONE(event.action)}>{event.action}</Badge>
          <span className="audit-entity">
            {titleCase(event.entity)}
            {event.entity_id && <em className="mono"> · {event.entity_id}</em>}
          </span>
        </span>
        <span className="audit-actor">{event.actor}</span>
        <span className="audit-toggle">{hasDetail ? open ? <ChevronDown size={15} /> : <ChevronRight size={15} /> : null}</span>
      </button>
      {open && hasDetail && (
        <div className="audit-detail">
          <JsonBlock value={event.detail_json} />
        </div>
      )}
    </div>
  );
}

function OutboxRow({ event, onRetry }: { event: OutboxEvent; onRetry: (id: number) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`outbox-row ${open ? "open" : ""}`}>
      <div className="outbox-main">
        <button className="outbox-expand" onClick={() => setOpen((value) => !value)} aria-label="Toggle payload">
          {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        </button>
        <span className="outbox-time">
          <strong>{fmtTime(event.ts_ms)}</strong>
          <em>{fmtRelative(event.ts_ms)}</em>
        </span>
        <span className="outbox-type">{event.event_type}</span>
        <Badge tone={OUTBOX_TONE(event.status)} dot>{event.status}</Badge>
        <span className="outbox-attempts">attempts {event.attempts}</span>
        <span className="outbox-error">{event.last_error ?? "—"}</span>
        {event.status !== "delivered" && (
          <button className="icon-btn" onClick={() => onRetry(event.id)} title="Retry now" aria-label="Retry outbox event">
            <RotateCcw size={14} />
          </button>
        )}
      </div>
      {open && (
        <div className="outbox-detail">
          <div className="outbox-payload-meta">
            <span>next attempt {event.next_attempt_ms ? fmtDateTime(event.next_attempt_ms) : "—"}</span>
            <span>delivered {event.delivered_at_ms ? fmtDateTime(event.delivered_at_ms) : "—"}</span>
          </div>
          <JsonBlock value={event.payload_json} label="Signed event payload" />
        </div>
      )}
    </div>
  );
}

export default function AuditPage() {
  const audit = usePolling(() => api.audit(250), 8000);
  const outbox = usePolling<OutboxResponse>(() => api.outbox(200), 8000);
  const inbound = usePolling(() => api.mesEvents(50), 12000);
  const [tab, setTab] = useState<Tab>("audit");
  const [query, setQuery] = useState("");
  const [actionFilter, setActionFilter] = useState("all");
  const [outboxFilter, setOutboxFilter] = useState("all");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const events = audit.data?.items ?? [];
  const outboxItems = outbox.data?.items ?? [];
  const status = outbox.data?.status;
  const counts = outbox.data?.counts ?? { inspections: 0, open_reviews: 0, outbox_pending: 0 };

  const actions = useMemo(() => {
    const prefixes = new Set<string>();
    events.forEach((event) => prefixes.add(event.action.split(".")[0]));
    return ["all", ...Array.from(prefixes).sort()];
  }, [events]);

  const filteredAudit = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return events.filter((event) => {
      if (actionFilter !== "all" && !event.action.startsWith(actionFilter)) return false;
      if (!needle) return true;
      return (
        event.action.toLowerCase().includes(needle) ||
        event.actor.toLowerCase().includes(needle) ||
        event.entity.toLowerCase().includes(needle) ||
        (event.entity_id ?? "").toLowerCase().includes(needle)
      );
    });
  }, [events, actionFilter, query]);

  const filteredOutbox = useMemo(() => {
    if (outboxFilter === "all") return outboxItems;
    return outboxItems.filter((event) => event.status === outboxFilter);
  }, [outboxItems, outboxFilter]);

  const flush = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.outboxFlush();
      await outbox.refresh();
      await audit.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const retry = async (id: number) => {
    setError(null);
    try {
      await api.outboxRetry(id);
      await outbox.refresh();
      await audit.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const outboxCounts = useMemo(() => {
    const result: Record<string, number> = { pending: 0, delivered: 0, dead_letter: 0 };
    outboxItems.forEach((event) => {
      result[event.status] = (result[event.status] ?? 0) + 1;
    });
    return result;
  }, [outboxItems]);

  const inboundItems = inbound.data?.items ?? [];

  return (
    <div className="audit-page">
      <div className="page-actions">
        <div className="control-group">
          <button className="btn btn-ghost" onClick={() => { void audit.refresh(); void outbox.refresh(); void inbound.refresh(); }}>
            <RefreshCw size={15} className={audit.loading || outbox.loading ? "spin" : ""} /> Refresh
          </button>
          <button className="btn btn-primary" onClick={() => void flush()} disabled={busy || !counts.outbox_pending}>
            <Send size={15} className={busy ? "spin" : ""} /> Flush outbox (demo sink)
          </button>
        </div>
        <span className="muted-line">
          {status?.mode === "signed_webhook" ? `MES webhook · ${status.url}` : "Outbox-only mode: events are retained locally until a MES URL is configured."}
        </span>
      </div>

      {error && <ErrorNote>{error}</ErrorNote>}
      {(audit.error || outbox.error) && <ErrorNote>{audit.error ?? outbox.error}</ErrorNote>}

      <div className="stats-grid audit-stats">
        <div className="stat">
          <div className="stat-top"><span className="stat-label">Audit events</span><FileClock size={16} className="stat-icon" /></div>
          <div className="stat-value">{fmtInt(events.length)}</div>
          <div className="stat-hint">latest 250 loaded</div>
        </div>
        <div className="stat">
          <div className="stat-top"><span className="stat-label">Outbox pending</span><CloudUpload size={16} className="stat-icon" /></div>
          <div className="stat-value">{fmtInt(counts.outbox_pending)}</div>
          <div className="stat-hint">{status?.configured ? "signed webhook configured" : "local demo sink"}</div>
        </div>
        <div className="stat">
          <div className="stat-top"><span className="stat-label">Delivered</span><CheckCircle2 size={16} className="stat-icon" /></div>
          <div className="stat-value">{fmtInt(outboxCounts.delivered ?? 0)}</div>
          <div className="stat-hint">{status?.delivered ?? 0} by this worker</div>
        </div>
        <div className="stat">
          <div className="stat-top"><span className="stat-label">Dead letters</span><XCircle size={16} className="stat-icon" /></div>
          <div className="stat-value">{fmtInt(outboxCounts.dead_letter ?? 0)}</div>
          <div className="stat-hint">{status?.last_error ? "last error present" : "no terminal failures"}</div>
        </div>
        <div className="stat">
          <div className="stat-top"><span className="stat-label">Open reviews</span><ShieldCheck size={16} className="stat-icon" /></div>
          <div className="stat-value">{fmtInt(counts.open_reviews)}</div>
          <div className="stat-hint">{fmtInt(counts.inspections)} inspections total</div>
        </div>
      </div>

      <div className="audit-tabs">
        <button className={tab === "audit" ? "on" : ""} onClick={() => setTab("audit")}>
          <FileClock size={15} /> Audit trail <em>{events.length}</em>
        </button>
        <button className={tab === "outbox" ? "on" : ""} onClick={() => setTab("outbox")}>
          <ServerCog size={15} /> MES outbox <em>{counts.outbox_pending}</em>
        </button>
        <button className={tab === "inbound" ? "on" : ""} onClick={() => setTab("inbound")}>
          <Inbox size={15} /> Inbound events <em>{inboundItems.length}</em>
        </button>
      </div>

      {tab === "audit" && (
        <Panel
          title="Immutable operator trail"
          subtitle="Threshold changes, profile switches, dispositions, abandonments, and retries"
          actions={
            <div className="table-filters">
              <label className="search-box compact">
                <Search size={14} />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter events" />
              </label>
              <label className="inline-field">
                <Filter size={14} />
                <select value={actionFilter} onChange={(event) => setActionFilter(event.target.value)}>
                  {actions.map((action) => (
                    <option key={action} value={action}>
                      {action === "all" ? "All actions" : titleCase(action)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          }
          flush
        >
          <div className="audit-head">
            <span>When</span>
            <span>Action · entity</span>
            <span>Actor</span>
            <span />
          </div>
          {audit.loading && !events.length ? (
            <Spinner label="Loading audit events" />
          ) : filteredAudit.length ? (
            <div className="audit-list">
              {filteredAudit.map((event) => (
                <AuditRow key={event.id} event={event} />
              ))}
            </div>
          ) : (
            <Empty icon={<Search size={22} />} title="No matching events" hint="Clear the filters to see the full trail." />
          )}
        </Panel>
      )}

      {tab === "outbox" && (
        <div className="outbox-layout">
          <Panel
            title="MES outbox"
            subtitle="Transactional outbox with HMAC-signed delivery, exponential retry, and dead-lettering"
            actions={
              <label className="inline-field">
                <Filter size={14} />
                <select value={outboxFilter} onChange={(event) => setOutboxFilter(event.target.value)}>
                  <option value="all">All states</option>
                  <option value="pending">Pending</option>
                  <option value="delivered">Delivered</option>
                  <option value="dead_letter">Dead letter</option>
                </select>
              </label>
            }
            flush
          >
            <div className="outbox-status">
              <div className={`status-orb ${status?.configured ? "on" : "off"}`} />
              <div>
                <strong>{status?.mode === "signed_webhook" ? "Signed webhook delivery" : "Outbox-only (demo)"}</strong>
                <span>{status?.url ?? "No INDORE_MES_URL configured · local sink only"}</span>
              </div>
              <div className="outbox-status-facts">
                <span>last run {status?.last_run_ms ? fmtRelative(status.last_run_ms) : "—"}</span>
                <span>last error {status?.last_error || "none"}</span>
              </div>
            </div>

            <div className="outbox-head">
              <span />
              <span>When</span>
              <span>Event</span>
              <span>Status</span>
              <span>Attempts</span>
              <span>Last error</span>
              <span />
            </div>
            {outbox.loading && !outboxItems.length ? (
              <Spinner label="Loading outbox" />
            ) : filteredOutbox.length ? (
              <div className="outbox-list">
                {filteredOutbox.map((event) => (
                  <OutboxRow key={event.id} event={event} onRetry={(id) => void retry(id)} />
                ))}
              </div>
            ) : (
              <Empty icon={<Inbox size={22} />} title="Outbox is empty" hint="Inspections and dispositions append events here." />
            )}
          </Panel>

          <div className="outbox-side">
            <Panel title="Delivery contract" subtitle="What the MES receives">
              <ul className="contract-list">
                <li><TerminalSquare size={14} /> POST JSON with `X-Indore-Signature: sha256=…`</li>
                <li><Link2 size={14} /> Header secret from `INDORE_WEBHOOK_SECRET`</li>
                <li><Database size={14} /> Transactional: event persists before delivery</li>
                <li><RotateCcw size={14} /> 5 attempts, exponential backoff, then dead-letter</li>
                <li><ShieldCheck size={14} /> Inbound events validated against the same secret</li>
              </ul>
              <InfoNote tone="warn">
                The “flush” action marks pending rows delivered to a local demo sink. It is explicitly not a claim that
                an external MES acknowledged them.
              </InfoNote>
            </Panel>

            <Panel title="Event types" subtitle="Emitted by the inspection pipeline">
              <div className="event-types">
                <div><strong>inspection.closed</strong><span>verdict, route, severity, z, evidence URL</span></div>
                <div><strong>mes.route</strong><span>rework / quarantine destination</span></div>
                <div><strong>inspection.disposition</strong><span>operator decision + label</span></div>
              </div>
            </Panel>
          </div>
        </div>
      )}

      {tab === "inbound" && (
        <Panel
          title="Inbound MES events"
          subtitle="HMAC-verified callbacks stored verbatim"
          actions={<Badge tone="info" dot>POST /api/mes/events</Badge>}
          flush
        >
          {inbound.error && <ErrorNote>{inbound.error}</ErrorNote>}
          {inboundItems.length ? (
            <div className="inbound-list">
              {inboundItems.map((event: any) => (
                <div key={event.id} className="inbound-row">
                  <div className="inbound-head">
                    <Badge tone="info">{event.event_type ?? "unknown"}</Badge>
                    <span className="mono">{fmtDateTime(event.ts_ms)}</span>
                    <span className="mono inbound-sig">{event.signature ?? "unsigned"}</span>
                  </div>
                  <JsonBlock value={event.payload_json} />
                </div>
              ))}
            </div>
          ) : (
            <Empty
              icon={<Inbox size={22} />}
              title="No inbound events"
              hint="POST a signed JSON body to /api/mes/events to see it recorded here."
            />
          )}
          <InfoNote tone="warn">
            This UI is read-only for inbound events. Invalid signatures are rejected with HTTP 401 and leave no row.
          </InfoNote>
        </Panel>
      )}

      <div className="audit-footnote">
        <AlertTriangle size={14} /> Audit rows are append-only by application convention. This demo uses SQLite file
        access, not an immutable WORM store.
      </div>
    </div>
  );
}
