import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  ClipboardCheck,
  Database,
  FileClock,
  Gauge,
  Radio,
  Settings as SettingsIcon,
  ShieldCheck,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useLive } from "../hooks/useLive";
import { useNow } from "../hooks/usePolling";
import { fmtTime, fmtUptime, titleCase } from "../lib/format";
import { Badge } from "./ui";

const NAV = [
  { to: "/", label: "Live Cell", icon: Activity, end: true },
  { to: "/review", label: "Review Queue", icon: ClipboardCheck },
  { to: "/metrics", label: "Metrics", icon: BarChart3 },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/audit", label: "Audit & MES", icon: FileClock },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
];

const PAGE_META: Record<string, { title: string; subtitle: string }> = {
  "/": { title: "Live Inspection Cell", subtitle: "Detector + novelty gate fusion, part by part." },
  "/review": { title: "Human Review", subtitle: "Disposition every escalated part with evidence attached." },
  "/metrics": { title: "Production Metrics", subtitle: "Replayed ground truth, latency, and operating-point health." },
  "/datasets": { title: "Dataset Evidence", subtitle: "Source verification, held-out isolation, and gate quality." },
  "/audit": { title: "Audit & Integration", subtitle: "Immutable operator trail and signed MES outbox." },
  "/settings": { title: "Cell Configuration", subtitle: "Thresholds, models, and runtime contract." },
};

function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 64 64" aria-hidden="true">
      <rect width="64" height="64" rx="12" fill="#0d151d" />
      <path d="M14 18h8v28h-8zM42 18h8v28h-8z" fill="#2dd4bf" />
      <path d="M22 32h20v8H22z" fill="#f5a524" />
      <circle cx="32" cy="22" r="4" fill="#dbe7ef" />
    </svg>
  );
}

function Clock() {
  const now = useNow(1000);
  return (
    <div className="clock">
      <span className="clock-time">{fmtTime(now)}</span>
      <span className="clock-date">{new Date(now).toLocaleDateString("en-CA")}</span>
    </div>
  );
}

export default function Layout() {
  const location = useLocation();
  const { health, control, frame, fps, connected, lastFrameAt, error } = useLive();
  const now = useNow(1000);
  const meta = PAGE_META[location.pathname] ?? PAGE_META["/"];
  const source = control?.source ?? health?.source ?? frame?.source ?? "dataset";
  const detectorMode = health?.detector_mode ?? "auto";
  const fixture =
    detectorMode === "fixture" ||
    control?.detector?.warning?.includes("fixture") === true ||
    frame?.engine?.includes("fixture") === true;
  const gateUntrusted = control?.gate?.gate_trusted === false || health?.gate_trusted === false;
  const paused = control?.paused === true;
  const stale = !!lastFrameAt && now - lastFrameAt > 6000 && !paused;
  const openReviews = health?.db?.open_reviews ?? 0;
  const outboxPending = health?.db?.outbox_pending ?? 0;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <BrandMark />
          <div className="brand-text">
            <strong>INDORE</strong>
            <span>QA CELL · v{health?.version ?? "1.0.0"}</span>
          </div>
        </div>

        <nav className="nav" aria-label="Primary">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
            >
              <Icon size={17} strokeWidth={1.8} />
              <span>{label}</span>
              {to === "/review" && openReviews > 0 && <em className="nav-badge">{openReviews}</em>}
              {to === "/audit" && outboxPending > 0 && <em className="nav-badge nav-badge-muted">{outboxPending}</em>}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-card">
          <div className="sidebar-card-head">
            <Radio size={14} />
            <span>Cell status</span>
          </div>
          <dl>
            <div>
              <dt>Profile</dt>
              <dd>{titleCase(control?.profile_name ?? health?.profile)}</dd>
            </div>
            <div>
              <dt>Detector</dt>
              <dd className={fixture ? "text-warn" : "text-good"}>
                {fixture ? "Fixture GT" : health?.detector_engine ?? "—"}
              </dd>
            </div>
            <div>
              <dt>Gate</dt>
              <dd className={gateUntrusted ? "text-warn" : "text-good"}>
                {gateUntrusted ? "Untrusted" : health?.gate_ready ? "Armed" : "Absent"}
              </dd>
            </div>
            <div>
              <dt>Uptime</dt>
              <dd>{fmtUptime(health?.uptime_s)}</dd>
            </div>
          </dl>
        </div>

        <div className="sidebar-foot">
          <p>
            NEU-DET · MVTec AD
            <br />
            <span>Research demo. Not production-safety rated.</span>
          </p>
          <a href="/docs" target="_blank" rel="noreferrer">
            API documentation ↗
          </a>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <div className="page-heading">
            <h1>{meta.title}</h1>
            <p>{meta.subtitle}</p>
          </div>
          <div className="topbar-right">
            <span className={`conn ${connected ? "conn-on" : "conn-off"}`}>
              {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
              {connected ? "WS LINK" : "HTTP POLL"}
            </span>
            <span className={`fps ${stale ? "fps-stale" : ""}`}>
              <Gauge size={14} />
              {fps.toFixed(1)} FPS
            </span>
            <span className={`paused ${paused ? "is-paused" : ""}`}>
              {paused ? <AlertTriangle size={14} /> : <ShieldCheck size={14} />}
              {paused ? "PAUSED" : "RUNNING"}
            </span>
            <Clock />
          </div>
        </header>

        <div className="integrity-strip">
          {source === "dataset" && (
            <span className="integrity-tag integrity-sim">
              <Radio size={13} />
              SIMULATED FEED · DATASET REPLAY
            </span>
          )}
          {source === "camera" && (
            <span className="integrity-tag integrity-live">
              <Radio size={13} />
              LIVE CAMERA FEED
            </span>
          )}
          {fixture && (
            <span className="integrity-tag integrity-fixture">
              <AlertTriangle size={13} />
              FIXTURE INFERENCE · GROUND-TRUTH BOXES, NOT MODEL OUTPUT
            </span>
          )}
          {gateUntrusted && (
            <span className="integrity-tag integrity-gate">
              <ShieldCheck size={13} />
              NOVELTY GATE NOT VALIDATED · DETECTOR-FIRST
            </span>
          )}
          {stale && (
            <span className="integrity-tag integrity-stale">
              <WifiOff size={13} />
              TELEMETRY STALE
            </span>
          )}
          <span className="integrity-spacer" />
          {error && <span className="integrity-error">{error}</span>}
          <span className="integrity-meta">
            DB {health?.db?.inspections ?? 0} inspections · {openReviews} open reviews
          </span>
        </div>

        <main className="content">
          <Outlet />
        </main>

        <footer className="app-foot">
          <span>INDORE QA CELL · LOCAL-FIRST INSPECTION DEMO</span>
          <span>
            NEU-DET (CC BY 4.0) · MVTec AD (CC BY-NC-SA 4.0, non-commercial) · Built for research evaluation
          </span>
        </footer>
      </div>
    </div>
  );
}
