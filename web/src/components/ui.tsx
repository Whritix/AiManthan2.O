import type { ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Info, Loader2, ShieldAlert, XCircle } from "lucide-react";
import type { Verdict } from "../lib/types";
import { titleCase } from "../lib/format";

type Tone = "neutral" | "good" | "warn" | "bad" | "info" | "muted";

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className = "",
  bodyClassName = "",
  flush = false,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  flush?: boolean;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <header className="panel-head">
          <div className="panel-title-wrap">
            {title && <h2 className="panel-title">{title}</h2>}
            {subtitle && <p className="panel-subtitle">{subtitle}</p>}
          </div>
          {actions && <div className="panel-actions">{actions}</div>}
        </header>
      )}
      <div className={`panel-body ${flush ? "panel-body-flush" : ""} ${bodyClassName}`}>{children}</div>
    </section>
  );
}

export function Badge({
  tone = "neutral",
  children,
  dot = false,
  className = "",
}: {
  tone?: Tone;
  children: ReactNode;
  dot?: boolean;
  className?: string;
}) {
  return (
    <span className={`badge badge-${tone} ${className}`}>
      {dot && <span className="badge-dot" aria-hidden="true" />}
      {children}
    </span>
  );
}

const VERDICT_META: Record<
  Verdict,
  { label: string; tone: Tone; icon: typeof CheckCircle2; blurb: string }
> = {
  pass: {
    label: "PASS",
    tone: "good",
    icon: CheckCircle2,
    blurb: "Release to line",
  },
  review: {
    label: "REVIEW",
    tone: "warn",
    icon: AlertTriangle,
    blurb: "Route to human review",
  },
  fail: {
    label: "FAIL",
    tone: "bad",
    icon: XCircle,
    blurb: "Quarantine / rework",
  },
};

export function VerdictBadge({
  verdict,
  size = "sm",
  provisional = false,
}: {
  verdict: string | null | undefined;
  size?: "sm" | "lg" | "hero";
  provisional?: boolean;
}) {
  const key = (verdict === "pass" || verdict === "review" || verdict === "fail" ? verdict : "review") as Verdict;
  const meta = VERDICT_META[key];
  const Icon = meta.icon;
  return (
    <div className={`verdict verdict-${key} verdict-${size}`}>
      <Icon className="verdict-icon" aria-hidden="true" />
      <div className="verdict-copy">
        <span className="verdict-label">{meta.label}</span>
        <span className="verdict-blurb">{provisional ? `${meta.blurb} · provisional` : meta.blurb}</span>
      </div>
    </div>
  );
}

export function VerdictChip({ verdict }: { verdict: string | null | undefined }) {
  const key = verdict === "pass" || verdict === "review" || verdict === "fail" ? verdict : "review";
  return <span className={`verdict-chip verdict-chip-${key}`}>{VERDICT_META[key].label}</span>;
}

export function SeverityMeter({ value, compact = false }: { value: number; compact?: boolean }) {
  const level = Math.max(1, Math.min(4, value || 1));
  return (
    <span className={`severity severity-${level} ${compact ? "severity-compact" : ""}`} title={`Severity ${level} of 4`}>
      {[1, 2, 3, 4].map((n) => (
        <i key={n} className={n <= level ? "on" : ""} />
      ))}
      {!compact && <em>S{level}</em>}
    </span>
  );
}

export function Stat({
  label,
  value,
  unit,
  hint,
  tone = "neutral",
  icon,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  hint?: ReactNode;
  tone?: Tone;
  icon?: ReactNode;
}) {
  return (
    <div className={`stat stat-${tone}`}>
      <div className="stat-top">
        <span className="stat-label">{label}</span>
        {icon && <span className="stat-icon">{icon}</span>}
      </div>
      <div className="stat-value">
        {value}
        {unit && <span className="stat-unit">{unit}</span>}
      </div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="spinner-wrap">
      <Loader2 className="spin" size={18} />
      {label && <span>{label}</span>}
    </div>
  );
}

export function Empty({
  icon,
  title,
  hint,
  action,
}: {
  icon?: ReactNode;
  title: string;
  hint?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      {icon && <div className="empty-icon">{icon}</div>}
      <h3>{title}</h3>
      {hint && <p>{hint}</p>}
      {action && <div className="empty-action">{action}</div>}
    </div>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <div className="note note-bad">
      <AlertTriangle size={15} />
      <span>{children}</span>
    </div>
  );
}

export function InfoNote({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" | "bad" | "good" }) {
  const Icon = tone === "warn" || tone === "bad" ? ShieldAlert : tone === "good" ? CheckCircle2 : Info;
  return (
    <div className={`note note-${tone}`}>
      <Icon size={15} />
      <span>{children}</span>
    </div>
  );
}

export function WarningList({ warnings }: { warnings: string[] }) {
  if (!warnings.length) return <Badge tone="good" dot>no warnings</Badge>;
  return (
    <div className="warn-list">
      {warnings.map((warning) => (
        <Badge key={warning} tone={warningTone(warning)}>
          {humanWarning(warning)}
        </Badge>
      ))}
    </div>
  );
}

export function warningTone(warning: string): Tone {
  if (/failed|error|leak|untrusted|not_validated|does_not_gate|fixture/i.test(warning)) return "bad";
  if (/stale|missing|fallback|degraded|simulated|transient|low_confidence|skipped/i.test(warning)) return "warn";
  return "info";
}

export function humanWarning(warning: string): string {
  return titleCase(warning.replace(/:/g, " · "));
}

export function JsonBlock({ value, label }: { value: unknown; label?: string }) {
  return (
    <div className="json-block">
      {label && <div className="json-label">{label}</div>}
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </div>
  );
}

export function ProgressBar({
  value,
  tone = "accent",
  label,
}: {
  value: number;
  tone?: "accent" | "good" | "warn" | "bad" | "muted";
  label?: string;
}) {
  const pct = `${Math.max(0, Math.min(100, value * 100))}%`;
  return (
    <div className={`progress progress-${tone}`} title={label}>
      <span style={{ width: pct }} />
    </div>
  );
}
