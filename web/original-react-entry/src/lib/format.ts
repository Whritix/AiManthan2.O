export const fmtInt = (value: number | null | undefined): string =>
  value === null || value === undefined || Number.isNaN(value)
    ? "—"
    : new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);

export const fmtNum = (value: number | null | undefined, digits = 2): string =>
  value === null || value === undefined || Number.isNaN(value) ? "—" : value.toFixed(digits);

export const fmtPct = (value: number | null | undefined, digits = 1): string =>
  value === null || value === undefined || Number.isNaN(value) ? "—" : `${(value * 100).toFixed(digits)}%`;

export const fmtMs = (value: number | null | undefined, digits = 1): string =>
  value === null || value === undefined || Number.isNaN(value) ? "—" : `${value.toFixed(digits)} ms`;

export const fmtTime = (ts: number | null | undefined): string => {
  if (!ts) return "—";
  return new Date(ts).toLocaleTimeString("en-GB", { hour12: false });
};

export const fmtDateTime = (ts: number | null | undefined): string => {
  if (!ts) return "—";
  const d = new Date(ts);
  return `${d.toLocaleDateString("en-CA")} ${d.toLocaleTimeString("en-GB", { hour12: false })}`;
};

export const fmtRelative = (ts: number | null | undefined): string => {
  if (!ts) return "—";
  const delta = Math.max(0, Date.now() - ts);
  if (delta < 4_000) return "just now";
  if (delta < 60_000) return `${Math.round(delta / 1000)}s ago`;
  if (delta < 3_600_000) return `${Math.round(delta / 60_000)}m ago`;
  if (delta < 86_400_000) return `${Math.round(delta / 3_600_000)}h ago`;
  return `${Math.round(delta / 86_400_000)}d ago`;
};

export const titleCase = (value: string | null | undefined): string => {
  if (!value) return "—";
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
};

export const shortId = (value: string | null | undefined, head = 10): string =>
  value ? `${value.slice(0, head)}${value.length > head ? "…" : ""}` : "—";

export const shortHash = (value: string | null | undefined): string =>
  value ? `${value.slice(0, 10)}…${value.slice(-4)}` : "—";

export const fmtBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
};

export const fmtUptime = (seconds: number | null | undefined): string => {
  if (!seconds && seconds !== 0) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
};

export const pctWidth = (value: number): string => `${Math.max(0, Math.min(100, value * 100))}%`;
