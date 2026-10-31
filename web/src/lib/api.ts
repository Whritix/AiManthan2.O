import type {
  AuditEvent,
  ControlState,
  DatasetReport,
  Disposition,
  EvalSummary,
  FrameResult,
  GateSummary,
  Health,
  Inspection,
  Metrics,
  ModelInfo,
  OutboxResponse,
  ProfileInfo,
  Threshold,
} from "./types";

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

const qs = (params: Record<string, string | number | boolean | undefined>) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
};

export const api = {
  health: () => request<Health>("/api/health"),
  profiles: () => request<ProfileInfo[]>("/api/profiles"),
  profile: (id: string) => request<any>(`/api/profiles/${id}`),
  gate: () => request<GateSummary>("/api/gate/summary"),
  datasets: (force = false) => request<DatasetReport>(`/api/datasets/verify${force ? "?force=true" : ""}`),
  evalSummary: () => request<EvalSummary>("/api/eval/summary"),
  latest: () => request<{ seq: number; frame: FrameResult | null; fps: number }>("/api/frames/latest"),
  recentFrames: () => request<{ frames: FrameResult[]; fps: number }>("/api/frames/recent"),
  controlState: () => request<ControlState>("/api/control/state"),
  control: (action: string, value?: string) =>
    request<{ ok: boolean; state: ControlState }>("/api/control", {
      method: "POST",
      body: JSON.stringify({ action, value: value ?? null }),
    }),
  uploadInput: async (file: File) => {
    const res = await fetch(`/api/input?filename=${encodeURIComponent(file.name)}`, {
      method: "PUT",
      headers: { "Content-Type": file.type || "application/octet-stream" },
      body: file,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch { /* ignore non-JSON errors */ }
      throw new ApiError(res.status, detail);
    }
    return (await res.json()) as { ok: boolean; filename: string; bytes: number; source: string };
  },
  inspections: (params: {
    limit?: number;
    offset?: number;
    verdict?: string;
    profile?: string;
    status?: string;
  }) => request<{ items: Inspection[]; total: number; limit: number; offset: number }>(`/api/inspections${qs(params)}`),
  inspection: (id: string) => request<Inspection>(`/api/inspections/${id}`),
  reviewQueue: () => request<{ items: Inspection[]; count: number; open_reviews: number }>("/api/review/queue"),
  review: (id: string, disposition: Disposition, label?: string, note?: string) =>
    request<Inspection>(`/api/inspections/${id}/review`, {
      method: "POST",
      body: JSON.stringify({ disposition, label: label ?? null, note: note ?? null }),
    }),
  metrics: (profile?: string) => request<Metrics>(`/api/metrics${qs({ profile })}`),
  audit: (limit = 100) => request<{ items: AuditEvent[] }>(`/api/audit?limit=${limit}`),
  outbox: (limit = 100) => request<OutboxResponse>(`/api/outbox?limit=${limit}`),
  outboxFlush: () => request<{ ok: boolean; flushed: number }>("/api/outbox/flush", { method: "POST" }),
  outboxRetry: (id: number) => request<{ ok: boolean }>(`/api/outbox/${id}/retry`, { method: "POST" }),
  mesEvents: (limit = 50) => request<{ items: any[] }>(`/api/mes/events?limit=${limit}`),
  thresholds: () => request<Threshold>("/api/thresholds"),
  updateThresholds: (body: Partial<Threshold> & { name?: string }) =>
    request<Threshold>("/api/thresholds", { method: "PATCH", body: JSON.stringify(body) }),
  models: () => request<ModelInfo>("/api/models"),
  config: () => request<Record<string, any>>("/api/config"),
};

export const mediaUrl = (path: string | null | undefined) => {
  if (!path) return null;
  if (path.startsWith("http") || path.startsWith("data:")) return path;
  return path;
};

export const exportCsvUrl = (profile?: string) => `/api/export/inspections.csv${qs({ profile })}`;
