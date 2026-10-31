export type Verdict = "pass" | "review" | "fail";
export type Disposition = "true_defect" | "false_alarm" | "reclassify";

export interface Detection {
  id: string;
  class: string;
  conf: number;
  bbox: number[];
  severity: number;
  source: string;
  votes: number;
  area_frac: number;
}

export interface AnomalyRegion {
  bbox: number[];
  score: number;
  area: number;
}

export interface AnomalyInfo {
  raw_score: number;
  z: number;
  threshold: number;
  is_anomaly: boolean;
  model: string;
  gate_trusted: boolean;
  gate_quality: string;
  separation: number | null;
  regions: AnomalyRegion[];
  heatmap_url: string | null;
  cached: boolean;
}

export interface LatencyMs {
  capture: number;
  preprocess: number;
  infer_detector: number;
  infer_gate: number;
  post: number;
  encode: number;
  total: number;
}

export interface PartState {
  index: number;
  total: number;
  item_id: string;
  kind: string;
  ground_truth_class: string | null;
  ground_truth_kind: string | null;
  frame_in_part: number;
  frames_in_part: number;
}

export interface FrameResult {
  schema_version: string;
  frame_id: string;
  inspection_id: string;
  ts_ms: number;
  stream_id: string;
  profile: string;
  source: string;
  device: string;
  engine: string;
  model_version: string;
  threshold_profile: string;
  latency_ms: LatencyMs;
  detections: Detection[];
  anomaly: AnomalyInfo;
  verdict: Verdict;
  routing: { destination: string; reason: string } | null;
  severity: number;
  part_state: PartState;
  image_url: string | null;
  overlay_url: string | null;
  crop_url: string | null;
  fps: number;
  frame_index: number;
  warnings: string[];
}

export interface ProfileInfo {
  id: string;
  display_name: string;
  subtitle: string;
  detector: boolean;
  gate_ready: boolean;
  gate_model: string | null;
  gate_separation: number | null;
  gate_quality: string;
  license: string;
  classes: string[];
  counts: Record<string, number>;
  heldout_class: string | null;
  active?: boolean;
}

export interface Inspection {
  id: string;
  ts_ms: number;
  profile: string;
  source: string;
  status: string;
  verdict: Verdict | string;
  final_verdict: Verdict | string | null;
  severity: number;
  route: string | null;
  reason: string | null;
  anomaly_z: number;
  anomaly_raw: number;
  latency_total_ms: number;
  frame_count: number;
  detections: Detection[];
  ground_truth_kind: string | null;
  ground_truth_class: string | null;
  operator_disposition: string | null;
  operator_label: string | null;
  operator_note: string | null;
  reviewed_at_ms: number | null;
  image_url: string | null;
  heatmap_url: string | null;
  overlay_url: string | null;
  crop_url: string | null;
  image_sha?: string | null;
  heatmap_sha?: string | null;
  overlay_sha?: string | null;
  crop_sha?: string | null;
  model_version: string | null;
  threshold_profile: string | null;
  gate_trusted?: boolean;
  warnings: string[];
  defects?: { id: string; class_name: string; conf: number; bbox: number[]; severity: number }[];
}

export interface Metrics {
  total: number;
  verdicts: Record<string, number>;
  pass_rate: number;
  defect_rate: number;
  fp_per_1000: number | null;
  good_total: number;
  fp_events: number;
  escapes: number;
  escapes_per_1000: number | null;
  bad_total: number;
  unknown_routing_rate: number;
  unknown_events: number;
  operator_reject_rate: number | null;
  dispositions: Record<string, number>;
  latency: { p50: number; p95: number; p99: number; mean: number; n: number };
  anomaly_z: { p50: number; p95: number; max: number };
  throughput_per_min: number;
  profiles: Record<string, number>;
  severities: Record<string, number>;
  classes: Record<string, number>;
  gate: Record<string, { quality: string; trusted: boolean; separation: number | null; model: string | null }>;
  warnings: string[];
  counts: { inspections: number; open_reviews: number; outbox_pending: number };
}

export interface AuditEvent {
  id: number;
  ts_ms: number;
  actor: string;
  action: string;
  entity: string;
  entity_id: string | null;
  detail_json: Record<string, unknown>;
}

export interface OutboxEvent {
  id: number;
  ts_ms: number;
  event_type: string;
  payload_json: Record<string, unknown>;
  status: string;
  attempts: number;
  next_attempt_ms: number;
  last_error: string | null;
  delivered_at_ms: number | null;
}

export interface Threshold {
  id: string;
  name: string;
  pass_max: number;
  review_max: number;
  unknown_z: number;
  conf_min: number;
  active: number;
  updated_at_ms: number;
}

export interface Health {
  status: string;
  version: string;
  uptime_s: number;
  profile: string;
  source: string;
  detector_mode: string;
  detector_engine: string;
  gate_ready: boolean;
  gate_trusted: boolean;
  device: string;
  db: { inspections: number; open_reviews: number; outbox_pending: number };
  fps: number;
  ws_clients: number;
  warnings: string[];
}

export interface ControlState {
  paused: boolean;
  profile: string;
  profile_name: string;
  source: string;
  sequence: string;
  detector: {
    engine: string;
    device: string;
    model_version: string;
    ready: boolean;
    warning: string;
    weights?: string;
  };
  gate: {
    ready: boolean;
    device: string;
    error: string | null;
    load_ms: number;
    profile: string | null;
    gate_trusted: boolean;
    gate_quality: string;
    separation: number | null;
  };
  device: string;
  engine: string;
  fps: number;
  frame_index: number;
  part_index: number;
  part: {
    inspection_id: string;
    kind: string;
    gt_class: string | null;
    frame: number;
    frames: number;
  } | null;
  sequence_total: number;
  uptime_s: number;
  warnings: string[];
  profiles?: { id: string; name: string; detector: boolean; gate_quality: string; gate_trusted: boolean }[];
  sequence_modes?: string[];
  detector_modes?: string[];
}

export interface DatasetSplit {
  count: number;
  per_class: Record<string, number>;
  missing: number;
  sha256: string | null;
}

export interface DatasetProfile {
  id: string;
  display_name: string;
  detector: boolean;
  gate_quality: string;
  gate_trusted: boolean;
  gate_separation: number | null;
  gate_check: Record<string, any>;
  counts: Record<string, number>;
  missing: number;
  license: string;
  heldout_class: string | null;
}

export interface DatasetReport {
  neu: {
    dataset?: string;
    source?: string;
    seed?: number;
    classes?: string[];
    trained_classes?: string[];
    heldout_class?: string;
    splits?: Record<string, DatasetSplit>;
    heldout_leak_free?: boolean;
    heldout_leaks?: string[];
    total?: number;
  } & Record<string, any>;
  mvtec: {
    available?: boolean;
    root?: string;
    categories?: string[];
    category_count?: number;
    total_train_good?: number;
    total_test?: number;
    total_masks?: number;
    license?: string;
  } & Record<string, any>;
  profiles: DatasetProfile[];
  warnings: string[];
  verified_at: string;
}

export interface EvalOperatingPoint {
  conf: number;
  test_hit: number;
  heldout_hit: number;
  good_fp: number;
}

export interface EvalSummary {
  available: boolean;
  montage_url?: string | null;
  operating_points?: EvalOperatingPoint[];
  operating_scan_status?: string;
  operating_scan_note?: string;
  metrics?: {
    weights?: string;
    split?: string;
    imgsz?: number;
    overall?: {
      precision: number;
      recall: number;
      map50: number;
      map50_95: number;
      n_images?: number;
    };
    per_class?: Record<
      string,
      { precision: number; recall: number; map50: number; map50_95: number; n_images?: number | null }
    >;
    speed_ms?: Record<string, number>;
    heldout_scan?: {
      folder: string;
      n_images: number;
      images_with_detection: number;
      detection_rate: number;
      boxes: number;
      by_class: Record<string, number>;
      conf_max: number;
      conf_median: number;
    };
    good_scan?: {
      folder: string;
      n_images: number;
      images_with_detection: number;
      detection_rate: number;
      boxes: number;
      by_class: Record<string, number>;
      conf_max: number;
      conf_median: number;
    };
    created_at_ms?: number;
  };
}

export interface GateSummary {
  summary: Record<string, any>;
  active: {
    profile: string;
    quality: string;
    trusted: boolean;
    separation: number | null;
    ready: boolean;
    device: string;
    error: string | null;
  };
  profiles: { id: string; quality: string; trusted: boolean; separation: number | null; model: string | null }[];
}

export interface ModelVersion {
  id: string;
  kind: string;
  name: string;
  path: string;
  sha256: string;
  metrics_json: Record<string, any>;
  created_at_ms: number;
  active: number;
}

export interface ModelInfo {
  versions: ModelVersion[];
  artifacts: { name: string; size: number; modified_ms: number; kind: string }[];
  runtime: {
    detector: Record<string, any>;
    gate: Record<string, any>;
    device: string;
    engine: string;
    promoted_weights: string | null;
    training_weights: string | null;
  };
  profiles: { id: string; gate_model: string | null; gate_quality: string; gate_trusted: boolean; separation: number | null }[];
}

export interface OutboxResponse {
  items: OutboxEvent[];
  status: {
    configured: boolean;
    url: string | null;
    last_error: string;
    last_run_ms: number;
    delivered: number;
    failed: number;
    mode: string;
  };
  counts: { inspections: number; open_reviews: number; outbox_pending: number };
}
