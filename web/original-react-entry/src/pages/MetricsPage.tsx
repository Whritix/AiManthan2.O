import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  Clock,
  Download,
  Gauge,
  Info,
  Percent,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Target,
  TrendingUp,
  XCircle,
  Zap,
} from "lucide-react";
import { api, exportCsvUrl } from "../lib/api";
import { fmtInt, fmtMs, fmtNum, fmtPct, fmtRelative, titleCase } from "../lib/format";
import type { Metrics, ProfileInfo } from "../lib/types";
import { usePolling } from "../hooks/usePolling";
import { Badge, Empty, ErrorNote, InfoNote, Panel, ProgressBar, SeverityMeter, Spinner, Stat } from "../components/ui";

const VERDICT_COLORS: Record<string, string> = {
  pass: "#2dd4bf",
  review: "#f5a524",
  fail: "#f05252",
};

const TOOLTIP_STYLE = {
  background: "#0c131a",
  border: "1px solid #24333f",
  borderRadius: 8,
  fontSize: 12,
  color: "#dbe7ef",
};

function ChartFrame({ children, height = 210 }: { children: React.ReactNode; height?: number }) {
  return <div style={{ width: "100%", height }}>{children}</div>;
}

function MetricTable({ rows }: { rows: { label: string; value: React.ReactNode }[] }) {
  return (
    <dl className="metric-table">
      {rows.map((row) => (
        <div key={row.label}>
          <dt>{row.label}</dt>
          <dd>{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function MetricsPage() {
  const profiles = usePolling<ProfileInfo[]>(api.profiles, 60000);
  const evalSummary = usePolling(api.evalSummary, 0);
  const [profile, setProfile] = useState("");
  const metrics = usePolling<Metrics>(() => api.metrics(profile || undefined), 10000);

  useEffect(() => {
    void metrics.refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profile]);

  const data = metrics.data;
  const verdictData = useMemo(() => {
    if (!data) return [];
    return Object.entries(data.verdicts)
      .filter(([, value]) => value > 0)
      .map(([name, value]) => ({ name: titleCase(name), key: name, value }));
  }, [data]);
  const severityData = useMemo(() => {
    if (!data) return [];
    return ["1", "2", "3", "4"].map((level) => ({
      name: `S${level}`,
      count: data.severities[level] ?? 0,
      level: Number(level),
    }));
  }, [data]);
  const latencyData = useMemo(() => {
    if (!data) return [];
    return [
      { name: "p50", value: data.latency.p50 },
      { name: "p95", value: data.latency.p95 },
      { name: "p99", value: data.latency.p99 },
      { name: "mean", value: data.latency.mean },
    ];
  }, [data]);
  const classData = useMemo(() => {
    if (!data) return [];
    return Object.entries(data.classes)
      .sort((a, b) => b[1] - a[1])
      .map(([name, count]) => ({ name: titleCase(name), count }));
  }, [data]);

  const evalData = evalSummary.data;
  const overall = evalData?.metrics?.overall;
  const perClass = evalData?.metrics?.per_class ?? {};
  const operating = evalData?.operating_points ?? [];

  return (
    <div className="metrics-page">
      <div className="page-actions">
        <div className="control-group">
          <label className="inline-field">
            <span>Profile</span>
            <select value={profile} onChange={(event) => setProfile(event.target.value)}>
              <option value="">All profiles</option>
              {profiles.data?.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.display_name}
                </option>
              ))}
            </select>
          </label>
          <button className="btn btn-ghost" onClick={() => void metrics.refresh()} disabled={metrics.loading}>
            <RefreshCw size={15} className={metrics.loading ? "spin" : ""} /> Refresh
          </button>
        </div>
        <a className="btn btn-ghost" href={exportCsvUrl(profile || undefined)}>
          <Download size={15} /> Export CSV
        </a>
      </div>

      {metrics.error && <ErrorNote>{metrics.error}</ErrorNote>}
      {!data ? (
        <Panel>
          <Spinner label="Loading production metrics" />
        </Panel>
      ) : (
        <>
          <div className="stats-grid">
            <Stat
              label="Inspections"
              value={fmtInt(data.total)}
              hint={`${fmtNum(data.throughput_per_min, 1)} parts/min in last 5 min`}
              icon={<BarChart3 size={16} />}
            />
            <Stat
              label="Pass rate"
              value={fmtPct(data.pass_rate)}
              hint={`${data.verdicts.pass ?? 0} released silently`}
              tone="good"
              icon={<CheckCircle2 size={16} />}
            />
            <Stat
              label="Held or escalated"
              value={fmtInt((data.verdicts.fail ?? 0) + (data.verdicts.review ?? 0))}
              hint={`${data.verdicts.fail ?? 0} fail · ${data.verdicts.review ?? 0} review`}
              tone="warn"
              icon={<AlertTriangle size={16} />}
            />
            <Stat
              label="Good → non-pass"
              value={data.fp_per_1000 === null ? "—" : fmtNum(data.fp_per_1000, 1)}
              unit=" /1k"
              hint={`${data.fp_events} events over ${data.good_total} good parts`}
              tone={data.fp_per_1000 === null ? "neutral" : data.fp_per_1000 > 50 ? "bad" : "good"}
              icon={<XCircle size={16} />}
            />
            <Stat
              label="Defect escapes"
              value={data.escapes_per_1000 === null ? "—" : fmtNum(data.escapes_per_1000, 1)}
              unit=" /1k"
              hint={`${data.escapes} missed over ${data.bad_total} defect parts`}
              tone={data.escapes === 0 ? "good" : "warn"}
              icon={<ShieldAlert size={16} />}
            />
            <Stat
              label="Latency p95"
              value={data.latency.n ? fmtNum(data.latency.p95, 0) : "—"}
              unit="ms"
              hint={`p50 ${fmtNum(data.latency.p50, 0)} ms · n=${data.latency.n}`}
              icon={<Clock size={16} />}
            />
            <Stat
              label="Operator reject"
              value={data.operator_reject_rate === null ? "—" : fmtPct(data.operator_reject_rate)}
              hint={`${data.dispositions.false_alarm} false alarms / ${data.dispositions.true_defect} confirmed`}
              icon={<Gauge size={16} />}
            />
            <Stat
              label="Unknown routing"
              value={fmtPct(data.unknown_routing_rate)}
              hint={`${data.unknown_events} novelty escalations of ${(data.verdicts.fail ?? 0) + (data.verdicts.review ?? 0)} non-pass`}
              tone="info"
              icon={<Target size={16} />}
            />
          </div>

          <div className="metrics-grid">
            <Panel title="Verdict mix" subtitle={`${data.total} closed inspections`}>
              {verdictData.length ? (
                <ChartFrame>
                  <ResponsiveContainer>
                    <PieChart>
                      <Pie
                        data={verdictData}
                        dataKey="value"
                        nameKey="name"
                        innerRadius={54}
                        outerRadius={82}
                        paddingAngle={2}
                        stroke="none"
                      >
                        {verdictData.map((entry) => (
                          <Cell key={entry.key} fill={VERDICT_COLORS[entry.key] ?? "#64748b"} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={TOOLTIP_STYLE} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="pie-center">
                    <strong>{fmtInt(data.total)}</strong>
                    <span>parts</span>
                  </div>
                </ChartFrame>
              ) : (
                <Empty icon={<BarChart3 size={22} />} title="No completed inspections" />
              )}
              <div className="legend">
                {(["pass", "review", "fail"] as const).map((key) => (
                  <span key={key} className="legend-item">
                    <i style={{ background: VERDICT_COLORS[key] }} />
                    {titleCase(key)}
                    <em>{data.verdicts[key] ?? 0}</em>
                  </span>
                ))}
              </div>
            </Panel>

            <Panel title="Severity distribution" subtitle="Max severity committed per part">
              <ChartFrame height={190}>
                <ResponsiveContainer>
                  <BarChart data={severityData} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                    <CartesianGrid stroke="#1b2833" vertical={false} />
                    <XAxis dataKey="name" tick={{ fill: "#8ea3b3", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "#8ea3b3", fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(45,212,191,0.06)" }} />
                    <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                      {severityData.map((entry) => (
                        <Cell
                          key={entry.level}
                          fill={["#2dd4bf", "#a3e635", "#f5a524", "#f05252"][entry.level - 1]}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartFrame>
              <MetricTable
                rows={[
                  { label: "Mean severity", value: meanSeverity(data.severities) },
                  { label: "Gate z p95", value: fmtNum(data.anomaly_z.p95, 2) },
                  { label: "Gate z max", value: fmtNum(data.anomaly_z.max, 2) },
                ]}
              />
            </Panel>

            <Panel title="Latency" subtitle="Part-level total, committed frames">
              <ChartFrame height={190}>
                <ResponsiveContainer>
                  <BarChart data={latencyData} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                    <CartesianGrid stroke="#1b2833" vertical={false} />
                    <XAxis dataKey="name" tick={{ fill: "#8ea3b3", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "#8ea3b3", fontSize: 11 }} axisLine={false} tickLine={false} unit="ms" width={52} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(45,212,191,0.06)" }} />
                    <Bar dataKey="value" fill="#38bdf8" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartFrame>
              <MetricTable
                rows={[
                  { label: "p50", value: fmtMs(data.latency.p50) },
                  { label: "p99", value: fmtMs(data.latency.p99) },
                  { label: "Mean", value: fmtMs(data.latency.mean) },
                ]}
              />
            </Panel>

            <Panel title="Defect classes" subtitle="Supported classes committed to the defect table">
              {classData.length ? (
                <ChartFrame height={190}>
                  <ResponsiveContainer>
                    <BarChart data={classData} layout="vertical" margin={{ top: 4, right: 18, left: 8, bottom: 0 }}>
                      <CartesianGrid stroke="#1b2833" horizontal={false} />
                      <XAxis type="number" tick={{ fill: "#8ea3b3", fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} />
                      <YAxis
                        type="category"
                        dataKey="name"
                        tick={{ fill: "#8ea3b3", fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                        width={92}
                      />
                      <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(45,212,191,0.06)" }} />
                      <Bar dataKey="count" fill="#a78bfa" radius={[0, 3, 3, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartFrame>
              ) : (
                <Empty icon={<Target size={22} />} title="No defects committed" />
              )}
              <div className="legend">
                {Object.entries(data.profiles).map(([name, count]) => (
                  <span key={name} className="legend-item">
                    <i style={{ background: "#94a3b8" }} />
                    {titleCase(name)}
                    <em>{count}</em>
                  </span>
                ))}
              </div>
            </Panel>
          </div>

          <div className="metrics-lower">
            <Panel title="Gate health" subtitle="Per-profile novelty quality and trust">
              <div className="table-wrap">
                <table className="data-table compact">
                  <thead>
                    <tr>
                      <th>Profile</th>
                      <th>Quality</th>
                      <th>Separation</th>
                      <th>Trusted</th>
                      <th>Model</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.gate).map(([profileId, gate]) => (
                      <tr key={profileId}>
                        <td>{titleCase(profileId)}</td>
                        <td>
                          <Badge tone={gate.quality === "strong" || gate.quality === "good" ? "good" : gate.quality === "weak" ? "warn" : "bad"}>
                            {titleCase(gate.quality)}
                          </Badge>
                        </td>
                        <td className="mono">{gate.separation === null ? "—" : gate.separation.toFixed(2)}</td>
                        <td>{gate.trusted ? <Badge tone="good">yes</Badge> : <Badge tone="warn">no</Badge>}</td>
                        <td className="mono small">{gate.model ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <InfoNote tone="warn">
                Steel's gate has negative separation on held-out unknowns. It is displayed but never allowed to drive
                PASS/FAIL for that profile.
              </InfoNote>
            </Panel>

            <Panel title="Dispositions" subtitle="Operator outcome of escalated parts">
              <div className="disposition-bars">
                {(["true_defect", "false_alarm", "reclassify"] as const).map((key) => {
                  const total = Math.max(1, data.dispositions.true_defect + data.dispositions.false_alarm + data.dispositions.reclassify);
                  return (
                    <div key={key} className="disposition-bar">
                      <span>{titleCase(key)}</span>
                      <ProgressBar value={data.dispositions[key] / total} tone={key === "false_alarm" ? "warn" : key === "reclassify" ? "accent" : "bad"} />
                      <em>{data.dispositions[key]}</em>
                    </div>
                  );
                })}
              </div>
              <MetricTable
                rows={[
                  { label: "Open reviews", value: data.counts.open_reviews },
                  { label: "Outbox pending", value: data.counts.outbox_pending },
                  { label: "Total inspections", value: fmtInt(data.counts.inspections) },
                ]}
              />
            </Panel>
          </div>

          <Panel
            title="Offline detector evaluation"
            subtitle="NEU-DET test split, 225 images / 530 instances, 640×640, rect=False; P/R at 0.35"
            actions={
              evalData?.available ? (
                <Badge tone="info" dot>measured</Badge>
              ) : (
                <Badge tone="muted">not exported</Badge>
              )
            }
          >
            {!evalData?.available ? (
              <Empty icon={<BarChart3 size={22} />} title="No offline evaluation artifact" hint="Run ml/eval/evaluate.py to produce artifacts/eval/test_metrics.json." />
            ) : (
              <>
                <div className="eval-grid">
                  <div className="eval-stat">
                    <span>mAP@50</span>
                    <strong>{fmtNum(overall?.map50, 3)}</strong>
                    <em>all classes</em>
                  </div>
                  <div className="eval-stat">
                    <span>mAP@50–95</span>
                    <strong>{fmtNum(overall?.map50_95, 3)}</strong>
                    <em>strict IoU</em>
                  </div>
                  <div className="eval-stat">
                    <span>Precision</span>
                    <strong>{fmtNum(overall?.precision, 3)}</strong>
                    <em>at 0.35</em>
                  </div>
                  <div className="eval-stat">
                    <span>Recall</span>
                    <strong>{fmtNum(overall?.recall, 3)}</strong>
                    <em>at 0.35</em>
                  </div>
                  <div className="eval-stat">
                    <span>Heldout hit</span>
                    <strong>{fmtPct(evalData.metrics?.heldout_scan?.detection_rate)}</strong>
                    <em>rolled-in_scale @0.35</em>
                  </div>
                  <div className="eval-stat">
                    <span>Clean-crop FP</span>
                    <strong>{fmtPct(evalData.metrics?.good_scan?.detection_rate)}</strong>
                    <em>synthetic good @0.35</em>
                  </div>
                </div>

                <div className="table-wrap">
                  <table className="data-table compact">
                    <thead>
                      <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>mAP@50</th>
                        <th>mAP@50–95</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(perClass).map(([name, row]) => (
                        <tr key={name}>
                          <td>{titleCase(name)}</td>
                          <td className="mono">{fmtNum(row.precision, 3)}</td>
                          <td className="mono">{fmtNum(row.recall, 3)}</td>
                          <td className="mono">{fmtNum(row.map50, 3)}</td>
                          <td className="mono">{fmtNum(row.map50_95, 3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <h3 className="subhead">Operating-point scan</h3>
                {operating.length === 0 ? (
                  <InfoNote tone="warn">
                    {evalData?.operating_scan_note ||
                      "Operating-point scan pending a re-run on the promoted weights."}
                  </InfoNote>
                ) : (
                  <div className="table-wrap">
                    <table className="data-table compact">
                      <thead>
                        <tr>
                          <th>Conf floor</th>
                          <th>Known test hit</th>
                          <th>Heldout hit</th>
                          <th>Good-crop FP</th>
                        </tr>
                      </thead>
                      <tbody>
                        {operating.map((point) => (
                          <tr key={point.conf} className={Math.abs(point.conf - 0.35) < 0.001 ? "row-active" : ""}>
                            <td className="mono">{point.conf.toFixed(2)}</td>
                            <td className="mono">{fmtPct(point.test_hit)}</td>
                            <td className="mono">{fmtPct(point.heldout_hit)}</td>
                            <td className="mono">{fmtPct(point.good_fp)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                <div className="eval-notes">
                  <InfoNote tone="warn">
                    Held-out <strong>rolled-in_scale</strong> never enters detector training. On the 640 path, 24.3% of
                    held-out images fire at 0.35 and 10.3% at 0.55. These are detector fire rates only: the steel gate
                    AUROC is 0.632, so held-out is <strong>not</strong> escalated as UNKNOWN.
                  </InfoNote>
                  <InfoNote tone="warn">
                    The good-crop scan uses inpainting-derived clean steel. The 640 detector false-fires on 18.8% of
                    clean images at 0.35 and 7.8% at 0.55. This is a measured limitation, not a display bug.
                  </InfoNote>
                </div>
                {evalSummary.data?.montage_url && (
                  <a className="text-link" href={evalSummary.data.montage_url} target="_blank" rel="noreferrer">
                    View false-positive montage <ArrowUpRight size={13} />
                  </a>
                )}
              </>
            )}
          </Panel>

          <div className="metrics-footnote">
            <Info size={14} />
            Online metrics are computed from replayed ground truth and operator dispositions. Offline metrics are
            measured on the locked test split. The two must not be conflated.
          </div>
        </>
      )}
    </div>
  );
}

function meanSeverity(severities: Record<string, number>): string {
  const total = Object.values(severities).reduce((sum, value) => sum + value, 0);
  if (!total) return "—";
  return (Object.entries(severities).reduce((sum, [level, count]) => sum + Number(level) * count, 0) / total).toFixed(2);
}
