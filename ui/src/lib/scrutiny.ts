import {
  AgentDataItem,
  useCloudApiClient,
  useHandlers,
  useWorkflow,
} from "@llamaindex/ui";
import { useCallback, useEffect, useRef, useState } from "react";
import { downloadFile, downloadJSON } from "./export";

export type ResultState =
  | "defect_found"
  | "compliant"
  | "not_applicable"
  | "not_determined"
  | "needs_review";

export interface EvidenceRef {
  page: number | null;
  quote: string;
}

export interface SubcheckResult {
  subcheck_id: string;
  status: ResultState;
  confidence: number;
  reasoning: string;
  evidence: EvidenceRef[];
  suggested_fix: string | null;
  fix_rationale: string | null;
}

export interface LlmUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens?: number;
  reasoning_tokens?: number;
  calls?: number;
  cost_usd?: number | null;
  model?: string | null;
  generation_id?: string | null;
  generation_ids?: string[];
}

export interface UsageByCheck {
  check_id: string;
  serial_no: number;
  cost_usd?: number | null;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  llm_calls: number;
  share?: number | null;
}

export interface UsageSummary {
  cost_usd?: number | null;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens?: number;
  reasoning_tokens?: number;
  llm_calls: number;
  model?: string | null;
  highest_cost_check_id?: string | null;
  highest_cost_serial_no?: number | null;
  highest_cost_usd?: number | null;
  by_check?: UsageByCheck[];
  note?: string;
}

export interface Coverage {
  chunks_reviewed: number;
  pages_reviewed: number[];
  structured_record_available: boolean;
  evidence_complete: boolean;
}

export interface DefectFinding {
  check_id: string;
  serial_no?: number;
  title: string;
  main_category?: string;
  special_category?: string | null;
  status: ResultState;
  summary: string;
  confidence: number;
  reasoning?: string;
  evidence?: EvidenceRef[];
  suggested_fix?: string | null;
  fix_rationale?: string | null;
  how_to_cure?: string[];
  applicable_rule?: string | null;
  location?: string | null;
  location_source?: string | null;
  evidence_ids?: string[];
  coverage: Coverage;
  usage?: LlmUsage | null;
  error?: string | null;
  /** Present on reports saved before the flattened defect format. */
  severity?: string;
  subcheck_results?: SubcheckResult[];
  authority_refs?: string[];
}

export interface ScrutinySummary {
  total_defects: number;
  defects_found: number;
  compliant: number;
  needs_review: number;
  not_determined: number;
  not_applicable: number;
  overall_confidence: number;
}

export interface ScrutinyReport {
  schema_name: string;
  catalogue_id: string;
  catalogue_version: string;
  agent_data_id: string | null;
  file_hash: string | null;
  file_name: string | null;
  petition_type: string | null;
  model: string | null;
  generated_at: string;
  disclaimer: string | null;
  findings: DefectFinding[];
  summary: ScrutinySummary;
  usage?: UsageSummary | null;
  planned_checks?: number | null;
  stopped_early?: boolean;
}

export const RESULT_LABELS: Record<ResultState, string> = {
  defect_found: "Defect found",
  compliant: "Compliant",
  not_applicable: "Not applicable",
  not_determined: "Visual / undetermined",
  needs_review: "Needs review",
};

/** Tailwind classes per result state, ordered from most to least alarming. */
export const RESULT_STYLES: Record<ResultState, string> = {
  defect_found:
    "bg-red-50 text-red-800 border-red-200 dark:bg-red-950 dark:text-red-100 dark:border-red-900",
  needs_review:
    "bg-amber-50 text-amber-900 border-amber-200 dark:bg-amber-950 dark:text-amber-100 dark:border-amber-900",
  not_determined:
    "bg-slate-50 text-slate-700 border-slate-200 dark:bg-slate-900 dark:text-slate-200 dark:border-slate-700",
  compliant:
    "bg-emerald-50 text-emerald-800 border-emerald-200 dark:bg-emerald-950 dark:text-emerald-100 dark:border-emerald-900",
  not_applicable:
    "bg-slate-50 text-slate-500 border-slate-200 dark:bg-slate-900 dark:text-slate-400 dark:border-slate-700",
};

const RESULT_ORDER: ResultState[] = [
  "defect_found",
  "needs_review",
  "not_determined",
  "compliant",
  "not_applicable",
];

export function sortFindings(findings: DefectFinding[]): DefectFinding[] {
  return [...findings].sort((a, b) => {
    const byStatus =
      RESULT_ORDER.indexOf(a.status) - RESULT_ORDER.indexOf(b.status);
    return byStatus !== 0
      ? byStatus
      : (a.serial_no ?? Number.MAX_SAFE_INTEGER) -
          (b.serial_no ?? Number.MAX_SAFE_INTEGER);
  });
}

export function formatUsd(value?: number | null): string {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  const abs = Math.abs(value);
  if (abs >= 1) {
    return `$${value.toFixed(2)}`;
  }
  if (abs >= 0.01) {
    return `$${value.toFixed(4)}`;
  }
  return `$${value.toFixed(6)}`;
}

export function formatTokens(value?: number | null): string {
  if (value == null) {
    return "—";
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}k`;
  }
  return String(value);
}

export function isApproved(item: AgentDataItem): boolean {
  return (
    (item.data as Record<string, unknown> | undefined)?.status === "approved"
  );
}

function fileNameOf(item: AgentDataItem): string | undefined {
  const data = item.data as Record<string, unknown> | undefined;
  return typeof data?.file_name === "string" ? data.file_name : undefined;
}

export interface ScrutinyTarget {
  itemId: string;
  fileName?: string;
  mode: "run" | "view";
}

/** Stored on the extraction item at `data.metadata.scrutiny_report`. */
export const SCRUTINY_REPORT_KEY = "scrutiny_report";

function asReport(value: unknown): ScrutinyReport | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  if (
    "findings" in value &&
    Array.isArray((value as ScrutinyReport).findings)
  ) {
    return value as ScrutinyReport;
  }
  if (
    "report" in value &&
    (value as { report?: unknown }).report &&
    typeof (value as { report?: unknown }).report === "object"
  ) {
    return asReport((value as { report: unknown }).report);
  }
  return undefined;
}

export function reportFromItem(
  item: AgentDataItem | undefined,
): ScrutinyReport | undefined {
  const data = item?.data as Record<string, unknown> | undefined;
  const metadata = data?.metadata as Record<string, unknown> | undefined;
  return (
    asReport(metadata?.[SCRUTINY_REPORT_KEY]) ??
    asReport(data?.[SCRUTINY_REPORT_KEY])
  );
}

export function downloadScrutinyReport(report: ScrutinyReport) {
  downloadJSON(report, scrutinyFilename(report, "json"));
}

export function downloadScrutinyDoc(report: ScrutinyReport) {
  downloadFile(
    `\ufeff${buildScrutinyDocHtml(report)}`,
    scrutinyFilename(report, "doc"),
    "application/msword",
  );
}

function scrutinyFilename(report: ScrutinyReport, ext: string) {
  const fileName = (report.file_name || "filing").replace(/\.[^.]+$/, "");
  const stamp = (report.generated_at || new Date().toISOString()).slice(0, 10);
  return `${fileName}-scrutiny-${stamp}.${ext}`;
}

function escapeHtml(value: string) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const STATUS_COLORS: Record<ResultState, { bg: string; fg: string }> = {
  defect_found: { bg: "#FEE2E2", fg: "#991B1B" },
  needs_review: { bg: "#FEF3C7", fg: "#92400E" },
  not_determined: { bg: "#F1F5F9", fg: "#334155" },
  compliant: { bg: "#D1FAE5", fg: "#065F46" },
  not_applicable: { bg: "#F1F5F9", fg: "#64748B" },
};

function statusBadge(status: ResultState) {
  const colors = STATUS_COLORS[status];
  return `<span style="background:${colors.bg};color:${colors.fg};padding:2px 8px;border-radius:999px;font-size:10pt;font-weight:600;">${escapeHtml(RESULT_LABELS[status])}</span>`;
}

function buildScrutinyDocHtml(report: ScrutinyReport): string {
  const findings = sortFindings(report.findings);
  const generated = report.generated_at
    ? new Date(report.generated_at).toLocaleString()
    : new Date().toLocaleString();
  const title = report.file_name || "Filing";

  const usage = report.usage;
  const usageBlock =
    usage && (usage.llm_calls || usage.total_tokens || usage.cost_usd != null)
      ? `<h2 style="margin:24px 0 8px;font-size:14pt;">OpenRouter spend</h2>
        <p><b>${escapeHtml(formatUsd(usage.cost_usd))}</b> · ${escapeHtml(formatTokens(usage.prompt_tokens))} in · ${escapeHtml(formatTokens(usage.completion_tokens))} out · ${usage.llm_calls} call${usage.llm_calls === 1 ? "" : "s"}${usage.model ? ` · ${escapeHtml(usage.model)}` : ""}</p>
        ${usage.highest_cost_check_id ? `<p>Highest: S.No. ${usage.highest_cost_serial_no} ${escapeHtml(usage.highest_cost_check_id)} · ${escapeHtml(formatUsd(usage.highest_cost_usd))}</p>` : ""}
        <table style="width:100%;border-collapse:collapse;margin:8px 0 16px;font-size:10pt;">
          <tr style="background:#F8FAFC;text-align:left;">
            <th style="padding:6px 8px;border-bottom:1px solid #E2E8F0;">S.No.</th>
            <th style="padding:6px 8px;border-bottom:1px solid #E2E8F0;">Check</th>
            <th style="padding:6px 8px;border-bottom:1px solid #E2E8F0;">Charge</th>
            <th style="padding:6px 8px;border-bottom:1px solid #E2E8F0;">Tokens</th>
            <th style="padding:6px 8px;border-bottom:1px solid #E2E8F0;">Share</th>
          </tr>
          ${(usage.by_check ?? [])
            .map(
              (row) =>
                `<tr>
                  <td style="padding:6px 8px;border-bottom:1px solid #E2E8F0;">${row.serial_no}</td>
                  <td style="padding:6px 8px;border-bottom:1px solid #E2E8F0;">${escapeHtml(row.check_id)}</td>
                  <td style="padding:6px 8px;border-bottom:1px solid #E2E8F0;">${escapeHtml(formatUsd(row.cost_usd))}</td>
                  <td style="padding:6px 8px;border-bottom:1px solid #E2E8F0;">${escapeHtml(formatTokens(row.total_tokens))} (${escapeHtml(formatTokens(row.prompt_tokens))} in / ${escapeHtml(formatTokens(row.completion_tokens))} out)</td>
                  <td style="padding:6px 8px;border-bottom:1px solid #E2E8F0;">${row.share != null ? `${Math.round(row.share * 100)}%` : "—"}</td>
                </tr>`,
            )
            .join("")}
        </table>
        ${usage.note ? `<p style="color:#64748B;font-size:9pt;">${escapeHtml(usage.note)}</p>` : ""}`
      : "";

  const findingBlocks = findings
    .map((finding) => {
      const evidenceItems = finding.evidence ?? [];
      const evidence =
        evidenceItems.length > 0
          ? `<ul>${evidenceItems
              .map(
                (ref) =>
                  `<li>${ref.page !== null ? `<b>Page ${ref.page}:</b> ` : ""}<i>“${escapeHtml(ref.quote)}”</i></li>`,
              )
              .join("")}</ul>`
          : "";
      const fix = finding.suggested_fix
        ? `<p style="background:#EFF6FF;border-left:4px solid #2563EB;padding:8px 12px;"><b>Suggested fix:</b> ${escapeHtml(finding.suggested_fix)}${finding.fix_rationale ? `<br/><span style="color:#1E3A8A;">${escapeHtml(finding.fix_rationale)}</span>` : ""}</p>`
        : "";
      const cure = finding.how_to_cure?.length
        ? `<p><b>How to cure</b></p><ol>${finding.how_to_cure.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ol>`
        : "";
      const heading =
        finding.serial_no != null
          ? `S.No. ${finding.serial_no} · ${finding.check_id}`
          : finding.check_id;
      const meta = [
        finding.serial_no != null ? `Sheet S.No. ${finding.serial_no}` : null,
        finding.location,
        finding.applicable_rule,
        finding.location_source,
        finding.main_category,
      ]
        .filter((value): value is string => Boolean(value))
        .join(" · ");
      const legacySubchecks = (finding.subcheck_results ?? [])
        .map((sub) => {
          const subEvidence =
            sub.evidence.length > 0
              ? `<ul>${sub.evidence
                  .map(
                    (ref) =>
                      `<li>${ref.page !== null ? `<b>Page ${ref.page}:</b> ` : ""}<i>“${escapeHtml(ref.quote)}”</i></li>`,
                  )
                  .join("")}</ul>`
              : "";
          const subFix = sub.suggested_fix
            ? `<p style="background:#EFF6FF;border-left:4px solid #2563EB;padding:8px 12px;"><b>Suggested fix:</b> ${escapeHtml(sub.suggested_fix)}</p>`
            : "";
          return `<tr>
            <td style="padding:8px 10px;border-bottom:1px solid #E2E8F0;vertical-align:top;width:110px;"><b>${escapeHtml(sub.subcheck_id)}</b></td>
            <td style="padding:8px 10px;border-bottom:1px solid #E2E8F0;vertical-align:top;">
              ${statusBadge(sub.status)}
              <p>${escapeHtml(sub.reasoning)}</p>
              ${subEvidence}
              ${subFix}
            </td>
          </tr>`;
        })
        .join("");

      return `<h2 style="margin:28px 0 8px;font-size:14pt;">${escapeHtml(heading)}</h2>
        <p>${statusBadge(finding.status)} <span style="color:#64748B;">${Math.round(finding.confidence * 100)}% confident${finding.usage?.cost_usd != null ? ` · ${escapeHtml(formatUsd(finding.usage.cost_usd))}` : ""}</span></p>
        <p>${escapeHtml(finding.title)}</p>
        <p>${escapeHtml(finding.summary)}</p>
        ${finding.reasoning ? `<p>${escapeHtml(finding.reasoning)}</p>` : ""}
        ${evidence}
        ${fix}
        ${cure}
        ${meta ? `<p style="color:#64748B;font-size:10pt;"><b>Authority:</b> ${escapeHtml(meta)}</p>` : ""}
        ${legacySubchecks ? `<table style="width:100%;border-collapse:collapse;">${legacySubchecks}</table>` : ""}`;
    })
    .join("");

  return `<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40">
<head>
<meta charset="utf-8">
<title>Registry defect check — ${escapeHtml(title)}</title>
<!--[if gte mso 9]><xml><w:WordDocument><w:View>Print</w:View><w:Zoom>100</w:Zoom></w:WordDocument></xml><![endif]-->
<style>
  body { font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: #0F172A; line-height: 1.45; max-width: 800px; }
  h1 { font-size: 20pt; margin-bottom: 4px; }
  table.summary td { padding: 10px 14px; text-align: center; }
</style>
</head>
<body>
  <h1>Registry defect check</h1>
  <p style="color:#475569;margin-top:0;">${escapeHtml(title)}${report.petition_type ? ` · ${escapeHtml(report.petition_type)}` : ""}</p>
  <p>${generated}${report.catalogue_id ? ` · ${escapeHtml(report.catalogue_id)} v${escapeHtml(report.catalogue_version)}` : ""}</p>

  <table class="summary" style="width:100%;border-collapse:collapse;margin:16px 0 24px;">
    <tr>
      <td style="background:#FEE2E2;"><b style="font-size:18pt;">${report.summary.defects_found}</b><br/>Defects found</td>
      <td style="background:#FEF3C7;"><b style="font-size:18pt;">${report.summary.needs_review}</b><br/>Needs review</td>
      <td style="background:#F1F5F9;"><b style="font-size:18pt;">${report.summary.not_determined}</b><br/>Visual / undetermined</td>
      <td style="background:#D1FAE5;"><b style="font-size:18pt;">${report.summary.compliant}</b><br/>Compliant</td>
    </tr>
  </table>

  ${usageBlock}

  ${report.disclaimer ? `<p style="background:#F8FAFC;border:1px solid #E2E8F0;padding:10px 12px;font-size:10pt;color:#475569;">${escapeHtml(report.disclaimer)}</p>` : ""}

  ${findingBlocks}

  <p style="margin-top:32px;color:#64748B;font-size:9pt;">Pre-filing assistance only. This report does not represent the Supreme Court Registry and does not replace review by an Advocate-on-Record.</p>
</body>
</html>`;
}

function stopSubscription(sub: {
  disconnect?: () => void;
  unsubscribe?: () => void;
} | null) {
  sub?.disconnect?.();
  sub?.unsubscribe?.();
}

export function useScrutiny() {
  const wf = useWorkflow("scrutiny-check");
  const client = useCloudApiClient();
  const handlersService = useHandlers({
    query: { workflow_name: ["scrutiny-check"] },
    sync: false,
  });
  const [target, setTarget] = useState<ScrutinyTarget | undefined>();
  const [handlerId, setHandlerId] = useState<string | undefined>();
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<ScrutinyReport | undefined>();
  const [error, setError] = useState<string | undefined>();
  const [progress, setProgress] = useState<string | undefined>();
  const reportRef = useRef<ScrutinyReport | undefined>(undefined);
  const itemIdRef = useRef<string | undefined>(undefined);

  const busy = running || loading;

  const run = useCallback(
    async (item: AgentDataItem) => {
      if (!item.id) {
        setError("This item has no id, so it cannot be checked.");
        return;
      }
      const data = item.data as Record<string, unknown> | undefined;
      itemIdRef.current = item.id;
      reportRef.current = undefined;
      setTarget({ itemId: item.id, fileName: fileNameOf(item), mode: "run" });
      setReport(undefined);
      setError(undefined);
      setProgress("Starting defect check…");
      setRunning(true);
      setHandlerId(undefined);

      try {
        // createHandler returns as soon as the run is queued. runToCompletion
        // hits POST /run and is killed by the cloud nginx 60s gateway timeout
        // once SCRUTINY_DEFECTS=all (~74 checks) is enabled.
        const created = await wf.createHandler({
          agent_data_id: item.id,
          file_hash: (data?.file_hash as string | undefined) ?? null,
        });
        handlersService.setHandler(created);
        setHandlerId(created.handler_id);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setRunning(false);
        setProgress(undefined);
      }
    },
    [wf, handlersService],
  );

  useEffect(() => {
    if (!handlerId) {
      return;
    }
    const sub = handlersService.actions(handlerId).subscribeToEvents({
      onData(event) {
        const payload = event.data as Record<string, unknown> | undefined;
        if (
          event.type === "Status" &&
          payload &&
          typeof payload.message === "string"
        ) {
          setProgress(payload.message);
        }
        if (
          payload &&
          typeof payload.completed === "number" &&
          typeof payload.total === "number"
        ) {
          const stopped =
            payload.stopped_early === true ? " — stopped after an error" : "";
          setProgress(
            `Completed ${payload.completed} of ${payload.total} checks${stopped}`,
          );
        }
        const parsed = asReport(payload);
        if (parsed) {
          reportRef.current = parsed;
          setReport(parsed);
          if (parsed.stopped_early) {
            setError(
              "An earlier check failed. Remaining LLM calls were cancelled. Showing completed results.",
            );
          }
        }
      },
      onComplete() {
        void (async () => {
          if (!reportRef.current && itemIdRef.current) {
            try {
              const fresh = await client.beta.agentData.get(itemIdRef.current);
              const saved = reportFromItem(fresh as AgentDataItem);
              if (saved) {
                reportRef.current = saved;
                setReport(saved);
              }
            } catch {
              /* stream already ended; fall through to the empty-result error */
            }
          }
          if (!reportRef.current) {
            setError("The check finished but returned no findings.");
          }
          setRunning(false);
          setProgress(undefined);
        })();
      },
      onError(err) {
        setError(err instanceof Error ? err.message : String(err));
        setRunning(false);
      },
    });
    return () => stopSubscription(sub);
  }, [handlerId, handlersService, client]);

  const viewSaved = useCallback(
    async (item: AgentDataItem) => {
      if (!item.id) {
        setError("This item has no id, so a saved check cannot be loaded.");
        return;
      }
      setTarget({ itemId: item.id, fileName: fileNameOf(item), mode: "view" });
      setReport(undefined);
      setError(undefined);
      setProgress(undefined);
      setLoading(true);

      try {
        let parsed: ScrutinyReport | undefined;
        try {
          const fresh = await client.beta.agentData.get(item.id);
          parsed = reportFromItem(fresh as AgentDataItem);
        } catch {
          parsed = reportFromItem(item);
        }
        if (!parsed) {
          setError(
            "No saved defect check for this filing yet. Run Check defects first.",
          );
          return;
        }
        setReport(parsed);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [client],
  );

  const close = useCallback(() => {
    setTarget(undefined);
    setReport(undefined);
    setError(undefined);
    setProgress(undefined);
    setHandlerId(undefined);
    setRunning(false);
    setLoading(false);
  }, []);

  return {
    target,
    running,
    loading,
    busy,
    report,
    error,
    progress,
    run,
    viewSaved,
    close,
  };
}
