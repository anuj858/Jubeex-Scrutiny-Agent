import { Dialog, DialogContent, Button } from "@llamaindex/ui";
import { AlertTriangle, ChevronDown, Download, Loader2 } from "lucide-react";
import { useState, type ReactNode } from "react";
import { cn } from "./utils";
import {
  DefectFinding,
  RESULT_LABELS,
  RESULT_STYLES,
  ResultState,
  ScrutinyReport,
  ScrutinyTarget,
  sortFindings,
  downloadScrutinyReport,
  downloadScrutinyDoc,
  formatUsd,
  formatTokens,
} from "./scrutiny";

function StatusBadge({
  status,
  children,
}: {
  status: ResultState;
  children?: ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium whitespace-nowrap",
        RESULT_STYLES[status],
      )}
    >
      {children ?? RESULT_LABELS[status]}
    </span>
  );
}

function EvidenceList({
  evidence,
}: {
  evidence: { page: number | null; quote: string }[];
}) {
  if (evidence.length === 0) {
    return null;
  }
  return (
    <ul className="mt-2 space-y-1.5">
      {evidence.map((ref, i) => (
        <li
          key={i}
          className="border-l-2 border-border pl-3 text-xs text-muted-foreground"
        >
          {ref.page !== null && (
            <span className="font-medium">Page {ref.page}: </span>
          )}
          <span className="italic">“{ref.quote}”</span>
        </li>
      ))}
    </ul>
  );
}

function SuggestedFix({
  suggestedFix,
  rationale,
}: {
  suggestedFix?: string | null;
  rationale?: string | null;
}) {
  if (!suggestedFix) {
    return null;
  }
  return (
    <div className="mt-2.5 rounded-md border border-blue-200 bg-blue-50 px-3 py-2 dark:border-blue-900 dark:bg-blue-950">
      <div className="text-xs font-semibold text-blue-900 dark:text-blue-100">
        Suggested fix
      </div>
      <p className="mt-0.5 text-sm text-blue-900 dark:text-blue-100">
        {suggestedFix}
      </p>
      {rationale && (
        <p className="mt-1 text-xs text-blue-800 dark:text-blue-200 opacity-80">
          {rationale}
        </p>
      )}
    </div>
  );
}

function FindingCard({ finding }: { finding: DefectFinding }) {
  const [open, setOpen] = useState(
    finding.status === "defect_found" || finding.status === "needs_review",
  );

  return (
    <div className="rounded-lg border border-border overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-start justify-between gap-3 px-4 py-3 text-left hover:bg-muted/50 transition-colors"
      >
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            {finding.serial_no != null && (
              <span className="inline-flex items-center rounded-md border border-border bg-muted px-2 py-0.5 text-sm font-semibold tabular-nums">
                S.No. {finding.serial_no}
              </span>
            )}
            <span className="text-xs font-medium text-muted-foreground">
              {finding.check_id}
            </span>
          </div>
          <p className="mt-1 text-sm">{finding.title}</p>
          <p className="mt-1 text-sm text-muted-foreground">
            {finding.summary}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <span className="text-xs text-muted-foreground">
            {Math.round(finding.confidence * 100)}%
            {finding.usage?.cost_usd != null &&
              ` · ${formatUsd(finding.usage.cost_usd)}`}
          </span>
          <StatusBadge status={finding.status} />
          <ChevronDown
            className={cn(
              "h-4 w-4 text-muted-foreground transition-transform",
              open && "rotate-180",
            )}
            aria-hidden="true"
          />
        </div>
      </button>

      {open && (
        <div className="bg-muted/20">
          <div className="border-t border-border px-4 py-3">
            {finding.reasoning && (
              <p className="text-sm text-muted-foreground">{finding.reasoning}</p>
            )}
            <EvidenceList evidence={finding.evidence ?? []} />
            <SuggestedFix
              suggestedFix={finding.suggested_fix}
              rationale={finding.fix_rationale}
            />
            {finding.usage &&
              (finding.usage.calls || finding.usage.total_tokens) > 0 && (
                <div className="mt-2 text-xs text-muted-foreground">
                  OpenRouter {formatUsd(finding.usage.cost_usd)} ·{" "}
                  {formatTokens(finding.usage.prompt_tokens)} in /{" "}
                  {formatTokens(finding.usage.completion_tokens)} out
                  {finding.usage.calls && finding.usage.calls > 1
                    ? ` · ${finding.usage.calls} calls`
                    : ""}
                </div>
              )}
            {finding.how_to_cure && finding.how_to_cure.length > 0 && (
              <div className="mt-3">
                <div className="text-xs font-semibold">How to cure</div>
                <ol className="mt-1 list-decimal space-y-1 pl-4 text-sm text-muted-foreground">
                  {finding.how_to_cure.map((step, i) => (
                    <li key={i}>{step}</li>
                  ))}
                </ol>
              </div>
            )}
          </div>
          {(finding.subcheck_results ?? []).map((result) => (
            <div key={result.subcheck_id} className="border-t border-border px-4 py-3">
              <div className="flex items-start justify-between gap-3">
                <div className="text-sm font-medium">{result.subcheck_id}</div>
                <StatusBadge status={result.status} />
              </div>
              <p className="mt-1.5 text-sm text-muted-foreground">
                {result.reasoning}
              </p>
              <EvidenceList evidence={result.evidence} />
              <SuggestedFix
                suggestedFix={result.suggested_fix}
                rationale={result.fix_rationale}
              />
            </div>
          ))}
          {!finding.coverage.evidence_complete && (
            <div className="border-t border-border px-4 py-2 text-xs text-muted-foreground">
              Based on {finding.coverage.chunks_reviewed} excerpt(s)
              {finding.coverage.pages_reviewed.length > 0 &&
                ` from page ${finding.coverage.pages_reviewed.join(", ")}`}
              . Coverage was incomplete, so undetermined results may simply mean
              the relevant text was not retrieved.
            </div>
          )}
          {(finding.serial_no != null ||
            finding.applicable_rule ||
            finding.location_source ||
            (finding.authority_refs ?? []).length > 0) && (
            <div className="border-t border-border px-4 py-2 text-xs text-muted-foreground">
              {[
                finding.serial_no != null ? `Sheet S.No. ${finding.serial_no}` : null,
                finding.applicable_rule,
                finding.location_source,
                ...(finding.authority_refs ?? []),
              ]
                .filter(Boolean)
                .join(" · ")}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function UsagePanel({ report }: { report: ScrutinyReport }) {
  const usage = report.usage;
  if (!usage || (!usage.llm_calls && !usage.total_tokens && usage.cost_usd == null)) {
    return null;
  }
  const rows = usage.by_check ?? [];
  return (
    <div className="rounded-lg border border-border px-4 py-3 space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            OpenRouter spend
          </div>
          <div className="mt-1 text-2xl font-semibold tabular-nums">
            {formatUsd(usage.cost_usd)}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {formatTokens(usage.prompt_tokens)} in ·{" "}
            {formatTokens(usage.completion_tokens)} out · {usage.llm_calls} call
            {usage.llm_calls === 1 ? "" : "s"}
            {usage.model ? ` · ${usage.model}` : ""}
          </p>
        </div>
        {usage.highest_cost_check_id && (
          <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-950 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-100">
            Highest: S.No. {usage.highest_cost_serial_no}{" "}
            {usage.highest_cost_check_id} · {formatUsd(usage.highest_cost_usd)}
          </div>
        )}
      </div>
      {rows.length > 0 && (
        <div className="space-y-1.5">
          {rows.map((row) => {
            const pct = Math.round((row.share ?? 0) * 100);
            return (
              <div key={row.check_id} className="space-y-0.5">
                <div className="flex items-center justify-between gap-2 text-xs">
                  <span className="min-w-0 truncate">
                    S.No. {row.serial_no} · {row.check_id}
                  </span>
                  <span className="shrink-0 tabular-nums text-muted-foreground">
                    {formatUsd(row.cost_usd)} · {formatTokens(row.total_tokens)} ·{" "}
                    {pct}%
                  </span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-amber-500/80"
                    style={{ width: `${Math.min(100, Math.max(pct, pct > 0 ? 2 : 0))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
      {usage.note && (
        <p className="text-xs text-muted-foreground">{usage.note}</p>
      )}
    </div>
  );
}

function SummaryBar({ report }: { report: ScrutinyReport }) {
  const stats: { label: string; value: number; status: ResultState }[] = [
    {
      label: "Defects",
      value: report.summary.defects_found,
      status: "defect_found",
    },
    {
      label: "Needs review",
      value: report.summary.needs_review,
      status: "needs_review",
    },
    {
      label: "Visual / undetermined",
      value: report.summary.not_determined,
      status: "not_determined",
    },
    {
      label: "Compliant",
      value: report.summary.compliant,
      status: "compliant",
    },
  ];

  return (
    <div className="flex flex-wrap gap-2">
      {stats.map((stat) => (
        <div
          key={stat.label}
          className={cn(
            "flex-1 min-w-24 rounded-lg border px-3 py-2",
            RESULT_STYLES[stat.status],
          )}
        >
          <div className="text-xl font-semibold">{stat.value}</div>
          <div className="text-xs">{stat.label}</div>
        </div>
      ))}
    </div>
  );
}

export function ScrutinyDialog({
  target,
  running,
  loading,
  report,
  error,
  progress,
  onClose,
}: {
  target?: ScrutinyTarget;
  running: boolean;
  loading?: boolean;
  report?: ScrutinyReport;
  error?: string;
  progress?: string;
  onClose: () => void;
}) {
  const busy = running || loading;
  return (
    <Dialog open={!!target} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
        <div className="space-y-4">
          <div>
            <h2 className="text-lg font-semibold">Registry defect check</h2>
            <p className="text-sm text-muted-foreground">
              {target?.fileName ?? "Selected filing"}
            </p>
          </div>

          {busy && (
            <div className="flex items-start gap-2 rounded-lg border border-border bg-muted/40 px-4 py-4 text-sm text-muted-foreground">
              <Loader2
                className="mt-0.5 h-4 w-4 shrink-0 animate-spin"
                aria-hidden="true"
              />
              <div>
                <div>
                  {running
                    ? "Checking this filing against the Supreme Court registry defect catalogue. Findings appear below as each check finishes."
                    : "Loading the last saved defect check for this filing."}
                </div>
                {running && progress && (
                  <div className="mt-1 text-xs">{progress}</div>
                )}
              </div>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-100">
              <AlertTriangle
                className="mt-0.5 h-4 w-4 shrink-0"
                aria-hidden="true"
              />
              <div>
                <div className="font-medium">
                  {target?.mode === "view"
                    ? "No saved check to show"
                    : report
                      ? "Check stopped early"
                      : "The check could not be run"}
                </div>
                <div className="mt-0.5 break-words opacity-90">{error}</div>
              </div>
            </div>
          )}

          {report && (
            <>
              {report.stopped_early && !error && (
                <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-950 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-100">
                  Stopped after an error so remaining checks were not sent to
                  the model. Showing {report.findings.length}
                  {report.planned_checks
                    ? ` of ${report.planned_checks}`
                    : ""}{" "}
                  completed results.
                </div>
              )}
              <SummaryBar report={report} />
              <UsagePanel report={report} />

              <div className="space-y-2">
                {sortFindings(report.findings).map((finding) => (
                  <FindingCard key={finding.check_id} finding={finding} />
                ))}
              </div>

              <div className="border-t border-border pt-3 text-xs text-muted-foreground space-y-2">
                <p>
                  {report.catalogue_id} v{report.catalogue_version}
                  {report.model && ` · ${report.model}`} ·{" "}
                  {new Date(report.generated_at).toLocaleString()}
                  {report.planned_checks
                    ? ` · ${report.findings.length}/${report.planned_checks} checks`
                    : ""}
                </p>
                {report.disclaimer && (
                  <p className="mt-1">{report.disclaimer}</p>
                )}
                {!running && (
                  <div className="flex flex-wrap gap-2">
                    <Button
                      size="sm"
                      label="Download Word report"
                      startIcon={<Download className="h-3.5 w-3.5" />}
                      onClick={() => downloadScrutinyDoc(report)}
                    />
                    <Button
                      size="sm"
                      variant="ghost"
                      label="Download JSON"
                      onClick={() => downloadScrutinyReport(report)}
                    />
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
