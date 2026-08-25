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
  SubcheckResult,
  ScrutinyTarget,
  sortFindings,
  downloadScrutinyReport,
  downloadScrutinyDoc,
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

function SubcheckRow({ result }: { result: SubcheckResult }) {
  return (
    <div className="border-t border-border px-4 py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="text-sm font-medium">{result.subcheck_id}</div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">
            {Math.round(result.confidence * 100)}% confident
          </span>
          <StatusBadge status={result.status} />
        </div>
      </div>

      <p className="mt-1.5 text-sm text-muted-foreground">{result.reasoning}</p>

      {result.evidence.length > 0 && (
        <ul className="mt-2 space-y-1.5">
          {result.evidence.map((ref, i) => (
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
      )}

      {result.suggested_fix && (
        <div className="mt-2.5 rounded-md border border-blue-200 bg-blue-50 px-3 py-2 dark:border-blue-900 dark:bg-blue-950">
          <div className="text-xs font-semibold text-blue-900 dark:text-blue-100">
            Suggested fix
          </div>
          <p className="mt-0.5 text-sm text-blue-900 dark:text-blue-100">
            {result.suggested_fix}
          </p>
          {result.fix_rationale && (
            <p className="mt-1 text-xs text-blue-800 dark:text-blue-200 opacity-80">
              {result.fix_rationale}
            </p>
          )}
        </div>
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
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold">{finding.check_id}</span>
            <span className="text-sm">{finding.title}</span>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            {finding.summary}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <span className="text-xs text-muted-foreground">
            {Math.round(finding.confidence * 100)}%
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
          {finding.subcheck_results.map((result) => (
            <SubcheckRow key={result.subcheck_id} result={result} />
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
          {finding.authority_refs.length > 0 && (
            <div className="border-t border-border px-4 py-2 text-xs text-muted-foreground">
              Authority: {finding.authority_refs.join(" · ")}
            </div>
          )}
        </div>
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
      label: "Undetermined",
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
  onClose,
}: {
  target?: ScrutinyTarget;
  running: boolean;
  loading?: boolean;
  report?: ScrutinyReport;
  error?: string;
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
            <div className="flex items-center gap-2 rounded-lg border border-border bg-muted/40 px-4 py-6 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              {running
                ? "Checking this filing against the Supreme Court registry defect catalogue. This usually takes under a minute."
                : "Loading the last saved defect check for this filing."}
            </div>
          )}

          {error && !busy && (
            <div className="flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-100">
              <AlertTriangle
                className="mt-0.5 h-4 w-4 shrink-0"
                aria-hidden="true"
              />
              <div>
                <div className="font-medium">
                  {target?.mode === "view"
                    ? "No saved check to show"
                    : "The check could not be run"}
                </div>
                <div className="mt-0.5 break-words opacity-90">{error}</div>
              </div>
            </div>
          )}

          {report && !busy && (
            <>
              <SummaryBar report={report} />

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
                </p>
                {report.disclaimer && (
                  <p className="mt-1">{report.disclaimer}</p>
                )}
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
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
