import { AgentDataItem, useWorkflow } from "@llamaindex/ui";
import { useCallback, useState } from "react";

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

export interface Coverage {
  chunks_reviewed: number;
  pages_reviewed: number[];
  structured_record_available: boolean;
  evidence_complete: boolean;
}

export interface DefectFinding {
  check_id: string;
  title: string;
  severity: string;
  status: ResultState;
  summary: string;
  confidence: number;
  subcheck_results: SubcheckResult[];
  evidence_ids: string[];
  coverage: Coverage;
  authority_refs: string[];
  error: string | null;
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
}

export const RESULT_LABELS: Record<ResultState, string> = {
  defect_found: "Defect found",
  compliant: "Compliant",
  not_applicable: "Not applicable",
  not_determined: "Not determined",
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
    return byStatus !== 0 ? byStatus : a.check_id.localeCompare(b.check_id);
  });
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
}

export function useScrutiny() {
  const wf = useWorkflow("scrutiny-check");
  const [target, setTarget] = useState<ScrutinyTarget | undefined>();
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<ScrutinyReport | undefined>();
  const [error, setError] = useState<string | undefined>();

  const run = useCallback(
    async (item: AgentDataItem) => {
      if (!item.id) {
        setError("This item has no id, so it cannot be checked.");
        return;
      }
      const data = item.data as Record<string, unknown> | undefined;
      setTarget({ itemId: item.id, fileName: fileNameOf(item) });
      setReport(undefined);
      setError(undefined);
      setRunning(true);

      try {
        const handler = await wf.runToCompletion({
          agent_data_id: item.id,
          file_hash: (data?.file_hash as string | undefined) ?? null,
        });

        if (handler.status !== "completed") {
          setError(
            handler.error || `Workflow ended with status: ${handler.status}`,
          );
          return;
        }

        const result = handler.result?.data as unknown as
          { report?: ScrutinyReport } | ScrutinyReport | undefined;
        const parsed =
          result && "report" in (result as object)
            ? (result as { report?: ScrutinyReport }).report
            : (result as ScrutinyReport | undefined);

        if (!parsed?.findings) {
          setError("The check finished but returned no findings.");
          return;
        }
        setReport(parsed);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setRunning(false);
      }
    },
    [wf],
  );

  const close = useCallback(() => {
    setTarget(undefined);
    setReport(undefined);
    setError(undefined);
  }, []);

  return { target, running, report, error, run, close };
}
