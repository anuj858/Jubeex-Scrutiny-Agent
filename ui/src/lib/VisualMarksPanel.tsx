import { useEffect, useState } from "react";
import { ExternalLink, FileJson, Loader2 } from "lucide-react";
import type { Highlight } from "@llamaindex/ui";
import {
  readS3Artifacts,
  readVisualSummary,
  type S3ArtifactLink,
  type VisualFailure,
  type VisualTargetRow,
  type VisualUsage,
} from "./utils";
import { formatUsd } from "./scrutiny";

type VisualMarkRow = {
  page: number;
  document_type: string;
  marking_type: string;
  signature_role?: string | null;
  bbox?: { x: number; y: number; w: number; h: number };
  confidence?: number | null;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function parseMarks(payload: unknown): VisualMarkRow[] {
  const root = asRecord(payload);
  const raw = root?.marks;
  if (!Array.isArray(raw)) {
    return [];
  }
  const rows: VisualMarkRow[] = [];
  for (const item of raw) {
    const mark = asRecord(item);
    if (!mark) {
      continue;
    }
    const page = asNumber(mark.page);
    const documentType =
      typeof mark.document_type === "string" ? mark.document_type : "";
    const markingType =
      typeof mark.marking_type === "string" ? mark.marking_type : "";
    if (page == null || !documentType || !markingType) {
      continue;
    }
    const box = asRecord(mark.bbox);
    rows.push({
      page,
      document_type: documentType,
      marking_type: markingType,
      signature_role:
        typeof mark.signature_role === "string" ? mark.signature_role : null,
      bbox: box
        ? {
            x: asNumber(box.x) ?? 0,
            y: asNumber(box.y) ?? 0,
            w: asNumber(box.w) ?? 0,
            h: asNumber(box.h) ?? 0,
          }
        : undefined,
      confidence: asNumber(mark.confidence),
    });
  }
  return rows;
}

function parseTargets(payload: unknown): VisualTargetRow[] {
  const root = asRecord(payload);
  if (!Array.isArray(root?.targets)) {
    return [];
  }
  const rows: VisualTargetRow[] = [];
  for (const item of root.targets) {
    const record = asRecord(item);
    if (!record) {
      continue;
    }
    const page = asNumber(record.page);
    if (page == null) {
      continue;
    }
    const documentTypes = Array.isArray(record.document_types)
      ? record.document_types
          .filter((name): name is string => typeof name === "string" && name.trim().length > 0)
          .map((name) => name.trim())
      : [];
    rows.push({ page, document_types: documentTypes });
  }
  return rows;
}

function parseFailures(payload: unknown): VisualFailure[] {
  const root = asRecord(payload);
  if (!Array.isArray(root?.failures)) {
    return [];
  }
  const rows: VisualFailure[] = [];
  for (const item of root.failures) {
    const record = asRecord(item);
    if (!record) {
      continue;
    }
    const page = asNumber(record.page);
    const reason = typeof record.reason === "string" ? record.reason.trim() : "";
    if (page == null || !reason) {
      continue;
    }
    const documentTypes = Array.isArray(record.document_types)
      ? record.document_types
          .filter((name): name is string => typeof name === "string" && name.trim().length > 0)
          .map((name) => name.trim())
      : [];
    rows.push({
      page,
      slot_id: typeof record.slot_id === "string" ? record.slot_id : null,
      document_types: documentTypes,
      reason,
      detail: typeof record.detail === "string" ? record.detail : null,
    });
  }
  return rows;
}

function parseUsage(payload: unknown): VisualUsage | null {
  const root = asRecord(payload);
  const record = asRecord(root?.usage);
  if (!record) {
    return null;
  }
  const cost = asNumber(record.cost_usd);
  const calls = asNumber(record.calls);
  const model = typeof record.model === "string" ? record.model : null;
  if (cost == null && calls == null && !model) {
    return null;
  }
  return {
    cost_usd: cost,
    prompt_tokens: asNumber(record.prompt_tokens),
    completion_tokens: asNumber(record.completion_tokens),
    total_tokens: asNumber(record.total_tokens),
    calls,
    model,
  };
}

function formatBox(bbox?: VisualMarkRow["bbox"]): string {
  if (!bbox) {
    return "—";
  }
  return `${bbox.x.toFixed(2)}, ${bbox.y.toFixed(2)}, ${bbox.w.toFixed(2)}×${bbox.h.toFixed(2)}`;
}

function ArtifactLink({ item }: { item: S3ArtifactLink }) {
  const label = item.label.trim().toLowerCase().endsWith("json")
    ? item.label
    : `${item.label} JSON`;
  if (!item.url) {
    return (
      <span className="text-xs text-slate-500" title={item.key}>
        {label}: stored at {item.key}
      </span>
    );
  }
  return (
    <a
      href={item.url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 text-xs font-medium text-blue-800 hover:underline"
      title={item.key || item.url}
    >
      <ExternalLink className="h-3 w-3 shrink-0" />
      {label}
    </a>
  );
}

function formatTypes(names: string[]): string {
  return names.length ? names.join(", ") : "—";
}

export function VisualMarksPanel({
  extractedData,
  onHighlight,
}: {
  extractedData: unknown;
  onHighlight?: (highlight: Highlight) => void;
}) {
  const artifacts = readS3Artifacts(extractedData);
  const summary = readVisualSummary(extractedData);
  const visual = artifacts.find((item) => item.id === "visual");
  const others = artifacts.filter((item) => item.id !== "visual");
  const [marks, setMarks] = useState<VisualMarkRow[]>([]);
  const [fetchedTargets, setFetchedTargets] = useState<VisualTargetRow[]>([]);
  const [fetchedFailures, setFetchedFailures] = useState<VisualFailure[]>([]);
  const [fetchedStatus, setFetchedStatus] = useState<string | null>(null);
  const [fetchedError, setFetchedError] = useState<string | null>(null);
  const [fetchedUsage, setFetchedUsage] = useState<VisualUsage | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">(
    visual?.url ? "loading" : "idle",
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!visual?.url) {
      setMarks([]);
      setFetchedTargets([]);
      setFetchedFailures([]);
      setFetchedStatus(null);
      setFetchedError(null);
      setFetchedUsage(null);
      setStatus("idle");
      setError(null);
      return;
    }
    const controller = new AbortController();
    setStatus("loading");
    setError(null);
    fetch(visual.url, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Could not load visual JSON (${response.status})`);
        }
        return response.json();
      })
      .then((payload) => {
        const root = asRecord(payload);
        setMarks(parseMarks(payload));
        setFetchedTargets(parseTargets(payload));
        setFetchedFailures(parseFailures(payload));
        setFetchedStatus(typeof root?.status === "string" ? root.status : null);
        setFetchedError(typeof root?.error === "string" ? root.error : null);
        setFetchedUsage(parseUsage(payload));
        setStatus("ready");
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setMarks([]);
        setFetchedTargets([]);
        setFetchedFailures([]);
        setFetchedStatus(null);
        setFetchedError(null);
        setFetchedUsage(null);
        setStatus("error");
        setError(
          cause instanceof Error
            ? cause.message
            : "Could not preview visual JSON in the browser",
        );
      });
    return () => controller.abort();
  }, [visual?.url]);

  const targets = fetchedTargets.length ? fetchedTargets : summary?.targets || [];
  const failures =
    fetchedFailures.length ? fetchedFailures : summary?.failures || [];
  const visionStatus = fetchedStatus || summary?.status || null;
  const visionError = fetchedError || summary?.error || null;
  const usage = fetchedUsage || summary?.usage || null;
  const markCount = status === "ready" ? marks.length : (summary?.mark_count ?? 0);
  const hasVision =
    Boolean(visual) || Boolean(summary) || others.length > 0;

  if (!hasVision) {
    return (
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
        <div className="text-sm font-semibold text-slate-800">Visual marks</div>
        <p className="mt-1 text-xs text-slate-600">
          No S3 visual JSON is stored on this filing yet. After you submit a PDF,
          signatures, seals, and stamps are saved to visualfiles/ and the link
          appears here.
        </p>
      </div>
    );
  }

  const showEmptyInk =
    status === "ready" &&
    marks.length === 0 &&
    visionStatus !== "error" &&
    failures.length === 0;

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
            <FileJson className="h-4 w-4 text-slate-600" />
            Visual marks (S3)
          </div>
          <p className="mt-1 text-xs text-slate-600">
            Ink detector output for this filing. Open the JSON to verify page,
            document type, marking type, and coordinates.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {visual ? <ArtifactLink item={visual} /> : null}
          {others.map((item) => (
            <ArtifactLink key={item.id} item={item} />
          ))}
        </div>
      </div>
      {visual?.key ? (
        <div className="break-all font-mono text-[11px] text-slate-500">
          {visual.key}
        </div>
      ) : null}
      {visionStatus || summary ? (
        <div className="rounded-md border border-slate-200 bg-white px-2 py-1.5 text-xs text-slate-700 space-y-1">
          <div>
            Status: <span className="font-medium">{visionStatus || "unknown"}</span>
            {visionError ? ` · ${visionError}` : ""}
            {` · ${summary?.target_count ?? targets.length} page(s) sent`}
            {` · ${markCount} mark(s)`}
            {failures.length ? ` · ${failures.length} failed` : ""}
            {usage?.cost_usd != null
              ? ` · OpenRouter ${formatUsd(usage.cost_usd)}`
              : ""}
            {usage?.calls != null
              ? ` · ${usage.calls} call${usage.calls === 1 ? "" : "s"}`
              : ""}
            {usage?.model ? ` · ${usage.model}` : ""}
          </div>
        </div>
      ) : null}
      {targets.length > 0 ? (
        <div>
          <div className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
            Pages sent
          </div>
          <ul className="mt-1 space-y-0.5 text-xs text-slate-700">
            {targets.map((target) => (
              <li key={`${target.page}-${formatTypes(target.document_types)}`}>
                page {target.page}: {formatTypes(target.document_types)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {failures.length > 0 ? (
        <div>
          <div className="text-[11px] font-medium uppercase tracking-wide text-amber-800">
            Page failures
          </div>
          <ul className="mt-1 space-y-0.5 text-xs text-amber-900">
            {failures.map((item) => (
              <li key={`${item.page}-${item.reason}-${item.slot_id || ""}`}>
                page {item.page}
                {item.document_types.length
                  ? ` ${formatTypes(item.document_types)}`
                  : ""}
                {`: ${item.reason}`}
                {item.detail ? ` — ${item.detail}` : ""}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {status === "loading" ? (
        <div className="flex items-center gap-2 text-xs text-slate-600">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Loading marks from S3…
        </div>
      ) : null}
      {status === "error" ? (
        <p className="text-xs text-amber-800">
          Browser could not preview the S3 file
          {error ? ` (${error})` : ""}. The visual JSON is on S3 — use{" "}
          <span className="font-medium">Visual marks JSON</span> to download
          boxes. Status above is from Agent Data.
        </p>
      ) : null}
      {showEmptyInk ? (
        <p className="text-xs text-slate-600">
          Visual JSON is stored, but no ink marks were found on the formality
          pages.
        </p>
      ) : null}
      {visionStatus === "error" && marks.length === 0 && failures.length === 0 ? (
        <p className="text-xs text-amber-800">
          Vision failed
          {visionError ? ` (${visionError})` : ""}. Open Visual marks JSON for
          the stored targets and failure details.
        </p>
      ) : null}
      {marks.length > 0 ? (
        <div className="overflow-x-auto rounded-md border border-slate-200 bg-white">
          <table className="min-w-full text-left text-xs">
            <thead className="bg-slate-100 text-slate-600">
              <tr>
                <th className="px-2 py-1.5 font-medium">Page</th>
                <th className="px-2 py-1.5 font-medium">Document</th>
                <th className="px-2 py-1.5 font-medium">Mark</th>
                <th className="px-2 py-1.5 font-medium">Role</th>
                <th className="px-2 py-1.5 font-medium">Box (x, y, w×h)</th>
                <th className="px-2 py-1.5 font-medium">Conf.</th>
              </tr>
            </thead>
            <tbody>
              {marks.map((mark, index) => (
                <tr
                  key={`${mark.page}-${mark.marking_type}-${index}`}
                  className={
                    onHighlight && mark.bbox
                      ? "cursor-pointer hover:bg-blue-50"
                      : undefined
                  }
                  onClick={() => {
                    if (!onHighlight || !mark.bbox) {
                      return;
                    }
                    onHighlight({
                      page: mark.page,
                      x: mark.bbox.x,
                      y: mark.bbox.y,
                      width: mark.bbox.w,
                      height: mark.bbox.h,
                    });
                  }}
                >
                  <td className="px-2 py-1.5 tabular-nums">{mark.page}</td>
                  <td className="px-2 py-1.5">{mark.document_type}</td>
                  <td className="px-2 py-1.5">{mark.marking_type}</td>
                  <td className="px-2 py-1.5 text-slate-600">
                    {mark.signature_role &&
                    mark.signature_role !== "not_applicable"
                      ? mark.signature_role
                      : "—"}
                  </td>
                  <td className="px-2 py-1.5 font-mono text-[11px] text-slate-600">
                    {formatBox(mark.bbox)}
                  </td>
                  <td className="px-2 py-1.5 tabular-nums">
                    {mark.confidence != null
                      ? `${Math.round(mark.confidence * 100)}%`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
