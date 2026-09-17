import { useEffect, useState } from "react";
import { ExternalLink, FileJson, Loader2 } from "lucide-react";
import type { Highlight } from "@llamaindex/ui";
import { readS3Artifacts, type S3ArtifactLink } from "./utils";

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

function formatBox(bbox?: VisualMarkRow["bbox"]): string {
  if (!bbox) {
    return "—";
  }
  return `${bbox.x.toFixed(2)}, ${bbox.y.toFixed(2)}, ${bbox.w.toFixed(2)}×${bbox.h.toFixed(2)}`;
}

function ArtifactLink({ item }: { item: S3ArtifactLink }) {
  if (!item.url) {
    return (
      <span className="text-xs text-slate-500" title={item.key}>
        {item.label}: stored at {item.key}
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
      {item.label} JSON
    </a>
  );
}

export function VisualMarksPanel({
  extractedData,
  onHighlight,
}: {
  extractedData: unknown;
  onHighlight?: (highlight: Highlight) => void;
}) {
  const artifacts = readS3Artifacts(extractedData);
  const visual = artifacts.find((item) => item.id === "visual");
  const others = artifacts.filter((item) => item.id !== "visual");
  const [marks, setMarks] = useState<VisualMarkRow[]>([]);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">(
    visual?.url ? "loading" : "idle",
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!visual?.url) {
      setMarks([]);
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
        setMarks(parseMarks(payload));
        setStatus("ready");
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setMarks([]);
        setStatus("error");
        setError(
          cause instanceof Error
            ? cause.message
            : "Could not preview visual JSON in the browser",
        );
      });
    return () => controller.abort();
  }, [visual?.url]);

  if (!visual && others.length === 0) {
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
      {status === "loading" ? (
        <div className="flex items-center gap-2 text-xs text-slate-600">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Loading marks from S3…
        </div>
      ) : null}
      {status === "error" ? (
        <p className="text-xs text-amber-800">
          {error}. Use the Visual marks JSON link above to download and inspect
          the file.
        </p>
      ) : null}
      {status === "ready" && marks.length === 0 ? (
        <p className="text-xs text-slate-600">
          Visual JSON is stored, but no ink marks were found on the formality
          pages.
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
