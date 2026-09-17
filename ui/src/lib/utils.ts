import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { Highlight, FieldCitation } from "@llamaindex/ui";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDuration(seconds: number | null | undefined): string | null {
  if (seconds == null || !Number.isFinite(seconds)) {
    return null;
  }
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours) {
    return `${hours}h ${minutes}m ${secs}s`;
  }
  if (minutes) {
    return `${minutes}m ${secs}s`;
  }
  if (seconds < 10) {
    return `${seconds.toFixed(1)}s`;
  }
  return `${total}s`;
}

export type JobTiming = {
  file_name?: string | null;
  classify_split_seconds?: number | null;
  parse_extract_seconds?: number | null;
  total_seconds?: number | null;
  classify_split?: string | null;
  parse_extract?: string | null;
  total?: string | null;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

export type S3ArtifactLink = {
  id: string;
  label: string;
  url?: string;
  key?: string;
};

const ARTIFACT_FIELDS: { id: string; label: string; urlKey: string; keyKey: string }[] =
  [
    {
      id: "visual",
      label: "Visual marks",
      urlKey: "visual_artifact_url",
      keyKey: "visual_artifact_key",
    },
    {
      id: "layout",
      label: "Page layout",
      urlKey: "layout_artifact_url",
      keyKey: "layout_artifact_key",
    },
    {
      id: "parse",
      label: "Parse",
      urlKey: "parse_artifact_url",
      keyKey: "parse_artifact_key",
    },
  ];

export function readItemMetadata(source: unknown): Record<string, unknown> | null {
  const record = asRecord(source);
  if (!record) {
    return null;
  }
  const nested = asRecord(record.data);
  return asRecord(record.metadata) ?? asRecord(nested?.metadata);
}

export function readS3Artifacts(source: unknown): S3ArtifactLink[] {
  const metadata = readItemMetadata(source);
  if (!metadata) {
    return [];
  }
  const links: S3ArtifactLink[] = [];
  for (const field of ARTIFACT_FIELDS) {
    const url =
      typeof metadata[field.urlKey] === "string"
        ? (metadata[field.urlKey] as string).trim()
        : "";
    const key =
      typeof metadata[field.keyKey] === "string"
        ? (metadata[field.keyKey] as string).trim()
        : "";
    if (!url && !key) {
      continue;
    }
    links.push({
      id: field.id,
      label: field.label,
      url: url || undefined,
      key: key || undefined,
    });
  }
  return links;
}

export function visualArtifactUrl(source: unknown): string | undefined {
  return readS3Artifacts(source).find((item) => item.id === "visual")?.url;
}

export type VisualFailure = {
  page: number;
  slot_id?: string | null;
  document_types: string[];
  reason: string;
  detail?: string | null;
};

export type VisualTargetRow = {
  page: number;
  document_types: string[];
};

export type VisualUsage = {
  cost_usd?: number | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  calls?: number | null;
  model?: string | null;
};

export type VisualSummary = {
  status?: string | null;
  error?: string | null;
  mark_count: number;
  target_count: number;
  failures: VisualFailure[];
  targets: VisualTargetRow[];
  usage?: VisualUsage | null;
};

function asFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function parseVisualTargets(raw: unknown): VisualTargetRow[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  const rows: VisualTargetRow[] = [];
  for (const item of raw) {
    const record = asRecord(item);
    if (!record) {
      continue;
    }
    const page = asFiniteNumber(record.page);
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

function parseVisualFailures(raw: unknown): VisualFailure[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  const rows: VisualFailure[] = [];
  for (const item of raw) {
    const record = asRecord(item);
    if (!record) {
      continue;
    }
    const page = asFiniteNumber(record.page);
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

function parseVisualUsage(raw: unknown): VisualUsage | null {
  const record = asRecord(raw);
  if (!record) {
    return null;
  }
  const cost = asFiniteNumber(record.cost_usd);
  const calls = asFiniteNumber(record.calls);
  const prompt = asFiniteNumber(record.prompt_tokens);
  const completion = asFiniteNumber(record.completion_tokens);
  const total = asFiniteNumber(record.total_tokens);
  const model = typeof record.model === "string" ? record.model : null;
  if (
    cost == null &&
    calls == null &&
    prompt == null &&
    completion == null &&
    total == null &&
    !model
  ) {
    return null;
  }
  return {
    cost_usd: cost,
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: total,
    calls: calls,
    model,
  };
}

export function readVisualSummary(source: unknown): VisualSummary | null {
  const metadata = readItemMetadata(source);
  const raw = asRecord(metadata?.visual_summary);
  if (!raw) {
    return null;
  }
  const markCount = asFiniteNumber(raw.mark_count) ?? 0;
  const targetCount = asFiniteNumber(raw.target_count);
  const targets = parseVisualTargets(raw.targets);
  return {
    status: typeof raw.status === "string" ? raw.status : null,
    error: typeof raw.error === "string" ? raw.error : null,
    mark_count: markCount,
    target_count: targetCount ?? targets.length,
    failures: parseVisualFailures(raw.failures),
    targets,
    usage: parseVisualUsage(raw.usage),
  };
}

export function readJobTiming(source: unknown): JobTiming | null {
  const record = asRecord(source);
  if (!record) {
    return null;
  }
  const nested = asRecord(record.data);
  const metadata = asRecord(record.metadata) ?? asRecord(nested?.metadata);
  const timing =
    asRecord(record.timing) ??
    asRecord(metadata?.timing) ??
    asRecord(nested?.timing);
  if (!timing) {
    return null;
  }
  const classify =
    typeof timing.classify_split_seconds === "number"
      ? timing.classify_split_seconds
      : null;
  const parse =
    typeof timing.parse_extract_seconds === "number"
      ? timing.parse_extract_seconds
      : null;
  const total =
    typeof timing.total_seconds === "number" ? timing.total_seconds : null;
  return {
    file_name: typeof timing.file_name === "string" ? timing.file_name : null,
    classify_split_seconds: classify,
    parse_extract_seconds: parse,
    total_seconds: total,
    classify_split:
      typeof timing.classify_split === "string"
        ? timing.classify_split
        : formatDuration(classify),
    parse_extract:
      typeof timing.parse_extract === "string"
        ? timing.parse_extract
        : formatDuration(parse),
    total: typeof timing.total === "string" ? timing.total : formatDuration(total),
  };
}

export function timingLabel(timing: JobTiming | null | undefined): string | null {
  if (!timing) {
    return null;
  }
  const bits: string[] = [];
  if (timing.classify_split) {
    bits.push(`bundle upload ${timing.classify_split}`);
  }
  if (timing.parse_extract) {
    bits.push(`slot submit ${timing.parse_extract}`);
  }
  if (timing.total && bits.length > 1) {
    bits.push(`total ${timing.total}`);
  }
  if (bits.length === 0) {
    return null;
  }
  const name = timing.file_name?.trim();
  return name ? `${name}: ${bits.join(" · ")}` : bits.join(" · ");
}

export function convertBoundingBoxesToHighlights(
  citations: FieldCitation[] | undefined,
): Highlight[] {
  if (!citations || citations.length === 0) return [];

  const highlights: Highlight[] = [];

  for (const citation of citations) {
    const page = citation.page ?? 1;
    const boundingBoxes = citation.bounding_boxes;

    if (boundingBoxes && boundingBoxes.length > 0) {
      for (const bbox of boundingBoxes) {
        highlights.push({
          page,
          x: bbox.x,
          y: bbox.y,
          width: bbox.w,
          height: bbox.h,
        });
      }
    } else if (citation.page !== undefined) {
      highlights.push({
        page,
        x: 0,
        y: 0,
        width: 0,
        height: 0,
      });
    }
  }

  return highlights;
}
