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
