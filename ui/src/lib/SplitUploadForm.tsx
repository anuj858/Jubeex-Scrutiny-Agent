import { useEffect, useMemo, useRef, useState } from "react";
import {
  Button,
  useCloudApiClient,
  useHandlers,
  useWorkflow,
  type HandlerState,
} from "@llamaindex/ui";
import { Download } from "lucide-react";
import { toast } from "sonner";
import { useMetadataContext } from "./MetadataProvider";
import type { SplitUploadSlot } from "./useMetadata";
import { clickDownloadUrl, downloadFile } from "./export";
import styles from "./SplitUploadForm.module.css";

type UploadedPart = {
  fileId: string;
  fileHash: string | null;
  filename: string;
};

export type PreparedPart = {
  slot_id: string;
  file_id: string;
  file_hash?: string | null;
  filename?: string | null;
};

export type BundlePrepared = {
  filing_type: string;
  parts: PreparedPart[];
  slot_pages?: Record<string, string>;
};

type PresignedFile = {
  url?: string;
  download_url?: string | { url?: string };
};

type CloudFiles = {
  files: {
    create: (args: {
      file: File;
      purpose: string;
    }) => Promise<{ id?: string; fileId?: string }>;
    get?: (fileId: string) => Promise<PresignedFile>;
    content?: (fileId: string) => Promise<PresignedFile>;
  };
};

async function sha256Hex(file: File): Promise<string> {
  const digest = await window.crypto.subtle.digest(
    "SHA-256",
    await file.arrayBuffer(),
  );
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function isPdf(file: File): boolean {
  return (
    file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")
  );
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function asPreparedParts(value: unknown): PreparedPart[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const parts: PreparedPart[] = [];
  for (const item of value) {
    const row = asRecord(item);
    if (!row) {
      continue;
    }
    const slotId = typeof row.slot_id === "string" ? row.slot_id : "";
    const fileId = typeof row.file_id === "string" ? row.file_id : "";
    if (!slotId || !fileId) {
      continue;
    }
    parts.push({
      slot_id: slotId,
      file_id: fileId,
      file_hash:
        typeof row.file_hash === "string" || row.file_hash === null
          ? row.file_hash
          : null,
      filename: typeof row.filename === "string" ? row.filename : null,
    });
  }
  return parts;
}

function presignedFileUrl(value: unknown): string | null {
  const raw = asRecord(value);
  if (!raw) {
    return null;
  }
  if (typeof raw.url === "string" && raw.url) {
    return raw.url;
  }
  if (typeof raw.download_url === "string" && raw.download_url) {
    return raw.download_url;
  }
  const nested = asRecord(raw.download_url);
  if (nested && typeof nested.url === "string" && nested.url) {
    return nested.url;
  }
  return null;
}

async function downloadCloudPdf(
  cloud: CloudFiles,
  fileId: string,
  filename: string,
) {
  const getter = cloud.files.get ?? cloud.files.content;
  if (!getter) {
    throw new Error("This LlamaCloud client cannot download files");
  }
  const url = presignedFileUrl(await getter(fileId));
  if (!url) {
    throw new Error(`No download URL for ${filename}`);
  }
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Download failed (${response.status})`);
    }
    downloadFile(await response.blob(), filename, "application/pdf");
  } catch {
    clickDownloadUrl(url, filename);
  }
}

function asSlotPages(value: unknown): Record<string, string> {
  const raw = asRecord(value);
  if (!raw) {
    return {};
  }
  const pages: Record<string, string> = {};
  for (const [key, item] of Object.entries(raw)) {
    if (typeof item === "string" && item) {
      pages[key] = item;
    }
  }
  return pages;
}

export function readBundlePrepared(payload: unknown): BundlePrepared | null {
  const event = asRecord(payload);
  if (!event) {
    return null;
  }
  const nested = asRecord(event.data) ?? asRecord(event.result);
  const source =
    event.type === "BundlePrepared" && nested
      ? nested
      : typeof event.filing_type === "string"
        ? event
        : nested && typeof nested.filing_type === "string"
          ? nested
          : null;
  if (!source || typeof source.filing_type !== "string") {
    return null;
  }
  return {
    filing_type: source.filing_type,
    parts: asPreparedParts(source.parts),
    slot_pages: asSlotPages(source.slot_pages),
  };
}

function stopSubscription(sub: {
  disconnect?: () => void;
  unsubscribe?: () => void;
} | null) {
  sub?.disconnect?.();
  sub?.unsubscribe?.();
}

export function SplitUploadForm({
  onStarted,
  prepareHandler,
}: {
  onStarted: (handler: HandlerState) => void;
  prepareHandler?: HandlerState | null;
}) {
  const { metadata } = useMetadataContext();
  const types = metadata.split_upload_types ?? {};
  const typeIds = useMemo(() => Object.keys(types), [types]);
  const [filingType, setFilingType] = useState(typeIds[0] ?? "");
  const [uploads, setUploads] = useState<Record<string, UploadedPart>>({});
  const [slotPages, setSlotPages] = useState<Record<string, string>>({});
  const [typeLocked, setTypeLocked] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [uploadingSlot, setUploadingSlot] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [downloadingSlot, setDownloadingSlot] = useState<string | null>(null);
  const inputRefs = useRef<Record<string, HTMLInputElement | null>>({});
  const preparedFor = useRef<string | null>(null);
  const cloud = useCloudApiClient() as unknown as CloudFiles;
  const wf = useWorkflow("process-split-files");
  const handlersService = useHandlers({
    query: { workflow_name: ["process-file"] },
    sync: false,
  });

  const catalog = filingType ? types[filingType] : undefined;
  const slots = catalog?.slots ?? [];

  const requiredReady = slots
    .filter((slot) => slot.required)
    .every((slot) => Boolean(uploads[slot.id]));

  const applyPrepared = (prepared: BundlePrepared, handlerId?: string) => {
    if (handlerId && preparedFor.current === handlerId) {
      setPreparing(false);
      return;
    }
    if (handlerId) {
      preparedFor.current = handlerId;
    }
    if (!types[prepared.filing_type]) {
      toast.error(
        `No split-upload form for ${prepared.filing_type}. Upload the documents manually.`,
      );
      setPreparing(false);
      return;
    }
    const nextUploads: Record<string, UploadedPart> = {};
    for (const part of prepared.parts) {
      nextUploads[part.slot_id] = {
        fileId: part.file_id,
        fileHash: part.file_hash ?? null,
        filename: part.filename || `${part.slot_id}.pdf`,
      };
    }
    setFilingType(prepared.filing_type);
    setUploads(nextUploads);
    setSlotPages(prepared.slot_pages ?? {});
    setTypeLocked(true);
    setPreparing(false);
    const found = Object.keys(nextUploads).length;
    toast.success(
      found
        ? `Loaded ${found} sliced file${found === 1 ? "" : "s"} from the bundled PDF`
        : "Split finished. Upload the missing required documents, then Submit.",
    );
  };

  useEffect(() => {
    if (!prepareHandler) {
      setPreparing(false);
      return;
    }
    const handlerId = prepareHandler.handler_id;
    if (preparedFor.current === handlerId) {
      setPreparing(false);
      return;
    }
    setPreparing(true);
    handlersService.setHandler(prepareHandler);
    const already = readBundlePrepared(prepareHandler.result);
    if (already) {
      applyPrepared(already, handlerId);
      return;
    }
    const sub = handlersService.actions(handlerId).subscribeToEvents({
      onData(event) {
        const prepared = readBundlePrepared(event);
        if (prepared) {
          applyPrepared(prepared, handlerId);
        }
      },
      onComplete() {
        const handler = handlersService.state.handlers[handlerId];
        const prepared =
          readBundlePrepared(handler?.result) ||
          readBundlePrepared(prepareHandler.result);
        if (prepared) {
          applyPrepared(prepared, handlerId);
          return;
        }
        setPreparing(false);
      },
      onError(error) {
        toast.error(
          error instanceof Error
            ? error.message
            : "Could not split the uploaded PDF",
        );
        setPreparing(false);
      },
    });
    return () => stopSubscription(sub);
  }, [prepareHandler?.handler_id]);

  const onChangeType = (next: string) => {
    if (typeLocked) {
      return;
    }
    setFilingType(next);
    setUploads({});
    setSlotPages({});
    setUploadingSlot(null);
  };

  const pickFile = (slot: SplitUploadSlot) => {
    inputRefs.current[slot.id]?.click();
  };

  const onFileChosen = async (slot: SplitUploadSlot, file: File | undefined) => {
    if (!file) {
      return;
    }
    if (!isPdf(file)) {
      toast.error(`${slot.label} must be a PDF`);
      const input = inputRefs.current[slot.id];
      if (input) {
        input.value = "";
      }
      return;
    }
    setUploadingSlot(slot.id);
    try {
      const fileHash = await sha256Hex(file);
      const created = await cloud.files.create({
        file,
        purpose: "extract",
      });
      const fileId = created.id ?? created.fileId;
      if (!fileId) {
        throw new Error("Upload did not return a file id");
      }
      setUploads((prev) => ({
        ...prev,
        [slot.id]: {
          fileId,
          fileHash,
          filename: file.name,
        },
      }));
      setSlotPages((prev) => {
        const next = { ...prev };
        delete next[slot.id];
        return next;
      });
    } catch (error) {
      toast.error(
        `Could not upload ${slot.label}: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    } finally {
      setUploadingSlot(null);
      const input = inputRefs.current[slot.id];
      if (input) {
        input.value = "";
      }
    }
  };

  const downloadPart = async (slot: SplitUploadSlot) => {
    const uploaded = uploads[slot.id];
    if (!uploaded || downloadingSlot) {
      return;
    }
    setDownloadingSlot(slot.id);
    try {
      await downloadCloudPdf(cloud, uploaded.fileId, uploaded.filename);
    } catch (error) {
      toast.error(
        `Could not download ${slot.label}: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    } finally {
      setDownloadingSlot(null);
    }
  };

  const downloadAllParts = async () => {
    if (downloadingSlot) {
      return;
    }
    const ready = slots
      .map((slot) => ({ slot, uploaded: uploads[slot.id] }))
      .filter(
        (row): row is { slot: SplitUploadSlot; uploaded: UploadedPart } =>
          Boolean(row.uploaded),
      );
    if (ready.length === 0) {
      return;
    }
    setDownloadingSlot("all");
    let failed = 0;
    for (const { slot, uploaded } of ready) {
      try {
        await downloadCloudPdf(cloud, uploaded.fileId, uploaded.filename);
      } catch {
        failed += 1;
        toast.error(`Could not download ${slot.label}`);
      }
    }
    if (failed === 0) {
      toast.success(
        `Downloaded ${ready.length} sliced file${ready.length === 1 ? "" : "s"}`,
      );
    }
    setDownloadingSlot(null);
  };

  const removeFile = (slotId: string) => {
    setUploads((prev) => {
      const next = { ...prev };
      delete next[slotId];
      return next;
    });
    setSlotPages((prev) => {
      const next = { ...prev };
      delete next[slotId];
      return next;
    });
  };

  const onSubmit = async () => {
    if (
      !filingType ||
      !catalog ||
      !requiredReady ||
      submitting ||
      preparing
    ) {
      return;
    }
    setSubmitting(true);
    try {
      const parts: {
        slot_id: string;
        document_parts: string[];
        file_id: string;
        file_hash: string | null;
        filename: string;
      }[] = [];
      for (const slot of slots) {
        const uploaded = uploads[slot.id];
        if (!uploaded) {
          continue;
        }
        parts.push({
          slot_id: slot.id,
          document_parts: slot.parts,
          file_id: uploaded.fileId,
          file_hash: uploaded.fileHash,
          filename: uploaded.filename,
        });
      }
      const created = await wf.createHandler({
        filing_type: filingType,
        parts,
      });
      onStarted(created);
      toast.success(`Started ${catalog.label} split upload`);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : String(error),
      );
    } finally {
      setSubmitting(false);
    }
  };

  if (typeIds.length === 0) {
    return null;
  }

  const formBusy =
    preparing ||
    submitting ||
    Boolean(uploadingSlot) ||
    Boolean(downloadingSlot);
  const slicedCount = Object.keys(uploads).length;

  return (
    <section className={styles.panel}>
      <div className={styles.header}>
        <div>
          <h2 className={styles.title}>Split document upload</h2>
          <p className={styles.subtitle}>
            {preparing
              ? "Classifying and splitting the uploaded PDF into document files…"
              : typeLocked
                ? "Files below were sliced from the bundled PDF. Upload any missing required documents, then Submit."
                : "Choose the matter type, then upload each document. Submit maps each PDF to its document part — the filename is not used."}
          </p>
        </div>
        <label className={styles.typeLabel}>
          Matter type
          <select
            className={styles.select}
            value={filingType}
            disabled={typeLocked || preparing}
            onChange={(event) => onChangeType(event.target.value)}
          >
            {typeIds.map((id) => (
              <option key={id} value={id}>
                {types[id]?.label || id}
              </option>
            ))}
          </select>
        </label>
      </div>

      <ul className={styles.list}>
        {slots.map((slot) => {
          const uploaded = uploads[slot.id];
          const busy = uploadingSlot === slot.id;
          const pages = slotPages[slot.id];
          return (
            <li key={slot.id} className={styles.row}>
              <div className={styles.name}>
                <span>{slot.label}</span>
                {slot.required ? (
                  <span className={styles.required}>required</span>
                ) : (
                  <span className={styles.optional}>optional</span>
                )}
                {pages ? (
                  <span className={styles.pageSpan}>{pages}</span>
                ) : null}
              </div>
              <div className={styles.actions}>
                {uploaded ? (
                  <span className={styles.filename} title={uploaded.filename}>
                    {uploaded.filename}
                  </span>
                ) : null}
                {uploaded ? (
                  <Button
                    size="sm"
                    variant="ghost"
                    label={
                      downloadingSlot === slot.id ? "Downloading…" : "Download"
                    }
                    startIcon={<Download className="h-3.5 w-3.5" />}
                    disabled={formBusy}
                    title={`Download the ${slot.label} PDF LlamaSplit assigned to this slot`}
                    onClick={() => void downloadPart(slot)}
                  />
                ) : null}
                <input
                  ref={(node) => {
                    inputRefs.current[slot.id] = node;
                  }}
                  type="file"
                  accept="application/pdf,.pdf"
                  className={styles.hiddenInput}
                  onChange={(event) =>
                    void onFileChosen(slot, event.target.files?.[0])
                  }
                />
                <Button
                  size="sm"
                  variant="outline"
                  label={
                    busy
                      ? "Uploading…"
                      : uploaded
                        ? "Replace"
                        : "Upload"
                  }
                  disabled={formBusy}
                  onClick={() => pickFile(slot)}
                />
                {uploaded ? (
                  <Button
                    size="sm"
                    variant="ghost"
                    label="Remove"
                    disabled={formBusy}
                    onClick={() => removeFile(slot.id)}
                  />
                ) : null}
              </div>
            </li>
          );
        })}
      </ul>

      <div className={styles.footer}>
        {slicedCount > 0 ? (
          <Button
            size="sm"
            variant="outline"
            label={
              downloadingSlot === "all" ? "Downloading…" : "Download all"
            }
            startIcon={<Download className="h-3.5 w-3.5" />}
            disabled={formBusy}
            title="Download every sliced PDF LlamaSplit assigned to a slot"
            onClick={() => void downloadAllParts()}
          />
        ) : (
          <span />
        )}
        <Button
          label={
            preparing
              ? "Preparing…"
              : submitting
                ? "Submitting…"
                : "Submit"
          }
          disabled={!requiredReady || formBusy}
          onClick={() => void onSubmit()}
        />
      </div>
    </section>
  );
}
