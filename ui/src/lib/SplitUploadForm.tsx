import { useMemo, useRef, useState } from "react";
import {
  Button,
  hashFile,
  useCloudApiClient,
  useWorkflow,
  type HandlerState,
} from "@llamaindex/ui";
import { toast } from "sonner";
import { useMetadataContext } from "./MetadataProvider";
import type { SplitUploadSlot } from "./useMetadata";
import styles from "./SplitUploadForm.module.css";

type UploadedPart = {
  fileId: string;
  fileHash: string | null;
  filename: string;
};

type CloudFileCreate = {
  files: {
    create: (args: {
      file: File;
      purpose: string;
    }) => Promise<{ id?: string; fileId?: string }>;
  };
};

function isPdf(file: File): boolean {
  return (
    file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")
  );
}

export function SplitUploadForm({
  onStarted,
}: {
  onStarted: (handler: HandlerState) => void;
}) {
  const { metadata } = useMetadataContext();
  const types = metadata.split_upload_types ?? {};
  const typeIds = useMemo(() => Object.keys(types), [types]);
  const [filingType, setFilingType] = useState(typeIds[0] ?? "");
  const [uploads, setUploads] = useState<Record<string, UploadedPart>>({});
  const [uploadingSlot, setUploadingSlot] = useState<string | None>(null);
  const [submitting, setSubmitting] = useState(false);
  const inputRefs = useRef<Record<string, HTMLInputElement | null>>({});
  const cloud = useCloudApiClient() as unknown as CloudFileCreate;
  const wf = useWorkflow("process-split-files");

  const catalog = filingType ? types[filingType] : undefined;
  const slots = catalog?.slots ?? [];

  const requiredReady = slots
    .filter((slot) => slot.required)
    .every((slot) => Boolean(uploads[slot.id]));

  const onChangeType = (next: string) => {
    setFilingType(next);
    setUploads({});
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
      const fileHash = await hashFile(file);
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

  const removeFile = (slotId: string) => {
    setUploads((prev) => {
      const next = { ...prev };
      delete next[slotId];
      return next;
    });
  };

  const onSubmit = async () => {
    if (!filingType || !catalog || !requiredReady || submitting) {
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

  return (
    <section className={styles.panel}>
      <div className={styles.header}>
        <div>
          <h2 className={styles.title}>Split document upload</h2>
          <p className={styles.subtitle}>
            Choose the matter type, then upload each document. Submit maps each
            PDF to its document part — the filename is not used.
          </p>
        </div>
        <label className={styles.typeLabel}>
          Matter type
          <select
            className={styles.select}
            value={filingType}
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
          return (
            <li key={slot.id} className={styles.row}>
              <div className={styles.name}>
                <span>{slot.label}</span>
                {slot.required ? (
                  <span className={styles.required}>required</span>
                ) : (
                  <span className={styles.optional}>optional</span>
                )}
              </div>
              <div className={styles.actions}>
                {uploaded ? (
                  <span className={styles.filename} title={uploaded.filename}>
                    {uploaded.filename}
                  </span>
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
                  disabled={busy || submitting}
                  onClick={() => pickFile(slot)}
                />
                {uploaded ? (
                  <Button
                    size="sm"
                    variant="ghost"
                    label="Remove"
                    disabled={busy || submitting}
                    onClick={() => removeFile(slot.id)}
                  />
                ) : null}
              </div>
            </li>
          );
        })}
      </ul>

      <div className={styles.footer}>
        <Button
          label={submitting ? "Submitting…" : "Submit"}
          disabled={!requiredReady || submitting || Boolean(uploadingSlot)}
          onClick={() => void onSubmit()}
        />
      </div>
    </section>
  );
}
