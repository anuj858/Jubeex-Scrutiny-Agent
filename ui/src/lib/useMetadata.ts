import { useWorkflow } from "@llamaindex/ui";
import { useEffect, useRef, useState } from "react";

export interface SplitUploadSlot {
  id: string;
  label: string;
  parts: string[];
  required: boolean;
}

export interface SplitUploadType {
  label: string;
  slots: SplitUploadSlot[];
}

export interface Metadata {
  schemas: Record<string, any>;
  extracted_data_collection: string;
  split_upload_types?: Record<string, SplitUploadType>;
}

export interface UseMetadataResult {
  metadata: Metadata;
  loading: boolean;
  error: string | undefined;
}

const METADATA_CACHE_KEY = "jubeex-metadata-v2";

function readCachedMetadata(): Metadata | undefined {
  try {
    const raw = sessionStorage.getItem(METADATA_CACHE_KEY);
    if (!raw) {
      return undefined;
    }
    const parsed = JSON.parse(raw) as Metadata;
    if (parsed?.schemas && parsed.extracted_data_collection) {
      return parsed;
    }
  } catch {
    /* ignore invalid cache */
  }
  return undefined;
}

function writeCachedMetadata(metadata: Metadata) {
  try {
    sessionStorage.setItem(METADATA_CACHE_KEY, JSON.stringify(metadata));
  } catch {
    /* quota / private mode */
  }
}

export function useMetadata() {
  const wf = useWorkflow("metadata");
  const cached = useRef(readCachedMetadata()).current;
  const [error, setError] = useState<string | undefined>(undefined);
  const [loading, setLoading] = useState(!cached);
  const [metadata, setMetadata] = useState<Metadata | undefined>(cached);
  const strictModeWorkaround = useRef(false);
  useEffect(() => {
    if (strictModeWorkaround.current) {
      return;
    }
    strictModeWorkaround.current = true;
    if (!cached) {
      setLoading(true);
    }
    wf.runToCompletion({})
      .then((handler) => {
        if (handler.status === "completed") {
          const result = handler.result?.data as unknown as Metadata;
          writeCachedMetadata(result);
          setMetadata(result);
        } else if (!cached) {
          setError(
            handler.error || `Unexpected workflow status: ${handler.status}`,
          );
        }
      })
      .catch((error) => {
        if (!cached) {
          setError(error instanceof Error ? error.message : String(error));
        }
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  return { metadata, loading, error };
}
