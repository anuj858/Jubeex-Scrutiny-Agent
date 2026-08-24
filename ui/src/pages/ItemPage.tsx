import { useEffect, useMemo, useState } from "react";
import {
  AcceptReject,
  ExtractedDataDisplay,
  FilePreview,
  useItemData,
  type Highlight,
  type ExtractedData,
  Button,
} from "@llamaindex/ui";
import { Clock, XCircle, Download, AlertCircle } from "lucide-react";
import { useParams } from "react-router-dom";
import { useToolbar } from "@/lib/ToolbarContext";
import { useNavigate } from "react-router-dom";
import { modifyJsonSchema } from "@llamaindex/ui/lib";
import { APP_TITLE } from "@/lib/config";
import { downloadExtractedDataItem } from "@/lib/export";
import { useMetadataContext } from "@/lib/MetadataProvider";
import { convertBoundingBoxesToHighlights } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Schema-driven validation: reads "required" arrays from config.json schema
// ---------------------------------------------------------------------------

type MissingField = {
  field_path: string;
  label: string;
};

/**
 * Walk the JSON schema and the extracted data together.
 * For every field listed in a "required" array, check if the value
 * in the extracted data is actually present (non-null, non-empty).
 */
function findMissingRequired(
  schema: Record<string, any>,
  data: Record<string, any> | null | undefined,
  prefix: string = "",
): MissingField[] {
  if (!schema || !schema.properties) return [];

  const requiredSet = new Set<string>(schema.required || []);
  const missing: MissingField[] = [];

  for (const [key, fieldSchema] of Object.entries<any>(schema.properties)) {
    const fullPath = prefix ? `${prefix}.${key}` : key;
    const value = data ? data[key] : undefined;
    const label = fieldSchema.title || key;

    // Check if this field is in the required array
    if (requiredSet.has(key)) {
      if (value == null) {
        missing.push({ field_path: fullPath, label });
      } else if (typeof value === "string" && value.trim() === "") {
        missing.push({ field_path: fullPath, label });
      } else if (Array.isArray(value) && value.length === 0) {
        missing.push({ field_path: fullPath, label });
      }
    }

    // Recurse into nested objects to check their required fields too
    // Handle "anyOf" pattern: [{ type: "object", required: [...], properties: {...} }, { type: "null" }]
    const objectSchema = resolveObjectSchema(fieldSchema);
    if (objectSchema && value && typeof value === "object" && !Array.isArray(value)) {
      missing.push(...findMissingRequired(objectSchema, value, fullPath));
    }
  }

  return missing;
}

/** Extract the object schema from an anyOf/oneOf wrapper or direct definition. */
function resolveObjectSchema(fieldSchema: any): Record<string, any> | null {
  if (!fieldSchema) return null;

  // Direct object with properties
  if (fieldSchema.type === "object" && fieldSchema.properties) {
    return fieldSchema;
  }

  // anyOf: [{ type: "object", ... }, { type: "null" }]
  const candidates = fieldSchema.anyOf || fieldSchema.oneOf || [];
  for (const candidate of candidates) {
    if (candidate.type === "object" && candidate.properties) {
      return candidate;
    }
  }

  return null;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ItemPage() {
  const { itemId } = useParams<{ itemId: string }>();
  const { setButtons, setBreadcrumbs } = useToolbar();
  const [highlight, setHighlight] = useState<Highlight | undefined>(undefined);
  const { metadata } = useMetadataContext();

  // Use the hook to fetch item data (initially with a default schema)
  const itemHookData = useItemData<any>({
    // We'll update the schema based on classification once data loads
    jsonSchema: modifyJsonSchema(metadata.schemas["10-K"] || {}, {}),
    itemId: itemId as string,
    isMock: false,
  });

  // Determine the correct schema based on classification
  const classificationData = itemHookData.item?.data as
    | ExtractedData<any>
    | undefined;
  const classification = (
    (classificationData?.metadata?.classification as string | undefined) ||
    "10-K"
  ).toUpperCase();
  const correctSchema =
    metadata.schemas[classification] || metadata.schemas["10-K"];

  // Update the schema in itemHookData if classification is available
  const [schemaKey, setSchemaKey] = useState(0);
  const [appliedSchema, setAppliedSchema] = useState(correctSchema);

  useEffect(() => {
    if (classification && metadata.schemas[classification]) {
      setAppliedSchema(modifyJsonSchema(metadata.schemas[classification], {}));
      setSchemaKey(schemaKey + 1);
    }
  }, [classification, metadata.schemas]);

  const navigate = useNavigate();

  useEffect(() => {
    const extractedData = itemHookData.item?.data as
      | ExtractedData<unknown>
      | undefined;
    const fileName = extractedData?.file_name;
    if (fileName) {
      setBreadcrumbs([
        { label: APP_TITLE, href: "/" },
        {
          label: fileName,
          isCurrentPage: true,
        },
      ]);
    }

    return () => {
      setBreadcrumbs([{ label: APP_TITLE, href: "/" }]);
    };
  }, [itemHookData.item?.data, setBreadcrumbs]);

  useEffect(() => {
    setButtons(() => [
      <div className="ml-auto flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            if (itemData) {
              downloadExtractedDataItem(itemData);
            }
          }}
          disabled={!itemData}
          startIcon={<Download className="h-4 w-4" />}
          label="Export JSON"
        />
        <AcceptReject<any>
          itemData={itemHookData}
          onComplete={() => navigate("/")}
        />
      </div>,
    ]);
    return () => {
      setButtons(() => []);
    };
  }, [itemHookData.data, setButtons]);

  const {
    item: itemData,
    updateData,
    loading: isLoading,
    error,
  } = itemHookData;

  const classificationReasoning = (itemData?.data as ExtractedData<any>)
    ?.metadata?.classification_reasoning as string | undefined;

  // --- Validation: read "required" from the schema, check against data ---
  const missingFields = useMemo(() => {
    const rawSchema = correctSchema; // the JSON schema from config.json
    const ed = itemData?.data as ExtractedData<any> | undefined;
    const extractedRecord = ed?.data as Record<string, any> | undefined;
    if (!rawSchema || !extractedRecord) return [];
    return findMissingRequired(rawSchema, extractedRecord);
  }, [correctSchema, itemData?.data]);

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <Clock className="h-8 w-8 animate-spin mx-auto mb-2" />
          <div className="text-sm text-gray-500">Loading item...</div>
        </div>
      </div>
    );
  }

  if (error || !itemData) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <XCircle className="h-8 w-8 text-red-500 mx-auto mb-2" />
          <div className="text-sm text-gray-500">
            Error loading item: {error || "Item not found"}
          </div>
        </div>
      </div>
    );
  }

  const extractedData = itemData.data as ExtractedData<any>;
  const fileId = extractedData.file_id;

  return (
    <div className="flex h-full bg-gray-50">
      <div className="w-1/2 border-r h-full border-gray-200 bg-white">
        {fileId && (
          <FilePreview
            fileId={fileId}
            onBoundingBoxClick={(box, pageNumber) => {
              console.log("Bounding box clicked:", box, "on page:", pageNumber);
            }}
            highlight={highlight}
          />
        )}
      </div>

      <div className="flex-1 bg-white h-full overflow-y-auto">
        <div className="p-4 space-y-4">
          {/* Classification Info */}
          {classification && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
              <div className="text-sm font-semibold text-blue-900">
                Document Type: {classification}
              </div>
              {classificationReasoning && (
                <div className="text-xs text-blue-600 mt-1">
                  {classificationReasoning}
                </div>
              )}
            </div>
          )}

          {/* Missing Required Fields */}
          {missingFields.length > 0 && (
            <div
              className="rounded-lg border border-red-300 bg-red-50 p-3 mb-4"
            >
              <div className="flex items-center gap-2 mb-2">
                <AlertCircle className="h-4 w-4 text-red-600 shrink-0" />
                <span className="text-sm font-semibold text-red-800">
                  {missingFields.length} required field(s) missing
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {missingFields.map((f, i) => (
                  <span
                    key={`${f.field_path}-${i}`}
                    className="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium"
                    style={{
                      backgroundColor: "#fee2e2",
                      color: "#991b1b",
                      border: "1px solid #fca5a5",
                    }}
                  >
                    <AlertCircle className="h-3 w-3" />
                    {f.label}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* All clear */}
          {missingFields.length === 0 && itemData && (
            <div className="rounded-lg border border-green-200 bg-green-50 p-3 mb-4">
              <div className="flex items-center gap-2">
                <svg className="h-4 w-4 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-sm font-semibold text-green-800">
                  All required fields present
                </span>
              </div>
            </div>
          )}

          <ExtractedDataDisplay<any>
            key={schemaKey}
            extractedData={extractedData}
            title="Extracted Data"
            onChange={(updatedData) => {
              updateData(updatedData);
            }}
            onHoverField={(args) => {
              const highlights = convertBoundingBoxesToHighlights(
                args?.metadata?.citation,
              );
              setHighlight(highlights[0]);
            }}
            jsonSchema={appliedSchema}
          />
        </div>
      </div>
    </div>
  );
}
