import {
  ItemCount,
  WorkflowTrigger,
  ExtractedDataItemGrid,
  HandlerState,
  AgentDataItem,
  Button,
} from "@llamaindex/ui";
import styles from "./HomePage.module.css";
import { useNavigate } from "react-router-dom";
import { useMemo, useState } from "react";
import { ShieldCheck } from "lucide-react";
import { WorkflowProgress } from "@/lib/WorkflowProgress";
import { ScrutinyDialog } from "@/lib/ScrutinyDialog";
import { isApproved, useScrutiny } from "@/lib/scrutiny";

export default function HomePage() {
  return <TaskList />;
}

function TaskList() {
  const navigate = useNavigate();
  const goToItem = (item: AgentDataItem) => {
    navigate(`/item/${item.id}`);
  };
  const [reloadSignal, setReloadSignal] = useState(0);
  const [handlers, setHandlers] = useState<HandlerState[]>([]);
  const scrutiny = useScrutiny();
  const { run: runScrutiny, running: scrutinyRunning } = scrutiny;

  const scrutinyColumn = useMemo(
    () => ({
      key: "scrutiny",
      header: "",
      getValue: (item: AgentDataItem) => item,
      renderCell: (value: unknown) => {
        const item = value as AgentDataItem;
        const approved = isApproved(item);
        return (
          <div
            onClick={(e) => {
              // Keep the row click from navigating to the item page.
              e.stopPropagation();
            }}
          >
            <Button
              size="sm"
              variant="outline"
              label="Check defects"
              startIcon={<ShieldCheck className="h-3.5 w-3.5" />}
              disabled={!approved || scrutinyRunning}
              title={
                approved
                  ? "Check this filing against the registry defect catalogue"
                  : "Only approved filings can be checked"
              }
              onClick={() => runScrutiny(item)}
            />
          </div>
        );
      },
    }),
    [runScrutiny, scrutinyRunning],
  );

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <div className={styles.grid}>
          <ItemCount title="Total Items" key={`total-items-${reloadSignal}`} />
          <ItemCount
            title="Reviewed"
            filter={{
              status: { includes: ["approved", "rejected"] },
            }}
            key={`reviewed-${reloadSignal}`}
          />
          <ItemCount
            title="Needs Review"
            filter={{
              status: { eq: "pending_review" },
            }}
            key={`needs-review-${reloadSignal}`}
          />
        </div>
        <div className={styles.commandBar}>
          <WorkflowProgress
            workflowName="process-file"
            handlers={handlers}
            onWorkflowCompletion={() => {
              setReloadSignal(reloadSignal + 1);
            }}
          />
          <WorkflowTrigger
            workflowName="process-file"
            contentHash={{ enabled: true }}
            customWorkflowInput={(files) => {
              return {
                file_id: files[0].fileId,
                file_hash: files[0].contentHash ?? null,
              };
            }}
            onSuccess={(handler) => {
              setHandlers([...handlers, handler]);
            }}
          />
        </div>

        <ExtractedDataItemGrid
          key={reloadSignal}
          onRowClick={goToItem}
          customColumns={[scrutinyColumn]}
          builtInColumns={{
            fileName: true,
            status: true,
            createdAt: true,
            itemsToReview: true,
            actions: true,
          }}
        />

        <ScrutinyDialog
          target={scrutiny.target}
          running={scrutiny.running}
          report={scrutiny.report}
          error={scrutiny.error}
          onClose={scrutiny.close}
        />
      </main>
    </div>
  );
}
