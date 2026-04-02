import { configure } from '@dnd-kit/abstract';
import { DragDropProvider, DragOverlay, KeyboardSensor, PointerSensor } from '@dnd-kit/react';
import { useMemo, type ReactNode } from 'react';

import { cn } from '../../lib/utils';
import { GPU_SYSTEM_OVERHEAD_BYTES } from '../../lib/vram';

type DragModelData = {
  type: 'model';
  modelName: string;
  sizeBytes: number;
  nodeIds: string[];
};

type DragSplitAssignmentData = {
  type: 'split-assignment';
  assignmentId: string;
  sourceNodeId: string;
  modelName: string;
  sizeBytes?: number;
};

function isDragModelData(data: unknown): data is DragModelData {
  if (!data || typeof data !== 'object') return false;
  const record = data as Record<string, unknown>;
  return record.type === 'model' && typeof record.modelName === 'string' && typeof record.sizeBytes === 'number';
}

function isDragSplitAssignmentData(data: unknown): data is DragSplitAssignmentData {
  if (!data || typeof data !== 'object') return false;
  const record = data as Record<string, unknown>;
  return (
    record.type === 'split-assignment'
    && typeof record.assignmentId === 'string'
    && typeof record.sourceNodeId === 'string'
    && typeof record.modelName === 'string'
  );
}

type DndContextProps = {
  children: ReactNode;
  selectedNodeId: string | null;
  selectedNodeVramBytes?: number;
  onAssignModel?: (modelName: string, sizeBytes: number, nodeId: string) => void;
  onMoveSplitAssignment?: (move: { assignmentId: string; sourceNodeId: string; targetNodeId: string }) => void;
};

export const DRAG_INTERACTIVE_ATTRIBUTE = 'data-dnd-interactive';
export const VRAM_DROP_TARGET_PREFIX = 'vram-container:';
export const CONFIG_NODE_DROP_TARGET_PREFIX = 'config-node:';

function hasInteractiveDragAncestor(target: EventTarget | null) {
  return target instanceof Element && target.closest(`[${DRAG_INTERACTIVE_ATTRIBUTE}]`) != null;
}

function resolveDropTargetNodeId(targetId: string, selectedNodeId: string | null): string | null {
  if (targetId.startsWith(VRAM_DROP_TARGET_PREFIX)) {
    return targetId.slice(VRAM_DROP_TARGET_PREFIX.length) || null;
  }
  if (targetId.startsWith(CONFIG_NODE_DROP_TARGET_PREFIX)) {
    return targetId.slice(CONFIG_NODE_DROP_TARGET_PREFIX.length) || null;
  }
  if (targetId === 'vram-container') {
    return selectedNodeId;
  }
  return null;
}

type OverlayFitStatus = 'fits' | 'tight' | 'too-large';

function computeOverlayFitStatus(sizeBytes: number, vramBytes: number): OverlayFitStatus {
  if (sizeBytes * 1.1 + GPU_SYSTEM_OVERHEAD_BYTES <= vramBytes) return 'fits';
  if (sizeBytes <= vramBytes) return 'tight';
  return 'too-large';
}

function formatSizeGb(sizeBytes: number): string {
  const gb = sizeBytes / 1e9;
  return `${gb >= 100 ? Math.round(gb) : gb.toFixed(1)} GB`;
}

function OverlayFitBadge({ status }: { status: OverlayFitStatus }) {
  const base = 'flex-none rounded px-1.5 py-0.5 text-[10px] font-medium';
  if (status === 'fits') {
    return (
      <span
        data-testid="overlay-fit-badge"
        className={cn(base, 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300')}
      >
        ✅ Fits
      </span>
    );
  }
  if (status === 'tight') {
    return (
      <span
        data-testid="overlay-fit-badge"
        className={cn(base, 'bg-amber-500/10 text-amber-700 dark:text-amber-300')}
      >
        ⚠️ Tight
      </span>
    );
  }
  return (
    <span
      data-testid="overlay-fit-badge"
      className={cn(base, 'bg-destructive/10 text-destructive')}
    >
      ❌ Too Large
    </span>
  );
}

export type { DragModelData, DragSplitAssignmentData };

export function DndContext({ children, selectedNodeId, selectedNodeVramBytes, onAssignModel, onMoveSplitAssignment }: DndContextProps) {
  const sensors = useMemo(
    () => [
      configure(PointerSensor, {
        preventActivation: (event) => hasInteractiveDragAncestor(event.target),
      }),
      configure(KeyboardSensor, {
        preventActivation: (event) => hasInteractiveDragAncestor(event.target),
      }),
    ],
    [],
  );

  return (
    <DragDropProvider
      sensors={sensors}
      onDragEnd={(event) => {
        if (event.canceled) return;

        const target = event.operation.target;
        const source = event.operation.source;

        if (!target || !source) return;

        const targetNodeId = resolveDropTargetNodeId(String(target.id), selectedNodeId);
        if (!targetNodeId) return;

        const data = source.data;
        if (isDragModelData(data)) {
          onAssignModel?.(data.modelName, data.sizeBytes, targetNodeId);
          return;
        }

        if (isDragSplitAssignmentData(data)) {
          onMoveSplitAssignment?.({
            assignmentId: data.assignmentId,
            sourceNodeId: data.sourceNodeId,
            targetNodeId,
          });
        }
      }}
    >
      {children}
      <DragOverlay>
        {(source) => {
          const data = source.data as unknown;

          let modelName: string;
          let sizeBytes: number | undefined;

          if (isDragModelData(data)) {
            modelName = data.modelName;
            sizeBytes = data.sizeBytes;
          } else if (isDragSplitAssignmentData(data)) {
            modelName = data.modelName;
            sizeBytes = data.sizeBytes;
          } else {
            return null;
          }

          const fitStatus =
            sizeBytes != null && selectedNodeVramBytes && selectedNodeVramBytes > 0
              ? computeOverlayFitStatus(sizeBytes, selectedNodeVramBytes)
              : null;

          return (
            <div
              data-testid="drag-overlay-card"
              className="flex items-center gap-2 rounded-md border border-border/80 bg-card px-3 py-2 text-sm shadow-lg ring-1 ring-primary/20"
            >
              <span className="min-w-0 flex-1 truncate font-medium text-foreground">{modelName}</span>
              {sizeBytes != null && (
                <span className="flex-none text-xs text-muted-foreground">{formatSizeGb(sizeBytes)}</span>
              )}
              {fitStatus != null && <OverlayFitBadge status={fitStatus} />}
            </div>
          );
        }}
      </DragOverlay>
    </DragDropProvider>
  );
}
