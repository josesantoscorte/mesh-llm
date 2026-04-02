import { useDragOperation, useDroppable } from '@dnd-kit/react';

import type { OwnedNode } from '../../hooks/useOwnedNodes';
import { cn } from '../../lib/utils';
import { Badge } from '../ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { ScrollArea } from '../ui/scroll-area';
import { CONFIG_NODE_DROP_TARGET_PREFIX } from './DndContext';

function formatVram(vramGb: number) {
  return `${new Intl.NumberFormat('en-US', {
    minimumFractionDigits: vramGb >= 100 ? 0 : 1,
    maximumFractionDigits: vramGb >= 100 ? 0 : 1,
  }).format(vramGb)} GB`;
}

function statusDotClass(statusTone: OwnedNode['statusTone']) {
  if (statusTone === 'serving') return 'bg-emerald-500';
  if (statusTone === 'host') return 'bg-indigo-500';
  if (statusTone === 'worker') return 'bg-sky-500';
  if (statusTone === 'client') return 'bg-zinc-400';
  return 'bg-zinc-500';
}

type NodeListProps = {
  nodes: OwnedNode[];
  selectedNodeId: string | null;
  onSelectNode: (node: OwnedNode) => void;
  className?: string;
  listClassName?: string;
};

export function NodeList({
  nodes,
  selectedNodeId,
  onSelectNode,
  className,
  listClassName = 'h-[clamp(18rem,42vh,36rem)]',
}: NodeListProps) {
  return (
    <Card className={cn('flex min-h-[22rem] flex-col overflow-hidden', className)}>
      <CardHeader className="border-b border-border">
        <div className="flex items-center justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className="text-lg font-semibold">Owned nodes</CardTitle>
            <p className="text-sm text-muted-foreground">
              Select a node that shares this console&apos;s verified owner fingerprint.
            </p>
          </div>
          <Badge>{nodes.length}</Badge>
        </div>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 overflow-hidden p-0">
        {nodes.length === 0 ? (
          <div className="m-4 flex min-h-[18rem] flex-col items-center justify-center rounded-md border border-dashed bg-muted/20 px-6 text-center">
            <div className="text-sm font-medium">No configurable nodes found</div>
            <p className="mt-1 max-w-xs text-xs text-muted-foreground">
              Start each node with the same owner key so matching verified owner fingerprints appear here for configuration.
            </p>
          </div>
        ) : (
          <ScrollArea className={cn('min-h-0 flex-1', listClassName)}>
            <div className="space-y-3 p-4">
              {nodes.map((node) => (
                <DroppableNodeCard
                  key={node.id}
                  node={node}
                  isSelected={selectedNodeId === node.id}
                  onSelectNode={onSelectNode}
                />
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}

function DroppableNodeCard({
  node,
  isSelected,
  onSelectNode,
}: {
  node: OwnedNode;
  isSelected: boolean;
  onSelectNode: (node: OwnedNode) => void;
}) {
  const { ref, isDropTarget } = useDroppable({ id: `${CONFIG_NODE_DROP_TARGET_PREFIX}${node.id}` });
  const { source } = useDragOperation();
  const isSplitDrag = (source?.data as { type?: string } | undefined)?.type === 'split-assignment';

  return (
    <div ref={ref}>
      <button
        type="button"
        onClick={() => onSelectNode(node)}
        aria-pressed={isSelected}
        data-testid={`config-node-card-${node.id}`}
        className={cn(
          'w-full rounded-lg border border-border/70 bg-background p-3 text-left shadow-soft transition-all duration-200 ease-out hover:-translate-y-0.5 hover:bg-muted/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
          isSelected && 'border-primary/40 bg-primary/5 ring-2 ring-primary',
          isSplitDrag && isDropTarget && 'border-primary/60 bg-primary/10 ring-2 ring-primary/50',
        )}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 space-y-1">
            <div className="flex items-center gap-2">
              <span className="truncate text-sm font-medium leading-5">{node.hostname}</span>
              {node.isSelf ? <Badge className="px-2 py-0.5 text-[11px]">This node</Badge> : null}
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className={cn('h-2 w-2 rounded-full', statusDotClass(node.statusTone))} aria-hidden="true" />
              <span>{node.statusLabel}</span>
              <span aria-hidden="true">·</span>
              <span>{node.role}</span>
            </div>
          </div>
          <div className="text-right text-xs text-muted-foreground">{formatVram(node.vramGb)}</div>
        </div>

        <div className="mt-3 grid gap-2 text-xs text-muted-foreground">
          <div>
            <div className="flex items-center gap-2">
              <span className="font-medium text-foreground">{node.hardwareLabel}</span>
              <span className="truncate">{node.gpuName}</span>
              {(node.gpuTargets?.length ?? 0) > 1 && (
                <Badge className="shrink-0 px-1.5 py-0 text-[10px]">
                  {node.gpuTargets.length} GPUs
                </Badge>
              )}
            </div>
            {(node.separateCapable || node.mixedGpuWarning) && (
              <div className="mt-1 flex flex-wrap gap-1">
                {node.separateCapable && (
                  <Badge className="border-sky-500/30 bg-sky-500/10 px-1.5 py-0 text-[10px] text-sky-600 dark:text-sky-400">
                    Separate mode available
                  </Badge>
                )}
                {node.mixedGpuWarning && (
                  <Badge className="border-amber-500/30 bg-amber-500/10 px-1.5 py-0 text-[10px] text-amber-600 dark:text-amber-400">
                    Mixed GPUs
                  </Badge>
                )}
              </div>
            )}
          </div>
          <div>
            <span className="font-medium text-foreground">Models</span>
            <div className="mt-1 flex flex-wrap gap-1">
              {node.models.length > 0 ? (
                node.models.map((model) => <Badge key={model}>{model}</Badge>)
              ) : (
                <Badge>No models advertised</Badge>
              )}
            </div>
          </div>
        </div>
      </button>
    </div>
  );
}
