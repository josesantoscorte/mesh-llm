import { HardDrive, Layers, MousePointerClick, RotateCcw, TriangleAlert } from 'lucide-react';

import { Button } from '../ui/button';
import { Alert, AlertDescription, AlertTitle } from '../ui/alert';

export function EmptyNoNodes() {
  return (
    <div
      data-testid="empty-no-nodes"
      className="flex min-h-[18rem] flex-col items-center justify-center rounded-md border border-dashed bg-muted/20 px-6 py-8 text-center"
    >
      <Layers className="mb-3 h-8 w-8 text-muted-foreground/60 transition-transform duration-200" aria-hidden="true" />
      <div className="text-sm font-medium">This console does not have a trusted owner fingerprint yet</div>
      <p className="mt-1.5 max-w-sm text-xs text-muted-foreground">
        Start mesh-llm with your owner key on this node, then rejoin peers so matching verified owner fingerprints appear here for configuration.
      </p>
      <pre className="mt-4 rounded-md border bg-muted/40 px-4 py-2.5 text-left text-[11px] text-muted-foreground">
        mesh-llm --owner-key ~/.mesh-llm/owner-key --model Qwen2.5-32B
      </pre>
    </div>
  );
}

export function EmptyNoModels({ onRefresh }: { onRefresh?: () => void }) {
  return (
    <div
      data-testid="empty-no-models"
      className="flex min-h-[18rem] flex-col items-center justify-center rounded-md border border-dashed bg-muted/20 px-6 py-8 text-center"
    >
      <HardDrive className="mb-3 h-8 w-8 text-muted-foreground/60 transition-transform duration-200" aria-hidden="true" />
      <div className="text-sm font-medium">No models found</div>
      <p className="mt-1.5 max-w-sm text-xs text-muted-foreground">
        Add GGUF files to <code className="rounded bg-muted px-1.5 py-0.5 text-[11px]">~/.models/</code> and click Refresh.
      </p>
      {onRefresh ? (
        <Button variant="outline" size="sm" className="mt-4 gap-1.5" onClick={onRefresh}>
          <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
          Refresh
        </Button>
      ) : null}
    </div>
  );
}

export function EmptyNoSelection() {
  return (
    <div
      data-testid="empty-no-selection"
      className="flex min-h-[18rem] flex-col items-center justify-center rounded-md border border-dashed bg-muted/20 px-6 py-8 text-center"
    >
      <MousePointerClick className="mb-3 h-8 w-8 text-muted-foreground/60 transition-transform duration-200" aria-hidden="true" />
      <div className="text-sm font-medium">Select a node to configure</div>
      <p className="mt-1.5 max-w-xs text-xs text-muted-foreground">
        Pick a node from the list on the left to manage its model assignments and VRAM allocation.
      </p>
    </div>
  );
}

export function ConfigLoadingError({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <Alert variant="destructive" data-testid="config-loading-error">
      <TriangleAlert className="h-4 w-4" />
      <AlertTitle>Failed to load configuration</AlertTitle>
      <AlertDescription className="space-y-2">
        <p>{error}</p>
        {onRetry ? (
          <Button variant="outline" size="sm" className="mt-1 gap-1.5" onClick={onRetry}>
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            Retry
          </Button>
        ) : null}
      </AlertDescription>
    </Alert>
  );
}

export function NodeListSkeleton() {
  return (
    <div data-testid="node-list-skeleton" className="space-y-2">
      {['skeleton-node-1', 'skeleton-node-2', 'skeleton-node-3'].map((key) => (
        <div
          key={key}
          className="animate-pulse rounded-lg border border-border/70 bg-muted/20 p-3"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1 space-y-2">
              <div className="h-4 w-32 rounded bg-muted/60" />
              <div className="flex gap-2">
                <div className="h-2 w-2 rounded-full bg-muted/60" />
                <div className="h-3 w-16 rounded bg-muted/40" />
                <div className="h-3 w-12 rounded bg-muted/40" />
              </div>
            </div>
            <div className="h-3.5 w-14 rounded bg-muted/40" />
          </div>
          <div className="mt-3 space-y-2">
            <div className="h-3 w-20 rounded bg-muted/40" />
            <div className="flex gap-1">
              <div className="h-5 w-28 rounded bg-muted/30" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export function CatalogSkeleton() {
  return (
    <div data-testid="catalog-skeleton" className="grid gap-3 lg:grid-cols-2">
      {['skeleton-cat-1', 'skeleton-cat-2', 'skeleton-cat-3', 'skeleton-cat-4', 'skeleton-cat-5', 'skeleton-cat-6'].map((key) => (
        <div
          key={key}
          className="animate-pulse rounded-lg border border-border/70 bg-muted/20 p-4"
        >
          <div className="space-y-2">
            <div className="h-4 w-40 rounded bg-muted/60" />
            <div className="flex gap-1.5">
              <div className="h-5 w-14 rounded bg-muted/40" />
              <div className="h-5 w-16 rounded bg-muted/30" />
            </div>
          </div>
          <div className="mt-4 space-y-2">
            <div className="h-3 w-20 rounded bg-muted/40" />
            <div className="flex gap-1">
              <div className="h-5 w-20 rounded bg-muted/30" />
              <div className="h-5 w-16 rounded bg-muted/30" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
