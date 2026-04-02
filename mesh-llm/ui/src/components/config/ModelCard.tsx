import { useEffect, useRef, type PointerEvent as ReactPointerEvent } from 'react';
import { animate } from 'animejs';

import { cn } from '../../lib/utils';
import { DRAG_INTERACTIVE_ATTRIBUTE } from './DndContext';

type FitStatus = 'fits' | 'overcommit' | 'unknown';

type ModelCardProps = {
  name: string;
  widthPercent: number;
  fitStatus: FitStatus;
  isSelected?: boolean;
  onSelect?: () => void;
  onRemove: () => void;
  errorMessage?: string;
};

function stopDragPointerDown(event: ReactPointerEvent<HTMLElement>) {
  event.stopPropagation();
}

function FitIcon({ status }: { status: FitStatus }) {
  if (status === 'fits') {
    return (
      <span
        data-testid="fit-icon-ok"
        className="inline-block h-2 w-2 rounded-full bg-emerald-500"
        title="Fits in VRAM"
      />
    );
  }
  if (status === 'overcommit') {
    return (
      <span
        data-testid="fit-icon-warn"
        className="inline-block h-2 w-2 rounded-full bg-destructive"
        title="Exceeds VRAM"
      />
    );
  }
  return (
    <span
      data-testid="fit-icon-unknown"
      className="inline-block h-2 w-2 rounded-full bg-muted-foreground/40"
      title="Unknown fit status"
    />
  );
}

export function ModelCard({ name, widthPercent, fitStatus, isSelected, onSelect, onRemove, errorMessage }: ModelCardProps) {
  const cardRef = useRef<HTMLButtonElement>(null);
  const hasAnimated = useRef(false);

  useEffect(() => {
    const el = cardRef.current;
    if (!el || hasAnimated.current) return;
    hasAnimated.current = true;

    animate(el, {
      scale: [0.85, 1],
      opacity: [0, 1],
      duration: 280,
      ease: 'outQuart',
    });
  }, []);

  return (
    <button
      ref={cardRef}
      type="button"
      data-testid="vram-model-card"
      onClick={onSelect}
      className={cn(
        'relative flex min-w-0 items-center gap-2 rounded-md border px-2.5 py-2 text-xs shadow-sm cursor-pointer text-left',
        'transition-all duration-150 hover:shadow-soft',
        isSelected
          ? 'border-primary/60 bg-primary/5 ring-1 ring-primary/30'
          : 'border-border bg-card',
      )}
      style={{ width: `${Math.max(widthPercent, 12)}%`, minWidth: '5rem' }}
    >
      <FitIcon status={fitStatus} />
      <span className="min-w-0 flex-1 truncate text-sm font-medium leading-5 text-foreground" title={name}>
        {name}
      </span>
      {errorMessage ? (
        <span
          data-testid="model-card-error-badge"
          className="flex-none rounded bg-destructive/15 px-1.5 py-0.5 text-[10px] font-medium text-destructive"
          title={errorMessage}
        >
          !
        </span>
      ) : null}
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onRemove();
        }}
        onPointerDown={stopDragPointerDown}
        aria-label={`Remove ${name}`}
        {...{ [DRAG_INTERACTIVE_ATTRIBUTE]: '' }}
        className="flex-none rounded px-0.5 text-muted-foreground transition-colors hover:bg-destructive/15 hover:text-destructive"
      >
        &times;
      </button>
    </button>
  );
}
