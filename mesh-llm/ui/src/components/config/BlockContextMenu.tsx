import { useCallback, useEffect, useRef } from 'react';
import { Info } from 'lucide-react';

import { cn } from '../../lib/utils';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip';
import type { VramAssignment } from './VramContainer';

export type ContextMenuPosition = { x: number; y: number };

type BlockContextMenuProps = {
  position: ContextMenuPosition;
  assignment: VramAssignment;
  canSplit: boolean;
  splitReason?: string | null;
  onRecombine?: (() => void) | null;
  onSplit: () => void;
  onRemove: () => void;
  onClose: () => void;
};

export function BlockContextMenu({ position, canSplit, splitReason, onRecombine, onSplit, onRemove, onClose }: BlockContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);
  const splitUnavailableReason = !canSplit ? (splitReason ?? null) : null;

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    },
    [onClose],
  );

  const handlePointerDown = useCallback(
    (e: PointerEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    },
    [onClose],
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('pointerdown', handlePointerDown, true);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('pointerdown', handlePointerDown, true);
    };
  }, [handleKeyDown, handlePointerDown]);

  return (
    <TooltipProvider>
      <div
        ref={menuRef}
        data-testid="block-context-menu"
        data-config-assignment-interactive="true"
        role="menu"
        className="fixed z-50 min-w-[10rem] overflow-hidden rounded-md border border-border bg-popover py-1 text-popover-foreground shadow-md"
        style={{
          left: position.x,
          top: position.y,
          animationName: 'context-menu-in',
          animationDuration: '120ms',
          animationTimingFunction: 'ease-out',
          animationFillMode: 'both',
        }}
      >
        <SplitMenuItem
          disabled={!canSplit}
          splitReason={splitUnavailableReason}
          onClick={() => { onSplit(); onClose(); }}
        />
        {onRecombine ? (
          <MenuItem
            testId="ctx-menu-recombine"
            onClick={() => { onRecombine(); onClose(); }}
            label="Recombine"
          />
        ) : null}
        <MenuItem
          testId="ctx-menu-remove"
          onClick={() => { onRemove(); onClose(); }}
          label="Remove"
        />
      </div>
    </TooltipProvider>
  );
}

function MenuItem({ testId, label, disabled, onClick }: { testId: string; label: string; disabled?: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      role="menuitem"
      data-testid={testId}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        'flex w-full items-center px-3 py-1.5 text-left text-xs transition-colors',
        disabled
          ? 'cursor-not-allowed text-muted-foreground/40'
          : 'cursor-pointer text-foreground hover:bg-accent hover:text-accent-foreground',
      )}
    >
      {label}
    </button>
  );
}

function SplitMenuItem({ disabled, splitReason, onClick }: { disabled: boolean; splitReason: string | null; onClick: () => void }) {
  return (
    <div className="flex items-center gap-1 px-1">
      <button
        type="button"
        role="menuitem"
        data-testid="ctx-menu-split"
        disabled={disabled}
        onClick={onClick}
        className={cn(
          'flex min-w-0 flex-1 items-center rounded-sm px-2 py-1.5 text-left text-xs transition-colors',
          disabled
            ? 'cursor-not-allowed text-muted-foreground/40'
            : 'cursor-pointer text-foreground hover:bg-accent hover:text-accent-foreground',
        )}
      >
        Split
      </button>
      {splitReason ? (
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              aria-label="Why Split is unavailable"
              className="inline-flex h-6 w-6 flex-none items-center justify-center rounded-sm text-muted-foreground/70 transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              <Info className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          </TooltipTrigger>
          <TooltipContent side="right" className="max-w-64 text-pretty leading-relaxed">
            {splitReason}
          </TooltipContent>
        </Tooltip>
      ) : null}
    </div>
  );
}
