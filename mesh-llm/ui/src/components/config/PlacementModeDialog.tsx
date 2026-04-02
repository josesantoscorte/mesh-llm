import * as DialogPrimitive from "@radix-ui/react-dialog";

import { Button } from "../ui/button";

export type PlacementModeDialogProps = {
  open: boolean;
  pendingNodeId: string | undefined;
  onConfirm: () => void;
  onCancel: () => void;
};

export function PlacementModeDialog({
  open,
  pendingNodeId,
  onConfirm,
  onCancel,
}: PlacementModeDialogProps) {
  return (
    <DialogPrimitive.Root
      open={open}
      onOpenChange={(isOpen) => {
        if (!isOpen) onCancel();
      }}
    >
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/50 transition-opacity duration-200" />
        <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-xl border bg-card p-6 shadow-lg transition-all duration-200">
          <DialogPrimitive.Title className="mb-2 text-base font-semibold">
            Pool GPUs?
          </DialogPrimitive.Title>
          <DialogPrimitive.Description asChild>
            <div className="mb-4 space-y-3 text-sm text-muted-foreground">
              <p>
                Pooling GPUs splits inference across slower communication
                channels, which results in slower token generation.
              </p>
              <p>
                Use this when a model doesn't fit on a single device.
              </p>
            </div>
          </DialogPrimitive.Description>
          <div className="flex justify-end gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={onCancel}
            >
              Cancel
            </Button>
            <Button
              size="sm"
              data-testid={`node-${pendingNodeId}-mode-confirm`}
              onClick={onConfirm}
            >
              Confirm
            </Button>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
