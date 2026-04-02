import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { useEffect, useState } from 'react';
import { describe, expect, it, vi } from 'vitest';

class MockIntersectionObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return [];
  }
}

vi.stubGlobal('IntersectionObserver', MockIntersectionObserver);

if (!HTMLElement.prototype.setPointerCapture) {
  HTMLElement.prototype.setPointerCapture = () => {};
}

if (!HTMLElement.prototype.releasePointerCapture) {
  HTMLElement.prototype.releasePointerCapture = () => {};
}

if (!Document.prototype.elementFromPoint) {
  Document.prototype.elementFromPoint = () => document.body;
}

if (typeof ShadowRoot !== 'undefined' && !ShadowRoot.prototype.elementFromPoint) {
  ShadowRoot.prototype.elementFromPoint = () => document.body;
}

if (!Document.prototype.getAnimations) {
  Document.prototype.getAnimations = () => [];
}

if (!Element.prototype.getAnimations) {
  Element.prototype.getAnimations = () => [];
}

if (!window.matchMedia) {
  window.matchMedia = () => ({
    matches: false,
    media: '',
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  });
}

vi.mock('animejs', () => ({
  animate: vi.fn(),
  createScope: vi.fn(() => ({
    add: vi.fn(),
    revert: vi.fn(),
  })),
}));

import { DndContext } from '../DndContext';
import { ModelCatalog } from '../ModelCatalog';
import { VramContainer, type VramAssignment } from '../VramContainer';
import { useDragDropManager } from '@dnd-kit/react';

const peers = [
  {
    id: 'node-a',
    hostname: 'alpha.local',
    models: ['GLM-4.7-Flash-Q4_K_M'],
    model_sizes: [['GLM-4.7-Flash-Q4_K_M', 10_000_000_000] as const],
  },
];

function setRect(element: HTMLElement, rect: Omit<DOMRect, 'toJSON'> & { toJSON?: () => unknown }) {
  Object.defineProperty(element, 'getBoundingClientRect', {
    configurable: true,
    value: () => ({
      ...rect,
      toJSON: rect.toJSON ?? (() => rect),
    }),
  });
}

function ManagerCapture({ onReady }: { onReady: (manager: ReturnType<typeof useDragDropManager>) => void }) {
  const manager = useDragDropManager();

  useEffect(() => {
    onReady(manager);
  }, [manager, onReady]);

  return null;
}

function DragHarness({ onManagerReady }: { onManagerReady: (manager: ReturnType<typeof useDragDropManager>) => void }) {
  const [assignments, setAssignments] = useState<VramAssignment[]>([]);

  return (
    <DndContext
      selectedNodeId="node-a"
       onAssignModel={(modelName, sizeBytes) => {
         setAssignments((current) => [
           ...current,
           {
             id: modelName,
             name: modelName,
             sizeBytes,
             fullSizeBytes: sizeBytes,
             weightsBytes: 0,
             contextBytes: 0,
             sizeGb: sizeBytes / 1e9,
           },
         ]);
       }}
    >
      <div>
        <ManagerCapture onReady={onManagerReady} />
        <ModelCatalog
          peers={peers}
          selectedNode={{ id: 'node-a', hostname: 'alpha.local', vramBytes: 24_000_000_000 }}
        />
        <VramContainer
          nodeId="node-a"
          nodeHostname="alpha.local"
          totalVramBytes={24_000_000_000}
          assignments={assignments}
          onRemoveModel={vi.fn()}
          selectedAssignmentId={null}
          onSelectAssignment={vi.fn()}
        />
      </div>
    </DndContext>
  );
}

function GpuDragHarness({
  onManagerReady,
  onAssignCapture,
}: {
  onManagerReady: (manager: ReturnType<typeof useDragDropManager>) => void;
  onAssignCapture: (nodeId: string) => void;
}) {
  const [assignments, setAssignments] = useState<VramAssignment[]>([]);

  return (
    <DndContext
      selectedNodeId="node-a"
       onAssignModel={(modelName, sizeBytes, nodeId) => {
         onAssignCapture(nodeId);
         setAssignments((current) => [
           ...current,
           {
             id: modelName,
             name: modelName,
             sizeBytes,
             fullSizeBytes: sizeBytes,
             weightsBytes: 0,
             contextBytes: 0,
             sizeGb: sizeBytes / 1e9,
           },
         ]);
       }}
    >
      <div>
        <ManagerCapture onReady={onManagerReady} />
        <ModelCatalog
          peers={peers}
          selectedNode={{ id: 'node-a', hostname: 'alpha.local', vramBytes: 24_000_000_000 }}
        />
        <VramContainer
          nodeId="node-a::gpu-1"
          nodeHostname="GPU 1 · RTX 4090 · 24.0 GB"
          totalVramBytes={24_000_000_000}
          assignments={assignments}
          onRemoveModel={vi.fn()}
          selectedAssignmentId={null}
          onSelectAssignment={vi.fn()}
          placementTarget="node-a:gpu-1"
        />
      </div>
    </DndContext>
  );
}

describe('ModelCatalog drag activation', () => {
  it('uses a full-surface drag handle across the visible card body', () => {
    render(<ModelCatalog peers={peers} selectedNode={null} />);

    const handle = screen.getByTestId('model-card-drag-handle');

    expect(handle).toHaveClass('absolute', 'inset-0', 'rounded-lg');
    expect(screen.getByRole('button', { name: 'GLM-4.7-Flash-Q4_K_M' })).toBe(handle);
    expect(handle).toHaveAttribute('aria-label', 'GLM-4.7-Flash-Q4_K_M');
  });

  it('starts dragging from the card body handle and assigns the model when dropped into the VRAM container', async () => {
    let manager: ReturnType<typeof useDragDropManager> | null = null;

    render(
      <DragHarness
        onManagerReady={(value) => {
          manager = value;
        }}
      />,
    );

    const card = screen.getByTestId('model-card');
    const handle = screen.getByTestId('model-card-drag-handle');
    const dropZone = screen.getByTestId('vram-container');

    setRect(card, {
      x: 20,
      y: 20,
      top: 20,
      left: 20,
      right: 260,
      bottom: 180,
      width: 240,
      height: 160,
    } as DOMRect);
    setRect(handle, {
      x: 20,
      y: 20,
      top: 20,
      left: 20,
      right: 260,
      bottom: 180,
      width: 240,
      height: 160,
    } as DOMRect);
    setRect(dropZone, {
      x: 320,
      y: 20,
      top: 20,
      left: 320,
      right: 720,
      bottom: 260,
      width: 400,
      height: 240,
    } as DOMRect);

    fireEvent.pointerDown(handle, {
      button: 0,
      buttons: 1,
      isPrimary: true,
      pointerId: 1,
      pointerType: 'mouse',
      clientX: 120,
      clientY: 100,
    });

    fireEvent.pointerMove(document, {
      buttons: 1,
      isPrimary: true,
      pointerId: 1,
      pointerType: 'mouse',
      clientX: 420,
      clientY: 120,
    });

    await waitFor(() => {
      expect(card).toHaveAttribute('data-dragging', 'true');
    });

    await waitFor(() => {
      expect(manager).not.toBeNull();
    });

    await act(async () => {
      await manager?.actions.setDropTarget('vram-container:node-a');
    });

    fireEvent.pointerUp(document, {
      button: 0,
      isPrimary: true,
      pointerId: 1,
      pointerType: 'mouse',
      clientX: 420,
      clientY: 120,
    });

    await waitFor(() => {
      expect(screen.getAllByTestId('vram-model-block')).toHaveLength(1);
      expect(within(screen.getByTestId('vram-capacity-bar')).getByText('GLM-4.7-Flash-Q4_K_M')).toBeVisible();
    });
  });

  it('resolves the GPU-specific composite nodeId when dropped onto a separate-GPU container with data-placement-target', async () => {
    let manager: ReturnType<typeof useDragDropManager> | null = null;
    let capturedNodeId: string | null = null;

    render(
      <GpuDragHarness
        onManagerReady={(value) => {
          manager = value;
        }}
        onAssignCapture={(nodeId) => {
          capturedNodeId = nodeId;
        }}
      />,
    );

    const container = screen.getByTestId('vram-container');
    expect(container).toHaveAttribute('data-placement-target', 'node-a:gpu-1');

    const card = screen.getByTestId('model-card');
    const handle = screen.getByTestId('model-card-drag-handle');

    setRect(card, {
      x: 20, y: 20, top: 20, left: 20, right: 260, bottom: 180, width: 240, height: 160,
    } as DOMRect);
    setRect(handle, {
      x: 20, y: 20, top: 20, left: 20, right: 260, bottom: 180, width: 240, height: 160,
    } as DOMRect);
    setRect(container, {
      x: 320, y: 20, top: 20, left: 320, right: 720, bottom: 260, width: 400, height: 240,
    } as DOMRect);

    fireEvent.pointerDown(handle, {
      button: 0, buttons: 1, isPrimary: true, pointerId: 1, pointerType: 'mouse', clientX: 120, clientY: 100,
    });

    fireEvent.pointerMove(document, {
      buttons: 1, isPrimary: true, pointerId: 1, pointerType: 'mouse', clientX: 420, clientY: 120,
    });

    await waitFor(() => {
      expect(card).toHaveAttribute('data-dragging', 'true');
    });

    await waitFor(() => {
      expect(manager).not.toBeNull();
    });

    await act(async () => {
      await manager?.actions.setDropTarget('vram-container:node-a::gpu-1');
    });

    fireEvent.pointerUp(document, {
      button: 0, isPrimary: true, pointerId: 1, pointerType: 'mouse', clientX: 420, clientY: 120,
    });

    await waitFor(() => {
      expect(capturedNodeId).toBe('node-a::gpu-1');
    });

    await waitFor(() => {
      expect(screen.getAllByTestId('vram-model-block')).toHaveLength(1);
    });
  });
});
