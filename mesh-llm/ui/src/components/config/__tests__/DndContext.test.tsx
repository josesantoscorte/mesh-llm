import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { useDraggable } from '@dnd-kit/react';
import { describe, expect, it, vi } from 'vitest';

import { DndContext } from '../DndContext';

if (!HTMLElement.prototype.setPointerCapture) {
  HTMLElement.prototype.setPointerCapture = () => {};
}

if (!HTMLElement.prototype.releasePointerCapture) {
  HTMLElement.prototype.releasePointerCapture = () => {};
}

if (!Document.prototype.elementFromPoint) {
  Document.prototype.elementFromPoint = () => document.body;
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

function setRect(element: HTMLElement, rect: Omit<DOMRect, 'toJSON'> & { toJSON?: () => unknown }) {
  Object.defineProperty(element, 'getBoundingClientRect', {
    configurable: true,
    value: () => ({
      ...rect,
      toJSON: rect.toJSON ?? (() => rect),
    }),
  });
}

function TestModelDraggable({ modelName, sizeBytes }: { modelName: string; sizeBytes: number }) {
  const { ref, handleRef } = useDraggable({
    id: `model:${modelName}`,
    data: { type: 'model', modelName, sizeBytes, nodeIds: [] },
  });

  return (
    <div ref={ref as React.RefCallback<HTMLDivElement>} data-testid="test-draggable">
      <button
        ref={handleRef as React.RefCallback<HTMLButtonElement>}
        type="button"
        data-testid="drag-handle"
      >
        Drag me
      </button>
    </div>
  );
}

describe('DndContext', () => {
  it('renders children within the drag-drop provider', () => {
    render(
      <DndContext selectedNodeId="node-a">
        <div data-testid="child">Hello</div>
      </DndContext>,
    );

    expect(screen.getByTestId('child')).toBeVisible();
    expect(screen.getByText('Hello')).toBeVisible();
  });

  it('renders children even when no node is selected', () => {
    render(
      <DndContext selectedNodeId={null}>
        <div data-testid="child">Content</div>
      </DndContext>,
    );

    expect(screen.getByTestId('child')).toBeVisible();
  });

  it('accepts an onAssignModel callback without error', () => {
    const onAssign = vi.fn();

    render(
      <DndContext selectedNodeId="node-a" onAssignModel={onAssign}>
        <div>Content</div>
      </DndContext>,
    );

    expect(screen.getByText('Content')).toBeVisible();
    expect(onAssign).not.toHaveBeenCalled();
  });

  it('DragOverlay renders with model name text when a drag is active', async () => {
    render(
      <DndContext selectedNodeId="node-a" selectedNodeVramBytes={24_000_000_000}>
        <TestModelDraggable modelName="GLM-4.7-Flash-Q4_K_M" sizeBytes={10_000_000_000} />
      </DndContext>,
    );

    const handle = screen.getByTestId('drag-handle');
    const draggable = screen.getByTestId('test-draggable');

    setRect(draggable as HTMLElement, {
      x: 20, y: 20, top: 20, left: 20, right: 260, bottom: 80, width: 240, height: 60,
    } as DOMRect);
    setRect(handle as HTMLElement, {
      x: 20, y: 20, top: 20, left: 20, right: 260, bottom: 80, width: 240, height: 60,
    } as DOMRect);

    fireEvent.pointerDown(handle, {
      button: 0, buttons: 1, isPrimary: true, pointerId: 1, pointerType: 'mouse', clientX: 120, clientY: 50,
    });

    await act(async () => {
      fireEvent.pointerMove(document, {
        buttons: 1, isPrimary: true, pointerId: 1, pointerType: 'mouse', clientX: 400, clientY: 300,
      });
    });

    await waitFor(() => {
      const overlay = screen.queryByTestId('drag-overlay-card');
      expect(overlay).not.toBeNull();
    });

    const overlay = screen.getByTestId('drag-overlay-card');
    expect(overlay).toHaveTextContent('GLM-4.7-Flash-Q4_K_M');
  });
});
