import { DragDropProvider } from '@dnd-kit/react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ScannedModelMetadata } from '../../../types/config';
import type { VramAssignment } from '../VramContainer';
import { VramContainer } from '../VramContainer';

const dragOperationState: {
  source:
    | {
        id: string;
        data: {
          type: string;
          sizeBytes?: number;
          assignmentId?: string;
          sourceNodeId?: string;
          modelName?: string;
        };
      }
    | null;
} = {
  source: null,
};

const droppableState = { isDropTarget: false };

vi.mock('@dnd-kit/react', async () => {
  const actual = await vi.importActual<typeof import('@dnd-kit/react')>('@dnd-kit/react');
  return {
    ...actual,
    useDragOperation: vi.fn(() => dragOperationState),
    useDroppable: vi.fn(() => ({ ref: vi.fn(), isDropTarget: droppableState.isDropTarget })),
  };
});

function renderWithDnd(ui: React.ReactElement) {
  return render(<DragDropProvider>{ui}</DragDropProvider>);
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

function makeAssignment(overrides: Partial<VramAssignment> = {}): VramAssignment {
  return {
    id: 'Qwen3-30B-A3B-Q4_K_M',
    name: 'Qwen3-30B-A3B-Q4_K_M',
    sizeBytes: 22_000_000_000,
    fullSizeBytes: 22_000_000_000,
    weightsBytes: 20_000_000_000,
    contextBytes: 2_000_000_000,
    sizeGb: 22,
    model_key: 'abc123',
    ...overrides,
  };
}

const assignments: VramAssignment[] = [makeAssignment()];

const scansLookup = new Map<string, ScannedModelMetadata>([
  ['Qwen3-30B-A3B-Q4_K_M', { total_offloadable_layers: 48, total_layers: 47, architecture: 'qwen3moe' }],
]);

describe('VramContainer', () => {
  beforeEach(() => {
    droppableState.isDropTarget = false;
    dragOperationState.source = null;
  });

  it('uses the same target highlight treatment for split drags as catalog model drags', () => {
    droppableState.isDropTarget = true;
    dragOperationState.source = {
      id: 'split-block:left',
      data: {
        type: 'split-assignment',
        assignmentId: 'left',
        sourceNodeId: 'node-b',
        modelName: 'Qwen3-30B-A3B-Q4_K_M',
        sizeBytes: 11_000_000_000,
      },
    };

    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={assignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
      />,
    );

    expect(screen.getByTestId('vram-container').className).toContain('border-emerald-500/50');
    expect(screen.getByTestId('vram-drag-preview')).toHaveTextContent('Drop to move split');

    droppableState.isDropTarget = false;
    dragOperationState.source = null;
  });

  it('shows total, used, and free VRAM labels', () => {
    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={assignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
      />,
    );

    expect(screen.getByTestId('vram-container')).toHaveTextContent(/VRAM\s+alpha\.local/);
    expect(screen.getByText('48.0 GB')).toBeVisible();
    expect(screen.getByText('26.0 GB')).toBeVisible();
    expect(screen.getAllByText('22.0 GB').length).toBeGreaterThanOrEqual(1);
  });

  it('switches the free VRAM label to MB when less than 2 GB remains', () => {
    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={23_500_000_000}
        assignments={assignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
      />,
    );

    expect(screen.getByText('1500 MB')).toBeVisible();
  });

  it('renders assigned models as proportional blocks in a capacity bar', () => {
    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={assignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
      />,
    );

    expect(screen.getByTestId('vram-capacity-bar')).toBeVisible();
    const blocks = screen.getAllByTestId('vram-model-block');
    expect(blocks).toHaveLength(1);
    expect(within(blocks[0]).getByText('Qwen3-30B-A3B-Q4_K_M')).toBeDefined();
  });

  it('shows a free space block when capacity is not fully committed', () => {
    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={assignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
      />,
    );

    expect(screen.getByTestId('vram-free-block')).toBeVisible();
  });

  it('shows empty drop zone when no models are assigned', () => {
    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={24_000_000_000}
        assignments={[]}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
      />,
    );

    expect(screen.getByTestId('vram-empty')).toBeVisible();
    expect(screen.getByText('Drag models from the catalog to assign them')).toBeVisible();
  });

  it('shows overcommit warning when assigned exceeds total VRAM', () => {
    const overcommittedAssignments: VramAssignment[] = [
      makeAssignment({ id: 'llama', name: 'Llama-3.3-70B-Q4_K_M', sizeBytes: 43_000_000_000, fullSizeBytes: 43_000_000_000, sizeGb: 43 }),
      makeAssignment(),
    ];

    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="beta.local"
        totalVramBytes={24_000_000_000}
        assignments={overcommittedAssignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
      />,
    );

    expect(screen.getByTestId('vram-overcommit-warning')).toBeVisible();
    expect(screen.getByText(/Overcommitted by/)).toBeVisible();
  });

  it('shows an assignment-level invalid signal when a model exceeds available VRAM', () => {
    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="beta.local"
        totalVramBytes={24_000_000_000}
        assignments={[
          makeAssignment({
            sizeBytes: 26_000_000_000,
            fullSizeBytes: 26_000_000_000,
            sizeGb: 26,
            invalidMessage: 'Exceeds available VRAM by 2.0 GB',
          }),
        ]}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
      />,
    );

    expect(screen.getByTestId('model-block-invalid-badge')).toHaveTextContent('invalid');
    expect(screen.getByTestId('model-block-invalid-message')).toHaveTextContent('Exceeds available VRAM by 2.0 GB');
  });

  it('renders assignment warning badges without requiring an external tooltip provider', () => {
    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={[
          makeAssignment({
            id: 'Qwen3-split-warning',
            errorMessage: 'This split is no longer advertised by the source node.',
            split: { start: 0, end: 23, total: 48 },
          }),
        ]}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
      />,
    );

    expect(screen.getByTestId('model-block-error')).toBeVisible();
  });

  it('calls onSelectAssignment when a model block is clicked', async () => {
    const user = userEvent.setup();
    const onSelectAssignment = vi.fn();

    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={assignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={onSelectAssignment}
      />,
    );

    await user.click(screen.getByTestId('vram-model-block'));
    expect(onSelectAssignment).toHaveBeenCalledWith('Qwen3-30B-A3B-Q4_K_M');
  });

  it('highlights the selected model block', () => {
    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={assignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId="Qwen3-30B-A3B-Q4_K_M"
        onSelectAssignment={vi.fn()}
      />,
    );

    const block = screen.getByTestId('vram-model-block');
    expect(block.className).toContain('ring-1');
    expect(block.getAttribute('style')).toContain('oklch');
    expect(block.getAttribute('style')).toContain('box-shadow');
  });

  it('shows weights and context separately on the block', () => {
    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={assignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
      />,
    );

    expect(screen.getByText('20.0 GB + 2.0 GB ctx')).toBeVisible();
  });

  it('scales weights and context labels for split blocks', () => {
    const splitAssignments: VramAssignment[] = [
      makeAssignment({ id: 'left', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, split: { start: 0, end: 23, total: 48 } }),
      makeAssignment({ id: 'right', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, split: { start: 24, end: 47, total: 48 } }),
    ];

    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={splitAssignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
      />,
    );

    expect(screen.getAllByText('10.0 GB + 1.0 GB ctx')).toHaveLength(2);
  });

  it('highlights every related split block in a selected group', () => {
    const splitAssignments: VramAssignment[] = [
      makeAssignment({ id: 'left', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 0, end: 23, total: 48 } }),
      makeAssignment({ id: 'right', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 24, end: 47, total: 48 } }),
    ];

    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={splitAssignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        selectedAssignmentIds={['left', 'right']}
        onSelectAssignment={vi.fn()}
      />,
    );

    for (const block of screen.getAllByTestId('vram-model-block')) {
      expect(block.className).toContain('ring-1');
    }
  });

  it('shows split label on split blocks', () => {
    const splitAssignments: VramAssignment[] = [
      makeAssignment({ id: 'left', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 0, end: 23, total: 48 } }),
      makeAssignment({ id: 'right', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 24, end: 47, total: 48 } }),
    ];

    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={splitAssignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
      />,
    );

    const labels = screen.getAllByTestId('model-block-split-label');
    expect(labels).toHaveLength(2);
    expect(labels[0].textContent).toContain('L0');
    expect(labels[1].textContent).toContain('L24');
  });

  it('renders a resize handle between contiguous split siblings', () => {
    const splitAssignments: VramAssignment[] = [
      makeAssignment({ id: 'left', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 0, end: 23, total: 48 } }),
      makeAssignment({ id: 'right', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 24, end: 47, total: 48 } }),
    ];

    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={splitAssignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
        onResizeSplitBoundary={vi.fn()}
      />,
    );

    const handle = screen.getByTestId('split-resize-handle');
    expect(handle).toBeVisible();
    const track = screen.getByTestId('split-group-block');
    expect(track.children).toHaveLength(3);
    expect(track.children[1]).toBe(handle.parentElement);
  });

  it('commits a changed split boundary through a real mouse drag sequence on the resize handle', async () => {
    const onResizeSplitBoundary = vi.fn();
    const splitAssignments: VramAssignment[] = [
      makeAssignment({ id: 'left', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 0, end: 23, total: 48 } }),
      makeAssignment({ id: 'right', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 24, end: 47, total: 48 } }),
    ];

    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={splitAssignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
        onResizeSplitBoundary={onResizeSplitBoundary}
      />,
    );

    const handle = screen.getByTestId('split-resize-handle');
    const group = screen.getByTestId('split-group-block');
    Object.defineProperty(handle, 'setPointerCapture', { configurable: true, value: vi.fn() });
    Object.defineProperty(handle, 'releasePointerCapture', { configurable: true, value: vi.fn() });
    Object.defineProperty(handle, 'hasPointerCapture', { configurable: true, value: vi.fn(() => true) });

    setRect(group, {
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: 240,
      bottom: 48,
      width: 240,
      height: 48,
    } as DOMRect);

    fireEvent.pointerDown(handle, { clientX: 120, pointerId: 1, button: 0 });
    fireEvent.pointerMove(window, { clientX: 150, pointerId: 1, buttons: 1 });
    fireEvent.pointerUp(window, { clientX: 150, pointerId: 1, button: 0 });

    await waitFor(() => {
      expect(onResizeSplitBoundary).toHaveBeenCalledWith('left', 'right', 30);
    });
  });

  it('does not allow resizing the first split boundary down to a 1-layer leading shard', async () => {
    const onResizeSplitBoundary = vi.fn();
    const splitAssignments: VramAssignment[] = [
      makeAssignment({ id: 'left', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 0, end: 23, total: 48 } }),
      makeAssignment({ id: 'right', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 24, end: 47, total: 48 } }),
    ];

    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={splitAssignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
        onResizeSplitBoundary={onResizeSplitBoundary}
      />,
    );

    const handle = screen.getByTestId('split-resize-handle');
    const group = screen.getByTestId('split-group-block');
    Object.defineProperty(handle, 'setPointerCapture', { configurable: true, value: vi.fn() });
    Object.defineProperty(handle, 'releasePointerCapture', { configurable: true, value: vi.fn() });
    Object.defineProperty(handle, 'hasPointerCapture', { configurable: true, value: vi.fn(() => true) });

    setRect(group, {
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: 240,
      bottom: 48,
      width: 240,
      height: 48,
    } as DOMRect);

    fireEvent.pointerDown(handle, { clientX: 120, pointerId: 1, button: 0 });
    fireEvent.pointerMove(window, { clientX: 0, pointerId: 1, buttons: 1 });
    fireEvent.pointerUp(window, { clientX: 0, pointerId: 1, button: 0 });

    await waitFor(() => {
      expect(onResizeSplitBoundary).toHaveBeenCalledWith('left', 'right', 2);
    });
  });

  it('disables resize handle and shows tooltip for cross-node splits', async () => {
    const splitAssignments: VramAssignment[] = [
      makeAssignment({ id: 'left', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 0, end: 23, total: 48 } }),
      makeAssignment({ id: 'right', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 24, end: 47, total: 48 } }),
    ];

    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={splitAssignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
        onResizeSplitBoundary={vi.fn()}
        crossNodeSplitGroupIds={new Set(['Qwen3-30B-A3B-Q4_K_M::abc123::48'])}
      />,
    );

    const handle = screen.getByTestId('split-resize-handle');
    expect(handle).toBeDisabled();
    expect(handle).toHaveAttribute('title', 'Cross-node splits cannot be resized from this view. To adjust, recombine and re-split.');
  });

  describe('context menu', () => {
    it('opens on right-click and shows split/remove/recombine actions without focus details', () => {
      const splitAssignments: VramAssignment[] = [
        makeAssignment({ id: 'left', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 0, end: 23, total: 48 } }),
        makeAssignment({ id: 'right', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 24, end: 47, total: 48 } }),
      ];

      renderWithDnd(
        <VramContainer
          nodeId="node-a"
          nodeHostname="alpha.local"
          totalVramBytes={48_000_000_000}
          assignments={splitAssignments}
          onRemoveModel={vi.fn()}
          selectedAssignmentId={null}
          onSelectAssignment={vi.fn()}
          onSplitModel={vi.fn()}
          modelScansLookup={scansLookup}
          onRecombineGroup={vi.fn()}
        />,
      );

      fireEvent.contextMenu(screen.getAllByTestId('vram-model-block')[0]!);

      expect(screen.getByTestId('block-context-menu')).toBeVisible();
      expect(screen.getByTestId('ctx-menu-recombine')).toBeVisible();
      expect(screen.getByTestId('ctx-menu-remove')).toBeVisible();
      expect(screen.queryByTestId('ctx-menu-focus')).not.toBeInTheDocument();
    });

    it('Recombine calls onRecombineGroup for split assignments', async () => {
      const user = userEvent.setup();
      const onRecombineGroup = vi.fn();
      const splitAssignments: VramAssignment[] = [
        makeAssignment({ id: 'left', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 0, end: 23, total: 48 } }),
        makeAssignment({ id: 'right', sizeBytes: 11_000_000_000, fullSizeBytes: 22_000_000_000, sizeGb: 11, split: { start: 24, end: 47, total: 48 } }),
      ];

      renderWithDnd(
        <VramContainer
          nodeId="node-a"
          nodeHostname="alpha.local"
          totalVramBytes={48_000_000_000}
          assignments={splitAssignments}
          onRemoveModel={vi.fn()}
          selectedAssignmentId={null}
          onSelectAssignment={vi.fn()}
          onSplitModel={vi.fn()}
          modelScansLookup={scansLookup}
          onRecombineGroup={onRecombineGroup}
        />,
      );

      fireEvent.contextMenu(screen.getAllByTestId('vram-model-block')[0]!);
      await user.click(screen.getByTestId('ctx-menu-recombine'));

      expect(onRecombineGroup).toHaveBeenCalledWith('Qwen3-30B-A3B-Q4_K_M::abc123::48');
    });
  
    it('Remove calls onRemoveModel', async () => {
      const user = userEvent.setup();
      const onRemove = vi.fn();

      renderWithDnd(
        <VramContainer
          nodeId="node-a"
          nodeHostname="alpha.local"
          totalVramBytes={48_000_000_000}
          assignments={assignments}
          onRemoveModel={onRemove}
          selectedAssignmentId={null}
          onSelectAssignment={vi.fn()}
          onSplitModel={vi.fn()}
          modelScansLookup={scansLookup}
        />,
      );

      fireEvent.contextMenu(screen.getByTestId('vram-model-block'));
      await user.click(screen.getByTestId('ctx-menu-remove'));
      expect(onRemove).toHaveBeenCalledWith('Qwen3-30B-A3B-Q4_K_M', 'Qwen3-30B-A3B-Q4_K_M');
    });

    it('Split calls onSplitModel with 50/50 layer ranges', async () => {
      const user = userEvent.setup();
      const onSplit = vi.fn();

      renderWithDnd(
        <VramContainer
          nodeId="node-a"
          nodeHostname="alpha.local"
          totalVramBytes={48_000_000_000}
          assignments={assignments}
          onRemoveModel={vi.fn()}
          selectedAssignmentId={null}
          onSelectAssignment={vi.fn()}
          onSplitModel={onSplit}
          modelScansLookup={scansLookup}
        />,
      );

      fireEvent.contextMenu(screen.getByTestId('vram-model-block'));
      expect(screen.queryByRole('button', { name: /why split is unavailable/i })).not.toBeInTheDocument();
      await user.click(screen.getByTestId('ctx-menu-split'));

      expect(onSplit).toHaveBeenCalledWith(
        'Qwen3-30B-A3B-Q4_K_M',
        { model_key: 'abc123', split: { start: 0, end: 23, total: 48 } },
        { model_key: 'abc123', split: { start: 24, end: 47, total: 48 } },
      );
    });

    it('uses the node model-key lookup when a full assignment was created without model_key', async () => {
      const user = userEvent.setup();
      const onSplit = vi.fn();
      const assignmentWithoutKey = makeAssignment({ model_key: null });

      renderWithDnd(
        <VramContainer
          nodeId="node-a"
          nodeHostname="alpha.local"
          totalVramBytes={48_000_000_000}
          assignments={[assignmentWithoutKey]}
          onRemoveModel={vi.fn()}
          selectedAssignmentId={null}
          onSelectAssignment={vi.fn()}
          onSplitModel={onSplit}
          modelScansLookup={scansLookup}
          modelKeyLookup={new Map([['Qwen3-30B-A3B-Q4_K_M', 'abc123']])}
        />,
      );

      fireEvent.contextMenu(screen.getByTestId('vram-model-block'));
      await user.click(screen.getByTestId('ctx-menu-split'));

      expect(onSplit).toHaveBeenCalledWith(
        'Qwen3-30B-A3B-Q4_K_M',
        { model_key: 'abc123', split: { start: 0, end: 23, total: 48 } },
        { model_key: 'abc123', split: { start: 24, end: 47, total: 48 } },
      );
    });

    it('shows the split-unavailable reason in a tooltip instead of inline copy when scan metadata is missing', async () => {
      const user = userEvent.setup();

      renderWithDnd(
        <VramContainer
          nodeId="node-a"
          nodeHostname="alpha.local"
          totalVramBytes={48_000_000_000}
          assignments={[makeAssignment({ model_key: null })]}
          onRemoveModel={vi.fn()}
          selectedAssignmentId={null}
          onSelectAssignment={vi.fn()}
          onSplitModel={vi.fn()}
          modelScansLookup={scansLookup}
        />,
      );

      fireEvent.contextMenu(screen.getByTestId('vram-model-block'));

      expect(screen.getByTestId('ctx-menu-split')).toBeDisabled();
      expect(screen.queryByTestId('ctx-menu-split-reason')).not.toBeInTheDocument();

      const infoTrigger = screen.getByRole('button', { name: /why split is unavailable/i });
      expect(infoTrigger).toBeVisible();

      await user.hover(infoTrigger);
      expect(await screen.findByRole('tooltip')).toHaveTextContent('Split unavailable: matching scan metadata is missing for this node.');
    });

    it('renders data-placement-target when placementTarget prop is set', () => {
    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={assignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
        placementTarget="node-a:pooled"
      />,
    );

    expect(screen.getByTestId('vram-container')).toHaveAttribute('data-placement-target', 'node-a:pooled');
  });

  it('renders data-placement-target with gpu target for separate mode', () => {
    renderWithDnd(
      <VramContainer
        nodeId="node-a::gpu-1"
        nodeHostname="GPU 1 · RTX 4090 · 24.0 GB"
        totalVramBytes={24_000_000_000}
        assignments={[]}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
        placementTarget="node-a:gpu-1"
      />,
    );

    expect(screen.getByTestId('vram-container')).toHaveAttribute('data-placement-target', 'node-a:gpu-1');
  });

  it('does not render data-placement-target when placementTarget prop is omitted', () => {
    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={assignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={vi.fn()}
      />,
    );

    expect(screen.getByTestId('vram-container')).not.toHaveAttribute('data-placement-target');
  });

  it('shows the split-unavailable reason on keyboard focus when only one node is available', async () => {
      renderWithDnd(
        <VramContainer
          nodeId="node-a"
          nodeHostname="alpha.local"
          totalVramBytes={48_000_000_000}
          assignments={assignments}
          onRemoveModel={vi.fn()}
          selectedAssignmentId={null}
          onSelectAssignment={vi.fn()}
          onSplitModel={vi.fn()}
          modelScansLookup={scansLookup}
          availableNodeCount={1}
        />,
      );

      fireEvent.contextMenu(screen.getByTestId('vram-model-block'));

      expect(screen.getByTestId('ctx-menu-split')).toBeDisabled();
      expect(screen.queryByTestId('ctx-menu-split-reason')).not.toBeInTheDocument();

      const infoTrigger = screen.getByRole('button', { name: /why split is unavailable/i });
      fireEvent.focus(infoTrigger);

      await waitFor(() => {
        expect(screen.getByRole('tooltip')).toHaveTextContent('Split unavailable: add another available node to distribute this model.');
      });
    });
  });

  it('navigates between model blocks with arrow keys', async () => {
    const user = userEvent.setup();
    const onSelectAssignment = vi.fn();

    const twoAssignments: VramAssignment[] = [
      makeAssignment({ id: 'model-1', name: 'Model-1' }),
      makeAssignment({ id: 'model-2', name: 'Model-2', sizeBytes: 11_000_000_000, fullSizeBytes: 11_000_000_000, weightsBytes: 10_000_000_000, contextBytes: 1_000_000_000, sizeGb: 11 }),
    ];

    renderWithDnd(
      <VramContainer
        nodeId="node-a"
        nodeHostname="alpha.local"
        totalVramBytes={48_000_000_000}
        assignments={twoAssignments}
        onRemoveModel={vi.fn()}
        selectedAssignmentId={null}
        onSelectAssignment={onSelectAssignment}
      />,
    );

    const modelBlocks = screen.getAllByTestId('vram-model-block');
    expect(modelBlocks).toHaveLength(2);

    // Tab to first model block
    await user.tab();
    expect(modelBlocks[0]).toHaveFocus();

    // Arrow down to second model block
    await user.keyboard('{ArrowDown}');
    expect(modelBlocks[1]).toHaveFocus();

    // Arrow up back to first model block
    await user.keyboard('{ArrowUp}');
    expect(modelBlocks[0]).toHaveFocus();

    // Enter should trigger click
    await user.keyboard('{Enter}');
    expect(onSelectAssignment).toHaveBeenCalledWith('model-1');

    // Space should also trigger click
    onSelectAssignment.mockClear();
    await user.keyboard(' ');
    expect(onSelectAssignment).toHaveBeenCalledWith('model-1');
  });
});
