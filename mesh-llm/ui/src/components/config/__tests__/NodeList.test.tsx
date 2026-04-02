import { DragDropProvider } from '@dnd-kit/react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';

import type { OwnedNode } from '../../../hooks/useOwnedNodes';
import { NodeList } from '../NodeList';

const nodes: OwnedNode[] = [
  {
    id: 'node-a',
    hostname: 'alpha.local',
    role: 'Host',
    hardwareLabel: 'GPU',
    hardwareNames: ['RTX 4090'],
    gpuName: 'RTX 4090',
    vramGb: 24,
    models: ['Qwen3-30B-A3B-Q4_K_M'],
    statusLabel: 'Serving',
    statusTone: 'serving',
    isSelf: true,
    gpuTargets: [],
    aggregateVramGb: 24,
    separateCapable: false,
    mixedGpuWarning: false,
  },
  {
    id: 'node-b',
    hostname: 'beta.local',
    role: 'Worker',
    hardwareLabel: 'SoC',
    hardwareNames: ['M4 Max'],
    gpuName: 'M4 Max',
    vramGb: 16,
    models: ['GLM-4.5-Air-Q4_K_M'],
    statusLabel: 'Worker',
    statusTone: 'worker',
    isSelf: false,
    gpuTargets: [],
    aggregateVramGb: 16,
    separateCapable: false,
    mixedGpuWarning: false,
  },
];

describe('NodeList', () => {
  it('shows an instructional empty state when no owned nodes are available', () => {
    render(
      <DragDropProvider>
        <NodeList nodes={[]} selectedNodeId={null} onSelectNode={vi.fn()} />
      </DragDropProvider>,
    );

    expect(screen.getByText('No configurable nodes found')).toBeVisible();
    expect(screen.getByText(/same owner key/i)).toBeVisible();
  });

  it('updates the selected node highlight when a node card is clicked', async () => {
    const user = userEvent.setup();

    function Harness() {
      const [selectedNodeId, setSelectedNodeId] = useState<string | null>(nodes[0].id);
      return <NodeList nodes={nodes} selectedNodeId={selectedNodeId} onSelectNode={(node) => setSelectedNodeId(node.id)} />;
    }

    render(
      <DragDropProvider>
        <Harness />
      </DragDropProvider>,
    );

    const alphaCard = screen.getByRole('button', { name: /alpha\.local/i });
    const betaCard = screen.getByRole('button', { name: /beta\.local/i });

    expect(alphaCard).toHaveClass('ring-2');
    expect(alphaCard).toHaveClass('ring-primary');
    expect(betaCard).not.toHaveClass('ring-2');

    await user.click(betaCard);

    expect(betaCard).toHaveClass('ring-2');
    expect(betaCard).toHaveClass('ring-primary');
    expect(alphaCard).not.toHaveClass('ring-2');
  });
});
