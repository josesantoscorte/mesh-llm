import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { ModelCatalogPeer } from '../../../lib/models';
import { ModelCatalog } from '../ModelCatalog';

const peers: ModelCatalogPeer[] = [
  {
    id: 'node-a',
    hostname: 'alpha.local',
    models: ['Qwen3-30B-A3B-Q4_K_M', 'GLM-4.5-Air-Q4_K_M'],
    model_sizes: [
      ['Qwen3-30B-A3B-Q4_K_M', 22_000_000_000],
      ['GLM-4.5-Air-Q4_K_M', 10_000_000_000],
    ],
    mesh_models: [
      {
        name: 'Qwen3-30B-A3B-Q4_K_M',
        size_gb: 22,
        moe: {
          n_expert: 128,
          n_expert_used: 8,
          min_experts_per_node: 24,
        },
      },
    ],
  },
  {
    id: 'node-b',
    hostname: 'beta.local',
    models: ['Llama-3.3-70B-Q4_K_M', 'Qwen3-30B-A3B-Q4_K_M'],
    model_sizes: [
      ['Llama-3.3-70B-Q4_K_M', 43_000_000_000],
      ['Qwen3-30B-A3B-Q4_K_M', 21_000_000_000],
    ],
  },
];

function makeManyModelPeers(count: number): ModelCatalogPeer[] {
  const names = Array.from({ length: count }, (_, i) => `Model-${String(i).padStart(3, '0')}-Q4_K_M`);
  names.push('Llama-3.3-70B-Q4_K_M');
  return [
    {
      id: 'node-many',
      hostname: 'many.local',
      models: names,
      model_sizes: names.map((n) => [n, 5_000_000_000] as const),
    },
  ];
}

describe('ModelCatalog', () => {
  it('renders one card per aggregated model', () => {
    render(
      <ModelCatalog
        peers={peers}
        selectedNode={{ id: 'node-a', hostname: 'alpha.local', vramBytes: 24_000_000_000 }}
      />,
    );

    expect(screen.getByTestId('model-catalog')).toBeVisible();
    expect(screen.getAllByTestId('model-card')).toHaveLength(3);
    expect(screen.getByText('MoE 8/128')).toBeVisible();
  });

  it('filters visible cards as the search input changes', async () => {
    const user = userEvent.setup();
    const manyPeers = makeManyModelPeers(22);

    render(<ModelCatalog peers={manyPeers} selectedNode={null} />);

    expect(screen.getAllByTestId('model-card').length).toBeGreaterThan(20);

    await user.type(screen.getByTestId('model-search'), 'llama');

    expect(screen.getAllByTestId('model-card')).toHaveLength(1);
    expect(screen.getByRole('button', { name: /llama-3\.3-70b-q4_k_m/i })).toBeVisible();
  });

  it('shows the search input once the catalog exceeds 8 aggregated models', () => {
    render(<ModelCatalog peers={makeManyModelPeers(8)} selectedNode={null} />);

    expect(screen.getByTestId('model-search')).toBeVisible();
  });

  it('shows fit and no-fit states when a node is selected', () => {
    render(
      <ModelCatalog
        peers={peers}
        selectedNode={{ id: 'node-a', hostname: 'alpha.local', vramBytes: 24_000_000_000 }}
      />,
    );

    expect(screen.getByRole('button', { name: /glm-4\.5-air-q4_k_m/i })).toHaveAttribute('data-fits', 'true');
    expect(screen.getByRole('button', { name: /llama-3\.3-70b-q4_k_m/i })).toHaveAttribute('data-fits', 'false');
  });

  it('keeps the model card hover/focus group class for detail reveal states', () => {
    render(<ModelCatalog peers={peers} selectedNode={null} />);

    expect(screen.getAllByTestId('model-card')[0]?.className).toContain('group/model');
  });

  it('shows the GGUF empty state when no models are available', () => {
    render(<ModelCatalog peers={[]} selectedNode={null} />);

    expect(screen.getByText('Catalog is empty')).toBeVisible();
    expect(screen.getByText(/No models found\. Add GGUF files to/i)).toBeVisible();
    expect(screen.getByText('0 models')).toBeVisible();
  });

  it('shows a filter-aware no-results state with recovery actions', async () => {
    const user = userEvent.setup();

    render(<ModelCatalog peers={makeManyModelPeers(22)} selectedNode={null} />);

    await user.click(screen.getByTestId('filter-pill-vision'));
    await user.type(screen.getByTestId('model-search'), 'qwen');

    expect(screen.getByText('No matching models')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Clear search' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Reset filters' })).toBeVisible();
  });

  it('hides the filter lane while the search input is focused on larger layouts', async () => {
    const user = userEvent.setup();
    const manyPeers = makeManyModelPeers(22);

    render(<ModelCatalog peers={manyPeers} selectedNode={null} />);

    const search = screen.getByTestId('model-search');
    const searchShell = search.parentElement;
    const searchLane = searchShell?.parentElement;
    const searchRow = searchLane?.parentElement;
    const filterFieldset = screen.getByTestId('filter-pill-all').closest('fieldset');
    const filterLane = filterFieldset?.parentElement;

    expect(searchShell?.className).toContain('h-9');
    expect(searchLane?.className).toContain('lg:min-w-[10rem]');
    expect(searchLane?.className).toContain('lg:w-[12rem]');
    expect(searchLane?.className).toContain('lg:max-w-[13rem]');
    expect(searchRow?.className).toContain('lg:gap-3');
    expect(searchRow?.className).toContain('lg:min-h-10');
    expect(filterFieldset).not.toBeNull();
    expect(filterFieldset?.className).not.toContain('lg:opacity-0');
    expect(filterFieldset?.className).not.toContain('lg:h-0');
    expect(filterLane?.className).not.toContain('lg:w-0');

    await user.click(search);

    expect(searchShell?.className).toContain('h-9');
    expect(searchLane?.className).toContain('lg:flex-1');
    expect(searchLane?.className).toContain('lg:max-w-none');
    expect(searchRow?.className).toContain('lg:gap-0');
    expect(filterFieldset?.className).toContain('lg:opacity-0');
    expect(filterFieldset?.className).toContain('lg:pointer-events-none');
    expect(filterFieldset?.className).toContain('lg:h-0');
    expect(filterFieldset?.className).toContain('lg:overflow-hidden');
    expect(filterLane?.className).toContain('lg:w-0');
    expect(filterLane?.className).toContain('lg:max-w-0');
    expect(screen.queryByText(/^Search$/)).not.toBeInTheDocument();

    await user.tab();

    expect(searchShell?.className).toContain('h-9');
    expect(searchLane?.className).toContain('lg:min-w-[10rem]');
    expect(searchLane?.className).toContain('lg:w-[12rem]');
    expect(searchLane?.className).toContain('lg:max-w-[13rem]');
    expect(searchRow?.className).toContain('lg:gap-3');
    expect(filterFieldset?.className).not.toContain('lg:opacity-0');
    expect(filterFieldset?.className).not.toContain('lg:h-0');
    expect(filterLane?.className).not.toContain('lg:w-0');
  });
});
