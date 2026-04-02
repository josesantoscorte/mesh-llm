import { describe, expect, it } from 'vitest';

import { aggregateModels, type ModelCatalogPeer } from '../models';

describe('aggregateModels', () => {
  it('deduplicates by model name, prefers the largest size, and tracks node ids', () => {
    const peers: ModelCatalogPeer[] = [
      {
        id: 'node-a',
        models: ['Qwen3-30B-A3B-Q4_K_M', 'Llama-3.3-70B-Q4_K_M'],
        model_sizes: [
          ['Qwen3-30B-A3B-Q4_K_M', 21_000_000_000],
          ['Llama-3.3-70B-Q4_K_M', 43_000_000_000],
        ],
        mesh_models: [
          {
            name: 'Qwen3-30B-A3B-Q4_K_M',
            size_gb: 21,
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
        models: ['Qwen3-30B-A3B-Q4_K_M', 'GLM-4.5-Air-Q4_K_M'],
        model_sizes: [
          ['Qwen3-30B-A3B-Q4_K_M', 22_000_000_000],
          ['GLM-4.5-Air-Q4_K_M', 10_000_000_000],
        ],
      },
    ];

    const aggregated = aggregateModels(peers);

    expect(aggregated).toHaveLength(3);
    expect(aggregated[0]).toMatchObject({
      name: 'Qwen3-30B-A3B-Q4_K_M',
      sizeBytes: 22_000_000_000,
      nodeIds: ['node-a', 'node-b'],
      moe: {
        nExpert: 128,
        nExpertUsed: 8,
        minExpertsPerNode: 24,
      },
    });
    expect(aggregated.map((model) => model.name)).toEqual([
      'Qwen3-30B-A3B-Q4_K_M',
      'GLM-4.5-Air-Q4_K_M',
      'Llama-3.3-70B-Q4_K_M',
    ]);
  });

  it('matches model sizes case-insensitively when peer.models and peer.model_sizes use different casing', () => {
    // Real-world scenario: election assigns lowercase names like "qwen2.5-3b-instruct-q4_k_m"
    // but model_sizes uses GGUF filename casing like "Qwen2.5-3B-Instruct-Q4_K_M"
    const peers: ModelCatalogPeer[] = [
      {
        id: 'carrack',
        models: ['qwen2.5-3b-instruct-q4_k_m'],
        model_sizes: [
          ['Qwen2.5-3B-Instruct-Q4_K_M', 2_104_932_768],
        ],
      },
    ];

    const aggregated = aggregateModels(peers);

    expect(aggregated).toHaveLength(1);
    expect(aggregated[0].sizeBytes).toBe(2_104_932_768);
    expect(aggregated[0].name).toBe('Qwen2.5-3B-Instruct-Q4_K_M');
    expect(aggregated[0].nodeIds).toEqual(['carrack']);
  });

  it('merges case-variant names into one entry across multiple peers', () => {
    const peers: ModelCatalogPeer[] = [
      {
        id: 'node-a',
        models: ['qwen2.5-3b-instruct-q4_k_m'],
        model_sizes: [
          ['Qwen2.5-3B-Instruct-Q4_K_M', 2_104_932_768],
        ],
      },
      {
        id: 'node-b',
        models: ['Qwen2.5-3B-Instruct-Q4_K_M'],
        model_sizes: [
          ['Qwen2.5-3B-Instruct-Q4_K_M', 2_104_932_768],
        ],
      },
    ];

    const aggregated = aggregateModels(peers);

    const matching = aggregated.filter(
      (m) => m.name.toLowerCase() === 'qwen2.5-3b-instruct-q4_k_m',
    );
    expect(matching).toHaveLength(1);
    expect(matching[0].sizeBytes).toBe(2_104_932_768);
    expect(matching[0].nodeIds).toEqual(['node-a', 'node-b']);
  });

  it('includes models discovered only from size tuples when local model names are absent', () => {
    const aggregated = aggregateModels([
      {
        id: 'node-a',
        models: [],
        model_sizes: [['Qwen2.5-7B-Instruct-Q4_K_M', 5_000_000_000]],
      },
    ]);

    expect(aggregated).toEqual([
      {
        name: 'Qwen2.5-7B-Instruct-Q4_K_M',
        sizeBytes: 5_000_000_000,
        sizeGb: 5,
        nodeIds: ['node-a'],
      },
    ]);
  });
});
