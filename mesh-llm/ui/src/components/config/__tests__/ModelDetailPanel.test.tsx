import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { AggregatedModel, AggregatedModelMoe } from '../../../lib/models';
import type { ModelAssignment, ScannedModelMetadata } from '../../../types/config';
import { ModelDetailPanel } from '../ModelDetailPanel';

const baseMetadata: ScannedModelMetadata = {
  architecture: 'llama',
  context_length: 32768,
  embedding_length: 4096,
  quantization_type: 'Q4_K_M',
  attention: {
    head_count: 32,
    head_count_kv: 8,
    key_length: 128,
    value_length: 128,
  },
  total_layers: 64,
  total_offloadable_layers: 65,
  dense_split_capable: true,
  rope: { kind: 'linear', freq_base: 10000, factor: 1, original_context_length: 8192 },
  experts: undefined,
  tokenizer: { model: 'gpt2', pre: 'default', vocab_size: 32000 },
};

const baseAssignment: ModelAssignment = {
  name: 'Qwen3-30B-A3B-Q4_K_M',
  ctx_size: 8192,
};

const baseAggregated: AggregatedModel = {
  name: 'Qwen3-30B-A3B-Q4_K_M',
  sizeBytes: 22_000_000_000,
  sizeGb: 22,
  nodeIds: ['node-a'],
};

describe('ModelDetailPanel', () => {
  it('shows empty state when no model is selected', () => {
    render(
      <ModelDetailPanel
        modelName={null}
        assignment={null}
        aggregated={null}
        metadata={null}
      />,
    );

    expect(screen.getByTestId('detail-panel-empty')).toBeVisible();
    expect(screen.getByText('Select a model block to inspect it')).toBeVisible();
  });

  it('renders rich metadata when a model is selected', () => {
    render(
      <ModelDetailPanel
        modelName="Qwen3-30B-A3B-Q4_K_M"
        assignment={baseAssignment}
        aggregated={baseAggregated}
        metadata={baseMetadata}
      />,
    );

    expect(screen.getByTestId('model-detail-panel')).toBeVisible();
    expect(screen.getByTestId('detail-panel-metadata')).toHaveClass('grid', 'grid-cols-2', 'sm:grid-cols-4', 'xl:grid-cols-6');
    expect(screen.getByText('Qwen3-30B-A3B-Q4_K_M')).toBeVisible();
    expect(screen.getByText('llama')).toBeVisible();
    expect(screen.getByText('Q4_K_M')).toBeVisible();
    expect(screen.getByText('32K')).toBeVisible();
    expect(screen.getByText('Core weights')).toBeVisible();
    expect(screen.getByText('22.0 GB')).toBeVisible();
    expect(screen.getByText('Context cache')).toBeVisible();
    expect(screen.getByText('1.1 GB')).toBeVisible();
    expect(screen.getByText('Estimated total')).toBeVisible();
    expect(screen.getByText('23.1 GB')).toBeVisible();
    expect(screen.getByText('4,096')).toBeVisible();
    expect(screen.getByText('64')).toBeVisible();
    expect(screen.getAllByText('32').length).toBeGreaterThan(0);
    expect(screen.getByText('8')).toBeVisible();
    expect(screen.getByText('65')).toBeVisible();
    expect(screen.getByText('gpt2')).toBeVisible();
    expect(screen.getByText('Capable')).toBeVisible();
  });

  it('shows split range when assignment has a split', () => {
    const splitAssignment: ModelAssignment = {
      ...baseAssignment,
      split: { start: 0, end: 31, total: 64 },
    };

    render(
      <ModelDetailPanel
        modelName="Qwen3-30B-A3B-Q4_K_M"
        assignment={splitAssignment}
        aggregated={baseAggregated}
        metadata={baseMetadata}
      />,
    );

    expect(screen.getByText(/0–31 of 64/)).toBeVisible();
  });

  it('shows context window and MoE controls when onUpdateModel is provided', () => {
    const onUpdate = vi.fn();
    const moe: AggregatedModelMoe = { nExpert: 64, nExpertUsed: 8, minExpertsPerNode: 4 };
    const aggregatedWithMoe: AggregatedModel = { ...baseAggregated, moe };

    render(
      <ModelDetailPanel
        modelName="Qwen3-30B-A3B-Q4_K_M"
        assignment={{ ...baseAssignment, moe_experts: 8 }}
        aggregated={aggregatedWithMoe}
        metadata={baseMetadata}
        onUpdateModel={onUpdate}
      />,
    );

    expect(screen.getByTestId('ctx-resize-handle')).toBeVisible();
    expect(screen.getByTestId('moe-slider')).toBeVisible();
  });

  it('shows fallback message when no metadata is available', () => {
    render(
      <ModelDetailPanel
        modelName="Qwen3-30B-A3B-Q4_K_M"
        assignment={baseAssignment}
        aggregated={baseAggregated}
        metadata={null}
      />,
    );

    expect(screen.getByText('No GGUF metadata available')).toBeVisible();
  });

  it('shows expert info when metadata has experts', () => {
    const moeMetadata: ScannedModelMetadata = {
      ...baseMetadata,
      experts: { expert_count: 64, expert_used_count: 8 },
    };

    render(
      <ModelDetailPanel
        modelName="Qwen3-30B-A3B-Q4_K_M"
        assignment={baseAssignment}
        aggregated={baseAggregated}
        metadata={moeMetadata}
      />,
    );

    expect(screen.getByText('64 total, 8 active')).toBeVisible();
  });

  it('shows RoPE details when present in metadata', () => {
    render(
      <ModelDetailPanel
        modelName="Qwen3-30B-A3B-Q4_K_M"
        assignment={baseAssignment}
        aggregated={baseAggregated}
        metadata={baseMetadata}
      />,
    );

    expect(screen.getByText('linear')).toBeVisible();
    expect(screen.getByText('10,000')).toBeVisible();
  });
});
