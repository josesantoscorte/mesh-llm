import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { ScannedModelMetadata } from '../../../types/config';
import { ContextWindowControl } from '../ContextWindowControl';

const qwenMetadata: ScannedModelMetadata = {
  architecture: 'qwen3',
  context_length: 32768,
  embedding_length: 5120,
  total_layers: 64,
  attention: {
    head_count: 24,
    head_count_kv: 4,
    key_length: 128,
    value_length: 128,
  },
};

describe('ContextWindowControl', () => {
  it('renders with correct default context size label', () => {
    render(
      <ContextWindowControl
        modelName="Qwen3-30B-A3B"
        currentCtxSize={8192}
        modelSizeBytes={22_000_000_000}
        metadata={qwenMetadata}
        onCtxSizeChange={vi.fn()}
      />,
    );

    expect(screen.getByText('Context')).toBeVisible();
    expect(screen.getByText('8K ctx')).toBeVisible();
  });

  it('uses a log-scale slider with 0-1000 range', () => {
    render(
      <ContextWindowControl
        modelName="Qwen3-30B-A3B"
        currentCtxSize={4096}
        modelSizeBytes={22_000_000_000}
        metadata={qwenMetadata}
        onCtxSizeChange={vi.fn()}
      />,
    );

    const slider = screen.getByTestId('ctx-resize-handle') as HTMLInputElement;
    expect(slider.type).toBe('range');
    expect(slider.min).toBe('0');
    expect(slider.max).toBe('1000');
    expect(slider.step).toBe('1');
  });

  it('slider value maps to log scale position', () => {
    render(
      <ContextWindowControl
        modelName="Qwen3-30B-A3B"
        currentCtxSize={4096}
        modelSizeBytes={22_000_000_000}
        metadata={qwenMetadata}
        onCtxSizeChange={vi.fn()}
      />,
    );

    const slider = screen.getByTestId('ctx-resize-handle') as HTMLInputElement;
    const pos = parseFloat(slider.value);
    expect(pos).toBeGreaterThan(0);
    expect(pos).toBeLessThan(1000);
  });

  it('shows model-aware KV cache estimate', () => {
    render(
      <ContextWindowControl
        modelName="Qwen3-30B-A3B"
        currentCtxSize={8192}
        modelSizeBytes={22_000_000_000}
        metadata={qwenMetadata}
        onCtxSizeChange={vi.fn()}
      />,
    );

    expect(screen.getByText('+0.5 GB KV cache')).toBeVisible();
  });

  it('renders scale marks for min and metadata max', () => {
    render(
      <ContextWindowControl
        modelName="TestModel"
        currentCtxSize={2048}
        modelSizeBytes={22_000_000_000}
        metadata={qwenMetadata}
        onCtxSizeChange={vi.fn()}
      />,
    );

    const scaleMarks = screen.getByTestId('ctx-scale-marks');
    expect(within(scaleMarks).getByText('512')).toBeInTheDocument();
    expect(within(scaleMarks).getByText('32K')).toBeInTheDocument();
  });

  it('renders a 1M scale mark for the one-million fallback max', () => {
    render(
      <ContextWindowControl
        modelName="TestModel"
        currentCtxSize={2048}
        modelSizeBytes={22_000_000_000}
        metadata={{ ...qwenMetadata, context_length: undefined }}
        onCtxSizeChange={vi.fn()}
      />,
    );

    const scaleMarks = screen.getByTestId('ctx-scale-marks');
    expect(within(scaleMarks).getByText('1M')).toBeInTheDocument();
  });

  it('highlights scale marks at or below current context value', () => {
    render(
      <ContextWindowControl
        modelName="TestModel"
        currentCtxSize={4096}
        modelSizeBytes={22_000_000_000}
        metadata={qwenMetadata}
        onCtxSizeChange={vi.fn()}
      />,
    );

    const scaleMarks = screen.getByTestId('ctx-scale-marks');
    const mark512 = within(scaleMarks).getByText('512');
    const mark2K = within(scaleMarks).getByText('2K');
    const mark4K = within(scaleMarks).getByText('4K');
    const mark8K = within(scaleMarks).getByText('8K');

    expect(mark512.className).toContain('text-foreground');
    expect(mark2K.className).toContain('text-foreground');
    expect(mark4K.className).toContain('text-foreground');
    expect(mark8K.className).toContain('text-muted-foreground/50');
  });

  it('fires onCtxSizeChange when a scale mark is clicked', () => {
    const onChange = vi.fn();

    render(
      <ContextWindowControl
        modelName="TestModel"
        currentCtxSize={2048}
        modelSizeBytes={22_000_000_000}
        metadata={qwenMetadata}
        onCtxSizeChange={onChange}
      />,
    );

    const scaleMarks = screen.getByTestId('ctx-scale-marks');
    const mark8K = within(scaleMarks).getByText('8K');
    fireEvent.click(mark8K.closest('button')!);

    expect(onChange).toHaveBeenCalledWith(8192);
  });


});
