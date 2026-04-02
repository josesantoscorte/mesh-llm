import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { AggregatedModelMoe } from '../../../lib/models';
import { MoeSlider } from '../MoeSlider';

const moe: AggregatedModelMoe = {
  nExpert: 64,
  nExpertUsed: 8,
  minExpertsPerNode: 4,
};

describe('MoeSlider', () => {
  it('renders label with correct expert count and total', () => {
    render(
      <MoeSlider
        modelName="Qwen3-30B-A3B"
        moe={moe}
        currentExperts={8}
        modelSizeBytes={20_000_000_000}
        onExpertsChange={vi.fn()}
      />,
    );

    expect(screen.getByText('8 of 64')).toBeVisible();
    expect(screen.getByText('Experts')).toBeVisible();
  });

  it('has correct min, max, step and value on the range input', () => {
    render(
      <MoeSlider
        modelName="Qwen3-30B-A3B"
        moe={moe}
        currentExperts={16}
        modelSizeBytes={20_000_000_000}
        onExpertsChange={vi.fn()}
      />,
    );

    const slider = screen.getByTestId('moe-slider') as HTMLInputElement;
    expect(slider.type).toBe('range');
    expect(slider.min).toBe('1');
    expect(slider.max).toBe('64');
    expect(slider.step).toBe('1');
    expect(slider.value).toBe('16');
  });

  it('fires onExpertsChange with integer value on change', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();

    const { container } = render(
      <MoeSlider
        modelName="Qwen3-30B-A3B"
        moe={moe}
        currentExperts={8}
        modelSizeBytes={20_000_000_000}
        onExpertsChange={onChange}
      />,
    );

    const slider = screen.getByTestId('moe-slider') as HTMLInputElement;

    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(slider, '32');
    slider.dispatchEvent(new Event('change', { bubbles: true }));

    expect(onChange).toHaveBeenCalledWith(32);
  });

  it('displays VRAM impact estimate', () => {
    render(
      <MoeSlider
        modelName="Qwen3-30B-A3B"
        moe={moe}
        currentExperts={8}
        modelSizeBytes={64_000_000_000}
        onExpertsChange={vi.fn()}
      />,
    );

    // 8 * 64e9 / 64 / 1e9 = 8.0 GB
    expect(screen.getByText('8.0 GB for 8 experts')).toBeVisible();
  });

  it('clamps value to valid range', () => {
    render(
      <MoeSlider
        modelName="Qwen3-30B-A3B"
        moe={moe}
        currentExperts={100}
        modelSizeBytes={20_000_000_000}
        onExpertsChange={vi.fn()}
      />,
    );

    const slider = screen.getByTestId('moe-slider') as HTMLInputElement;
    expect(slider.value).toBe('64');
    expect(screen.getByText('64 of 64')).toBeVisible();
  });
});
