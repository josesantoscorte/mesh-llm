import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

vi.mock('animejs', () => ({
  animate: vi.fn(),
}));

import { ModelCard } from '../ModelCard';

describe('ModelCard', () => {
  it('renders compact name and fit icon', () => {
    render(
      <ModelCard
        name="Qwen3-30B-A3B-Q4_K_M"
        widthPercent={50}
        fitStatus="fits"
        onRemove={vi.fn()}
      />,
    );

    expect(screen.getByText('Qwen3-30B-A3B-Q4_K_M')).toBeVisible();
    expect(screen.getByTestId('fit-icon-ok')).toBeVisible();
  });

  it('shows overcommit icon when fitStatus is overcommit', () => {
    render(
      <ModelCard
        name="Big-Model"
        widthPercent={80}
        fitStatus="overcommit"
        onRemove={vi.fn()}
      />,
    );

    expect(screen.getByTestId('fit-icon-warn')).toBeVisible();
  });

  it('shows unknown icon when fitStatus is unknown', () => {
    render(
      <ModelCard
        name="Unknown-Model"
        widthPercent={30}
        fitStatus="unknown"
        onRemove={vi.fn()}
      />,
    );

    expect(screen.getByTestId('fit-icon-unknown')).toBeVisible();
  });

  it('shows error badge when errorMessage is provided', () => {
    render(
      <ModelCard
        name="Error-Model"
        widthPercent={50}
        fitStatus="fits"
        onRemove={vi.fn()}
        errorMessage="Split validation failed"
      />,
    );

    expect(screen.getByTestId('model-card-error-badge')).toBeVisible();
    expect(screen.getByTestId('model-card-error-badge')).toHaveAttribute('title', 'Split validation failed');
  });

  it('highlights when selected', () => {
    render(
      <ModelCard
        name="Selected-Model"
        widthPercent={50}
        fitStatus="fits"
        isSelected
        onRemove={vi.fn()}
      />,
    );

    const card = screen.getByTestId('vram-model-card');
    expect(card.className).toContain('ring-1');
    expect(card.className).toContain('border-primary/60');
  });

  it('fires onSelect when the card body is clicked', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();

    render(
      <ModelCard
        name="Clickable-Model"
        widthPercent={50}
        fitStatus="fits"
        onSelect={onSelect}
        onRemove={vi.fn()}
      />,
    );

    await user.click(screen.getByTestId('vram-model-card'));
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it('keeps remove button clickable without firing onSelect', async () => {
    const user = userEvent.setup();
    const onRemove = vi.fn();
    const onSelect = vi.fn();

    render(
      <ModelCard
        name="Qwen3-30B-A3B-Q4_K_M"
        widthPercent={50}
        fitStatus="fits"
        onSelect={onSelect}
        onRemove={onRemove}
      />,
    );

    await user.click(screen.getByRole('button', { name: /remove qwen3-30b-a3b-q4_k_m/i }));
    expect(onRemove).toHaveBeenCalledTimes(1);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it('does not bubble pointerdown from remove button to parent drag surfaces', () => {
    const onParentPointerDown = vi.fn();

    render(
      <div onPointerDown={onParentPointerDown}>
        <ModelCard
          name="Qwen3-30B-A3B-Q4_K_M"
          widthPercent={50}
          fitStatus="fits"
          onRemove={vi.fn()}
        />
      </div>,
    );

    const removeButton = screen.getByRole('button', { name: /remove qwen3-30b-a3b-q4_k_m/i });
    fireEvent.pointerDown(removeButton);
    expect(onParentPointerDown).not.toHaveBeenCalled();
  });
});
