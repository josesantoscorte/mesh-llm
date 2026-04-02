import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { PlacementModeDialog } from '../PlacementModeDialog';

describe('PlacementModeDialog', () => {
  it('confirm button testid contains the node id when provided', () => {
    render(
      <PlacementModeDialog
        open={true}
        pendingNodeId="node-abc123"
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByTestId('node-node-abc123-mode-confirm')).toBeInTheDocument();
  });

  it('regression: confirm button testid must not contain the string "undefined"', () => {
    render(
      <PlacementModeDialog
        open={true}
        pendingNodeId=""
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    const allTestIds = document
      .querySelectorAll('[data-testid]');
    const testIds = Array.from(allTestIds).map((el) => el.getAttribute('data-testid') ?? '');
    const hasUndefined = testIds.some((id) => id.includes('undefined'));
    expect(hasUndefined).toBe(false);
  });
});
