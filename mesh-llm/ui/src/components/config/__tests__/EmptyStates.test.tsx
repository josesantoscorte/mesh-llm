import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import {
  CatalogSkeleton,
  ConfigLoadingError,
  EmptyNoModels,
  EmptyNoNodes,
  EmptyNoSelection,
  NodeListSkeleton,
} from '../EmptyStates';

describe('EmptyStates', () => {
  it('empty-no-nodes renders owner-key guidance without legacy owner-secret copy', () => {
    render(<EmptyNoNodes />);

    const container = screen.getByTestId('empty-no-nodes');
    expect(container).toBeVisible();
    expect(
      screen.getByText('This console does not have a trusted owner fingerprint yet'),
    ).toBeVisible();
    expect(screen.getByText(/owner key/i)).toBeVisible();
    expect(screen.getByText(/verified owner fingerprints appear here/i)).toBeVisible();
    expect(
      screen.getByText(/mesh-llm --owner-key ~\/\.mesh-llm\/owner-key --model Qwen2.5-32B/i),
    ).toBeVisible();
    expect(screen.queryByText(/mesh-llm --owner\s+<your-secret>/i)).toBeNull();
    expect(screen.queryByText(/my-secret-phrase/)).toBeNull();
  });

  it('empty-no-models renders refresh instruction', () => {
    const onRefresh = vi.fn();
    render(<EmptyNoModels onRefresh={onRefresh} />);

    const container = screen.getByTestId('empty-no-models');
    expect(container).toBeVisible();
    expect(screen.getByText('No models found')).toBeVisible();
    expect(screen.getByText(/~\/\.models\//)).toBeVisible();
    expect(screen.getByRole('button', { name: /refresh/i })).toBeVisible();
  });

  it('empty-no-models refresh button calls onRefresh', async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn();
    render(<EmptyNoModels onRefresh={onRefresh} />);

    await user.click(screen.getByRole('button', { name: /refresh/i }));
    expect(onRefresh).toHaveBeenCalledOnce();
  });

  it('empty-no-selection renders selection placeholder', () => {
    render(<EmptyNoSelection />);

    const container = screen.getByTestId('empty-no-selection');
    expect(container).toBeVisible();
    expect(screen.getByText('Select a node to configure')).toBeVisible();
  });

  it('config-loading-error renders error with retry', async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    render(<ConfigLoadingError error="Connection refused" onRetry={onRetry} />);

    const container = screen.getByTestId('config-loading-error');
    expect(container).toBeVisible();
    expect(screen.getByText('Connection refused')).toBeVisible();
    expect(screen.getByRole('button', { name: /retry/i })).toBeVisible();

    await user.click(screen.getByRole('button', { name: /retry/i }));
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it('node-list-skeleton renders 3 shimmer placeholders', () => {
    render(<NodeListSkeleton />);

    const container = screen.getByTestId('node-list-skeleton');
    expect(container).toBeVisible();

    const skeletons = container.querySelectorAll('.animate-pulse');
    expect(skeletons).toHaveLength(3);
  });

  it('catalog-skeleton renders placeholder grid', () => {
    render(<CatalogSkeleton />);

    const container = screen.getByTestId('catalog-skeleton');
    expect(container).toBeVisible();

    const skeletons = container.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBeGreaterThanOrEqual(4);
  });
});
