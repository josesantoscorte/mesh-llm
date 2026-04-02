import { describe, it, expect, vi, beforeEach } from 'vitest';
import { broadcastConfig, type ConfigValidationError } from '../api';

describe('broadcastConfig', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns structured errors array when backend validation fails', async () => {
    const mockErrors: ConfigValidationError[] = [
      { code: 'split_gap', path: 'nodes[0].models[0].split', message: 'Gap between layers 5 and 10' }
    ];

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      text: () => Promise.resolve(JSON.stringify({
        error: 'Config invalid',
        errors: mockErrors
      }))
    }));

    const result = await broadcastConfig('...');
    expect(result.ok).toBe(false);
    expect(result.errors).toHaveLength(1);
    expect(result.errors![0].code).toBe('split_gap');
    expect(result.errors![0].path).toBe('nodes[0].models[0].split');
    expect(result.errors![0].message).toBe('Gap between layers 5 and 10');
  });
});
