import { describe, it, expect, vi, beforeEach } from 'vitest';
import { broadcastConfig, broadcastScan, fetchAuthoredConfig, type ConfigValidationError } from '../api';
import { parseConfig } from '../config';

describe('broadcastConfig', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('regression: ok is true when backend omits the failed field', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ saved: 2, total: 2 }),
    }));

    const result = await broadcastConfig('version = 1\n');
    expect(result.ok).toBe(true);
    expect(result.saved).toBe(2);
    expect(result.total).toBe(2);
    expect(result.failed).toEqual([]);
  });

  it('returns ok and errors array when backend validation fails', async () => {
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

describe('broadcastScan', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('regression: ok is true when backend omits the failed field', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ refreshed: 3, total: 3 }),
    }));

    const result = await broadcastScan();
    expect(result.ok).toBe(true);
    expect(result.refreshed).toBe(3);
    expect(result.total).toBe(3);
    expect(result.failed).toEqual([]);
  });
});

describe('fetchAuthoredConfig', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns raw response text as the TOML config string', async () => {
    const tomlBody = 'version = 1\n';
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve(tomlBody),
    }));

    const result = await fetchAuthoredConfig();
    expect(result.ok).toBe(true);
    expect(result.config).toBe(tomlBody);
    expect(parseConfig(result.config!)).toEqual({ version: 1, nodes: [] });
  });

  it('regression: JSON-encoded response body is not parseable as TOML', async () => {
    const jsonBody = JSON.stringify({ ok: true, config: '' });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve(jsonBody),
    }));

    const result = await fetchAuthoredConfig();
    expect(result.ok).toBe(true);
    expect(result.config).toBe(jsonBody);
    expect(parseConfig(result.config!)).toBeNull();
  });

  it('returns an error when the request fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      text: () => Promise.resolve('Internal Server Error'),
    }));

    const result = await fetchAuthoredConfig();
    expect(result.ok).toBe(false);
    expect(result.error).toBe('Internal Server Error');
  });
});
