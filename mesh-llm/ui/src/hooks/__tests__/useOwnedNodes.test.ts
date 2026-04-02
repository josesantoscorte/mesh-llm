import { renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { StatusPayload } from '../../App';
import { useOwnedNodes } from '../useOwnedNodes';

type TestStatusPayload = StatusPayload & {
  mesh_models?: unknown[];
  owner_id?: string | null;
  owner_fingerprint?: string | null;
  owner_fingerprint_verified?: boolean;
  owner_fingerprint_transitive?: boolean;
  peers: Array<
    StatusPayload['peers'][number] & {
      owner_id?: string | null;
      owner_fingerprint?: string | null;
      owner_fingerprint_verified?: boolean;
      owner_fingerprint_transitive?: boolean;
    }
  >;
};

function createStatus(overrides: Partial<TestStatusPayload> = {}): TestStatusPayload {
  return {
    node_id: 'self-node-1234',
    token: 'invite-token',
    node_status: 'Serving',
    is_host: true,
    is_client: false,
    llama_ready: true,
    model_name: 'Qwen3-30B-A3B-Q4_K_M',
    serving_models: ['Qwen3-30B-A3B-Q4_K_M'],
    api_port: 3131,
    my_vram_gb: 48,
    model_size_gb: 20,
    peers: [],
    mesh_models: [],
    inflight_requests: 0,
    my_hostname: 'charlie.local',
    gpus: [{ name: 'RTX 6000 Ada', vram_bytes: 48_000_000_000 }],
    owner_id: 'shared-owner',
    owner_fingerprint: 'fingerprint-shared-owner',
    owner_fingerprint_verified: true,
    owner_fingerprint_transitive: false,
    ...overrides,
  };
}

describe('useOwnedNodes', () => {
  it('filters by trusted owner fingerprint, includes self, and sorts by hostname', () => {
    const status = createStatus({
      peers: [
        {
          id: 'peer-b',
          role: 'Worker',
          models: ['Llama-3.3-70B-Q4_K_M'],
          vram_gb: 24,
          hostname: 'bravo.local',
          owner_id: 'shared-owner',
          owner_fingerprint: 'fingerprint-shared-owner',
          owner_fingerprint_verified: true,
          owner_fingerprint_transitive: false,
          gpus: [{ name: 'RTX 4090', vram_bytes: 24_000_000_000 }],
        },
        {
          id: 'peer-a',
          role: 'Host',
          models: ['GLM-4.5-Air-Q4_K_M'],
          vram_gb: 16,
          hostname: 'alpha.local',
          owner_id: 'shared-owner',
          owner_fingerprint: 'fingerprint-shared-owner',
          owner_fingerprint_verified: true,
          owner_fingerprint_transitive: false,
          serving_models: ['GLM-4.5-Air-Q4_K_M'],
          gpus: [{ name: 'M4 Max', vram_bytes: 16_000_000_000 }],
        },
        {
          id: 'peer-z',
          role: 'Client',
          models: ['ShouldNotAppear'],
          vram_gb: 0,
          hostname: 'zulu.local',
          owner_id: 'different-owner',
          owner_fingerprint: 'fingerprint-different-owner',
          owner_fingerprint_verified: true,
          owner_fingerprint_transitive: false,
          gpus: [],
        },
      ],
    });

    const { result } = renderHook(() => useOwnedNodes(status));

    expect(result.current.map((node) => node.hostname)).toEqual(['alpha.local', 'bravo.local', 'charlie.local']);
    expect(result.current.map((node) => node.id)).toEqual(['peer-a', 'peer-b', 'self-node-1234']);
    expect(result.current[2]).toMatchObject({
      isSelf: true,
      gpuName: 'RTX 6000 Ada',
      models: ['Qwen3-30B-A3B-Q4_K_M'],
      statusLabel: 'Serving',
    });
    expect(result.current[1]).toMatchObject({
      isSelf: false,
      gpuName: 'RTX 4090',
      hardwareLabel: 'GPU',
    });
  });

  it('groups nodes by owner fingerprint when owner_id is absent', () => {
    const status = createStatus({
      owner_id: null,
      owner_fingerprint: 'fingerprint-shared-owner',
      owner_fingerprint_verified: true,
      peers: [
        {
          id: 'peer-alpha',
          role: 'Worker',
          models: ['GLM-4.7-Flash-Q4_K_M'],
          vram_gb: 32,
          hostname: 'alpha.local',
          owner_fingerprint: 'fingerprint-shared-owner',
          owner_fingerprint_verified: true,
          gpus: [{ name: 'RTX 4090', vram_bytes: 24_000_000_000 }],
        },
        {
          id: 'peer-echo',
          role: 'Worker',
          models: ['ShouldNotAppear'],
          vram_gb: 16,
          hostname: 'echo.local',
          owner_fingerprint: 'fingerprint-other-owner',
          owner_fingerprint_verified: true,
          gpus: [{ name: 'RTX 3080', vram_bytes: 10_000_000_000 }],
        },
      ],
    });

    const { result } = renderHook(() => useOwnedNodes(status));

    expect(result.current.map((node) => node.id)).toEqual([
      'peer-alpha',
      'self-node-1234',
    ]);
    expect(result.current[0]).toMatchObject({
      hostname: 'alpha.local',
      isSelf: false,
    });
  });

  it('excludes unverified or transitive owner fingerprint peers even when fingerprints match', () => {
    const status = createStatus({
      owner_id: null,
      owner_fingerprint: 'fingerprint-shared-owner',
      owner_fingerprint_verified: true,
      owner_fingerprint_transitive: false,
      peers: [
        {
          id: 'peer-trusted',
          role: 'Worker',
          models: ['GLM-4.7-Flash-Q4_K_M'],
          vram_gb: 32,
          hostname: 'trusted.local',
          owner_fingerprint: 'fingerprint-shared-owner',
          owner_fingerprint_verified: true,
          owner_fingerprint_transitive: false,
          gpus: [{ name: 'RTX 4090', vram_bytes: 24_000_000_000 }],
        },
        {
          id: 'peer-transitive',
          role: 'Worker',
          models: ['GLM-4.7-Flash-Q4_K_M'],
          vram_gb: 32,
          hostname: 'transitive.local',
          owner_fingerprint: 'fingerprint-shared-owner',
          owner_fingerprint_verified: true,
          owner_fingerprint_transitive: true,
          gpus: [{ name: 'RTX 4090', vram_bytes: 24_000_000_000 }],
        },
        {
          id: 'peer-unverified',
          role: 'Worker',
          models: ['GLM-4.7-Flash-Q4_K_M'],
          vram_gb: 32,
          hostname: 'unverified.local',
          owner_fingerprint: 'fingerprint-shared-owner',
          owner_fingerprint_verified: false,
          gpus: [{ name: 'RTX 4090', vram_bytes: 24_000_000_000 }],
        },
      ],
    });

    const { result } = renderHook(() => useOwnedNodes(status));

    expect(result.current.map((node) => node.id)).toEqual([
      'self-node-1234',
      'peer-trusted',
    ]);
    expect(result.current.some((node) => node.id === 'peer-transitive')).toBe(false);
    expect(result.current.some((node) => node.id === 'peer-unverified')).toBe(false);
  });

  it('returns an empty list when the current status has no trusted owner fingerprint', () => {
    const status = createStatus({
      owner_id: null,
      owner_fingerprint: null,
      owner_fingerprint_verified: false,
      peers: [
        {
          id: 'peer-a',
          role: 'Worker',
          models: ['Llama-3.3-70B-Q4_K_M'],
          vram_gb: 24,
          hostname: 'alpha.local',
          owner_id: 'shared-owner',
          owner_fingerprint: 'fingerprint-shared-owner',
          owner_fingerprint_verified: true,
          owner_fingerprint_transitive: false,
          gpus: [{ name: 'RTX 4090', vram_bytes: 24_000_000_000 }],
        },
      ],
    });

    const { result } = renderHook(() => useOwnedNodes(status));

    expect(result.current).toEqual([]);
  });

  it('shortens enumerated NVIDIA peer hardware names while keeping raw names for tooltips', () => {
    const status = createStatus({
      my_is_soc: true,
      gpus: [{ name: 'Apple M4 Pro', vram_bytes: 24_000_000_000 }],
      peers: [
        {
          id: 'peer-gpu',
          role: 'Worker',
          models: ['Qwen3.5-0.8B-UD-Q8_K_XL'],
          vram_gb: 80,
          hostname: 'atlas.local',
          owner_id: 'shared-owner',
          owner_fingerprint: 'fingerprint-shared-owner',
          owner_fingerprint_verified: true,
          owner_fingerprint_transitive: false,
          gpus: [
            { name: 'NVIDIA GeForce RTX 5090', vram_bytes: 32_000_000_000 },
            { name: 'NVIDIA RTX 6000 Ada', vram_bytes: 48_000_000_000 },
          ],
        },
      ],
    });

    const { result } = renderHook(() => useOwnedNodes(status));

    expect(result.current[0]).toMatchObject({
      id: 'peer-gpu',
      hardwareLabel: 'GPU',
      gpuName: 'RTX 5090, RTX 6000 Ada',
      hardwareNames: ['NVIDIA GeForce RTX 5090', 'NVIDIA RTX 6000 Ada'],
    });
    expect(result.current[1]).toMatchObject({
      id: 'self-node-1234',
      hardwareLabel: 'SoC',
      gpuName: 'Apple M4 Pro',
    });
  });

  it('single-GPU node: separateCapable false, mixedGpuWarning false, gpuTargets length 1', () => {
    const status = createStatus({
      gpus: [{ name: 'RTX 6000 Ada', vram_bytes: 48_000_000_000 }],
    });

    const { result } = renderHook(() => useOwnedNodes(status));
    const self = result.current.find((n) => n.isSelf)!;

    expect(self.separateCapable).toBe(false);
    expect(self.mixedGpuWarning).toBe(false);
    expect(self.gpuTargets).toHaveLength(1);
    expect(self.aggregateVramGb).toBeCloseTo(self.vramGb);
  });

  it('dual homogenous GPU node: separateCapable true, mixedGpuWarning false, correct labels', () => {
    const status = createStatus({
      peers: [
        {
          id: 'peer-dual-homogenous',
          role: 'Worker',
          models: [],
          vram_gb: 48,
          hostname: 'alpha.local',
          owner_id: 'shared-owner',
          owner_fingerprint: 'fingerprint-shared-owner',
          owner_fingerprint_verified: true,
          owner_fingerprint_transitive: false,
          gpus: [
            { name: 'RTX 4090', vram_bytes: 24_000_000_000 },
            { name: 'RTX 4090', vram_bytes: 24_000_000_000 },
          ],
        },
      ],
    });

    const { result } = renderHook(() => useOwnedNodes(status));
    const node = result.current.find((n) => n.id === 'peer-dual-homogenous')!;

    expect(node.separateCapable).toBe(true);
    expect(node.mixedGpuWarning).toBe(false);
    expect(node.gpuTargets).toHaveLength(2);
    expect(node.gpuTargets[0].label).toBe('GPU 0 · RTX 4090 · 24.0 GB');
    expect(node.gpuTargets[1].label).toBe('GPU 1 · RTX 4090 · 24.0 GB');
  });

  it('dual mixed GPU node: separateCapable true, mixedGpuWarning true', () => {
    const status = createStatus({
      peers: [
        {
          id: 'peer-dual-mixed',
          role: 'Worker',
          models: [],
          vram_gb: 40,
          hostname: 'alpha.local',
          owner_id: 'shared-owner',
          owner_fingerprint: 'fingerprint-shared-owner',
          owner_fingerprint_verified: true,
          owner_fingerprint_transitive: false,
          gpus: [
            { name: 'RTX 4090', vram_bytes: 24_000_000_000 },
            { name: 'RTX 4080', vram_bytes: 16_000_000_000 },
          ],
        },
      ],
    });

    const { result } = renderHook(() => useOwnedNodes(status));
    const node = result.current.find((n) => n.id === 'peer-dual-mixed')!;

    expect(node.separateCapable).toBe(true);
    expect(node.mixedGpuWarning).toBe(true);
    expect(node.gpuTargets).toHaveLength(2);
  });

  it('legacy no-gpus node: separateCapable false, gpuTargets empty, aggregateVramGb from my_vram_gb', () => {
    const status = createStatus({
      gpus: undefined,
      my_vram_gb: 24,
    });

    const { result } = renderHook(() => useOwnedNodes(status));
    const self = result.current.find((n) => n.isSelf)!;

    expect(self.separateCapable).toBe(false);
    expect(self.gpuTargets).toHaveLength(0);
    expect(self.aggregateVramGb).toBeCloseTo(24);
  });

  it('derives node VRAM from reported GPU bytes before falling back to vram_gb', () => {
    const status = createStatus({
      my_vram_gb: 38.7,
      gpus: [{ name: 'Apple M4 Pro', vram_bytes: 24_000_000_000 }],
      peers: [
        {
          id: 'peer-vram',
          role: 'Host',
          models: ['Qwen3.5-0.8B-UD-Q8_K_XL'],
          vram_gb: 61.7,
          hostname: 'atlas.local',
          owner_id: 'shared-owner',
          owner_fingerprint: 'fingerprint-shared-owner',
          owner_fingerprint_verified: true,
          owner_fingerprint_transitive: false,
          gpus: [
            { name: 'NVIDIA GeForce RTX 5090', vram_bytes: 32_000_000_000 },
            { name: 'NVIDIA GeForce RTX 3080', vram_bytes: 10_000_000_000 },
          ],
        },
      ],
    });

    const { result } = renderHook(() => useOwnedNodes(status));

    expect(result.current[0]).toMatchObject({
      id: 'peer-vram',
      vramGb: 42,
    });
    expect(result.current[1]).toMatchObject({
      id: 'self-node-1234',
      vramGb: 24,
    });
  });
});
