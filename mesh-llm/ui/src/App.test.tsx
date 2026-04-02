import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

type ConfigPageProps = {
  status: unknown;
  onRefreshStatus?: () => Promise<void> | void;
};

const configPageSpy = vi.fn<(props: ConfigPageProps) => void>();

vi.mock('./pages/ConfigPage', () => ({
  ConfigPage: (props: ConfigPageProps) => {
    configPageSpy(props);
    return <div data-testid="config-page-mock" />;
  },
}));

import { App } from './App';

const statusPayload = {
  node_id: 'node-a',
  token: 'token-1',
  node_status: 'Serving',
  is_host: true,
  is_client: false,
  llama_ready: true,
  model_name: 'auto',
  api_port: 3131,
  my_vram_gb: 24,
  model_size_gb: 0,
  peers: [],
  mesh_models: [],
  inflight_requests: 0,
};

class EventSourceMock {
  onopen: ((this: EventSource, ev: Event) => unknown) | null = null;
  onmessage: ((this: EventSource, ev: MessageEvent) => unknown) | null = null;
  onerror: ((this: EventSource, ev: Event) => unknown) | null = null;

  constructor(public url: string) {}

  close() {}
}

describe('App config route regression', () => {
  beforeEach(() => {
    configPageSpy.mockClear();
    window.history.pushState({}, '', '/config');

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => statusPayload,
      }),
    );

    vi.stubGlobal('EventSource', EventSourceMock);

    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: vi.fn().mockImplementation(() => ({
        matches: false,
        media: '(prefers-color-scheme: dark)',
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders the config route and passes a callable refresh callback to ConfigPage', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId('config-page-mock')).toBeInTheDocument();
    });

    expect(configPageSpy).toHaveBeenCalled();
    const props = configPageSpy.mock.calls[configPageSpy.mock.calls.length - 1]?.[0];
    expect(props?.onRefreshStatus).toEqual(expect.any(Function));

    vi.mocked(fetch).mockClear();
    await props?.onRefreshStatus?.();

    expect(fetch).toHaveBeenCalledWith('/api/status');
  });
});
