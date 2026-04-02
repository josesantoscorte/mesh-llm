export type ConfigValidationError = { code: string; path: string; message: string };

export async function broadcastConfig(
  config: string,
): Promise<{ ok: boolean; saved?: number; total?: number; failed?: string[]; error?: string; errors?: ConfigValidationError[] }> {
  try {
    const resp = await fetch('/api/config/broadcast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/toml' },
      body: config,
    });
    if (!resp.ok) {
      const text = await resp.text();
      try {
        const body = JSON.parse(text) as { error?: string; errors?: ConfigValidationError[] };
        return { ok: false, error: body.error ?? `Failed to save config (${resp.status})`, errors: body.errors };
      } catch {
        return { ok: false, error: text || `Failed to save config (${resp.status})` };
      }
    }
    const body = await resp.json();
    return {
      ok: body.failed.length === 0,
      saved: body.saved,
      total: body.total,
      failed: body.failed,
    };
  } catch (e) {
    return { ok: false, error: `Network error: ${(e as Error).message}` };
  }
}

export async function fetchAuthoredConfig(): Promise<{ ok: boolean; config?: string; error?: string }> {
  try {
    const resp = await fetch('/api/config');
    const body = await resp.text();
    if (!resp.ok) {
      return {
        ok: false,
        error: body || `Failed to load config (${resp.status})`,
      };
    }
    return { ok: true, config: body };
  } catch (e) {
    return { ok: false, error: `Network error: ${(e as Error).message}` };
  }
}

export async function broadcastScan(): Promise<{
  ok: boolean;
  refreshed?: number;
  total?: number;
  failed?: string[];
}> {
  try {
    const resp = await fetch('/api/scan/broadcast', { method: 'POST' });
    if (!resp.ok) {
      return { ok: false };
    }
    const body = await resp.json();
    return {
      ok: body.failed.length === 0,
      refreshed: body.refreshed,
      total: body.total,
      failed: body.failed,
    };
  } catch {
    return { ok: false };
  }
}
