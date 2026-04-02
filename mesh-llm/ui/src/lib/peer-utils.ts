/**
 * Shared peer normalization utilities
 */

function shortNodeId(id: string) {
  return id.length > 10 ? `${id.slice(0, 6)}…${id.slice(-4)}` : id;
}

export function normalizeHostname(hostname: string | undefined, id: string) {
  const trimmed = hostname?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : shortNodeId(id);
}

export function normalizeModels(values: Array<string | null | undefined>) {
  return Array.from(new Set(values.map((value) => value?.trim()).filter((value): value is string => Boolean(value))));
}
