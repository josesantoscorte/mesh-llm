import type { ModelSplit } from '../types/config';

export function clampResizeBoundaryStart(left: ModelSplit, right: ModelSplit, boundaryStart: number): number {
  const minBoundaryStart = left.start === 0 ? left.start + 2 : left.start + 1;
  const maxBoundaryStart = right.end === right.total - 1 ? right.end - 1 : right.end;

  return Math.max(minBoundaryStart, Math.min(maxBoundaryStart, boundaryStart));
}
