import type { ModelSplit } from '../types/config';

const MIN_SPLIT_WIDTH = 2;

export function clampResizeBoundaryStart(left: ModelSplit, right: ModelSplit, boundaryStart: number): number {
  const minBoundaryStart = left.start + MIN_SPLIT_WIDTH;
  const maxBoundaryStart = right.end - MIN_SPLIT_WIDTH;

  return Math.max(minBoundaryStart, Math.min(maxBoundaryStart, boundaryStart));
}
