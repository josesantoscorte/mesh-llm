export type ModelSizeTuple = readonly [string, number];

export type AggregatedModelMoe = {
  nExpert: number;
  nExpertUsed: number;
  minExpertsPerNode: number;
};

export type ModelCatalogMeshModel = {
  name: string;
  size_gb?: number;
  n_expert?: number | null;
  n_expert_used?: number | null;
  min_experts_per_node?: number | null;
  moe?: {
    n_expert?: number | null;
    n_expert_used?: number | null;
    min_experts_per_node?: number | null;
  } | null;
};

export type ModelCatalogPeer = {
  id: string;
  hostname?: string;
  models: string[];
  model_sizes?: ReadonlyArray<ModelSizeTuple> | null;
  mesh_models?: ReadonlyArray<ModelCatalogMeshModel> | null;
};

export interface AggregatedModel {
  name: string;
  sizeBytes: number;
  sizeGb: number;
  nodeIds: string[];
  moe?: AggregatedModelMoe;
}

function normalizeModelName(name: string | null | undefined) {
  const trimmed = name?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : null;
}

function bytesFromSizeGb(sizeGb: number | undefined) {
  if (sizeGb == null || !Number.isFinite(sizeGb) || sizeGb <= 0) return null;
  return Math.round(sizeGb * 1e9);
}

function toMoeInfo(meshModel: ModelCatalogMeshModel): AggregatedModelMoe | undefined {
  const source = meshModel.moe ?? meshModel;
  const nExpert = source.n_expert;
  const nExpertUsed = source.n_expert_used;
  const minExpertsPerNode = source.min_experts_per_node;

  if (
    nExpert == null ||
    nExpertUsed == null ||
    minExpertsPerNode == null ||
    !Number.isFinite(nExpert) ||
    !Number.isFinite(nExpertUsed) ||
    !Number.isFinite(minExpertsPerNode)
  ) {
    return undefined;
  }

  return {
    nExpert,
    nExpertUsed,
    minExpertsPerNode,
  };
}

export function aggregateModels(peers: ModelCatalogPeer[]): AggregatedModel[] {
  const aggregated = new Map<string, AggregatedModel>();

  for (const peer of peers) {
    const localSizes = new Map<string, number>();
    const localMoe = new Map<string, AggregatedModelMoe>();
    const displayNames = new Map<string, string>();

    for (const [rawName, rawSize] of peer.model_sizes ?? []) {
      const name = normalizeModelName(rawName);
      if (!name || !Number.isFinite(rawSize) || rawSize <= 0) continue;
      const key = name.toLowerCase();
      localSizes.set(key, Math.max(localSizes.get(key) ?? 0, rawSize));
      if (!displayNames.has(key) || (displayNames.get(key) === key && name !== key)) {
        displayNames.set(key, name);
      }
    }

    for (const meshModel of peer.mesh_models ?? []) {
      const name = normalizeModelName(meshModel.name);
      if (!name) continue;
      const key = name.toLowerCase();

      const sizeBytes = bytesFromSizeGb(meshModel.size_gb);
      if (sizeBytes != null) {
        localSizes.set(key, Math.max(localSizes.get(key) ?? 0, sizeBytes));
      }
      if (!displayNames.has(key) || (displayNames.get(key) === key && name !== key)) {
        displayNames.set(key, name);
      }

      const moe = toMoeInfo(meshModel);
      if (moe) {
        localMoe.set(key, moe);
      }
    }

    const modelKeys = new Set<string>();

    for (const rawName of peer.models) {
      const name = normalizeModelName(rawName);
      if (name) {
        const key = name.toLowerCase();
        modelKeys.add(key);
        if (!displayNames.has(key)) {
          displayNames.set(key, name);
        }
      }
    }

    for (const key of localSizes.keys()) {
      modelKeys.add(key);
    }

    for (const key of localMoe.keys()) {
      modelKeys.add(key);
    }

    for (const key of modelKeys) {
      const displayName = displayNames.get(key) ?? key;
      const existing = aggregated.get(key) ?? {
        name: displayName,
        sizeBytes: 0,
        sizeGb: 0,
        nodeIds: [],
      };

      if (existing.name.toLowerCase() === existing.name && displayName.toLowerCase() !== displayName) {
        existing.name = displayName;
      }

      if (!existing.nodeIds.includes(peer.id)) {
        existing.nodeIds.push(peer.id);
      }

      const sizeBytes = localSizes.get(key);
      if (sizeBytes != null && sizeBytes > existing.sizeBytes) {
        existing.sizeBytes = sizeBytes;
        existing.sizeGb = sizeBytes / 1e9;
      }

      const moe = localMoe.get(key);
      if (moe) {
        existing.moe = moe;
      }

      aggregated.set(key, existing);
    }
  }

  return [...aggregated.values()]
    .map((model) => ({
      ...model,
      nodeIds: [...model.nodeIds].sort((a, b) => a.localeCompare(b)),
    }))
    .sort((a, b) => (b.nodeIds.length - a.nodeIds.length) || a.name.localeCompare(b.name));
}
