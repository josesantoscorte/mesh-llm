import type { NodeStatusTone, OwnedNode } from "../hooks/useOwnedNodes";
import { getSplitGroupId } from "../pages/config/configReducer";
import type { MeshConfig, ModelAssignment } from "../types/config";
import type { ModelSizeTuple } from "./models";

export type OwnerFingerprintFields = {
  owner_fingerprint?: string | null;
  owner_fingerprint_verified?: boolean | null;
  owner_fingerprint_transitive?: boolean | null;
};

export function formatGb(bytes: number) {
  const gb = bytes / 1e9;
  return `${gb >= 100 ? Math.round(gb) : gb.toFixed(1)} GB`;
}

export function trustedOwnerFingerprint(owner: OwnerFingerprintFields | null | undefined) {
  const fingerprint = owner?.owner_fingerprint?.trim();
  if (!fingerprint) return null;
  if (owner?.owner_fingerprint_verified !== true) return null;
  if (owner?.owner_fingerprint_transitive === true) return null;
  return fingerprint;
}

export function modelNamesFromSizes(sizes: ModelSizeTuple[] | null | undefined) {
  return (sizes ?? []).map(([name]) => name);
}

const STATUS_ONLY_MODEL_NAMES = new Set(["(idle)", "(client)", "(standby)"]);

export function catalogModelName(name: string | null | undefined) {
  const trimmed = name?.trim();
  if (!trimmed || STATUS_ONLY_MODEL_NAMES.has(trimmed)) {
    return null;
  }
  return trimmed;
}

export function configDerivedStatus(
  node: OwnedNode,
  nodeModels: ModelAssignment[],
): { label: string; tone: NodeStatusTone } {
  if (node.role === "Client") return { label: "Client", tone: "client" };
  if (nodeModels.length === 0) return { label: "Standby", tone: "idle" };
  const hasSplit = nodeModels.some((m) => m.split != null);
  if (hasSplit) return { label: "Serving (split)", tone: "serving" };
  return { label: "Serving", tone: "serving" };
}

export function getCrossNodeSplitGroupIds(config: MeshConfig): Set<string> {
  const splitGroupNodeCount = new Map<string, Set<string>>();

  for (const node of config.nodes) {
    for (const model of node.models) {
      const groupId = getSplitGroupId(model);
      if (!groupId) continue;

      if (!splitGroupNodeCount.has(groupId)) {
        splitGroupNodeCount.set(groupId, new Set());
      }
      splitGroupNodeCount.get(groupId)!.add(node.node_id);
    }
  }

  const crossNodeGroups = new Set<string>();
  for (const [groupId, nodeIds] of splitGroupNodeCount.entries()) {
    if (nodeIds.size > 1) {
      crossNodeGroups.add(groupId);
    }
  }

  return crossNodeGroups;
}
