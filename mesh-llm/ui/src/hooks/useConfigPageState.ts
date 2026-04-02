import { useMemo } from "react";
import type { StatusPayload } from "../App";
import type { OwnedNode, NodeStatusTone } from "./useOwnedNodes";
import { getAssignmentId, getSplitGroupId } from "../pages/config/configReducer";
import { validateSplits } from "../lib/config";
import { aggregateModels } from "../lib/models";
import type { AggregatedModel, ModelCatalogMeshModel, ModelCatalogPeer, ModelSizeTuple } from "../lib/models";
import { normalizeHostname, normalizeModels } from "../lib/peer-utils";
import { estimateAssignmentBreakdownBytes, estimateAssignmentSizeBytes } from "../lib/vram";
import { collectSelectedAssignmentIds, splitLayerCount } from "../lib/configSplitOps";
import {
  catalogModelName,
  configDerivedStatus,
  formatGb,
  modelNamesFromSizes,
  trustedOwnerFingerprint,
  vramBytesFromStatus,
} from "../lib/configPageHelpers";
import type { MeshConfig, ModelAssignment, ScannedModel, ScannedModelMetadata } from "../types/config";
import type { VramAssignment } from "../components/config/VramContainer";
import type { NodeAssignTarget } from "../components/config/CatalogContextMenu";

type CatalogPeerPayload = StatusPayload["peers"][number] & {
  model_sizes?: ModelSizeTuple[] | null;
  model_scans?: ScannedModel[];
};

type CatalogStatusPayload = StatusPayload & {
  model_sizes?: ModelSizeTuple[];
  mesh_models: ModelCatalogMeshModel[];
  peers: CatalogPeerPayload[];
  model_scans?: ScannedModel[];
};

type OwnedCatalogPeer = ModelCatalogPeer & {
  hostname: string;
  vramBytes: number;
};

type AssignmentWithUiError = ModelAssignment & { _errorMessage?: string };

export function useConfigPageState({
  status,
  config,
  selectedNodeId,
  selectedAssignmentId,
  ownedNodes,
  assignmentErrors,
  loadError,
  tomlParseError,
}: {
  status: StatusPayload | null;
  config: MeshConfig;
  selectedNodeId: string | null;
  selectedAssignmentId: string | null;
  ownedNodes: OwnedNode[];
  assignmentErrors: Record<string, string>;
  loadError: string | null;
  tomlParseError: string | null;
}) {
  const trustedOwnerFingerprintValue = useMemo(
    () => trustedOwnerFingerprint(status),
    [status],
  );

  const modelScansByNodeAndName = useMemo(() => {
    const ownerAwareStatus = status as CatalogStatusPayload | null;
    const nodeMap = new Map<string, Map<string, ScannedModel>>();

    const appendScans = (
      nodeId: string,
      scans: ScannedModel[] | null | undefined,
    ) => {
      if (!scans || scans.length === 0) return;
      const byName = new Map<string, ScannedModel>();
      for (const scan of scans) {
        byName.set(scan.name, scan);
      }
      nodeMap.set(nodeId, byName);
    };

    if (ownerAwareStatus) {
      appendScans(ownerAwareStatus.node_id, ownerAwareStatus.model_scans);
      for (const peer of ownerAwareStatus.peers) {
        appendScans(peer.id, peer.model_scans);
      }
    }

    return nodeMap;
  }, [status]);

  const ownedModelPeers = useMemo<OwnedCatalogPeer[]>(() => {
    const ownerAwareStatus = status as CatalogStatusPayload | null;
    const ownerFingerprint = trustedOwnerFingerprintValue;

    if (!ownerAwareStatus || !ownerFingerprint) return [];

    const peers: OwnedCatalogPeer[] = [
      {
        id: ownerAwareStatus.node_id,
        hostname: normalizeHostname(
          ownerAwareStatus.my_hostname,
          ownerAwareStatus.node_id,
        ),
        vramBytes: vramBytesFromStatus(
          ownerAwareStatus.my_vram_gb,
          ownerAwareStatus.gpus,
        ),
        models: normalizeModels([
          catalogModelName(ownerAwareStatus.model_name),
          ...(ownerAwareStatus.serving_models ?? []),
          ...modelNamesFromSizes(ownerAwareStatus.model_sizes),
        ]),
        model_sizes: ownerAwareStatus.model_sizes,
        mesh_models: ownerAwareStatus.mesh_models,
      },
    ];

    for (const peer of ownerAwareStatus.peers) {
      if (trustedOwnerFingerprint(peer) !== ownerFingerprint) continue;

      peers.push({
        id: peer.id,
        hostname: normalizeHostname(peer.hostname, peer.id),
        vramBytes: vramBytesFromStatus(peer.vram_gb, peer.gpus),
        models: normalizeModels([
          ...peer.models,
          ...modelNamesFromSizes(peer.model_sizes),
        ]),
        model_sizes: peer.model_sizes,
      });
    }

    return peers.sort(
      (a, b) =>
        a.hostname.localeCompare(b.hostname, undefined, {
          sensitivity: "base",
          numeric: true,
        }) || a.id.localeCompare(b.id),
    );
  }, [status, trustedOwnerFingerprintValue]);

  const ownershipNotice = useMemo(() => {
    const ownerAwareStatus = status as CatalogStatusPayload | null;
    if (!ownerAwareStatus) return null;

    const untrustedMatchingPeerCount = ownerAwareStatus.peers.filter((peer) => {
      const peerFingerprint = peer.owner_fingerprint?.trim();
      if (!peerFingerprint) return false;
      if (peerFingerprint !== ownerAwareStatus.owner_fingerprint?.trim()) return false;
      return peer.owner_fingerprint_verified !== true || peer.owner_fingerprint_transitive === true;
    }).length;

    if (!trustedOwnerFingerprintValue) {
      return {
        tone: "warning" as const,
        title: "Configuration is read-only until you claim your nodes",
        description:
          ownerAwareStatus.owner_fingerprint?.trim()
            ? "This console does not have a directly verified owner fingerprint yet. Start mesh-llm with your owner key on this node, then rejoin peers so verified ownership can unlock configuration."
            : "This console is running without a trusted owner fingerprint. Start mesh-llm with your owner key to claim this node before editing owned-node placement.",
      };
    }

    if (untrustedMatchingPeerCount > 0) {
      return {
        tone: "info" as const,
        title:
          untrustedMatchingPeerCount === 1
            ? "1 peer stays read-only until ownership is verified"
            : `${untrustedMatchingPeerCount} peers stay read-only until ownership is verified`,
        description:
          "Only peers with a directly verified owner fingerprint appear here as configurable owned nodes. Unverified or transitive ownership claims stay visible in status, but they are excluded from placement changes.",
      };
    }

    return null;
  }, [status, trustedOwnerFingerprintValue]);

  const nodeConfigLookup = useMemo(
    () => new Map(config.nodes.map((node) => [node.node_id, node])),
    [config],
  );

  const configAwareNodes = useMemo(
    () =>
      ownedNodes.map((node) => {
        const nodeModels = nodeConfigLookup.get(node.id)?.models ?? [];
        const { label, tone } = configDerivedStatus(node, nodeModels);
        return { ...node, statusLabel: label, statusTone: tone };
      }),
    [ownedNodes, nodeConfigLookup],
  );

  const selectedNode = useMemo<OwnedNode | null>(
    () => configAwareNodes.find((node) => node.id === selectedNodeId) ?? null,
    [configAwareNodes, selectedNodeId],
  );

  const selectedCatalogNode = useMemo(
    () => ownedModelPeers.find((node) => node.id === selectedNodeId) ?? null,
    [ownedModelPeers, selectedNodeId],
  );

  const modelSizeLookup = useMemo(() => {
    const models = aggregateModels(ownedModelPeers);
    const map = new Map<string, AggregatedModel>();
    for (const model of models) {
      map.set(model.name, model);
      map.set(model.name.toLowerCase(), model);
    }
    return map;
  }, [ownedModelPeers]);

  const advertisedModelKeysByNodeAndName = useMemo(() => {
    const ownerAwareStatus = status as CatalogStatusPayload | null;
    const nodeMap = new Map<string, Map<string, Set<string>>>();

    const appendScans = (
      nodeId: string,
      scans: ScannedModel[] | null | undefined,
    ) => {
      if (!scans || scans.length === 0) return;

      const byName = nodeMap.get(nodeId) ?? new Map<string, Set<string>>();
      for (const scan of scans) {
        const keys = byName.get(scan.name) ?? new Set<string>();
        keys.add(scan.model_key);
        byName.set(scan.name, keys);
      }
      nodeMap.set(nodeId, byName);
    };

    if (ownerAwareStatus) {
      appendScans(ownerAwareStatus.node_id, ownerAwareStatus.model_scans);
      for (const peer of ownerAwareStatus.peers) {
        appendScans(peer.id, peer.model_scans);
      }
    }

    return nodeMap;
  }, [status]);

  const advertisedModelsByNode = useMemo(
    () =>
      new Map(ownedModelPeers.map((peer) => [peer.id, new Set(peer.models)])),
    [ownedModelPeers],
  );

  const vramAssignmentsByNode = useMemo(() => {
    const map = new Map<string, VramAssignment[]>();
    for (const node of ownedNodes) {
      const nodeConfig = nodeConfigLookup.get(node.id);
      const totalVramBytes = ownedModelPeers.find((peer) => peer.id === node.id)?.vramBytes
        ?? Math.round(node.vramGb * 1e9);
      let assignedBytes = 0;
      const assignments = (nodeConfig?.models ?? []).map((assignment) => {
        const model = modelSizeLookup.get(assignment.name);
        const metadata = modelScansByNodeAndName.get(node.id)?.get(assignment.name)?.metadata ?? null;
        const breakdown = estimateAssignmentBreakdownBytes(
          model?.sizeBytes ?? 0,
          assignment.ctx_size,
          metadata,
        );
        const fullSizeBytes = estimateAssignmentSizeBytes(
          model?.sizeBytes ?? 0,
          assignment.ctx_size,
          metadata,
        );
        const split = assignment.split;
        const sizeBytes = split
          ? Math.round(fullSizeBytes * (splitLayerCount(split) / split.total))
          : fullSizeBytes;
        const assignmentId = getAssignmentId(assignment);
        const overcommitBytes = Math.max(0, assignedBytes + sizeBytes - totalVramBytes);
        const invalidMessage = overcommitBytes > 0
          ? `Exceeds available VRAM by ${formatGb(overcommitBytes)}`
          : undefined;

        assignedBytes += sizeBytes;

        return {
          id: assignmentId,
          name: assignment.name,
          sizeBytes,
          fullSizeBytes,
          weightsBytes: breakdown.weightsBytes,
          contextBytes: breakdown.contextBytes,
          sizeGb: sizeBytes / 1e9,
          moeExperts: assignment.moe_experts,
          ctxSize: assignment.ctx_size,
          errorMessage: assignmentErrors[assignmentId],
          invalidMessage,
          model_key: assignment.model_key ?? null,
          split: assignment.split ?? null,
        } satisfies VramAssignment;
      });
      map.set(node.id, assignments);
    }
    return map;
  }, [assignmentErrors, modelScansByNodeAndName, modelSizeLookup, nodeConfigLookup, ownedModelPeers, ownedNodes]);

  const assignedBytesByNode = useMemo(
    () =>
      new Map(
        Array.from(vramAssignmentsByNode.entries()).map(
          ([nodeId, assignments]) => [
            nodeId,
            assignments.reduce(
              (sum, assignment) => sum + assignment.sizeBytes,
              0,
            ),
          ],
        ),
      ),
    [vramAssignmentsByNode],
  );

  const totalVramByNode = useMemo(
    () => new Map(ownedModelPeers.map((peer) => [peer.id, peer.vramBytes])),
    [ownedModelPeers],
  );

  const catalogAssignTargets = useMemo<NodeAssignTarget[]>(
    () =>
      ownedModelPeers.map((peer) => ({
        id: peer.id,
        hostname: peer.hostname ?? peer.id,
        vramBytes: peer.vramBytes,
        assignedBytes: assignedBytesByNode.get(peer.id) ?? 0,
        assignedModelNames: new Set(
          nodeConfigLookup.get(peer.id)?.models.map((m) => m.name) ?? [],
        ),
      })),
    [ownedModelPeers, assignedBytesByNode, nodeConfigLookup],
  );

  const hasInvalidAssignments = useMemo(
    () =>
      Array.from(vramAssignmentsByNode.values()).some((assignments) =>
        assignments.some((assignment) => Boolean(assignment.invalidMessage)),
      ),
    [vramAssignmentsByNode],
  );

  const allDropTargetsOvercommitted = useMemo(() => {
    const containers: { assignedBytes: number; capacityBytes: number }[] = [];

    for (const node of ownedNodes) {
      const assignments = vramAssignmentsByNode.get(node.id) ?? [];
      const placementMode = nodeConfigLookup.get(node.id)?.placement_mode ?? 'pooled';
      const nodeTotalVramBytes =
        ownedModelPeers.find((peer) => peer.id === node.id)?.vramBytes
        ?? Math.round(node.vramGb * 1e9);

      if (placementMode === 'separate' && node.gpuTargets.length > 0) {
        for (const gpu of node.gpuTargets) {
          const gpuAssignedBytes = assignments
            .filter((a) => a.id.match(new RegExp(`::gpu-${gpu.index}$`)))
            .reduce((sum, a) => sum + a.sizeBytes, 0);
          containers.push({ assignedBytes: gpuAssignedBytes, capacityBytes: gpu.vramBytes });
        }
      } else {
        const assignedBytes = assignments.reduce((sum, a) => sum + a.sizeBytes, 0);
        containers.push({ assignedBytes, capacityBytes: nodeTotalVramBytes });
      }
    }

    if (containers.length === 0) return false;
    return containers.every((c) => c.assignedBytes > c.capacityBytes);
  }, [nodeConfigLookup, ownedModelPeers, ownedNodes, vramAssignmentsByNode]);

  const splitValidationErrors = useMemo(() => validateSplits(config.nodes), [config]);

  const hasAssignmentErrors = Object.keys(assignmentErrors).length > 0;

  const invalidReason =
    loadError
    ?? tomlParseError
    ?? Object.values(assignmentErrors)[0]
    ?? Array.from(vramAssignmentsByNode.values())
      .flat()
      .find((assignment) => assignment.invalidMessage)?.invalidMessage
    ?? splitValidationErrors[0]?.message
    ?? null;

  const isConfigValid =
    !loadError
    && !tomlParseError
    && !hasAssignmentErrors
    && !hasInvalidAssignments
    && splitValidationErrors.length === 0;

  const selectedAssignment = useMemo<AssignmentWithUiError | null>(() => {
    if (!selectedAssignmentId) return null;

    for (const node of config.nodes) {
      const assignment = node.models.find(
        (model) => getAssignmentId(model) === selectedAssignmentId,
      );
      if (!assignment) continue;
      const errorMessage = assignmentErrors[getAssignmentId(assignment)];
      return errorMessage
        ? { ...assignment, _errorMessage: errorMessage }
        : assignment;
    }

    return null;
  }, [assignmentErrors, config, selectedAssignmentId]);

  const selectedModelName = selectedAssignment?.name ?? null;

  const selectedAssignmentIds = useMemo(
    () => collectSelectedAssignmentIds(config, selectedAssignmentId),
    [config, selectedAssignmentId],
  );

  const nodeModelScansLookup = useMemo(() => {
    const map = new Map<string, Map<string, ScannedModelMetadata>>();
    for (const [nodeId, scans] of modelScansByNodeAndName.entries()) {
      map.set(
        nodeId,
        new Map(
          Array.from(scans.entries()).map(([name, scan]) => [
            name,
            scan.metadata,
          ]),
        ),
      );
    }
    return map;
  }, [modelScansByNodeAndName]);

  const maxCtxByModel = useMemo(() => {
    const map = new Map<string, number>();
    for (const scans of modelScansByNodeAndName.values()) {
      for (const [modelName, scan] of scans.entries()) {
        const ctxLen = scan.metadata?.context_length;
        if (ctxLen != null && ctxLen > 0) {
          const existing = map.get(modelName);
          if (existing == null || ctxLen < existing) {
            map.set(modelName, ctxLen);
          }
        }
      }
    }
    return map;
  }, [modelScansByNodeAndName]);

  const nodeModelKeyLookup = useMemo(() => {
    const map = new Map<string, Map<string, string>>();
    for (const [nodeId, scans] of modelScansByNodeAndName.entries()) {
      map.set(
        nodeId,
        new Map(
          Array.from(scans.entries()).map(([name, scan]) => [
            name,
            scan.model_key,
          ]),
        ),
      );
    }
    return map;
  }, [modelScansByNodeAndName]);

  return {
    trustedOwnerFingerprintValue,
    modelScansByNodeAndName,
    ownedModelPeers,
    ownershipNotice,
    nodeConfigLookup,
    configAwareNodes,
    selectedNode,
    selectedCatalogNode,
    modelSizeLookup,
    advertisedModelKeysByNodeAndName,
    advertisedModelsByNode,
    vramAssignmentsByNode,
    assignedBytesByNode,
    totalVramByNode,
    catalogAssignTargets,
    hasInvalidAssignments,
    allDropTargetsOvercommitted,
    splitValidationErrors,
    hasAssignmentErrors,
    invalidReason,
    isConfigValid,
    selectedAssignment,
    selectedModelName,
    selectedAssignmentIds,
    nodeModelScansLookup,
    maxCtxByModel,
    nodeModelKeyLookup,
  };
}
