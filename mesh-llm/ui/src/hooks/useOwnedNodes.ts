import { useMemo } from 'react';

import type { StatusPayload } from '../App';
import {
  formatHardwareNames,
  gpuTargets,
  hardwareVramGb,
  isHomogenousGpuSet,
  type GpuTarget,
  type ReportedGpu,
} from '../lib/hardware';
import { trustedOwnerFingerprint } from '../lib/configPageHelpers';
import { normalizeHostname, normalizeModels } from '../lib/peer-utils';

export type NodeStatusTone = 'serving' | 'host' | 'worker' | 'client' | 'idle';

export type OwnedNode = {
  id: string;
  hostname: string;
  role: string;
  gpuName: string;
  hardwareLabel: 'GPU' | 'SoC';
  hardwareNames: string[];
  vramGb: number;
  models: string[];
  statusLabel: string;
  statusTone: NodeStatusTone;
  isSelf: boolean;
  gpuTargets: GpuTarget[];
  aggregateVramGb: number;
  separateCapable: boolean;
  mixedGpuWarning: boolean;
};

function normalizeHardware(
  gpus: ReportedGpu[] | undefined,
  isSoc: boolean | undefined,
) {
  const hardwareLabel: OwnedNode['hardwareLabel'] = isSoc ? 'SoC' : 'GPU';
  const hardwareNames = formatHardwareNames(gpus);
  const shortenedHardwareNames = isSoc
    ? hardwareNames
    : formatHardwareNames(gpus, { shorten: true });
  const fallbackName = isSoc ? 'SoC not reported' : 'GPU not reported';

  return {
    hardwareLabel,
    hardwareNames,
    gpuName:
      shortenedHardwareNames.length > 0
        ? shortenedHardwareNames.join(', ')
        : fallbackName,
  };
}

function selfStatus(status: StatusPayload): { label: string; tone: NodeStatusTone } {
  const label = status.node_status || (status.is_client ? 'Client' : status.is_host ? 'Host' : 'Idle');
  if (label === 'Serving' || label === 'Serving (split)') return { label, tone: 'serving' };
  if (label === 'Client' || status.is_client) return { label: 'Client', tone: 'client' };
  if (label === 'Worker') return { label, tone: 'worker' };
  if (label === 'Host' || status.is_host) return { label: 'Host', tone: 'host' };
  return { label, tone: 'idle' };
}

function peerStatus(peer: StatusPayload['peers'][number]): { label: string; tone: NodeStatusTone } {
  if (peer.serving_models && peer.serving_models.length > 0) {
    return { label: 'Serving', tone: 'serving' };
  }
  if (peer.role === 'Host') return { label: 'Host', tone: 'host' };
  if (peer.role === 'Worker') return { label: 'Worker', tone: 'worker' };
  if (peer.role === 'Client') return { label: 'Client', tone: 'client' };
  return { label: peer.role || 'Idle', tone: 'idle' };
}

function sortByHostname(a: OwnedNode, b: OwnedNode) {
  return a.hostname.localeCompare(b.hostname, undefined, { sensitivity: 'base', numeric: true }) || a.id.localeCompare(b.id);
}

export function useOwnedNodes(status: StatusPayload | null) {
  return useMemo<OwnedNode[]>(() => {
    const ownerFingerprint = trustedOwnerFingerprint(status);

    if (!status || !ownerFingerprint) return [];

    const self = selfStatus(status);
    const selfHardware = normalizeHardware(status.gpus, status.my_is_soc);
    const selfGpuTargets = gpuTargets(status.gpus);
    const selfVramGb = hardwareVramGb(status.my_vram_gb, status.gpus);
    const nodes: OwnedNode[] = [
      {
        id: status.node_id,
        hostname: normalizeHostname(status.my_hostname, status.node_id),
        role: status.is_client ? 'Client' : status.is_host ? 'Host' : 'Worker',
        gpuName: selfHardware.gpuName,
        hardwareLabel: selfHardware.hardwareLabel,
        hardwareNames: selfHardware.hardwareNames,
        vramGb: selfVramGb,
        models: normalizeModels(
          status.serving_models && status.serving_models.length > 0
            ? status.serving_models
            : [status.model_name],
        ),
        statusLabel: self.label,
        statusTone: self.tone,
        isSelf: true,
        gpuTargets: selfGpuTargets,
        aggregateVramGb: selfVramGb,
        separateCapable: selfGpuTargets.length >= 2,
        mixedGpuWarning: selfGpuTargets.length >= 2 && !isHomogenousGpuSet(status.gpus),
      },
    ];

    for (const peer of status.peers) {
      if (trustedOwnerFingerprint(peer) !== ownerFingerprint) continue;
      const peerState = peerStatus(peer);
      const peerHardware = normalizeHardware(peer.gpus, peer.is_soc);
      const peerGpuTargets = gpuTargets(peer.gpus);
      const peerVramGb = hardwareVramGb(peer.vram_gb, peer.gpus);
      nodes.push({
        id: peer.id,
        hostname: normalizeHostname(peer.hostname, peer.id),
        role: peer.role,
        gpuName: peerHardware.gpuName,
        hardwareLabel: peerHardware.hardwareLabel,
        hardwareNames: peerHardware.hardwareNames,
        vramGb: peerVramGb,
        models: normalizeModels(
          peer.serving_models && peer.serving_models.length > 0 ? peer.serving_models : peer.models,
        ),
        statusLabel: peerState.label,
        statusTone: peerState.tone,
        isSelf: false,
        gpuTargets: peerGpuTargets,
        aggregateVramGb: peerVramGb,
        separateCapable: peerGpuTargets.length >= 2,
        mixedGpuWarning: peerGpuTargets.length >= 2 && !isHomogenousGpuSet(peer.gpus),
      });
    }

    return nodes.sort(sortByHostname);
  }, [status]);
}
