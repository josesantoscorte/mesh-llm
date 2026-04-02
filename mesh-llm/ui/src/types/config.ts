export interface MeshConfig {
  version: number;
  nodes: NodeConfig[];
}

export type PlacementMode = 'pooled' | 'separate';

export interface NodeConfig {
  node_id: string;
  hostname?: string;
  placement_mode?: PlacementMode;
  models: ModelAssignment[];
}

export interface ModelAssignment {
  name: string;
  model_key?: string;
  split?: ModelSplit;
  path?: string;
  ctx_size?: number;
  moe_experts?: number;
  gpu_index?: number;
}

export interface ModelSplit {
  start: number;
  end: number;
  total: number;
}

/** Compact GGUF metadata shipped by backend in /api/status → model_scans */
export interface ScannedModelMetadata {
  architecture?: string;
  context_length?: number;
  embedding_length?: number;
  quantization_type?: string;
  attention?: {
    head_count?: number;
    head_count_kv?: number;
    key_length?: number;
    value_length?: number;
  };
  total_layers?: number;
  total_offloadable_layers?: number;
  file_size?: number;
  dense_split_capable?: boolean;
  experts?: {
    expert_count?: number;
    expert_used_count?: number;
    expert_shared_count?: number;
    expert_group_count?: number;
  };
  rope?: {
    kind?: string;
    factor?: number;
    freq_base?: number;
    original_context_length?: number;
  };
  tokenizer?: {
    model?: string;
    pre?: string;
    vocab_size?: number;
  };
}

export interface ScannedModel {
  name: string;
  model_key: string;
  size_bytes: number;
  metadata: ScannedModelMetadata;
}
