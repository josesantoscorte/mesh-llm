use super::super::{quant_params_for, quantize_stacked_weights, ModelConfig, TensorPrefixes};
use anyhow::{Context, Result};
use mlx_rs::ops::dequantize;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use serde_json::Value;
use std::collections::HashMap;

pub(crate) fn transform_deepseek_v3_tensors(
    tensors: &mut HashMap<String, Array>,
    prefixes: &TensorPrefixes,
    config: &ModelConfig,
    config_json: &Value,
    default_group_size: i32,
    default_bits: i32,
) -> Result<()> {
    let num_heads = config.num_attention_heads;
    let qk_nope_head_dim = config
        .qk_nope_head_dim
        .context("missing qk_nope_head_dim for DeepSeekV3")?;
    let v_head_dim = config
        .v_head_dim
        .context("missing v_head_dim for DeepSeekV3")?;
    let kv_lora_rank = config
        .kv_lora_rank
        .context("missing kv_lora_rank for DeepSeekV3")?;

    for i in 0..config.num_hidden_layers {
        let prefix = format!("{}.layers.{i}.self_attn", prefixes.model);
        if !tensors.contains_key(&format!("{prefix}.kv_b_proj.weight"))
            || tensors.contains_key(&format!("{prefix}.embed_q.weight"))
        {
            continue;
        }

        let (group_size, bits) = quant_params_for(
            config_json,
            &format!("{prefix}.kv_b_proj"),
            default_group_size,
            default_bits,
        );
        let weight = tensors
            .get(&format!("{prefix}.kv_b_proj.weight"))
            .cloned()
            .with_context(|| format!("missing {prefix}.kv_b_proj.weight"))?;
        let scales = tensors
            .get(&format!("{prefix}.kv_b_proj.scales"))
            .cloned()
            .with_context(|| format!("missing {prefix}.kv_b_proj.scales"))?;
        let biases = tensors
            .get(&format!("{prefix}.kv_b_proj.biases"))
            .cloned()
            .with_context(|| format!("missing {prefix}.kv_b_proj.biases"))?;
        let dense = dequantize(&weight, &scales, &biases, group_size, bits)?;
        let dense = dense.reshape(&[num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank])?;
        let wk = dense
            .index((std::ops::RangeFull, ..qk_nope_head_dim, std::ops::RangeFull))
            .transpose_axes(&[0, 2, 1])?;
        let wv = dense.index((std::ops::RangeFull, qk_nope_head_dim.., std::ops::RangeFull));
        let (wk_q, wk_s, wk_b) = quantize_stacked_weights(&wk, group_size, bits)?;
        let (wv_q, wv_s, wv_b) = quantize_stacked_weights(&wv, group_size, bits)?;
        tensors.insert(format!("{prefix}.embed_q.weight"), wk_q);
        tensors.insert(format!("{prefix}.embed_q.scales"), wk_s);
        tensors.insert(format!("{prefix}.embed_q.biases"), wk_b);
        tensors.insert(format!("{prefix}.unembed_out.weight"), wv_q);
        tensors.insert(format!("{prefix}.unembed_out.scales"), wv_s);
        tensors.insert(format!("{prefix}.unembed_out.biases"), wv_b);
    }

    Ok(())
}
