use super::family::ModelArchitecture;
use super::{ModelConfig, TensorPrefixes};
use anyhow::Result;
use mlx_rs::Array;
use serde_json::Value;
use std::collections::HashMap;

mod deepseek_v3;
mod phi3;

pub(crate) fn apply_family_tensor_transforms(
    arch: ModelArchitecture,
    tensors: &mut HashMap<String, Array>,
    prefixes: &TensorPrefixes,
    config: &ModelConfig,
    config_json: &Value,
    default_group_size: i32,
    default_bits: i32,
) -> Result<()> {
    if arch.is_deepseek_v3() || arch.is_kimi_linear() {
        deepseek_v3::transform_deepseek_v3_tensors(
            tensors,
            prefixes,
            config,
            config_json,
            default_group_size,
            default_bits,
        )?;
    }

    if config_json
        .get("model_type")
        .and_then(|value| value.as_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("phi3"))
    {
        phi3::transform_phi3_tensors(tensors, prefixes, config)?;
    }

    Ok(())
}
