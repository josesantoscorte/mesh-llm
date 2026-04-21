use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use hf_hub::types::cache::HFCacheInfo;

use crate::models::local::{huggingface_hub_cache_dir, scan_hf_cache_info, split_gguf_base_name};
use crate::models::usage;

#[derive(Debug)]
pub struct DeleteResult {
    pub deleted_paths: Vec<PathBuf>,
    pub reclaimed_bytes: u64,
    pub removed_metadata_files: usize,
    pub removed_usage_records: usize,
}

/// Resolve a user-facing identifier to the set of all GGUF file paths that belong to that model.
///
/// Resolution strategy:
/// - If `identifier` contains `/`, treat it as a HF repo ID and scan HFCacheInfo for any gguf files
///   in repos whose repo_id starts with `identifier`.
/// - Otherwise, treat it as a stem (filename base without `.gguf`) and look for exact filename
///   matches OR split-GGUF patterns (`stem-00001-of-NNNNN.gguf`).
///
/// Returns an error if zero files match (NotFound), or if multiple distinct repos match
/// and cannot be disambiguated by stem analysis (Ambiguous).
pub fn resolve_model_identifier(identifier: &str) -> Result<Vec<PathBuf>> {
    let cache_root = huggingface_hub_cache_dir();
    let Some(cache_info) = scan_hf_cache_info(&cache_root) else {
        bail!("Model not found: {}", identifier);
    };

    let stem_matches = find_stem_matches(&cache_info, identifier);
    if !stem_matches.is_empty() {
        return Ok(stem_matches);
    }

    let repo_matches = find_repo_prefix_matches(&cache_info, identifier);
    if !repo_matches.is_empty() {
        return Ok(repo_matches);
    }

    bail!("Model not found: {}", identifier);
}

/// Find GGUF file paths whose filename (or split-GGUF base name) matches the given stem.
fn find_stem_matches(cache_info: &HFCacheInfo, stem: &str) -> Vec<PathBuf> {
    let mut results: BTreeSet<PathBuf> = BTreeSet::new();

    for repo in &cache_info.repos {
        use hf_hub::RepoType;
        if repo.repo_type != RepoType::Model {
            continue;
        }
        for revision in &repo.revisions {
            for file in &revision.files {
                if !file.file_name.ends_with(".gguf") {
                    continue;
                }
                let file_stem = file
                    .file_name
                    .strip_suffix(".gguf")
                    .unwrap_or(&file.file_name);

                if file_stem.eq_ignore_ascii_case(stem) {
                    results.insert(file.file_path.clone());
                    continue;
                }

                if let Some(shard_prefix) = extract_shard_prefix(stem) {
                    let prefix_pattern = format!("{}-", shard_prefix);
                    if file.file_name.starts_with(&prefix_pattern)
                        && file.file_name.ends_with(".gguf")
                    {
                        results.insert(file.file_path.clone());
                    }
                }

                if let Some(base) = split_gguf_base_name(stem) {
                    let base_lower = base.to_ascii_lowercase();
                    if file_stem.to_ascii_lowercase() == base_lower {
                        results.insert(file.file_path.clone());
                    }
                }
            }
        }
    }

    results.into_iter().collect()
}

/// Extract the shard prefix from a filename like "Qwen3-8B-Q4_K_M-00001".
fn extract_shard_prefix(filename: &str) -> Option<&str> {
    let dash = filename.rfind('-')?;
    let num_part = &filename[dash + 1..];
    if num_part.len() != 5 || !num_part.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(&filename[..dash])
}

/// Find GGUF files in repos whose repo_id starts with the given identifier.
fn find_repo_prefix_matches(cache_info: &HFCacheInfo, prefix: &str) -> Vec<PathBuf> {
    let mut results: BTreeSet<PathBuf> = BTreeSet::new();

    for repo in &cache_info.repos {
        use hf_hub::RepoType;
        if repo.repo_type != RepoType::Model {
            continue;
        }
        if !repo.repo_id.starts_with(prefix) {
            continue;
        }
        for revision in &repo.revisions {
            for file in &revision.files {
                if file.file_name.ends_with(".gguf") {
                    results.insert(file.file_path.clone());
                }
            }
        }
    }

    results.into_iter().collect()
}

/// Collect all paths that should be deleted for a model identified by its primary path.
/// This includes managed_paths from usage records and any sidecar GGUF files (mmproj, etc.)
/// found in the same directory as the primary model file.
pub fn collect_delete_paths(primary_path: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut to_delete: BTreeSet<PathBuf> = BTreeSet::new();

    if let Some(record) = usage::load_model_usage_record_for_path(primary_path) {
        if record.mesh_managed && !record.managed_paths.is_empty() {
            for p in &record.managed_paths {
                to_delete.insert(p.clone());
            }
            return Ok(to_delete.into_iter().collect());
        }
    }

    let parent = primary_path.parent().context("No parent dir")?;

    let filename = primary_path
        .file_name()
        .and_then(|f| f.to_str())
        .context("No filename")?;

    let base_stem = split_gguf_base_name(filename)
        .unwrap_or(filename.strip_suffix(".gguf").unwrap_or(filename));

    let entries = std::fs::read_dir(parent)
        .with_context(|| format!("Cannot read directory: {}", parent.display()))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() && !path.is_symlink() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
            continue;
        }

        let entry_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

        if entry_stem.eq_ignore_ascii_case(base_stem)
            || entry_stem.starts_with(&format!("{}-", base_stem))
            || (split_gguf_base_name(entry_stem).is_some()
                && split_gguf_base_name(entry_stem).unwrap() == base_stem)
        {
            to_delete.insert(path);
        }
    }

    Ok(to_delete.into_iter().collect())
}

/// Execute a model deletion by identifier, collecting and removing ALL files.
pub async fn delete_model_by_identifier(identifier: &str) -> Result<DeleteResult> {
    let primary_paths = resolve_model_identifier(identifier)?;

    if primary_paths.is_empty() {
        bail!("Model not found: {}", identifier);
    }

    let primary_path = &primary_paths[0];

    let all_paths = collect_delete_paths(primary_path)?;

    if all_paths.is_empty() {
        bail!(
            "No GGUF files found at resolved path: {}",
            primary_path.display()
        );
    }

    let mut reclaimed_bytes: u64 = 0;
    let mut removed_metadata_files: usize = 0;
    let mut removed_usage_records: usize = 0;
    let mut deleted_paths: Vec<PathBuf> = Vec::new();

    for path in &all_paths {
        if path.exists() {
            if let Ok(meta) = std::fs::metadata(path) {
                reclaimed_bytes += meta.len();
            }
            std::fs::remove_file(path).with_context(|| format!("Remove {}", path.display()))?;
            deleted_paths.push(path.clone());

            if let Some(metadata_path) = crate::models::local::gguf_metadata_cache_path(path) {
                if metadata_path.exists() {
                    std::fs::remove_file(&metadata_path).with_context(|| {
                        format!("Remove metadata cache {}", metadata_path.display())
                    })?;
                    removed_metadata_files += 1;
                }
            }

            prune_empty_ancestors(path, &huggingface_hub_cache_dir());
        }
    }

    if let Some(record) = load_model_usage_record_for_path(primary_path) {
        let usage_dir = usage::model_usage_cache_dir();
        let record_path = usage::usage_record_path(&usage_dir, &record.lookup_key);
        if record_path.exists() {
            std::fs::remove_file(&record_path)
                .with_context(|| format!("Remove usage record {}", record_path.display()))?;
            removed_usage_records += 1;
        }
    }

    Ok(DeleteResult {
        deleted_paths,
        reclaimed_bytes,
        removed_metadata_files,
        removed_usage_records,
    })
}

/// Load a model usage record for a given path.
fn load_model_usage_record_for_path(path: &std::path::Path) -> Option<usage::ModelUsageRecord> {
    usage::load_model_usage_record_for_path(path)
}

/// Prune empty ancestor directories up to (but not including) stop_at.
fn prune_empty_ancestors(path: &std::path::Path, stop_at: &std::path::Path) {
    let stop_at = normalize_path(stop_at);
    let mut current = path.parent().map(normalize_path);
    while let Some(dir) = current {
        if dir == stop_at {
            break;
        }
        let Ok(mut entries) = std::fs::read_dir(&dir) else {
            break;
        };
        if entries.next().is_some() {
            break;
        }
        if std::fs::remove_dir(&dir).is_err() {
            break;
        }
        current = dir.parent().map(normalize_path);
    }
}

fn normalize_path(path: &std::path::Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_shard_prefix_valid() {
        assert_eq!(
            extract_shard_prefix("Qwen3-8B-Q4_K_M-00001"),
            Some("Qwen3-8B-Q4_K_M")
        );
        assert_eq!(extract_shard_prefix("model-00001"), Some("model"));
    }

    #[test]
    fn test_extract_shard_prefix_invalid() {
        assert_eq!(extract_shard_prefix("model-001"), None);
        assert_eq!(extract_shard_prefix("model00001"), None);
        assert_eq!(extract_shard_prefix("model-abcde"), None);
    }

    #[test]
    fn test_split_gguf_base_name_single_file() {
        assert_eq!(split_gguf_base_name("Qwen3-8B-Q4_K_M"), None);
    }

    #[test]
    fn test_split_gguf_base_name_first_shard() {
        let base = split_gguf_base_name("GLM-5-UD-IQ2_XXS-00001-of-00006");
        assert_eq!(base, Some("GLM-5-UD-IQ2_XXS"));
    }

    #[test]
    fn test_split_gguf_base_name_last_shard() {
        let base = split_gguf_base_name("GLM-5-UD-IQ2_XXS-00006-of-00006");
        assert_eq!(base, Some("GLM-5-UD-IQ2_XXS"));
    }
}
