//! Inference performance tracker — locally observed TTFT per model + target.
//!
//! Each node tracks the time-to-first-token it observes when proxying inference
//! requests. This is purely local data (not gossiped) and feeds into routing
//! decisions to prefer faster hosts and avoid slow ones.
//!
//! Design:
//! - Ring buffer of the last N samples per (model, target) pair
//! - Reported TTFT is the best of the last 3 samples (filters outliers)
//! - Entries expire after 30 minutes of no updates
//! - Background probes populate cold entries when new hosts appear

use crate::inference::election::InferenceTarget;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_SAMPLES: usize = 8;
const BEST_OF_N: usize = 3;
const ENTRY_TTL: Duration = Duration::from_secs(30 * 60);
const MAX_ENTRIES: usize = 512;

/// Compact target identifier for map keys (EndpointId is 32 bytes, port is 2).
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum TargetKey {
    Local(u16),
    Remote([u8; 32]),
}

impl TargetKey {
    fn from_inference_target(target: &InferenceTarget) -> Option<Self> {
        match target {
            InferenceTarget::Local(port) | InferenceTarget::MoeLocal(port) => {
                Some(TargetKey::Local(*port))
            }
            InferenceTarget::Remote(id) | InferenceTarget::MoeRemote(id) => {
                Some(TargetKey::Remote(*id.as_bytes()))
            }
            InferenceTarget::None => None,
        }
    }

    fn display_short(&self) -> String {
        match self {
            TargetKey::Local(port) => format!("local:{port}"),
            TargetKey::Remote(bytes) => {
                // Same format as iroh's fmt_short: first 5 bytes hex
                format!(
                    "{}",
                    bytes[..5]
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>()
                )
            }
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PerfKey {
    model: String,
    target: TargetKey,
}

#[derive(Clone, Debug)]
struct Sample {
    ttft_ms: u64,
    _recorded_at: Instant,
}

#[derive(Clone, Debug)]
struct PerfEntry {
    samples: Vec<Sample>,
    last_updated: Instant,
}

impl PerfEntry {
    fn new() -> Self {
        Self {
            samples: Vec::with_capacity(MAX_SAMPLES),
            last_updated: Instant::now(),
        }
    }

    fn push(&mut self, ttft_ms: u64) {
        self.last_updated = Instant::now();
        if self.samples.len() >= MAX_SAMPLES {
            self.samples.remove(0);
        }
        self.samples.push(Sample {
            ttft_ms,
            _recorded_at: Instant::now(),
        });
    }

    fn is_expired(&self) -> bool {
        self.last_updated.elapsed() > ENTRY_TTL
    }

    /// Best (lowest) TTFT among the last N samples.
    /// Uses the most recent `BEST_OF_N` samples, returns the minimum.
    /// This filters out transient spikes from contention or cold cache.
    fn best_ttft_ms(&self) -> Option<u64> {
        if self.samples.is_empty() {
            return None;
        }
        let recent_count = self.samples.len().min(BEST_OF_N);
        let start = self.samples.len() - recent_count;
        self.samples[start..].iter().map(|s| s.ttft_ms).min()
    }

    fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

#[derive(Default)]
struct TrackerState {
    entries: HashMap<PerfKey, PerfEntry>,
}

impl TrackerState {
    fn prune_expired(&mut self) {
        self.entries.retain(|_, entry| !entry.is_expired());
        // Hard cap to prevent unbounded growth
        if self.entries.len() > MAX_ENTRIES {
            // Remove oldest entries
            let mut by_age: Vec<_> = self
                .entries
                .iter()
                .map(|(k, v)| (k.clone(), v.last_updated))
                .collect();
            by_age.sort_by_key(|(_, t)| *t);
            let to_remove = self.entries.len() - MAX_ENTRIES;
            for (key, _) in by_age.into_iter().take(to_remove) {
                self.entries.remove(&key);
            }
        }
    }
}

/// Thread-safe inference performance tracker.
/// Clone-cheap (inner Arc).
#[derive(Clone, Default)]
pub struct InferenceTracker {
    inner: Arc<Mutex<TrackerState>>,
}

impl InferenceTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an observed TTFT for a (model, target) pair.
    pub fn record_ttft(&self, model: &str, target: &InferenceTarget, ttft_ms: u64) {
        let Some(target_key) = TargetKey::from_inference_target(target) else {
            return;
        };
        let key = PerfKey {
            model: model.to_string(),
            target: target_key,
        };
        let mut state = self.inner.lock().unwrap();
        state.prune_expired();
        let entry = state.entries.entry(key).or_insert_with(PerfEntry::new);
        entry.push(ttft_ms);
    }

    /// Get the best observed TTFT for a model across all targets.
    /// Used by routing to prefer faster models (will be wired in a follow-up).
    #[allow(dead_code)]
    pub fn best_ttft_for_model(&self, model: &str) -> Option<u64> {
        let state = self.inner.lock().unwrap();
        state
            .entries
            .iter()
            .filter(|(k, v)| k.model == model && !v.is_expired())
            .filter_map(|(_, v)| v.best_ttft_ms())
            .min()
    }

    /// Get the best observed TTFT for a specific (model, target) pair.
    /// Used by routing to prefer faster targets (will be wired in a follow-up).
    #[allow(dead_code)]
    pub fn best_ttft_for_target(&self, model: &str, target: &InferenceTarget) -> Option<u64> {
        let target_key = TargetKey::from_inference_target(target)?;
        let key = PerfKey {
            model: model.to_string(),
            target: target_key,
        };
        let state = self.inner.lock().unwrap();
        state
            .entries
            .get(&key)
            .filter(|e| !e.is_expired())
            .and_then(|e| e.best_ttft_ms())
    }

    /// Snapshot for API exposure.
    pub fn snapshot(&self) -> InferencePerfSnapshot {
        let mut state = self.inner.lock().unwrap();
        state.prune_expired();

        // Aggregate by model: for each model, find best target + overall stats
        let mut by_model: HashMap<String, Vec<(&PerfKey, &PerfEntry)>> = HashMap::new();
        for (key, entry) in &state.entries {
            by_model
                .entry(key.model.clone())
                .or_default()
                .push((key, entry));
        }

        let mut models = Vec::new();
        for (model_name, entries) in &by_model {
            let mut best_ttft: Option<u64> = None;
            let mut best_target: Option<String> = None;
            let mut total_samples: usize = 0;
            let mut targets = Vec::new();

            for (key, entry) in entries {
                let ttft = entry.best_ttft_ms();
                let count = entry.sample_count();
                total_samples += count;

                targets.push(ModelTargetPerf {
                    target: key.target.display_short(),
                    best_ttft_ms: ttft,
                    samples: count,
                });

                if let Some(t) = ttft {
                    if best_ttft.map(|b| t < b).unwrap_or(true) {
                        best_ttft = Some(t);
                        best_target = Some(key.target.display_short());
                    }
                }
            }

            targets.sort_by_key(|t| t.best_ttft_ms.unwrap_or(u64::MAX));

            models.push(ModelPerfSummary {
                model: model_name.clone(),
                best_ttft_ms: best_ttft,
                best_target,
                total_samples,
                targets,
            });
        }

        models.sort_by(|a, b| a.model.cmp(&b.model));

        InferencePerfSnapshot { models }
    }
}

// ── API types ───────────────────────────────────────────────────────

#[derive(Clone, Debug, Default, Serialize)]
pub struct InferencePerfSnapshot {
    pub models: Vec<ModelPerfSummary>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelPerfSummary {
    pub model: String,
    pub best_ttft_ms: Option<u64>,
    pub best_target: Option<String>,
    pub total_samples: usize,
    pub targets: Vec<ModelTargetPerf>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelTargetPerf {
    pub target: String,
    pub best_ttft_ms: Option<u64>,
    pub samples: usize,
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::SecretKey;

    fn make_remote(seed: u8) -> InferenceTarget {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        InferenceTarget::Remote(SecretKey::from_bytes(&bytes).public())
    }

    #[test]
    fn test_record_and_retrieve() {
        let tracker = InferenceTracker::new();
        let target = InferenceTarget::Local(8080);
        tracker.record_ttft("qwen", &target, 500);
        tracker.record_ttft("qwen", &target, 300);
        tracker.record_ttft("qwen", &target, 800);

        // Best of last 3: min(500, 300, 800) = 300
        assert_eq!(tracker.best_ttft_for_target("qwen", &target), Some(300));
        assert_eq!(tracker.best_ttft_for_model("qwen"), Some(300));
    }

    #[test]
    fn test_best_of_3_filters_spikes() {
        let tracker = InferenceTracker::new();
        let target = InferenceTarget::Local(8080);

        // First few samples include a spike
        tracker.record_ttft("model", &target, 200);
        tracker.record_ttft("model", &target, 5000); // spike
        tracker.record_ttft("model", &target, 250);

        // Best of last 3: min(200, 5000, 250) = 200
        assert_eq!(tracker.best_ttft_for_target("model", &target), Some(200));
    }

    #[test]
    fn test_multiple_targets_same_model() {
        let tracker = InferenceTracker::new();
        let fast = InferenceTarget::Local(8080);
        let slow = make_remote(1);

        tracker.record_ttft("qwen", &fast, 100);
        tracker.record_ttft("qwen", &slow, 2000);

        assert_eq!(tracker.best_ttft_for_target("qwen", &fast), Some(100));
        assert_eq!(tracker.best_ttft_for_target("qwen", &slow), Some(2000));
        assert_eq!(tracker.best_ttft_for_model("qwen"), Some(100));
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let tracker = InferenceTracker::new();
        let target = InferenceTarget::Local(8080);

        // Push more than MAX_SAMPLES
        for i in 0..MAX_SAMPLES + 5 {
            tracker.record_ttft("m", &target, (i as u64 + 1) * 100);
        }

        let state = tracker.inner.lock().unwrap();
        let key = PerfKey {
            model: "m".to_string(),
            target: TargetKey::Local(8080),
        };
        assert_eq!(state.entries[&key].samples.len(), MAX_SAMPLES);
    }

    #[test]
    fn test_snapshot_structure() {
        let tracker = InferenceTracker::new();
        let local = InferenceTarget::Local(8080);
        let remote = make_remote(1);

        tracker.record_ttft("modelA", &local, 200);
        tracker.record_ttft("modelA", &remote, 500);
        tracker.record_ttft("modelB", &local, 150);

        let snap = tracker.snapshot();
        assert_eq!(snap.models.len(), 2);

        let a = snap.models.iter().find(|m| m.model == "modelA").unwrap();
        assert_eq!(a.best_ttft_ms, Some(200));
        assert_eq!(a.targets.len(), 2);
        assert_eq!(a.total_samples, 2);

        let b = snap.models.iter().find(|m| m.model == "modelB").unwrap();
        assert_eq!(b.best_ttft_ms, Some(150));
        assert_eq!(b.targets.len(), 1);
    }

    #[test]
    fn test_none_target_ignored() {
        let tracker = InferenceTracker::new();
        tracker.record_ttft("model", &InferenceTarget::None, 100);
        assert_eq!(tracker.best_ttft_for_model("model"), None);
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = InferenceTracker::new();
        assert_eq!(tracker.best_ttft_for_model("nonexistent"), None);
        let snap = tracker.snapshot();
        assert!(snap.models.is_empty());
    }
}
