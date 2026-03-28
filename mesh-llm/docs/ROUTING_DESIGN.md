# Routing & Session Design

## The Problem

A request arrives at a mesh proxy. We need to decide:
1. **Which model** — auto or user-specified
2. **Which host** — one of potentially many serving that model
3. **Sticky or not** — should follow-up requests go to the same host?

These three decisions interact. Getting them wrong wastes GPU memory (cold KV),
sends requests to overloaded hosts, or routes agentic work to weak models.

## Current State (v0.49)

```
Request → proxy → pick model → pick host (round-robin or first-reachable) → tunnel TCP
```

- No session stickiness in the normal path (only MoE has hash-based routing)
- Load signal is wrong: each proxy counts its own inflight, not actual server load
- Auto mode picks one model; if it fails, gives up or falls back

## Proposed Design

### Layer 1: Model Selection

Two modes, both produce a **ranked model list**:

**Auto mode** (`model=auto` or unset):
- Classify the request (agentic/chat/code/reasoning)
- Rank all served models by quality for that category + load penalty
- Agentic: only tool-capable models. Chat: all models (more flexibility)
- Result: ordered list like `[Qwen3-Coder-Next, Qwen2.5-Coder-32B, Qwen3-8B]`

**Named model** (`model=Qwen2.5-Coder-32B`):
- That model is the primary choice
- Alternatives are only tried if ALL hosts for the named model are **unreachable**
  (not busy — unreachable, i.e. connection fails)
- This is important: user asked for a specific model, respect that

### Layer 2: Host Selection (within a model)

Once a model is chosen, pick a host. This is where stickiness and load matter.

**Session affinity:**
- Extract session hint from request (`user` field, `session_id`, or client IP)
- Hash the session hint → sticky host assignment
- Same session always goes to the same host (warm KV cache, faster TTFT)
- Stickiness holds as long as the host is **reachable and not saturated**

**When stickiness breaks:**
- Host is unreachable (connection fails) → try next host, re-bind session
- Host is saturated (all slots full, confirmed by the host) → try next host
- Host left the mesh → re-bind

**Without a session hint:**
- Round-robin or least-loaded (no KV to preserve anyway)

### Layer 3: Load Signal (source of truth)

The load signal must come from the **worker node running llama-server**, not from proxies guessing.

**What the worker knows:**
- llama-server `/health` returns: `{"status":"ok"}` or `{"status":"no slot available","slots_idle":0,"slots_processing":1}`
- llama-server `/slots` returns per-slot state: idle, processing, prompt length, tokens generated
- The worker node can poll this cheaply (localhost HTTP, every 5-10s)

**What gets gossipped:**
```rust
pub struct PeerAnnouncement {
    // ... existing fields ...
    #[serde(default)]
    pub slots_idle: u8,      // how many KV slots are free
    #[serde(default)]
    pub slots_total: u8,     // total slots (usually 1 for big models)
}
```

**Why slots, not inflight:**
- `slots_idle=0` means the GPU literally cannot accept another request right now
- `slots_idle>0` means there's capacity, regardless of how many proxies are sending work
- This is the real signal. A proxy's local inflight count is meaningless in a multi-proxy mesh.

**Staleness (60s gossip):**
- Acceptable for sustained load (agentic sessions last minutes)
- For bursty chat: stale=idle is fine (worst case you send to a now-busy host, it queues)
- For bursty chat: stale=busy is fine (worst case you avoid a host that freed up, hit another)
- The only bad case: stale=available but actually crashed. Connection failure handles that.

### How It All Fits Together

```
Request arrives at proxy
  │
  ├─ auto mode?
  │   ├─ classify request (agentic/chat/code)
  │   └─ rank models by (quality for category - load penalty)
  │       chat: more models eligible, bigger pool
  │       agentic: only tool-capable models
  │
  ├─ named model?
  │   └─ that model is the list (fallback only on unreachable)
  │
  ▼
  For each model in ranked list:
    │
    ├─ session hint present?
    │   ├─ sticky host assigned and healthy? → use it
    │   └─ otherwise → pick least-loaded host, bind session
    │
    ├─ no session hint?
    │   └─ pick least-loaded host
    │
    ├─ try tunnel to host
    │   ├─ success → done
    │   └─ fail → try next host for this model
    │
    └─ all hosts failed → try next model in ranked list
```

### Load Penalty Details

The penalty applies at **model selection** (Layer 1), using the *minimum* slots_idle
across hosts serving that model. This is conservative — if any host has capacity, the
model isn't considered overloaded.

| Scenario | Agentic | Chat |
|----------|---------|------|
| All hosts have idle slots | No penalty | No penalty |
| Some hosts saturated | No penalty (others have capacity) | No penalty |
| All hosts saturated | Penalty (but never hard-block) | Larger penalty |

Chat gets penalized harder → spills to smaller models earlier → frees up big models for agentic.
Agentic holds the line → keeps the best tool model even under load.

### Session Stickiness Details

**Binding:**
- Key: `(model_name, session_hint)` → `host_id`
- Created on first request for a session
- Stored in-memory on the proxy (not gossipped — each proxy has its own bindings)

**Expiry:**
- Binding expires after 10 minutes of no requests (KV cache likely evicted anyway)
- Binding breaks immediately on connection failure

**No session hint:**
- Client IP as fallback? Or just no stickiness.
- Most agentic clients send `user` field. Chat UIs vary.

### What Changes

1. **Worker nodes poll llama-server `/health`** every 5-10s, store slots_idle/slots_total
2. **Gossip includes slots_idle/slots_total** instead of proxy inflight count
3. **Proxy maintains session→host bindings** (simple HashMap with TTL)
4. **Host selection prefers sticky host**, falls back to least-loaded
5. **Model selection uses slots_idle** for load penalty instead of inflight

### What Doesn't Change

- Gossip frequency stays at 60s (no extra traffic)
- Wire protocol is backward compatible (`#[serde(default)]`)
- Named model still gets priority (fallback only on unreachable)
- Never returns 503 when any host is reachable (soft penalties only)

### Open Questions

- Should session bindings be shared across proxies (gossip them)? Probably not — adds complexity, and if a client switches proxy entry point, cold KV is the least of the problems.
- Should we use `/slots` for richer signal (how many tokens generated per slot = how close to eviction)? Probably overkill for now.
- llama-server with `--parallel 1` (default for big models): slots_total=1, slots_idle is 0 or 1. Binary signal. Is that enough? Probably yes — it means "busy or not."
- What about split/tensor-parallel setups where one llama-server spans multiple nodes? The host election already handles this — the elected host is the one running llama-server, and it's the one that reports load.
