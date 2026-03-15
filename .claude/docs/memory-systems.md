# Memory Systems (Shared & Individual)

## Overview

The codebase implements a dual-memory architecture: **SharedMemory** for thread-safe cross-agent coordination and **per-agent memory** (BaseMemory, ShortTermMemory, LongTermMemory) for individual agent state. Agents use SharedMemory to coordinate rendezvous points for cooperative carry operations, while maintaining private memories for local observations and action history.

## Shared Memory Architecture

SharedMemory provides thread-safe inter-agent communication using a simple key-value store with Python's `threading.Lock`. Located in `memory/shared_memory.py`, it exposes three methods: `update(key, info)`, `retrieve(key)`, and `retrieve_all()`. All operations acquire the lock using `with self.lock:` to prevent race conditions.

**Instantiation**: One SharedMemory instance per simulation, created in `worlds1/WorldBuilder.py` as `marble_shared_memory = SharedMemory()` and passed to all SearchRescueAgent instances during construction.

**Usage Pattern**: Agents publish coordination events (e.g., "I'm waiting at location X with victim Y") and poll for events from teammates (e.g., "Is anyone waiting for cooperative carry?").

## Rendezvous Mechanism

The primary SharedMemory use case is cooperative victim transport. When an agent initiates `CarryObjectTogether`, it enters a retry loop (`_handle_carry_retry` in `agents1/llm_agent_base.py`) and publishes to `carry_rendezvous` key with `{agent, victim_id, location, status: 'waiting_for_partner'}`. Other agents poll this key in `_handle_rendezvous()` and navigate to the rendezvous location if status is `waiting_for_partner` and they're not the waiting agent. After delivery or timeout (100 ticks), the key is cleared by setting it to `None`.

Additionally, agents broadcast completed carry/obstacle actions to keys like `victim_{obj_id}` and `obstacle_{obj_id}` via `_maybe_share_observation()`, though these are not actively polled—they serve as a coordination log.

## Per-Agent Memory

Each agent maintains a private `self.memory` instance (BaseMemory by default, initialized in `LLMAgentBase.__init__`). BaseMemory is a simple append-only list (`self.storage: List[Any]`) with methods `update(key, info)` (appends), `retrieve_latest()` (returns last item), and `retrieve_all()` (returns shallow copy).

**Usage**: Agents log actions (`self.memory.update('action', {...})`) and carry failures (`self.memory.update('carry_failure', {...})`). SearchRescueAgent retrieves the last 15 entries when building LLM prompts: `self.memory.retrieve_all()[-15:]`.

The `key` parameter in `update()` is unused in BaseMemory—kept for API consistency with SharedMemory—but memory entries are typically dicts with a `type` field for semantic filtering.

## Short-Term Memory with Compression

`memory/short_term_memory.py` extends BaseMemory with LLM-based compression. When storage exceeds `memory_limit` (default 20 entries), `_compress_oldest()` pops the two oldest entries, sends them to Ollama for summarization, and prepends a `{type: 'old_memory_summary', entries: [...]}` entry. This keeps token usage bounded when injecting memory into prompts.

**Configuration**: Takes `memory_limit`, `llm_model` (default `qwen3:8b`), and `api_url` (Ollama endpoint). Exposes `get_compact_str()` for JSON serialization with minimal separators.

## Long-Term Memory (Embedding-Based)

`memory/long_term_memory.py` (imported conditionally in `memory/__init__.py` due to sklearn/litellm dependencies) stores entries as `(info, embedding)` tuples using OpenAI's `text-embedding-3-small`. Provides `retrieve_most_relevant(query, n=1)` via cosine similarity and optional LLM summarization. Not currently used by SearchRescueAgent—present for extensibility.

## Global State (Pseudo-Shared Memory)

Agents also maintain `WORLD_STATE_GLOBAL` (in `agents1/modules/perception_module.py`), a dict tracking all observed victims, obstacles, doors, and teammate positions across the entire simulation. This is **agent-local** (not thread-safe across agents) and accumulates data from each tick's observations via `process_observations()`. When building LLM prompts, SearchRescueAgent merges local `WORLD_STATE` (1-block radius) with `WORLD_STATE_GLOBAL['victims']`, `obstacles`, `doors` to give the LLM a "known map" of previously seen objects.

**Key Difference**: SharedMemory coordinates **inter-agent synchronization** (rendezvous), while WORLD_STATE_GLOBAL provides **intra-agent memory** of the map.

## Thread Safety Summary

- **SharedMemory**: Thread-safe via `threading.Lock` (all methods acquire lock)
- **BaseMemory / ShortTermMemory / LongTermMemory**: Not thread-safe (single-agent ownership)
- **WORLD_STATE_GLOBAL**: Agent-local dict, no cross-agent sharing
