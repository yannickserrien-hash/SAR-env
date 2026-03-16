# Memory Systems

## Overview

SAR-env uses a dual-memory architecture: thread-safe `SharedMemory` for cross-agent coordination and per-agent `BaseMemory`/`ShortTermMemory` for individual state. SharedMemory enables cooperative actions through a rendezvous mechanism that synchronizes agents for two-person carry operations.

## SharedMemory: Thread-Safe Coordination

**Location**: `memory/shared_memory.py`

Simple Lock-protected key-value store using `threading.Lock`. All operations (`update()`, `retrieve()`, `retrieve_all()`) wrap dict access with `with self.lock:` to prevent race conditions when multiple agent threads read/write simultaneously.

**Instantiation**: One instance per simulation, created in `worlds1/WorldBuilder.py` (line 92: `marble_shared_memory = SharedMemory()`) and injected into each `SearchRescueAgent` during initialization.

### Rendezvous Mechanism

**Purpose**: Coordinate two agents for `CarryObjectTogether` (critically injured victims) and `RemoveObjectTogether` (large obstacles).

**Flow** (implemented in `agents1/llm_agent_base.py`):

1. **Initiator**: Agent attempts cooperative carry but partner not adjacent. Enters `_handle_carry_retry_loop()` and publishes rendezvous request:
   ```python
   self.shared_memory.update('carry_rendezvous', {
       'agent': self.agent_id,
       'victim_id': obj_id,
       'location': agent_loc,
       'status': 'waiting_for_partner',
   })
   ```

2. **Partner Discovery**: Other agents poll `carry_rendezvous` key each tick in `_handle_rendezvous()`. If status is `'waiting_for_partner'` and agent ID differs, they navigate to the published location.

3. **Blocking Wait**: Initiating agent retries carry action every tick (line 335) up to `CARRY_WAIT_TIMEOUT_TICKS = 100`. While waiting, agent cannot reason or perform other actions—stuck in retry loop until partner arrives or timeout.

4. **Resolution**: On success (victim no longer nearby, line 296) or timeout (line 305), rendezvous cleared via `shared_memory.update('carry_rendezvous', None)`.

**Key Insight**: This is cooperative blocking through shared state polling, not true async synchronization. The waiting agent burns ticks idling.

## Per-Agent Memory: BaseMemory

**Location**: `memory/base_memory.py`

Simple append-only list (`self.storage: List[Any]`) instantiated per agent in `LLMAgentBase.__init__()` (line 121). Methods:
- `update(key, info)`: Appends `info` to list (key parameter ignored, kept for API consistency with SharedMemory)
- `retrieve_latest()`: Returns last entry or None
- `retrieve_all()`: Returns shallow copy of list

**Usage**: Logs action feedback, carry failures, etc. Not currently injected into `SearchRescueAgent` prompts (legacy `RescueAgent` used it). Exists for debugging and future memory-augmented reasoning.

## ShortTermMemory: LLM-Compressed History

**Location**: `memory/short_term_memory.py`

Extends `BaseMemory` with structured dict storage and automatic LLM summarization. Configuration: `memory_limit=20`, `llm_model='qwen3:8b'`, `api_url=None` (per-agent Ollama endpoint).

**Storage Constraint**: Each entry must be dict with `type` key (e.g., `'victim_found'`, `'room_explored'`). When `len(storage) >= memory_limit`, `_compress_oldest()` pops oldest 2 entries, sends to Ollama via `call_llm_sync()` (from `agents1/async_model_prompting.py`), and replaces with single `{'type': 'old_memory_summary', 'entries': [...]}` entry.

**Token Optimization**: `get_compact_str()` serializes to JSON with minimal separators (`json.dumps(..., separators=(',', ':'))`) for efficient LLM prompt injection.

**Current Status**: Only used by deprecated `RescueAgent` in `agents1/agents_graveyard/`. Active `SearchRescueAgent` uses `BaseMemory`. ShortTermMemory is a working prototype for bounded-memory agents.

## Thread Safety Summary

- **SharedMemory**: Thread-safe via `threading.Lock` (all methods acquire lock before dict access)
- **BaseMemory / ShortTermMemory**: Not thread-safe (single-agent ownership, no cross-thread access)

## Key Files

- `memory/shared_memory.py` - Lock-protected cross-agent coordination
- `memory/base_memory.py` - Simple per-agent append-only list
- `memory/short_term_memory.py` - LLM-compressed bounded memory
- `agents1/llm_agent_base.py` - Rendezvous logic (lines 272-375: `_handle_rendezvous`, `_handle_carry_retry_loop`)
- `worlds1/WorldBuilder.py` - SharedMemory instantiation (line 92)
