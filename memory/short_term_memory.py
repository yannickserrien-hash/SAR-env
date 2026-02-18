"""
Short-term memory module with LLM-based summarization.

Extends BaseMemory with:
- Structured dict storage (each entry has a 'type' key)
- Configurable capacity limit
- Automatic compression of oldest entries via Ollama LLM when full
- Compact serialization for token-efficient LLM prompt injection
"""

import json
import logging
from typing import Any, Dict, List, Optional

from memory.base_memory import BaseMemory

logger = logging.getLogger('ShortTermMemory')


class ShortTermMemory(BaseMemory):
    """
    Short-term memory that stores structured dict entries and automatically
    compresses the oldest entries when the capacity limit is reached.

    Each entry must be a dict with at least a ``type`` key (e.g.
    ``victim_found``, ``room_explored``).  Compression uses an Ollama LLM
    to merge old entries, keeping the storage within ``memory_limit``.
    """

    def __init__(self, memory_limit: int = 20, llm_model: str = 'llama3:8b') -> None:
        """
        Args:
            memory_limit: Maximum number of entries before compression kicks in.
            llm_model: Ollama model name used for entry summarization.
        """
        super().__init__()
        self.memory_limit: int = memory_limit
        self.llm_model: str = llm_model
        # Override parent's generic list with typed list
        self.storage: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, key: str, information: Dict[str, Any]) -> None:
        """
        Append a structured entry to memory.

        If storage is at capacity, the two oldest entries are popped and
        compressed into a single ``memory_summary`` entry first.

        Args:
            key: Ignored (kept for BaseMemory signature compatibility).
            information: A dict with at least a ``type`` key.
        """
        if len(self.storage) >= self.memory_limit:
            self._compress_oldest()
        self.storage.append(information)

    def retrieve_by_type(self, entry_type: str) -> List[Dict[str, Any]]:
        """Return all entries whose ``type`` matches *entry_type*."""
        return [e for e in self.storage if e.get('type') == entry_type]

    def get_compact_str(self) -> str:
        """
        Compact JSON serialization of all entries.

        Uses minimal separators (no spaces) for token efficiency when
        injected into an LLM prompt.
        """
        return json.dumps(self.storage, default=str, separators=(',', ':'))

    def retrieve_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recently added entry, or ``None``."""
        return self.storage[-1] if self.storage else None

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of all entries."""
        return self.storage.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress_oldest(self) -> None:
        """Pop up to 2 oldest entries and replace with a compressed summary."""
        count = min(2, len(self.storage))
        if count == 0:
            return
        oldest = [self.storage.pop(0) for _ in range(count)]
        compressed = self._summarize_entries(oldest)
        self.storage.insert(0, {
            'type': 'memory_summary',
            'entries': compressed,
        })

    def _summarize_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use the Ollama LLM to merge *entries* into fewer structured dicts.

        Falls back to returning the original entries unchanged if the LLM
        call fails or returns unparseable output.
        """
        # Late import to avoid circular dependency at module load time
        from engine.llm_utils import query_llm, parse_json_response

        system_prompt = (
            "You are compressing old memory entries for a search-and-rescue "
            "robot. Merge the given entries into fewer structured entries, "
            "preserving all critical facts (victim locations, explored rooms, "
            "obstacles). Output ONLY valid JSON: {\"entries\": [...]}"
        )
        user_prompt = (
            "Compress these old memory entries:\n"
            f"{json.dumps(entries, default=str)}\n\n"
            "Return JSON: {\"entries\": [...]}"
        )

        try:
            response = query_llm(
                model=self.llm_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1000,
                temperature=0.1,
            )
            parsed = parse_json_response(response)
            if parsed and 'entries' in parsed and isinstance(parsed['entries'], list):
                logger.info(f"Compressed {len(entries)} entries -> {len(parsed['entries'])}")
                return parsed['entries']
        except Exception as exc:
            logger.warning(f"LLM summarization failed ({exc}), keeping originals")

        # Fallback: return originals unchanged
        return entries
