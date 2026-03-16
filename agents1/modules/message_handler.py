"""
MessageHandler — Minimal message parsing and context building for agent communication.

Parses incoming MATRX messages (with [tag:xxx] prefix), tracks new/private messages,
and builds compact context strings for LLM prompts.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


_TAG_RE = re.compile(r'^\[tag:(\w+)\]\s*(.*)', re.DOTALL)

VALID_TAGS = frozenset({'ask_help', 'share_info', 'request_task', 'task_update', 'reply'})


@dataclass
class MessageRecord:
    from_id: str
    tag: str            # ask_help | share_info | request_task | task_update | reply
    content: str
    is_private: bool    # True if addressed to this specific agent
    processed: bool = False


class MessageHandler:
    """Parses MATRX messages and provides context for LLM prompts.

    Args:
        agent_id: This agent's ID (set after MATRX initialization).
    """

    def __init__(self, agent_id: str = ''):
        self.agent_id = agent_id
        self._last_processed_index: int = 0
        self._history: List[MessageRecord] = []

    # ── Parsing ────────────────────────────────────────────────────────────

    def parse_new_messages(self, received_messages: list) -> List[MessageRecord]:
        """Parse MATRX messages received since the last call.

        Args:
            received_messages: ``self.received_messages`` from MATRX AgentBrain
                               (list of Message objects, accumulated across ticks).

        Returns:
            List of newly parsed ``MessageRecord`` objects.
        """
        new_records: List[MessageRecord] = []
        new_msgs = received_messages[self._last_processed_index:]
        self._last_processed_index = len(received_messages)

        for msg in new_msgs:
            from_id = getattr(msg, 'from_id', None) or '?'
            # Skip own messages
            if from_id == self.agent_id:
                continue

            raw_content = getattr(msg, 'content', '')
            if isinstance(raw_content, dict):
                raw_content = str(raw_content)

            # Parse [tag:xxx] prefix
            tag, content = self._parse_tag(raw_content)

            # Determine if private (to_id targets this agent specifically)
            to_id = getattr(msg, 'to_id', None)
            is_private = (
                to_id is not None
                and isinstance(to_id, str)
                and to_id == self.agent_id
            )

            record = MessageRecord(
                from_id=from_id,
                tag=tag,
                content=content,
                is_private=is_private,
            )
            self._history.append(record)
            new_records.append(record)

        return new_records

    # ── Context for prompts ────────────────────────────────────────────────

    def get_context_for_prompt(self, max_messages: int = 10) -> str:
        """Return recent messages as a compact string for inclusion in LLM prompts.

        Format per line: ``[from_id|tag] content``

        Returns:
            Formatted string, or ``"(no recent messages)"`` if empty.
        """
        recent = self._history[-max_messages:] if self._history else []
        if not recent:
            return '(no recent messages)'

        lines = []
        for r in recent:
            lines.append(f'[{r.from_id}|{r.tag}] {r.content}')
        return '\n'.join(lines)

    # ── Query helpers ──────────────────────────────────────────────────────

    def get_unprocessed_private(self) -> List[MessageRecord]:
        """Return private messages that have not yet been processed."""
        return [r for r in self._history if r.is_private and not r.processed]

    def get_unprocessed(self) -> List[MessageRecord]:
        """Return all messages that have not yet been processed."""
        return [r for r in self._history if not r.processed]

    # ── Internal ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_tag(raw: str) -> tuple:
        """Extract ``(tag, content)`` from a ``[tag:xxx] content`` string."""
        m = _TAG_RE.match(raw)
        if m:
            tag = m.group(1) if m.group(1) in VALID_TAGS else 'share_info'
            return tag, m.group(2)
        return 'share_info', raw
