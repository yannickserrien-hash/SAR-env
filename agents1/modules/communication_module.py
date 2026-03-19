"""
Communication Module — environment-sided message processing.

Follows the Perception module pattern: processes raw MATRX received_messages
into structured data for LLM prompts. Does NOT handle sending — outbound
logic lives in LLMAgentBase._apply_communication().

Supports pluggable strategies that control which messages appear in prompts.
"""

import json
import logging
from concurrent.futures import Future
from typing import Any, Dict, List, Optional

logger = logging.getLogger('CommunicationModule')

VALID_MESSAGE_TYPES = frozenset({'ask_help', 'help', 'message'})

_COMMUNICATION_PROMPT = """You are a communication processor for a Search and Rescue agent team.
Extract actionable information from the following inter-agent messages.
If information is ambiguous or contradictory, prefer the most recent message.

Respond in JSON format:
{
    "discovered_victims": [{"id": "victim_id", "location": [x,y], "severity": "mild|critical"}],
    "discovered_obstacles": [{"id": "obstacle_id", "location": [x,y], "type": "rock|stone|tree"}],
    "help_requests": [{"from": "agent_id", "task": "description", "location": [x,y]}],
    "explored_areas": [{"area": "area N", "by": "agent_id", "status": "complete|partial|not_started"}],
    "teammate_updates": [{"agent": "agent_id", "status": "description", "location": [x,y]}],
    "summary": "One sentence prioritizing: (1) urgent help requests, (2) new discoveries, (3) status updates"
}

Only include fields with actual data. If nothing relevant, return {"summary": "No actionable information"}.
Respond with valid JSON only."""


# ── Strategies ────────────────────────────────────────────────────────────────

class CommStrategy:
    """Base — controls which messages the agent sees in prompts."""

    def filter_for_prompt(self, messages: List[dict], agent_busy: bool) -> List[dict]:
        raise NotImplementedError


class AlwaysRespondStrategy(CommStrategy):
    """Show all messages. LLM decides whether to respond or continue task."""

    def filter_for_prompt(self, messages: List[dict], agent_busy: bool = False) -> List[dict]:
        return messages


class BusyAwareStrategy(CommStrategy):
    """When busy, only show ask_help messages. When idle, show all."""

    def filter_for_prompt(self, messages: List[dict], agent_busy: bool = False) -> List[dict]:
        if not agent_busy:
            return messages
        return [m for m in messages if m.get('message_type') == 'ask_help']


_STRATEGY_REGISTRY: Dict[str, type] = {
    'always_respond': AlwaysRespondStrategy,
    'busy_aware': BusyAwareStrategy,
}


def _resolve_strategy(name: str) -> CommStrategy:
    cls = _STRATEGY_REGISTRY.get(name)
    if cls is None:
        logger.warning("Unknown comm strategy '%s', defaulting to always_respond", name)
        cls = AlwaysRespondStrategy
    return cls()


# ── Communication Module ─────────────────────────────────────────────────────

class CommunicationModule:
    """Environment-sided message processing (like Perception for observations).

    Processes raw MATRX received_messages into structured data.
    Handles async summarization of old message history.
    Does NOT handle sending — that's in LLMAgentBase._apply_communication().
    """

    def __init__(
        self,
        agent_id: str,
        strategy: str = 'always_respond',
        llm_model: Optional[str] = None,
        api_base: Optional[str] = None,
        summary_threshold: int = 10,
    ) -> None:
        self.agent_id = agent_id
        self._messages: List[dict] = []
        self._processed_count: int = 0
        self._strategy = _resolve_strategy(strategy)

        # Async summarization
        self._llm_model = llm_model
        self._api_base = api_base
        self._summary_threshold = summary_threshold
        self._summary_future: Optional[Future] = None
        self._summary_text: str = ''

    # ── Inbound processing ────────────────────────────────────────────────

    def process_messages(self, received_messages: list) -> None:
        """Ingest new MATRX messages since last call.

        Uses a counter to track which messages have been processed
        (received_messages only appends, never reorders).
        """
        self._poll_summary()

        new_count = len(received_messages)
        if new_count <= self._processed_count:
            return

        for msg in received_messages[self._processed_count:]:
            entry = self._extract_message(msg)
            if entry is not None:
                self._messages.append(entry)

        self._processed_count = new_count
        self._maybe_summarize()

    def _extract_message(self, msg) -> Optional[dict]:
        """Extract structured data from a MATRX Message object."""
        from_id = getattr(msg, 'from_id', None)
        to_id = getattr(msg, 'to_id', None)
        content = getattr(msg, 'content', None)

        # Skip messages from self
        if from_id == self.agent_id:
            return None

        # Handle structured content (dict with message_type + text)
        if isinstance(content, dict):
            text = content.get('text', '')
            msg_type = content.get('message_type', 'message')
            if msg_type not in VALID_MESSAGE_TYPES:
                msg_type = 'message'
        elif isinstance(content, str):
            text = content
            msg_type = 'message'
        else:
            return None

        if not text:
            return None

        return {
            'from': from_id or 'unknown',
            'to': 'all' if to_id is None else (to_id if isinstance(to_id, str) else str(to_id)),
            'message_type': msg_type,
            'text': text,
        }

    # ── Async summarization ───────────────────────────────────────────────

    def _poll_summary(self) -> None:
        """Non-blocking poll of pending summary future."""
        if self._summary_future is None:
            return
        from agents1.async_model_prompting import get_llm_result
        try:
            result = get_llm_result(self._summary_future)
        except Exception as exc:
            logger.warning('Summary LLM call failed: %s', exc)
            self._summary_future = None
            return

        if result is None:
            return  # still in flight

        # Extract text from LLM response
        text = getattr(result[0], 'content', '') if result else ''
        if text:
            self._summary_text = text.strip()
        self._summary_future = None

    def _maybe_summarize(self) -> None:
        """If messages exceed threshold, summarize old ones asynchronously."""
        if self._summary_future is not None:
            return  # already in flight
        if self._llm_model is None:
            return  # no LLM configured
        if len(self._messages) <= self._summary_threshold:
            return

        # Split: keep recent, summarize old
        keep_count = self._summary_threshold
        old_messages = self._messages[:-keep_count]
        self._messages = self._messages[-keep_count:]

        # Build summary prompt
        old_text = '\n'.join(
            f"[{m['from']} -> {m['to']}] ({m['message_type']}) {m['text']}"
            for m in old_messages
        )
        existing = f"Previous summary: {self._summary_text}\n\n" if self._summary_text else ''

        messages = [
            {"role": "system", "content": "Summarize the following agent communication history concisely. Focus on key information: help requests, discoveries, task updates, and commitments made."},
            {"role": "user", "content": f"{existing}New messages to incorporate:\n{old_text}"},
        ]

        from agents1.async_model_prompting import submit_llm_call
        llm_model = self._llm_model if self._llm_model.startswith('ollama/') else f'ollama/{self._llm_model}'
        self._summary_future = submit_llm_call(
            llm_model=llm_model,
            messages=messages,
            max_token_num=300,
            temperature=0.2,
            api_base=self._api_base,
        )

    # ── Prompt generation ─────────────────────────────────────────────────

    def get_messages_for_prompt(self, limit: int = 10, agent_busy: bool = False) -> List[dict]:
        """Return messages formatted for the reasoning prompt.

        1. Strategy filters (BusyAware may hide non-urgent messages)
        2. Take most recent `limit` messages
        3. If summary exists, prepend as context
        """
        filtered = self._strategy.filter_for_prompt(self._messages, agent_busy)
        recent = filtered[-limit:] if len(filtered) > limit else filtered

        result = []
        if self._summary_text:
            result.append({
                'from': 'system',
                'to': 'all',
                'message_type': 'message',
                'text': f'[Earlier conversation summary] {self._summary_text}',
            })
        result.extend(recent)
        return result

    # ── Communication stage prompt ───────────────────────────────────────

    def get_communication_prompt(self, messages: list, agent_context: dict = None) -> list:
        """Build a prompt to extract actionable information from incoming messages.

        Returns an OpenAI-style message list for the COMMUNICATION pipeline stage.
        Optionally includes agent context (position, task, carrying) for relevance assessment.
        """
        formatted = '\n'.join(
            f"[{m['from']} -> {m['to']}] ({m['message_type']}) {m['text']}"
            for m in messages
        )
        if agent_context:
            context_str = (
                f"Your current state:\n"
                f"- Position: {agent_context.get('position')}\n"
                f"- Current task: {agent_context.get('current_task')}\n"
                f"- Carrying: {agent_context.get('carrying')}\n"
            )
            formatted = context_str + "\nMessages:\n" + formatted
        return [
            {"role": "system", "content": _COMMUNICATION_PROMPT},
            {"role": "user", "content": formatted},
        ]

    # ── Query helpers ─────────────────────────────────────────────────────

    def has_pending_ask_help(self, from_agent: str) -> bool:
        """Check if there's a recent ask_help from `from_agent`.

        Used by _apply_communication() to decide auto-announce.
        """
        for msg in reversed(self._messages):
            if msg['from'] == from_agent and msg['message_type'] == 'ask_help':
                return True
        return False