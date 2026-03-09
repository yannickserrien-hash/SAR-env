"""
CommunicationModule for LLM-MAS RescueAgent.

Handles all inter-agent communication through a unified generate_message() method:
1. Observation broadcasts (BroadcastObservation action)
2. Help requests (SendMessage action)
3. Help acceptance (SendAcceptHelpMessage action)
4. Inbound message parsing (automatic, extracts intel to memory)

All LLM calls are non-blocking via query_llm_async() and futures.
Multiple outbound messages can be in-flight simultaneously.
"""

import json
import logging
import concurrent.futures
from typing import Any, Dict, List

from engine.llm_utils import query_llm_async, parse_json_response, load_few_shot

logger = logging.getLogger('CommunicationModule')


class CommunicationModule:

    def __init__(
        self,
        llm_model: str,
        prompts: dict,
        agent_id: str,
        send_message_fn,
        api_url: str = None,
    ):
        self._llm_model = llm_model
        self._prompts = prompts
        self._agent_id = agent_id
        self._send_message = send_message_fn
        self._api_url = api_url

        # Outbound: list of pending futures (supports concurrent messages)
        self._outbound_futures: List[concurrent.futures.Future] = []

        # Inbound: raw message store (no LLM parsing)
        self._stored_messages: List[dict] = []
        self._last_processed_msg_index: int = 0

    # ------------------------------------------------------------------
    # Outbound: unified message generation
    # ------------------------------------------------------------------

    def has_pending_llm(self) -> bool:
        """True if any outbound LLM future is still in-flight."""
        return any(
            not (f[0] if isinstance(f, tuple) else f).done()
            for f in self._outbound_futures
        )

    def generate_message(self, msg_type: str, tick: int, **kwargs) -> None:
        """
            Submit an async LLM call to generate a message.
        """
        if self.has_pending_llm():
            logger.debug("Outbound LLM call pending — skipping new message generation")
            return
        system_prompt = self._prompts.get('comm_generate_system', '').format()
        user_prompt_key = f'comm_{msg_type}_user'
        user_template = self._prompts.get(user_prompt_key, '')

        try:
            user_prompt = user_template.format(tick=tick, **kwargs)
        except KeyError as e:
            logger.warning(f"Missing prompt variable {e} for {user_prompt_key}")
            return

        future = query_llm_async(
            model=self._llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_url=self._api_url,
            few_shot_messages=load_few_shot('communication'),
        )
        self._outbound_futures.append((future, None))

    def generate_direct_message(self, target_agent_id: str, tick: int,
                                message_intent: str, context: str = '') -> None:
        """Submit an async LLM call to generate a direct message to a specific agent."""
        if self.has_pending_llm():
            logger.debug("Outbound LLM call pending — skipping new direct message generation")
            return
        system_prompt = self._prompts.get('comm_direct_system', '').format()
        user_template = self._prompts.get('comm_direct_user', '')
        try:
            user_prompt = user_template.format(
                tick=tick,
                target_agent=target_agent_id,
                message_intent=message_intent,
                context=context,
            )
        except KeyError as e:
            logger.warning(f"Missing prompt variable {e} for comm_direct_user")
            return

        future = query_llm_async(
            model=self._llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_url=self._api_url,
            few_shot_messages=load_few_shot('communication'),
        )
        self._outbound_futures.append((future, target_agent_id))

    def poll_outbound(self) -> None:
        """Harvest completed outbound futures and send messages.

        Called every tick. Iterates all pending futures, sends messages
        for completed ones, keeps pending ones for the next tick.
        Supports targeted messages via (future, target_id) tuples.
        """
        still_pending = []
        for item in self._outbound_futures:
            if isinstance(item, tuple):
                future, target_id = item
            else:
                future, target_id = item, None

            if future.done():
                try:
                    raw = future.result()
                    parsed = parse_json_response(raw)
                    if parsed and 'message' in parsed:
                        msg_text = parsed['message']
                    elif raw:
                        msg_text = raw.strip()[:200]
                    else:
                        msg_text = None

                    if msg_text:
                        self._send_message(msg_text, self._agent_id, target_id=target_id)
                        kind = 'direct' if target_id else 'broadcast'
                        logger.info(f"Sent {kind} message: {msg_text[:80]}")
                except Exception as e:
                    logger.warning(f"Outbound message future failed: {e}")
            else:
                still_pending.append(item)
        self._outbound_futures = still_pending

    # ------------------------------------------------------------------
    # Inbound: parse incoming messages for actionable intelligence
    # ------------------------------------------------------------------

    def poll_inbound(self, received_messages: list, action_count: int = 0) -> None:
        """Store new inbound messages directly (no LLM parsing).

        Called every tick. Appends any new messages to self._stored_messages.

        Args:
            received_messages: RescueAgent.received_messages (list of Message).
            action_count: Unused; kept for API compatibility.
        """
        new_msgs = received_messages[self._last_processed_msg_index:]
        if not new_msgs:
            return

        for m in new_msgs:
            content = m.content if hasattr(m, 'content') else str(m)
            from_id = m.from_id if hasattr(m, 'from_id') else 'unknown'
            self._stored_messages.append({'from_id': from_id, 'content': content})
            logger.debug(f"Stored inbound message from {from_id}: {content[:80]}")

        self._last_processed_msg_index += len(new_msgs)

    # ------------------------------------------------------------------
    # Context for reasoning prompt
    # ------------------------------------------------------------------

    def get_communication_context(self, received_messages: list) -> str:
        """Build a compact string of recent messages for inclusion in the
        reasoning prompt.

        Args:
            received_messages: RescueAgent.received_messages (list of Message).

        Returns:
            Formatted string of recent messages, or "(no recent messages)".
        """
        recent = received_messages[-5:] if received_messages else []
        if not recent:
            return "(no recent messages)"

        lines = []
        for m in recent:
            content = m.content if hasattr(m, 'content') else str(m)
            from_id = m.from_id if hasattr(m, 'from_id') else '?'
            lines.append(f"[{from_id}]: {content}")

        return '\n'.join(lines)
