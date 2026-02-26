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
from typing import Any, Dict, List, Optional

from engine.llm_utils import query_llm_async, parse_json_response

logger = logging.getLogger('CommunicationModule')


class CommunicationModule:
    """Non-blocking communication module for RescueAgent.

    Uses a single generate_message() method for all outbound message types.
    The Reasoning Module decides when to communicate via LLM actions;
    this module handles the async message generation and delivery.
    """

    MAX_INBOUND_BATCH = 5

    def __init__(
        self,
        llm_model: str,
        prompts: dict,
        agent_id: str,
        send_message_fn,
        memory,
        world_memory: Optional[Dict[str, Any]] = None,
        api_url: str = None,
    ):
        """
        Args:
            llm_model: Ollama model name (e.g. 'llama3:8b').
            prompts: YAML prompt dict (from prompts_rescue_agent.yaml).
            agent_id: This agent's MATRX ID.
            send_message_fn: Reference to RescueAgent._send_message(content, sender).
            memory: Reference to RescueAgent.memory (ShortTermMemory).
            world_memory: Reference to RescueAgent.MEMORY (structured world-knowledge dict).
            api_url: Ollama base URL for this agent (None = default).
        """
        self._llm_model = llm_model
        self._prompts = prompts
        self._agent_id = agent_id
        self._send_message = send_message_fn
        self._memory = memory
        self._world_memory = world_memory
        self._api_url = api_url

        # Outbound: list of pending futures (supports concurrent messages)
        self._outbound_futures: List[concurrent.futures.Future] = []

        # Inbound: single future for sequential batch processing
        self._inbound_future: Optional[concurrent.futures.Future] = None
        self._last_processed_msg_index: int = 0

    # ------------------------------------------------------------------
    # Outbound: unified message generation
    # ------------------------------------------------------------------

    def generate_message(self, msg_type: str, tick: int, **kwargs) -> None:
        """
            Submit an async LLM call to generate a message.
        """
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
            max_tokens=120,
            temperature=0.2,
            api_url=self._api_url,
        )
        self._outbound_futures.append(future)

    def poll_outbound(self) -> None:
        """Harvest completed outbound futures and send messages.

        Called every tick. Iterates all pending futures, sends messages
        for completed ones, keeps pending ones for the next tick.
        """
        still_pending = []
        for future in self._outbound_futures:
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
                        self._send_message(msg_text, self._agent_id)
                        logger.info(f"Sent message: {msg_text[:80]}")
                except Exception as e:
                    logger.warning(f"Outbound message future failed: {e}")
            else:
                still_pending.append(future)
        self._outbound_futures = still_pending

    # ------------------------------------------------------------------
    # Inbound: parse incoming messages for actionable intelligence
    # ------------------------------------------------------------------

    def poll_inbound(self, received_messages: list) -> None:
        """Process new inbound messages via async LLM parsing.

        Called every tick. Checks for new messages since last processing,
        submits them to the LLM for extraction, and writes results to memory.

        Args:
            received_messages: RescueAgent.received_messages (list of Message).
        """
        # Harvest completed inbound future
        if self._inbound_future is not None and self._inbound_future.done():
            try:
                raw = self._inbound_future.result()
                parsed = parse_json_response(raw)
                if parsed:
                    self._process_parsed_inbound(parsed)
            except Exception as e:
                logger.warning(f"Inbound processing future failed: {e}")
            finally:
                self._inbound_future = None

        # Submit new batch if new messages and no pending parse
        if self._inbound_future is not None:
            return

        new_msgs = received_messages[self._last_processed_msg_index:]
        if not new_msgs:
            return

        batch = new_msgs[:self.MAX_INBOUND_BATCH]
        self._last_processed_msg_index += len(batch)

        msg_texts = []
        for m in batch:
            content = m.content if hasattr(m, 'content') else str(m)
            from_id = m.from_id if hasattr(m, 'from_id') else 'unknown'
            msg_texts.append(f"[{from_id}]: {content}")

        system_prompt = self._prompts.get('comm_inbound_system', '').format()
        user_template = self._prompts.get('comm_inbound_user', 'Parse: {messages}')
        user_prompt = user_template.format(messages='\n'.join(msg_texts))

        self._inbound_future = query_llm_async(
            model=self._llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=300,
            temperature=0.1,
            api_url=self._api_url,
        )

    def _process_parsed_inbound(self, parsed: dict) -> None:
        """Write extracted intel entries from parsed inbound LLM result to memory.

        Expected format:
            {"entries": [{"type": "intel_...", ...}, ...]}

        Entries of type "intel_help_requested" are additionally written into
        self._world_memory['pending_help_requests'] (if world_memory was provided).
        """
        entries = parsed.get('entries', [])
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            if not (isinstance(entry, dict) and entry.get('type')):
                continue
            self._memory.update('comm_intel', entry)

            # Mirror help requests into the structured MEMORY dict
            if entry.get('type') == 'intel_help_requested' and self._world_memory is not None:
                request = {
                    'location': entry.get('location'),
                    'message': entry.get('message', ''),
                    'sender': entry.get('sender', 'unknown'),
                }
                existing = self._world_memory.setdefault('pending_help_requests', [])
                # Avoid duplicates: skip if same sender+message already present
                is_dup = any(
                    r.get('sender') == request['sender'] and r.get('message') == request['message']
                    for r in existing
                )
                if not is_dup:
                    existing.append(request)

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
