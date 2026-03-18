"""
CommunicationModule for LLM-MAS RescueAgent.

Template-based inter-agent communication (no LLM calls):
1. Observation broadcasts (BroadcastObservation action)
2. Help requests (SendMessage action)
3. Help acceptance (SendAcceptHelpMessage action)
4. Direct messages (SendDirectMessage action)

All messages are sent synchronously and logged to a shared message log
that persists across all agents for the entire game.
"""

import logging
from typing import Any, Dict, List, Optional
import concurrent.futures

logger = logging.getLogger('CommunicationModule')

# Premade templates for each message type
TEMPLATES = {
    'broadcast': "[Tick {tick}] {agent_id} | Task: {current_task} | {observation_json}",
    'help_request': "[HelpRequest][Tick {tick}] Need help at {target_location}. Action: {action_needed}. Object: {object_id}",
    'accept_help': "[HelpAccepted][Tick {tick}] On my way from {agent_location}. Re: {help_message}",
    'direct': "[Tick {tick}] {message_intent}",
}


class CommunicationModule:

    def __init__(
        self,
        agent_id: str,
        send_message_fn,
        shared_message_log: list,
    ):
        self._agent_id = agent_id
        self._send_message = send_message_fn
        self._shared_message_log = shared_message_log

        # Outbound: list of pending futures (supports concurrent messages)
        self._outbound_futures: List[concurrent.futures.Future] = []

        # Inbound: raw message store (no LLM parsing)
        self._stored_messages: List[dict] = []
        self._last_processed_msg_index: int = 0

    # ------------------------------------------------------------------
    # Outbound: template-based message sending
    # ------------------------------------------------------------------

    def send_templated_message(self, msg_type: str, tick: int, **kwargs) -> None:
        """Format a premade template and send immediately.

        Args:
            msg_type: One of 'broadcast', 'help_request', 'accept_help'.
            tick: Current simulation tick.
            **kwargs: Template variables (e.g. current_task, observation_json).
        """
        template = TEMPLATES.get(msg_type)
        if template is None:
            logger.warning(f"Unknown message type: {msg_type}")
            return

        try:
            content = template.format(tick=tick, agent_id=self._agent_id, **kwargs)
        except KeyError as e:
            logger.warning(f"Missing template variable {e} for msg_type={msg_type}")
            return

        # Truncate to 200 chars for MATRX compatibility
        content = content[:200]

        # Send via MATRX (appears in game chat UI)
        self._send_message(content, self._agent_id, target_id=None)

        # Log to shared message log
        self._shared_message_log.append({
            'tick': tick,
            'sender': self._agent_id,
            'recipient': 'all',
            'content': content,
            'msg_type': msg_type,
        })
        logger.info(f"Sent {msg_type} message: {content[:80]}")

    def send_direct_message(self, target_agent_id: str, tick: int,
                            message_intent: str, context: str = '') -> None:
        """Send a direct message to a specific agent using a template.

        Args:
            target_agent_id: ID of the target agent.
            tick: Current simulation tick.
            message_intent: The message text/intent (provided by Reasoning module).
            context: Additional context (currently unused in template).
        """
        template = TEMPLATES['direct']
        content = template.format(tick=tick, message_intent=message_intent)
        content = content[:200]

        # Send via MATRX (appears in game chat UI)
        self._send_message(content, self._agent_id, target_id=target_agent_id)

        # Log to shared message log
        self._shared_message_log.append({
            'tick': tick,
            'sender': self._agent_id,
            'recipient': target_agent_id,
            'content': content,
            'msg_type': 'direct',
        })
        logger.info(f"Sent direct message to {target_agent_id}: {content[:80]}")

    # ------------------------------------------------------------------
    # Inbound: track new messages (no LLM parsing)
    # ------------------------------------------------------------------

    def poll_inbound(self, received_messages: list) -> None:
        """Track new inbound messages. No LLM parsing — just updates the index.

        Args:
            received_messages: RescueAgent.received_messages (list of Message).
        """
        self._last_processed_msg_index = len(received_messages)

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

    def get_message_log_context(self, last_n: int = 10) -> str:
        """Return a formatted string of the last N entries from the shared
        message log for inclusion in reasoning prompts.

        Args:
            last_n: Number of recent log entries to include.

        Returns:
            Formatted string of recent log entries, or "(no message history)".
        """
        if not self._shared_message_log:
            return "(no message history)"

        recent = self._shared_message_log[-last_n:]
        lines = []
        for entry in recent:
            recipient = entry.get('recipient', 'all')
            to_str = f"to {recipient}" if recipient != 'all' else 'to all'
            lines.append(
                f"[Tick {entry['tick']}] {entry['sender']} {to_str}: {entry['content']}"
            )
        return '\n'.join(lines)
