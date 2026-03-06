"""
PlannerChannel: Thread-safe bidirectional communication between agents and EnginePlanner.

Agents push questions into an inbound queue; the planner drains them,
generates LLM-powered answers in background threads, and posts responses
into per-agent slots that agents poll each tick.
"""

import queue
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PlannerMessage:
    """A message from an agent to the EnginePlanner."""
    msg_id: str
    agent_id: str
    content: str
    tick: int
    context: dict = field(default_factory=dict)


@dataclass
class PlannerResponse:
    """A response from the EnginePlanner to an agent."""
    msg_id: str
    agent_id: str
    content: str
    tick: int


class PlannerChannel:
    """Thread-safe bidirectional channel between agents and EnginePlanner.

    Agents push PlannerMessages into the inbound queue.
    The planner drains the queue, processes questions, and places
    PlannerResponses into per-agent response slots.
    Agents poll their response slot each tick.
    """

    def __init__(self):
        self._inbound: queue.Queue = queue.Queue()
        self._responses: Dict[str, List[PlannerResponse]] = {}
        self._responses_lock = threading.Lock()
        self._seq = 0
        self._seq_lock = threading.Lock()

    # --- Agent-facing API ---

    def submit_question(self, agent_id: str, content: str, tick: int,
                        context: dict = None) -> str:
        """Agent submits a question. Returns the msg_id for tracking."""
        with self._seq_lock:
            self._seq += 1
            msg_id = f"{agent_id}_{tick}_{self._seq}"
        msg = PlannerMessage(
            msg_id=msg_id,
            agent_id=agent_id,
            content=content,
            tick=tick,
            context=context or {},
        )
        self._inbound.put_nowait(msg)
        return msg_id

    def poll_responses(self, agent_id: str) -> List[PlannerResponse]:
        """Agent polls for responses. Returns list and clears them."""
        with self._responses_lock:
            return self._responses.pop(agent_id, [])

    # --- Planner-facing API ---

    def drain_questions(self) -> List[PlannerMessage]:
        """Planner drains all pending questions. Returns list of PlannerMessage."""
        messages = []
        while True:
            try:
                messages.append(self._inbound.get_nowait())
            except queue.Empty:
                break
        return messages

    def post_response(self, response: PlannerResponse) -> None:
        """Planner posts a response for a specific agent."""
        with self._responses_lock:
            self._responses.setdefault(response.agent_id, []).append(response)
