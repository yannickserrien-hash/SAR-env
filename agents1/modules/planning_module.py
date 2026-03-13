import json
import logging
from typing import Dict, List, Any, Optional
from agents1.modules.utils_prompting import to_toon

logger = logging.getLogger('Planning')


TASK_DECOMPOSITION_PROMPT = """
You are an expert planner for a team of agents in a search and rescue simulation.
Your job is to decompose high-level tasks into actionable subgoals for the agents.
"""


class Planning:
    def __init__(self) -> None:
        # Task decomposition
        self.task_decomposition = []

        # Clarification state (AskPlanner discussion loop)
        self._needs_clarification: bool = False
        self._question: str = ''
        self._feedback: str = ''           # accumulated Q&A from prior rounds
        self._clarification_round: int = 0
        self.current_task = ''

    def update_current_task(self, task: str) -> None:
        self.current_task = task

    def set_task_decomposition(self, decomposition: str) -> None:
        self.task_decomposition = decomposition

    def set_manual_task_decomposition(self, decomposition: List[str]) -> None:
        """Override PlanningModule by directly setting the plan text.

        Must be called after set_current_task() (which resets self.task_decomposition).
        """
        self.task_decomposition = decomposition

    def get_task_decomposition_prompt(
        self, information: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        world_state = information.get('world_state', {})
        memory = information.get('memory', '') or 'none'
        feedback = information.get('feedback', '') or 'none'

        info_dict: Dict[str, Any] = {
            "task": self.current_task,
            "world_state": world_state,
            "memory": memory,
            "feedback": feedback,
        }

        return [
            {"role": "system", "content": TASK_DECOMPOSITION_PROMPT},
            {"role": "user",   "content": to_toon(info_dict)},
        ]
