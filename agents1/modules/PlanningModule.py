"""
PlanningModule for LLM-MAS RescueAgent.

Decomposes high-level tasks from EnginePlanner into ordered sub-tasks.
Uses query_llm_async for non-blocking operation within the tick loop.

If the LLM response contains "AskPlanner", the module signals that it needs
clarification from the EnginePlanner. The caller (RescueAgent) submits the
question via PlannerChannel, waits for an answer, and calls receive_answer()
to trigger a re-plan with the planner's feedback.
"""

import json
import logging
import concurrent.futures
from typing import Dict, List, Optional

from engine.llm_utils import query_llm_async

logger = logging.getLogger('PlanningModule')

MAX_CLARIFICATION_ROUNDS = 3


class PlanningModule:

    def __init__(
        self,
        llm_model: str,
        prompts: dict,
        memory=None,
        api_url: str = None,
    ):
        self._llm_model = llm_model
        self._prompts = prompts
        self._memory = memory
        self._api_url = api_url

        # Plan state
        self.PLAN = ''
        self._planning_future: Optional[concurrent.futures.Future] = None
        self._plan_ready: bool = False

        # Clarification state (AskPlanner discussion loop)
        self._needs_clarification: bool = False
        self._question: str = ''
        self._feedback: str = ''           # accumulated Q&A from prior rounds
        self._clarification_round: int = 0

        # Stored args for re-plan after receiving planner answer
        self._last_task: str = ''
        self._last_world_state: str = ''
        self._last_memory: str = ''

    @property
    def is_planning(self) -> bool:
        """True if a planning LLM call is in flight."""
        return self._planning_future is not None

    @property
    def plan_ready(self) -> bool:
        """True if plan is populated."""
        return self._plan_ready

    @property
    def needs_clarification(self) -> bool:
        """True if the LLM returned AskPlanner and a question is pending."""
        return self._needs_clarification

    @property
    def get_plan(self):
        return self.PLAN

    def get_question(self) -> str:
        """Return the pending question for the planner."""
        return self._question

    def plan(self, task, world_state: str = '', memory: str = '', feedback: str = '') -> None:
        """Submit async LLM call for task decomposition.

        Args:
            task: High-level task string from EnginePlanner.
            world_state: Current world state (TOON or text).
            memory: Compact memory string.
            feedback: Accumulated planner Q&A (empty on first call).
        """
        task_str = task if isinstance(task, str) else json.dumps(task, default=str)

        # Store args for re-plan
        self._last_task = task_str
        self._last_world_state = world_state
        self._last_memory = memory

        self._plan_ready = False
        self._needs_clarification = False

        system_prompt = self._prompts.get('planning_decompose_system', '').strip()

        user_prompt_template = self._prompts.get('planning_decompose_user', '').strip()
        user_prompt = (user_prompt_template
                       .replace('{task}', task_str)
                       .replace('{world_state}', world_state)
                       .replace('{memory}', memory)
                       .replace('{feedback}', feedback if feedback else '(none)'))

        self._planning_future = query_llm_async(
            model=self._llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_url=self._api_url,
        )

    def is_plan_ready(self) -> bool:
        """Non-blocking poll of the planning Future. Call every tick.

        Returns True the tick the plan becomes ready, False otherwise.
        If the LLM returned AskPlanner, sets needs_clarification and returns False.
        """
        if self._planning_future is None:
            return False
        if not self._planning_future.done():
            return False

        raw = None
        try:
            raw = self._planning_future.result()
        except Exception as e:
            logger.warning(f"PlanningModule: planning future raised {e}")
        finally:
            self._planning_future = None

        # Check if LLM wants to ask the planner for more info
        if (raw and 'AskPlanner' in raw
                and self._clarification_round < MAX_CLARIFICATION_ROUNDS):
            # Extract question text (everything after "AskPlanner")
            idx = raw.index('AskPlanner')
            question = raw[idx + len('AskPlanner'):].strip().strip(':').strip()
            if not question:
                question = raw.strip()  # fallback: use entire response
            self._question = question
            self._needs_clarification = True
            logger.info(f"PlanningModule needs clarification: {question}")
            return False

        # Valid plan (or fallback)
        if raw:
            self.PLAN = raw
        else:
            self.PLAN = '1. Explore nearest area'
        self._plan_ready = True
        self._needs_clarification = False
        logger.info(f"PlanningModule: plan ready: {self.PLAN[:100]}")
        return True

    def receive_answer(self, answer: str) -> None:
        """Receive the planner's answer and re-plan with accumulated feedback.

        Args:
            answer: The planner's response to the agent's question.
        """
        self._clarification_round += 1
        self._feedback += f"\nQ: {self._question}\nA: {answer}\n"
        self._needs_clarification = False
        self._question = ''

        # Re-plan with accumulated feedback
        self.plan(
            task=self._last_task,
            world_state=self._last_world_state,
            memory=self._last_memory,
            feedback=self._feedback,
        )

    def reset(self) -> None:
        """Clear all planning state for a new task."""
        self.PLAN = ''
        self._planning_future = None
        self._plan_ready = False
        self._needs_clarification = False
        self._question = ''
        self._feedback = ''
        self._clarification_round = 0
        self._last_task = ''
        self._last_world_state = ''
        self._last_memory = ''
