"""
PlanningModule for LLM-MAS RescueAgent.

Decomposes high-level tasks from EnginePlanner into ordered sub-tasks.
Uses query_llm_async for non-blocking operation within the tick loop.
"""

import json
import logging
import concurrent.futures
from typing import Dict, List, Optional

from engine.llm_utils import query_llm_async

logger = logging.getLogger('PlanningModule')


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

    @property
    def is_planning(self) -> bool:
        """True if a planning LLM call is in flight."""
        return self._planning_future is not None

    @property
    def plan_ready(self) -> bool:
        """True if plan is populated."""
        return self._plan_ready

    @property
    def get_plan(self):
        return self.PLAN

    def plan(self, task, world_state: str = '', memory: str = '', feedback="") -> None:
        task_str = task if isinstance(task, str) else json.dumps(task, default=str)
        self._plan_ready = False

        system_prompt = self._prompts.get('planning_decompose_system', '').strip()

        user_prompt_template = self._prompts.get('planning_decompose_user', '').strip()
        user_prompt = (user_prompt_template
                       .replace('{task}', task_str)
                       .replace('{world_state}', world_state)
                       .replace('{memory}', memory)
                       .replace('{feedback}', feedback))

        self._planning_future = query_llm_async(
            model=self._llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_url=self._api_url,
        )

    def is_plan_ready(self) -> bool:
        """Non-blocking poll of the planning Future.  Call every tick.

        Returns True the tick the plan becomes ready, False otherwise.
        Also handles evaluation responses when _evaluating is True.
        """
        if self._planning_future is None:
            return False
        if not self._planning_future.done():
            return False

        try:
            self.PLAN = self._planning_future.result()
        except Exception as e:
            logger.warning(f"PlanningModule: planning future raised {e}, using original task")
        finally:
            self._planning_future = None
            self._plan_ready = True

        logger.info(f"PlanningModule: plan ready {self.PLAN}")
        return True

    def reset(self) -> None:
        """Clear all planning state."""
        self.PLAN = ''
        self._planning_future = None
        self._plan_ready = False