"""
EnginePlanner: Coordinates multi-agent task planning and termination decisions.

Based on MARBLE's EnginePlanner pattern, adapted for MATRX simulation.
Uses an LLM (via Ollama) to generate task assignments and summarize results.

LLM calls are non-blocking: each call is dispatched to a background thread via
concurrent.futures.ThreadPoolExecutor. The planner uses a prefetch strategy —
the next iteration's tasks are generated in the background while the current
iteration's ticks are running, so submit_generate_tasks() returns almost
instantly for every iteration after the first.
"""

import json
import logging
import os
import concurrent.futures
from typing import List, Dict, Any, Optional

import yaml

from engine.iteration_data import IterationData
from engine.llm_utils import query_llm, parse_json_response

# Load all LLM prompts from the companion YAML file (once at import time).
_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), 'prompts_engine_planner.yaml')
with open(_PROMPTS_FILE, 'r') as _f:
    PROMPTS = yaml.safe_load(_f)

DEFAULT_TASK_DESCRIPTION = PROMPTS['default_task_description'].strip()


class EnginePlanner:
    """
    Coordinates multi-agent task planning and termination decisions.

    Responsibilities:
    - Generate task assignments via LLM (async, prefetched)
    - Summarize iteration results via LLM (async, background)
    - Check termination conditions (block_hit_rate, max_iterations)
    - Track iteration history
    """

    def __init__(
        self,
        max_iterations: int = 100,
        score_file: str = "logs/score.json",
        llm_model: str = 'llama3:8b',
        task_description: str = '',
        ticks_per_iteration: int = 100
    ):
        """
        Initialize EnginePlanner.

        Args:
            max_iterations: Maximum number of planning iterations
            score_file: Path to score.json for checking termination
            llm_model: Ollama model name for LLM calls
            task_description: High-level mission description for LLM context
            ticks_per_iteration: Number of MATRX ticks per planning iteration
        """
        self.max_iterations = max_iterations
        self.score_file = score_file
        self.llm_model = llm_model
        self.task_description = task_description or DEFAULT_TASK_DESCRIPTION
        self.ticks_per_iteration = ticks_per_iteration
        self.iteration_history: List[IterationData] = []
        self.logger = logging.getLogger('EnginePlanner')
        self._last_summary = ""

        # Background thread pool for async LLM calls (3 workers: planner + summarizer + prefetch)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=3, thread_name_prefix='planner_worker'
        )
        # Prefetched task-generation future for the NEXT iteration.
        # Populated at the end of each iteration by submit_summarize().
        self._prefetch_future: Optional[concurrent.futures.Future] = None
        # Cached agent list so prefetch submissions don't need a caller-supplied list.
        self._agents_cache: list = []

    # ------------------------------------------------------------------
    # Private synchronous cores (run inside background threads)
    # ------------------------------------------------------------------

    def _generate_tasks_sync(self, world_state_summary: str, agents: list) -> Dict[str, str]:
        """
        Synchronous core of task generation — called inside a background thread.
        """
        num_agents = len(agents)

        system_prompt = PROMPTS['generate_tasks_system'].format(
            num_agents=num_agents
        )
        user_prompt = PROMPTS['generate_tasks_user'].format(
            task_description=self.task_description,
            world_state_summary=world_state_summary,
            num_agents=num_agents,
            previous_summary=(
                self._last_summary if self._last_summary
                else "This is the first iteration."
            ),
        )

        response = query_llm(
            model=self.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1000,
            temperature=0.2
        )

        result = parse_json_response(response)
        if result and 'rescuebot_task' in result:
            self.logger.info(f"  LLM task generation: {result.get('reasoning', '')}")
            return result

        # Fallback if LLM fails
        self.logger.warning("  LLM task generation failed, using fallback")
        return {
            'rescuebot_task': 'explore nearest unexplored area',
            'human_task': 'Please explore areas and rescue victims',
            'reasoning': 'Fallback: LLM unavailable'
        }

    def _summarize_sync(self, iteration_data: IterationData,
                        world_state_summary: str) -> str:
        """
        Synchronous core of iteration summarization — called inside a background thread.
        """
        # Read current score data
        score_info = ""
        try:
            with open(self.score_file, 'r') as f:
                score_data = json.load(f)
                score_info = (
                    f"Victims rescued: {score_data.get('victims_rescued', 0)}/"
                    f"{score_data.get('total_victims', 8)}, "
                    f"Score: {score_data.get('score', 0)}, "
                    f"Block hit rate: {score_data.get('block_hit_rate', 0.0):.2f}"
                )
        except (FileNotFoundError, json.JSONDecodeError):
            score_info = "Score data unavailable"

        # Build task results text
        task_results_text = ""
        for result in iteration_data.task_results:
            task_results_text += (
                f"- {result.get('agent_id', '?')}: task='{result.get('task', '?')}' "
                f"status={result.get('result', {}).get('status', '?')}\n"
            )
        if not task_results_text:
            task_results_text = "No task results recorded this iteration."

        system_prompt = PROMPTS['summarize_system'].strip()
        user_prompt = PROMPTS['summarize_user'].format(
            task_assignments_json=json.dumps(iteration_data.task_assignments, indent=2),
            task_results_text=task_results_text,
            world_state_summary=world_state_summary,
            score_info=score_info,
        )

        response = query_llm(
            model=self.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=200,
            temperature=0.5
        )

        if response:
            return response

        # Fallback: simple concatenation
        parts = [
            f"{r.get('agent_id', '?')}: "
            f"{r.get('task', '?')} -> "
            f"{r.get('result', {}).get('status', '?')}"
            for r in iteration_data.task_results
        ]
        return "; ".join(parts) if parts else "No results"

    # ------------------------------------------------------------------
    # Non-blocking API (returns Futures, never blocks the caller)
    # ------------------------------------------------------------------

    def submit_generate_tasks(self, world_state_summary: str,
                              agents: list) -> concurrent.futures.Future:
        """
        Submit task generation to a background thread. Returns a Future immediately.

        Never blocks the caller. The caller polls the returned Future
        via .done() and retrieves the result when ready.
        """
        if agents:
            self._agents_cache = agents

        # If a prefetch is available and already done, wrap it in a resolved Future
        if self._prefetch_future is not None and self._prefetch_future.done():
            prefetch = self._prefetch_future
            self._prefetch_future = None
            try:
                result = prefetch.result()
                if result and 'rescuebot_task' in result:
                    self.logger.info(
                        f"[Planner] Used prefetched tasks: {result.get('reasoning', '')}"
                    )
                    f = concurrent.futures.Future()
                    f.set_result(result)
                    return f
            except Exception as e:
                self.logger.error(f"[Planner] Prefetch future raised: {e}")

        # If prefetch exists but not done yet, return it (caller will poll)
        if self._prefetch_future is not None:
            future = self._prefetch_future
            self._prefetch_future = None
            return future

        # No prefetch available: submit fresh task generation
        return self._executor.submit(
            self._generate_tasks_sync, world_state_summary, agents
        )

    def submit_summarize(self, iteration_data: IterationData,
                         world_state_summary: str) -> concurrent.futures.Future:
        """
        Submit summarization to a background thread. Returns a Future immediately.

        Also pre-fetches the next iteration's task assignments.
        Never blocks the caller.
        """
        # Submit summarization
        summary_future = self._executor.submit(
            self._summarize_sync, iteration_data, world_state_summary
        )

        # Pre-fetch NEXT iteration's tasks in background
        agents_for_prefetch = self._agents_cache if self._agents_cache else []
        self._prefetch_future = self._executor.submit(
            self._generate_tasks_sync, world_state_summary, agents_for_prefetch
        )

        # Update _last_summary when the summary completes (callback runs on pool thread)
        def _on_summary_done(fut):
            try:
                self._last_summary = fut.result()
            except Exception:
                pass

        summary_future.add_done_callback(_on_summary_done)
        return summary_future

    def decide_next_step(self, iteration_data: IterationData) -> bool:
        """
        Determine if simulation should continue (rule-based, no LLM).

        Termination conditions:
        1. block_hit_rate >= 1.0 (all victims rescued)
        2. iteration >= max_iterations

        Args:
            iteration_data: Current iteration data

        Returns:
            True if simulation should continue, False to terminate
        """
        # Read score.json
        try:
            with open(self.score_file, 'r') as f:
                score_data = json.load(f)
                block_hit_rate = score_data.get('block_hit_rate', 0.0)
                iteration_data.block_hit_rate = block_hit_rate
                iteration_data.score = score_data.get('score', 0)

                self.logger.info(f"  Score: {iteration_data.score}, "
                               f"Block hit rate: {block_hit_rate:.2f}")

                # Check termination: all victims rescued
                if block_hit_rate >= 1.0:
                    self.logger.info("  All victims rescued (block_hit_rate == 1.0)")
                    return False
        except FileNotFoundError:
            self.logger.warning(f"  Score file {self.score_file} not found, continuing...")
        except json.JSONDecodeError:
            self.logger.warning(f"  Score file {self.score_file} invalid JSON, continuing...")

        # Check termination: max iterations
        if iteration_data.iteration >= self.max_iterations - 1:
            self.logger.info(f"  Max iterations ({self.max_iterations}) reached")
            return False

        return True

    def update_progress(self, iteration_data: IterationData):
        """
        Update cumulative progress tracking.

        Args:
            iteration_data: Current iteration data
        """
        self.logger.info(f"Iteration {iteration_data.iteration} complete. "
                        f"Score: {iteration_data.score}, "
                        f"Block hit rate: {iteration_data.block_hit_rate:.2f}")
