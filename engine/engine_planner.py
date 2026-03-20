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

from engine.toon_utils import to_toon

import yaml

from engine.iteration_data import IterationData
from agents1.async_model_prompting import call_llm_sync
from engine.parsing_utils import parse_json_response, load_few_shot
from engine.planner_channel import PlannerChannel, PlannerMessage, PlannerResponse

# Load all LLM prompts from the companion YAML file (once at import time).
_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), 'prompts_engine_planner.yaml')
with open(_PROMPTS_FILE, 'r') as _f:
    PROMPTS = yaml.safe_load(_f)

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
        llm_model: str = 'qwen3:8b',
        ticks_per_iteration: int = 100,
        include_human: bool = True,
        api_url: str = None,
        manual_plans_file: str = None,
    ):
        """
        Initialize EnginePlanner.

        Args:
            max_iterations: Maximum number of planning iterations
            score_file: Path to score.json for checking termination
            llm_model: Ollama model name for LLM calls
            ticks_per_iteration: Number of MATRX ticks per planning iteration
            include_human: Whether a human agent is present in the simulation
            api_url: Ollama base URL (None = default)
        """
        self.max_iterations = max_iterations
        self.score_file = score_file
        self.llm_model = f"ollama/{llm_model}" if not llm_model.startswith("ollama/") else llm_model
        self._api_base = api_url
        self.ticks_per_iteration = ticks_per_iteration
        self.include_human = include_human
        self.iteration_history: List[IterationData] = []
        self.logger = logging.getLogger('EnginePlanner')
        self._last_summary = ""
        self.world_state = {}  # latest world state received from channel

        # Background thread pool for async LLM calls (4 workers: planner + summarizer + prefetch + Q&A)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix='planner_worker'
        )
        # Prefetched task-generation future for the NEXT iteration.
        # Populated at the end of each iteration by submit_summarize().
        self._prefetch_future: Optional[concurrent.futures.Future] = None
        # Cached agent list so prefetch submissions don't need a caller-supplied list.
        self._agents_cache: list = []

        # Manual plans override (optional — replaces LLM task/plan generation)
        self._manual_data: Optional[dict] = None
        self._manual_iteration: int = 0
        if manual_plans_file:
            import yaml
            with open(manual_plans_file, 'r') as _mf:
                self._manual_data = yaml.safe_load(_mf)
            self.logger.info(f"[Planner] Loaded manual plans from {manual_plans_file}")

        # Agent ↔ Planner communication channel
        self._planner_channel: Optional[PlannerChannel] = None
        self._pending_answers: list = []  # list of (PlannerMessage, Future)

    @staticmethod
    def _strip_location_from_id(task_value):
        """Remove any '@...' location suffix from target_id inside a task dict.

        The LLM sometimes produces 'victim_1@[3,5]' despite explicit instructions.
        This strips everything from the '@' onward, leaving only the bare object id.
        Non-dict task values (e.g. plain strings) are returned unchanged.
        """
        if not isinstance(task_value, dict):
            return task_value
        raw_id = task_value.get('target_id')
        if isinstance(raw_id, str) and '@' in raw_id:
            task_value['target_id'] = raw_id.split('@')[0].strip()
        return task_value

    # ------------------------------------------------------------------
    # Agent ↔ Planner communication
    # ------------------------------------------------------------------

    def set_channel(self, channel: PlannerChannel):
        """Set the communication channel (called once during setup)."""
        self._planner_channel = channel
    
    def set_world_state(self, world_state: Dict[str, Any]):
        """Receive the current world state (called every tick from run_with_planner)."""
        self.world_state = world_state

    def _read_score_info(self) -> str:
        """Read current score data from score.json."""
        try:
            with open(self.score_file, 'r') as f:
                score_data = json.load(f)
                return (
                    f"Victims rescued: {score_data.get('victims_rescued', 0)}/"
                    f"{score_data.get('total_victims', 8)}, "
                    f"Score: {score_data.get('score', 0)}, "
                    f"Block hit rate: {score_data.get('block_hit_rate', 0.0):.2f}"
                )
        except (FileNotFoundError, json.JSONDecodeError):
            return "Score data unavailable"

    def _answer_question_sync(self, question: PlannerMessage) -> str:
        """Answer an agent's question using the planner's broad context.

        Runs synchronously in a background thread.
        """

        current_tasks = {}
        if self.iteration_history:
            current_tasks = self.iteration_history[-1].task_assignments
        elif self._agents_cache:
            current_tasks = {getattr(a, 'agent_id', '?'): '(pending)' for a in self._agents_cache}

        system_prompt = PROMPTS['answer_question_system']
        user_prompt = PROMPTS['answer_question_user'].format(
            agent_id=question.agent_id,
            question=question.content,
            world_state=to_toon(self.world_state),
            current_tasks=json.dumps(current_tasks, default=str),
        )

        response = call_llm_sync(
            llm_model=self.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_base=self._api_base,
            max_token_num=2000,
            temperature=0.2,
        )
        return response or "Unable to answer at this time."

    def process_agent_questions(self) -> None:
        """Drain agent questions, submit LLM answers, post completed responses.

        Called every tick from run_with_planner(). Non-blocking.
        """
        if self._planner_channel is None:
            return

        # Drain new questions and submit to executor immediately
        new_questions = self._planner_channel.drain_questions()
        for q in new_questions:
            future = self._executor.submit(self._answer_question_sync, q)
            self._pending_answers.append((q, future))
            self.logger.info(
                f"[Planner] Processing question from {q.agent_id}: {q.content[:80]}"
            )

        # Harvest completed answers and post responses
        still_pending = []
        for question, future in self._pending_answers:
            if future.done():
                try:
                    answer_text = future.result()
                except Exception as e:
                    answer_text = f"Error processing question: {e}"
                    self.logger.warning(f"[Planner] Answer future raised: {e}")

                response = PlannerResponse(
                    msg_id=question.msg_id,
                    agent_id=question.agent_id,
                    content=answer_text,
                    tick=question.tick,
                )
                self._planner_channel.post_response(response)
                self.logger.info(
                    f"[Planner] Answered {question.agent_id}: {answer_text[:80]}"
                )
            else:
                still_pending.append((question, future))
        self._pending_answers = still_pending

    # ------------------------------------------------------------------
    # Manual task/plan override
    # ------------------------------------------------------------------

    def _build_manual_tasks(self, agent_ids: list) -> Dict[str, Any]:
        """Build task assignments (and optional plans) from the manual plans file."""
        tasks: Dict[str, str] = {}
        plans: Dict[str, str] = {}

        # Per-iteration section: maps agent_id → {task, plan?}
        iterations = self._manual_data.get('iterations', [])
        if iterations:
            idx = min(self._manual_iteration, len(iterations) - 1)
            entry = iterations[idx]
            self._manual_iteration += 1
            for aid in agent_ids:
                if aid in entry:
                    agent_entry = entry[aid]
                    tasks[aid] = agent_entry.get('task', 'explore nearest area')
                    if 'plan' in agent_entry:
                        plans[aid] = agent_entry['plan']

        # agent_plans section: fallback plans keyed by agent_id
        for aid in agent_ids:
            if aid not in plans:
                agent_plan = self._manual_data.get('agent_plans', {}).get(aid)
                if agent_plan:
                    plans[aid] = agent_plan
            if aid not in tasks:
                tasks[aid] = 'explore nearest area'

        result: Dict[str, Any] = {'tasks': tasks}
        if plans:
            result['plans'] = plans
        self.logger.info(f"[Planner] Manual tasks: {tasks}")
        return result

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------

    def _generate_tasks_sync(self, agents: list) -> Dict[str, str]:
        """
        Synchronous core of task generation — called inside a background thread.

        Returns a dict with:
        - 'agent_tasks': {agent_id: task_string} for each AI agent
        - 'human_task': suggested task for human (if include_human)
        - 'reasoning': explanation of plan
        - 'rescuebot_task': (backward compat) shared fallback task
        """
        num_agents = len(agents)
        # Build per-agent ID list for the prompt
        agent_ids = [getattr(a, 'agent_id', f'rescuebot{i}') for i, a in enumerate(agents)]

        # Use manual plans file if configured (bypasses LLM entirely)
        if self._manual_data is not None:
            return self._build_manual_tasks(agent_ids)

        # Build a formatted agent listing for the prompt (include capabilities if available)
        agent_lines = []
        for i, aid in enumerate(agent_ids):
            caps = getattr(agents[i], '_capabilities', None) if i < len(agents) else None
            if caps:
                cap_str = ', '.join(f"{k}: {v}" for k, v in caps.items())
                agent_lines.append(f"  - {aid} ({cap_str})")
            else:
                agent_lines.append(f"  - {aid}")
        agent_ids_formatted = '\n'.join(agent_lines)
        agent_tasks_schema = ', '.join(f'"{aid}": "task for {aid}"' for aid in agent_ids)

        json_schema = f'{{"tasks": {{{agent_tasks_schema}}}"'
        if self.include_human:
            json_schema += ', "human_task": "suggested task for human"'

        system_prompt = PROMPTS['generate_tasks_system'].format(
            num_agents=num_agents
        )

        human_line = ""
        if self.include_human:
            human_line = "- Human"

        user_prompt = PROMPTS['generate_tasks_user'].format(
            world_state_summary=to_toon(self.world_state),
            num_agents=num_agents,
            json_schema = json_schema,
            agent_ids_formatted=agent_ids_formatted,
            human_line=human_line,
            previous_summary=(
                self._last_summary if self._last_summary
                else "First iteration."
            ),
        )
        
        response = call_llm_sync(
            llm_model=self.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_base=self._api_base,
            few_shot_messages=load_few_shot('generate_tasks'),
        )

        result = parse_json_response(response)
        print("LLM task generation response:", result)

        if result:
            # Accept new format (agent_tasks) or old format (rescuebot_task)
            if 'tasks' in result:
                # Sanitise: strip any "@location" suffix the LLM may have appended to target_id
                result['tasks'] = {
                    aid: self._strip_location_from_id(task)
                    for aid, task in result['tasks'].items()
                }
                return result
            if 'tasks' in result:
                return result

        # Fallback if LLM fails
        self.logger.warning("LLM task generation failed, using fallback")
        fallback = {
            'tasks': {aid: 'explore area 1 and find victims' for aid in agent_ids},
        }
        if self.include_human:
            fallback['human_task'] = 'Please explore areas and rescue victims'
        return fallback

    def _summarize_sync(self, iteration_data: IterationData,
                        world_state_summary: str) -> str:
        """
        Synchronous core of iteration summarization — called inside a background thread.
        """
        score_info = self._read_score_info()

        # Build task results text
        task_results_text = ""
        for result in iteration_data.task_results:
            task_results_text += (
                f"- {result.get('agent_id', '?')}: task='{result.get('task', '?')}' "
                f"status={result.get('result', {}).get('status', '?')}\n"
            )
        if not task_results_text:
            task_results_text = "No task results recorded this iteration."

        from engine.toon_utils import to_toon
        system_prompt = PROMPTS['summarize_system'].strip()
        user_prompt = PROMPTS['summarize_user'].format(
            task_assignments_json=to_toon(iteration_data.task_assignments),
            task_results_text=task_results_text,
            world_state_summary=world_state_summary,
            score_info=score_info,
        )

        response = call_llm_sync(
            llm_model=self.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_base=self._api_base,
            few_shot_messages=load_few_shot('summarize'),
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

    def submit_generate_tasks(self,
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
                if result and ('tasks' in result):
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
            self._generate_tasks_sync, agents
        )

    def request_new_task(self, world_state_summary: str,
                         agents: list) -> concurrent.futures.Future:
        """
        Agent-triggered mid-iteration task re-allocation.

        Submits a fresh LLM task generation call to the background executor and
        returns a Future immediately — never blocks the MATRX tick loop.
        The caller (RescueAgent) polls .done() each tick and calls
        set_current_task() with the result when the Future resolves.

        Bypasses the prefetch cache so the world state used is current.
        """
        if agents:
            self._agents_cache = agents
        self.logger.info("[Planner] Mid-iteration re-task requested by agent.")
        return self._executor.submit(
            self._generate_tasks_sync, agents
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
