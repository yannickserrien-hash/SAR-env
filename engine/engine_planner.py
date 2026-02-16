"""
EnginePlanner: Coordinates multi-agent task planning and termination decisions.

Based on MARBLE's EnginePlanner pattern, adapted for MATRX simulation.
Uses an LLM (via Ollama) to generate task assignments and summarize results.
"""

import json
import logging
from typing import List, Dict, Any
from engine.iteration_data import IterationData
from engine.llm_utils import query_llm, parse_json_response

# TODO: Customize this task description based on your project needs.
# This is sent to the LLM as context for generating task assignments.
DEFAULT_TASK_DESCRIPTION = (
    "You are coordinating a search and rescue mission in a 25x24 grid world. "
    "There are 14 areas (rooms) that may contain victims. Victims must be found "
    "and carried to the drop-off zone at x=23, y=8-15. "
    "There are 8 target victims: 4 critically injured (+6 pts each) and "
    "4 mildly injured (+3 pts each). "
    "Critically injured victims require BOTH agents to carry together. "
    "Mildly injured victims can be carried by one agent alone. "
    "Obstacles include: trees (RescueBot removes alone), small stones "
    "(either agent removes alone), and big rocks (require both agents). "
    "The human agent is controlled via keyboard and acts independently."
)


class EnginePlanner:
    """
    Coordinates multi-agent task planning and termination decisions.

    Responsibilities:
    - Generate task assignments via LLM
    - Summarize iteration results via LLM
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

    def generate_tasks(self, world_state_summary: str, agents: list) -> Dict[str, str]:
        """
        Use LLM to generate subtask assignments for agents.

        Args:
            world_state_summary: Text description of current world state
            agents: List of RescueAgent brain instances

        Returns:
            Dict with 'rescuebot_task', 'human_task', and 'reasoning' keys
        """
        num_agents = len(agents)

        system_prompt = (
            "You are the planning coordinator for a search and rescue mission. "
            "You assign high-level subtasks to agents based on the current situation. "
            "Always respond with valid JSON only, no other text."
        )

        user_prompt = f"""## Mission Description
{self.task_description}

## Current World State
{world_state_summary}

## Available Agents
- RescueBot (AI agent, count: {num_agents}): Can explore areas, carry mildly injured victims alone, remove trees and small stones alone, cooperate with human for critical victims and big rocks.
- Human (keyboard-controlled): Acts independently. You can suggest a task but cannot control them.

## Previous Iteration Summary
{self._last_summary if self._last_summary else "This is the first iteration."}

## Instructions
Assign the next high-level subtask to each agent. Consider what areas might have been explored, what victims may have been found, and what obstacles might remain.

Respond in JSON format:
{{
  "rescuebot_task": "description of RescueBot's next subtask",
  "human_task": "suggested task for human (sent as chat message)",
  "reasoning": "brief explanation of your plan"
}}"""

        response = query_llm(
            model=self.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=300,
            temperature=0.7
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

    def summarize_output(self, iteration_data: IterationData,
                         world_state_summary: str = '') -> str:
        """
        Use LLM to summarize what happened in this iteration.

        Args:
            iteration_data: Current iteration data
            world_state_summary: Text description of current world state

        Returns:
            Summary string
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

        system_prompt = (
            "You are summarizing the results of one planning iteration in a "
            "search and rescue simulation. Be concise (under 100 words)."
        )

        user_prompt = f"""## Task Assignments This Iteration
{json.dumps(iteration_data.task_assignments, indent=2)}

## Task Results
{task_results_text}

## Current World State
{world_state_summary}

## Score
{score_info}

Provide a concise summary of what happened and what remains to be done."""

        response = query_llm(
            model=self.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=200,
            temperature=0.5
        )

        if response:
            self._last_summary = response
            return response

        # Fallback: simple concatenation
        parts = []
        for result in iteration_data.task_results:
            parts.append(
                f"{result.get('agent_id', '?')}: "
                f"{result.get('task', '?')} -> "
                f"{result.get('result', {}).get('status', '?')}"
            )
        fallback = "; ".join(parts) if parts else "No results"
        self._last_summary = fallback
        return fallback

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
