import re
from engine.llm_utils import (
    query_llm_async, query_llm_with_tools_async,
    parse_json_response, load_few_shot,
)
from agents1.agents_graveyard.matrx_tool_description import REASONING_TOOLS
from memory.short_term_memory import ShortTermMemory


class ReasoningBase:
    def __init__(self, llm_model, prompts=None, api_url=None):
        self.llm_model = llm_model
        self._prompts = prompts
        self._api_url = api_url


class ReasoningIO(ReasoningBase):
    """
        LLM reasoning module that converts task descriptions into action decisions.
        Uses Ollama tool calling: the model receives structured tool descriptions
        and returns a tool_call instead of free-form JSON.
    """

    def __call__(self, task_description: str, observation, previous_action,
                 world_state=None, feedback: str = ''):
        system_prompt = self._prompts['reason_system'].strip().format()
        user_prompt = self._build_reasoning_prompt(
            task_description, observation, previous_action, world_state, feedback)

        print("User Prompt:\n", user_prompt)

        return query_llm_with_tools_async(
            model=self.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=REASONING_TOOLS,
            api_url=self._api_url,
        )

    def _build_reasoning_prompt(self, task_description, observation, previous_action,
                                world_state=None, feedback: str = '') -> str:
        import json as _json

        # --- Extract agent pos + carrying from observation dict ---
        agent_info = observation.get('agent', {}) if isinstance(observation, dict) else {}
        agent_pos = agent_info.get('location', 'unknown')
        carrying = agent_info.get('carrying', [])
        carrying_str = ', '.join(carrying) if carrying else 'nothing'

        # --- Nearby objects: everything except the 'agent' key ---
        nearby = {k: v for k, v in observation.items() if k != 'agent'} if isinstance(observation, dict) else {}
        nearby_json = _json.dumps(nearby, separators=(',', ':')) if nearby else '{}'

        # --- World state: accumulated knowledge of the map ---
        if world_state and isinstance(world_state, dict):
            world_state_str = _json.dumps(world_state, separators=(',', ':'))
        else:
            world_state_str = '{}'

        if not previous_action:
            previous_action = 'None'
        else:
            previous_action = '\n  '.join(
                f"{i + 1}. {a}" for i, a in enumerate(previous_action)
            )

        user_prompt = self._prompts['reasoning_user'].format(
            current_task=task_description,
            agent_pos=agent_pos,
            carrying=carrying_str,
            nearby_json=nearby_json,
            prev_action=previous_action,
            action_feedback=feedback if feedback else 'none',
        )

        return user_prompt
