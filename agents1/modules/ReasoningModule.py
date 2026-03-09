import re
from engine.llm_utils import query_llm_async, parse_json_response, load_few_shot
from memory.short_term_memory import ShortTermMemory


class ReasoningBase:
    def __init__(self, llm_model, prompts=None, api_url=None):
        self.llm_model = llm_model
        self._prompts = prompts 
        self._api_url = api_url


class ReasoningIO(ReasoningBase):
    """
        LLM reasoning module that converts task descriptions into action decisions.
    """

    def __call__(self, task_description: str, observation, previous_action,
                 feedback: str = ''):
        system_prompt = self._prompts['reason_system'].strip().format()
        user_prompt = self._build_reasoning_prompt(
            task_description, observation, previous_action)

        return query_llm_async(
            model=self.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_url=self._api_url,
            few_shot_messages=load_few_shot('reasoning'),
        )

    def _build_reasoning_prompt(self, task_description, observation, previous_action) -> str:
        import json as _json

        # --- Extract agent pos + carrying from observation dict ---
        agent_info = observation.get('agent', {}) if isinstance(observation, dict) else {}
        agent_pos = agent_info.get('location', 'unknown')
        carrying = agent_info.get('carrying', [])
        carrying_str = ', '.join(carrying) if carrying else 'nothing'

        # --- Nearby objects: everything except the 'agent' key ---
        nearby = {k: v for k, v in observation.items() if k != 'agent'} if isinstance(observation, dict) else {}
        nearby_json = _json.dumps(nearby, separators=(',', ':')) if nearby else '{}'

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
        )

        return user_prompt
