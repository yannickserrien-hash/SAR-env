from typing import Dict, List, Any
from agents1.modules.utils_prompting import to_toon


REASONING_PROMPT = """
You are a Search and Rescue agent in a city hit by an earthquake. Your goal is to find and rescue as many victims as possible.

You are given a subtask. Return a tool call to advance or complete it.
- Do not repeat the same action from your last 3 actions (visible in memory).
- Every tool call has a `task_completing` field. Set it to the exact subtask text if this action completes the subtask. Otherwise set it to "N/A".
- Before marking a task completed, verify from your observation that it is actually done.
- If your subtask involves sending a message, use SendMessage with the appropriate `message_type` ("ask_help", "help", or "message").
"""


class ReasoningBase:
    def __init__(self, plan):
        self.plan = plan


class ReasoningIO(ReasoningBase):
    def get_reasoning_prompt(self, information: Dict[str, Any]) -> List[Dict[str, str]]:
        observation = information.get('observation', {})
        task_decomposition = information.get('task_decomposition', '')
        memory = information.get('memory', '') or 'none'
        critic_feedback = information.get('critic_feedback', '')

        info_dict: Dict[str, Any] = {
            "current_subtask": task_decomposition,
            "observation": observation,
            "memory": memory,
            "critic_feedback": critic_feedback,
        }
        print(to_toon(info_dict))

        return [
            {"role": "system", "content": REASONING_PROMPT},
            {"role": "user",   "content": to_toon(info_dict)},
        ]
