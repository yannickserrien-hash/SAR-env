from typing import Dict, List, Any
from agents1.modules.utils_prompting import to_toon


REASONING_PROMPT = """Return tool calls to complete the tasks. Do not repeat action the same action if already taken recently.
If a task is completed, move to next one. Do not keep returning actions to complete the same task.
"""


class ReasoningBase:
    def __init__(self, plan):
        self.plan = plan


class ReasoningIO(ReasoningBase):
    def get_reasoning_prompt(self, information: Dict[str, Any]) -> List[Dict[str, str]]:
        observation = information.get('observation', {})
        task_decomposition = information.get('task_decomposition', '')
        feedback = information.get('feedback', '') or 'none'
        memory = information.get('memory', '') or 'none'
        previous_action = information.get('previous_action', '')

        info_dict: Dict[str, Any] = {
            "observation": observation,
            "tasks": task_decomposition,
            "feedback": feedback,
            "memory": memory,
        }
        if previous_action:
            info_dict["previous_action"] = previous_action

        return [
            {"role": "system", "content": REASONING_PROMPT},
            {"role": "user",   "content": to_toon(info_dict)},
        ]
