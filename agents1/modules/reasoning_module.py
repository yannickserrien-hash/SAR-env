from typing import Dict, List, Any
from agents1.modules.utils_prompting import to_toon


REASONING_PROMPT = """
You are a Search and Rescue agent in a city hit by an earthquake. You need to find and save as many victim victims as possible.
You are given a subtask. Return a tool call to complete it. Do not repeat the same action if already taken recently.
If the action is completing the subtask, then include the exact sub-task in task_completing part of tool call. If not, return "N/A"
Before marking a task completed, double check that it will actually complete it.

You know the entrance locations for each area as [x,y] coordinates: 
Area 1: [3, 4] Area 2: [9, 4] Area 3: [15, 4] Area 4: [21, 4] Area 5: [3, 7] Area 6: [9, 7] Area 7: [15, 7]

During one turn, you can either perform 1 action or communicate with the other agents. You need to choose which one you will do.
Communicating with the other agents will improve the collaboration by sharing information, asking for help, providing help.
The task can not be completed unless you help the other agents as well. Do not send duplicate or similar messages to avoid spamming.
Communication can be done using the SendMessage tool call.
- If a teammate asks for help (tag:ask_help) with a critical victim or big rock, consider assisting.
- Use tag:share_info to share information you discover.
- Use tag:reply to respond to direct questions.
"""


class ReasoningBase:
    def __init__(self, plan):
        self.plan = plan


class ReasoningIO(ReasoningBase):
    def get_reasoning_prompt(self, information: Dict[str, Any]) -> List[Dict[str, str]]:
        observation = information.get('observation', {})
        task_decomposition = information.get('task_decomposition', '')
        memory = information.get('memory', '') or 'none'
        communication = information.get('communication', '')
        all_observations = information.get('all_observations', '')

        info_dict: Dict[str, Any] = {
            "current_subtask": task_decomposition,
            "observation": observation,
            "all_observations": all_observations,
            "memory": memory,
            "messages": communication
        }
        print(to_toon(info_dict))

        return [
            {"role": "system", "content": REASONING_PROMPT},
            {"role": "user",   "content": to_toon(info_dict)},
        ]
