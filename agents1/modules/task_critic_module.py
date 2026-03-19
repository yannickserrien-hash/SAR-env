from typing import Dict, List, Any
from agents1.modules.utils_prompting import to_toon


CRITIC_PROMPT = """
You are an assistant that assesses my progress of a search and rescue mission.
You are required to evaluate if I have met the task requirements. Exceeding the task requirements is also considered a success while failing to meet them requires you to provide critique to help me improve.

I will give you the following information:
Current subtask: the objective that I need to accomplish.
Last action attempted: the action that was just executed and its arguments. Evaluate whether this action moved us closer to completing the subtask.
Position: Agent's current position.
Nearby objects: The surrounding objects. Can be house walls, victims or obstacles.
Carrying: what victim I am carrying right now.
Rescued victims: which victims have been rescued.
You know the entrance locations for each area as [x,y] coordinates: 
Area 1: [3, 4] Area 2: [9, 4] Area 3: [15, 4] Area 4: [21, 4] Area 5: [3, 7] Area 6: [9, 7] Area 7: [15, 7]

You should only respond in JSON format as described below:
{
    "reasoning": "reasoning",
    "success": boolean,
    "critique": "critique",
}

Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc.

Here are some examples:
INPUT:
Position: [3,5]
Carrying: mildly_injured_woman
Nearby objects: none
Rescued victims: none
Last action: CarryObject(object_id="mildly_injured_woman")

Task: Pick up victim in at location [3, 5]

RESPONSE:
{
    "reasoning": "You are at location [3, 5], there are no objects around you and you are carrying a victim. The CarryObject action succeeded.",
    "success": true,
    "critique": ""
}

INPUT:
Position: [3,5]
Carrying: None
Nearby objects: mildly_injured_woman
Rescued victims: none
Last action: CarryObject(object_id="mildly_injured_woman")

Task: Pick up victim in at location [3, 5]

RESPONSE:
{
    "reasoning": "You are at location [3, 5], the victim is still on the ground. You are not carrying anything despite attempting CarryObject.",
    "success": false,
    "critique": "The CarryObject action failed. You should check if you can do this alone or jointly. Retry carrying the victim"
}

INPUT:
Position: [15,5]
Carrying: None
Nearby objects: stone_1 at [15,4]
Rescued victims: none
Last action: RemoveObject(object_id="stone_1")

Task: Remove obstacle blocking the door of Area 3

RESPONSE:
{
    "reasoning": "The door of Area 3 is at [15, 4]. The obstacle stone_1 is still in nearby objects, so RemoveObject did not succeed.",
    "success": false,
    "critique": "Given that the stone_1 is still observable in the environment, it means it was not removed. Try RemoveObjectTogether if it requires cooperation"
}

INPUT:
Position: [5,10]
Carrying: None
Nearby objects: critically_injured_man at [5,10]
Rescued victims: none
Last action: CarryObjectTogether(object_id="critically_injured_man")

Task: Carry critical victim at [5, 10] to drop zone cooperatively

RESPONSE:
{
    "reasoning": "CarryObjectTogether failed because no teammate is adjacent to the victim. Critical victims require two agents to carry together. The victim is still on the ground.",
    "success": false,
    "critique": "Ask a teammate for help using SendMessage with message_type ask_help. Include the victim ID and your location so the teammate can navigate to you. Wait for the teammate to arrive before retrying CarryObjectTogether."
}

"""


class CriticBase:
    def __init__(self, plan):
        self.plan = plan

    def get_critic_prompt(self, information: Dict[str, Any]) -> List[Dict[str, str]]:
        observation = information.get('observation', {})
        current_task = information.get('current_task', '')
        communication = information.get('communication', '')
        all_observations = information.get('all_observations', '')
        last_action = information.get('last_action', {})

        info_dict: Dict[str, Any] = {
            "current_task": current_task,
            "last_action": last_action,
            "observation": observation,
            "all_observations": all_observations,
            "messages": communication
        }
        print(to_toon(info_dict))

        return [
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user",   "content": to_toon(info_dict)},
        ]