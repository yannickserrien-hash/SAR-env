import re
from engine.llm_utils import query_llm_async, parse_json_response


class ReasoningBase:
    def __init__(self, profile_type_prompt, memory, _llm_model, prompts=None):
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self._llm_model = _llm_model[0]
        self.task_name_cache = None
        self._prompts = prompts  # YAML prompt dict (may be None)

    def process_task_description(self, task_description):
        task_name = re.findall(r'Your task is to:\s*(.*?)\s*>', task_description)
        if self.memory is not None:
            if self.task_name_cache is not None and self.task_name_cache == task_name:
                pass
            else:
                self.task_name_cache = task_name
                self.memory_cache = self.memory(task_description)
        else:
            self.memory_cache = ''
        split_text = task_description.rsplit('You are in the', 1)
        examples = split_text[0]
        task_description = 'You are in the' + split_text[1]

        return examples, task_description


class ReasoningIO(ReasoningBase):
    """LLM reasoning module that converts task descriptions into action decisions.

    If a ``prompts`` dict is provided (loaded from YAML), the system prompt
    is read from ``prompts['reasoning_system']``.  Otherwise a hardcoded
    fallback is used for backwards compatibility.
    """

    # Fallback system prompt (used when no YAML prompts are provided)
    _FALLBACK_SYSTEM_PROMPT = (
        "Role: You are RescueBot, an AI agent in a 25x24 grid search-and-rescue simulation.\n"
        "Objective: Find victims and carry them to the drop zone.\n\n"
        "Available actions and their parameters:\n"
        "- MoveTo: {'x': int, 'y': int} - navigate to a grid coordinate\n"
        "- RemoveObject: {'object_id': str} - remove a stone or tree blocking your path "
        "(you can do this ALONE, must be within 1 block)\n"
        "- RemoveObjectTogether: {'object_id': str} - remove a rock (requires BOTH you "
        "AND the human within 1 block)\n"
        "- OpenDoorAction: {'object_id': str} - open a closed door\n"
        "- CarryObject: {'object_id': str} - pick up a mildly injured victim alone\n"
        "- CarryObjectTogether: {'object_id': str} - pick up a victim together with human\n"
        "- Drop: {'object_id': str} - drop a carried victim\n"
        "- DropObjectTogether: {'object_id': str} - drop a victim carried together\n"
        "- Idle: {} - wait one tick\n\n"
        "Obstacle rules:\n"
        "- Stone: You can remove it ALONE with RemoveObject. Must be within 1 block.\n"
        "- Tree: You can remove it ALONE with RemoveObject. Must be within 1 block.\n"
        "- Rock: Requires BOTH agents. Use RemoveObjectTogether. Both must be within 1 block.\n"
        "- If navigation is blocked by a removable obstacle, REMOVE it instead of retrying MoveTo.\n"
        "- If blocked by a rock, ask the human for help or find an alternative route.\n\n"
        "Constraints: Respond ONLY in valid JSON.\n"
        "Schema: {'reasoning': str, 'action': str, 'params': dict}"
    )

    def __call__(self, task_description: str, feedback: str = ''):
        # Prefer YAML prompt; fall back to hardcoded string
        if self._prompts and 'reasoning_system' in self._prompts:
            system_prompt = self._prompts['reasoning_system'].strip()
        else:
            system_prompt = self._FALLBACK_SYSTEM_PROMPT

        user_prompt = task_description
        return query_llm_async(
            model=self._llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1000,
            temperature=0.1
        )
