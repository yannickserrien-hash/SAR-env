import re
from engine.llm_utils import query_llm_async, parse_json_response
from memory.short_term_memory import ShortTermMemory 


class ReasoningBase:
    def __init__(self, profile_type_prompt, memory: ShortTermMemory, llm_model, prompts=None):
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self.llm_model = llm_model[0]
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
    """
        LLM reasoning module that converts task descriptions into action decisions.
    """

    def __call__(self, task_description: str, observation, previous_action, world_state, feedback: str = ''):
        system_prompt = self._prompts['reasoning_system'].strip().format()
        user_prompt = self._build_reasoning_prompt(task_description, observation, previous_action, world_state)

        return query_llm_async(
            model=self.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=200,
            temperature=0.1
        )
        
    def _compact_memory(self, world_state: dict) -> str:
        """Serialize self.MEMORY to a short, lossless string.

        IDs are preserved for victims and obstacles (needed for action params).
        Doors and teammates are location-only (no ID needed for those actions).

        Format per line:
          V: <sym>[<id>]@<loc> ...   (victims, critical first)
          O: <sym>[<id>]@<loc> ...   (obstacles, big_rock first)
          D: <loc> ...               (door/house entrances)
          H: <loc> ...               (teammate positions)
          REQ: <sender>@<loc>:<msg>  (help requests, only if present)

        Symbols: !=critical  v=mild  o=healthy  R=big_rock  s=stone  t=tree
        """
        parts = []

        # Victims — critical first; id needed for CarryObject / CarryObjectTogether
        victims = world_state.get('known_victims', [])
        if victims:
            ORDER = {'critical': 0, 'mild': 1, 'healthy': 2}
            SYM   = {'critical': '!', 'mild': 'v', 'healthy': 'o'}
            sv = sorted(victims, key=lambda x: ORDER.get(x.get('type', 'healthy'), 2))
            tokens = [f"{SYM.get(v['type'], '?')}[{v['id']}]@{v['location']}" for v in sv]
            parts.append('V:' + ' '.join(tokens))

        # Obstacles — big_rock first; id needed for RemoveObject / RemoveObjectTogether
        obstacles = world_state.get('known_obstacles', [])
        if obstacles:
            ORDER = {'big_rock': 0, 'stone': 1, 'tree': 2}
            SYM   = {'big_rock': 'R', 'stone': 's', 'tree': 't'}
            so = sorted(obstacles, key=lambda x: ORDER.get(x.get('type', 'stone'), 1))
            tokens = [f"{SYM.get(o['type'], '?')}[{o['id']}]@{o['location']}" for o in so]
            parts.append('O:' + ' '.join(tokens))

        # Doors — location only (agent uses MoveTo, no id needed)
        doors = world_state.get('door_houses', [])
        if doors:
            tokens = [str(d['door']) for d in doors]
            parts.append('D:' + ' '.join(tokens))

        # Teammate — location only
        teammates = world_state.get('teammate_positions', {})
        if teammates:
            tokens = [str(v) for v in teammates.values()]
            parts.append('H:' + ' '.join(tokens))

        # Help requests — only include when non-empty
        help_reqs = world_state.get('pending_help_requests', [])
        if help_reqs:
            tokens = [
                f"{r.get('sender', '?')}@{r.get('location')}:{r.get('message', '')[:30]}"
                for r in help_reqs
            ]
            parts.append('REQ:' + ' | '.join(tokens))

        return '\n'.join(parts) if parts else '(empty)'

    def _build_reasoning_prompt(self, task_description, observation, previous_action, world_state) -> str:
        import json as _json

        # --- Extract agent pos + carrying from observation dict ---
        agent_info = observation.get('agent', {}) if isinstance(observation, dict) else {}
        agent_pos = agent_info.get('location', 'unknown')
        carrying = agent_info.get('carrying', [])
        carrying_str = ', '.join(carrying) if carrying else 'nothing'

        # --- Nearby objects: everything except the 'agent' key ---
        # Compact JSON kept here: location strings like "(5,6)" contain commas, which TOON
        # must quote, yielding no savings over already-compact JSON separators.
        nearby = {k: v for k, v in observation.items() if k != 'agent'} if isinstance(observation, dict) else {}
        nearby_json = _json.dumps(nearby, separators=(',', ':')) if nearby else '{}'

        # --- Compact world memory: lossless short format, no truncation needed ---
        if isinstance(world_state, dict) and world_state:
            memory_compact = self._compact_memory(world_state)
        elif self.memory.storage:
            memory_compact = self.memory.get_compact_str()[:400]
        else:
            memory_compact = '(empty)'

        if previous_action is None:
            previous_action = 'None'

        user_prompt = self._prompts['reasoning_user'].format(
            current_task=task_description,
            agent_pos=agent_pos,
            carrying=carrying_str,
            nearby_json=nearby_json,
            memory_compact=memory_compact,
            prev_action=previous_action,
        )
        print(user_prompt)

        return user_prompt
