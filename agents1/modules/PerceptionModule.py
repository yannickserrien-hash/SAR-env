from typing import Tuple, Optional

from matrx.agents.agent_utils.state import State


class PerceptionModule:
    """
        Perception Module that implements different strategies for converting the observations to input for the LLM-powered ReasoningModule.
    """

    def observation_to_text(self, state: State) -> str:
        """Convert filtered MATRX state to text for LLM prompt."""
        agent_info = state[self.agent_id]
        agent_loc = agent_info['location']

        parts = [f"Your position: {agent_loc}"]

        # Check if carrying
        is_carrying = agent_info.get('is_carrying', [])
        if is_carrying:
            carried_names = [obj.get('obj_id', str(obj)) if isinstance(obj, dict)
                           else str(obj) for obj in is_carrying]
            parts.append(f"Carrying: {carried_names}")
            self._carrying = True
        else:
            parts.append("Carrying: nothing")
            self._carrying = False

        # Teammate positions (human and other AI agents)
        if self._include_human:
            human_info = state.get('humanagent', None) or state.get(self._human_name, None)
            if human_info and isinstance(human_info, dict):
                parts.append(f"Human teammate '{self._human_name}' at {human_info.get('location', 'unknown')}")

        # Other AI agents in range
        for obj_id, obj_data in state.items():
            if obj_id == self.agent_id or obj_id == 'World':
                continue
            if not isinstance(obj_data, dict):
                continue
            class_inh = obj_data.get('class_inheritance', [])
            if isinstance(class_inh, list) and 'AgentBody' in class_inh and obj_id.startswith('rescuebot'):
                parts.append(f"AI teammate '{obj_id}' at {obj_data.get('location', 'unknown')}")

        # Nearby objects — use class_inheritance for reliable type detection
        nearby = []
        skip_ids = {self.agent_id, 'World'}
        if self._include_human:
            skip_ids.add('humanagent')
            skip_ids.add(self._human_name)
        for obj_id, obj_data in state.items():
            if obj_id in skip_ids or obj_id.startswith('rescuebot'):
                continue
            if not isinstance(obj_data, dict):
                continue
            loc = obj_data.get('location', 'unknown')
            img = obj_data.get('img_name', '')
            class_inh = obj_data.get('class_inheritance', [])
            is_traversable = obj_data.get('is_traversable', True)

            # --- Victims (collectable objects) ---
            if obj_data.get('is_collectable', False):
                victim_type = 'unknown'
                if 'critical' in str(img).lower():
                    victim_type = 'critically injured'
                elif 'mild' in str(img).lower():
                    victim_type = 'mildly injured'
                elif 'healthy' in str(img).lower():
                    victim_type = 'healthy'
                nearby.append(f"  Victim '{obj_id}' ({victim_type}) at {loc}")
            # --- Walls ---
            elif 'Wall' in class_inh:
                nearby.append(f"  Wall at {loc} [non-traversable]")
            # --- Doors ---
            elif 'Door' in class_inh:
                door_open = obj_data.get('is_open', True)
                door_name = str(obj_id).split('_-_door')[0] if '_-_door' in str(obj_id) else obj_id
                nearby.append(f"  Door '{door_name}' at {loc} ({'open' if door_open else 'closed, non-traversable'})")
            # --- Obstacles: type-specific info ---
            elif 'ObstacleObject' in class_inh:
                oid_lower = str(obj_id).lower()
                if 'rock' in oid_lower:
                    nearby.append(
                        f"  Rock '{obj_id}' at {loc} "
                        f"[requires BOTH agents to remove with RemoveObjectTogether]")
                elif 'stone' in oid_lower:
                    nearby.append(
                        f"  Stone '{obj_id}' at {loc} "
                        f"[removable by you alone with RemoveObject]")
                elif 'tree' in oid_lower:
                    nearby.append(
                        f"  Tree '{obj_id}' at {loc} "
                        f"[removable by you alone with RemoveObject]")
                else:
                    nearby.append(
                        f"  Obstacle '{obj_id}' at {loc} [non-traversable]")
            # --- Roof tiles (indicate agent is inside a room) ---
            elif 'roof' in str(obj_id).lower():
                nearby.append(f"  Roof at {loc}")
            # --- Any other non-traversable object ---
            elif not is_traversable:
                nearby.append(f"  Blocked object '{obj_id}' at {loc} [non-traversable]")
            # --- Skip traversable non-interesting objects (AreaTile, floor, etc.) ---

        if nearby:
            parts.append("Nearby objects (within 1 block):")
            parts.extend(nearby)
        else:
            parts.append("Nearby objects: none visible")

        # Recent messages
        if hasattr(self, 'received_messages') and self.received_messages:
            msg_texts = [m.content for m in self.received_messages[-3:]]
            parts.append(f"Recent messages: {msg_texts}")

        return "\n".join(parts)

    def observation_to_map(self) -> str:
        """Build a persistent 25x24 ASCII grid map from self.state_from_engine.

        Updates self._known_cells (spatial memory) each call, then renders the
        full grid with coordinate axes, a legend, and a status line.  Unknown
        cells are shown as '?' — the LLM may assume they are traversable until
        proven otherwise.  Returns a string used as {state_text} in the
        reasoning prompt and as {observation} in the memory-extraction prompt.

        Symbol priority (higher-priority symbols overwrite lower ones):
          @  You (agent)           — priority 0  (highest)
          H  Human teammate        — priority 1
          !  Critically injured    — priority 2
          v  Mildly injured        — priority 3
          o  Healthy victim        — priority 4
          R  Rock                  — priority 5
          s  Stone                 — priority 6
          t  Tree                  — priority 7
          #  Wall                  — priority 8
          +  Door (closed)         — priority 9
          '  Door (open)           — priority 10
          D  Drop zone             — priority 11
          .  Explored / empty      — priority 12
          ?  Unknown               — priority 13  (lowest / default)
        """
        GRID_W = 23
        GRID_H = 22
        DROP_OFF_SIZE = 8
        PRIORITY = {
            '@': 0, 'H': 1, 'A': 1, '!': 2, 'v': 3, 'o': 4,
            'R': 5, 's': 6, 't': 7, '#': 8,
            '+': 9, "'": 10, 'D': 11, '.': 12, '?': 13,
        }

        # ------------------------------------------------------------------
        # Drop zone: x=23, y=8..15
        # ------------------------------------------------------------------
        drop_x = 23
        for dy in range(1, 9):
            cell = (drop_x, (GRID_H - DROP_OFF_SIZE) / 2 + dy)
            existing = self._known_cells.get(cell, '?')
            if PRIORITY.get(existing, 13) >= PRIORITY['D']:
                self._known_cells[cell] = 'D'

        # ------------------------------------------------------------------
        # Phase 2 — Stale-entry eviction
        # ------------------------------------------------------------------
        if self.state_from_engine is not None:
            current_ids = set(self.state_from_engine.keys())
            stale = [
                cell for cell, oid in list(self._known_cell_owners.items())
                if oid not in current_ids
            ]
            for cell in stale:
                del self._known_cell_owners[cell]
                if self._known_cells.get(cell) not in ('#', 'D'):
                    self._known_cells[cell] = '.'

        # ------------------------------------------------------------------
        # Phase 2.5 — Clear stale agent/human ghost symbols
        # ------------------------------------------------------------------
        if self.state_from_engine is not None:
            _agent_info = self.state_from_engine.get(self.agent_id, {})
            _cur_agent_loc = None
            if isinstance(_agent_info, dict):
                _aloc = _agent_info.get('location')
                if _aloc is not None:
                    _cur_agent_loc = (int(_aloc[0]), int(_aloc[1]))

            # Clear stale self-agent ghost
            _prev_agent = getattr(self, '_prev_agent_loc', None)
            if _prev_agent is not None and _prev_agent != _cur_agent_loc:
                if self._known_cells.get(_prev_agent) == '@':
                    self._known_cells[_prev_agent] = '.'
            self._prev_agent_loc = _cur_agent_loc

            # Clear stale human ghost (only if human exists)
            if self._include_human:
                _human_info = self.state_from_engine.get('humanagent', {})
                if not isinstance(_human_info, dict):
                    _human_info = self.state_from_engine.get(self._human_name, {})
                _cur_human_loc = None
                if isinstance(_human_info, dict):
                    _hloc = _human_info.get('location')
                    if _hloc is not None:
                        _cur_human_loc = (int(_hloc[0]), int(_hloc[1]))
                _prev_human = getattr(self, '_prev_human_loc', None)
                if _prev_human is not None and _prev_human != _cur_human_loc:
                    if self._known_cells.get(_prev_human) == 'H':
                        self._known_cells[_prev_human] = '?'
                self._prev_human_loc = _cur_human_loc

            # Clear stale AI teammate ghosts
            _prev_teammates = getattr(self, '_prev_teammate_locs', {})
            _cur_teammate_locs = {}
            for _tid, _tdata in self.state_from_engine.items():
                if _tid == self.agent_id or _tid == 'World' or not isinstance(_tdata, dict):
                    continue
                if isinstance(_tid, str) and _tid.startswith('rescuebot'):
                    _tloc = _tdata.get('location')
                    if _tloc is not None:
                        _cur_teammate_locs[_tid] = (int(_tloc[0]), int(_tloc[1]))
            for _tid, _ploc in _prev_teammates.items():
                _cloc = _cur_teammate_locs.get(_tid)
                if _ploc != _cloc and self._known_cells.get(_ploc) == 'A':
                    self._known_cells[_ploc] = '.'
            self._prev_teammate_locs = _cur_teammate_locs

        # ------------------------------------------------------------------
        # Phase 3 — Update from current state_from_engine
        # ------------------------------------------------------------------
        if self.state_from_engine is not None:
            for obj_id, obj_data in self.state_from_engine.items():
                if obj_id == 'World':
                    continue
                if not isinstance(obj_data, dict):
                    continue
                loc = obj_data.get('location')
                if loc is None:
                    continue
                x, y = int(loc[0]), int(loc[1])
                if not (0 < x <= GRID_W and 0 < y <= GRID_H):
                    continue

                cell = (x, y)
                class_inh = obj_data.get('class_inheritance', [])
                img = str(obj_data.get('img_name', ''))
                oid_lower = str(obj_id).lower()

                if obj_id == self.agent_id:
                    symbol = '@'
                elif obj_id == 'humanagent' or (self._include_human and obj_id == self._human_name):
                    symbol = 'H'
                elif isinstance(obj_id, str) and obj_id.startswith('rescuebot') and obj_id != self.agent_id:
                    symbol = 'A'
                elif obj_data.get('is_collectable', False):
                    if 'critical' in img.lower():
                        symbol = '!'
                    elif 'mild' in img.lower():
                        symbol = 'v'
                    else:
                        symbol = 'o'
                elif 'ObstacleObject' in class_inh:
                    if 'rock' in oid_lower:
                        symbol = 'R'
                    elif 'stone' in oid_lower:
                        symbol = 's'
                    elif 'tree' in oid_lower:
                        symbol = 't'
                    else:
                        symbol = 's'
                elif 'Wall' in class_inh:
                    symbol = '#'
                elif 'Door' in class_inh:
                    symbol = "'" if obj_data.get('is_open', True) else '+'
                elif obj_data.get('is_drop_zone', False):
                    symbol = 'D'
                elif not obj_data.get('is_traversable', True):
                    symbol = '#'
                else:
                    if self._known_cells.get(cell, '?') == '?':
                        self._known_cells[cell] = '.'
                    continue

                existing = self._known_cells.get(cell, '?')
                if PRIORITY.get(symbol, 13) <= PRIORITY.get(existing, 13):
                    self._known_cells[cell] = symbol
                    if symbol not in ('#', "'", '+', 'D'):
                        self._known_cell_owners[cell] = obj_id

            # Phase 4 — Mark Chebyshev-1 neighbourhood around agent as explored
            agent_info = self.state_from_engine.get(self.agent_id, {})
            if isinstance(agent_info, dict):
                aloc = agent_info.get('location')
                if aloc is not None:
                    ax, ay = int(aloc[0]), int(aloc[1])
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nc = (ax + dx, ay + dy)
                            if 0 < nc[0] <= GRID_W and 0 < nc[1] <= GRID_H:
                                if self._known_cells.get(nc, '?') == '?':
                                    self._known_cells[nc] = '.'

        # ------------------------------------------------------------------
        # Phase 5 — Render grid
        # ------------------------------------------------------------------
        rows = []
        for y in range(1, GRID_H + 1):
            row_chars = [self._known_cells.get((x, y), '?') for x in range(1, GRID_W + 1)]
            rows.append(''.join(row_chars))

        legend = (
            "Legend: @=You  H=Human  A=AI teammate  !=critical victim  v=mild victim  o=healthy victim\n"
            "        R=rock(coop)  s=stone(solo)  t=tree(solo)  #=wall\n"
            "        +=door(closed)  '=door(open)  D=dropzone  .=explored  ?=unknown"
        )

        status_lines = []
        if self.state_from_engine is not None:
            agent_info = self.state_from_engine.get(self.agent_id, {})
            if isinstance(agent_info, dict):
                aloc = agent_info.get('location', 'unknown')
                is_carrying = agent_info.get('is_carrying', [])
                if is_carrying:
                    carried_names = [
                        obj.get('obj_id', str(obj)) if isinstance(obj, dict) else str(obj)
                        for obj in is_carrying
                    ]
                    self._carrying = True
                    carry_str = f"carrying {carried_names}"
                else:
                    self._carrying = False
                    carry_str = "carrying nothing"
                status_lines.append(f"You: position={aloc}, {carry_str}")

            if self._include_human:
                human_info = self.state_from_engine.get('humanagent', {})
                if not isinstance(human_info, dict):
                    human_info = self.state_from_engine.get(self._human_name, {})
                if isinstance(human_info, dict) and human_info.get('location'):
                    hloc = human_info.get('location', 'unknown')
                    status_lines.append(f"Human '{self._human_name}': position={hloc}")

            # Other AI teammates in state
            for _tid, _tdata in self.state_from_engine.items():
                if isinstance(_tid, str) and _tid.startswith('rescuebot') and _tid != self.agent_id:
                    if isinstance(_tdata, dict) and _tdata.get('location'):
                        status_lines.append(f"AI teammate '{_tid}': position={_tdata['location']}")

        if hasattr(self, 'received_messages') and self.received_messages:
            msg_texts = [m.content for m in self.received_messages[-3:]]
            status_lines.append(f"Recent messages: {msg_texts}")

        grid_str = '\n'.join(rows)
        status_str = '\n'.join(status_lines)
        return f"{grid_str}\n{legend}\n{status_str}"

    TYPE_ACTION = {
        'crit':       'CarryObjectTogether',
        'mild':       'CarryObject',
        'healthy':    'CarryObject',
        'rock':       'RemoveObjectTogether',
        'stone':      'RemoveObject',
        'tree':       'RemoveObject',
        'door_open':  'Navigate',
        'door_closed':'Navigate',
    }

    @staticmethod
    def _classify_object(obj_id: str, obj_data: dict) -> Tuple[Optional[str], Optional[str]]:
        """Classify a MATRX object into (type_str, action_str).

        Returns (None, None) for uninteresting objects (walls, area tiles, etc.).
        """
        if not isinstance(obj_data, dict):
            return None, None

        img = obj_data.get('img_name', '')
        class_inh = obj_data.get('class_inheritance', [])
        oid_lower = str(obj_id).lower()

        if obj_data.get('is_collectable', False):
            img_l = str(img).lower()
            if 'critical' in img_l:
                typ = 'crit'
            elif 'mild' in img_l:
                typ = 'mild'
            else:
                typ = 'healthy'
        elif 'ObstacleObject' in class_inh:
            if 'rock' in oid_lower:
                typ = 'rock'
            elif 'stone' in oid_lower:
                typ = 'stone'
            elif 'tree' in oid_lower:
                typ = 'tree'
            else:
                typ = 'stone'
        elif 'Door' in class_inh:
            typ = 'door_open' if obj_data.get('is_open', True) else 'door_closed'
        elif 'Wall' in class_inh:
            typ = 'wall'
        elif not obj_data.get('is_traversable', True):
            typ = 'blocked'
        else:
            return None, None

        action = PerceptionModule.TYPE_ACTION.get(typ)
        return typ, action

    def observation_to_json(self, state) -> dict:
        """Return the current MEMORY dict (updated via add_new_obs each tick).

        The MEMORY dict is the single source of truth for the agent's world
        knowledge and is passed to the LLM as structured context.

        Schema:
          {
            "known_victims": [
              {"id": "v1", "type": "critical"|"mild"|"healthy",
               "location": [x, y], "rescued": false}
            ],
            "known_obstacles": [
              {"id": "r1", "type": "big_rock"|"stone"|"tree", "location": [x, y]}
            ],
            "door_houses": [
              {"id": "Area 1", "door": [x, y]}
            ],
            "pending_help_requests": [
              {"location": [x, y], "message": "...", "sender": "agent_id"}
            ],
            "teammate_positions": {"agent_id": [x, y]}
          }
        """
        if state is None:
            return self.MEMORY

        self.add_new_obs(state)
        self._last_obs_dict = self.MEMORY
        return self.MEMORY

    def add_new_obs(self, state) -> None:
        """Merge a filtered MATRX state snapshot into self.MEMORY.

        Updates agent_position, carrying, teammate_positions, and accumulates
        known_victims, known_obstacles, and door_houses from newly visible objects.
        Existing entries are updated in-place (matched by id); new ones are appended.
        Call this every tick (or whenever a new filtered state is available).
        """
        if state is None:
            return

        # --- Teammate positions ---
        if self._include_human:
            human_info = state.get(self._human_name, None)
            if human_info and isinstance(human_info, dict):
                hloc = human_info.get('location')
                if hloc is not None:
                    self.MEMORY['teammate_positions'][self._human_name] = list(hloc)

        # Track other AI agent positions
        for obj_id, obj_data in state.items():
            if isinstance(obj_id, str) and obj_id.startswith('rescuebot') and obj_id != self.agent_id:
                if isinstance(obj_data, dict):
                    tloc = obj_data.get('location')
                    if tloc is not None:
                        self.MEMORY['teammate_positions'][obj_id] = list(tloc)

        # --- Nearby objects ---
        skip_ids = {self.agent_id, 'World'}
        if self._include_human:
            skip_ids.add(self._human_name)
        for obj_id, obj_data in state.items():
            if obj_id in skip_ids or (isinstance(obj_id, str) and obj_id.startswith('rescuebot')):
                continue
            if not isinstance(obj_data, dict):
                continue
            loc = obj_data.get('location')
            if loc is None:
                continue

            typ, _ = self._classify_object(obj_id, obj_data)
            if typ is None:
                continue

            pos = list(loc)

            # --- Victims ---
            if typ in ('crit', 'mild', 'healthy'):
                victim_type = {'crit': 'critical', 'mild': 'mild', 'healthy': 'healthy'}[typ]
                existing = next(
                    (v for v in self.MEMORY['known_victims'] if v['id'] == obj_id), None
                )
                if existing is None:
                    self.MEMORY['known_victims'].append({
                        'id': obj_id,
                        'type': victim_type,
                        'location': pos,
                    })
                else:
                    existing['location'] = pos

            # --- Obstacles ---
            elif typ in ('rock', 'stone', 'tree'):
                obstacle_type = {'rock': 'big_rock', 'stone': 'stone', 'tree': 'tree'}[typ]
                existing = next(
                    (o for o in self.MEMORY['known_obstacles'] if o['id'] == obj_id), None
                )
                if existing is None:
                    self.MEMORY['known_obstacles'].append({
                        'id': obj_id,
                        'type': obstacle_type,
                        'location': pos,
                    })
                else:
                    existing['location'] = pos

            # --- Doors (area entrances) ---
            elif typ in ('door_open', 'door_closed'):
                area_id = str(obj_id).split('_-_door')[0] if '_-_door' in str(obj_id) else str(obj_id)
                existing = next(
                    (d for d in self.MEMORY['door_houses'] if d['id'] == area_id), None
                )
                if existing is None:
                    self.MEMORY['door_houses'].append({
                        'id': area_id,
                        'door': pos,
                    })
                else:
                    existing['door'] = pos

    def state_to_json(self, state) -> dict:
        """Convert a filtered MATRX State snapshot to a simple {label: [loc_keys]} dict.

        At most ~8 objects are visible at any tick (Chebyshev-1 filter).
        Obstacles (stone/tree/rock) additionally carry an "approach:<id>" key listing
        the free adjacent cells the agent can MoveTo in order to then act on them.
        Agent, human, and world metadata are included as special top-level keys.
        """
        if state is None:
            return {}

        result = {}

        agent_info = state[self.agent_id]
        agent_location = agent_info['location']

        # Collect all non-traversable locations in the current state for approach filtering
        blocked_locs = set()
        for oid, odata in state.items():
            if not isinstance(odata, dict):
                continue
            if odata.get('is_traversable', True):
                continue
            loc = odata.get('location')
            if loc is not None:
                blocked_locs.add((int(loc[0]), int(loc[1])))

        for obj_id, obj_data in state.items():

            if obj_id == 'World':
                continue
            if not isinstance(obj_data, dict):
                continue
            loc = obj_data.get('location')

            # --- Agent self ---
            if obj_id == self.agent_id:
                is_carrying = obj_data.get('is_carrying', [])
                carrying_ids = [
                    o.get('obj_id', str(o)) if isinstance(o, dict) else str(o)
                    for o in is_carrying
                ]
                result['agent'] = {
                    'location': list(loc) if loc else None,
                    'carrying': carrying_ids,
                }
                continue

            # --- Human teammate ---
            if self._include_human and (obj_id == self._human_name or obj_id == 'humanagent'):
                result['human'] = list(loc) if loc else None
                continue

            # --- Other AI teammates ---
            if isinstance(obj_id, str) and obj_id.startswith('rescuebot') and obj_id != self.agent_id:
                teammates = result.setdefault('teammates', {})
                teammates[obj_id] = list(loc) if loc else None
                continue

            if loc is None:
                continue

            if self._is_within_range(agent_location, obj_data.get('location'), radius=1):

                loc_key = str(tuple(int(c) for c in loc))
                typ, action = self._classify_object(obj_id, obj_data)

                if typ == 'crit':
                    label = f"critical_victim:{obj_id}"
                elif typ == 'mild':
                    label = f"mild_victim:{obj_id}"
                elif typ == 'healthy':
                    label = f"healthy_victim:{obj_id}"
                elif typ == 'rock':
                    label = f"rock:{obj_id}"
                elif typ == 'stone':
                    label = f"stone:{obj_id}"
                elif typ == 'tree':
                    label = f"tree:{obj_id}"
                elif typ == 'door_open':
                    area_id = str(obj_id).split('_-_door')[0] if '_-_door' in str(obj_id) else str(obj_id)
                    label = f"{area_id}"
                elif typ == 'door_closed':
                    area_id = str(obj_id).split('_-_door')[0] if '_-_door' in str(obj_id) else str(obj_id)
                    label = f"door_closed:{area_id}"
                elif typ == 'wall':
                    label = f"wall"
                elif typ == 'blocked':
                    label = f"blocked:{obj_id}"
                else:
                    continue  # skip uninteresting objects

                result.setdefault(label, []).append(loc_key)

        return result

    def _is_within_range(self, pos1: Tuple[int, int], pos2: Tuple[int, int], radius: int) -> bool:
        """Check if two positions are within a given radius using Chebyshev distance."""
        if pos1 is None or pos2 is None:
            return False
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) <= radius
