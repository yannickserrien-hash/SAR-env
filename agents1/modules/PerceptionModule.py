from typing import Tuple, Optional

from matrx.agents.agent_utils.state import State


class PerceptionModule:
    """
        Perception Module that implements different strategies for converting the observations to input for the LLM-powered ReasoningModule.
    """

    TYPE_ACTION = {
        'crit':       'CarryObjectTogether',
        'mild':       'CarryObject',
        'healthy':    'CarryObject',
        'rock':       'RemoveObjectTogether',
        'stone':      'RemoveObject',
        'tree':       'RemoveObject',
        'door_open':  'Navigate',
    }

    @staticmethod
    def _classify_object(obj_id: str, obj_data: dict) -> Optional[str]:
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
            return None

        return typ

    def process_observations(self, state) -> dict:
        if state is None:
            return self.WORLD_STATE_FILTERED

        self.add_new_obs(state)
        self._last_obs_dict = self.WORLD_STATE_FILTERED
        return self.WORLD_STATE_FILTERED

    def add_new_obs(self, state) -> None:
        if state is None:
            return

        # --- Teammate positions ---
        if self._include_human:
            human_info = state.get(self._human_name, None)
            if human_info and isinstance(human_info, dict):
                hloc = human_info.get('location')
                if hloc is not None:
                    self.WORLD_STATE_FILTERED['teammate_positions'][self._human_name] = list(hloc)

        # Track other AI agent positions
        for obj_id, obj_data in state.items():
            if isinstance(obj_id, str) and obj_id.startswith('rescuebot') and obj_id != self.agent_id:
                if isinstance(obj_data, dict):
                    tloc = obj_data.get('location')
                    if tloc is not None:
                        self.WORLD_STATE_FILTERED['teammate_positions'][obj_id] = list(tloc)

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

            typ = self._classify_object(obj_id, obj_data)
            if typ is None:
                continue

            pos = list(loc)

            # --- Victims ---
            if typ in ('crit', 'mild', 'healthy'):
                victim_type = {'crit': 'critical', 'mild': 'mild', 'healthy': 'healthy'}[typ]
                existing = next(
                    (v for v in self.WORLD_STATE_FILTERED['victims'] if v['id'] == obj_id), None
                )
                if existing is None:
                    self.WORLD_STATE_FILTERED['victims'].append({
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
                    (o for o in self.WORLD_STATE_FILTERED['obstacles'] if o['id'] == obj_id), None
                )
                if existing is None:
                    self.WORLD_STATE_FILTERED['obstacles'].append({
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
                    (d for d in self.WORLD_STATE_FILTERED['doors'] if d['area'] == area_id), None
                )
                if existing is None:
                    self.WORLD_STATE_FILTERED['doors'].append({
                        'area': area_id,
                        'location': pos,
                    })
                else:
                    existing['location'] = pos

    def observation_to_dict(self, state) -> dict:
        """Convert a filtered state to a dict.
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
                typ = self._classify_object(obj_id, obj_data)

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
                    label = f"door:{area_id}"
                elif typ == 'wall':
                    label = f"wall"
                else:
                    continue 

                result.setdefault(label, []).append(loc_key)

        return result

    def _is_within_range(self, pos1: Tuple[int, int], pos2: Tuple[int, int], radius: int) -> bool:
        """Check if two positions are within a given radius using Chebyshev distance."""
        if pos1 is None or pos2 is None:
            return False
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) <= radius
