from typing import Dict, Any, Optional, List, Tuple


class Perception:
    """Converts a filtered MATRX WorldState dict to an LLM-friendly dict
    that can be passed to to_toon() for token-efficient prompting."""

    # Drop zone coordinates (matches WorldBuilder.py)
    DROP_ZONE_LOCATION: Tuple[int, int] = (23, 8)

    def percept_state(
        self,
        state: Dict[str, Any],
        agent_id: str,
        teammates: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Serialize filtered state to a structured dict for TOON encoding.

        Args:
            state:     Filtered MATRX state dict (from filter_observations).
            agent_id:  The agent's own ID (used to extract self-info).
            teammates: Set of (agent_id, location) tuples for all known teammates.

        Returns:
            Dict ready to be passed to to_toon().
        """
        # Extract plain teammate IDs for object classification.
        teammate_ids: set = {t[0] for t in teammates} if teammates else set()

        result = {
            "agent": self._serialize_agent(state, agent_id),
            "current_observation": self._serialize_nearby(state, agent_id, teammate_ids),
            # Use flat scalar fields so TOON's tabular format renders cleanly:
            #   teammates[1]{id,x,y}:
            #     rescuebot1,21,11
            # (A nested list or dict as a column value would be quoted by TOON.)
            "teammates": [
                {"id": t[0], "x": int(t[1][0]), "y": int(t[1][1])}
                for t in teammates
            ] if teammates else [],
        }
        return result

    # ── Private helpers ────────────────────────────────────────────────────

    def _serialize_agent(
        self, state: Dict[str, Any], agent_id: str
    ) -> Dict[str, Any]:
        """Extract self-position and carrying status."""
        agent_data = state.get(agent_id, {})
        loc = agent_data.get('location', [0, 0])
        raw_carrying = agent_data.get('is_carrying', [])
        carrying_ids = [
            obj.get('obj_id', str(obj)) if isinstance(obj, dict) else str(obj)
            for obj in raw_carrying
        ]
        return {
            "location": list(loc),
            "carrying": carrying_ids,
        }

    def _serialize_nearby(
        self,
        state: Dict[str, Any],
        agent_id: str,
        teammate_ids: set,
    ) -> List[Dict[str, Any]]:
        """Extract all interesting nearby objects."""
        skip = {agent_id, 'World'}
        objects: List[Dict[str, Any]] = []
        walls = {"blocked": []}

        for obj_id, obj_data in state.items():
            if obj_id in skip:
                continue
            if not isinstance(obj_data, dict):
                continue
            loc = obj_data.get('location')
            if loc is None:
                continue

            obj_type = self._classify_type(obj_id, obj_data, teammate_ids)
            if obj_type is None:
                continue

            if obj_type == 'door':
                continue
            #     obj_id = str(obj_id).split('_-_door')[0] if '_-_door' in str(obj_id) else str(obj_id)

            if 'wall' in obj_type:
                walls['blocked'].append([int(c) for c in loc])
                continue

            entry: Dict[str, Any] = {
                "id": obj_id,
                "type": obj_type,
                # Convert MATRX tuples (x, y) to lists so TOON renders
                # them as  location[2]: x,y  instead of "(x, y)".
                "location": [int(c) for c in loc],
            }

            # Add severity for victims so the LLM knows if coop carry is needed
            if obj_type == 'victim':
                img = str(obj_data.get('img_name', '')).lower()
                if 'critical' in img:
                    entry["severity"] = "critical"
                elif 'mild' in img:
                    entry["severity"] = "mild"
                else:
                    entry["severity"] = "healthy"

            objects.append(entry)
        objects.append(walls)

        return objects

    def _classify_type(
        self,
        obj_id: str,
        obj_data: Dict[str, Any],
        teammate_ids: set,
    ) -> Optional[str]:
        """Return a semantic type string, or None to skip the object.

        Types: 'victim', 'rock', 'stone', 'tree', 'door', 'agent', 'wall'
        """
        oid_lower = str(obj_id).lower()
        class_inh = obj_data.get('class_inheritance', [])

        # Human / other AI agents — check against plain ID set, not tuple set.
        if obj_id in teammate_ids:
            return 'agent'
        if 'AgentBody' in class_inh:
            return 'agent'

        # Victims (collectable)
        if obj_data.get('is_collectable', False):
            return 'victim'

        # Obstacles
        if 'ObstacleObject' in class_inh:
            if 'rock' in oid_lower:
                return 'rock'
            if 'tree' in oid_lower:
                return 'tree'
            return 'stone'

        # Doors
        if 'door' in oid_lower:
            return 'door'

        # Walls
        if 'Wall' in class_inh:
            return 'wall'

        if not obj_data.get('is_traversable', True):
            return 'blocked'

        return None
    
    
    def init_global_state(self) -> None:
        """Initialise the persistent world-knowledge store.

        Must be called once before ``process_observations`` is used
        (e.g. from ``SearchRescueAgent.initialize()``).
        """
        self.WORLD_STATE_GLOBAL: Dict[str, Any] = {
            'victims':   [],   # [{'id', 'severity', 'location'}]
            'obstacles': [],   # [{'id', 'type', 'location'}]
        }

    def update_observation(self, state) -> Dict[str, Any]:
        """Update and return the persistent global world state.

        Call every tick with the filtered state so that objects seen
        at any point in the episode are remembered even after the agent
        moves away.
        """
        if not hasattr(self, 'WORLD_STATE_GLOBAL'):
            self.init_global_state()
        if state is not None:
            self.add_new_obs(state)
        return self.WORLD_STATE_GLOBAL

    def add_new_obs(self, state) -> None:
        """Merge *state* into ``WORLD_STATE_GLOBAL``."""
        if state is None:
            return

        partner_name = getattr(self, '_partner_name', None)
        include_human = getattr(self, '_include_human', False)

        # Build the set of teammate IDs so _classify_type can recognise them.
        teammate_ids: set = set()

        # --- Teammate positions ---
        # if include_human and partner_name:
        #     human_info = state.get(partner_name)
        #     if isinstance(human_info, dict):
        #         hloc = human_info.get('location')
        #         if hloc is not None:
        #             self.WORLD_STATE_GLOBAL['teammate_positions'][partner_name] = [int(c) for c in hloc]
        #             teammate_ids.add(partner_name)

        # for obj_id, obj_data in state.items():
        #     if isinstance(obj_id, str) and obj_id.startswith('rescuebot') and obj_id != self.agent_id:
        #         if isinstance(obj_data, dict):
        #             tloc = obj_data.get('location')
        #             if tloc is not None:
        #                 self.WORLD_STATE_GLOBAL['teammate_positions'][obj_id] = [int(c) for c in tloc]
        #                 teammate_ids.add(obj_id)

        # --- Nearby objects ---
        skip_ids = {self.agent_id, 'World'} | teammate_ids
        if include_human and partner_name:
            skip_ids.add(partner_name)

        for obj_id, obj_data in state.items():
            if obj_id in skip_ids:
                continue
            if not isinstance(obj_data, dict):
                continue
            loc = obj_data.get('location')
            if loc is None:
                continue

            typ = self._classify_type(obj_id, obj_data, teammate_ids)
            if typ is None:
                continue

            pos = [int(c) for c in loc]

            # --- Victims ---
            if typ == 'victim':
                img = str(obj_data.get('img_name', '')).lower()
                if 'critical' in img:
                    severity = 'critical'
                elif 'mild' in img:
                    severity = 'mild'
                else:
                    severity = 'healthy'
                existing = next((v for v in self.WORLD_STATE_GLOBAL['victims'] if v['id'] == obj_id), None)
                if existing is None:
                    self.WORLD_STATE_GLOBAL['victims'].append(
                        {'id': obj_id, 'severity': severity, 'location': pos}
                    )
                else:
                    existing['location'] = pos

            # --- Obstacles ---
            elif typ in ('rock', 'stone', 'tree'):
                existing = next((o for o in self.WORLD_STATE_GLOBAL['obstacles'] if o['id'] == obj_id), None)
                if existing is None:
                    self.WORLD_STATE_GLOBAL['obstacles'].append(
                        {'id': obj_id, 'type': typ, 'location': pos}
                    )
                else:
                    existing['location'] = pos

            # --- Doors ---
            # elif typ == 'door':
            #     area_id = str(obj_id).split('_-_door')[0] if '_-_door' in str(obj_id) else str(obj_id)
            #     existing = next((d for d in self.WORLD_STATE_GLOBAL['doors'] if d['area'] == area_id), None)
            #     if existing is None:
            #         self.WORLD_STATE_GLOBAL['doors'].append(
            #             {'area': area_id, 'location': pos}
            #         )
            #     else:
            #         existing['location'] = pos
