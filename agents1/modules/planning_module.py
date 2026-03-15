import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
from agents1.modules.utils_prompting import to_toon

logger = logging.getLogger('Planning')


TASK_DECOMPOSITION_PROMPT = """
You are an expert planner for a team of agents in a search and rescue simulation.
Your job is to decompose high-level tasks into actionable subgoals for the agents.
"""

# ── TaskGraph data structures ─────────────────────────────────────────────────


class TaskStatus(Enum):
    PENDING = 'pending'
    ACTIVE = 'active'
    COMPLETED = 'completed'
    SKIPPED = 'skipped'


@dataclass
class TaskNode:
    """A single node in the task graph."""
    id: int
    description: str
    status: TaskStatus = TaskStatus.PENDING
    is_condition: bool = False
    condition_action: str = ''
    next_id: Optional[int] = None

    def full_description(self) -> str:
        """Description including the conditional action (for LLM prompts)."""
        if self.is_condition and self.condition_action:
            return f"{self.description}. If so, {self.condition_action}"
        return self.description


# Regex patterns for detecting conditional tasks
_COND_INLINE = re.compile(
    r'^(.*?)\.\s*[Ii]f\s+(.+?),\s*(.+)$'
)
_COND_SUBBULLET = re.compile(
    r'\n\s*-\s*[Ii]f\s+(.+?),\s*(.+)$'
)


class TaskGraph:
    """Directed acyclic graph of tasks with conditional branching support."""

    def __init__(self) -> None:
        self._nodes: Dict[int, TaskNode] = {}
        self._head_id: Optional[int] = None

    @classmethod
    def from_task_list(cls, tasks: List[str]) -> 'TaskGraph':
        """Build a TaskGraph from a flat list of task description strings.

        Detects conditional patterns (e.g. "Check for X. If present, Y")
        and marks those nodes as condition nodes with a condition_action.
        """
        graph = cls()
        if not tasks:
            return graph

        nodes: List[TaskNode] = []
        for idx, raw in enumerate(tasks):
            node_id = idx + 1
            # Try sub-bullet conditional first (multiline)
            m = _COND_SUBBULLET.search(raw)
            if m:
                # Parent description is everything before the sub-bullet
                desc = raw[:m.start()].strip().rstrip('.')
                action = m.group(2).strip()
                nodes.append(TaskNode(
                    id=node_id,
                    description=desc,
                    is_condition=True,
                    condition_action=action,
                ))
                continue

            # Try inline conditional
            m = _COND_INLINE.match(raw.strip())
            if m:
                desc = m.group(1).strip()
                action = m.group(3).strip()
                nodes.append(TaskNode(
                    id=node_id,
                    description=desc,
                    is_condition=True,
                    condition_action=action,
                ))
                continue

            # Normal task
            nodes.append(TaskNode(id=node_id, description=raw.strip()))

        # Link nodes sequentially
        for i in range(len(nodes) - 1):
            nodes[i].next_id = nodes[i + 1].id

        # Populate graph
        for node in nodes:
            graph._nodes[node.id] = node

        # Activate the first node
        if nodes:
            graph._head_id = nodes[0].id
            nodes[0].status = TaskStatus.ACTIVE

        return graph

    def get_current_task(self) -> Optional[TaskNode]:
        if self._head_id is None:
            return None
        return self._nodes.get(self._head_id)

    def advance(self, action_name: str) -> None:
        """Mark the current task as completed, remove it, and advance to next.

        For condition nodes the graph always advances.  The LLM decides
        whether to perform the conditional action (any action != Idle) or
        skip it (Idle).  The graph does not need to branch — the branching
        is implicit in the LLM's action choice.
        """
        node = self._nodes.get(self._head_id)
        if node is None:
            return

        node.status = TaskStatus.COMPLETED
        next_id = node.next_id

        # Remove completed node
        del self._nodes[node.id]

        # Advance head
        self._head_id = next_id
        if next_id is not None and next_id in self._nodes:
            self._nodes[next_id].status = TaskStatus.ACTIVE
        else:
            self._head_id = None

        logger.info("Task completed: '%s' | next: %s", node.description, next_id)

    def is_empty(self) -> bool:
        return len(self._nodes) == 0

    def remaining_count(self) -> int:
        return len(self._nodes)

    def get_tasks_for_prompt(self) -> List[str]:
        """Return the current task + up to 2 upcoming tasks for LLM context."""
        result: List[str] = []
        current = self.get_current_task()
        if current is None:
            return result

        result.append(current.full_description())

        # Walk the chain for upcoming context
        nid = current.next_id
        for _ in range(2):
            if nid is None or nid not in self._nodes:
                break
            upcoming = self._nodes[nid]
            result.append(upcoming.full_description())
            nid = upcoming.next_id

        return result

    def __repr__(self) -> str:
        parts = []
        for nid, node in sorted(self._nodes.items()):
            marker = '>' if node.status == TaskStatus.ACTIVE else ' '
            cond = ' [COND]' if node.is_condition else ''
            parts.append(f"  {marker} {node.id}. {node.description}{cond} ({node.status.value})")
        return "TaskGraph:\n" + "\n".join(parts) if parts else "TaskGraph: (empty)"


# ── Planning class ────────────────────────────────────────────────────────────


class Planning:
    def __init__(self, mode: str = 'simple') -> None:
        self.mode = mode  # 'simple' or 'dag'

        # Task decomposition (used by both modes)
        self.task_decomposition: List[str] = []

        # DAG mode only
        self.task_graph: Optional[TaskGraph] = None

        # Clarification state (AskPlanner discussion loop)
        self._needs_clarification: bool = False
        self._question: str = ''
        self._feedback: str = ''           # accumulated Q&A from prior rounds
        self._clarification_round: int = 0
        self.current_task = ''

    def update_current_task(self, task: str) -> None:
        self.current_task = task

    def set_task_decomposition(self, decomposition: str) -> None:
        self.task_decomposition = decomposition

    def set_manual_task_decomposition(self, decomposition: List[str]) -> None:
        """Override PlanningModule by directly setting the plan text.

        Must be called after set_current_task() (which resets self.task_decomposition).
        """
        self.task_decomposition = decomposition
        if self.mode == 'dag':
            self.task_graph = TaskGraph.from_task_list(decomposition)

    # ── Unified interface (works for both modes) ──────────────────────────

    def has_remaining_tasks(self) -> bool:
        if self.mode == 'dag':
            return self.task_graph is not None and not self.task_graph.is_empty()
        return len(self.task_decomposition) > 0

    def get_tasks_for_reasoning(self, task_num: int) -> List[str]:
        """Return tasks for the reasoning prompt.

        In 'dag' mode: returns the current task + up to 2 upcoming.
        In 'simple' mode: returns the last *task_num* entries (existing behavior).
        """
        if self.mode == 'dag':
            return self.task_graph.get_tasks_for_prompt() if self.task_graph else []
        return self.task_decomposition[-task_num:] if task_num > 0 else []

    def advance_task(self, action_name: str) -> None:
        """Advance the task graph. No-op in simple mode."""
        if self.mode == 'dag' and self.task_graph:
            self.task_graph.advance(action_name)

    # ── Prompt generation ─────────────────────────────────────────────────

    def get_task_decomposition_prompt(
        self, information: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        world_state = information.get('world_state', {})
        memory = information.get('memory', '') or 'none'
        feedback = information.get('feedback', '') or 'none'

        info_dict: Dict[str, Any] = {
            "task": self.current_task,
            "world_state": world_state,
            "memory": memory,
            "feedback": feedback,
        }

        return [
            {"role": "system", "content": TASK_DECOMPOSITION_PROMPT},
            {"role": "user",   "content": to_toon(info_dict)},
        ]
