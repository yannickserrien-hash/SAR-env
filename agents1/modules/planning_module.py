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

_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'to', 'at', 'for', 'and', 'or', 'is', 'in', 'of',
    'it', 'its', 'this', 'that', 'be', 'by', 'on', 'with', 'from',
})


# ── Task status ──────────────────────────────────────────────────────────────


class TaskStatus(Enum):
    PENDING = 'pending'
    ACTIVE = 'active'
    COMPLETED = 'completed'
    SKIPPED = 'skipped'


# ── SubTask (simple mode) ────────────────────────────────────────────────────


@dataclass
class SubTask:
    description: str
    status: TaskStatus = TaskStatus.PENDING

    def __str__(self) -> str:
        return self.description


# ── TaskGraph data structures (DAG mode) ─────────────────────────────────────


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
        if self.is_condition and self.condition_action:
            return f"{self.description}. If so, {self.condition_action}"
        return self.description


_COND_INLINE = re.compile(r'^(.*?)\.\s*[Ii]f\s+(.+?),\s*(.+)$')
_COND_SUBBULLET = re.compile(r'\n\s*-\s*[Ii]f\s+(.+?),\s*(.+)$')


class TaskGraph:
    """Directed acyclic graph of tasks with conditional branching support."""

    def __init__(self) -> None:
        self._nodes: Dict[int, TaskNode] = {}
        self._head_id: Optional[int] = None

    @classmethod
    def from_task_list(cls, tasks: List[str]) -> 'TaskGraph':
        graph = cls()
        if not tasks:
            return graph

        nodes: List[TaskNode] = []
        for idx, raw in enumerate(tasks):
            node_id = idx + 1
            m = _COND_SUBBULLET.search(raw)
            if m:
                desc = raw[:m.start()].strip().rstrip('.')
                nodes.append(TaskNode(
                    id=node_id, description=desc,
                    is_condition=True, condition_action=m.group(2).strip(),
                ))
                continue
            m = _COND_INLINE.match(raw.strip())
            if m:
                nodes.append(TaskNode(
                    id=node_id, description=m.group(1).strip(),
                    is_condition=True, condition_action=m.group(3).strip(),
                ))
                continue
            nodes.append(TaskNode(id=node_id, description=raw.strip()))

        for i in range(len(nodes) - 1):
            nodes[i].next_id = nodes[i + 1].id
        for node in nodes:
            graph._nodes[node.id] = node
        if nodes:
            graph._head_id = nodes[0].id
            nodes[0].status = TaskStatus.ACTIVE

        return graph

    def get_current_task(self) -> Optional[TaskNode]:
        if self._head_id is None:
            return None
        return self._nodes.get(self._head_id)

    def advance(self) -> None:
        """Mark current task completed, remove it, activate next."""
        node = self._nodes.get(self._head_id)
        if node is None:
            return
        node.status = TaskStatus.COMPLETED
        next_id = node.next_id
        del self._nodes[node.id]
        self._head_id = next_id
        if next_id is not None and next_id in self._nodes:
            self._nodes[next_id].status = TaskStatus.ACTIVE
        else:
            self._head_id = None
        logger.info("DAG task completed: '%s' | next: %s", node.description, next_id)

    def is_empty(self) -> bool:
        return len(self._nodes) == 0

    def __repr__(self) -> str:
        parts = []
        for nid, node in sorted(self._nodes.items()):
            marker = '>' if node.status == TaskStatus.ACTIVE else ' '
            cond = ' [COND]' if node.is_condition else ''
            parts.append(f"  {marker} {node.id}. {node.description}{cond} ({node.status.value})")
        return "TaskGraph:\n" + "\n".join(parts) if parts else "TaskGraph: (empty)"


# ── Keyword-overlap matching ─────────────────────────────────────────────────


def _is_task_match(task_completing: str, task_description: str) -> bool:
    """Check if task_completing keywords overlap >=50% with task description."""
    tc_words = set(task_completing.lower().split()) - _STOP_WORDS
    td_words = set(task_description.lower().split()) - _STOP_WORDS
    if not tc_words:
        return False
    return len(tc_words & td_words) / len(tc_words) >= 0.5


# ── Planning class ───────────────────────────────────────────────────────────


class Planning:
    def __init__(self, mode: str = 'simple') -> None:
        self.mode = mode
        self.task_decomposition: List[SubTask] = []
        self.task_graph: Optional[TaskGraph] = None
        self.current_task = ''

        # Clarification state (AskPlanner discussion loop)
        self._needs_clarification: bool = False
        self._question: str = ''
        self._feedback: str = ''
        self._clarification_round: int = 0

    def update_current_task(self, task: str) -> None:
        self.current_task = task

    def set_task_decomposition(self, decomposition: str) -> None:
        self.task_decomposition = decomposition

    def set_manual_task_decomposition(self, decomposition: List[str]) -> None:
        """Set plan from a list of task description strings."""
        self.task_decomposition = [SubTask(desc) for desc in decomposition]
        if self.task_decomposition:
            self.task_decomposition[0].status = TaskStatus.ACTIVE
        if self.mode == 'dag':
            self.task_graph = TaskGraph.from_task_list(decomposition)

    # ── Unified interface ────────────────────────────────────────────────

    def has_remaining_tasks(self) -> bool:
        if self.mode == 'dag':
            return self.task_graph is not None and not self.task_graph.is_empty()
        return any(
            st.status in (TaskStatus.PENDING, TaskStatus.ACTIVE)
            for st in self.task_decomposition
        )

    def get_tasks_for_reasoning(self) -> List[str]:
        """Return ONLY the currently active task."""
        if self.mode == 'dag':
            node = self.task_graph.get_current_task() if self.task_graph else None
            return [node.full_description()] if node else []
        for st in self.task_decomposition:
            if st.status == TaskStatus.ACTIVE:
                return [st.description]
        return []

    def advance_task(self, task_completing: str = '') -> None:
        """Advance the active task if task_completing matches it."""
        if not task_completing:
            return
        if self.mode == 'dag':
            self._advance_dag(task_completing)
        else:
            self._advance_simple(task_completing)

    def _advance_simple(self, task_completing: str) -> None:
        active = next(
            (st for st in self.task_decomposition if st.status == TaskStatus.ACTIVE),
            None,
        )
        if active is None:
            return
        if _is_task_match(task_completing, active.description):
            active.status = TaskStatus.COMPLETED
            for st in self.task_decomposition:
                if st.status == TaskStatus.PENDING:
                    st.status = TaskStatus.ACTIVE
                    break
            logger.info("Task completed: '%s' (matched '%s')", active.description, task_completing)

    def _advance_dag(self, task_completing: str) -> None:
        if not self.task_graph:
            return
        node = self.task_graph.get_current_task()
        if node is None:
            return
        if _is_task_match(task_completing, node.description):
            self.task_graph.advance()

    # ── Prompt generation ────────────────────────────────────────────────

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
