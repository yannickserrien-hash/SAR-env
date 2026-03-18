The Planning class in agents1/modules/planning_module.py has two modes (simple/dag) but inconsistent task tracking. 

Simple mode uses a raw List[str] with an external task_num counter; 
DAG mode has TaskStatus but blindly advances after every action. 

The task_completing field from tool returns (e.g., "carrying victim") is defined in tool_registry.py but never used. The agent also sees upcoming tasks in prompts when it should only see the active one.

Goal: Unify both modes around TaskStatus, use task_completing for completion detection via keyword overlap, and show only the ACTIVE task in prompts.

Changes
1. agents1/modules/planning_module.py — Core rewrite

Add SubTask dataclass for simple mode:

python  @dataclass
  class SubTask:
      description: str
      status: TaskStatus = TaskStatus.PENDING

Change task_decomposition type from List[str] to List[SubTask]
Rewrite set_manual_task_decomposition(): Create SubTask list, mark first as ACTIVE
Rewrite get_tasks_for_reasoning(): Drop task_num param. Return ONLY the active task for both modes:

Simple: find first SubTask with status == ACTIVE
DAG: task_graph.get_current_task()


Rewrite advance_task(action_name, task_completing): Accept task_completing string. Use _is_task_match() keyword overlap (50% threshold) to decide if the active task should be marked COMPLETED. If match found, mark COMPLETED and activate next PENDING task.
Add _is_task_match(task_completing, task_description): Keyword overlap — tokenize both strings, remove stop words, check if ≥50% of task_completing words appear in task_description.
Rewrite has_remaining_tasks(): Simple mode checks for any ACTIVE/PENDING SubTask.
Keep TaskGraph/TaskNode mostly as-is — the graph's advance() method still handles pointer updates, but is now called conditionally (only when match succeeds).

2. agents1/modules/execution_module.py — Return task_completing

Pop task_completing from args at the top of execute_action() before dispatching
Add _DEFAULT_TASK_COMPLETING dict for tools without the parameter (MoveNorth→"moving north", Drop→"dropping carried victim", etc.)
Change return type to Tuple[str, Dict, str] — 3rd element is task_completing
Update all return statements to include the task_completing string

3. agents1/llm_agent_base.py — Wire task_completing through

_handle_llm_result(): Destructure 3-tuple from execute_action():

python  action_name, kwargs, task_completing = execute_action(...)

Pass task_completing to advance_task():

python  self.planner.advance_task(action_name, task_completing)

Remove self.task_num field from __init__ (line 154)
Remove self.task_num = len(entries) from set_manual_task_decomposition() (line 279)

4. agents1/search_rescue_agent.py — Cleanup

Remove task_num -= 1 (lines 117-118)
Simplify no_tasks check (lines 91-95) to just not self.planner.has_remaining_tasks()
Update get_tasks_for_reasoning() call — drop task_num argument

5. agents1/action_mapper.py — Update parse() return type

parse() calls execute_action() which now returns 3-tuple → update return type to Tuple[str, Dict, str]
