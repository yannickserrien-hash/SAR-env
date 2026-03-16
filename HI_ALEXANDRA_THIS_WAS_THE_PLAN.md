I LOVE YOUUUU!

User approved Claude's plan
  ⎿  Plan saved to: ~/.claude/plans/purring-snuggling-waffle.md · /plan to edit
     Plan: Consolidate LLM Calls to MARBLE Executor

     Context

     The codebase has two LLM execution paths. Per the project spec (Section 1.2), we must consolidate to one. Additionally, each agent must be
     able to route to its own Ollama instance (different ports) for truly parallel non-blocking inference.

     Current state:
     - MARBLE path (agents1/async_model_prompting.py): Used by agents. Calls model_prompting() which wraps litellm.completion(). No per-agent
     port routing — all agents hit default Ollama.
     - Ollama direct path (engine/llm_utils.py): Used by EnginePlanner, ShortTermMemory. Has per-port routing via api_url param.

     Target state:
     - Single LLM module in agents1/async_model_prompting.py calling litellm.completion() directly with base_url support for per-agent Ollama
     routing.
     - engine/llm_utils.py deleted. Utilities (parse_json_response, load_few_shot) relocated.

     Key Insight: base_url Routing & No MARBLE Imports

     MARBLE's model_prompting() hardcodes base_url=None for non-together_ai models, so we can't pass per-agent Ollama URLs through it. Important
      constraint: No files from the MARBLE package (marble/) should be imported. All needed logic (retry decorator, litellm call) must be
     self-contained in our codebase.

     Solution: call litellm.completion() directly in our wrapper with an inlined retry decorator (simple exponential backoff, no MARBLE
     dependency).

     Implementation Steps

     Step 1: Rewrite agents1/async_model_prompting.py

     Remove from marble.llms.model_prompting import model_prompting. Call litellm.completion directly. Add inlined retry decorator (no MARBLE
     import). Add api_base parameter for per-agent Ollama routing. Add call_llm_sync() convenience wrapper for EnginePlanner.

     import time
     import logging
     import litellm
     from functools import wraps

     def _retry_with_backoff(retries=5, base_wait_time=1):
         """Simple retry decorator with exponential backoff (replaces MARBLE's error_handler)."""
         def decorator(func):
             @wraps(func)
             def wrapper(*args, **kwargs):
                 for attempt in range(retries):
                     try:
                         return func(*args, **kwargs)
                     except Exception as e:
                         if attempt == retries - 1:
                             raise
                         wait = base_wait_time * (2 ** attempt)
                         logging.getLogger('llm').warning(
                             "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                             attempt + 1, retries, e, wait
                         )
                         time.sleep(wait)
             return wrapper
         return decorator

     @_retry_with_backoff(retries=5, base_wait_time=1)
     def _llm_completion(
         llm_model: str,
         messages: list,
         max_token_num: int = 512,
         temperature: float = 0.0,
         tools=None,
         tool_choice=None,
         api_base: str = None,
     ) -> list:
         """Call litellm.completion with retry logic. Returns List[Message]."""
         completion = litellm.completion(
             model=llm_model,
             messages=messages,
             max_tokens=max_token_num,
             temperature=temperature,
             tools=tools,
             tool_choice=tool_choice,
             base_url=api_base,
         )
         msg = completion.choices[0].message
         return [msg]


     def submit_llm_call(..., api_base=None) -> Future:
         """Existing API + new api_base param. Calls _llm_completion instead of model_prompting."""

     def call_llm_sync(
         llm_model, system_prompt, user_prompt,
         max_token_num=512, temperature=0.0,
         few_shot_messages=None, tools=None, tool_choice=None,
         api_base=None,
     ) -> Optional[str]:
         """Sync convenience wrapper. Builds messages list, calls _llm_completion,
         extracts .content. For callers that manage their own threads (EnginePlanner)."""

     Step 2: Create engine/parsing_utils.py

     Move from engine/llm_utils.py:
     - parse_json_response() (lines 291-342) — 3-stage JSON parser
     - load_few_shot() (lines 40-70) — YAML few-shot loader + cache

     Step 3: Migrate engine/engine_planner.py

     Imports:
     # Remove: from engine.llm_utils import query_llm, parse_json_response, load_few_shot
     # Add:
     from agents1.async_model_prompting import call_llm_sync
     from engine.parsing_utils import parse_json_response, load_few_shot

     __init__:
     - Normalize model string: self.llm_model = f"ollama/{llm_model}" if not llm_model.startswith("ollama/") else llm_model
     - Convert self._api_url to self._api_base (same value, different param name): self._api_base = api_url

     3 call sites — all follow same pattern:
     # Before:
     response = query_llm(model=self.llm_model, system_prompt=..., user_prompt=..., api_url=self._api_url, ...)
     # After:
     response = call_llm_sync(llm_model=self.llm_model, system_prompt=..., user_prompt=..., api_base=self._api_base, ...)

     Keep: EnginePlanner's own self._executor (4 workers). It still manages its own concurrency — call_llm_sync runs inside those worker
     threads.

     Step 4: Migrate memory/short_term_memory.py

     # Remove: from engine.llm_utils import query_llm, parse_json_response
     # Add:
     from agents1.async_model_prompting import call_llm_sync
     from engine.parsing_utils import parse_json_response

     Replace query_llm(...) call with call_llm_sync(...).

     Step 5: Migrate main.py

     # Remove: from engine.llm_utils import init_llm_pool / init_llm_pool(...)
     # Add:
     from agents1.async_model_prompting import init_marble_pool
     init_marble_pool(num_rescue_agents)

     Optionally normalize planner_model to include ollama/ prefix, or let EnginePlanner handle it.

     Step 6: Add per-agent Ollama port routing in worlds1/WorldBuilder.py

     Currently all MARBLE agents use llm_model='ollama/qwen3:8b' with no port differentiation.

     # Before (line 107-118):
     brain = SearchRescueAgent(..., llm_model='ollama/qwen3:8b', ...)

     # After: pass api_base per agent
     api_base = f"http://localhost:{ollama_base_port + agent_nr}"
     brain = SearchRescueAgent(..., llm_model='ollama/qwen3:8b', api_base=api_base, ...)

     This requires adding api_base parameter to SearchRescueAgent.__init__ → LLMAgentBase.__init__ → stored as self._api_base, and passed
     through in _submit_llm().

     Files to change:
     - agents1/llm_agent_base.py: Add api_base param to __init__, pass to submit_llm_call() in _submit_llm()
     - agents1/search_rescue_agent.py: Add api_base param, pass to super
     - worlds1/WorldBuilder.py: Pass api_base per agent

     Step 7: Update graveyard imports

     - agents1/agents_graveyard/RescueAgent.py: parse_json_response → from engine.parsing_utils
     - agents1/agents_graveyard/PlanningModule.py: query_llm_async → from agents1.async_model_prompting
     - agents1/agents_graveyard/ReasoningModule.py: Update all imports

     Step 8: Delete engine/llm_utils.py

     After all callers migrated, delete the file entirely.

     Files to Modify (in order)

     ┌─────┬─────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────┐
     │  #  │                    File                     │                                   Action                                   │
     ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
     │ 1   │ agents1/async_model_prompting.py            │ Rewrite: use litellm.completion directly, add api_base, add call_llm_sync  │
     ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
     │ 2   │ engine/parsing_utils.py                     │ Create: relocate parse_json_response + load_few_shot                       │
     ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
     │ 3   │ engine/engine_planner.py                    │ Migrate 3 query_llm calls → call_llm_sync, update imports, normalize model │
     ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
     │ 4   │ memory/short_term_memory.py                 │ Migrate 1 query_llm call → call_llm_sync, update imports                   │
     ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
     │ 5   │ main.py                                     │ init_llm_pool → init_marble_pool, remove llm_utils import                  │
     ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
     │ 6   │ agents1/llm_agent_base.py                   │ Add api_base param to __init__ and _submit_llm                             │
     ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
     │ 7   │ agents1/search_rescue_agent.py              │ Add api_base param, pass to super                                          │
     ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
     │ 8   │ worlds1/WorldBuilder.py                     │ Pass per-agent api_base URL based on ollama_base_port + agent_nr           │
     ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
     │ 9   │ agents1/agents_graveyard/RescueAgent.py     │ Update import                                                              │
     ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
     │ 10  │ agents1/agents_graveyard/PlanningModule.py  │ Update import                                                              │
     ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
     │ 11  │ agents1/agents_graveyard/ReasoningModule.py │ Update imports                                                             │
     ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
     │ 12  │ engine/llm_utils.py                         │ Delete                                                                     │
     └─────┴─────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────┘

     Verification

     1. grep -r "from engine.llm_utils" . --include="*.py" → zero results
     2. grep -r "from engine import llm_utils" . --include="*.py" → zero results
     3. python -c "from engine.engine_planner import EnginePlanner; from memory.short_term_memory import ShortTermMemory; from
     agents1.async_model_prompting import call_llm_sync, submit_llm_call; print('OK')" → no import errors
     4. python main.py with Ollama running → EnginePlanner generates tasks, agents receive and execute them on separate Ollama instances