"""
Async wrapper around MARBLE's model_prompting() for non-blocking LLM calls
in the MATRX tick-based game loop.

Uses a ThreadPoolExecutor so that LLM calls from multiple agents don't block
each other. Each tick an agent submits a call and polls for the result.

Usage:
    from agents1.async_model_prompting import submit_llm_call, get_llm_result

    future = submit_llm_call(
        "ollama/llama3", messages,
        max_token_num=512,
        tools=tool_schemas,          # OpenAI-style tool dicts (optional)
        tool_choice='auto',          # or 'none' / a specific tool name
    )
    ...
    result = get_llm_result(future)  # None if still running, List[Message] if done
"""

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List, Dict, Any

from marble.llms.model_prompting import model_prompting

# Shared executor for all SearchRescueAgent instances (lazy init).
_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix='marble_llm'
        )
    return _executor


def init_marble_pool(num_agents: int = 1) -> None:
    """Resize the executor pool based on the number of agents.

    Call once at startup (optional). Workers = max(8, num_agents * 3).
    Replaces any existing pool.
    """
    global _executor
    workers = max(8, num_agents * 3)
    _executor = ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix='marble_llm'
    )


def submit_llm_call(
    llm_model: str,
    messages: List[Dict[str, str]],
    max_token_num: int = 512,
    temperature: float = 0.0,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> Future:
    """Submit an LLM call non-blocking; returns a Future immediately.

    The Future's result will be ``List[Message]`` on success, or ``None``
    on failure (after all retries from MARBLE's exponential-backoff handler).

    Args:
        llm_model:     LiteLLM model string, e.g. ``"ollama/llama3"``.
        messages:      OpenAI-style message list (role/content dicts).
        max_token_num: Max tokens for the completion.
        temperature:   Sampling temperature.
        tools:         OpenAI-compatible tool schemas for structured tool calling.
                       Pass ``None`` to use plain text generation.
        tool_choice:   ``'auto'``, ``'none'``, or a specific tool name.
                       Ignored when *tools* is None.
        **kwargs:      Additional keyword args forwarded to model_prompting().
    """
    return _get_executor().submit(
        model_prompting,
        llm_model,
        messages,
        max_token_num=max_token_num,
        temperature=temperature,
        tools=tools if tools else None,
        tool_choice=tool_choice if tools else None,
        **kwargs,
    )


def get_llm_result(future: Future):
    """Poll a Future without blocking.

    Returns:
        ``List[Message]`` if the call is done, ``None`` if still in flight.

    Raises:
        Exception: Propagates any exception the LLM call raised (caller
                   should catch and handle accordingly).
    """
    if future.done():
        return future.result()   # may be None if all retries exhausted
    return None
