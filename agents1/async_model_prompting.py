"""
Unified LLM interface — single execution path for all LLM calls in the codebase.

Calls litellm.completion() directly with per-agent Ollama port routing via
the `api_base` parameter. No external framework dependencies (MARBLE, etc.).

Uses a ThreadPoolExecutor so that LLM calls from multiple agents don't block
each other. Each tick an agent submits a call and polls for the result.

Usage (async, for agents):
    from agents1.async_model_prompting import submit_llm_call, get_llm_result

    future = submit_llm_call(
        "ollama/llama3", messages,
        max_token_num=512,
        tools=tool_schemas,
        tool_choice='auto',
        api_base="http://localhost:11435",
    )
    result = get_llm_result(future)  # None if still running, List[Message] if done

Usage (sync, for EnginePlanner / memory):
    from agents1.async_model_prompting import call_llm_sync

    text = call_llm_sync(
        llm_model="ollama/qwen3:8b",
        system_prompt="You are helpful.",
        user_prompt="What should the agents do?",
        api_base="http://localhost:11434",
    )
"""

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import wraps
from typing import Any, Dict, List, Optional

import litellm

logger = logging.getLogger('async_model_prompting')

# ---------------------------------------------------------------------------
# Retry decorator (replaces MARBLE's error_handler — zero external deps)
# ---------------------------------------------------------------------------

def _retry_with_backoff(retries: int = 5, base_wait_time: float = 1.0):
    """Simple retry decorator with exponential backoff."""
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
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, retries, e, wait,
                    )
                    time.sleep(wait)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Core LLM completion (with retry)
# ---------------------------------------------------------------------------

@_retry_with_backoff(retries=5, base_wait_time=1)
def _llm_completion(
    llm_model: str,
    messages: list,
    max_token_num: int = 512,
    temperature: float = 0.0,
    tools: Optional[list] = None,
    tool_choice: Optional[str] = None,
    api_base: Optional[str] = None,
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


# ---------------------------------------------------------------------------
# Shared ThreadPoolExecutor (lazy init)
# ---------------------------------------------------------------------------

_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix='llm_pool'
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
        max_workers=workers, thread_name_prefix='llm_pool'
    )


# ---------------------------------------------------------------------------
# Async API (for agents — submit & poll)
# ---------------------------------------------------------------------------

def submit_llm_call(
    llm_model: str,
    messages: List[Dict[str, str]],
    max_token_num: int = 512,
    temperature: float = 0.0,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs: Any,
) -> Future:
    """Submit an LLM call non-blocking; returns a Future immediately.

    The Future's result will be ``List[Message]`` on success.

    Args:
        llm_model:     LiteLLM model string, e.g. ``"ollama/llama3"``.
        messages:      OpenAI-style message list (role/content dicts).
        max_token_num: Max tokens for the completion.
        temperature:   Sampling temperature.
        tools:         OpenAI-compatible tool schemas (optional).
        tool_choice:   ``'auto'``, ``'none'``, or a specific tool name.
        api_base:      Per-agent Ollama base URL, e.g. ``"http://localhost:11435"``.
        **kwargs:      Reserved for future use.
    """
    return _get_executor().submit(
        _llm_completion,
        llm_model,
        messages,
        max_token_num=max_token_num,
        temperature=temperature,
        tools=tools if tools else None,
        tool_choice=tool_choice if tools else None,
        api_base=api_base,
    )


def get_llm_result(future: Future):
    """Poll a Future without blocking.

    Returns:
        ``List[Message]`` if the call is done, ``None`` if still in flight.

    Raises:
        Exception: Propagates any exception the LLM call raised.
    """
    if future.done():
        return future.result()
    return None


# ---------------------------------------------------------------------------
# Sync API (for EnginePlanner, ShortTermMemory — runs in caller's thread)
# ---------------------------------------------------------------------------

def call_llm_sync(
    llm_model: str,
    system_prompt: str,
    user_prompt: str,
    max_token_num: int = 5000,
    temperature: float = 0.4,
    few_shot_messages: Optional[List[Dict]] = None,
    tools: Optional[list] = None,
    tool_choice: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Optional[str]:
    """Synchronous LLM call that builds a message list and returns text content.

    Convenience wrapper for callers that manage their own threads
    (EnginePlanner, ShortTermMemory). Mirrors the old ``query_llm()`` signature.

    Returns:
        The assistant's text content, or ``None`` on failure.
    """
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot_messages:
        messages.extend(few_shot_messages)
    messages.append({"role": "user", "content": user_prompt})

    try:
        result = _llm_completion(
            llm_model=llm_model,
            messages=messages,
            max_token_num=max_token_num,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            api_base=api_base,
        )
        if result and len(result) > 0:
            return getattr(result[0], 'content', None)
        return None
    except Exception as e:
        logger.error("LLM call failed after all retries: %s", e)
        return None
