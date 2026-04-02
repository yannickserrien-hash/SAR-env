"""
Unified LLM interface — single execution path for all LLM calls in the codebase.

Supports two backends for local Ollama inference:
  - "ollama_sdk": Uses the official Ollama Python SDK (ollama.Client.chat)
  - "requests":   Direct HTTP calls to Ollama's OpenAI-compatible endpoint

Uses a ThreadPoolExecutor so that LLM calls from multiple agents don't block
each other. Each tick an agent submits a call and polls for the result.

Usage (async, for agents):
    from agents1.async_model_prompting import submit_llm_call, get_llm_result

    future = submit_llm_call(
        "ollama/llama3", messages,
        max_token_num=512,
        tools=tool_schemas,
        tool_choice='auto',
        api_base="http://localhost:11434",
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

Backend selection:
    from agents1.async_model_prompting import set_backend
    set_backend("ollama_sdk")   # default
    set_backend("requests")     # direct HTTP

    # Or via environment variable:  LLM_BACKEND=requests
    # Or at startup:  init_marble_pool(num_agents, backend="requests")
"""

import json
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, List, Optional

import requests as _requests

try:
    import ollama as _ollama_sdk
except ImportError:
    _ollama_sdk = None

logger = logging.getLogger('async_model_prompting')

# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

_backend: str = os.environ.get("LLM_BACKEND", "ollama_sdk")


def set_backend(backend: str) -> None:
    """Set the LLM backend. Options: 'ollama_sdk', 'requests'."""
    global _backend
    if backend not in ("ollama_sdk", "requests"):
        raise ValueError(
            f"Unknown backend: {backend!r}. Use 'ollama_sdk' or 'requests'."
        )
    _backend = backend
    logger.info("LLM backend set to: %s", backend)


# ---------------------------------------------------------------------------
# Response dataclasses (compatible with OpenAI message attribute access)
# ---------------------------------------------------------------------------

@dataclass
class _Function:
    name: str
    arguments: str  # always a JSON string

@dataclass
class _ToolCall:
    function: _Function
    id: str = ""
    type: str = "function"

@dataclass
class _Message:
    content: Optional[str] = None
    tool_calls: Optional[List[_ToolCall]] = None
    role: str = "assistant"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_model_prefix(model: str) -> str:
    """Strip the 'ollama/' prefix from model names."""
    if model.startswith("ollama/"):
        return model[len("ollama/"):]
    return model


# ---------------------------------------------------------------------------
# Retry decorator
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
# Backend: Ollama Python SDK
# ---------------------------------------------------------------------------

def _completion_ollama_sdk(
    model: str,
    messages: list,
    max_token_num: int,
    temperature: float,
    tools: Optional[list],
    tool_choice: Optional[str],
    api_base: Optional[str],
) -> List[_Message]:
    if _ollama_sdk is None:
        raise ImportError(
            "ollama package required for 'ollama_sdk' backend. "
            "Install with: pip install ollama>=0.4.0"
        )
    client = _ollama_sdk.Client(host=api_base) if api_base else _ollama_sdk.Client()

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "options": {"num_predict": max_token_num, "temperature": temperature},
    }
    if tools:
        kwargs["tools"] = tools

    response = client.chat(**kwargs)

    # Convert Ollama SDK response to our _Message format
    msg_data = response.get("message", response) if isinstance(response, dict) else getattr(response, 'message', response)
    if isinstance(msg_data, dict):
        content = msg_data.get("content") or None
        raw_tool_calls = msg_data.get("tool_calls")
    else:
        content = getattr(msg_data, 'content', None) or None
        raw_tool_calls = getattr(msg_data, 'tool_calls', None)

    tool_calls = None
    if raw_tool_calls:
        tool_calls = []
        for i, tc in enumerate(raw_tool_calls):
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", {})
            else:
                fn = getattr(tc, 'function', tc)
                fn_name = getattr(fn, 'name', '') if not isinstance(fn, dict) else fn.get('name', '')
                fn_args = getattr(fn, 'arguments', {}) if not isinstance(fn, dict) else fn.get('arguments', {})
            args_str = json.dumps(fn_args) if isinstance(fn_args, dict) else str(fn_args)
            tool_calls.append(_ToolCall(
                function=_Function(name=fn_name, arguments=args_str),
                id=f"call_{i}",
            ))

    return [_Message(content=content, tool_calls=tool_calls or None)]


# ---------------------------------------------------------------------------
# Backend: Direct HTTP (requests) via Ollama's OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

def _completion_requests(
    model: str,
    messages: list,
    max_token_num: int,
    temperature: float,
    tools: Optional[list],
    tool_choice: Optional[str],
    api_base: Optional[str],
) -> List[_Message]:
    base = (api_base or "http://localhost:11434").rstrip("/")
    url = f"{base}/v1/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_token_num,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice and tools:
        payload["tool_choice"] = tool_choice

    resp = _requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Parse OpenAI-format response
    choice = data["choices"][0]["message"]
    content = choice.get("content") or None
    raw_tool_calls = choice.get("tool_calls")

    tool_calls = None
    if raw_tool_calls:
        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            tool_calls.append(_ToolCall(
                function=_Function(
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", "{}"),
                ),
                id=tc.get("id", ""),
            ))

    return [_Message(content=content, tool_calls=tool_calls or None)]


# ---------------------------------------------------------------------------
# Core LLM completion dispatcher (with retry)
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
    """Call the configured LLM backend with retry logic. Returns List[_Message]."""
    model = _strip_model_prefix(llm_model)

    if _backend == "ollama_sdk":
        return _completion_ollama_sdk(
            model, messages, max_token_num, temperature, tools, tool_choice, api_base
        )
    elif _backend == "requests":
        return _completion_requests(
            model, messages, max_token_num, temperature, tools, tool_choice, api_base
        )
    else:
        raise ValueError(f"Unknown LLM backend: {_backend!r}")


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


def init_marble_pool(num_agents: int = 1, backend: Optional[str] = None) -> None:
    """Resize the executor pool based on the number of agents.

    Call once at startup (optional). Workers = max(8, num_agents * 3).
    Replaces any existing pool.

    Args:
        num_agents: Number of concurrent agents.
        backend:    Optionally set the LLM backend ('ollama_sdk' or 'requests').
    """
    global _executor
    if backend is not None:
        set_backend(backend)
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

    The Future's result will be ``List[_Message]`` on success.

    Args:
        llm_model:     Model string, e.g. ``"ollama/llama3"`` or ``"qwen3:8b"``.
        messages:      OpenAI-style message list (role/content dicts).
        max_token_num: Max tokens for the completion.
        temperature:   Sampling temperature.
        tools:         OpenAI-compatible tool schemas (optional).
        tool_choice:   ``'auto'``, ``'none'``, or a specific tool name.
        api_base:      Ollama base URL, e.g. ``"http://localhost:11434"``.
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
        ``List[_Message]`` if the call is done, ``None`` if still in flight.

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
