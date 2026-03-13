"""
LLM utility functions for the Engine and RescueAgent.

Supports Ollama (local) via HTTP API at http://localhost:11434.
"""

import json
import os
import re
import random
import logging
import threading
import concurrent.futures
import requests
from typing import Optional, List, Dict
from ollama import Client

OLLAMA_BASE_URL = "http://localhost:11434"

# ── Ollama SDK client cache (thread-safe, one Client per host URL) ────────
_client_cache: Dict[str, Client] = {}
_client_lock = threading.Lock()


def _get_client(api_url: str = None) -> Client:
    """Return a cached Ollama Client for the given host URL."""
    host = api_url or OLLAMA_BASE_URL
    with _client_lock:
        if host not in _client_cache:
            _client_cache[host] = Client(host=host)
        return _client_cache[host]

# ── Few-shot loader ─────────────────────────────────────────────────────────
_FEW_SHOT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'few_shot_examples.yaml'
)
_few_shot_cache: Optional[Dict] = None


def load_few_shot(key: str) -> List[Dict]:
    """Return few-shot messages for *key* from few_shot_examples.yaml.

    Returns a list of {"role": "user"/"assistant", "content": str} dicts
    ready to be injected between the system message and the real user prompt.
    Returns [] if the key is absent, empty, or the file is missing.
    """
    global _few_shot_cache
    if _few_shot_cache is None:
        try:
            import yaml
            with open(_FEW_SHOT_FILE, 'r') as f:
                _few_shot_cache = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logging.getLogger('llm_utils').warning(
                "few_shot_examples.yaml not found at %s", _FEW_SHOT_FILE
            )
            _few_shot_cache = {}
        except Exception as e:
            logging.getLogger('llm_utils').warning(
                "Failed to load few_shot_examples.yaml: %s", e
            )
            _few_shot_cache = {}

    examples = _few_shot_cache.get(key) or []
    messages = []
    for ex in examples:
        if isinstance(ex, dict) and 'user' in ex and 'assistant' in ex:
            messages.append({"role": "user", "content": ex['user'].strip()})
            messages.append({"role": "assistant", "content": ex['assistant'].strip()})
    return messages

logger = logging.getLogger('llm_utils')

# Module-level thread pool shared by all async LLM callers.
# Lazily initialised — call init_llm_pool(num_agents) at startup to size it.
_llm_executor = None


def init_llm_pool(num_agents: int = 1) -> None:
    """Initialise the LLM thread pool based on the number of agents.

    Call once at startup, before any LLM queries are submitted.
    Workers = max(4, num_agents * 3) so each agent can have ~3 concurrent
    LLM calls (reasoning + memory extraction + communication) in flight.
    """
    global _llm_executor
    workers = max(4, num_agents * 3)
    _llm_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix='llm_worker'
    )
    logger.info("LLM thread pool initialised with %d workers (num_agents=%d)", workers, num_agents)


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the LLM executor, creating a default 4-worker pool if needed."""
    global _llm_executor
    if _llm_executor is None:
        init_llm_pool(1)
    return _llm_executor


def query_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 5000,
    temperature: float = 0.4,
    api_url: str = None,
    few_shot_messages: Optional[List[Dict]] = None,
) -> Optional[str]:
    """
    Query an LLM via Ollama's chat API.
    """
    base_url = api_url or OLLAMA_BASE_URL
    # messages = [{"role": "system", "content": system_prompt}]
    # if few_shot_messages:
    #     messages.extend(few_shot_messages)
    # messages.append({"role": "user", "content": user_prompt})
    prompt_parts = []

    if system_prompt:
        prompt_parts.append(f"\n{system_prompt.strip()}")

    if few_shot_messages:
        for msg in few_shot_messages:
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if role == "user":
                prompt_parts.append(f"\n{content}")
            elif role == "assistant":
                prompt_parts.append(f":\n{content}")

    prompt_parts.append(f":\n{user_prompt.strip()}")
    prompt_parts.append(":\n")

    full_prompt = "\n".join(prompt_parts)
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                # "keep_alive": 0,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
            }
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama at %s. Is it running?", base_url)
        return None
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out")
        return None
    except Exception as e:
        logger.error("LLM query failed: %s", e)
        return None


def query_llm_async(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 5000,
    temperature: float = 0.4,
    api_url: str = None,
    few_shot_messages: Optional[List[Dict]] = None,
) -> concurrent.futures.Future:
    """
    Submit an LLM query to the background thread pool and return immediately.

    Returns a concurrent.futures.Future whose result is Optional[str]
    (same type as query_llm). Callers poll future.done() each tick and
    retrieve the result with future.result() once it is ready.

    This function NEVER blocks — the requests.post call runs in a daemon
    thread managed by _llm_executor.
    """
    print("LLM queried")
    return _get_executor().submit(
        query_llm,
        model,
        system_prompt,
        user_prompt,
        max_tokens,
        temperature,
        api_url,
        few_shot_messages,
    )


def query_llm_with_tools(
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: List[Dict],
    max_tokens: int = 1000,
    temperature: float = 0.1,
    api_url: str = None,
    few_shot_messages: Optional[List[Dict]] = None,
) -> Optional[dict]:
    """Query an LLM via the Ollama Python SDK with tool calling.

    Sends structured tool descriptions so the model returns a
    ``tool_calls`` response instead of free-form text.

    Returns:
        A dict ``{"name": str, "arguments": dict}`` from the first tool
        call, or a fallback string if the model returned plain text, or
        *None* on error.
    """
    client = _get_client(api_url)

    # Build messages array with proper role separation
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot_messages:
        messages.extend(few_shot_messages)
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat(
            model=model,
            messages=messages,
            tools=tools,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        )

        # ── Structured tool call ──
        if response.message.tool_calls:
            call = response.message.tool_calls[0]
            name = call.function.name
            arguments = call.function.arguments or {}
            if name:
                return {"name": name, "arguments": arguments}

        # ── Fallback: model returned plain text instead of a tool call ──
        content = response.message.content or ""
        if content:
            logger.info("Tool-call model returned text instead of tool_call, "
                        "falling back to text parsing.")
            return content  # caller can run parse_json_response on this

        return None

    except Exception as e:
        logger.error("LLM tool-call query failed: %s", e)
        return None


def query_llm_with_tools_async(
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: List[Dict],
    max_tokens: int = 1000,
    temperature: float = 0.1,
    api_url: str = None,
    few_shot_messages: Optional[List[Dict]] = None,
) -> concurrent.futures.Future:
    """Non-blocking version of ``query_llm_with_tools``.

    Returns a ``Future`` whose result is the same type as
    ``query_llm_with_tools``: either a tool-call dict, a fallback
    string, or None.
    """
    print("LLM queried (tool-calling)")
    return _get_executor().submit(
        query_llm_with_tools,
        model,
        system_prompt,
        user_prompt,
        tools,
        max_tokens,
        temperature,
        api_url,
        few_shot_messages,
    )


def parse_json_response(text: str) -> Optional[dict]:
    """
    Extract JSON from LLM response text.

    Tries in order:
    1. ```json ... ``` fenced code block
    2. Raw JSON object (strict json.loads)
    3. Python dict literal via ast.literal_eval  ← handles single-quoted responses

    Args:
        text: Raw LLM response text

    Returns:
        Parsed dict, or None if parsing fails
    """
    import ast

    if not text:
        return None

    # 1. Fenced ```json block
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Isolate the outermost { ... } span for attempts 2 & 3
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end + 1]

    # 2. Strict JSON (double-quoted keys/values)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 3. Python dict literal — handles single-quoted strings from the LLM
    try:
        result = ast.literal_eval(candidate)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    logger.warning("Failed to parse JSON/dict from LLM response: %s", text[:200])
    return None
