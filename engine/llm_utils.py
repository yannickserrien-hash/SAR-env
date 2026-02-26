"""
LLM utility functions for the Engine and RescueAgent.

Supports Ollama (local) via HTTP API at http://localhost:11434.
"""

import json
import re
import logging
import threading
import concurrent.futures
import requests
from typing import Optional

OLLAMA_BASE_URL = "http://localhost:11434"

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
    max_tokens: int = 1000,
    temperature: float = 0.1,
    api_url: str = None
) -> Optional[str]:
    """
    Query an LLM via Ollama's chat API.

    Args:
        model: Model name (e.g., 'llama3:8b')
        system_prompt: System message
        user_prompt: User message
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        api_url: Ollama base URL (e.g. 'http://localhost:11435'). Defaults to OLLAMA_BASE_URL.

    Returns:
        Response text string, or None if the call fails
    """
    base_url = api_url or OLLAMA_BASE_URL
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
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
    temperature: float = 0.1,
    api_url: str = None
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
    print(user_prompt)
    return _get_executor().submit(
        query_llm,
        model,
        system_prompt,
        user_prompt,
        max_tokens,
        temperature,
        api_url
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

    print("[JSON:] " + text)

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
