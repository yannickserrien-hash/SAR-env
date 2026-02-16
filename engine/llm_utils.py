"""
LLM utility functions for the Engine and RescueAgent.

Supports Ollama (local) via HTTP API at http://localhost:11434.
"""

import json
import re
import logging
import requests
from typing import Optional

OLLAMA_BASE_URL = "http://localhost:11434"

logger = logging.getLogger('llm_utils')


def query_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7
) -> Optional[str]:
    """
    Query an LLM via Ollama's chat API.

    Args:
        model: Model name (e.g., 'llama3:8b')
        system_prompt: System message
        user_prompt: User message
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        Response text string, or None if the call fails
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
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
        logger.error("Cannot connect to Ollama at %s. Is it running?", OLLAMA_BASE_URL)
        return None
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out")
        return None
    except Exception as e:
        logger.error("LLM query failed: %s", e)
        return None


def parse_json_response(text: str) -> Optional[dict]:
    """
    Extract JSON from LLM response text.
    Handles responses with ```json code blocks or raw JSON.

    Args:
        text: Raw LLM response text

    Returns:
        Parsed dict, or None if parsing fails
    """
    if not text:
        return None
    try:
        # Try to find JSON in ```json code block
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Try to find raw JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON from LLM response: %s", e)

    return None
