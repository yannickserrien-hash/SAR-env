"""
ActionMapper: parses LLM JSON output and maps it to MATRX
(action_class_name, kwargs) tuples.

The LLM is instructed to return:
    {"action": "<name>", "args": {"object_id": "...", ...}}

Dispatch logic is delegated to action_dispatch.dispatch_action() — this
module's unique value is the robust text-JSON extraction (fenced blocks,
raw JSON, Python dict literals).

Usage:
    mapper = ActionMapper(partner_name='humanagent')
    action_name, kwargs = mapper.parse(llm_response_text)
"""

import json
import re
import logging
from typing import Optional, Tuple, Dict, Any

from actions1.CustomActions import Idle
from agents1.modules.execution_module import execute_action

logger = logging.getLogger('ActionMapper')


class ActionMapper:
    """Maps LLM JSON responses to MATRX (action_name, kwargs) pairs.

    Falls back to Idle on any parse error or unknown action.
    """

    def __init__(self, partner_name: str = 'humanagent') -> None:
        self._partner_name = partner_name

    def parse(self, llm_text: str) -> Tuple[str, Dict[str, Any]]:
        """Parse LLM response text into a MATRX (action_name, kwargs) pair.

        Args:
            llm_text: Raw text returned by the LLM.

        Returns:
            Tuple of (action_class_name, kwargs_dict).
        """
        parsed = self._extract_json(llm_text)
        if parsed is None:
            logger.warning(
                "ActionMapper: could not parse JSON from response: %.120s", llm_text
            )
            return Idle.__name__, {'duration_in_ticks': 1}

        action = parsed.get('action', 'Idle')
        # Support both "args" and "params" keys from different LLM formats
        args: Dict[str, Any] = parsed.get('args') or parsed.get('params') or {}

        return execute_action(action, args, self._partner_name, None)

    def parse_raw(self, llm_text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Extract action name and args from LLM text *without* dispatching.

        Use this when you need to validate the action before executing it.

        Returns:
            Tuple of ``(action_name, args_dict)``, or ``(None, {})`` on parse failure.
        """
        parsed = self._extract_json(llm_text)
        if parsed is None:
            logger.warning(
                "ActionMapper: could not parse JSON from response: %.120s", llm_text
            )
            return None, {}
        action = parsed.get('action', 'Idle')
        args: Dict[str, Any] = parsed.get('args') or parsed.get('params') or {}
        return action, args

    # ── JSON extraction ────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON dict from free-form LLM text.

        Tries in order:
        1. ```json ... ``` fenced block
        2. First outermost { ... } span (strict JSON)
        3. Python dict literal via ast.literal_eval
        """
        import ast

        if not text:
            return None

        # 1. Fenced ```json block
        m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 2. First { ... } span
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end <= start:
            return None
        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 3. Python dict literal (handles single-quoted LLM output)
        try:
            result = ast.literal_eval(candidate)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass

        return None
