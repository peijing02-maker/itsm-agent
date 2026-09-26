"""Helpers for results of MCP tools."""

import json
from typing import Any


def mcp_json(result: Any) -> Any:
    """MCP tools return content blocks (or their text); our server puts JSON in the text."""
    if isinstance(result, list):
        result = "".join(b.get("text", "") for b in result if isinstance(b, dict))
    return json.loads(result)
