"""Provider-neutral helpers for wiring the canonical tools into a tool-calling loop.

The canonical tool definitions live in `tools.py` as plain JSON Schema. Every
provider's tool/function-calling format is built on that same schema; this module
holds the small, mechanical translations from the canonical shape into each
provider's envelope — the "adapt thrice" seam. Today only the Anthropic mapping
exists; `to_openai_tools` / `to_gemini_tools` belong here when those providers
grow tool support.

Keeping these as pure functions (no SDK imports, no provider imports) means this
module has no heavy dependencies and never causes an import cycle.
"""

from __future__ import annotations

from .tools import TOOL_SCHEMAS

#: Upper bound on tool-call round trips in a single analysis, so a model that
#: keeps calling tools can't loop forever.
MAX_TOOL_ITERATIONS = 6


def to_anthropic_tools(schemas: list[dict] | None = None) -> list[dict]:
    """Render the canonical tool schemas into Anthropic's tool format.

    Claude expects ``{"name", "description", "input_schema"}`` — the canonical
    ``parameters`` JSON Schema maps directly onto ``input_schema``.
    """
    schemas = schemas if schemas is not None else TOOL_SCHEMAS
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }
        for tool in schemas
    ]


def to_openai_tools(schemas: list[dict] | None = None) -> list[dict]:
    """Render the canonical tool schemas into OpenAI's function-calling format.

    OpenAI-compatible endpoints (incl. NVIDIA NIM) expect
    ``{"type": "function", "function": {"name", "description", "parameters"}}`` —
    the canonical ``parameters`` JSON Schema maps directly onto ``function.parameters``.
    """
    schemas = schemas if schemas is not None else TOOL_SCHEMAS
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in schemas
    ]
