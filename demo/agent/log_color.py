"""ANSI log coloring — copied verbatim from agent/react_agent.py so the LLM-led
experiment's console logs look exactly like the structured agent's.

Color is embedded per-message via _tag()/_line_end()/_tokens()/_clock(), gated by
_detect_color() (AGENT_LOG_COLOR env / isatty / WT_SESSION...). The log formatter
stays plain text; nothing here touches the file handler.
"""

from __future__ import annotations

import os
import sys
from typing import Any


def _detect_color() -> bool:
    override = os.environ.get("AGENT_LOG_COLOR")
    if override is not None:
        return override.strip() not in {"0", "false", "False", ""}
    try:
        if sys.stderr.isatty() or sys.stdout.isatty():
            return True
    except Exception:
        pass
    return any(os.environ.get(name) for name in ("WT_SESSION", "ANSICON", "TERM_PROGRAM"))


_USE_COLOR = _detect_color()

_COLOR_RESET = "\033[0m"
_TOKEN_COLOR = "\033[1;93m"  # bold bright yellow
_CLOCK_COLOR = "\033[1;96m"  # bold bright cyan
_TOTAL_COLOR = "\033[1;95m"  # bold bright magenta — 累计总 token 醒目
_TAG_COLORS = {
    "STEP_BEGIN": "\033[1;36m",   # bold cyan
    "SNAPSHOT": "\033[2;37m",     # dim white/gray
    "REACT": "\033[32m",          # green
    "TOOL": "\033[34m",           # blue
    "OBS": "\033[2;34m",          # dim blue
    "PATHS": "\033[35m",          # magenta
    "PREPLAN": "\033[2;35m",      # dim magenta
    "REACT_FAIL": "\033[1;31m",   # bold red
    "FALLBACK": "\033[33m",       # yellow
    "NOTES": "\033[36m",          # cyan
    "DECISION": "\033[1;32m",     # bold green
    "TOKENS": "\033[2;33m",       # dim yellow (cumulative TOTAL 用 _total 单独高亮)
}


def _tag(name: str) -> str:
    if not _USE_COLOR:
        return f"[{name}]"
    color = _TAG_COLORS.get(name, "")
    if not color:
        return f"[{name}]"
    return f"{color}[{name}]"


def _line_color(name: str) -> str:
    if not _USE_COLOR:
        return ""
    return _TAG_COLORS.get(name, "")


def _line_end() -> str:
    return _COLOR_RESET if _USE_COLOR else ""


def _tokens(tag_name: str, **values: Any) -> str:
    text = " ".join(f"{key}={value}" for key, value in values.items())
    if not _USE_COLOR:
        return text
    return f"{_TOKEN_COLOR}{text}{_line_color(tag_name)}"


def _clock(tag_name: str, value: Any) -> Any:
    if not _USE_COLOR:
        return value
    return f"{_CLOCK_COLOR}{value}{_line_color(tag_name)}"


def _total(tag_name: str, value: Any) -> Any:
    """高亮累计总 token(粗亮品红),用完返回该 tag 的行色。"""
    if not _USE_COLOR:
        return value
    return f"{_TOTAL_COLOR}{value}{_line_color(tag_name)}"
