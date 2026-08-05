"""Harness v2（复核 top route 第一跳 · 新架构 C4）。

旧 harness 审"模型为何接这单"；新 harness 在 plan_route 选出 top route(r0) 后、执行第一跳前，复核三件事：
  1. **虚拟单是否正确**：该有的 active 虚拟单在不在、窗口/价是否合理（manager 漏注/错注）。
  2. **top route 第一跳是否违反硬约束**：休息窗内作业 / 禁区禁品类禁端点 / blocking 事件赶不上。
  3. **pricing 是否离谱**：adjusted_yield 与偏好罚款账明显矛盾（如违规单却高分）。

任一违反 → 返回 violations，loop 把原因回灌给 Virtual Manager 重维护一轮（有界 HARNESS_MAX_REPLANS）。
契约：review(api, context, logger) -> {"ok": bool, "violations": [{"category","reason"}], "note": str}
容错：模型/解析失败 → ok=True, violations=[]（fails-open，绝不阻断；确定性墙仍兜底）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import model_io
from .log_color import _line_end, _tag

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "harness_review_prompt.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip() if PROMPT_PATH.exists() else ""

_log = logging.getLogger("llm_agent.harness")


def review(api: Any, context: dict[str, Any], logger: logging.Logger | None = None) -> dict[str, Any]:
    """复核 top route 第一跳。返回 {ok, violations, note}。出错 → ok=True（fails-open）。"""
    log = logger or _log
    if api is None or not SYSTEM_PROMPT:
        return {"ok": True, "violations": [], "note": ""}
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
        "temperature": 0.0,
    }
    try:
        content = model_io.call_content_with_retry(api, payload, log)
        obj = json.loads(content)
        if not isinstance(obj, dict):
            raise ValueError("harness response is not a JSON object")
    except Exception as exc:  # noqa: BLE001 — fails-open
        log.warning("%s harness_v2 failed: %s%s", _tag("HARNESS_FAIL"), exc, _line_end())
        return {"ok": True, "violations": [], "note": ""}
    violations = obj.get("violations") if isinstance(obj.get("violations"), list) else []
    violations = [v for v in violations if isinstance(v, dict) and v.get("reason")]
    ok = bool(obj.get("ok", not violations)) and not violations
    log.info("%s ok=%s violations=%s%s", _tag("HARNESS"), ok,
             json.dumps(violations, ensure_ascii=False), _line_end())
    return {"ok": ok, "violations": violations, "note": str(obj.get("note") or "")[:200]}
