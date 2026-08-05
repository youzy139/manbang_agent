"""The preference-LLM: the only writer of the preference_note.

Two modes share one prompt (prompts/preference_llm_prompt.txt):
  INIT  — once per driver: standardize + classify preferences, seed obligations.
  STEP  — on each completed action / day rollover: refresh today's directives,
          update obligation counters, mark event substeps, brief the decision-LLM.

Both calls use the same OpenAI-style ``model_chat_completion`` interface as the
existing agent, asking for a strict JSON object back.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import model_io
from .log_color import _line_end, _tag
from .preference_parser import preference_llm_enabled

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "preference_llm_prompt.txt"
# 偏好 LLM 默认关闭、prompt 已删 → import 容忍缺失（置 PREFERENCE_LLM_ENABLED=1 重启需放回该文件）。
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip() if PROMPT_PATH.exists() else ""

# 「改写器」专用 prompt（PREFERENCE_REWRITE_ONLY 模式用）：只理清逻辑 + 规范单位,不改含义。
_REWRITE_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "preference_rewrite_prompt.txt"
_REWRITE_PROMPT = _REWRITE_PROMPT_PATH.read_text(encoding="utf-8").strip() if _REWRITE_PROMPT_PATH.exists() else ""

_log = logging.getLogger("llm_agent.preference_llm")


def rewrite_clarify(api: Any, raw_text: str, log: logging.Logger | None = None) -> str | None:
    """「改写器」(rewrite-only):把一条偏好原文 → **无歧义、单位规范**的清楚陈述,供决策 LLM 直接读
    (原文仍由 ledger 的 raw_text 保留作兜底,pref_id 仍按原文 md5、不变)。**只清理表达、规范单位,不改
    含义、不增删约束**。每条偏好**一次性**调用(parse 层按内容缓存),非每步。失败/空/无 prompt → 返回
    None(下游回退到原文)。"""
    lg = log or _log
    raw = str(raw_text or "").strip()
    if not raw or not _REWRITE_PROMPT or api is None:  # 无 api(如离线单测)→ 直接回退,不空跑重试退避
        return None
    payload = {
        "messages": [
            {"role": "system", "content": _REWRITE_PROMPT},
            {"role": "user", "content": json.dumps({"raw": raw}, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        # 开思考:改写是**一次性**(parse 层按内容缓存)、需要仔细推敲偏好逻辑(如"晚N小时再休息"= 推迟开始
        # 还是整体平移)的任务,值得多想;思考内容走 reasoning_content,load_content 仍只取最终 content(JSON)。
        "enable_thinking": True,
    }
    try:
        content = model_io.call_content_with_retry(api, payload, lg)
        obj = json.loads(content)
        clarified = str((obj or {}).get("clarified", "") or "").strip() if isinstance(obj, dict) else ""
        if not clarified:
            return None
        lg.info("%s preference_rewrite raw=%r -> %r%s", _tag("NOTES"), raw[:80], clarified[:120], _line_end())
        return clarified
    except Exception as exc:  # noqa: BLE001 — 改写失败绝不打断,回退原文
        lg.warning("%s preference_rewrite failed: %s%s", _tag("REACT_FAIL"), exc, _line_end())
        return None


def _call_json(api: Any, user_payload: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
    }
    # Same retry-then-degrade policy as the decision-LLM (model_io): a single
    # preference-LLM timeout used to fall straight through to an empty delta,
    # silently dropping a whole step's directives + counter updates.
    content = model_io.call_content_with_retry(api, payload, _log)
    _log.info("%s preference_llm output=%s%s", _tag("NOTES"), content, _line_end())
    obj = json.loads(content)
    if not isinstance(obj, dict):
        raise ValueError("model response is not a JSON object")
    return obj


def standardize_init(api: Any, preferences: list[dict[str, Any]]) -> dict[str, Any]:
    """Mode A. Returns {prefs_std, obligations_seed, freeform_seed}."""
    if not preference_llm_enabled():
        # [偏好 LLM 关闭] 跳过 standardize/classify LLM：原文直通 prefs_std（无 std/class），
        # 不种任何 obligation/event（结构化义务由确定性层放弃，决策 LLM 读原文自行权衡）。
        return {"prefs_std": [{"src": i, "raw": str(p.get("content", "") or "")}
                              for i, p in enumerate(preferences or [])],
                "obligations_seed": [], "freeform_seed": [], "_init_ok": True}
    payload = {
        "mode": "INIT",
        "preferences": [
            {"src": i, "content": str(p.get("content", "")),
             "penalty_amount": p.get("penalty_amount"), "penalty_cap": p.get("penalty_cap"),
             "start_time": p.get("start_time"), "end_time": p.get("end_time")}
            for i, p in enumerate(preferences)
        ],
    }
    try:
        out = _call_json(api, payload)
    except Exception as exc:  # noqa: BLE001 — experiment harness, log and degrade
        _log.warning("%s preference_llm INIT failed: %s%s", _tag("REACT_FAIL"), exc, _line_end())
        return {"prefs_std": [], "obligations_seed": [], "freeform_seed": []}
    out.setdefault("prefs_std", [])
    out.setdefault("obligations_seed", [])
    out.setdefault("freeform_seed", [])
    out["_init_ok"] = True  # success sentinel; the except-path omits it so the loop can retry
    return out
