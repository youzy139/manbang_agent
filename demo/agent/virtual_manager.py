"""Virtual Manager LLM（策略场维护者 · 新架构 C3）。

**不直接决策动作**——它像编译器：读偏好状态 + 账本事实 + 当前虚拟单，输出 **virtual patch**
（add / cancel / update / no_change），系统应用后由 plan_route 在策略场里做确定性最优选择。

契约：
    maintain(api, context, logger) -> {"add":[spec...], "cancel":[id...],
                                       "update":[{id,...}...], "no_change":[id...], "reason": str}

``context``（系统侧组装，见 loop._build_manager_context）：
  now {date, weekday, is_weekend, hhmm, sim_min, day}
  pos [lat, lng]
  pref_status[]   {id, raw_text(司机原话/最终判准), canonical_text(去歧义清楚稿), facts}  —— canonical 是改写辅助稿、
                  可能曲解，编单的钟点/坐标/条件/数量以 raw_text 为准；facts 含 penalty_amount/penalty_cap、剔除文本字段；
                  **无预标 status**，manager 自行据 facts + ledger_facts 判定 due/behind/at_risk（见 prompt SOP 第3步）
  ledger_facts    {month_day_range, completed_orders, month_category_counts,
                  prev_month, prev_month_category_counts(上月品类完成数·跨月补欠额用),
                  month_long_haul_count, month_deadhead_km(累计空驶), month_haul_km,
                  day_meters{today, by_date(近7天)·含gross}(日级米表·"每天≤X/当日收入"类·任意日期查 loop.day_meters_for_date),
                  pacing{day_of_month, days_in_month, days_left, month_elapsed_frac}(配额节奏锚·判进度是否落后),
                  continuous_driving{minutes,text,break_reset_min}(自上次休息连续驾驶·工时触发休息类用),
                  rest_days{month_zero_work_days,last_zero_work_date,days_since_last_zero_work}("每月/周/N天休M天"类用),
                  last_real_order, recent_orders[](均带 category/haul_km/haul_minutes/pickup起点/drop终点/end_min/gross)}
  geo_hints{}    偏好提到的地名→[lat,lng]（GEO_HINTS 参考表命中项·原文无坐标时编 geo 谓词用）
  hot_zones[]     LongMemory 观测的热区中心坐标（注 deadhead 目标用，避免凭空编坐标）
  virtuals[]      registry.view()：active/pending/consumed/expired，含 kind/window/value/predicate；
                  category 谓词的 cargo_modifier 另绑 category_count_this_month(本月该品类实接数,判达标只读它)
                  + category_count_prev_month(上月,仅补欠用)——达标(≥目标)必把该升值单 id 落到 cancel
  modifier_effects (可空)：**上一步规划实测效果**（loop._collect_modifier_effects）——
                  per_modifier[{id, pool_hits(候选池命中跳数), market_count(市场满足谓词货数)}] +
                  top_route_yields(前5候选 yield 标尺)。manager 据实调场（治哑火与盲调抖动）
  harness_feedback (可空)：上一轮 harness 打回的原因 → 据此修正 patch

容错：模型/解析失败 → 返回空 patch（no-op，绝不打断主循环）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import model_io
from .log_color import _line_end, _tag

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "virtual_manager_prompt.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip() if PROMPT_PATH.exists() else ""

_log = logging.getLogger("llm_agent.virtual_manager")

_EMPTY_PATCH = {"add": [], "cancel": [], "update": [], "no_change": [], "reason": ""}


def maintain(api: Any, context: dict[str, Any], logger: logging.Logger | None = None) -> dict[str, Any]:
    """跑一次策略场维护，返回 virtual patch。任何错误降级为空 patch（no-op）。"""
    log = logger or _log
    if api is None or not SYSTEM_PROMPT:
        return dict(_EMPTY_PATCH)
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
        "temperature": 0.0,  # 编译器式：同输入尽量同输出（复现性 + 注入不抖动）
    }
    try:
        content = model_io.call_content_with_retry(api, payload, log)
        log.info("%s virtual_manager patch=%s%s", _tag("VMGR"), content, _line_end())
        obj = json.loads(content)
        if not isinstance(obj, dict):
            raise ValueError("virtual_manager response is not a JSON object")
    except Exception as exc:  # noqa: BLE001 — 实验台，降级为空 patch
        log.warning("%s virtual_manager failed: %s%s", _tag("VMGR_FAIL"), exc, _line_end())
        return dict(_EMPTY_PATCH)
    return normalize_patch(obj)


def normalize_patch(obj: dict[str, Any]) -> dict[str, Any]:
    """把模型输出规范成 {add,cancel,update,no_change,reason}，剔除明显非法项。
    registry.apply_patch 再做 schema 校验（非法 spec 自然被拒，不崩）。"""
    def _list(x: Any) -> list:
        return list(x) if isinstance(x, list) else []
    add = [s for s in _list(obj.get("add")) if isinstance(s, dict) and s.get("kind")]
    cancel = [str(i) for i in _list(obj.get("cancel")) if str(i).strip()]
    update = [s for s in _list(obj.get("update")) if isinstance(s, dict) and s.get("id")]
    no_change = [str(i) for i in _list(obj.get("no_change")) if str(i).strip()]
    return {
        "add": add, "cancel": cancel, "update": update, "no_change": no_change,
        "reason": str(obj.get("reason") or "")[:300],
    }
