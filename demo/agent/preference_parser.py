"""Parse driver preferences once at driver initialization.

The parser LLM turns natural-language preference content into structured fields
that the ReAct loop and route planner can consume:
- categories: priced as soft penalties in plan_routes, not auto-banned.
- rest_type in {continuous_daily, fixed_window, monthly_days}: exposed as
  compliance context and, when possible, priced into planned routes.
- first_order_deadline_hour: priced when a route starts a day's first order late.
- distance limits and daily location deadlines: priced into planned routes.
- geo_bbox / geo_circle: priced when geo_constraint_type is allowed_region or forbidden_region.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from simkit.ports import SimulationApiPort

_SIM_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
_WALL_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _coerce_float(x: Any, default: float | None) -> float | None:
    """容错转 float；bool/None/不可转 → default。透传服务端 penalty_amount/penalty_cap 用。"""
    if isinstance(x, bool) or x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _wall_to_sim_min(text: str) -> int | None:
    """Parse 'YYYY-MM-DD HH:MM:SS' to minutes since the simulation epoch."""
    try:
        dt = datetime.strptime(str(text).strip(), _WALL_TIME_FORMAT)
    except (TypeError, ValueError):
        return None
    delta = dt - _SIM_EPOCH
    return int(delta.total_seconds() // 60)


def _sim_min_to_wall(sim_min: Any) -> str | None:
    """Inverse of ``_wall_to_sim_min``: an absolute simulation minute back to its
    'YYYY-MM-DD HH:MM' wall-clock text. Used to render itinerary time fields for
    the review LLM: a raw sim_min like 43200 (= 3/31 00:00) reads to a human as
    "12:00"/"720-of-day" and the reviewer kept "correcting" already-correct
    parses into garbage — showing the walltime removes that whole failure mode."""
    try:
        m = int(sim_min)
    except (TypeError, ValueError):
        return None
    return (_SIM_EPOCH + timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M")


def _read_prompt_or_empty(name: str) -> str:
    """读 prompts/<name>，文件不存在则返回 ""。偏好 LLM 默认关闭(PREFERENCE_LLM_ENABLED=0)、其 prompt
    已删，故 import 必须容忍缺失;若日后置 =1 重启解析,需把对应 prompt 文件放回。"""
    p = Path(__file__).resolve().parent / "prompts" / name
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


PARSE_SYSTEM_PROMPT = _read_prompt_or_empty("preference_parse_prompt.txt")
REWRITE_SYSTEM_PROMPT = _read_prompt_or_empty("preference_rewrite_prompt.txt")
REVIEW_SYSTEM_PROMPT = _read_prompt_or_empty("preference_review_prompt.txt")

REST_TYPE_CONTINUOUS_DAILY = "continuous_daily"
REST_TYPE_FIXED_WINDOW = "fixed_window"
REST_TYPE_MONTHLY_DAYS = "monthly_days"
_VALID_REST_TYPES = {REST_TYPE_CONTINUOUS_DAILY, REST_TYPE_FIXED_WINDOW, REST_TYPE_MONTHLY_DAYS}
GEO_CONSTRAINT_ALLOWED_REGION = "allowed_region"
GEO_CONSTRAINT_FORBIDDEN_REGION = "forbidden_region"
GEO_CONSTRAINT_VISIT_TARGET = "visit_target"
_VALID_GEO_CONSTRAINT_TYPES = {
    GEO_CONSTRAINT_ALLOWED_REGION,
    GEO_CONSTRAINT_FORBIDDEN_REGION,
    GEO_CONSTRAINT_VISIT_TARGET,
}

MINUTES_PER_DAY = 1440


@dataclass
class ParsedPreference:
    raw_content: str
    # 「改写器」(PREFERENCE_REWRITE_ONLY)产出的无歧义清楚稿:非空时 ledger 用它当 canonical_text 喂决策 LLM,
    # 原文仍由 raw_content 保留(pref_id 仍按 raw_content md5,不变)。关时为 None → canonical 回退到原文。
    clarified_text: str | None = None
    penalty_amount: float = 0.0
    penalty_cap: float | None = None
    # Graduated per-occurrence amounts (越接越贵 / 越漏越贵): the k-th violation costs
    # penalty_tiers[min(k, len-1)]. None = flat penalty_amount. Applied to this
    # preference's PER_OCCURRENCE constraints at compile time.
    penalty_tiers: list[float] | None = None
    active_start_min: int | None = None
    active_end_min: int | None = None
    excluded_categories: list[str] = field(default_factory=list)
    required_categories: list[str] = field(default_factory=list)
    # City/address text constraints: if a cargo pickup or drop endpoint text
    # contains any of these locations, accepting that cargo is penalized once.
    forbidden_endpoint_locations: list[str] = field(default_factory=list)
    # Positive endpoint-location target: accepting real cargos whose pickup or
    # drop text matches these locations earns a planning bonus on distinct days
    # until required_endpoint_location_days is reached.
    required_endpoint_locations: list[str] = field(default_factory=list)
    required_endpoint_location_days: int | None = None
    required_cargo_ids: list[str] = field(default_factory=list)
    # Metadata for the required cargo: pickup location + release/deadline times.
    # Populated when required_cargo_ids is non-empty, so plan_routes can pull
    # the driver toward the pickup point even before the cargo is visible.
    required_cargo_pickup: dict[str, Any] | None = None
    required_cargo_release_time_min: int | None = None
    required_cargo_deadline_time_min: int | None = None
    # itinerary_commitment: ordered list of {event_id, type, lat/lng or
    # location_name, dwell_min, must_complete_before_min, until_min,
    # start_after_event_id} describing a
    # temporal-commitment task (e.g. 先接人再回家再静止数日).
    itinerary_commitment: list[dict[str, Any]] = field(default_factory=list)
    rest_type: str | None = None
    rest_continuous_hours: float | None = None
    rest_window_start_hour: int | None = None
    rest_window_end_hour: int | None = None
    rest_window_crosses_midnight: bool = False
    rest_monthly_days: int | None = None
    first_order_deadline_hour: int | None = None
    max_haul_km: float | None = None
    max_pickup_deadhead_km: float | None = None
    max_deadhead_km: float | None = None
    forbidden_hours: list[dict[str, Any]] = field(default_factory=list)
    # max_idle_gap_between_orders_min: 相邻两笔已接订单的空闲间隔上限(分钟);超一次罚一次。
    # 时间相邻约束(空间衔接已由 max_pickup_deadhead_km 覆盖)。
    max_idle_gap_between_orders_min: int | None = None
    daily_location_deadline_hour: int | None = None
    daily_location_circle: dict[str, Any] | None = None
    # aggregate_threshold KPI constraints: each item is
    # {metric, window, aggregate, op, value, n_days?, penalty_fn}. Aggregates a
    # metric (accepted_orders/income/distance_km) over a window
    # (natural_day/whole_month/rolling_N_days) and penalizes windows that fail the
    # comparator. Empty for all current preferences (additive new capability).
    aggregate_constraints: list[dict[str, Any]] = field(default_factory=list)
    # recurring_visits: "每隔 N 天去某地" — each item is
    # {center_lat, center_lng, radius_km?, period_days, dwell_min?, anchor_min?}
    # (or {circle:{center_lat,center_lng,radius_km}, period_days, ...}). Requires at
    # least one qualifying visit per consecutive N-day bucket. Additive new capability.
    recurring_visits: list[dict[str, Any]] = field(default_factory=list)
    # activation_guard: {metric, op, value} — a finite predicate that gates this
    # preference's prohibit penalties (e.g. only forbid cold-chain once
    # daily_income > 800). metric ∈ {daily_income, daily_accepted_orders,
    # monthly_income}; op ∈ {>,>=,<,<=,==}. None for all current preferences.
    activation_guard: dict[str, Any] | None = None
    # cargo_attribute_filters: per-order prohibit predicates over a cargo attribute.
    # Each item is {attribute, op, value} for numeric attributes
    # (cargo_value=货值/价格, cost_time_minutes=装运耗时) or
    # {attribute:"truck_length", value:"4.2米"} for a truck-length membership test.
    # Optional "negate":true flips it into a whitelist (penalize cargo NOT matching).
    cargo_attribute_filters: list[dict[str, Any]] = field(default_factory=list)
    # sequence_constraints: relations over the ordered stream of accepted orders. Each
    # item is {relation, ...}: adjacency_implication {antecedent, consequent},
    # adjacency_distinct {distinct_key}, or window_quota {antecedent, window_n,
    # comparator, value}. Predicates are {attribute(haul_km/cargo_value/
    # cost_time_minutes), op, value} | {category:<t>} | {region:<t>} (+optional negate).
    # The compiler validates again; malformed items are dropped.
    sequence_constraints: list[dict[str, Any]] = field(default_factory=list)
    # or_alternatives: OR / fallback — a list of already-parsed sub-preferences that
    # are ALTERNATIVES ("满足 A 或 B 即可"). The compiler tags each alternative's
    # (single) constraint with a shared or_group, so satisfying any one satisfies the
    # whole group. None for all current preferences. Each sub-pref's own
    # or_alternatives is cleared (no nesting).
    or_alternatives: "list[ParsedPreference] | None" = None
    geo_constraint_type: str | None = None
    geo_bbox: dict[str, Any] | None = None
    geo_circle: dict[str, Any] | None = None
    # When geo_constraint_type == "visit_target", how many distinct days the
    # driver should reach the circle. Used to size the per-visit virtual cargo
    # price (price = penalty_cap / visit_target_days, or penalty_amount fallback).
    visit_target_days: int | None = None
    # --- operational obligations (运营义务): generic stateful constraints whose trigger
    # is a CUMULATIVE COUNTER over the trajectory, not a clause over one order/position.
    # All three reuse existing machinery (rest virtual / reach virtual / route pricing);
    # only the TRIGGER (a meter threshold or a category gate) is new. Numbers/coords/
    # categories come 100% from the LLM parse — zero literals in code, so the SAME type in
    # a different scenario/wording (电车每300km充电 / 连开5h歇半小时 / 冰淇淋2h送达) is
    # recognized just as well. See operational_meters.py + obligation_injector.py.
    #
    # driving_limits: 疲劳驾驶计时. {max_continuous_drive_min, required_break_min,
    # max_daily_drive_min?}. Continuous-drive meter >= limit -> forced in-place rest; daily
    # meter near cap -> pricing disincentive + signal. NOT a rest_type (那是"每天一段休息";
    # 这是"连续驾驶到点强制歇"——不同语义, 走这里不污染 is_rest).
    driving_limits: dict[str, Any] | None = None
    # range_obligation: 续航/补给. {max_range_km, stations:[{lat,lng,label?}], service_min}.
    # km-since-last-supply meter >= max_range_km -> reach virtual to the NEAREST station.
    # Generic over 油/电/水 — only distance + a supply-point set.
    range_obligation: dict[str, Any] | None = None
    # transit_time_limits: 品类时效. [{categories:[...], max_minutes}]. A candidate route
    # whose first cargo's category matches and whose deliver-time-from-now exceeds the
    # limit is priced down (same shape as the deadhead / first-order penalties).
    transit_time_limits: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------- parsing -----------------


def preference_llm_enabled() -> bool:
    """偏好解析/改写/复审 LLM 总开关。**默认关闭**（``PREFERENCE_LLM_ENABLED`` 未设或≠1）：偏好不经
    任何 LLM，原文裸包成 ParsedPreference 直喂决策 LLM（无结构化字段 → override/约束内核自然 no-op）。
    设环境变量 ``PREFERENCE_LLM_ENABLED=1`` 恢复 standardize/rewrite/parse/review 全套结构化解析。
    每次调用都重读 env，便于运行中切换 / 单测 monkeypatch。"""
    return os.environ.get("PREFERENCE_LLM_ENABLED", "0").strip() in ("1", "true", "True", "yes", "on")


def preference_rewrite_only_enabled() -> bool:
    """「只跑改写器」开关。**默认开启**(用户 2026-06-09 直接打开;设 ``PREFERENCE_REWRITE_ONLY=0`` 可关)。
    在纯 LLM-led(总开关 PREFERENCE_LLM_ENABLED 仍关)的基础上,**只**额外跑一道「改写器」——把每条偏好原文 →
    无歧义、单位规范的清楚稿(clarified_text),决策 LLM 读清楚稿、原文仍作兜底;**parse/分类/复审/结构化闸门/
    override/约束内核一律不开**(架构维持纯 LLM-led+虚拟单)。改写一次性(parse 层按内容缓存),失败回退原文。
    若总开关 PREFERENCE_LLM_ENABLED 已开,则走完整结构化流程、本开关无意义。每次读 env,便于切换/单测。"""
    return os.environ.get("PREFERENCE_REWRITE_ONLY", "1").strip() in ("1", "true", "True", "yes", "on")


def parse_driver_preferences(
    api: SimulationApiPort,
    preferences: list[dict[str, Any]],
    logger: logging.Logger | None = None,
) -> list[ParsedPreference]:
    """LLM-based parse, one LLM call per preference.

    Parsing each preference in its own call removes the cross-preference
    failure modes of a single batched call: the model can no longer mis-assign
    one preference's structured fields to another preference's index, nor drop a
    preference from the returned array. A failed single call degrades only that
    preference (it falls back to raw-text field extractors), never the whole
    driver. The per-call result's reported index is forced to the real index,
    then the existing per-field assembly in ``_build_parsed_list`` runs unchanged.
    """
    log = logger or logging.getLogger("agent.preference_parser")
    if not preferences:
        return []
    if not preference_llm_enabled():
        # [偏好 LLM 关闭] 跳过 解析/分类/复审/结构化:每条原文裸包成 ParsedPreference(只有 raw_content),
        # 下游 is_cargo/is_event/is_rest 全 False → 归入 other_prefs(long_tail)喂决策 LLM。
        # **可选只跑「改写器」**(PREFERENCE_REWRITE_ONLY=1):额外把原文→无歧义清楚稿 clarified_text(原文仍留),
        # 决策 LLM 读清楚稿;其余结构化/override/内核仍不开。改写一次性(parse 层按内容缓存),失败回退原文。
        rewrite_on = preference_rewrite_only_enabled()
        out: list[ParsedPreference] = []
        for p in preferences:
            raw = str(p.get("content", "") or "")
            clarified = None
            if rewrite_on and raw.strip():
                try:
                    from . import preference_llm  # 延迟导入避免与 preference_llm 的循环依赖
                    clarified = preference_llm.rewrite_clarify(api, raw, log)
                except Exception:  # noqa: BLE001 — 改写绝不打断,回退原文
                    clarified = None
            # 透传服务端罚款口径（get_driver_status 每条偏好仅含 content/penalty_amount/penalty_cap，见数据说明），
            # 别再硬置 0.0——Virtual Manager 据此给 cargo_modifier 定价（罚额≥违规罚款才能把违规货 adjusted 驱负）。
            out.append(ParsedPreference(
                raw_content=raw, clarified_text=clarified,
                penalty_amount=_coerce_float(p.get("penalty_amount"), 0.0),
                penalty_cap=_coerce_float(p.get("penalty_cap"), None),
            ))
        log.info("[%s] preference passthrough (%d prefs)%s",
                 "PREF_REWRITE_ONLY" if rewrite_on else "PREF_LLM_OFF", len(preferences),
                 " — rewrote to clarified text" if rewrite_on else " — raw, no rewrite/parse/review")
        return out
    out: list[ParsedPreference] = []
    for i, p in enumerate(preferences):
        out.append(_parse_one_with_selfcheck(api, p, i, log))
    # Cross-preference review pass: a review LLM inspects every parsed result
    # together (original text + structured fields), flags the ones that look
    # wrong, and each flagged preference is re-parsed once with the reviewer's
    # targeted complaint. This catches mistakes the per-preference deterministic
    # self-check cannot — type confusions (端点禁接 vs forbidden_region,
    # visit_target vs allowed_region), lat/lng swaps, closed-vocab overshoots,
    # and partial extraction of multi-rule preferences. Runs before the location
    # registry so the registry fills coordinates on the repaired list.
    _review_and_repair_preferences(api, preferences, out, log)
    # Cross-preference pass: a coordinate stated in one preference (e.g. the
    # 增城≥4日 rule's "（23.15，113.67）") resolves a sibling itinerary commitment
    # that only names the place ("增城老档口"), avoiding cargo-graph drift.
    apply_preference_location_registry(out, log)
    return out


def _parse_one_with_selfcheck(
    api: SimulationApiPort,
    pref: Any,
    index: int,
    log: logging.Logger,
) -> ParsedPreference:
    """Parse one preference, then run a deterministic self-check and, on a
    high-confidence extraction gap, retry that single preference once with a
    targeted complaint. Keeps whichever attempt has fewer detected gaps.

    The self-check compares raw-text signals against the assembled
    ``ParsedPreference`` (post raw-text fallbacks), so it only fires when the
    structured result genuinely failed to carry something the text clearly
    states. This bounds extra LLM calls to at most one per under-extracted
    preference.
    """
    if not isinstance(pref, dict):
        return ParsedPreference(raw_content="", penalty_amount=0.0)
    content = str(pref.get("content", ""))

    def build(item: dict[str, Any] | None) -> ParsedPreference:
        # _build_parsed_list maps results to prefs by the result's reported
        # ``index`` against the local enumerate position. Here the local list
        # has exactly one pref at position 0, so the single item must report
        # index 0 — otherwise (for any real index != 0) the lookup misses and
        # the LLM fields are silently dropped to the raw-text fallback. The real
        # index is only used for logging / retry feedback above.
        if isinstance(item, dict):
            item = {**item, "index": 0}
        items = [item] if item is not None else []
        return _build_parsed_list([pref], {"preferences": items})[0]

    rewrite = _rewrite_preference(api, pref, index, log)
    item = _parse_single_preference(api, pref, index, log, rewrite=rewrite)
    obj = build(item)
    gaps = _detect_extraction_gaps(obj, content)
    if not gaps:
        return obj

    log.info("[PREF_SELFCHECK] idx=%s gaps=%s -> retry", index, gaps)
    retry_item = _parse_single_preference(
        api, pref, index, log, extra_feedback=_gap_feedback_text(gaps), rewrite=rewrite
    )
    if retry_item is None:
        return obj
    retry_obj = build(retry_item)
    retry_gaps = _detect_extraction_gaps(retry_obj, content)
    # Prefer the attempt with fewer gaps; on a tie prefer the one that extracted
    # more structured rule types; final tie keeps the retry (latest signal).
    if len(retry_gaps) < len(gaps):
        return retry_obj
    if len(retry_gaps) == len(gaps) and len(_constraint_types(retry_obj)) >= len(_constraint_types(obj)):
        return retry_obj
    return obj


# Raw-text tokens that signal an actionable preference rule should exist.
_ACTIONABLE_GAP_TOKENS = (
    "不接", "推掉", "干不了", "不想接", "不要", "一律", "禁",
    "休息", "睡觉", "熄火", "停车", "整天", "歇", "保养", "检修",
    "回家", "赶到", "停留", "停一", "打卡", "拜访", "档口",
    "不超过", "不得超过", "公里", "点前", "之前", "避开", "不进", "不往", "不出车",
)

# (X月X号 / X月X日) date markers, Chinese numerals or digits.
_DATE_GAP_RE = re.compile(r"[一二三四五六七八九十百零0-9]+月[一二三四五六七八九十百零0-9]+\s*[号日]")
# Explicit coordinate pair like （23.15，113.67） or (23.32, 112.83).
_COORD_GAP_RE = re.compile(r"[（(]\s*\d{1,3}\.\d+\s*[，,]\s*\d{1,3}\.\d+")


def _detect_extraction_gaps(p: ParsedPreference, content: str) -> list[str]:
    """Deterministic high-confidence under-extraction signals for one preference."""
    text = content or ""
    gaps: list[str] = []
    if p.penalty_amount > 0 and not _constraint_types(p) and any(t in text for t in _ACTIONABLE_GAP_TOKENS):
        gaps.append("penalty>0 但未抽到任何结构化规则")
    if _DATE_GAP_RE.search(text):
        dated = bool(p.itinerary_commitment) or p.rest_type == REST_TYPE_MONTHLY_DAYS or bool(p.required_cargo_ids)
        if not dated:
            gaps.append("原文含具体日期(X月X号)，但未抽到行程/时限/月度承诺")
    if _COORD_GAP_RE.search(text):
        events = p.itinerary_commitment or []
        has_coord_carrier = bool(
            p.geo_circle
            or p.geo_bbox
            or p.daily_location_circle
            or p.required_cargo_pickup
            or any((ev.get("lat") is not None and ev.get("lng") is not None) or ev.get("location_name") for ev in events)
        )
        if not has_coord_carrier:
            gaps.append("原文含坐标，但结构化字段未携带坐标/地名")
    return gaps


def _gap_feedback_text(gaps: list[str]) -> str:
    return (
        "上一次解析可能遗漏了以下内容，请重新仔细解析这条偏好并补全相应结构化字段："
        + "；".join(gaps)
    )


def _nonempty_parsed_fields(p: ParsedPreference) -> dict[str, Any]:
    """Compact view of an assembled ``ParsedPreference`` for the review payload:
    drop ``raw_content`` and every empty/default field, then attach the derived
    ``_constraint_types`` so the reviewer sees exactly what was (and wasn't)
    extracted without wading through nulls."""
    out: dict[str, Any] = {}
    for key, value in p.to_dict().items():
        if key == "raw_content":
            continue
        if value in (None, "", [], {}, False):
            continue
        if key == "itinerary_commitment":
            value = _itinerary_review_view(value)
        out[key] = value
    out["_constraint_types"] = _constraint_types(p)
    return out


# Itinerary time fields are stored as ABSOLUTE simulation minutes (minutes since
# 2026-03-01 00:00). The review LLM cannot do that arithmetic — it reads 43200 as
# "12:00", 43919 as "7:19" etc., flags correct parses as wrong, and pushes
# day-of-clock numbers like 720 (= 3/1 12:00) that wreck a 3/31 banquet. Render
# every *_min field as wall-clock text in the review view so the reviewer reasons
# on what a human sees, not on raw minutes.
_ITINERARY_MIN_FIELDS = ("not_before_min", "must_complete_before_min", "until_min")


def _itinerary_review_view(events: Any) -> Any:
    if not isinstance(events, list):
        return events
    view: list[Any] = []
    for ev in events:
        if not isinstance(ev, dict):
            view.append(ev)
            continue
        shown = dict(ev)
        for fld in _ITINERARY_MIN_FIELDS:
            if shown.get(fld) is None:
                continue
            wall = _sim_min_to_wall(shown.pop(fld))
            if wall is not None:
                shown[fld[:-4] if fld.endswith("_min") else fld] = wall
        view.append(shown)
    return view


def _review_parsed_preferences(
    api: SimulationApiPort,
    preferences: list[dict[str, Any]],
    parsed: list[ParsedPreference],
    log: logging.Logger,
) -> dict[int, str]:
    """Cross-preference review: hand the reviewer every preference's original
    text + structured result and collect per-index complaints for the ones that
    look wrong. Returns {index: feedback}. On any failure returns {} so parsing
    proceeds on the un-reviewed results (the review is strictly additive)."""
    items: list[dict[str, Any]] = []
    for i, (pref, p) in enumerate(zip(preferences, parsed)):
        if not isinstance(pref, dict):
            continue
        items.append(
            {
                "index": i,
                "content": str(pref.get("content", "")),
                "penalty_amount": float(pref.get("penalty_amount", 0) or 0),
                "penalty_cap": _penalty_cap(pref.get("penalty_cap")),
                "start_time": pref.get("start_time"),
                "end_time": pref.get("end_time"),
                "parsed": _nonempty_parsed_fields(p),
            }
        )
    if not items:
        return {}
    try:
        resp = api.model_chat_completion(
            {
                "messages": [
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({"preferences": items}, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
                "enable_thinking": False,
            }
        )
        choices = resp.get("choices") if isinstance(resp, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError("review response missing choices")
        text = choices[0].get("message", {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("review response content empty")
        obj = json.loads(text)
        issues = obj.get("issues") if isinstance(obj, dict) else None
        if not isinstance(issues, list):
            return {}
        feedback: dict[int, str] = {}
        for it in issues:
            if not isinstance(it, dict):
                continue
            try:
                idx = int(it.get("index"))
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(parsed)):
                continue
            problem = str(it.get("problem") or "").strip()
            suggestion = str(it.get("suggestion") or "").strip()
            merged = "；".join(x for x in (problem, suggestion) if x)
            if not merged:
                continue
            # Same index may appear twice; concatenate rather than overwrite.
            feedback[idx] = f"{feedback[idx]}；{merged}" if idx in feedback else merged
        return feedback
    except Exception as exc:
        log.warning("preference_review_skip: %s", exc)
        return {}


def _choose_repaired_preference(
    old: ParsedPreference, new: ParsedPreference, content: str
) -> ParsedPreference:
    """Accept the review-driven re-parse unless it regresses on the deterministic
    under-extraction signals (introduces strictly more gaps than before). This
    guards against a reparse that, while addressing the reviewer's point, nukes
    correctly-extracted constraints."""
    if len(_detect_extraction_gaps(new, content)) > len(_detect_extraction_gaps(old, content)):
        return old
    return new


# Max review→repair rounds. Each round is one review LLM call plus one
# rewrite+parse per flagged preference. Capped to bound LLM cost and stop the
# review/reparse pair from oscillating on a preference the reviewer keeps
# disliking. Round 1 fixes, round 2 confirms (and fixes anything still flagged).
_MAX_REVIEW_ROUNDS = 2


def _repair_one_preference(
    api: SimulationApiPort,
    pref: dict[str, Any],
    idx: int,
    complaint: str,
    current: ParsedPreference,
    log: logging.Logger,
) -> ParsedPreference | None:
    """Re-parse one flagged preference with the reviewer's complaint as targeted
    feedback. Returns the repaired ``ParsedPreference`` only when it is both
    valid and an actual improvement (changed and not regressing on deterministic
    gaps); returns ``None`` to keep the current parse."""
    rewrite = _rewrite_preference(api, pref, idx, log)
    item = _parse_single_preference(
        api,
        pref,
        idx,
        log,
        extra_feedback="审校发现本条解析可能有问题，请据此重新解析并修正：" + complaint,
        rewrite=rewrite,
    )
    if item is None:
        log.info("[PREF_REVIEW] idx=%s reparse failed, keeping current. issue=%s", idx, complaint)
        return None
    # _build_parsed_list maps results to prefs by the result's reported
    # ``index`` against the local enumerate position (here a single pref at
    # position 0), so force index 0 — see _parse_one_with_selfcheck.build.
    new_obj = _build_parsed_list([pref], {"preferences": [{**item, "index": 0}]})[0]
    chosen = _choose_repaired_preference(current, new_obj, str(pref.get("content", "")))
    if chosen is not new_obj:
        log.info("[PREF_REVIEW] idx=%s reparse rejected (would add gaps), keeping current. issue=%s", idx, complaint)
        return None
    if chosen.to_dict() == current.to_dict():
        # Reviewer flagged it but the reparse produced the same fields — treat as
        # no-op so the loop can settle instead of re-reviewing an unchanged list.
        log.info("[PREF_REVIEW] idx=%s reparse unchanged, keeping current. issue=%s", idx, complaint)
        return None
    log.info("[PREF_REVIEW] idx=%s reparsed per review. issue=%s", idx, complaint)
    return chosen


def _review_and_repair_preferences(
    api: SimulationApiPort,
    preferences: list[dict[str, Any]],
    parsed: list[ParsedPreference],
    log: logging.Logger,
) -> None:
    """Iteratively review all parsed preferences together and re-parse the
    flagged ones, then review again to confirm. Mutates ``parsed`` in place.

    Bounded by ``_MAX_REVIEW_ROUNDS``: each round runs one review call and at
    most one rewrite+parse per flagged preference. The loop stops early once a
    round produces no complaints (everything confirmed clean) or no preference
    actually changes (the reviewer keeps disliking something but the reparse
    can't improve it), so it never oscillates."""
    for round_no in range(1, _MAX_REVIEW_ROUNDS + 1):
        feedback = _review_parsed_preferences(api, preferences, parsed, log)
        if not feedback:
            if round_no > 1:
                log.info("[PREF_REVIEW] round=%s clean, all preferences confirmed", round_no)
            break
        repaired_any = False
        for idx, complaint in feedback.items():
            pref = preferences[idx]
            if not isinstance(pref, dict):
                continue
            repaired = _repair_one_preference(api, pref, idx, complaint, parsed[idx], log)
            if repaired is not None:
                parsed[idx] = repaired
                repaired_any = True
        if not repaired_any:
            log.info("[PREF_REVIEW] round=%s flagged but nothing improved, stopping", round_no)
            break


def _rewrite_preference(
    api: SimulationApiPort,
    pref: dict[str, Any],
    index: int,
    log: logging.Logger,
) -> dict[str, Any] | None:
    """Normalization-rewrite pass: expand one vague preference into controlled
    text + atomic clauses + parser hints before structured parsing.

    This only normalizes (Chinese numerals/time phrases/coords) and splits
    multi-action preferences; it never invents facts and never makes the final
    structured decision. Returns ``None`` on any failure so parsing proceeds on
    the original content alone (the rewrite is strictly additive context).
    """
    content = str(pref.get("content", ""))
    if not content.strip():
        return None
    payload = {
        "content": content,
        "penalty_amount": float(pref.get("penalty_amount", 0) or 0),
        "penalty_cap": _penalty_cap(pref.get("penalty_cap")),
        "start_time": pref.get("start_time"),
        "end_time": pref.get("end_time"),
    }
    try:
        resp = api.model_chat_completion(
            {
                "messages": [
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
                "enable_thinking": False,
            }
        )
        choices = resp.get("choices") if isinstance(resp, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError("rewrite response missing choices")
        text = choices[0].get("message", {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("rewrite response content empty")
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("rewrite response is not an object")
        return _sanitize_rewrite(obj)
    except Exception as exc:
        log.warning("preference_rewrite_skip idx=%s: %s", index, exc)
        return None


# Whitelisted parser_hints keys; anything else from the rewrite model is dropped.
_REWRITE_HINT_KEYS = {
    "hint_kind",
    "hint_polarity",
    "hint_combine",
    "contains_location",
    "contains_deadline",
    "contains_duration",
    "contains_cargo_endpoint_rule",
    "contains_vehicle_region_rule",
}
_REWRITE_MAX_CLAUSES = 12
_REWRITE_MAX_CLAUSE_LEN = 200
_REWRITE_MAX_NORMALIZED_LEN = 1200


def _sanitize_rewrite(obj: dict[str, Any]) -> dict[str, Any]:
    """Bound and type-clean the rewrite output before it reaches the parser.

    The rewrite is auxiliary context only, so we keep just three shapes:
    ``normalized_text`` (bounded string), ``atomic_clauses`` (bounded list of
    bounded strings), and ``parser_hints`` (whitelisted keys only). This stops a
    long/deeply-nested/odd-keyed rewrite from polluting the next parse payload.
    """
    out: dict[str, Any] = {}
    nt = obj.get("normalized_text")
    if isinstance(nt, str) and nt.strip():
        out["normalized_text"] = nt.strip()[:_REWRITE_MAX_NORMALIZED_LEN]
    clauses = obj.get("atomic_clauses")
    if isinstance(clauses, list):
        cleaned = [
            str(c).strip()[:_REWRITE_MAX_CLAUSE_LEN]
            for c in clauses
            if isinstance(c, str) and c.strip()
        ]
        if cleaned:
            out["atomic_clauses"] = cleaned[:_REWRITE_MAX_CLAUSES]
    hints = obj.get("parser_hints")
    if isinstance(hints, dict):
        kept = {k: hints[k] for k in _REWRITE_HINT_KEYS if k in hints}
        if kept:
            out["parser_hints"] = kept
    return out


def _parse_single_preference(
    api: SimulationApiPort,
    pref: dict[str, Any],
    index: int,
    log: logging.Logger,
    extra_feedback: str | None = None,
    rewrite: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Parse one preference via a single LLM call.

    Returns the model's structured item with its ``index`` forced to the real
    ``index`` (so it can never be mis-mapped), or ``None`` on any failure so the
    caller leaves that preference to the raw-text field extractors. When
    ``extra_feedback`` is given (a deterministic self-check complaint from a
    prior attempt), it is appended to the payload so the retry targets the gap.
    ``rewrite`` (the normalization pass output) is attached as auxiliary context;
    the parser still extracts from the original ``content``.
    """
    pref_payload: dict[str, Any] = {
        "index": 0,
        "content": str(pref.get("content", "")),
        "penalty_amount": float(pref.get("penalty_amount", 0) or 0),
        "penalty_cap": _penalty_cap(pref.get("penalty_cap")),
        "start_time": pref.get("start_time"),
        "end_time": pref.get("end_time"),
    }
    if isinstance(rewrite, dict):
        if rewrite.get("normalized_text"):
            pref_payload["normalized_text"] = str(rewrite["normalized_text"])[:1200]
        if rewrite.get("atomic_clauses"):
            pref_payload["atomic_clauses"] = rewrite["atomic_clauses"]
        if rewrite.get("parser_hints"):
            pref_payload["parser_hints"] = rewrite["parser_hints"]
    if extra_feedback:
        pref_payload["reparse_feedback"] = extra_feedback
    payload = {"preferences": [pref_payload]}
    try:
        resp = api.model_chat_completion(
            {
                "messages": [
                    {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
                "enable_thinking": False,
            }
        )
        choices = resp.get("choices") if isinstance(resp, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError("model response missing choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("model response content empty")
        parsed_obj = json.loads(content)
        items = parsed_obj.get("preferences") if isinstance(parsed_obj, dict) else None
        item = next((it for it in items if isinstance(it, dict)), None) if isinstance(items, list) else None
        if item is None and isinstance(parsed_obj, dict):
            # Some models drop the wrapper and return the single object directly.
            item = parsed_obj if any(k != "preferences" for k in parsed_obj) else None
        if not isinstance(item, dict):
            raise ValueError("model response missing preference item")
        item = dict(item)
        item["index"] = index
        return item
    except Exception as exc:
        log.warning("preference_parser_single_fallback idx=%s: %s", index, exc)
        return None


def _build_parsed_list(prefs: list[dict[str, Any]], parsed_obj: dict[str, Any]) -> list[ParsedPreference]:
    by_idx: dict[int, dict[str, Any]] = {}
    raw_items = parsed_obj.get("preferences") if isinstance(parsed_obj, dict) else None
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict) and "index" in item:
                try:
                    by_idx[int(item["index"])] = item
                except (TypeError, ValueError):
                    continue
    out: list[ParsedPreference] = []
    for i, pref in enumerate(prefs):
        data = by_idx.get(i, {})
        required_cargo_ids = _required_cargo_ids(pref, data)
        required_cargo_pickup = _required_cargo_pickup(pref, data) if required_cargo_ids else None
        required_cargo_release_min = _required_cargo_release_time_min(pref, data) if required_cargo_ids else None
        required_cargo_deadline_min = _required_cargo_deadline_time_min(pref, data) if required_cargo_ids else None
        itinerary_commitment = _itinerary_commitment(pref, data)
        out.append(
            ParsedPreference(
                raw_content=str(pref.get("content", "")),
                penalty_amount=float(pref.get("penalty_amount", 0) or 0),
                penalty_cap=_penalty_cap(pref.get("penalty_cap")),
                penalty_tiers=_penalty_tiers(data),
                active_start_min=_wall_to_sim_min(str(pref.get("start_time", ""))),
                active_end_min=_wall_to_sim_min(str(pref.get("end_time", ""))),
                excluded_categories=_excluded_categories(pref, data),
                required_categories=_str_list(data.get("required_categories")),
                forbidden_endpoint_locations=_forbidden_endpoint_locations(pref, data),
                required_endpoint_locations=_required_endpoint_locations(pref, data),
                required_endpoint_location_days=_required_endpoint_location_days(pref, data),
                required_cargo_ids=required_cargo_ids,
                required_cargo_pickup=required_cargo_pickup,
                required_cargo_release_time_min=required_cargo_release_min,
                required_cargo_deadline_time_min=required_cargo_deadline_min,
                itinerary_commitment=itinerary_commitment,
                rest_type=_rest_type_with_fallback(pref, data),
                rest_continuous_hours=_rest_continuous_hours(pref, data),
                rest_window_start_hour=_rest_window_start_hour(pref, data),
                rest_window_end_hour=_rest_window_end_hour(pref, data),
                rest_window_crosses_midnight=_rest_crosses_midnight_with_fallback(pref, data),
                rest_monthly_days=_pos_int(data.get("rest_monthly_days")),
                first_order_deadline_hour=_first_order_deadline_hour(pref, data),
                max_haul_km=_distance_limit_km(pref, data, "max_haul_km"),
                max_pickup_deadhead_km=_distance_limit_km(pref, data, "max_pickup_deadhead_km"),
                max_idle_gap_between_orders_min=_pos_int(data.get("max_idle_gap_between_orders_min")),
                daily_location_deadline_hour=_daily_location_deadline_hour(pref, data),
                daily_location_circle=_daily_location_circle(pref, data),
                geo_constraint_type=_geo_constraint_type(pref, data),
                geo_bbox=_valid_bbox(data.get("geo_bbox")),
                geo_circle=_valid_circle(data.get("geo_circle")),
                visit_target_days=_visit_target_days(pref, data),
                aggregate_constraints=_aggregate_constraints(data),
                recurring_visits=_recurring_visits(data),
                activation_guard=_activation_guard(data),
                cargo_attribute_filters=_cargo_attribute_filters(data),
                sequence_constraints=_sequence_constraints(data),
                driving_limits=_driving_limits(data),
                range_obligation=_range_obligation(data),
                transit_time_limits=_transit_time_limits(data),
                or_alternatives=_parse_or_alternatives(pref, data),
                notes=str(data.get("notes", "")).strip()[:200],
            )
        )
    for p in out:
        _strip_required_cargo_descriptor_constraints(p)
        _normalize_windowed_continuous_rest(p)
    return out


def _normalize_windowed_continuous_rest(p: ParsedPreference) -> None:
    """"时段窗口内连续休息 N 小时" → fixed_window 的有效时长按 N 小时算(用户口径:
    起点=窗口起点, 长度=N → end=起点+N)。LLM 心算 start+N 不稳(实测半数把 end 留成原窗口
    终点), 故改成确定性:只要是 fixed_window 且带了连续时长 N(由解析器从"连续N小时"读出)就用
    N 重算 end, 然后清掉 N(fixed_window 下游不需要 N, 留着反而可能被误当每日连续休息)。
    纯固定静止窗("23点至8点不接单"无"连续N小时")不带 N, 不受影响。"""
    if p.rest_type != REST_TYPE_FIXED_WINDOW:
        return
    n_hours = p.rest_continuous_hours
    start = p.rest_window_start_hour
    if n_hours is None or start is None or n_hours <= 0:
        return
    new_end = int(round(start + n_hours)) % 24
    if new_end != start:  # N 为 24 的整数倍时退化, 保持原样
        p.rest_window_end_hour = new_end
        p.rest_window_crosses_midnight = start > new_end
    p.rest_continuous_hours = None


def _strip_required_cargo_descriptor_constraints(p: ParsedPreference) -> None:
    """A 必接-specific-cargo pref (``required_cargo_ids`` set) names that cargo's pickup place and
    category only to IDENTIFY it. The LLM extractor sometimes ALSO mirrors those descriptors into
    general constraints on the SAME pref — excluded/required category + forbidden endpoint — which
    then contradict the 必接 itself: the cargo is at 东莞 / is 服饰, yet 东莞 gets walled off and 服饰
    gets both required AND excluded, so the required cargo can NEVER be taken (extra-D002: 254237
    missed = 20000 一次性罚) and a whole region is needlessly forbidden (lost gross). The required
    cargo is the explicit, penalised intent, identified by ``required_cargo_ids`` + ``required_cargo_
    pickup``; strip the mirrored descriptor-constraints so picking it up is legal. No-op for any pref
    without required_cargo_ids — an independent 不接 / 须接 pref is never touched (general, not keyed
    to any specific id/place/category)."""
    if not p.required_cargo_ids:
        return
    if p.excluded_categories or p.required_categories or p.forbidden_endpoint_locations:
        p.excluded_categories = []
        p.required_categories = []
        p.forbidden_endpoint_locations = []


def _opt_num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rest_crosses_midnight(data: dict[str, Any]) -> bool:
    """Deterministically derive cross-midnight from the hours: a window crosses
    midnight iff start_hour > end_hour (22->6 True; 0->6 False). The LLM's flag is
    ignored when both hours are known, so a mis-tagged 0-6 window can never be treated
    as a 30-hour span. Falls back to the LLM flag only if an hour is missing."""
    sh = _hour(data.get("rest_window_start_hour"))
    eh = _hour(data.get("rest_window_end_hour"))
    if sh is not None and eh is not None:
        return sh > eh
    return bool(data.get("rest_window_crosses_midnight", False))


_REST_WINDOW_SIGNALS = ("停运", "停驶", "不接单", "不空跑", "不出车", "不开车", "静止", "熄火", "睡觉", "休息", "停车")
_REST_WINDOW_RANGE_RE = re.compile(
    r"(\d{1,2}|[零一二两三四五六七八九十]{1,3})\s*[点时]"
    r"\s*(?:至|到|~|～|—|－|-|–)\s*"
    r"(次日|次晨|第二天|翌日|凌晨)?\s*"
    r"(\d{1,2}|[零一二两三四五六七八九十]{1,3})\s*[点时]"
)


def _rest_window_from_text(text: str) -> tuple[int, int, bool] | None:
    """Deterministic fallback for a fixed rest window the LLM didn't structure — typical for a
    COMBINED pref ('每天23点前回家(坐标);当天23点至次日8点不接单、不空跑'), where the LLM captures
    only the home half and drops the curfew. Returns (start_hour, end_hour, crosses_midnight) or
    None. Fires ONLY when an explicit rest/停运 signal is present, so a plain time range never
    becomes a rest window. Generic — no specific hours hardcoded."""
    t = text or ""
    if not any(s in t for s in _REST_WINDOW_SIGNALS):
        return None
    m = _REST_WINDOW_RANGE_RE.search(t)
    if not m:
        return None
    sh = int(m.group(1)) if m.group(1).isdigit() else _CN_NUM.get(m.group(1))
    eh = int(m.group(3)) if m.group(3).isdigit() else _CN_NUM.get(m.group(3))
    if sh is None or eh is None:
        return None
    sh = 0 if sh == 24 else sh
    eh = 0 if eh == 24 else eh
    if not (0 <= sh <= 23 and 0 <= eh <= 23) or sh == eh:
        return None
    return sh, eh, bool(m.group(2)) or sh > eh


def _rest_window_start_hour(pref: dict[str, Any], data: dict[str, Any]) -> int | None:
    explicit = _hour(data.get("rest_window_start_hour"))
    if explicit is not None:
        return explicit
    r = _rest_window_from_text(str(pref.get("content", "")))
    return r[0] if r else None


def _rest_window_end_hour(pref: dict[str, Any], data: dict[str, Any]) -> int | None:
    explicit = _hour(data.get("rest_window_end_hour"))
    if explicit is not None:
        return explicit
    r = _rest_window_from_text(str(pref.get("content", "")))
    return r[1] if r else None


def _rest_type_with_fallback(pref: dict[str, Any], data: dict[str, Any]) -> str | None:
    """rest_type, with a text fallback to fixed_window when an explicit curfew window is in the
    text but the LLM didn't tag rest_type (combined pref). continuous_daily / monthly keep their
    own LLM fields — this only recovers the explicit fixed window, never invents the other kinds.

    Sanity guard: an all-day "fixed rest window" (span >= 20h) is NEVER a real daily static window —
    it is a mis-parse (e.g. the parse-reviewer broadening an event day's "这天别排别的活" into a
    00:00-23:00 no_work/static window, which then froze the driver stationary ALL DAY → 6.5h 空等 +
    整天空耗, ds2-D002 家长会). Drop the fixed-window rest in that case; the event (到达+守候) and the
    soft "别接其他单" note still stand, and the driver stays free to earn the rest of the day. All
    downstream readers gate on rest_type=="fixed_window", so returning None neutralizes the bogus window."""
    explicit = _valid_rest_type(data.get("rest_type"))
    rest_type = explicit if explicit is not None else (
        REST_TYPE_FIXED_WINDOW if _rest_window_from_text(str(pref.get("content", ""))) is not None else None
    )
    if rest_type == REST_TYPE_FIXED_WINDOW:
        sh = _rest_window_start_hour(pref, data)
        eh = _rest_window_end_hour(pref, data)
        if sh is not None and eh is not None:
            span_h = (eh - sh) if eh > sh else (eh + 24 - sh)
            if span_h >= 20:
                return None
    return rest_type


def _rest_crosses_midnight_with_fallback(pref: dict[str, Any], data: dict[str, Any]) -> bool:
    sh = _rest_window_start_hour(pref, data)
    eh = _rest_window_end_hour(pref, data)
    if sh is not None and eh is not None:
        return sh > eh
    return bool(data.get("rest_window_crosses_midnight", False))


def _parse_or_alternatives(pref: dict[str, Any], data: dict[str, Any]) -> list[ParsedPreference] | None:
    """Recursively parse OR/fallback alternatives. Each alternative is itself a
    structured-fields object; it is run through the SAME field extractors (so it is
    fully validated) and inherits the parent's penalty/active window. Needs >= 2
    well-formed alternatives, else None. Nesting is not supported (each sub's own
    or_alternatives is cleared)."""
    raw = data.get("or_alternatives")
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    subs: list[ParsedPreference] = []
    for alt in raw:
        if not isinstance(alt, dict) or not alt:
            continue
        sub = _build_parsed_list([pref], {"preferences": [{**alt, "index": 0}]})[0]
        sub.or_alternatives = None
        subs.append(sub)
    return subs if len(subs) >= 2 else None


def _aggregate_constraints(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract aggregate_threshold KPI specs from the LLM output. Structural only —
    the compiler validates the closed vocabulary and drops anything out of range, so
    a hallucinated/partial spec can never corrupt evaluation. Requires metric/window/
    op/value to be present; carries optional aggregate/n_days/penalty_fn through."""
    items = data.get("aggregate_constraints")
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        spec: dict[str, Any] = {}
        for key in ("metric", "window", "aggregate", "op", "penalty_fn", "category", "region"):
            if it.get(key) is not None and str(it.get(key)).strip():
                spec[key] = str(it.get(key)).strip()
        value = _opt_num(it.get("value"))
        if value is not None:
            spec["value"] = value
        n_days = _pos_int(it.get("n_days"))
        if n_days is not None:
            spec["n_days"] = n_days
        # night_active_days carries the daily night window hours (e.g. 22→次日5 点);
        # accept 0-23 整点, the compiler derives crosses_midnight from start>end.
        for hk in ("night_start_hour", "night_end_hour"):
            hv = _hour(it.get(hk))
            if hv is not None:
                spec[hk] = hv
        if spec.get("metric") and spec.get("window") and spec.get("op") and "value" in spec:
            out.append(spec)
        if len(out) >= 10:
            break
    return out


def _recurring_visits(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract recurring_visit specs (每隔 N 天去某地) from the LLM output. Requires a
    center coordinate and a positive period_days; the compiler drops anything else."""
    items = data.get("recurring_visits")
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        circle = it.get("circle") if isinstance(it.get("circle"), dict) else it
        lat = _opt_num(circle.get("center_lat", circle.get("lat")))
        lng = _opt_num(circle.get("center_lng", circle.get("lng")))
        period = _pos_int(it.get("period_days"))
        if lat is None or lng is None or period is None:
            continue
        spec: dict[str, Any] = {"center_lat": lat, "center_lng": lng, "period_days": period}
        radius = _opt_num(circle.get("radius_km"))
        spec["radius_km"] = radius if radius and radius > 0 else 1.0
        dwell = _pos_int(it.get("dwell_min"))
        if dwell is not None:
            spec["dwell_min"] = dwell
        anchor = _pos_int(it.get("anchor_min"))
        if anchor is not None:
            spec["anchor_min"] = anchor
        if it.get("penalty_fn") is not None:
            spec["penalty_fn"] = str(it.get("penalty_fn")).strip()
        out.append(spec)
        if len(out) >= 10:
            break
    return out


def _driving_limits(data: dict[str, Any]) -> dict[str, Any] | None:
    """疲劳驾驶计时 (driving_limits). Generic shape, all in MINUTES (the LLM normalizes
    hours->minutes per the prompt): {max_continuous_drive_min, required_break_min,
    max_daily_drive_min?}. Requires at least a continuous OR a daily limit; nothing is
    keyed on wording. A bare/partial spec returns None so it can never mis-fire."""
    raw = data.get("driving_limits")
    if not isinstance(raw, dict):
        return None
    cont = _pos_int(raw.get("max_continuous_drive_min"))
    brk = _pos_int(raw.get("required_break_min"))
    daily = _pos_int(raw.get("max_daily_drive_min"))
    if cont is None and daily is None:
        return None
    spec: dict[str, Any] = {}
    if cont is not None:
        spec["max_continuous_drive_min"] = cont
        # A continuous-drive limit needs SOME break to reset it; default 1 min if unstated
        # (the meter still resets on any qualifying stationary span).
        spec["required_break_min"] = brk if brk is not None else 1
    if daily is not None:
        spec["max_daily_drive_min"] = daily
    return spec or None


def _range_obligation(data: dict[str, Any]) -> dict[str, Any] | None:
    """续航/补给义务 (range_obligation). Generic over 油/电/水: {max_range_km,
    stations:[{lat,lng,label?}], service_min}. Every max_range_km of driving must reach one
    of the supply points. Requires a positive range and >=1 valid station coordinate."""
    raw = data.get("range_obligation")
    if not isinstance(raw, dict):
        return None
    max_km = _opt_num(raw.get("max_range_km"))
    if max_km is None or max_km <= 0:
        return None
    stations: list[dict[str, Any]] = []
    for s in (raw.get("stations") or []):
        if not isinstance(s, dict):
            continue
        lat = _opt_num(s.get("lat", s.get("center_lat")))
        lng = _opt_num(s.get("lng", s.get("center_lng")))
        if lat is None or lng is None:
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
            continue
        station: dict[str, Any] = {"lat": lat, "lng": lng}
        label = str(s.get("label", "") or "").strip()
        if label:
            station["label"] = label[:40]
        stations.append(station)
        if len(stations) >= 20:
            break
    if not stations:
        return None
    service = _pos_int(raw.get("service_min"))
    return {"max_range_km": float(max_km), "stations": stations,
            "service_min": service if service is not None else 10}


def _transit_time_limits(data: dict[str, Any]) -> list[dict[str, Any]]:
    """品类时效 (transit_time_limits). [{categories:[...], max_minutes}]: cargo of the listed
    categories must be delivered within max_minutes of acceptance. Empty categories => applies
    to every cargo. Requires a positive max_minutes; malformed entries dropped."""
    items = data.get("transit_time_limits")
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        mins = _pos_int(it.get("max_minutes"))
        if mins is None:
            continue
        out.append({"categories": _str_list(it.get("categories")), "max_minutes": mins})
        if len(out) >= 10:
            break
    return out


_CARGO_ATTRS = {"cargo_value", "cost_time_minutes", "truck_length", "haul_km", "net_yield"}
_CARGO_ATTR_OPS = {">", "<", ">=", "<=", "=="}


def _daypart_hours(spec: dict[str, Any]) -> tuple[int, int] | None:
    """Resolve an optional daypart on a cargo-attribute filter into [start_hour,
    end_hour). Explicit daypart_start_hour/daypart_end_hour win; else the named
    morning/afternoon binary split of the day. None when no daypart is given."""
    sh = _opt_num(spec.get("daypart_start_hour"))
    eh = _opt_num(spec.get("daypart_end_hour"))
    if sh is not None and eh is not None and 0 <= int(sh) <= 24 and 0 <= int(eh) <= 24:
        return int(sh), int(eh)
    dp = str(spec.get("daypart") or "").strip().lower()
    if dp in ("morning", "am", "上午"):
        return 0, 12
    if dp in ("afternoon", "pm", "下午"):
        return 12, 24
    return None


def _penalty_tiers(data: dict[str, Any]) -> list[float] | None:
    """Graduated per-occurrence amounts from the LLM output. Non-negative numbers
    only; empty/absent -> None (flat penalty)."""
    items = data.get("penalty_tiers")
    if not isinstance(items, list):
        return None
    out: list[float] = []
    for v in items:
        n = _opt_num(v)
        if n is None or n < 0:
            continue
        out.append(n)
        if len(out) >= 12:
            break
    return out or None


def _cargo_attribute_filters(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-order cargo-attribute prohibit predicates. Numeric attributes need
    {attribute, op, value}; truck_length needs {attribute, value(text)}. Out-of-vocab
    or malformed entries are dropped (the compiler validates again)."""
    items = data.get("cargo_attribute_filters")
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        attr = str(it.get("attribute") or "").strip()
        if attr not in _CARGO_ATTRS:
            continue
        spec: dict[str, Any] = {"attribute": attr}
        if attr == "truck_length":
            text = str(it.get("value") if it.get("value") is not None else it.get("text") or "").strip()
            if not text:
                continue
            spec["text"] = text
        else:
            op = str(it.get("op") or "").strip()
            value = _opt_num(it.get("value"))
            if op not in _CARGO_ATTR_OPS or value is None:
                continue
            spec["op"] = op
            spec["value"] = value
        if it.get("negate"):
            spec["negate"] = True
        daypart = _daypart_hours(it)
        if daypart is not None:
            spec["daypart_start_hour"], spec["daypart_end_hour"] = daypart
        out.append(spec)
        if len(out) >= 10:
            break
    return out


_SEQ_PRED_ATTRS = {"haul_km", "cargo_value", "cost_time_minutes"}
_SEQ_RELATIONS = {"adjacency_implication", "adjacency_distinct", "window_quota", "max_consecutive_same"}
_SEQ_DISTINCT_KEYS = {"category", "pickup_region", "dropoff_region"}


def _seq_predicate(spec: Any) -> dict[str, Any] | None:
    """Normalize a per-order predicate for a sequence relation: {attribute(haul_km/
    cargo_value/cost_time_minutes), op, value} | {category:<t>} | {region:<t>}, with an
    optional negate. Out-of-vocab/malformed → None (the compiler validates again)."""
    if not isinstance(spec, dict):
        return None
    negate = bool(spec.get("negate"))
    if spec.get("category") is not None:
        text = str(spec.get("category")).strip()
        return {"category": text, **({"negate": True} if negate else {})} if text else None
    if spec.get("region") is not None:
        text = str(spec.get("region")).strip()
        return {"region": text, **({"negate": True} if negate else {})} if text else None
    attr = str(spec.get("attribute") or "").strip()
    op = str(spec.get("op") or "").strip()
    value = _opt_num(spec.get("value"))
    if attr not in _SEQ_PRED_ATTRS or op not in _CARGO_ATTR_OPS or value is None:
        return None
    out = {"attribute": attr, "op": op, "value": value}
    if negate:
        out["negate"] = True
    return out


def _sequence_constraints(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract relations over the ordered accepted-order stream. Validates the closed
    relation/key vocabulary and the per-relation required fields; malformed items are
    dropped (the compiler re-validates)."""
    items = data.get("sequence_constraints")
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        relation = str(it.get("relation") or "").strip()
        if relation not in _SEQ_RELATIONS:
            continue
        spec: dict[str, Any] = {"relation": relation}
        if relation == "adjacency_implication":
            ant = _seq_predicate(it.get("antecedent"))
            con = _seq_predicate(it.get("consequent"))
            if ant is None or con is None:
                continue
            spec["antecedent"], spec["consequent"] = ant, con
        elif relation == "adjacency_distinct":
            key = str(it.get("distinct_key") or "category").strip()
            if key not in _SEQ_DISTINCT_KEYS:
                continue
            spec["distinct_key"] = key
        elif relation == "max_consecutive_same":
            # 同一 distinct_key 值（默认品类）连续出现不得超过 max_run 单；可选 category 谓词把
            # "连续"限定到某个具体品类（建材最多连接2单），非该品类的单子打断连续段。
            key = str(it.get("distinct_key") or "category").strip()
            if key not in _SEQ_DISTINCT_KEYS:
                continue
            max_run = _pos_int(it.get("max_run"))
            if max_run is None:
                continue
            spec["distinct_key"] = key
            spec["max_run"] = max_run
            cat = it.get("category")
            if cat is not None and str(cat).strip():
                spec["category"] = str(cat).strip()
        else:  # window_quota
            ant = _seq_predicate(it.get("antecedent") or it.get("predicate"))
            n = _opt_num(it.get("window_n"))
            cmp = str(it.get("comparator") or "").strip()
            value = _opt_num(it.get("value"))
            if ant is None or n is None or int(n) <= 0 or cmp not in _CARGO_ATTR_OPS or value is None:
                continue
            spec["antecedent"], spec["window_n"], spec["comparator"], spec["value"] = ant, int(n), cmp, value
        if str(it.get("penalty_fn") or "") == "all_or_nothing":
            spec["penalty_fn"] = "all_or_nothing"
        out.append(spec)
        if len(out) >= 10:
            break
    return out


def _activation_guard(data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract an activation_guard from the LLM output. Leaf {metric, op, value} or a
    compound {"all":[...]} / {"any":[...]} / {"not": guard}. Structure is normalized
    recursively here; the compiler validates the metric/op vocabulary on leaves."""
    return _normalize_guard_spec(data.get("activation_guard"))


def _normalize_guard_spec(guard: Any) -> dict[str, Any] | None:
    if not isinstance(guard, dict):
        return None
    for key in ("all", "any"):
        if isinstance(guard.get(key), list):
            subs = [s for s in (_normalize_guard_spec(x) for x in guard[key]) if s is not None]
            return {key: subs} if subs else None
    if "not" in guard:
        inner = _normalize_guard_spec(guard.get("not"))
        return {"not": inner} if inner is not None else None
    metric = str(guard.get("metric") or "").strip()
    op = str(guard.get("op") or "").strip()
    value = _opt_num(guard.get("value"))
    if not metric or not op or value is None:
        return None
    return {"metric": metric, "op": op, "value": value}


def _str_list(value: Any, limit: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for v in value:
        s = str(v).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
            if len(out) >= limit:
                break
    return out


def _valid_rest_type(value: Any) -> str | None:
    return value if value in _VALID_REST_TYPES else None


def _excluded_categories(pref: dict[str, Any], parsed_data: dict[str, Any]) -> list[str]:
    explicit = _str_list(parsed_data.get("excluded_categories"))
    if explicit:
        return explicit
    text = str(pref.get("content", ""))
    if not any(token in text for token in ("不接", "推掉", "干不了", "不想接", "不要", "一律", "扣钱", "扣", "罚")):
        return []
    candidates: list[str] = []
    patterns = (
        r"凡是\s*([\u4e00-\u9fff]{2,12})\s*(?:货源|货|订单|活儿)",
        r"([\u4e00-\u9fff]{2,12})\s*(?:这类|这一类|类)\s*(?:活儿|货|订单).*?(?:干不了|不接|推掉|扣钱|扣|罚)",
        r"(?:货源品类|品类)\s*(?:为|是|叫)?\s*「?([\u4e00-\u9fff]{2,12})」?",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            name = str(match.group(1)).strip(" ，,。；;：:「」")
            if not name:
                continue
            # If the phrase contains concrete examples before the actual
            # category, keep the tail after punctuation/conjunctions.
            for sep in ("、", "，", ",", "和", "及"):
                if sep in name:
                    name = name.split(sep)[-1].strip()
            if name and name not in candidates:
                candidates.append(name)
    return candidates


def effective_penalty_amount(pref: ParsedPreference, amount: float | None = None) -> float:
    base = float(pref.penalty_amount if amount is None else amount)
    return max(0.0, base)


def _pos_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _penalty_cap(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pos_int(value: Any) -> int | None:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _hour(value: Any) -> int | None:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if 0 <= v <= 23 else None


# 中文数字 -> int(覆盖小时表达 0-24)。通用时点解析用,不针对特定钟点硬编码。
_CN_NUM = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
    "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
    "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20, "二十一": 21, "二十二": 22,
    "二十三": 23, "二十四": 24,
}
_DAYPART_PM = ("下午", "午后", "晚上", "傍晚", "夜里", "夜晚")
_CLOCK_RE = re.compile(
    r"(上午|下午|中午|午后|早上|早晨|凌晨|清晨|晚上|傍晚|夜里|夜晚)?\s*"
    r"(\d{1,2}|[零一二两三四五六七八九十]{1,3})\s*[点时]"
)


def _parse_clock_hour(text: str) -> int | None:
    """从中文文本抽一个"时点(小时,24h 制)":识别**阿拉伯或中文**数字 + 时段(上午/下午/中午/
    晚上/凌晨…)并归一化。通用解析,不针对特定钟点(如 12)硬编码;找不到返回 None。
    例:"上午10点"->10、"下午两点"->14、"中午"->12、"晚上8点"->20、"二十三点"->23。"""
    t = text or ""
    m = _CLOCK_RE.search(t)
    if not m:
        return 12 if "中午" in t else None
    daypart, num = m.group(1), m.group(2)
    h = int(num) if num.isdigit() else _CN_NUM.get(num)
    if h is None or not (0 <= h <= 24):
        return None
    if h == 24:
        h = 0
    if daypart == "中午":
        return 12
    if daypart in _DAYPART_PM:
        if h == 12:
            # "晚上12点/夜里12点" = 午夜(0点);但"下午12点/午后12点" = 正午(12点)。
            h = 12 if daypart in ("下午", "午后") else 0
        elif h < 12:
            h += 12
    return h if 0 <= h <= 23 else None


# 连续休息时长(小时):LLM 常把"6个半小时"抽成 6.0(丢"半"),extra_drivers2-D001 因此每天少休 30min
# → 31 天休息违规。确定性兜底:从原文识别 X.5 / X个半小时 / X小时半,优先于 LLM 的取整值。
# 通用(中文/阿拉伯数字、个半/小时半/.5),不针对特定司机或时长硬编码。
_REST_DEC_RE = re.compile(r"(\d+\.5)\s*个?\s*小?时")
_REST_HALF_RE = re.compile(r"(\d+|[零一二两三四五六七八九十]{1,3})\s*个?\s*半\s*个?\s*小?时")
_REST_HALF2_RE = re.compile(r"(\d+|[零一二两三四五六七八九十]{1,3})\s*个?\s*小时\s*半")


def _continuous_rest_hours_from_text(text: str) -> float | None:
    """显式半小时的连续休息时长(小时),否则 None。"""
    t = text or ""
    m = _REST_DEC_RE.search(t)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    for rgx in (_REST_HALF_RE, _REST_HALF2_RE):
        m = rgx.search(t)
        if m:
            s = m.group(1)
            n = int(s) if s.isdigit() else _CN_NUM.get(s)
            if n is not None and 0 < n <= 24:
                return float(n) + 0.5
    return None


def _rest_continuous_hours(pref: dict, data: dict) -> float | None:
    """连续休息小时数:原文若写明半小时(X个半/X小时半/X.5)以确定性值为准(LLM 会丢"半"),
    否则用 LLM 抽的整数小时(整点它一般抽对)。"""
    det = _continuous_rest_hours_from_text(str(pref.get("content", "") or ""))
    if det is not None:
        return det
    return _pos_float(data.get("rest_continuous_hours"))


_COORD_PATTERN = re.compile(
    r"[（(]\s*(-?\d+(?:\.\d+)?)\s*[，,]\s*(-?\d+(?:\.\d+)?)\s*[）)]"
)
_TIME_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")
_CN_DATE_PATTERN = re.compile(
    r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*(\d{1,2})\s*[:：]\s*(\d{1,2})"
    r"(?:\s*[:：]\s*(\d{1,2}))?"
)
_CN_MONTH_DAY_PATTERN = re.compile(
    r"(?P<month>\d{1,2}|[一二两三四五六七八九十]{1,4})\s*月\s*"
    r"(?P<day>\d{1,2}|[一二两三四五六七八九十]{1,4})\s*(?:日|号)"
)
_CN_TIME_OF_DAY_PATTERN = re.compile(
    r"(?P<part>凌晨|早上|上午|中午|下午|傍晚|晚上|晚间)?\s*"
    r"(?P<hour>\d{1,2}|[一二两三四五六七八九十]{1,4})\s*点"
    r"(?:(?P<minute>\d{1,2}|[一二两三四五六七八九十]{1,4})\s*分?)?"
)
_CARGO_ID_REFERENCE_PATTERN = re.compile(
    r"(?:编号|编码|cargo[_ ]?id|货源|熟货源?)\s*[:：]?\s*(\d{2,})",
    re.IGNORECASE,
)
_LOCATION_NAME_PATTERN = re.compile(
    r"[\u4e00-\u9fff]{1,12}(?:老档口|档口|县城|城区|区|县|市|镇|街道|仓库|仓|工厂|园区|市场|工地|码头)"
)
_COMMITMENT_OBLIGATION_TOKENS = (
    "必须",
    "须",
    "务必",
    "需要",
    "得",
    "要",
    "要在",
    "不得晚于",
    "不晚于",
    "之前",
    "前",
    "完成",
    "到达",
    "抵达",
    "接上",
    "接到",
    "返回",
    "停留",
    "静止",
    "待到",
    "指定",
    "预留",
    "承诺",
)
_ITINERARY_SEQUENCE_TOKENS = (
    "先到",
    "先去",
    "先抵达",
    "再到",
    "再去",
    "然后",
    "返回",
    "回到",
    "回家",
    "接上",
    "接到",
    "送到",
    "进家门",
)
_ITINERARY_STAY_TOKENS = ("停留", "静止", "待到", "留在", "原处", "不得离开", "停一趟", "花")
_REQUIRED_CARGO_TOKENS = ("指定", "必须", "须", "需要", "要接", "接", "熟货", "预留", "上架", "装货地", "装货点")

# 守候到 <钟点> (continuous-presence-until). Keyed on the 待/守/候/留/呆 + 到 + 点 verbs so an arrival
# "9点前到" or 接到/开到/送到 never matches; "停满4小时" (flexible dwell) has no clock here either.
_STAY_UNTIL_CLOCK_RE = re.compile(
    r"(?:待|守|候|留|呆)\s*到\s*"
    r"(凌晨|早上|上午|中午|下午|傍晚|晚上|晚间)?\s*"
    r"(\d{1,2}|[一二两三四五六七八九十]{1,4})\s*点\s*(半)?"
)


def _stay_until_clock_min(text: str) -> int | None:
    """Minutes-of-day for an explicit 守候/待到 <钟点> phrase (e.g. '一直在酒店待到下午3点' -> 900).
    Returns None when no such phrase exists — '停满4小时' is a flexible dwell, NOT a stay-until, and
    must never match. Period words shift PM hours (下午/傍晚/晚上/晚间 +12, 中午 -> 12); 点半 adds 30."""
    m = _STAY_UNTIL_CLOCK_RE.search(str(text or ""))
    if not m:
        return None
    hour = _cn_num_to_int(m.group(2))
    if hour is None or not 0 <= hour <= 24:
        return None
    period = m.group(1) or ""
    if period in ("下午", "傍晚", "晚上", "晚间") and hour < 12:
        hour += 12
    elif period == "中午" and hour < 12:
        hour = 12
    minute = 30 if m.group(3) else 0
    total = hour * 60 + minute
    if total <= 0 or total >= MINUTES_PER_DAY:
        return None
    return total


def _pin_stay_until_from_text(events: list[dict[str, Any]], content: str) -> None:
    """Deterministic stay-until recovery for the LLM main path. When a multi-stop itinerary ends in
    a 守候 ('接到人 → 开到酒店 → 一直待到下午3点'), the LLM sometimes encodes that final guard as a
    ``stay`` carrying ``must_complete_before`` but NO ``until`` — which makes the completion override
    treat it as a flexible dwell (gated, deferred) rather than a stay-until (entered the moment the
    prior stop is done). That gap lets the model insert a multi-hour order between the prior stop and
    the guard and reach it too late (ds2-D001 3/28 接亲: 接到人 08:45 后偷接一单, 14:45 才到酒店, 错过
    13:00 到场判定). If the text carries an explicit 守候/待到 <钟点>, pin ``until_min`` onto the FINAL stop
    (and normalize its type to ``stay`` — a 守候-until is a stay no matter how the LLM typed it) so
    ``stay_active`` fires immediately after the prior stop. Only the last stop is eligible — the guard is
    always the itinerary's end ('待到下午3点把客人送走') — so a leading stop's own arrival deadline still
    defers the chain (no midnight over-pin). Single-stop stays (家长会/年检) and dwell-N guards
    (盘库'停满2h'/春运'停4h', no 待到钟点) are untouched: len<2, or the regex (待/守/候/留/呆+到+点) just
    does not match."""
    if len(events) < 2:
        return
    clock = _stay_until_clock_min(content)
    if clock is None:
        return
    ev = events[-1]
    if ev.get("until_min") is not None:
        return
    base = ev.get("must_complete_before_min")
    if base is None:
        base = ev.get("not_before_min")
    if base is None:
        return
    day_start = (int(base) // MINUTES_PER_DAY) * MINUTES_PER_DAY
    until = day_start + int(clock)
    nb = ev.get("not_before_min")
    if nb is not None and until <= int(nb):
        return
    ev["until_min"] = until
    ev["type"] = "stay"


def _cn_num_to_int(text: str) -> int | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    raw = raw.replace("两", "二")
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if raw == "十":
        return 10
    if "十" in raw:
        left, _, right = raw.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    if len(raw) == 1:
        return digits.get(raw)
    return None


def _date_context_from_text_or_pref(text: str, pref: dict[str, Any] | None = None) -> tuple[int, int] | None:
    m = _CN_MONTH_DAY_PATTERN.search(text)
    if m:
        month = _cn_num_to_int(m.group("month"))
        day = _cn_num_to_int(m.group("day"))
        if month and day:
            return month, day
    if pref is not None:
        for key in ("start_time", "end_time"):
            dt_min = _wall_to_sim_min(str(pref.get(key, "")))
            if dt_min is None:
                continue
            dt = _SIM_EPOCH + timedelta(minutes=int(dt_min))
            return dt.month, dt.day
    return None


def _time_of_day_from_text(text: str) -> tuple[int, int] | None:
    matches = list(_CN_TIME_OF_DAY_PATTERN.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    hour = _cn_num_to_int(m.group("hour"))
    minute = _cn_num_to_int(m.group("minute") or "0")
    if hour is None or minute is None:
        return None
    part = m.group("part") or ""
    if part in {"下午", "傍晚", "晚上", "晚间"} and hour < 12:
        hour += 12
    elif part == "中午" and hour < 11:
        hour += 12
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _clean_location_name(name: str) -> str:
    text = str(name or "").strip(" ，,。；;：:")
    text = re.sub(r"^.*(?:先过|先到|先去|再到|再去|赶到|到|去|过|返回|回到|抵达|到达)", "", text)
    text = re.sub(
        r"^(?:凌晨|早上|上午|中午|下午|傍晚|晚上|晚间)?\s*"
        r"(?:\d{1,2}|[一二两三四五六七八九十]{1,4})\s*点(?:前)?",
        "",
        text,
    )
    text = re.sub(r"^(?:当天|上午|下午|中午|晚上|凌晨|早上|得|要|先|再|过|到|去|赶到|返回|回到)+", "", text)
    text = re.sub(r"^(?:\d{1,2}|[一二两三四五六七八九十]{1,4})月(?:\d{1,2}|[一二两三四五六七八九十]{1,4})(?:日|号)", "", text)
    return text.strip(" ，,。；;：:")


def _extract_location_names(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in _LOCATION_NAME_PATTERN.finditer(str(text or "")):
        name = _clean_location_name(m.group(0))
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


# Avoidance markers: a place named right AFTER one of these is somewhere the driver
# must STAY AWAY from (禁入 / 回避), NOT a place to go. Used to stop "不往深圳跑" from
# being mis-compiled into a "去深圳赴约" itinerary commitment (the observed inversion that
# made the model reposition INTO the forbidden region).
_AVOID_MARKERS = (
    "不往", "不进", "不去", "别去", "别往", "避开", "绕开", "绕过", "不出现在",
    "不到", "不跑", "别跑", "不能去", "不要去", "别给我派", "不接",
)


def _text_avoids_location(text: str, location_name: str) -> bool:
    """True if the raw text tells the driver to AVOID this location (不往/不进/避开 X …),
    so it must NOT be turned into a 'go there' itinerary commitment. Matches the event's
    location CORE (admin suffixes stripped, e.g. 深圳市 -> 深圳) against an avoidance marker
    that precedes it within a few characters (handles '不往深圳跑'). Does not rely on the
    location-name regex, which misses bare city names like '深圳'."""
    loc = _clean_location_name(str(location_name or ""))
    core = re.sub(r"(?:市|区|县|镇|村|街道|老档口|档口|那边|那儿|那里)$", "", loc) or loc
    if len(core) < 2:
        return False
    s = str(text or "")
    for marker in _AVOID_MARKERS:
        if re.search(re.escape(marker) + r"[^，。；！？、\s]{0,6}?" + re.escape(core), s):
            return True
    return False


def _has_specific_datetime(text: str) -> bool:
    if _TIME_PATTERN.search(text) or _CN_DATE_PATTERN.search(text):
        return True
    return bool(_CN_MONTH_DAY_PATTERN.search(text))


def _has_short_active_window(pref: dict[str, Any]) -> bool:
    start = _wall_to_sim_min(str(pref.get("start_time", "")))
    end = _wall_to_sim_min(str(pref.get("end_time", "")))
    if start is None or end is None or end <= start:
        return False
    return (end - start) <= 14 * 1440


def _has_commitment_obligation(text: str) -> bool:
    return any(token in text for token in _COMMITMENT_OBLIGATION_TOKENS)


def _looks_like_required_cargo_commitment(text: str) -> bool:
    if not _CARGO_ID_REFERENCE_PATTERN.search(text):
        return False
    return any(token in text for token in _REQUIRED_CARGO_TOKENS)


def _forbidden_endpoint_locations(pref: dict[str, Any], parsed_data: dict[str, Any]) -> list[str]:
    # A daily_location("每天X点前回到[坐标]N公里内")命名的是「回家目标点」,不是要避开的禁接地;
    # 同句的宵禁子句("23点至次日8点不接单")是休息窗,其中的"不接"绝不能派生出一个 forbidden
    # endpoint——否则 agent 会把家当禁区绕开,与"回家"自相矛盾。故凡解析为 daily_location 的偏好
    # (deadline 已识别),一律不从其文本/LLM 抽禁接端点。通用,不针对具体坐标/司机。
    if _daily_location_deadline_hour(pref, parsed_data) is not None:
        return []
    explicit = _str_list(
        parsed_data.get("forbidden_endpoint_locations")
        or parsed_data.get("excluded_endpoint_locations")
        or parsed_data.get("forbidden_cargo_locations")
    )
    # Pure LLM source only. The old raw-text regex fallback (在/到/去/进 + 任意2-8汉字,
    # triggered by 扣/罚/不接/不要…) was a net-negative straggler from the "removed hardcoded
    # raw-text fallbacks" cleanup: it fired on 休息窗("到第二天早上6点")、空驶预算("每公里扣点钱")、
    # 净收益("逐单罚")、毛收入("每少一元扣") 等含 扣/罚/不接 的非端点偏好, 贪婪地把句子碎片当地名
    # (实测 39 处幻灵端点、横扫 20 类, 多为"第二天早上"这类碎片或极性反了的允许区/打卡目标)。
    # forbidden_endpoint 现完全交给解析器 LLM(prompt §1b 已明确指示), 合法的
    # forbidden_region_cargo 全部由 LLM 干净抽出(惠州/肇庆/清远…), 不依赖正则。
    if not explicit:
        return []
    # Drop generic endpoint nouns that are not real place names.
    blocked = {"装货地", "装货点", "卸货地", "卸货点", "起运地", "目的地"}
    out: list[str] = []
    for x in explicit:
        name = _clean_location_name(x)
        if name and name not in blocked and name not in out:
            out.append(name)
    return out


def _required_endpoint_locations(pref: dict[str, Any], parsed_data: dict[str, Any]) -> list[str]:
    explicit = _str_list(
        parsed_data.get("required_endpoint_locations")
        or parsed_data.get("required_cargo_endpoint_locations")
        or parsed_data.get("target_endpoint_locations")
    )
    if explicit:
        return [_clean_location_name(x) for x in explicit if _clean_location_name(x)]
    text = str(pref.get("content", ""))
    if not any(tok in text for tok in ("装货", "卸货", "起运", "目的地")):
        return []
    if not any(tok in text for tok in ("起码", "至少", "接够", "够", "不同的日子", "不同自然日")):
        return []
    names: list[str] = []
    for pattern in (
        r"(?:装货|卸货|起运|目的地).*?在\s*([\u4e00-\u9fff]{2,8})",
        r"在\s*([\u4e00-\u9fff]{2,8})\s*的货.*?(?:起码|至少|接够)",
    ):
        for match in re.finditer(pattern, text):
            name = _clean_location_name(match.group(1))
            if name and name not in names:
                names.append(name)
    return names


def _required_endpoint_location_days(pref: dict[str, Any], parsed_data: dict[str, Any]) -> int | None:
    explicit = _pos_int(
        parsed_data.get("required_endpoint_location_days")
        or parsed_data.get("target_endpoint_location_days")
        or parsed_data.get("required_endpoint_days")
    )
    if explicit is not None:
        return explicit
    text = str(pref.get("content", ""))
    if not _required_endpoint_locations(pref, parsed_data):
        return None
    patterns = (
        r"(?:起码|至少)?\s*(?:得|要)?\s*(?:接够)?\s*(\d+|[一二两三四五六七八九十]{1,4})\s*个?不同",
        r"(\d+|[一二两三四五六七八九十]{1,4})\s*个?不同(?:的)?(?:日子|自然日|天)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = _cn_num_to_int(match.group(1))
        if value and value > 0:
            return value
    return None


def _looks_like_itinerary_commitment(text: str, pref: dict[str, Any] | None = None) -> bool:
    coord_count = len(_COORD_PATTERN.findall(text))
    names = _extract_location_names(text)
    target_count = coord_count + len(names)
    if target_count < 1:
        return False
    # Pure avoidance ("不往深圳跑") with no explicit coords is NOT a go-there commitment —
    # it's a forbidden region, so don't let the fallback build an itinerary for it.
    if coord_count == 0 and names and all(_text_avoids_location(text, n) for n in names):
        return False
    if not (_has_specific_datetime(text) or (pref is not None and _has_short_active_window(pref))):
        return False
    if not _has_commitment_obligation(text):
        return False
    sequence_score = sum(1 for token in _ITINERARY_SEQUENCE_TOKENS if token in text)
    has_stay = any(token in text for token in _ITINERARY_STAY_TOKENS)
    if coord_count < 2:
        return has_stay or sequence_score >= 1
    return sequence_score >= 1 or has_stay


def _normalize_time_text(text: str, default_date: tuple[int, int] | None = None) -> str | None:
    """Convert any matched time literal (ISO or Chinese-style) to ISO format."""
    m_iso = _TIME_PATTERN.search(text)
    if m_iso:
        return m_iso.group(0)
    m_cn = _CN_DATE_PATTERN.search(text)
    if m_cn:
        year, month, day, hour, minute = (int(m_cn.group(i)) for i in range(1, 6))
        second = int(m_cn.group(6)) if m_cn.group(6) else 0
        return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
    date = _date_context_from_text_or_pref(text)
    time_of_day = _time_of_day_from_text(text)
    if date is None:
        date = default_date
    if date is not None and time_of_day is not None:
        month, day = date
        hour, minute = time_of_day
        return f"2026-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:00"
    if date is not None and _CN_MONTH_DAY_PATTERN.search(text):
        month, day = date
        return f"2026-{month:02d}-{day:02d} 00:00:00"
    return None


def _required_cargo_pickup(pref: dict[str, Any], parsed_data: dict[str, Any]) -> dict[str, Any] | None:
    """Pickup lat/lng for the required cargo. Prefer parser output, else regex
    scan ``装货地：...（lat，lng）`` style text."""
    explicit = _valid_circle({**(parsed_data.get("required_cargo_pickup") or {}), "radius_km": 0.001}) \
        if isinstance(parsed_data.get("required_cargo_pickup"), dict) else None
    if explicit is not None:
        return {"lat": explicit["center_lat"], "lng": explicit["center_lng"]}
    raw = parsed_data.get("required_cargo_pickup")
    if isinstance(raw, dict):
        try:
            return {"lat": float(raw["lat"]), "lng": float(raw["lng"])}
        except (KeyError, TypeError, ValueError):
            pass
    text = str(pref.get("content", ""))
    if not any(tok in text for tok in ("装货地", "装货点", "起运地")):
        return None
    match = _COORD_PATTERN.search(text)
    if not match:
        return None
    try:
        return {"lat": float(match.group(1)), "lng": float(match.group(2))}
    except ValueError:
        return None


def _required_cargo_release_time_min(pref: dict[str, Any], parsed_data: dict[str, Any]) -> int | None:
    """Release time in sim minutes. Looks for parser field then '上架时间' text."""
    parser_value = parsed_data.get("required_cargo_release_time")
    if isinstance(parser_value, str):
        m = _wall_to_sim_min(parser_value)
        if m is not None:
            return m
    text = str(pref.get("content", ""))
    if "上架时间" in text or "上架于" in text or "释放时间" in text:
        match = _TIME_PATTERN.search(text)
        if match:
            m = _wall_to_sim_min(match.group(0))
            if m is not None:
                return m
    return _wall_to_sim_min(str(pref.get("start_time", "")))


# 指定/熟货源是"预留的热货"。这个 deadline 是 **未观测到货时的兜底窗**(决定 __pickup_v_ 虚拟单还
# 引导司机去装货点多久)——按用户口径不超过上架 + 1h。**一旦该货被观测到(进了货图),guard ② 让真货
# 按它自己的真实 load_time 窗口被接走**,这个兜底就不再起作用。兜底取窄(1h)= 别为一个迟迟不出现/已
# 撤的货(如旧 240646)空耗太久;真货只要露面就走真实窗,不受此 1h 限制。副作用(正是要的):只有上架
# 时间、没写截止的偏好也得到一个确定 deadline → 第③道闸(无 deadline 不注入)能正常触发,不再 dormant。
_REQUIRED_CARGO_GRAB_WINDOW_MIN = 60


def _required_cargo_deadline_time_min(pref: dict[str, Any], parsed_data: dict[str, Any]) -> int | None:
    """FALLBACK pickup deadline (used only while the cargo is NOT yet observed) = min(stated, release
    + 1h). stated = LLM ``required_cargo_deadline_time`` else the pref's ``end_time``; release =
    ``_required_cargo_release_time_min``. With a known release the deadline is ALWAYS set (capped at
    release+1h), so a release-only pref no longer yields deadline=None. Falls back to the bare stated
    value when there is no release to anchor the cap. Once the real cargo is observed, the pickup
    virtual is suppressed (guard ②) and the real cargo is taken on its OWN load_time window — this
    cap only bounds how long an unobserved/phantom cargo is chased."""
    stated = None
    parser_value = parsed_data.get("required_cargo_deadline_time")
    if isinstance(parser_value, str):
        stated = _wall_to_sim_min(parser_value)
    if stated is None:
        stated = _wall_to_sim_min(str(pref.get("end_time", "")))
    release = _required_cargo_release_time_min(pref, parsed_data)
    if release is not None:
        cap = int(release) + _REQUIRED_CARGO_GRAB_WINDOW_MIN
        return min(int(stated), cap) if stated is not None else cap
    return stated


def _itinerary_commitment(pref: dict[str, Any], parsed_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile itinerary_commitment events from parser output, with a regex
    fallback for the time-bound itinerary pattern (visit + stay events with
    coordinates and deadlines)."""
    raw_events = parsed_data.get("itinerary_commitment") if isinstance(parsed_data, dict) else None
    out: list[dict[str, Any]] = []
    not_before_min = _wall_to_sim_min(str(pref.get("start_time", "")))
    # Drop any event whose location is one the driver must AVOID (e.g. "不往深圳跑"): an
    # avoidance preference must NEVER become a "go there" commitment — that inversion made
    # the model reposition INTO the forbidden region. The forbidden_region / endpoint
    # fields (also parsed) still carry the real avoidance constraint.
    _content = str(pref.get("content", ""))
    if isinstance(raw_events, list):
        for ev in raw_events:
            normalized = _normalize_itinerary_event(ev)
            if normalized is not None:
                if _text_avoids_location(_content, normalized.get("location_name")):
                    continue
                if normalized.get("not_before_min") is None and not_before_min is not None:
                    normalized["not_before_min"] = not_before_min
                _clamp_event_not_before_to_event_day(normalized)
                out.append(normalized)
        if out:
            _pin_stay_until_from_text(out, _content)
            return out
    return _itinerary_commitment_fallback(pref)


def _clamp_event_not_before_to_event_day(ev: dict[str, Any]) -> None:
    """An itinerary event must not be allowed to START before its own calendar
    day. A preference's ``start_time`` is when the driver LEARNS of the
    commitment (the sim reveals "三月三十一号舅公做寿" a couple of days early), NOT
    when the event happens — yet a per-event ``not_before`` is otherwise
    inherited from it. Left alone, the 寿宴 (deadline 3/31 noon) gets
    not_before=3/29 and the driver executes the whole itinerary a day or two
    early; the scorer then finds the banquet-day stops empty and fails the rule.

    Floor the not_before at the start of the event's own day, derived from its
    deadline (must_complete_before). Events with no deadline (e.g. a multi-day
    ``stay`` keyed only by ``until``) are left untouched: a stay may legitimately
    begin the evening before, and it is gated by ordering/start_after anyway."""
    deadline = ev.get("must_complete_before_min")
    if deadline is None:
        return
    day_start = (int(deadline) // MINUTES_PER_DAY) * MINUTES_PER_DAY
    nb = ev.get("not_before_min")
    if nb is None or int(nb) < day_start:
        ev["not_before_min"] = day_start
    elif int(nb) >= int(deadline):
        # A not_before at/after the event's OWN deadline is self-contradictory (the
        # driver can't both start after it and finish by it). It is almost always a
        # mis-parse — "中午十二点前赶到四会" read as not_before=12:00 instead of a
        # 12:00 arrival deadline. Floor it back to the event day so the completion
        # override can still depart in time instead of waiting past the deadline.
        ev["not_before_min"] = day_start


def _normalize_itinerary_event(ev: Any) -> dict[str, Any] | None:
    if not isinstance(ev, dict):
        return None
    location_name = str(ev.get("location_name") or ev.get("location") or ev.get("place_name") or "").strip()
    lat: float | None = None
    lng: float | None = None
    try:
        lat = float(ev["lat"])
        lng = float(ev["lng"])
    except (KeyError, TypeError, ValueError):
        if not location_name:
            return None
    event_type = str(ev.get("type", "visit")).strip().lower()
    if event_type not in {"visit", "stay"}:
        event_type = "visit"
    event_id = str(ev.get("event_id", "")).strip()
    if not event_id:
        event_id = f"event_{lat:.4f}_{lng:.4f}" if lat is not None and lng is not None else f"event_{_slug_location_name(location_name)}"
    dwell_min = _optional_pos_int(ev.get("dwell_min"))
    must_before_min: int | None = None
    if isinstance(ev.get("must_complete_before"), str):
        must_before_min = _wall_to_sim_min(ev["must_complete_before"])
    elif isinstance(ev.get("must_complete_before_min"), (int, float)):
        must_before_min = int(ev["must_complete_before_min"])
    until_min: int | None = None
    if isinstance(ev.get("until"), str):
        until_min = _wall_to_sim_min(ev["until"])
    elif isinstance(ev.get("until_min"), (int, float)):
        until_min = int(ev["until_min"])
    not_before_min: int | None = None
    if isinstance(ev.get("not_before"), str):
        not_before_min = _wall_to_sim_min(ev["not_before"])
    elif isinstance(ev.get("not_before_min"), (int, float)):
        not_before_min = int(ev["not_before_min"])
    start_after = ev.get("start_after") or ev.get("start_after_event") or ev.get("start_after_event_id")
    normalized = {
        "event_id": event_id,
        "type": event_type,
        "dwell_min": dwell_min,
        "not_before_min": not_before_min,
        "must_complete_before_min": must_before_min,
        "until_min": until_min,
        "start_after_event_id": str(start_after).strip() if isinstance(start_after, str) and start_after.strip() else None,
        "radius_km": 1.0,
    }
    if lat is not None and lng is not None:
        normalized["lat"] = lat
        normalized["lng"] = lng
    if location_name:
        normalized["location_name"] = location_name
    return normalized


def _slug_location_name(name: str) -> str:
    text = re.sub(r"\W+", "_", str(name or "").strip())
    return text.strip("_") or "location"


def _event_target(ev: dict[str, Any]) -> tuple[float, float] | None:
    try:
        return float(ev["lat"]), float(ev["lng"])
    except (KeyError, TypeError, ValueError):
        return None


def _location_aliases(name: str) -> set[str]:
    raw = _clean_location_name(name)
    if not raw:
        return set()
    aliases = {raw}
    compact = raw
    for token in ("老档口", "档口", "县城", "城区", "仓库", "工厂", "园区", "市场", "工地", "码头"):
        compact = compact.replace(token, "")
    aliases.add(compact)
    if compact.endswith(("区", "县", "市", "镇")) and len(compact) > 2:
        aliases.add(compact[:-1])
    # Strip a trailing VENUE / spot word (档口/市场/码头/…). Such a spot name never appears in observed
    # cargo ADDRESSES — they carry the enclosing region (增城区…) — so without this a "增城区档口"
    # itinerary stop matched NOTHING and stayed unresolved (the 寿宴 first stop the driver could never
    # be steered to). Reduce it to the region, then drop a trailing 区/市/县/镇 so "增城区档口" → "增城区"
    # → "增城" matches the 增城-region cargo. General venue list, no per-place hard-coding.
    # Operate on RAW, not ``compact``: the token loop above also strips "城区", over-reducing
    # "增城区档口" → "增城区" → "增" (1 char, filtered out), losing the region entirely. From RAW we
    # peel only the venue word, then a single trailing 区/市/县/镇, so "增城区档口" → "增城区" → "增城".
    for _venue in ("档口", "市场", "批发市场", "码头", "仓库", "园区", "广场", "商城", "档"):
        if raw.endswith(_venue) and len(raw) - len(_venue) >= 2:
            _base = raw[: -len(_venue)]
            aliases.add(_base)
            if _base[-1:] in "区市县镇" and len(_base) > 2:
                aliases.add(_base[:-1])
    return {a for a in aliases if len(a) >= 2}


def _string_values(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_string_values(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(_string_values(v))
    return out


def _endpoint_text_blob(cargo: dict[str, Any], endpoint: str) -> str:
    endpoint_obj = cargo.get(endpoint) if isinstance(cargo.get(endpoint), dict) else {}
    values = _string_values(endpoint_obj)
    key_markers = {
        "start": ("start", "origin", "pickup", "load", "from", "src", "发", "起", "装"),
        "end": ("end", "dest", "drop", "unload", "to", "dst", "收", "卸", "到"),
    }[endpoint]
    for key, value in cargo.items():
        key_text = str(key).lower()
        if any(marker in key_text for marker in key_markers):
            values.extend(_string_values(value))
    return " ".join(values)


def _coord_from_endpoint(cargo: dict[str, Any], endpoint: str) -> tuple[float, float] | None:
    point = cargo.get(endpoint)
    if not isinstance(point, dict):
        return None
    try:
        return float(point["lat"]), float(point["lng"])
    except (KeyError, TypeError, ValueError):
        return None


def _resolve_location_name_from_cargos(name: str, cargos: list[dict[str, Any]]) -> tuple[float, float] | None:
    aliases = _location_aliases(name)
    if not aliases:
        return None
    candidates: list[tuple[float, float]] = []
    for cargo in cargos or []:
        if not isinstance(cargo, dict):
            continue
        for endpoint in ("start", "end"):
            blob = _endpoint_text_blob(cargo, endpoint)
            if blob and any(alias in blob for alias in aliases):
                coord = _coord_from_endpoint(cargo, endpoint)
                if coord is not None:
                    candidates.append(coord)
        if candidates:
            continue
        full_blob = " ".join(_string_values(cargo))
        if full_blob and any(alias in full_blob for alias in aliases):
            coord = _coord_from_endpoint(cargo, "start")
            if coord is not None:
                candidates.append(coord)
    if not candidates:
        return None
    # Use a small centroid so "增城区" style region names become a stable
    # concrete anchor without needing a hard-coded gazetteer.
    limited = candidates[:50]
    return (
        sum(p[0] for p in limited) / len(limited),
        sum(p[1] for p in limited) / len(limited),
    )


_COORD_IN_TEXT_RE = re.compile(
    r"[（(]\s*(-?\d{1,3}(?:\.\d+)?)\s*[，,、\s]+\s*(-?\d{1,3}(?:\.\d+)?)\s*[)）]"
)

# Clause boundaries used to bind a coordinate to the place-name that immediately
# precedes it. One preference may name several places yet give coordinates for
# only some of them (e.g. "先过增城区档口捎寿礼…赶到四会县城（23.32，112.83）"), so a
# coordinate must attach to its own clause's name — NOT to every name in the text.
_CLAUSE_SEP_RE = re.compile(r"[，,。、；;：:！!？?\n（）()]")


def _build_preference_location_registry(
    parsed_prefs: list["ParsedPreference"],
) -> dict[str, tuple[float, float]]:
    """Harvest explicit '<地名>（lat，lng）' coordinates stated in ANY preference's
    raw text, keyed by place-name aliases. Lets a later preference that names the
    same place WITHOUT coordinates (e.g. an itinerary '增城老档口') resolve to the
    driver's own stated coordinate instead of drifting via the cargo graph. The
    coordinates come from the driver's preference text — not a hardcoded table.

    Each coordinate binds only to the place-name in its own clause (the span
    between the preceding clause separator and the '（lat，lng）'). Mapping every
    name in the text to a single coordinate is wrong when a clause like
    '先过增城区档口…赶到四会县城（坐标）' lists two places but only四会 has a coordinate —
    增城区档口 would otherwise be mis-bound to四会's point."""
    registry: dict[str, tuple[float, float]] = {}
    for pref in parsed_prefs:
        text = str(getattr(pref, "raw_content", "") or "")
        endpoint_names = [
            *(getattr(pref, "required_endpoint_locations", None) or []),
            *(getattr(pref, "forbidden_endpoint_locations", None) or []),
        ]
        for m in _COORD_IN_TEXT_RE.finditer(text):
            try:
                lat, lng = float(m.group(1)), float(m.group(2))
            except (TypeError, ValueError):
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
                continue
            coord = (lat, lng)
            # The place this coordinate labels sits in the clause right before it.
            prefix = text[: m.start()]
            seps = list(_CLAUSE_SEP_RE.finditer(prefix))
            clause = prefix[seps[-1].end():] if seps else prefix
            names: list[str] = list(_extract_location_names(clause))
            names.extend(en for en in endpoint_names if en and en in clause)
            for name in names:
                for alias in _location_aliases(name):
                    if len(alias) >= 2:
                        registry.setdefault(alias, coord)
    return registry


def _registry_lookup(
    location_name: str, registry: dict[str, tuple[float, float]]
) -> tuple[float, float] | None:
    if not location_name or not registry:
        return None
    for alias in sorted(_location_aliases(location_name) | {location_name}, key=len, reverse=True):
        if alias in registry:
            return registry[alias]
    for r_alias in sorted(registry, key=len, reverse=True):
        if len(r_alias) >= 2 and r_alias in location_name:
            return registry[r_alias]
    return None


def apply_preference_location_registry(
    parsed_prefs: list["ParsedPreference"], log: logging.Logger | None = None
) -> None:
    """Fill itinerary_commitment events that only carry a location_name with a
    coordinate harvested from a sibling preference's stated '（lat，lng）'."""
    registry = _build_preference_location_registry(parsed_prefs)
    if not registry:
        return
    for pref in parsed_prefs:
        for ev in getattr(pref, "itinerary_commitment", None) or []:
            if not isinstance(ev, dict) or _event_target(ev) is not None:
                continue
            loc = str(ev.get("location_name") or "").strip()
            coord = _registry_lookup(loc, registry)
            if coord is None:
                continue
            ev["lat"], ev["lng"] = coord[0], coord[1]
            ev["resolved_from_location_registry"] = loc
            if log is not None:
                log.info(
                    "[PREF_LOC_REGISTRY] event '%s' -> (%.5f,%.5f) from sibling preference coordinate",
                    loc,
                    coord[0],
                    coord[1],
                )


def _info_by_id_to_cargos(info_by_id: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Cargo-shaped dicts (start/end each with address text + lat/lng) rebuilt from the PERSISTENT
    ``cargo_info_by_id`` (it only grows and survives graph cleanup), so place-name resolution can use
    observations that have since expired out of the live graph."""
    out: list[dict[str, Any]] = []
    for entry in (info_by_id or {}).values():
        if not isinstance(entry, dict):
            continue
        out.append({
            "start": {"address": entry.get("start_address", ""),
                      "lat": entry.get("start_lat"), "lng": entry.get("start_lng")},
            "end": {"address": entry.get("end_address", ""),
                    "lat": entry.get("end_lat"), "lng": entry.get("end_lng")},
        })
    return out


def resolve_itinerary_event_locations(
    events: list[dict[str, Any]], cargos: list[dict[str, Any]],
    info_by_id: dict[str, Any] | None = None,
) -> int:
    """Resolve itinerary events that only have ``location_name`` using observed cargo city/address/
    name fields. Returns the number of events updated. ``info_by_id`` is the PERSISTENT observation
    store (cargo_info_by_id): when the CURRENT graph cannot resolve a name, fall back to it so a place
    the driver visited EARLIER (e.g. the 增城≥4日 visit) still resolves on a morning it is far from
    there — the 寿宴 增城档口 was unresolvable when D002 started 03-31 far away, so the event override
    could not steer toward it in time (四会 reached 13:14 vs the noon deadline). Purely ADDITIVE: it
    only resolves names the live graph could NOT, never changing an already-resolved one — a driver
    with no itinerary events (D001) is untouched."""
    resolved = 0
    persistent: list[dict[str, Any]] | None = None  # lazily built from info_by_id
    for ev in events or []:
        if not isinstance(ev, dict) or _event_target(ev) is not None:
            continue
        name = str(ev.get("location_name") or "").strip()
        if not name:
            continue
        coord = _resolve_location_name_from_cargos(name, cargos)
        if coord is None and info_by_id:
            if persistent is None:
                persistent = _info_by_id_to_cargos(info_by_id)
            coord = _resolve_location_name_from_cargos(name, persistent)
        if coord is None:
            continue
        ev["lat"], ev["lng"] = coord
        ev["resolved_from_location_name"] = name
        resolved += 1
    return resolved


def _optional_pos_int(value: Any) -> int | None:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _itinerary_commitment_fallback(pref: dict[str, Any]) -> list[dict[str, Any]]:
    """Regex fallback for time-bound multi-stop commitments.

    Looks for: 须先到（lat，lng）...（停留不少于N分钟）, 再(返回|到)（lat，lng）,
    须在TIME前进家门, 到家后...至少待到TIME.
    """
    text = str(pref.get("content", ""))
    if not _looks_like_itinerary_commitment(text, pref):
        return []
    coords = _COORD_PATTERN.findall(text)
    names = _extract_location_names(text)
    if len(coords) < 1 and not names:
        return []
    end_time_min = _wall_to_sim_min(str(pref.get("end_time", "")))
    not_before_min = _wall_to_sim_min(str(pref.get("start_time", "")))
    date_context = _date_context_from_text_or_pref(text, pref)
    content_has_day = bool(_CN_MONTH_DAY_PATTERN.search(text))
    day_start_min = None
    day_end_min = None
    if date_context is not None:
        day_start_min = _wall_to_sim_min(f"2026-{date_context[0]:02d}-{date_context[1]:02d} 00:00:00")
        day_end_min = _wall_to_sim_min(f"2026-{date_context[0]:02d}-{date_context[1]:02d} 23:59:00")
    if day_start_min is not None:
        if not_before_min is None or content_has_day:
            not_before_min = max(int(not_before_min or day_start_min), int(day_start_min))
    if day_end_min is not None and ("当天" in text or end_time_min is None):
        end_time_min = day_end_min
    # Look for the *earliest* qualifying deadline mentioned before "前/前进家门/前到家/抵达".
    pickup_deadline_min: int | None = None
    for chunk in re.split(r"前", text)[:-1]:
        normalized = _normalize_time_text(chunk[-40:] if len(chunk) >= 40 else chunk, date_context)
        if normalized is None:
            continue
        candidate = _wall_to_sim_min(normalized)
        if candidate is None:
            continue
        if pickup_deadline_min is None or candidate < pickup_deadline_min:
            pickup_deadline_min = candidate
    if pickup_deadline_min is None:
        # Fall back to the latest time literal anywhere in the text (e.g.
        # "须在 ... 前" patterns where Chinese parsing failed).
        latest_candidate: int | None = None
        for match in list(_CN_DATE_PATTERN.finditer(text)) + list(_CN_TIME_OF_DAY_PATTERN.finditer(text)):
            normalized = _normalize_time_text(match.group(0), date_context)
            ts = _wall_to_sim_min(normalized) if normalized is not None else None
            if ts is not None and (latest_candidate is None or ts < latest_candidate):
                latest_candidate = ts
        pickup_deadline_min = latest_candidate
    dwell_match = re.search(r"停留不少于\s*(\d+)\s*分钟", text)
    dwell_min = int(dwell_match.group(1)) if dwell_match else None
    if dwell_min is None:
        hour_match = re.search(r"(?:花|停留|停)\s*(\d+|[一二两三四五六七八九十]{1,4})\s*小时", text)
        if hour_match:
            hour_value = _cn_num_to_int(hour_match.group(1))
            if hour_value:
                dwell_min = int(hour_value) * 60
    until_min = None
    until_match = re.search(r"(?:到|待到)\s*(凌晨|早上|上午|中午|下午|傍晚|晚上|晚间)?\s*(\d{1,2}|[一二两三四五六七八九十]{1,4})\s*点", text)
    if until_match:
        normalized_until = _normalize_time_text(until_match.group(0), date_context)
        until_min = _wall_to_sim_min(normalized_until) if normalized_until is not None else None
    if end_time_min is None and day_end_min is not None:
        end_time_min = day_end_min
    events: list[dict[str, Any]] = []
    first_event: dict[str, Any] = {
        "event_id": "visit_1",
        "type": "visit",
        "dwell_min": dwell_min if len(coords) < 1 else None,
        "not_before_min": not_before_min,
        "must_complete_before_min": pickup_deadline_min or end_time_min,
        "until_min": None,
        "start_after_event_id": None,
        "radius_km": 1.0,
    }
    if len(coords) >= 2:
        first_event.update({"event_id": "pickup_1", "lat": float(coords[0][0]), "lng": float(coords[0][1]), "dwell_min": dwell_min})
    elif names:
        first_event["location_name"] = names[0]
    elif coords:
        first_event.update({"lat": float(coords[0][0]), "lng": float(coords[0][1])})
    events.append(first_event)
    if len(coords) < 1 or (len(coords) == 1 and not names):
        return events
    dest_coord = coords[1] if len(coords) >= 2 else coords[0]
    home_lat = float(dest_coord[0])
    home_lng = float(dest_coord[1])
    events.append(
        {
            "event_id": "arrive_2",
            "type": "visit",
            "lat": home_lat,
            "lng": home_lng,
            "dwell_min": None,
            "not_before_min": not_before_min,
            "must_complete_before_min": pickup_deadline_min,
            "until_min": None,
            "start_after_event_id": str(first_event["event_id"]),
            "radius_km": 1.0,
        }
    )
    # Stay-at-home event covers the locked period (deadline -> end_time).
    # For a stay, must_complete_before represents "by when must the driver be
    # at the location AND staying" — that is the stay's own end (end_time_min),
    # not the prior visit deadline. The per-minute leaving penalty is handled
    # separately via the raw text + daily_location-style logic.
    stay_until = until_min or end_time_min
    if stay_until is not None and pickup_deadline_min is not None and stay_until > pickup_deadline_min:
        events.append(
            {
                "event_id": "stay_2",
                "type": "stay",
                "lat": home_lat,
                "lng": home_lng,
                "dwell_min": None,
                "not_before_min": pickup_deadline_min,
                "must_complete_before_min": stay_until,
                "until_min": stay_until,
                "start_after_event_id": "arrive_2",
                "radius_km": 1.0,
            }
        )
    return events


def _required_cargo_ids(pref: dict[str, Any], parsed_data: dict[str, Any]) -> list[str]:
    """Extract specific cargo_ids the driver must take.

    例: "指定熟货源编号240646..." → ["240646"]
    """
    explicit = _str_list(parsed_data.get("required_cargo_ids"))
    if explicit:
        return explicit
    text = str(pref.get("content", ""))
    # Trigger only when the original preference clearly references a specific
    # cargo by id/code and contains obligation/cargo-task language. Do not rely
    # on labels such as "临时约定"; new event text may omit them.
    if not _looks_like_required_cargo_commitment(text):
        return []
    matches = _CARGO_ID_REFERENCE_PATTERN.findall(text)
    out: list[str] = []
    seen: set[str] = set()
    for m in matches:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _first_order_deadline_hour(pref: dict[str, Any], parsed_data: dict[str, Any]) -> int | None:
    explicit = _hour(parsed_data.get("first_order_deadline_hour"))
    if explicit is not None:
        return explicit
    text = str(pref.get("content", ""))
    if "首单" not in text and "第一单" not in text:
        return None
    # 通用时点解析(不再硬编码 12):需有截止语义,再抽任意钟点(阿拉伯/中文 + 时段)。
    if not any(t in text for t in ("不晚于", "不得晚于", "之前", "以前", "截止", "开工", "前")):
        return None
    return _parse_clock_hour(text)


def _distance_limit_km(pref: dict[str, Any], parsed_data: dict[str, Any], key: str) -> float | None:
    explicit = _pos_float(parsed_data.get(key))
    if explicit is not None:
        return explicit
    text = str(pref.get("content", ""))
    if key == "max_haul_km":
        if not any(token in text for token in ("装货点至卸货点", "装货点到卸货点", "单笔货", "运输距离", "货运距离")):
            return None
    elif key == "max_pickup_deadhead_km":
        if not any(token in text for token in ("赴装货点空驶", "到装货点空驶", "接单后赴装货点", "空驶距离")):
            return None
    match = re.search(r"(?:不得超过|不超过|不能超过|小于等于|≤|<=)\s*(\d+(?:\.\d+)?)\s*公里", text)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)\s*公里(?:以内|内)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _daily_location_deadline_hour(pref: dict[str, Any], parsed_data: dict[str, Any]) -> int | None:
    explicit = _hour(parsed_data.get("daily_location_deadline_hour"))
    if explicit is not None:
        return explicit
    text = str(pref.get("content", ""))
    if not any(token in text for token in ("每天", "每日")):
        return None
    if not any(token in text for token in ("位置", "自家", "家", "家里", "附近")):
        return None
    # 排除"固定时段静止/睡觉/熄火"类休息偏好:它们也常带钟点 + "家",但语义是"那段时间在某地
    # 静止",不是"某点前必须到定点"。不排除的话会把休息窗起点(如"夜里11点到6点静止")误抽成
    # daily_location 截止(把 11 当成"每天11点前到家")。
    # 但「回家定点」偏好常与「宵禁」写在同一句("每天23点前回到家(坐标)N公里内;23点至8点不接单"),
    # 这类**带到点定位信号**(坐标 / 回到·回家 / 在…N公里内)的,确有"某点前必须到定点"的语义,
    # **不能**因为同句也出现"不接单/停运/静止"就被当成纯休息窗排除掉(否则回家定点彻底丢失)。
    # 仅当**没有任何到点定位信号**(纯休息窗)时才按休息词排除。
    _has_arrival_point = (
        re.search(r"[（(]\s*-?\d+(?:\.\d+)?\s*[，,]\s*-?\d+(?:\.\d+)?\s*[）)]", text) is not None
        or any(t in text for t in ("回到", "回家", "赶回", "返回"))
        or (any(u in text for u in ("公里", "千米", "米")) and "内" in text)
    )
    if (not _has_arrival_point) and any(t in text for t in ("睡觉", "熄火", "静止", "停驶", "不接单", "休息", "停车")):
        return None
    # 再要求有"到点/截止"语义,才把钟点当 daily_location 截止时点。
    if not any(t in text for t in ("点前", "时前", "之前", "不晚于", "不得晚于", "回到", "回家", "赶回")):
        return None
    # 通用时点解析(阿拉伯/中文数字 + 时段),不再只认阿拉伯"X点前"。
    return _parse_clock_hour(text)


def _daily_location_circle(pref: dict[str, Any], parsed_data: dict[str, Any]) -> dict[str, Any] | None:
    explicit = _valid_circle(parsed_data.get("daily_location_circle"))
    if explicit is not None:
        return explicit
    text = str(pref.get("content", ""))
    if _daily_location_deadline_hour(pref, parsed_data) is None:
        return None
    coord = re.search(r"[（(]\s*(-?\d+(?:\.\d+)?)\s*[，,]\s*(-?\d+(?:\.\d+)?)\s*[）)]", text)
    if not coord:
        return None
    radius = 1.0
    radius_match = re.search(r"(\d+(?:\.\d+)?)\s*公里", text)
    if radius_match:
        radius = float(radius_match.group(1))
    elif "一公里" in text:
        radius = 1.0
    return {
        "center_lat": float(coord.group(1)),
        "center_lng": float(coord.group(2)),
        "radius_km": radius,
        "label": "daily_location_deadline",
    }


def _geo_constraint_type(pref: dict[str, Any], parsed_data: dict[str, Any]) -> str | None:
    value = parsed_data.get("geo_constraint_type")
    if value in _VALID_GEO_CONSTRAINT_TYPES:
        return str(value)
    if _daily_location_deadline_hour(pref, parsed_data) is not None:
        return None
    text = str(pref.get("content", ""))
    if not any(token in text for token in ("范围", "区域", "北纬", "东经", "坐标", "位置", "公里")):
        return None
    forbidden_signals = (
        "不得进入",
        "不能进入",
        "不进入",
        "禁止进入",
        "禁入",
        "避开",
        "不得驶入",
        "不能驶入",
        "不驶入",
        "不得靠近",
        "不能靠近",
    )
    allowed_signals = (
        "范围内",
        "区域内",
        "不得离开",
        "不能离开",
        "不出",
        "不离开",
        "只在",
        "仅在",
        "必须在",
        "须在",
        "限制在",
        "跑车",
        "活动",
    )
    visit_signals = (
        "到过",
        "到访",
        "去过",
        "经过",
        "至少",
        "打卡",
        "抵达",
        "前往",
    )
    if any(token in text for token in forbidden_signals):
        return GEO_CONSTRAINT_FORBIDDEN_REGION
    if any(token in text for token in allowed_signals):
        return GEO_CONSTRAINT_ALLOWED_REGION
    if any(token in text for token in visit_signals):
        return GEO_CONSTRAINT_VISIT_TARGET
    return None


def _visit_target_days(pref: dict[str, Any], parsed_data: dict[str, Any]) -> int | None:
    """Extract '至少N个不同的自然日' style count for visit_target prefs."""
    explicit = parsed_data.get("visit_target_days") if isinstance(parsed_data, dict) else None
    parsed = _pos_int(explicit)
    if parsed is not None:
        return parsed
    text = str(pref.get("content", ""))
    if "到过" not in text and "到访" not in text and "经过" not in text and "至少" not in text:
        return None
    match = re.search(r"至少\s*(\d+|[一二两三四五六七八九十]{1,3})\s*(?:个不同的自然日|天|个自然日|日)", text)
    if not match:
        return None
    raw = match.group(1)
    return _pos_int(raw) if raw.isdigit() else _cn_num_to_int(raw)


def _valid_bbox(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        out = {
            "min_lat": float(value["min_lat"]),
            "max_lat": float(value["max_lat"]),
            "min_lng": float(value["min_lng"]),
            "max_lng": float(value["max_lng"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if out["min_lat"] > out["max_lat"]:
        out["min_lat"], out["max_lat"] = out["max_lat"], out["min_lat"]
    if out["min_lng"] > out["max_lng"]:
        out["min_lng"], out["max_lng"] = out["max_lng"], out["min_lng"]
    out["label"] = str(value.get("label", "")).strip()[:80]
    return out


def _valid_circle(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        center_lat = float(value["center_lat"])
        center_lng = float(value["center_lng"])
        radius_km = float(value["radius_km"])
    except (KeyError, TypeError, ValueError):
        return None
    if radius_km <= 0:
        return None
    return {
        "center_lat": center_lat,
        "center_lng": center_lng,
        "radius_km": radius_km,
        "label": str(value.get("label", "")).strip()[:80],
    }


# ----------------- compliance / cap accounting -----------------


def count_category_violations(
    history_records: list[dict[str, Any]],
    categories: set[str],
    cargo_name_by_id: dict[str, str] | None = None,
) -> int:
    """Count accepted take_orders whose cargo_name matches one of categories."""
    if not categories:
        return 0
    cargo_names = cargo_name_by_id or {}
    n = 0
    for rec in history_records or []:
        action = rec.get("action") if isinstance(rec.get("action"), dict) else {}
        if str(action.get("action", "")).strip().lower() != "take_order":
            continue
        result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        if not bool(result.get("accepted", False)):
            continue
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        cargo_id = str(params.get("cargo_id") or result.get("cargo_id") or "").strip()
        cargo_name = str(
            result.get("cargo_name")
            or rec.get("cargo_name")
            or cargo_names.get(cargo_id, "")
        ).strip()
        if not cargo_name:
            continue
        if cargo_name in categories:
            n += 1
    return n


def _history_accepted_category_names(
    history_records: list[dict[str, Any]],
    cargo_name_by_id: dict[str, str] | None = None,
) -> set[str]:
    """Set of cargo_name (category) values the driver has already accepted.

    Feeds the required_categories reward so a "必接「冷链」" preference is credited
    once the category is taken (and not double-credited across planning paths)."""
    cargo_names = cargo_name_by_id or {}
    out: set[str] = set()
    for rec in history_records or []:
        action = rec.get("action") if isinstance(rec.get("action"), dict) else {}
        if str(action.get("action", "")).strip().lower() != "take_order":
            continue
        result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        if not bool(result.get("accepted", False)):
            continue
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        cargo_id = str(params.get("cargo_id") or result.get("cargo_id") or "").strip()
        name = str(
            result.get("cargo_name") or rec.get("cargo_name") or cargo_names.get(cargo_id, "")
        ).strip()
        if name:
            out.add(name)
    return out


def _constraint_types(p: ParsedPreference) -> list[str]:
    types: list[str] = []
    if p.excluded_categories:
        types.append("category")
    if p.required_categories:
        types.append("required_category")
    if p.forbidden_endpoint_locations:
        types.append("forbidden_endpoint_location")
    if p.required_endpoint_locations and p.required_endpoint_location_days:
        types.append("required_endpoint_location_days")
    if p.required_cargo_ids:
        types.append("required_cargo_ids")
    if p.itinerary_commitment:
        types.append("itinerary_commitment")
    if p.rest_type:
        types.append(f"rest_{p.rest_type}")
    if p.first_order_deadline_hour is not None:
        types.append("first_order_deadline")
    if p.max_haul_km is not None:
        types.append("max_haul_km")
    if p.max_pickup_deadhead_km is not None:
        types.append("max_pickup_deadhead_km")
    if p.daily_location_deadline_hour is not None:
        types.append("daily_location_deadline")
    # Operational obligations (Iter35): MUST be registered here so a pref carrying ONLY one of
    # these is not flagged by the self-check (line ~306: "penalty>0 but no constraint extracted")
    # nor by the reviewer as an empty parse — which would churn re-parses and, worse, let the
    # retry-accept logic (which compares len(_constraint_types)) accept a re-parse that DROPPED
    # the new field. Without this they were invisible to the audit even though correctly parsed.
    if getattr(p, "driving_limits", None):
        types.append("driving_limits")
    if getattr(p, "range_obligation", None):
        types.append("range_obligation")
    if getattr(p, "transit_time_limits", None):
        types.append("transit_time_limits")
    if _prices_geo_region(p):
        types.append(f"geo_{p.geo_constraint_type}")
    elif p.geo_bbox or p.geo_circle:
        types.append(f"geo_{p.geo_constraint_type or 'reference'}")
    return types


def penalty_accounting(
    parsed_prefs: list[ParsedPreference],
    history_records: list[dict[str, Any]],
    cargo_name_by_id: dict[str, str] | None = None,
    current_time_min: int | None = None,
    current_pos: tuple[float, float] | None = None,
    cargo_info_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Per-preference cap accounting across **all** structured constraint types.

    Returns one entry per parsed pref with penalty_amount > 0 and at least one
    recognized constraint. Each entry summarizes how much of the cap has been
    spent so far based on history, what the next single violation would cost,
    and whether the cap is fully exhausted (so future violations are free).
    """
    # daily_location / fixed_window / monthly_days are now priced as virtual
    # cargos in plan_routes (see __home_v_* / __rest_window_v_* / __no_order_explore_v_*),
    # not as compliance penalties — no more in-flight penalty accumulation here.
    _ = current_time_min, current_pos  # kept in signature for backwards compat
    state = _preference_history_state(parsed_prefs, history_records, cargo_name_by_id, cargo_info_by_id)
    paid_by_pref = state["paid_by_pref"]
    geo_days = state.get("geo_violation_days") or {}
    first_order_days = state.get("first_order_late_days") or {}

    detail: list[dict[str, Any]] = []
    for idx, p in enumerate(parsed_prefs):
        if p.penalty_amount <= 0:
            continue
        types = _constraint_types(p)
        if not types:
            continue
        category_violations = (
            count_category_violations(history_records, set(p.excluded_categories), cargo_name_by_id)
            if p.excluded_categories
            else 0
        )
        geo_violations = len(geo_days.get(idx, set())) if _prices_geo_region(p) else 0
        continuous_daily_violations = (
            len((state.get("continuous_daily_miss_days") or {}).get(idx, set()))
            if p.rest_type == REST_TYPE_CONTINUOUS_DAILY
            else 0
        )
        first_order_violations = (
            len(first_order_days.get(idx, set())) if p.first_order_deadline_hour is not None else 0
        )

        paid = float(paid_by_pref.get(idx, 0.0))
        if p.penalty_cap is None:
            remaining = None
            cap_exhausted = False
            raw_next_cost = float(p.penalty_amount)
        else:
            remaining = max(0.0, float(p.penalty_cap) - paid)
            cap_exhausted = remaining <= 0
            raw_next_cost = min(float(p.penalty_amount), remaining)
        next_cost = effective_penalty_amount(p, raw_next_cost)
        detail.append(
            {
                "index": idx,
                "preference": p.raw_content[:80],
                "constraint_types": types,
                "categories": list(p.excluded_categories),
                "forbidden_endpoint_locations": list(p.forbidden_endpoint_locations),
                "required_endpoint_locations": list(p.required_endpoint_locations),
                "required_endpoint_location_days": p.required_endpoint_location_days,
                "rest_type": p.rest_type,
                "geo_constraint_type": p.geo_constraint_type,
                "geo_bbox": p.geo_bbox,
                "geo_circle": p.geo_circle,
                "penalty_amount": p.penalty_amount,
                "penalty_cap": p.penalty_cap,
                "paid": round(paid, 2),
                "cap_remaining": None if remaining is None else round(remaining, 2),
                "cap_exhausted": cap_exhausted,
                "raw_next_violation_cost": round(raw_next_cost, 2),
                "next_violation_cost": round(next_cost, 2),
                "violations": {
                    "category": category_violations,
                    "geo": geo_violations,
                    "continuous_daily": continuous_daily_violations,
                    "first_order_deadline": first_order_violations,
                },
            }
        )
    return detail


def annotate_route_preference_penalties(
    paths: list[dict[str, Any]],
    parsed_prefs: list[ParsedPreference],
    history_records: list[dict[str, Any]],
    cargo_name_by_id: dict[str, str] | None = None,
    cargo_info_by_id: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Mutate planned paths with preference penalties and adjusted yields.

    This intentionally keeps cargos in the candidate set. Soft preferences are
    expressed as money: each hop/path carries the expected incremental penalty
    and an adjusted yield used by plan_routes ranking.
    """
    if not paths:
        return
    history_state = _preference_history_state(parsed_prefs, history_records, cargo_name_by_id, cargo_info_by_id)
    # History-only derived structures are identical across all paths in this
    # call, so compute them once and reuse. Each path takes a shallow dict
    # copy where it needs a mutable view.
    base_first_starts = _history_first_order_starts(history_records)
    base_required_endpoint_days = _history_required_endpoint_location_days(
        parsed_prefs, history_records, cargo_info_by_id
    )
    base_accepted_categories = (
        _history_accepted_category_names(history_records, cargo_name_by_id)
        if any(getattr(pref, "required_categories", None) for pref in parsed_prefs)
        else set()
    )
    itinerary_history_index = (
        _itinerary_history_index(history_records)
        if any(getattr(pref, "itinerary_commitment", None) for pref in parsed_prefs)
        else None
    )
    for path in paths:
        route_state = _copy_penalty_state(history_state)
        path_penalty = 0.0
        path_reward = 0.0
        path_reasons: list[dict[str, Any]] = []
        path_reward_reasons: list[dict[str, Any]] = []
        planned_first_starts = dict(base_first_starts)
        planned_required_endpoint_days = {
            int(k): set(v)
            for k, v in base_required_endpoint_days.items()
        }
        planned_required_categories: dict[int, set[str]] = {}
        planned_rest_window_days: dict[int, set[int]] = {}
        planned_continuous_rest_days: dict[int, set[int]] = {}
        continuous_rest_satisfied_days: dict[int, set[int]] = {}
        prev_pos = _coerce_pos(path.get("route_start_pos"))
        route_start_time = int(path.get("route_start_time_min", 0) or 0)
        for hop_index, hop in enumerate(path.get("hops") or [], start=1):
            hop_penalty = 0.0
            hop_reward = 0.0
            hop_reasons: list[dict[str, Any]] = []
            hop_reward_reasons: list[dict[str, Any]] = []
            cargo = hop.get("cargo") if isinstance(hop.get("cargo"), dict) else {}
            cargo_name = str(cargo.get("cargo_name", "")).strip()
            action_start = int(hop.get("action_start_min", hop.get("arrival_min", 0)) or 0)
            action_end = int(hop.get("finish_min", action_start) or action_start)
            day_idx = action_start // MINUTES_PER_DAY
            pickup_pos = _coerce_pos((cargo.get("start") or {}))
            drop_pos = _coerce_pos(hop.get("end_pos")) or _coerce_pos((cargo.get("end") or {}))

            for pref_idx, pref in enumerate(parsed_prefs):
                if pref.penalty_amount <= 0:
                    continue
                penalty_amount = effective_penalty_amount(pref)
                if cargo_name and cargo_name in set(pref.excluded_categories):
                    delta = _charge_penalty(route_state, pref_idx, pref, penalty_amount)
                    if delta > 0:
                        hop_penalty += delta
                        hop_reasons.append(
                            {
                                "type": "category",
                                "category": cargo_name,
                                "penalty": round(delta, 2),
                                "preference": pref.raw_content[:60],
                            }
                        )

                # Required-category reward ("必接「冷链」"): credit a path that
                # accepts a required category not yet satisfied (in history or
                # earlier in this path). Mirrors required_endpoint_location_day_bonus
                # so plan_routes ranks the compliant cargo up — without this the
                # preference had NO economic signal and was silently lost.
                if (
                    pref.required_categories
                    and cargo_name
                    and not _is_virtual_cargo_id(cargo.get("cargo_id", hop.get("cargo_id", "")))
                ):
                    req_cats = {c for c in pref.required_categories if c}
                    if cargo_name in req_cats:
                        sat = planned_required_categories.setdefault(
                            pref_idx, req_cats & base_accepted_categories
                        )
                        if cargo_name not in sat:
                            sat.add(cargo_name)
                            bonus = effective_penalty_amount(pref) / max(1, len(req_cats))
                            if bonus > 0:
                                hop_reward += bonus
                                hop_reward_reasons.append(
                                    {
                                        "type": "required_category_bonus",
                                        "category": cargo_name,
                                        "remaining_categories": sorted(req_cats - sat),
                                        "bonus": round(bonus, 2),
                                        "preference": pref.raw_content[:60],
                                    }
                                )

                if pref.forbidden_endpoint_locations:
                    pickup_hit, drop_hit = _cargo_matches_forbidden_endpoint_location(cargo, pref)
                    if pickup_hit or drop_hit:
                        delta = _charge_penalty(route_state, pref_idx, pref, penalty_amount)
                        if delta > 0:
                            hop_penalty += delta
                            hop_reasons.append(
                                {
                                    "type": "forbidden_endpoint_location",
                                    "locations": list(pref.forbidden_endpoint_locations),
                                    "where": _geo_violation_where(pickup_hit, drop_hit),
                                    "penalty": round(delta, 2),
                                    "preference": pref.raw_content[:60],
                                }
                            )

                if (
                    pref.required_endpoint_locations
                    and pref.required_endpoint_location_days
                    and not _is_virtual_cargo_id(cargo.get("cargo_id", hop.get("cargo_id", "")))
                ):
                    pickup_hit, drop_hit = _cargo_matches_endpoint_location(cargo, pref.required_endpoint_locations)
                    if pickup_hit or drop_hit:
                        day_set = planned_required_endpoint_days.setdefault(pref_idx, set())
                        if len(day_set) < int(pref.required_endpoint_location_days) and day_idx not in day_set:
                            day_set.add(day_idx)
                            bonus = _required_endpoint_location_day_bonus(pref)
                            if bonus > 0:
                                hop_reward += bonus
                                hop_reward_reasons.append(
                                    {
                                        "type": "required_endpoint_location_day_bonus",
                                        "locations": list(pref.required_endpoint_locations),
                                        "where": _geo_violation_where(pickup_hit, drop_hit),
                                        "day": day_idx,
                                        "target_days": pref.required_endpoint_location_days,
                                        "bonus": round(bonus, 2),
                                        "preference": pref.raw_content[:60],
                                    }
                                )

                # Fixed-window rest protection: a real order whose execution
                # OVERLAPS a day's rest window (e.g. 0:00-6:00) means the driver
                # was moving when they should have been parked, forfeiting that
                # day's rest. Overlap (not just finish-in-window) covers every
                # shape: finishing inside, starting inside, sitting entirely
                # inside, and — the case finish-only missed — starting the
                # evening before and running straight THROUGH the window (e.g.
                # 20:24 -> next-day 07:18 crosses all of 0-6, finish 07:18 is
                # OUTSIDE so the old check skipped it). Charge once per violated
                # day so the planner parks before the window instead of plowing
                # through it.
                if (
                    pref.rest_type == REST_TYPE_FIXED_WINDOW
                    and pref.rest_window_start_hour is not None
                    and pref.rest_window_end_hour is not None
                    and _hop_breaks_fixed_window_rest(cargo.get("cargo_id", hop.get("cargo_id", "")))
                ):
                    win_start_min = int(pref.rest_window_start_hour) * 60
                    win_end_min = int(pref.rest_window_end_hour) * 60
                    # Derive from the hours only; never trust a possibly mis-tagged
                    # crosses flag (a 0-6 window wrongly flagged crosses would otherwise
                    # probe the previous day and over-charge a normal 7-8am order).
                    crosses = win_start_min > win_end_min
                    start_day = action_start // MINUTES_PER_DAY
                    end_day = max(action_start, action_end - 1) // MINUTES_PER_DAY
                    # For a midnight-crossing window the instance that covers an
                    # early-morning order actually started the previous evening,
                    # so also probe start_day-1.
                    first_day = start_day - 1 if crosses else start_day
                    rest_days = planned_rest_window_days.setdefault(pref_idx, set())
                    for win_day in range(first_day, end_day + 1):
                        win_lo = win_day * MINUTES_PER_DAY + win_start_min
                        win_hi = (
                            (win_day + 1) * MINUTES_PER_DAY + win_end_min
                            if crosses
                            else win_day * MINUTES_PER_DAY + win_end_min
                        )
                        if win_hi <= win_lo:
                            continue
                        if not _interval_overlap(action_start, action_end, win_lo, win_hi):
                            continue
                        if win_day in rest_days:
                            continue
                        rest_days.add(win_day)
                        delta = _charge_penalty(route_state, pref_idx, pref, penalty_amount)
                        if delta > 0:
                            hop_penalty += delta
                            hop_reasons.append(
                                {
                                    "type": "fixed_window_rest_overlap",
                                    "window": f"{pref.rest_window_start_hour}:00-{pref.rest_window_end_hour}:00",
                                    "day": win_day,
                                    "action_start_min": action_start,
                                    "finish_min": action_end,
                                    "penalty": round(delta, 2),
                                    "preference": pref.raw_content[:60],
                                }
                            )

                # Continuous-daily rest protection (mirror of fixed_window above,
                # by finish time): when today's >=X-hour continuous rest is NOT
                # yet secured, a real order whose finish leaves the day's
                # remaining rest-able window (finish -> midnight) shorter than the
                # threshold has stranded that day's rest. Charge the rest penalty
                # once per such day so the planner schedules the rest earlier (or
                # skips the late order) instead of fragmenting the day past
                # rescue. A virtual rest hop (__rest_v_) covering the threshold
                # marks the day satisfied so a legit rest->work path is not hit.
                if (
                    pref.rest_type == REST_TYPE_CONTINUOUS_DAILY
                    and pref.rest_continuous_hours
                ):
                    threshold_min = int(float(pref.rest_continuous_hours) * 60)
                    cargo_id_str = str(cargo.get("cargo_id", hop.get("cargo_id", "")))
                    if threshold_min > 0 and _is_virtual_cargo_id(cargo_id_str):
                        if (
                            cargo.get("_virtual_rest_pref_idx") == pref_idx
                            and int(cargo.get("_virtual_rest_minutes", 0) or 0) >= threshold_min
                        ):
                            continuous_rest_satisfied_days.setdefault(pref_idx, set()).add(
                                action_start // MINUTES_PER_DAY
                            )
                    elif threshold_min > 0:
                        rest_day = action_end // MINUTES_PER_DAY
                        already_satisfied = (
                            rest_day in continuous_rest_satisfied_days.get(pref_idx, set())
                            or int(route_state["continuous_wait_max_by_day"].get(rest_day, 0) or 0)
                            >= threshold_min
                        )
                        if not already_satisfied:
                            finish_minute_of_day = action_end % MINUTES_PER_DAY
                            remaining_to_midnight = MINUTES_PER_DAY - finish_minute_of_day
                            if remaining_to_midnight < threshold_min:
                                charged = planned_continuous_rest_days.setdefault(pref_idx, set())
                                if rest_day not in charged:
                                    charged.add(rest_day)
                                    delta = _charge_penalty(route_state, pref_idx, pref, penalty_amount)
                                    if delta > 0:
                                        hop_penalty += delta
                                        hop_reasons.append(
                                            {
                                                "type": "continuous_rest_stranded",
                                                "threshold_min": threshold_min,
                                                "remaining_window_min": remaining_to_midnight,
                                                "finish_min": action_end,
                                                "day": rest_day,
                                                "penalty": round(delta, 2),
                                                "preference": pref.raw_content[:60],
                                            }
                                        )

                if pref.max_haul_km is not None:
                    haul_km = float(hop.get("haul_km", 0.0) or 0.0)
                    if haul_km > float(pref.max_haul_km):
                        delta = _charge_penalty(route_state, pref_idx, pref, penalty_amount)
                        if delta > 0:
                            hop_penalty += delta
                            hop_reasons.append(
                                {
                                    "type": "max_haul_km",
                                    "limit_km": pref.max_haul_km,
                                    "actual_km": round(haul_km, 2),
                                    "penalty": round(delta, 2),
                                    "preference": pref.raw_content[:60],
                                }
                            )

                if pref.max_pickup_deadhead_km is not None:
                    deadhead_km = float(hop.get("deadhead_km", 0.0) or 0.0)
                    if deadhead_km > float(pref.max_pickup_deadhead_km):
                        delta = _charge_penalty(route_state, pref_idx, pref, penalty_amount)
                        if delta > 0:
                            hop_penalty += delta
                            hop_reasons.append(
                                {
                                    "type": "max_pickup_deadhead_km",
                                    "limit_km": pref.max_pickup_deadhead_km,
                                    "actual_km": round(deadhead_km, 2),
                                    "penalty": round(delta, 2),
                                    "preference": pref.raw_content[:60],
                                }
                            )

                if pref.first_order_deadline_hour is not None:
                    first_start = planned_first_starts.get(day_idx)
                    if first_start is None or action_start < first_start:
                        planned_first_starts[day_idx] = action_start
                        deadline = day_idx * MINUTES_PER_DAY + int(pref.first_order_deadline_hour) * 60
                        if action_start >= deadline and day_idx not in route_state["first_order_late_days"].get(pref_idx, set()):
                            route_state["first_order_late_days"].setdefault(pref_idx, set()).add(day_idx)
                            delta = _charge_penalty(route_state, pref_idx, pref, penalty_amount)
                            if delta > 0:
                                hop_penalty += delta
                                hop_reasons.append(
                                    {
                                        "type": "first_order_deadline",
                                        "deadline_hour": pref.first_order_deadline_hour,
                                        "day": day_idx,
                                        "penalty": round(delta, 2),
                                        "preference": pref.raw_content[:60],
                                    }
                                )

                if (
                    _prices_geo_region(pref)
                    and _hop_breaks_fixed_window_rest(cargo.get("cargo_id", hop.get("cargo_id", "")))
                    and _pref_active_window_overlaps(pref, action_start, action_end)
                ):
                    pickup_bad = _violates_geo_constraint(pickup_pos, pref)
                    drop_bad = _violates_geo_constraint(drop_pos, pref)
                    if pickup_bad or drop_bad:
                        # Count once per natural day per pref to mirror the
                        # "any moment outside" semantics without spamming
                        # multi-hop intra-day routes.
                        day_set = route_state["geo_violation_days"].setdefault(pref_idx, set())
                        if day_idx not in day_set:
                            day_set.add(day_idx)
                            delta = _charge_penalty(route_state, pref_idx, pref, penalty_amount)
                            if delta > 0:
                                hop_penalty += delta
                                hop_reasons.append(
                                    {
                                        "type": f"geo_{pref.geo_constraint_type}",
                                        "where": _geo_violation_where(pickup_bad, drop_bad),
                                        "day": day_idx,
                                        "penalty": round(delta, 2),
                                        "preference": pref.raw_content[:60],
                                    }
                                )

            hop["preference_penalty"] = round(hop_penalty, 2)
            hop["preference_penalty_reasons"] = hop_reasons
            hop["preference_reward"] = round(hop_reward, 2)
            hop["preference_reward_reasons"] = hop_reward_reasons
            hop["adjusted_net_yield"] = float(hop.get("net_yield", 0.0)) - hop_penalty + hop_reward
            path_penalty += hop_penalty
            path_reward += hop_reward
            path_reasons.extend(hop_reasons)
            path_reward_reasons.extend(hop_reward_reasons)
            prev_pos = drop_pos or prev_pos

        finish_min = int(path.get("finish_min", route_start_time) or route_start_time)

        path_cargo_ids = {str(h.get("cargo_id", "")).strip() for h in (path.get("hops") or [])}
        required_penalty, required_reasons = _required_cargo_route_penalty(
            route_state,
            parsed_prefs,
            history_records,
            path_cargo_ids,
            route_start_time,
        )
        path_penalty += required_penalty
        path_reasons.extend(required_reasons)

        finish_pos_pref = _coerce_pos(path.get("finish_pos")) or prev_pos
        itinerary_penalty, itinerary_reasons = _itinerary_route_penalty(
            route_state,
            parsed_prefs,
            history_records,
            route_start_time,
            finish_min,
            finish_pos_pref,
            path_cargo_ids,
            history_index=itinerary_history_index,
        )
        path_penalty += itinerary_penalty
        path_reasons.extend(itinerary_reasons)

        # Daily-location / fixed-window / monthly-rest preferences are now
        # priced as virtual cargos injected into plan_routes (__home_v_*,
        # __rest_window_v_*, __no_order_explore_v_*) rather than as penalties on
        # paths that miss them.

        gross = float(path.get("total_net_yield", 0.0))
        path["preference_penalty"] = round(path_penalty, 2)
        path["preference_reward"] = round(path_reward, 2)
        path["preference_penalty_reasons"] = path_reasons[:20]
        path["preference_reward_reasons"] = path_reward_reasons[:20]
        path["preference_adjusted_yield"] = round(gross - path_penalty + path_reward, 2)
        time_min = max(1, int(path.get("total_time_min", 0) or 0))
        path["preference_adjusted_yield_per_min"] = round((gross - path_penalty + path_reward) / time_min, 4)


def _pref_active_window_overlaps(pref: ParsedPreference, action_start: int, action_end: int) -> bool:
    """A time-windowed preference (e.g. "三月四号五号不进深圳") only constrains
    actions that fall inside its active window. Returns True when the pref has no
    window (always active) or the action [start, end) overlaps it.

    Without this guard a forbidden_region rule counts a driver parked in the
    region on OUT-OF-WINDOW days (e.g. D001 resting 3/1-3/3 at a home that sits
    inside the 深圳 circle), phantom-exhausting the cap before the rule is even
    active — which then makes the model believe 深圳 is "free" on the real
    forbidden day."""
    start = pref.active_start_min
    end = pref.active_end_min
    if start is None and end is None:
        return True
    lo = int(start) if start is not None else 0
    hi = int(end) if end is not None else 10**12
    return _interval_overlap(int(action_start), int(action_end), lo, hi + 1)


def _preference_history_state(
    parsed_prefs: list[ParsedPreference],
    history_records: list[dict[str, Any]],
    cargo_name_by_id: dict[str, str] | None,
    cargo_info_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    paid_by_pref = {idx: 0.0 for idx, _p in enumerate(parsed_prefs)}
    continuous_daily_miss_days: dict[int, set[int]] = {}
    first_order_late_days: dict[int, set[int]] = {}
    first_starts = _history_first_order_starts(history_records)
    continuous_wait_max_by_day = _continuous_wait_max_by_day(history_records)
    history_end_min = _history_end_min(history_records)
    completed_day_count = history_end_min // MINUTES_PER_DAY
    cargo_names = cargo_name_by_id or {}
    # History records carry only cargo_id (no endpoint city text), so endpoint
    # matching must resolve the cargo via cargo_info_by_id, not the raw record.
    info_by_id = cargo_info_by_id or {}
    geo_violation_days: dict[int, set[int]] = {}
    for record, step_start, action_start, action_end, _step_end in _iter_history_with_time(history_records):
        action = record.get("action") if isinstance(record.get("action"), dict) else {}
        action_name = str(action.get("action", "")).strip().lower()
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        accepted = action_name == "take_order" and bool(result.get("accepted", False))
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        cargo_id = str(params.get("cargo_id") or result.get("cargo_id") or "").strip()
        cargo_name = str(result.get("cargo_name") or record.get("cargo_name") or cargo_names.get(cargo_id, "")).strip()
        pickup_deadhead_km = _optional_float(result.get("pickup_deadhead_km", result.get("distance_km")))
        haul_km = _optional_float(result.get("haul_distance_km"))
        position_after = _position_from_record(record, "position_after")
        position_before = _position_from_record(record, "position_before")
        action_day = action_start // MINUTES_PER_DAY
        for pref_idx, pref in enumerate(parsed_prefs):
            if pref.penalty_amount <= 0:
                continue
            if accepted and cargo_name and cargo_name in set(pref.excluded_categories):
                paid_by_pref[pref_idx] += pref.penalty_amount
            if (
                accepted
                and pref.forbidden_endpoint_locations
                and _pref_active_window_overlaps(pref, action_start, action_end)
            ):
                info = info_by_id.get(cargo_id)
                cargo_like = _endpoint_cargo_from_info(info) if isinstance(info, dict) else record
                endpoint_hit = _cargo_matches_forbidden_endpoint_location(cargo_like, pref)
                if endpoint_hit[0] or endpoint_hit[1]:
                    paid_by_pref[pref_idx] += pref.penalty_amount
            if accepted and pref.max_haul_km is not None and haul_km is not None and haul_km > float(pref.max_haul_km):
                paid_by_pref[pref_idx] += pref.penalty_amount
            if (
                accepted
                and pref.max_pickup_deadhead_km is not None
                and pickup_deadhead_km is not None
                and pickup_deadhead_km > float(pref.max_pickup_deadhead_km)
            ):
                paid_by_pref[pref_idx] += pref.penalty_amount
            # Forbidden-region (position) violation. Mirror the scorer: count only
            # MOVING actions (reposition / non-accepted take_order) — a parked wait
            # inside the region is NOT a violation — and only while the rule is
            # ACTIVE. An accepted take_order is judged by its cargo endpoint text
            # above (forbidden_endpoint), not by position, so it is excluded here
            # to avoid divergence/double-counting.
            geo_position_action = action_name == "reposition" or (action_name == "take_order" and not accepted)
            if (
                _prices_geo_region(pref)
                and geo_position_action
                and _pref_active_window_overlaps(pref, action_start, action_end)
            ):
                outside_after = _violates_geo_constraint(position_after, pref)
                outside_before = _violates_geo_constraint(position_before, pref)
                if outside_after or outside_before:
                    day_set = geo_violation_days.setdefault(pref_idx, set())
                    if action_day not in day_set:
                        day_set.add(action_day)
                        paid_by_pref[pref_idx] += pref.penalty_amount
    for pref_idx, pref in enumerate(parsed_prefs):
        if pref.rest_type != REST_TYPE_CONTINUOUS_DAILY or pref.penalty_amount <= 0:
            continue
        threshold_min = int((pref.rest_continuous_hours or 0) * 60)
        if threshold_min <= 0:
            continue
        miss_days = continuous_daily_miss_days.setdefault(pref_idx, set())
        for day_idx in range(max(0, completed_day_count)):
            if int(continuous_wait_max_by_day.get(day_idx, 0) or 0) >= threshold_min:
                continue
            miss_days.add(day_idx)
            paid_by_pref[pref_idx] += pref.penalty_amount
    for pref_idx, pref in enumerate(parsed_prefs):
        if pref.first_order_deadline_hour is None or pref.penalty_amount <= 0:
            continue
        late_days: set[int] = set()
        for day_idx, first_start in first_starts.items():
            deadline = day_idx * MINUTES_PER_DAY + int(pref.first_order_deadline_hour) * 60
            if first_start >= deadline:
                late_days.add(day_idx)
                paid_by_pref[pref_idx] += pref.penalty_amount
        first_order_late_days[pref_idx] = late_days
    for pref_idx, pref in enumerate(parsed_prefs):
        if pref.penalty_cap is not None:
            paid_by_pref[pref_idx] = min(paid_by_pref[pref_idx], pref.penalty_cap)
    return {
        "paid_by_pref": paid_by_pref,
        "continuous_daily_miss_days": continuous_daily_miss_days,
        "continuous_wait_max_by_day": continuous_wait_max_by_day,
        "first_order_late_days": first_order_late_days,
        "geo_violation_days": geo_violation_days,
    }


def _copy_penalty_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "paid_by_pref": dict(state.get("paid_by_pref") or {}),
        "continuous_daily_miss_days": {
            int(k): set(v)
            for k, v in (state.get("continuous_daily_miss_days") or {}).items()
        },
        "continuous_wait_max_by_day": dict(state.get("continuous_wait_max_by_day") or {}),
        "first_order_late_days": {
            int(k): set(v)
            for k, v in (state.get("first_order_late_days") or {}).items()
        },
        "geo_violation_days": {
            int(k): set(v)
            for k, v in (state.get("geo_violation_days") or {}).items()
        },
    }


def _charge_penalty(state: dict[str, Any], pref_idx: int, pref: ParsedPreference, amount: float) -> float:
    paid_by_pref = state["paid_by_pref"]
    already_paid = float(paid_by_pref.get(pref_idx, 0.0) or 0.0)
    if pref.penalty_cap is None:
        delta = float(amount)
    else:
        remaining = max(0.0, float(pref.penalty_cap) - already_paid)
        delta = min(float(amount), remaining)
    paid_by_pref[pref_idx] = already_paid + delta
    return delta


_DEFAULT_REPOSITION_SPEED_KM_H = 60.0


def _required_cargo_route_penalty(
    state: dict[str, Any],
    parsed_prefs: list[ParsedPreference],
    history_records: list[dict[str, Any]],
    path_cargo_ids: set[str],
    route_start_min: int = 0,
) -> tuple[float, list[dict[str, Any]]]:
    """Charge full penalty_amount on any path that does NOT include a required
    cargo_id, until the cargo has been accepted in history.

    Used for "熟货指定" type preferences where missing a specific cargo costs
    a fixed lump sum. By charging the SAME penalty to all routes that miss it,
    the one route that includes the required cargo emerges as preferred.

    Once the cargo's deadline has passed and the driver still hasn't accepted
    it, the loss is **sunk** — keeping the penalty on every future path is
    double-counting and pollutes the ranking. Skip in that case.
    """
    penalty = 0.0
    reasons: list[dict[str, Any]] = []
    accepted_ids = _history_accepted_cargo_ids(history_records)
    for pref_idx, pref in enumerate(parsed_prefs):
        if pref.penalty_amount <= 0 or not pref.required_cargo_ids:
            continue
        required = set(pref.required_cargo_ids)
        already_taken = required & accepted_ids
        if already_taken:
            continue  # satisfied already
        deadline_min = getattr(pref, "required_cargo_deadline_time_min", None)
        if deadline_min is not None and int(deadline_min) < int(route_start_min):
            continue  # deadline missed — sunk cost, don't keep penalising
        if _path_contains_required_cargo(pref_idx, required, path_cargo_ids):
            continue  # this path picks one of the required cargos
        delta = _charge_penalty(state, pref_idx, pref, pref.penalty_amount)
        if delta <= 0:
            continue
        penalty += delta
        reasons.append(
            {
                "type": "required_cargo_missed",
                "required_cargo_ids": sorted(required),
                "penalty": round(delta, 2),
                "preference": pref.raw_content[:60],
            }
        )
    return penalty, reasons


def _path_contains_required_cargo(pref_idx: int, required: set[str], path_cargo_ids: set[str]) -> bool:
    if required & path_cargo_ids:
        return True
    virtual_prefix = f"__pickup_v_{pref_idx}_"
    for cid in path_cargo_ids:
        cid_text = str(cid).strip()
        if not cid_text.startswith(virtual_prefix):
            continue
        target_id = cid_text[len(virtual_prefix):]
        if target_id in required:
            return True
    return False


ITINERARY_VIRTUAL_PREFIX = "__commit_v_"


def _itinerary_history_index(
    history_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Precompute the history-only data feeding ``completed_itinerary_event_ids``.

    ``history_records`` doesn't change across paths within a single planner
    call, so deriving the taken-virtual set, position trail and wait segments
    once and reusing them avoids paying the scan per (path, preference).
    """
    taken_virtual: set[str] = set()
    for rec in history_records or []:
        action = rec.get("action") if isinstance(rec.get("action"), dict) else {}
        if str(action.get("action", "")).strip().lower() != "take_order":
            continue
        result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        if not bool(result.get("accepted", False)):
            continue
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        cargo_id = str(params.get("cargo_id") or result.get("cargo_id") or "").strip()
        if cargo_id.startswith(ITINERARY_VIRTUAL_PREFIX):
            taken_virtual.add(cargo_id)
    history_points: list[tuple[int, tuple[float, float]]] = []
    wait_segments: list[tuple[int, int, tuple[float, float]]] = []
    for rec, _step_start, action_start, action_end, step_end in _iter_history_with_time(history_records):
        pos = _position_from_record(rec, "position_after")
        if pos is not None:
            history_points.append((int(step_end), pos))
        action = rec.get("action") if isinstance(rec.get("action"), dict) else {}
        if str(action.get("action", "")).strip().lower() == "wait" and action_end > action_start:
            wait_pos = pos or _position_from_record(rec, "position_before")
            if wait_pos is not None:
                wait_segments.append((int(action_start), int(action_end), wait_pos))
    return {
        "taken_virtual": taken_virtual,
        "history_points": history_points,
        "wait_segments": wait_segments,
    }


def completed_itinerary_event_ids(
    events: list[dict[str, Any]],
    history_records: list[dict[str, Any]],
    extra_taken_cargo_ids: set[str] | None = None,
    history_index: dict[str, Any] | None = None,
) -> set[str]:
    """Return event_ids that history evidence shows are already satisfied.

    Two completion paths:
    1) The driver chose a virtual itinerary cargo whose ``cargo_id`` carries
       the event_id (prefix ``__commit_v_{pref_idx}_{event_id}``); any such
       accepted take_order completes the event regardless of dwell.
    2) Heuristic: any action whose ``position_after`` is within ``radius_km``
       of the event center counts as a visit. Stay events still require the
       per-minute penalty path (handled elsewhere); planner-side we only need
       to know whether the visit has happened.
    """
    if history_index is None:
        history_index = _itinerary_history_index(history_records)
    taken_virtual = set(history_index["taken_virtual"])
    if extra_taken_cargo_ids:
        taken_virtual.update(extra_taken_cargo_ids)
    history_points = history_index["history_points"]
    wait_segments = history_index["wait_segments"]

    completed: set[str] = set()
    prev_event_id: str | None = None
    for ev in events:
        event_id = str(ev.get("event_id", "")).strip()
        if not event_id:
            continue
        # Sequential completion: events are listed in occurrence order, and the
        # scorer (_eval_route_stops) only credits stop[i] once it is reached
        # AFTER stop[i-1] (after_min ordering). Mirror that here — a later stop
        # visited out of order (e.g. arriving 四会 before 增城) must NOT mark the
        # commitment progressed, or the agent would falsely believe a multi-stop
        # itinerary is done. The explicit start_after guard below still applies
        # when the parser set it; this covers the common case where it didn't.
        if prev_event_id is not None and prev_event_id not in completed:
            break
        prev_event_id = event_id
        start_after = str(ev.get("start_after_event_id") or "").strip()
        if start_after and start_after not in completed:
            break
        target = _event_target(ev)
        if target is None:
            continue
        radius_km = float(ev.get("radius_km", 1.0))
        for taken_id in taken_virtual:
            if taken_id.endswith(f"_{event_id}"):
                completed.add(event_id)
                break
        if event_id in completed:
            continue
        not_before = ev.get("not_before_min")
        until_min = ev.get("until_min")
        event_type = str(ev.get("type") or "visit")
        dwell_min = int(ev.get("dwell_min") or 0)
        if event_type == "visit" and dwell_min > 0:
            # Require an explicit wait action at the position that covers the
            # full dwell — a transient drive-by (reposition / take_order drop)
            # is not enough to "接上配偶".
            for w_start, w_end, w_pos in wait_segments:
                if not_before is not None and w_end < int(not_before):
                    continue
                if w_end - w_start < dwell_min:
                    continue
                if _haversine_km(w_pos, target) > radius_km:
                    continue
                completed.add(event_id)
                break
            continue
        for step_end, pos in history_points:
            if not_before is not None and step_end < int(not_before):
                continue
            if event_type == "stay" and until_min is not None and step_end < int(until_min):
                continue
            if _haversine_km(pos, target) <= radius_km:
                completed.add(event_id)
                break
    return completed


def remaining_itinerary_events(
    events: list[dict[str, Any]],
    completed_event_ids: set[str],
) -> list[dict[str, Any]]:
    return [ev for ev in events if str(ev.get("event_id")) not in completed_event_ids]


def itinerary_feasible_from(
    events: list[dict[str, Any]],
    completed_event_ids: set[str],
    cur_pos: tuple[float, float],
    cur_time_min: int,
    speed_km_h: float,
) -> tuple[bool, dict[str, Any] | None]:
    """Simulate remaining events sequentially from (cur_pos, cur_time_min).

    Returns (feasible, first_blocker). ``feasible=False`` means at least one
    deadline cannot be honoured. Stay events block when their must_complete_before
    has already elapsed without the prerequisite visit.
    """
    remaining = remaining_itinerary_events(events, completed_event_ids)
    if not remaining:
        return True, None
    pos = (float(cur_pos[0]), float(cur_pos[1]))
    time_min = int(cur_time_min)
    speed = max(1.0, float(speed_km_h))
    for ev in remaining:
        target = _event_target(ev)
        if target is None:
            return False, {"event_id": ev.get("event_id"), "reason": "unresolved_location", "location_name": ev.get("location_name")}
        dist_km = _haversine_km(pos, target)
        travel_min = 0 if dist_km < 1e-6 else math.ceil(dist_km / speed * 60.0)
        arrival_min = time_min + travel_min
        not_before = ev.get("not_before_min")
        ready_min = max(arrival_min, int(not_before)) if not_before is not None else arrival_min
        event_type = str(ev.get("type") or "visit")
        until_min = ev.get("until_min")
        deadline = ev.get("must_complete_before_min")
        if deadline is None and event_type == "stay" and until_min is not None:
            deadline = until_min
        if deadline is not None and ready_min > int(deadline):
            return False, {"event_id": ev.get("event_id"), "reason": "arrival_after_deadline", "arrival_min": ready_min, "deadline_min": int(deadline)}
        dwell_min = int(ev.get("dwell_min") or 0)
        if event_type == "stay" and until_min is not None:
            done_min = max(ready_min, int(until_min))
        else:
            done_min = ready_min + dwell_min
        if deadline is not None and done_min > int(deadline):
            return False, {"event_id": ev.get("event_id"), "reason": "dwell_after_deadline", "done_min": done_min, "deadline_min": int(deadline)}
        time_min = done_min
        pos = target
    return True, None


def _itinerary_route_penalty(
    state: dict[str, Any],
    parsed_prefs: list[ParsedPreference],
    history_records: list[dict[str, Any]],
    route_start_min: int,
    finish_min: int,
    finish_pos: tuple[float, float] | None,
    path_cargo_ids: set[str],
    speed_km_h: float = 60.0,
    history_index: dict[str, Any] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Charge full penalty_amount on any path whose finish state cannot complete
    the remaining itinerary events on time.

    A path that *includes* a virtual itinerary cargo for an event is treated as
    satisfying that event (planner already places the driver at the location
    inside the path's time window).
    """
    penalty = 0.0
    reasons: list[dict[str, Any]] = []
    if finish_pos is None:
        return penalty, reasons
    in_path_virtual_ids = {cid for cid in path_cargo_ids if str(cid).startswith(ITINERARY_VIRTUAL_PREFIX)}
    for pref_idx, pref in enumerate(parsed_prefs):
        if pref.penalty_amount <= 0 or not pref.itinerary_commitment:
            continue
        # Skip if entire commitment window is already past
        deadlines = [ev.get("must_complete_before_min") for ev in pref.itinerary_commitment if ev.get("must_complete_before_min") is not None]
        until_values = [ev.get("until_min") for ev in pref.itinerary_commitment if ev.get("until_min") is not None]
        latest = max(deadlines + until_values) if (deadlines or until_values) else None
        if latest is not None and int(latest) < int(route_start_min):
            continue
        completed = completed_itinerary_event_ids(
            pref.itinerary_commitment,
            history_records,
            extra_taken_cargo_ids=in_path_virtual_ids,
            history_index=history_index,
        )
        feasible, blocker = itinerary_feasible_from(
            pref.itinerary_commitment,
            completed,
            finish_pos,
            finish_min,
            speed_km_h,
        )
        if feasible:
            continue
        delta = _charge_penalty(state, pref_idx, pref, pref.penalty_amount)
        if delta <= 0:
            continue
        penalty += delta
        reasons.append(
            {
                "type": "itinerary_commitment_infeasible",
                "blocker": blocker,
                "penalty": round(delta, 2),
                "preference": pref.raw_content[:60],
            }
        )
    return penalty, reasons


def _history_accepted_cargo_ids(history_records: list[dict[str, Any]]) -> set[str]:
    accepted: set[str] = set()
    for rec in history_records or []:
        action = rec.get("action") if isinstance(rec.get("action"), dict) else {}
        if str(action.get("action", "")).strip().lower() != "take_order":
            continue
        result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        if not bool(result.get("accepted", False)):
            continue
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        cid = str(params.get("cargo_id") or result.get("cargo_id") or "").strip()
        if cid:
            accepted.add(cid)
    return accepted


def _history_first_order_starts(history_records: list[dict[str, Any]]) -> dict[int, int]:
    first_starts: dict[int, int] = {}
    for record, _step_start, action_start, _action_end, _step_end in _iter_history_with_time(history_records):
        action = record.get("action") if isinstance(record.get("action"), dict) else {}
        if str(action.get("action", "")).strip().lower() != "take_order":
            continue
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        if not bool(result.get("accepted", False)):
            continue
        day_idx = action_start // MINUTES_PER_DAY
        if day_idx not in first_starts or action_start < first_starts[day_idx]:
            first_starts[day_idx] = action_start
    return first_starts


def _coerce_pos(value: Any) -> tuple[float, float] | None:
    try:
        if isinstance(value, dict):
            return float(value["lat"]), float(value["lng"])
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _position_from_record(record: dict[str, Any], key: str) -> tuple[float, float] | None:
    raw = record.get(key)
    if not isinstance(raw, dict):
        return None
    try:
        return (float(raw["lat"]), float(raw["lng"]))
    except (KeyError, TypeError, ValueError):
        return None


def _violates_geo_constraint(pos: tuple[float, float] | None, pref: ParsedPreference) -> bool:
    """Return True if pos violates an allowed/forbidden geo constraint."""
    if pos is None:
        return False
    if pref.geo_constraint_type == GEO_CONSTRAINT_ALLOWED_REGION:
        return not _inside_geo_region(pos, pref)
    if pref.geo_constraint_type == GEO_CONSTRAINT_FORBIDDEN_REGION:
        return _inside_any_geo_region(pos, pref)
    return False


def _cargo_matches_forbidden_endpoint_location(cargo: dict[str, Any], pref: ParsedPreference) -> tuple[bool, bool]:
    if not pref.forbidden_endpoint_locations:
        return False, False
    return _cargo_matches_endpoint_location(cargo, pref.forbidden_endpoint_locations)


def _cargo_matches_endpoint_location(cargo: dict[str, Any], locations: list[str]) -> tuple[bool, bool]:
    if not locations:
        return False, False
    aliases: set[str] = set()
    for loc in locations:
        aliases.update(_location_aliases(loc))
    if not aliases:
        return False, False
    pickup_blob = _endpoint_text_blob(cargo, "start")
    drop_blob = _endpoint_text_blob(cargo, "end")
    if not pickup_blob and not drop_blob:
        all_blob = " ".join(_string_values(cargo))
        return any(alias in all_blob for alias in aliases), any(alias in all_blob for alias in aliases)
    return (
        any(alias in pickup_blob for alias in aliases),
        any(alias in drop_blob for alias in aliases),
    )


def _is_virtual_cargo_id(cargo_id: Any) -> bool:
    return str(cargo_id or "").strip().startswith("__")


# Pickup obligation virtual prefix (defined in tools.py as VIRTUAL_PICKUP_PREFIX);
# mirrored here to avoid an import cycle. Commit uses ITINERARY_VIRTUAL_PREFIX.
_PICKUP_VIRTUAL_PREFIX = "__pickup_v_"


def _hop_breaks_fixed_window_rest(cargo_id: Any) -> bool:
    """A hop counts against a fixed-window rest only if the driver is actually
    out working/moving during it: real orders, plus the movement-bearing
    obligation virtuals (commit / pickup) — going to a stocktake/pickup inside
    0-6 forfeits that night's rest just like a real haul does. Stationary rest
    virtuals (__rest_v_ / __rest_window_v_ / __home_v_ / monthly rest) are
    exempt: they ARE the rest, not a violation of it."""
    cid = str(cargo_id or "").strip()
    if not _is_virtual_cargo_id(cid):
        return True
    return cid.startswith(ITINERARY_VIRTUAL_PREFIX) or cid.startswith(_PICKUP_VIRTUAL_PREFIX)


def _required_endpoint_location_day_bonus(pref: ParsedPreference) -> float:
    target = int(pref.required_endpoint_location_days or 0)
    if target <= 0:
        return 0.0
    if pref.penalty_cap is not None and pref.penalty_cap > 0:
        return effective_penalty_amount(pref, float(pref.penalty_cap) / float(target))
    return effective_penalty_amount(pref, pref.penalty_amount)


def _endpoint_cargo_from_info(info: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a cargo-like dict (start/end with city+address) from a
    cargo_info_by_id entry so endpoint-location matching has text to work with."""
    return {
        "start": {"city": info.get("start_city", ""), "address": info.get("start_address", "")},
        "end": {"city": info.get("end_city", ""), "address": info.get("end_address", "")},
    }


def _history_required_endpoint_location_days(
    parsed_prefs: list[ParsedPreference],
    history_records: list[dict[str, Any]],
    cargo_info_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    interesting = {
        idx: pref
        for idx, pref in enumerate(parsed_prefs)
        if pref.required_endpoint_locations and pref.required_endpoint_location_days
    }
    if not interesting:
        return out
    # History records carry NO cargo endpoint text (only cargo_id + accepted),
    # so matching the raw record never hits a region and the day-bonus would
    # never turn off. Resolve each accepted cargo's endpoint via cargo_info_by_id
    # (cargo_id -> start/end city) and match against that instead.
    info_by_id = cargo_info_by_id or {}
    for record, _step_start, action_start, _action_end, _step_end in _iter_history_with_time(history_records):
        action = record.get("action") if isinstance(record.get("action"), dict) else {}
        if str(action.get("action", "")).strip().lower() != "take_order":
            continue
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        if not bool(result.get("accepted", False)):
            continue
        cargo_id = str((action.get("params") or {}).get("cargo_id") or result.get("cargo_id") or "").strip()
        info = info_by_id.get(cargo_id)
        cargo_like = _endpoint_cargo_from_info(info) if isinstance(info, dict) else record
        day_idx = action_start // MINUTES_PER_DAY
        for pref_idx, pref in interesting.items():
            pickup_hit, drop_hit = _cargo_matches_endpoint_location(cargo_like, pref.required_endpoint_locations)
            if pickup_hit or drop_hit:
                out.setdefault(pref_idx, set()).add(day_idx)
    return out


def _geo_violation_where(pickup_bad: bool, drop_bad: bool) -> str:
    parts: list[str] = []
    if pickup_bad:
        parts.append("pickup")
    if drop_bad:
        parts.append("drop")
    return "+".join(parts) if parts else "unknown"


def _inside_geo_region(pos: tuple[float, float], pref: ParsedPreference) -> bool:
    has_region = bool(pref.geo_bbox or pref.geo_circle)
    if pref.geo_bbox:
        b = pref.geo_bbox
        try:
            lat = float(pos[0])
            lng = float(pos[1])
            if not (float(b["min_lat"]) <= lat <= float(b["max_lat"])):
                return False
            if not (float(b["min_lng"]) <= lng <= float(b["max_lng"])):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    if pref.geo_circle:
        if not _point_in_circle(pos, pref.geo_circle):
            return False
    return has_region


def _inside_any_geo_region(pos: tuple[float, float], pref: ParsedPreference) -> bool:
    if pref.geo_bbox:
        b = pref.geo_bbox
        try:
            lat = float(pos[0])
            lng = float(pos[1])
            if float(b["min_lat"]) <= lat <= float(b["max_lat"]) and float(b["min_lng"]) <= lng <= float(b["max_lng"]):
                return True
        except (KeyError, TypeError, ValueError):
            pass
    if pref.geo_circle and _point_in_circle(pos, pref.geo_circle):
        return True
    return False


def _prices_geo_region(pref: ParsedPreference) -> bool:
    return pref.geo_constraint_type in {GEO_CONSTRAINT_ALLOWED_REGION, GEO_CONSTRAINT_FORBIDDEN_REGION} and bool(
        pref.geo_bbox or pref.geo_circle
    )


def _point_in_circle(pos: tuple[float, float], circle: dict[str, Any]) -> bool:
    try:
        center = (float(circle["center_lat"]), float(circle["center_lng"]))
        radius_km = float(circle["radius_km"])
    except (KeyError, TypeError, ValueError):
        return False
    return _haversine_km(pos, center) <= radius_km


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(h)))


def _iter_history_with_time(records: list[dict[str, Any]]):
    sim_cursor = 0
    for record in records or []:
        step_elapsed = max(0, int(record.get("step_elapsed_minutes", 0) or 0))
        query_scan = max(0, int(record.get("query_scan_cost_minutes", 0) or 0))
        action_exec = max(0, int(record.get("action_exec_cost_minutes", 0) or 0))
        step_start = sim_cursor
        action_start = step_start + query_scan
        action_end = action_start + action_exec
        step_end = step_start + step_elapsed
        sim_cursor = step_end
        yield record, step_start, action_start, action_end, step_end


def _history_end_min(records: list[dict[str, Any]]) -> int:
    end_min = 0
    for _record, _step_start, _action_start, _action_end, step_end in _iter_history_with_time(records):
        end_min = step_end
    return end_min


def _continuous_wait_max_by_day(records: list[dict[str, Any]]) -> dict[int, int]:
    max_by_day: dict[int, int] = {}
    for record, _step_start, action_start, action_end, _step_end in _iter_history_with_time(records):
        action = record.get("action") if isinstance(record.get("action"), dict) else {}
        if str(action.get("action", "")).strip().lower() != "wait":
            continue
        cur = action_start
        while cur < action_end:
            day_idx = cur // MINUTES_PER_DAY
            day_end = (day_idx + 1) * MINUTES_PER_DAY
            seg_end = min(day_end, action_end)
            max_by_day[day_idx] = max(max_by_day.get(day_idx, 0), seg_end - cur)
            cur = seg_end
    return max_by_day


def _interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)
