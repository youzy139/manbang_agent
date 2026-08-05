"""把当前 DriverPersona 输出对接成新规则引擎的 ParsedPreference 列表。

当前画像提取器把全部偏好合并成一颗 persona dict；新引擎期望每项偏好一个 ParsedPreference。
本适配器按约束类型拆成多个 ParsedPreference，保留原文摘要与结构化 facts，供 Virtual Manager 消费。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from agent._helpers import _parse_dt_flexible, _wall_str_to_sim_min
from agent.preference_parser import (
    GEO_CONSTRAINT_FORBIDDEN_REGION,
    GEO_CONSTRAINT_VISIT_TARGET,
    ParsedPreference,
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _penalty(rule: dict[str, Any]) -> float:
    return _as_float(rule.get("penalty_amount", rule.get("penalty", 0)), 0.0)


def _cap(rule: dict[str, Any]) -> float | None:
    cap = rule.get("penalty_cap", rule.get("cap"))
    if cap is None or cap == "":
        return None
    try:
        return float(cap)
    except (TypeError, ValueError):
        return None


def _wall_to_min(text: Any) -> int | None:
    if text is None:
        return None
    try:
        return _wall_str_to_sim_min(str(text))
    except Exception:
        return _parse_dt_flexible(str(text))


def _hour_from_text(t: Any) -> int | None:
    if t is None:
        return None
    text = str(t)
    if ":" in text:
        try:
            return int(text.split(":")[0])
        except (ValueError, IndexError):
            return None
    return _as_int(t)


def _minute_from_text(t: Any) -> int | None:
    if t is None:
        return None
    text = str(t)
    if ":" in text:
        parts = text.split(":")
        try:
            return int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            return 0
    return _as_int(t) or 0


def persona_to_parsed_preferences(
    raw_preferences: list[Any],
    persona: dict[str, Any],
) -> list[ParsedPreference]:
    """把 DriverPersona.to_dict() 转成 ParsedPreference 列表。"""
    out: list[ParsedPreference] = []

    base_text = "; ".join(
        str(p.get("content", p) if isinstance(p, dict) else p)
        for p in raw_preferences
    )[:800]

    # 1. 品类回避
    avoided: set[str] = set()
    for name in persona.get("cargo_avoidance", []) or []:
        if isinstance(name, str) and name.strip():
            avoided.add(name.strip())
    if avoided:
        out.append(ParsedPreference(
            raw_content=f"{base_text}"[:400],
            clarified_text="品类回避: " + ", ".join(sorted(avoided)),
            excluded_categories=sorted(avoided),
        ))

    # 2. 距离上限
    for rule in persona.get("max_haul_km", []) or []:
        km = _as_float(rule.get("max_km"), 0.0)
        if km > 0:
            out.append(ParsedPreference(
                raw_content=str(rule.get("raw", "单笔运输距离上限")),
                max_haul_km=km,
                penalty_amount=_penalty(rule),
                penalty_cap=_cap(rule),
            ))
    for rule in persona.get("max_pickup_deadhead_km", []) or []:
        km = _as_float(rule.get("max_km"), 0.0)
        if km > 0:
            out.append(ParsedPreference(
                raw_content=str(rule.get("raw", "赴装空驶距离上限")),
                max_pickup_deadhead_km=km,
                penalty_amount=_penalty(rule),
                penalty_cap=_cap(rule),
            ))
    for rule in persona.get("max_deadhead_km", []) or []:
        km = _as_float(rule.get("max_km"), 0.0)
        if km > 0:
            out.append(ParsedPreference(
                raw_content=str(rule.get("raw", "月度空驶里程上限")),
                max_deadhead_km=km,
                penalty_amount=_penalty(rule),
                penalty_cap=_cap(rule),
            ))

    # 3. 禁行时段
    for rule in persona.get("forbidden_hours", []) or []:
        start_h = _as_int(rule.get("start_hour"))
        end_h = _as_int(rule.get("end_hour"))
        if start_h is None or end_h is None:
            continue
        out.append(ParsedPreference(
            raw_content=str(rule.get("raw", "禁行时段")),
            forbidden_hours=[{
                "start_hour": start_h,
                "start_min": _as_int(rule.get("start_min"), 0) or 0,
                "end_hour": end_h,
                "end_min": _as_int(rule.get("end_min"), 0) or 0,
                "no_order": bool(rule.get("no_order", True)),
                "no_reposition": bool(rule.get("no_reposition", False)),
            }],
            penalty_amount=_penalty(rule),
            penalty_cap=_cap(rule),
        ))

    # 3. 固定休息窗口
    fsw = persona.get("fixed_stationary_window")
    if isinstance(fsw, dict):
        for key, label in (("weekdays", "工作日"), ("weekends", "周末")):
            cfg = fsw.get(key)
            if not isinstance(cfg, dict):
                continue
            start = _hour_from_text(cfg.get("start"))
            end = _hour_from_text(cfg.get("end"))
            minutes = _as_int(cfg.get("minutes"), 0) or 0
            if start is None or end is None:
                continue
            cross = cfg.get("cross_day", start > end)
            out.append(ParsedPreference(
                raw_content=f"{label}固定休息窗口 {start}:00-{end}:00",
                rest_type="fixed_window",
                rest_window_start_hour=start,
                rest_window_end_hour=end,
                rest_window_crosses_midnight=bool(cross),
            ))

    # 4. 每日连续休息
    dcrm = persona.get("daily_continuous_rest_minutes")
    if dcrm:
        hours = _as_float(dcrm, 0.0) / 60.0
        if hours > 0:
            out.append(ParsedPreference(
                raw_content=f"每日连续休息 {hours} 小时",
                rest_type="continuous_daily",
                rest_continuous_hours=hours,
            ))

    # 5. 月度休息天数
    for rule in persona.get("monthly_rest_days", []) or []:
        days = _as_int(rule.get("min_days"))
        if days:
            out.append(ParsedPreference(
                raw_content=str(rule.get("raw", "月度休息天数")),
                rest_type="monthly_days",
                rest_monthly_days=days,
                penalty_amount=_penalty(rule),
                penalty_cap=_cap(rule),
            ))

    # 6. 禁入区域 / 活动区域边界
    for rule in persona.get("forbidden_zone", []) or []:
        lat = _as_float(rule.get("center_lat"), None)
        lng = _as_float(rule.get("center_lng"), None)
        r = _as_float(rule.get("radius_km"), 0.0)
        if lat is not None and lng is not None and r > 0:
            out.append(ParsedPreference(
                raw_content=str(rule.get("raw", "禁入区域")),
                geo_constraint_type=GEO_CONSTRAINT_FORBIDDEN_REGION,
                geo_circle={"center_lat": lat, "center_lng": lng, "radius_km": r},
                penalty_amount=_penalty(rule),
                penalty_cap=_cap(rule),
            ))
    for rule in persona.get("geo_boundary", []) or []:
        lat_min = _as_float(rule.get("lat_min"), None)
        lat_max = _as_float(rule.get("lat_max"), None)
        lng_min = _as_float(rule.get("lng_min"), None)
        lng_max = _as_float(rule.get("lng_max"), None)
        if None not in (lat_min, lat_max, lng_min, lng_max):
            out.append(ParsedPreference(
                raw_content=str(rule.get("raw", "活动区域边界")),
                geo_constraint_type="allowed_region",
                geo_bbox={"min_lat": lat_min, "max_lat": lat_max,
                          "min_lng": lng_min, "max_lng": lng_max},
                penalty_amount=_penalty(rule),
                penalty_cap=_cap(rule),
            ))

    # 7. 偏好接货区域（到访/端点落区）
    for rule in persona.get("preferred_order_regions", []) or []:
        keyword = str(rule.get("region_keyword") or "").strip()
        min_days = _as_int(rule.get("min_days"))
        if keyword:
            out.append(ParsedPreference(
                raw_content=str(rule.get("raw", f"接货区域 {keyword}")),
                required_endpoint_locations=[keyword],
                required_endpoint_location_days=min_days,
                penalty_amount=_penalty(rule),
                penalty_cap=_cap(rule),
            ))

    # 8. must_visit
    for rule in persona.get("must_visit", []) or []:
        lat = _as_float(rule.get("target_lat"), None)
        lng = _as_float(rule.get("target_lng"), None)
        r = _as_float(rule.get("radius_km"), 1.0)
        days = _as_int(rule.get("min_days"))
        if lat is not None and lng is not None:
            out.append(ParsedPreference(
                raw_content=str(rule.get("raw", "必须到访")),
                geo_constraint_type=GEO_CONSTRAINT_VISIT_TARGET,
                geo_circle={"center_lat": lat, "center_lng": lng, "radius_km": r},
                visit_target_days=days,
                penalty_amount=_penalty(rule),
                penalty_cap=_cap(rule),
            ))

    # 9. 熟货源 / 指定货源
    for rule in persona.get("must_take", []) or []:
        cid = str(rule.get("cargo_id") or "").strip()
        lat = _as_float(rule.get("pickup_lat"), None)
        lng = _as_float(rule.get("pickup_lng"), None)
        if cid:
            out.append(ParsedPreference(
                raw_content=str(rule.get("raw", f"必须接 {cid}")),
                required_cargo_ids=[cid],
                required_cargo_pickup=({"lat": lat, "lng": lng}
                                       if lat is not None and lng is not None else None),
                required_cargo_release_time_min=_wall_to_min(rule.get("available_at")),
                required_cargo_deadline_time_min=_wall_to_min(rule.get("end_time")),
                penalty_amount=_penalty(rule),
                penalty_cap=_cap(rule),
            ))

    # 10. 特殊事件 / 家事事件 -> itinerary_commitment
    events: list[dict[str, Any]] = []
    for rule in persona.get("special_events", []) or []:
        ev = _special_event_to_itinerary(rule)
        if ev:
            events.append(ev)
    for rule in persona.get("home_event", []) or []:
        ev = _home_event_to_itinerary(rule)
        if ev:
            events.append(ev)
    if events:
        out.append(ParsedPreference(
            raw_content="特殊/家事事件安排",
            itinerary_commitment=events,
        ))

    # 11. 月度 KPI
    kpi = persona.get("monthly_kpi")
    if isinstance(kpi, dict):
        kpi = [kpi]
    if isinstance(kpi, list):
        for item in kpi:
            if not isinstance(item, dict):
                continue
            cat = str(item.get("category") or "").strip()
            target = _as_int(item.get("target_count"))
            month = _as_int(item.get("month"))
            if cat and target:
                pp = ParsedPreference(
                    raw_content=f"月度KPI: {cat} {target} 单" + (f"(月份{month})" if month else ""),
                    required_categories=[cat],
                    penalty_amount=_as_float(item.get("penalty_amount", 0), 0.0),
                    penalty_cap=_cap(item),
                )
                if month:
                    start = f"2026-{month:02d}-01 00:00:00"
                    end_day = 31
                    try:
                        import calendar
                        end_day = calendar.monthrange(2026, month)[1]
                    except Exception:
                        pass
                    end = f"2026-{month:02d}-{end_day:02d} 23:59:59"
                    pp = replace(pp,
                                 active_start_min=_wall_to_min(start),
                                 active_end_min=_wall_to_min(end))
                out.append(pp)

    # 12. 首单截止时间
    for rule in persona.get("first_order_before", []) or []:
        h = _as_int(rule.get("hour"))
        if h is not None:
            out.append(ParsedPreference(
                raw_content=str(rule.get("raw", "首单截止")),
                first_order_deadline_hour=h,
                penalty_amount=_penalty(rule),
                penalty_cap=_cap(rule),
            ))

    # 13. 每日最多接单
    for rule in persona.get("daily_order_limit", []) or []:
        n = _as_int(rule.get("max_orders"))
        if n:
            out.append(ParsedPreference(
                raw_content=str(rule.get("raw", "每日接单上限")),
                aggregate_constraints=[{
                    "metric": "accepted_orders",
                    "window": "natural_day",
                    "aggregate": "sum",
                    "op": ">",
                    "value": n,
                }],
                penalty_amount=_penalty(rule),
                penalty_cap=_cap(rule),
            ))

    # 14. 月度长途上限
    mlhc = persona.get("monthly_long_haul_cap")
    if isinstance(mlhc, dict):
        max_hours = _as_float(mlhc.get("max_hours"), 0.0)
        max_orders = _as_int(mlhc.get("max_orders"))
        if max_hours > 0 and max_orders:
            out.append(ParsedPreference(
                raw_content=f"月度长途上限: >{max_hours}h 最多 {max_orders} 单",
                aggregate_constraints=[{
                    "metric": "haul_minutes",
                    "window": "whole_month",
                    "aggregate": "count_above",
                    "op": ">",
                    "value": max_hours * 60,
                    "n_orders": max_orders,
                }],
            ))

    # 15. 驾驶时限（规则引擎直接消费的 deterministic 字段）
    dlim = persona.get("driving_limits")
    if isinstance(dlim, dict):
        spec: dict[str, Any] = {}
        cont = _as_int(dlim.get("max_continuous_drive_min"))
        brk = _as_int(dlim.get("required_break_min"))
        daily = _as_int(dlim.get("max_daily_drive_min"))
        if cont is not None:
            spec["max_continuous_drive_min"] = cont
            spec["required_break_min"] = brk if brk is not None else 1
        if daily is not None:
            spec["max_daily_drive_min"] = daily
        if spec:
            out.append(ParsedPreference(
                raw_content=str(dlim.get("raw", "驾驶时限")),
                driving_limits=spec,
                penalty_amount=_penalty(dlim),
                penalty_cap=_cap(dlim),
            ))

    # 16. 序列约束（规则引擎直接消费的 deterministic 字段）
    seq_rules = persona.get("sequence_constraints") or []
    valid_seq: list[dict[str, Any]] = []
    for rule in seq_rules:
        if not isinstance(rule, dict):
            continue
        relation = str(rule.get("relation") or "").strip()
        if not relation:
            continue
        spec = {"relation": relation}
        if relation == "adjacency_implication":
            ant = rule.get("antecedent")
            con = rule.get("consequent")
            if isinstance(ant, dict) and isinstance(con, dict):
                spec["antecedent"] = ant
                spec["consequent"] = con
            else:
                continue
        elif relation == "adjacency_distinct":
            spec["distinct_key"] = str(rule.get("distinct_key") or "category").strip() or "category"
        elif relation == "max_consecutive_same":
            key = str(rule.get("distinct_key") or "category").strip() or "category"
            max_run = _as_int(rule.get("max_run"))
            if max_run is None:
                continue
            spec["distinct_key"] = key
            spec["max_run"] = max_run
            cat = rule.get("category")
            if cat is not None and str(cat).strip():
                spec["category"] = str(cat).strip()
        elif relation == "window_quota":
            ant = rule.get("antecedent") or rule.get("predicate")
            n = _as_int(rule.get("window_n"))
            cmp = str(rule.get("comparator") or "").strip()
            value = _as_float(rule.get("value"), None)
            if not isinstance(ant, dict) or n is None or n <= 0 or not cmp or value is None:
                continue
            spec["antecedent"] = ant
            spec["window_n"] = n
            spec["comparator"] = cmp
            spec["value"] = value
        else:
            continue
        if str(rule.get("penalty_fn") or "") == "all_or_nothing":
            spec["penalty_fn"] = "all_or_nothing"
        valid_seq.append(spec)
    if valid_seq:
        out.append(ParsedPreference(
            raw_content="序列约束: " + "; ".join(str(r.get("relation")) for r in valid_seq),
            sequence_constraints=valid_seq,
        ))

    # 17. 激活门（规则引擎直接消费的 deterministic 字段）
    guard = persona.get("activation_guard")
    normalized_guard = _normalize_guard_spec(guard)
    if normalized_guard:
        out.append(ParsedPreference(
            raw_content="激活门条件",
            activation_guard=normalized_guard,
        ))

    # 兜底：如果什么都没抽出来，至少把原文包进去让 manager 自己读
    if not out:
        out.append(ParsedPreference(raw_content=base_text))

    return out


def _normalize_guard_spec(guard: Any) -> dict[str, Any] | None:
    """归一化激活门：支持叶子 {metric, op, value} 或复合 {"all":[...]}/{"any":[...]}/{"not":guard}。"""
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
    op = str(guard.get("op") or guard.get("operator") or "").strip()
    value = guard.get("value")
    if not metric or not op or value is None:
        return None
    return {"metric": metric, "op": op, "value": value}


def _special_event_to_itinerary(rule: dict[str, Any]) -> dict[str, Any] | None:
    month = _as_int(rule.get("month"))
    day = _as_int(rule.get("day"))
    lat = _as_float(rule.get("target_lat"), None)
    lng = _as_float(rule.get("target_lng"), None)
    if month is None or day is None or lat is None or lng is None:
        return None
    try:
        import calendar
        from datetime import datetime
        year = 2026
        if day > calendar.monthrange(year, month)[1]:
            return None
        base = datetime(year, month, day)
    except Exception:
        return None
    dwell = _as_int(rule.get("wait_minutes"), 120) or 120
    deadline = rule.get("secondary_deadline_hour")
    deadline_min = None
    if deadline is not None:
        try:
            deadline_min = int((base.replace(hour=int(deadline))).timestamp() // 60) - int(datetime(2026, 3, 1).timestamp() // 60)
        except Exception:
            pass
    return {
        "event_id": str(rule.get("primary_name") or rule.get("description") or f"evt_{month}_{day}"),
        "type": str(rule.get("type") or "stay"),
        "lat": lat,
        "lng": lng,
        "dwell_min": dwell,
        "must_complete_before_min": deadline_min,
        "until_min": deadline_min,
    }


def _home_event_to_itinerary(rule: dict[str, Any]) -> dict[str, Any] | None:
    spouse_lat = _as_float(rule.get("spouse_lat"), None)
    spouse_lng = _as_float(rule.get("spouse_lng"), None)
    if spouse_lat is None or spouse_lng is None:
        return None
    deadline = _wall_to_min(rule.get("deadline"))
    event_end = _wall_to_min(rule.get("event_end"))
    return {
        "event_id": "home_event",
        "type": "visit",
        "lat": spouse_lat,
        "lng": spouse_lng,
        "dwell_min": _as_int(rule.get("stay_minutes"), 10) or 10,
        "must_complete_before_min": deadline,
        "until_min": event_end,
    }