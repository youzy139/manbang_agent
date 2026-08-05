"""验证 12 位司机的画像提取与规则引擎匹配情况。

运行方式（从仓库根目录）:
    python demo/verify_persona.py drivers.json

输出：
- 每位司机提取到的非空 schema 字段
- 全局字段覆盖统计
- 哪些 populated 字段没有对应 ParsedPreference 映射（匹配漏洞）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.driver_persona import DriverPersona
from agent.llm_persona_extractor import LLMPersonaConfig
from agent._persona_adapter import persona_to_parsed_preferences
from agent.preference_parser import ParsedPreference


SCHEMA_FIELDS = {
    "forbidden_hours",
    "max_pickup_deadhead_km",
    "max_haul_km",
    "max_deadhead_km",
    "cargo_avoidance",
    "monthly_rest_days",
    "daily_order_limit",
    "first_order_before",
    "geo_boundary",
    "forbidden_zone",
    "preferred_order_regions",
    "must_visit",
    "must_take",
    "special_events",
    "home_event",
    "known_locations",
    "fixed_stationary_window",
    "daily_continuous_rest_minutes",
    "monthly_kpi",
    "monthly_long_haul_cap",
    "driving_limits",
    "sequence_constraints",
    "activation_guard",
}

# ParsedPreference 字段 → 哪些 engine 模块会读它（按当前代码静态分析）
ENGINE_CONSUMPTION: dict[str, list[str]] = {
    "raw_content": ["loop._build_pref_status -> manager raw_text"],
    "clarified_text": ["loop._build_pref_status -> manager canonical_text"],
    "excluded_categories": ["plan_route.exclude_region (via loop)"],
    "required_categories": ["plan_route.required_cargo_categories (via loop)"],
    "required_endpoint_locations": ["_persona_adapter only (facts)"],
    "required_cargo_ids": ["loop._build_required_cargo_context?"],
    "itinerary_commitment": ["plan_route.inject_commitment_virtuals (defined but not called)"],
    "rest_type": ["loop._compute_pref_dims clock_rest"],
    "rest_continuous_hours": ["loop._compute_pref_dims clock_rest"],
    "rest_window_start_hour": ["loop._compute_pref_dims clock_rest"],
    "rest_window_end_hour": ["plan_route.inject_rest_virtuals (defined but not called)"],
    "rest_monthly_days": ["loop._compute_pref_dims clock_rest / plan_route.inject_rest_virtuals (not called)"],
    "first_order_deadline_hour": ["loop._build_pref_status facts"],
    "max_haul_km": ["loop._build_pref_status facts"],
    "max_pickup_deadhead_km": ["loop._build_pref_status facts"],
    "max_deadhead_km": ["loop._build_pref_status facts"],
    "forbidden_hours": ["loop._build_pref_status facts"],
    "aggregate_constraints": ["loop._compute_pref_dims daily_meter"],
    "driving_limits": ["loop._compute_pref_dims continuous_driving/daily_drive"],
    "sequence_constraints": ["loop._compute_pref_dims sequence"],
    "activation_guard": ["loop._compute_pref_dims daily_meter"],
    "geo_constraint_type": ["plan_route forbidden/allowed region filters (via loop)"],
    "geo_bbox": ["plan_route forbidden/allowed region filters"],
    "geo_circle": ["plan_route forbidden/allowed region filters"],
    "visit_target_days": ["plan_route visit target pricing"],
}


def _load_config() -> LLMPersonaConfig:
    cfg_path = Path("demo/server/config/config.json")
    if not cfg_path.is_file():
        return LLMPersonaConfig()
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    return LLMPersonaConfig(
        api_key=raw.get("model_api_key", ""),
        endpoint=raw.get("model_api_url", ""),
        model=raw.get("model_name", "deepseek-chat"),
        timeout=raw.get("model_timeout_seconds", 60),
    )


def _non_empty_fields(persona: dict) -> set[str]:
    out = set()
    for k, v in persona.items():
        if k not in SCHEMA_FIELDS:
            continue
        if v is None:
            continue
        if isinstance(v, (list, dict, str)) and not v:
            continue
        if isinstance(v, (int, float)) and v == 0:
            continue
        out.add(k)
    return out


def _persona_populated_fields(parsed_prefs: list[ParsedPreference]) -> set[str]:
    """从 ParsedPreference 列表反推哪些 schema 字段成功映射。"""
    populated: set[str] = set()
    for p in parsed_prefs:
        d = p.to_dict()
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, (list, dict, str)) and not v:
                continue
            if isinstance(v, (int, float)) and v == 0:
                continue
            # 把 ParsedPreference 字段名回映射到 schema 字段名
            if k == "excluded_categories" and v:
                populated.add("cargo_avoidance")
            elif k == "first_order_deadline_hour" and v is not None:
                populated.add("first_order_before")
            elif k == "rest_type" and v:
                if v == "fixed_window":
                    populated.add("fixed_stationary_window")
                elif v == "continuous_daily":
                    populated.add("daily_continuous_rest_minutes")
                elif v == "monthly_days":
                    populated.add("monthly_rest_days")
            elif k == "geo_constraint_type" and v:
                if v == "allowed_region":
                    populated.add("geo_boundary")
                elif v == "forbidden_region":
                    populated.add("forbidden_zone")
                elif v == "visit_target":
                    populated.add("must_visit")
            elif k == "required_endpoint_locations" and v:
                populated.add("preferred_order_regions")
            elif k == "required_cargo_ids" and v:
                populated.add("must_take")
            elif k == "itinerary_commitment" and v:
                for ev in v:
                    eid = str(ev.get("event_id") or "")
                    if eid == "home_event":
                        populated.add("home_event")
                    else:
                        populated.add("special_events")
            elif k == "aggregate_constraints" and v:
                for agg in v:
                    metric = str(agg.get("metric") or "")
                    if metric == "accepted_orders":
                        populated.add("daily_order_limit")
                    elif metric == "haul_minutes":
                        populated.add("monthly_long_haul_cap")
            elif k == "driving_limits" and v:
                populated.add("driving_limits")
            elif k == "sequence_constraints" and v:
                populated.add("sequence_constraints")
            elif k == "activation_guard" and v:
                populated.add("activation_guard")
            elif k in ENGINE_CONSUMPTION:
                populated.add(k)
    return populated


# known_locations 是元数据，不进入规则引擎，不算匹配漏洞
NON_RULE_FIELDS = {"known_locations"}


def main() -> None:
    drivers_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("drivers.json")
    raw = json.loads(drivers_path.read_text(encoding="utf-8"))
    config = _load_config()

    per_driver: dict[str, dict] = {}
    global_populated: dict[str, int] = {f: 0 for f in SCHEMA_FIELDS}

    for item in raw:
        driver_id = item.get("driver_id", "unknown")
        prefs = item.get("preferences", [])
        cost_per_km = float(item.get("cost_per_km", 1.5))
        print(f"\n=== {driver_id} ===")

        try:
            persona_obj = DriverPersona(
                driver_id=driver_id,
                raw_preferences=prefs,
                cost_per_km=cost_per_km,
                llm_config=config,
            )
            persona = persona_obj.to_dict(sparse=True)
            parsed_prefs = persona_to_parsed_preferences(prefs, persona)
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            per_driver[driver_id] = {"error": str(exc), "traceback": traceback.format_exc()}
            continue

        populated_schema = _non_empty_fields(persona)
        mapped_fields = _persona_populated_fields(parsed_prefs)
        unmapped = populated_schema - mapped_fields - NON_RULE_FIELDS

        per_driver[driver_id] = {
            "preferences_count": len(prefs),
            "populated_schema_fields": sorted(populated_schema),
            "mapped_to_parsedpreference": sorted(mapped_fields),
            "unmapped": sorted(unmapped),
            "parsed_prefs_count": len(parsed_prefs),
        }

        for f in populated_schema:
            global_populated[f] += 1

        print(f"  偏好数: {len(prefs)}")
        print(f"  提取到字段: {sorted(populated_schema)}")
        print(f"  映射到 ParsedPreference: {sorted(mapped_fields)}")
        if unmapped:
            print(f"  [WARN] unmapped: {sorted(unmapped)}")
        else:
            print(f"  [OK] all extracted fields mapped")

    print("\n=== 全局字段覆盖统计 ===")
    for f in sorted(SCHEMA_FIELDS):
        print(f"  {f}: {global_populated[f]}/{len(raw)} 位司机")

    print("\n=== 无司机覆盖的字段 ===")
    empty_fields = [f for f in sorted(SCHEMA_FIELDS) if global_populated[f] == 0]
    if empty_fields:
        print(f"  {empty_fields}")
    else:
        print("  无")

    out_path = Path("demo/verify_persona_result.json")
    out_path.write_text(json.dumps({
        "schema_fields": sorted(SCHEMA_FIELDS),
        "global_populated": global_populated,
        "per_driver": per_driver,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细结果已写入: {out_path}")


if __name__ == "__main__":
    main()