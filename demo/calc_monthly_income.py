"""计算何师傅（D001）仿真收益与偏好罚分。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT / "demo") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "demo"))

from simkit.simulation_actions import haversine_km

_DRIVER_ID = "D001"
_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
_REPOSITION_SPEED_KM_PER_HOUR = 60.0

_APRIL_FRUIT_MIN_ORDERS = 12
_MAY_JIANCAI_MIN_ORDERS = 12
_LONG_HAUL_MINUTES = 8 * 60
_LONG_HAUL_MONTHLY_CAP = 5


def _parse_epoch_minutes(ts: str) -> int:
    return int((_SIMULATION_EPOCH.fromisoformat(ts.strip().replace(" ", "T")) - _SIMULATION_EPOCH).total_seconds() // 60)


@dataclass(frozen=True)
class PreferenceRuleSpec:
    content: str
    start_minutes: int
    end_minutes: int
    penalty_amount: float
    penalty_cap: float | None


def _resolve_config_json(server_config_dir: Path) -> Path:
    primary = server_config_dir / "config.json"
    if primary.is_file():
        return primary
    fallback = server_config_dir / "config.example.json"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"缺少 server 配置: {primary} 或 {fallback}")


def load_reposition_speed_km_per_hour(config_path: Path) -> float:
    value = json.loads(config_path.read_text(encoding="utf-8")).get("reposition_speed_km_per_hour")
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"{config_path.name} 缺少有效 reposition_speed_km_per_hour")
    return float(value)


def load_drivers_path(config_path: Path, server_root: Path) -> Path:
    rel = json.loads(config_path.read_text(encoding="utf-8")).get("drivers_path")
    if not rel or not isinstance(rel, str):
        raise ValueError(f"{config_path.name} 缺少有效 drivers_path")
    path = Path(rel)
    return (path if path.is_absolute() else server_root / path).resolve()


def load_cargo_map(path: Path) -> dict[str, dict[str, Any]]:
    cargo_map: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cargo_id = str(item.get("cargo_id", "")).strip()
            if not cargo_id:
                continue
            start, end = item.get("start", {}), item.get("end", {})
            distance_km = haversine_km(float(start["lat"]), float(start["lng"]), float(end["lat"]), float(end["lng"]))
            load_start_minutes: int | None = None
            load_end_minutes: int | None = None
            load_window = item.get("load_time")
            if isinstance(load_window, list) and len(load_window) == 2:
                load_start_minutes = _parse_epoch_minutes(str(load_window[0]))
                load_end_minutes = _parse_epoch_minutes(str(load_window[1]))
                if load_end_minutes < load_start_minutes:
                    load_start_minutes = load_end_minutes = None
            cargo_map[cargo_id] = {
                "price": float(item.get("price", 0.0)) / 100.0,
                "distance_km": distance_km,
                "create_minutes": _parse_epoch_minutes(str(item["create_time"])),
                "remove_minutes": _parse_epoch_minutes(str(item["remove_time"])),
                "start_lat": float(start["lat"]),
                "start_lng": float(start["lng"]),
                "end_lat": float(end["lat"]),
                "end_lng": float(end["lng"]),
                "cost_time_minutes": int(item.get("cost_time_minutes", 0) or 0),
                "load_start_minutes": load_start_minutes,
                "load_end_minutes": load_end_minutes,
                "cargo_name": str(item.get("cargo_name", "") or "").strip(),
            }
    return cargo_map


def load_driver_profile(path: Path) -> tuple[float, list[PreferenceRuleSpec]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 1:
        raise ValueError(f"demo 仅支持 1 位司机，当前 drivers.json 条目数: {len(raw) if isinstance(raw, list) else '非法'}")
    item = raw[0]
    driver_id = str(item.get("driver_id", "")).strip()
    if driver_id != _DRIVER_ID:
        raise ValueError(f"期望司机 {_DRIVER_ID}，当前为 {driver_id or '空'}")
    rules: list[PreferenceRuleSpec] = []
    for entry in item.get("preferences") or []:
        if not isinstance(entry, dict):
            continue
        content = str(entry.get("content", "")).strip()
        if not content:
            continue
        cap_raw = entry.get("penalty_cap")
        rules.append(
            PreferenceRuleSpec(
                content=content,
                start_minutes=_parse_epoch_minutes(str(entry.get("start_time", "2026-03-01 00:00:00"))),
                end_minutes=_parse_epoch_minutes(str(entry.get("end_time", "2026-05-31 23:59:59"))),
                penalty_amount=float(entry.get("penalty_amount", 0.0) or 0.0),
                penalty_cap=None if cap_raw is None else float(cap_raw),
            )
        )
    return float(item.get("cost_per_km", 0.0)), rules


def find_latest_result_file(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob(f"actions_202603_{_DRIVER_ID}_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"未找到 {_DRIVER_ID} 仿真结果: {results_dir}/actions_202603_{_DRIVER_ID}_*.jsonl")
    return candidates[-1]


def load_simulation_duration_days(path: Path) -> int:
    value = json.loads(path.read_text(encoding="utf-8")).get("simulation_duration_days")
    if not isinstance(value, int) or value <= 0:
        raise ValueError("run_summary_202603.json 缺少有效 simulation_duration_days")
    return value


def load_simulate_time_seconds(path: Path) -> float | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get("simulate_time_seconds")
    return round(float(value), 2) if isinstance(value, (int, float)) else None


def _nearly_equal(a: float, b: float, eps: float = 1e-4) -> bool:
    return abs(float(a) - float(b)) <= eps


def _distance_minutes(distance_km: float, speed_km_per_hour: float) -> int:
    if distance_km <= 0:
        return 1
    return max(1, int(math.ceil((distance_km / speed_km_per_hour) * 60.0)))


def _interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def _ctx_overlaps_rule(ctx: dict[str, Any], rule: PreferenceRuleSpec) -> bool:
    return _interval_overlap(ctx["action_start"], ctx["action_end"], rule.start_minutes, rule.end_minutes + 1)


def _append_rule(detail_rules: list[dict[str, Any]], rule_label: str, penalty: float, rule: PreferenceRuleSpec, **extra: Any) -> None:
    detail_rules.append({"rule": rule_label, "penalty": round(penalty, 2), "preference_text": rule.content, **extra})


def _is_weekend_day(day: int) -> bool:
    return (_SIMULATION_EPOCH + timedelta(days=day)).weekday() >= 5


def _build_step_contexts(file_path: Path) -> list[dict[str, Any]]:
    ctxs: list[dict[str, Any]] = []
    prev_end = 0
    with file_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue
            record = json.loads(row)
            query_scan = int(record["query_scan_cost_minutes"])
            action_exec = int(record["action_exec_cost_minutes"])
            result = record.get("result", {})
            end_minutes = int(result["simulation_progress_minutes"])
            action_obj = record.get("action") or {}
            pos_before = record.get("position_before") or {}
            pos_after = record.get("position_after") or {}
            ctxs.append(
                {
                    "line_no": line_no,
                    "action_name": str(action_obj.get("action", "")).strip().lower(),
                    "params": action_obj.get("params") or {},
                    "result": result if isinstance(result, dict) else {},
                    "step_start": prev_end,
                    "action_start": prev_end + query_scan,
                    "action_end": prev_end + query_scan + action_exec,
                    "step_end": end_minutes,
                    "action_exec_cost": action_exec,
                }
            )
            prev_end = end_minutes
    return ctxs


def _eval_night_rest_with_weekend(
    ctxs: list[dict[str, Any]],
    days: list[int],
    rule: PreferenceRuleSpec,
    label: str,
    detail_rules: list[dict[str, Any]],
) -> float:
    violations = 0
    for day in days:
        day_start = day * 1440
        if not _interval_overlap(day_start, day_start + 1440, rule.start_minutes, rule.end_minutes + 1):
            continue
        rest_start_hour = 23 if _is_weekend_day(day) else 21
        window_start = day * 1440 + rest_start_hour * 60
        window_end = (day + 1) * 1440 + 6 * 60
        window_minutes = window_end - window_start

        active_violation = any(
            ctx["action_name"] in {"take_order", "reposition"}
            and _interval_overlap(ctx["action_start"], ctx["action_end"], window_start, window_end)
            for ctx in ctxs
        )
        wait_intervals: list[tuple[int, int]] = []
        for ctx in ctxs:
            if ctx["action_name"] != "wait" or ctx["action_exec_cost"] <= 0:
                continue
            s, e = max(ctx["step_start"], window_start), min(ctx["step_end"], window_end)
            if e > s:
                wait_intervals.append((s, e))
        rest_ok = sum(e - s for s, e in _merge_intervals(wait_intervals)) >= window_minutes
        if active_violation or not rest_ok:
            violations += 1

    penalty = violations * rule.penalty_amount
    _append_rule(detail_rules, label, penalty, rule, violations=violations)
    return penalty


def _count_category_orders(
    ctxs: list[dict[str, Any]],
    cargo_map: dict[str, dict[str, Any]],
    category: str,
    month: int,
    rule: PreferenceRuleSpec,
) -> int:
    count = 0
    for ctx in ctxs:
        if ctx["action_name"] != "take_order" or not bool(ctx["result"].get("accepted", False)):
            continue
        if not _ctx_overlaps_rule(ctx, rule):
            continue
        action_day = ctx["action_start"] // 1440
        if (_SIMULATION_EPOCH + timedelta(days=action_day)).month != month:
            continue
        cargo = cargo_map.get(str((ctx["params"] or {}).get("cargo_id", "")).strip())
        if cargo is not None and str(cargo.get("cargo_name", "")) == category:
            count += 1
    return count


def _eval_monthly_long_haul_cap(
    ctxs: list[dict[str, Any]],
    cargo_map: dict[str, dict[str, Any]],
    min_minutes: int,
    max_per_month: int,
    rule: PreferenceRuleSpec,
    label: str,
    detail_rules: list[dict[str, Any]],
) -> float:
    by_month: dict[int, int] = {}
    for ctx in ctxs:
        if ctx["action_name"] != "take_order" or not bool(ctx["result"].get("accepted", False)):
            continue
        if not _ctx_overlaps_rule(ctx, rule):
            continue
        cargo = cargo_map.get(str((ctx["params"] or {}).get("cargo_id", "")).strip())
        if cargo is None or int(cargo["cost_time_minutes"]) <= min_minutes:
            continue
        month = (_SIMULATION_EPOCH + timedelta(days=ctx["action_start"] // 1440)).month
        by_month[month] = by_month.get(month, 0) + 1

    violations = sum(max(0, count - max_per_month) for count in by_month.values())
    penalty = violations * rule.penalty_amount
    _append_rule(detail_rules, label, penalty, rule, violations=violations, by_month=by_month)
    return penalty


def evaluate_d001_preferences(
    ctxs: list[dict[str, Any]],
    cargo_map: dict[str, dict[str, Any]],
    rules: list[PreferenceRuleSpec],
    simulation_duration_days: int,
) -> tuple[float, dict[str, Any]]:
    if len(rules) != 4:
        raise ValueError(f"D001 需要 4 条偏好，当前 {len(rules)}")
    detail: list[dict[str, Any]] = []
    total = 0.0
    all_days = list(range(simulation_duration_days))
    night_rule, april_rule, may_rule, long_haul_rule = rules

    total += _eval_night_rest_with_weekend(ctxs, all_days, night_rule, "夜间停车休息", detail)

    april_fruit_orders = _count_category_orders(ctxs, cargo_map, "水果", 4, april_rule)
    april_shortfall = max(0, _APRIL_FRUIT_MIN_ORDERS - april_fruit_orders)
    april_penalty = april_shortfall * april_rule.penalty_amount
    total += april_penalty
    _append_rule(
        detail,
        "四月水果≥12单",
        april_penalty,
        april_rule,
        orders=april_fruit_orders,
        shortfall=april_shortfall,
    )

    may_fruit_makeup_orders = _count_category_orders(ctxs, cargo_map, "水果", 5, may_rule)
    remaining_april_shortfall = max(0, april_shortfall - may_fruit_makeup_orders)
    may_jiancai_orders = _count_category_orders(ctxs, cargo_map, "建材", 5, may_rule)
    may_shortfall = max(0, _MAY_JIANCAI_MIN_ORDERS - may_jiancai_orders)
    may_violations = remaining_april_shortfall + may_shortfall
    may_penalty = may_violations * may_rule.penalty_amount
    total += may_penalty
    _append_rule(
        detail,
        "五月建材指标及四月欠额",
        may_penalty,
        may_rule,
        april_shortfall=april_shortfall,
        may_fruit_makeup_orders=may_fruit_makeup_orders,
        remaining_april_shortfall=remaining_april_shortfall,
        may_jiancai_orders=may_jiancai_orders,
        may_shortfall=may_shortfall,
        violations=may_violations,
    )

    total += _eval_monthly_long_haul_cap(
        ctxs,
        cargo_map,
        _LONG_HAUL_MINUTES,
        _LONG_HAUL_MONTHLY_CAP,
        long_haul_rule,
        "月度长途>8h≤5单",
        detail,
    )
    return round(total, 2), {"rules": detail}


def _validate_and_compute_income(
    file_path: Path,
    cargo_map: dict[str, dict[str, Any]],
    cost_per_km: float,
    reposition_speed_km_per_hour: float,
    simulation_horizon_minutes: int,
) -> tuple[dict[str, float], dict[str, int]]:
    income = {"gross_income": 0.0, "distance_km": 0.0, "cost": 0.0, "net_income": 0.0}
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    prev_end_minutes = 0
    with file_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue
            record = json.loads(row)
            action_name = str(record.get("action", {}).get("action", "")).strip().lower()
            if action_name not in {"wait", "reposition", "take_order"}:
                raise ValueError(f"{file_path.name} 第 {line_no} 行 action 非法: {action_name}")
            result = record.get("result", {})
            params = record.get("action", {}).get("params", {})
            raw_usage = record.get("token_usage", {})
            if isinstance(raw_usage, dict):
                for key in token_usage:
                    token_usage[key] += int(raw_usage.get(key, 0))
            end_minutes = int(result["simulation_progress_minutes"])
            step_elapsed = int(record["step_elapsed_minutes"])
            query_scan = int(record["query_scan_cost_minutes"])
            action_exec = int(record["action_exec_cost_minutes"])
            if end_minutes - prev_end_minutes != step_elapsed:
                raise ValueError(f"{file_path.name} 第 {line_no} 行时间推进不一致")
            action_start = prev_end_minutes + query_scan
            before = record.get("position_before", {})
            after = record.get("position_after", {})
            before_lat, before_lng = float(before["lat"]), float(before["lng"])
            after_lat, after_lng = float(after["lat"]), float(after["lng"])
            if action_name == "wait":
                if action_exec != int((params or {}).get("duration_minutes", 1)):
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 wait 时间不一致")
                if not _nearly_equal(before_lat, after_lat) or not _nearly_equal(before_lng, after_lng):
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 wait 不应改变位置")
            elif action_name == "reposition":
                target_lat, target_lng = float(params["latitude"]), float(params["longitude"])
                expected_km = haversine_km(before_lat, before_lng, target_lat, target_lng)
                if action_exec != _distance_minutes(expected_km, reposition_speed_km_per_hour):
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 reposition 时间不一致")
                income["distance_km"] += float(result.get("distance_km", 0.0))
            elif action_name == "take_order":
                cargo_id = str((params or {}).get("cargo_id", "")).strip()
                cargo = cargo_map.get(cargo_id)
                if cargo is None:
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 cargo_id 不存在")
                if bool(result.get("accepted", False)):
                    if not (int(cargo["create_minutes"]) <= action_start <= int(cargo["remove_minutes"])):
                        raise ValueError(f"{file_path.name} 第 {line_no} 行接单时点不在货源有效期")
                    pickup_km = haversine_km(before_lat, before_lng, float(cargo["start_lat"]), float(cargo["start_lng"]))
                    pickup_minutes = _distance_minutes(pickup_km, reposition_speed_km_per_hour) if pickup_km > 1e-6 else 0
                    arrival = action_start + pickup_minutes
                    wait_minutes = 0
                    if isinstance(cargo.get("load_start_minutes"), int) and isinstance(cargo.get("load_end_minutes"), int):
                        if arrival > int(cargo["load_end_minutes"]):
                            raise ValueError(f"{file_path.name} 第 {line_no} 行成功接单但已超装货时间窗")
                        wait_minutes = max(0, int(cargo["load_start_minutes"]) - arrival)
                    if action_exec != pickup_minutes + wait_minutes + int(cargo["cost_time_minutes"]):
                        raise ValueError(f"{file_path.name} 第 {line_no} 行接单耗时不一致")
                    if end_minutes <= simulation_horizon_minutes:
                        income["gross_income"] += float(cargo["price"])
                    income["distance_km"] += float(result.get("pickup_deadhead_km", 0.0) or 0.0) + float(
                        result.get("haul_distance_km", 0.0) or cargo["distance_km"]
                    )
            prev_end_minutes = end_minutes
    income["cost"] = income["distance_km"] * cost_per_km
    income["net_income"] = income["gross_income"] - income["cost"]
    for key in ("gross_income", "distance_km", "cost", "net_income"):
        income[key] = round(float(income[key]), 2)
    return income, token_usage


def main(
    *,
    results_dir: Path,
    data_dir: Path | None = None,
    project_root: Path | None = None,
    reposition_speed_km_per_hour: float | None = None,
) -> None:
    layout_root = (project_root or _SCRIPT_DIR).resolve()
    results_dir = results_dir.resolve()
    server_root = layout_root / "server"
    if data_dir is not None:
        cargo_dataset = data_dir / "cargo_dataset.jsonl"
        drivers_dataset = data_dir / "drivers.json"
        speed = float(reposition_speed_km_per_hour or _REPOSITION_SPEED_KM_PER_HOUR)
    else:
        config_path = _resolve_config_json(server_root / "config")
        cargo_dataset = server_root / "data" / "cargo_dataset.jsonl"
        drivers_dataset = load_drivers_path(config_path, server_root)
        speed = float(reposition_speed_km_per_hour or load_reposition_speed_km_per_hour(config_path))

    output_file = results_dir / "monthly_income_202603.json"
    run_summary_file = results_dir / "run_summary_202603.json"
    if not cargo_dataset.is_file():
        raise FileNotFoundError(f"缺少货源数据: {cargo_dataset}")
    if not drivers_dataset.is_file():
        raise FileNotFoundError(f"缺少司机数据: {drivers_dataset}")

    cargo_map = load_cargo_map(cargo_dataset)
    cost_per_km, preference_rules = load_driver_profile(drivers_dataset)
    simulation_duration_days = load_simulation_duration_days(run_summary_file)
    result_file = find_latest_result_file(results_dir)
    horizon = simulation_duration_days * 1440

    validation_errors: dict[str, str] = {}
    preference_check: dict[str, Any] = {"rules": []}
    income = {"gross_income": 0.0, "distance_km": 0.0, "cost": 0.0, "preference_penalty": 0.0, "net_income": 0.0}
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    try:
        income, token_usage = _validate_and_compute_income(result_file, cargo_map, cost_per_km, speed, horizon)
        ctxs = _build_step_contexts(result_file)
        penalty, preference_check = evaluate_d001_preferences(ctxs, cargo_map, preference_rules, simulation_duration_days)
        income["preference_penalty"] = round(penalty, 2)
        income["net_income"] = round(income["net_income"] - penalty, 2)
    except Exception as exc:
        validation_errors[_DRIVER_ID] = f"{type(exc).__name__}: {exc}"

    drivers = [
        {
            "driver_id": _DRIVER_ID,
            "income": income,
            "token_usage": token_usage,
            "calculation_aborted": _DRIVER_ID in validation_errors,
            "validation_error": validation_errors.get(_DRIVER_ID),
            "preference_check": preference_check,
        }
    ]
    payload = {
        "month": "2026-03",
        "simulate_time_seconds": load_simulate_time_seconds(run_summary_file),
        "result_files_count": 1,
        "drivers": drivers,
        "summary": {
            "total_net_income_all_drivers": round(float(income["net_income"]), 2),
            "total_preference_penalty": round(float(income.get("preference_penalty", 0.0)), 2),
            "total_token_usage": token_usage,
            "failed_driver_count": len(validation_errors),
            "failed_drivers": validation_errors,
        },
        "cost_meaning": "cost = distance_km * cost_per_km",
        "cost_metric": "net_income = gross_income - cost - preference_penalty",
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算何师傅（D001）仿真收益")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--reposition-speed", type=float, default=None)
    args = parser.parse_args()
    layout_root = (args.project_root or _SCRIPT_DIR).resolve()
    results_dir = (args.results_dir or layout_root / "results").resolve()
    main(
        results_dir=results_dir,
        data_dir=args.data_dir,
        project_root=layout_root,
        reposition_speed_km_per_hour=args.reposition_speed,
    )
