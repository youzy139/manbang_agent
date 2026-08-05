"""Persistent cargo graph used by graph-based decision planning."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from simkit.simulation_actions import distance_to_minutes, haversine_km, parse_cost_time_to_minutes

_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
_WALL_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


Position = tuple[float, float]


def wall_time_to_sim_min(text: str) -> int:
    """Convert a cargo wall-clock timestamp to minutes since the simulation epoch."""
    dt = datetime.strptime(str(text).strip(), _WALL_TIME_FORMAT)
    delta = dt - _SIMULATION_EPOCH
    return int(delta.total_seconds() // 60)


def sim_min_to_wall(sim_min: int) -> str:
    """Convert a simulation minute back to wall-clock text used by cargo fields."""
    from datetime import timedelta as _td
    dt = _SIMULATION_EPOCH + _td(minutes=int(sim_min))
    return dt.strftime(_WALL_TIME_FORMAT)


def _point_distance_km(a: Position, b: Position) -> float:
    return haversine_km(float(a[0]), float(a[1]), float(b[0]), float(b[1]))


def _travel_minutes(distance_km: float, speed_km_h: float) -> int:
    if distance_km < 1e-6:
        return 0
    return distance_to_minutes(distance_km, speed_km_h)


def _load_window_minutes(cargo: dict[str, Any]) -> tuple[int, int] | None:
    raw = cargo.get("load_time")
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"cargo load_time must be a 2-item list: {raw!r}")
    start_min = wall_time_to_sim_min(str(raw[0]))
    end_min = wall_time_to_sim_min(str(raw[1]))
    if end_min < start_min:
        raise ValueError(f"cargo load_time ends before it starts: {raw!r}")
    return start_min, end_min


def _validate_cargo(cargo: dict[str, Any]) -> None:
    _cargo_start(cargo)
    _cargo_end(cargo)
    wall_time_to_sim_min(str(cargo["remove_time"]))
    _load_window_minutes(cargo)
    parse_cost_time_to_minutes(cargo)


def _cargo_start(cargo: dict[str, Any]) -> Position:
    start = cargo["start"]
    return (float(start["lat"]), float(start["lng"]))


def _cargo_end(cargo: dict[str, Any]) -> Position:
    end = cargo["end"]
    return (float(end["lat"]), float(end["lng"]))


def _cargo_at_position(cargo: dict[str, Any], pos: Position) -> dict[str, Any]:
    cloned = dict(cargo)
    cloned["start"] = {"lat": float(pos[0]), "lng": float(pos[1])}
    cloned["end"] = {"lat": float(pos[0]), "lng": float(pos[1])}
    return cloned


def _parse_static_cargo(cargo: dict[str, Any]) -> dict[str, Any]:
    """Parse the query-invariant fields of a cargo once.

    ``feasible_edges_from`` runs at every DFS node of a plan_paths search and
    used to re-derive these on every call even though none depend on the query
    ``(pos, time_min)``: remove_time / load_time / cost_time parsing (each a
    ``datetime.strptime``), the ``_virtual_*`` key scan, and the static
    pickup/dropoff coords + haul distance. They are memoised on the graph entry
    (see ``feasible_edges_from``) keyed by cargo object identity, so a replaced
    cargo recomputes. ``ok=False`` mirrors the old inline try/except — the
    caller skips that cargo. Dynamic-position virtuals (rest_v / rest_window /
    monthly_rest) deliberately omit start/end/haul because those follow the
    runtime position and are filled in per call.
    """
    try:
        is_virtual_rest = bool(cargo.get("_virtual_rest"))
        is_virtual_cargo = any(str(key).startswith("_virtual_") for key in cargo.keys())
        is_dynamic = bool(
            is_virtual_rest
            or cargo.get("_virtual_rest_window")
            or cargo.get("_virtual_monthly_rest")
        )
        static: dict[str, Any] = {
            "ref": cargo,
            "ok": True,
            "is_virtual_rest": is_virtual_rest,
            "is_virtual_cargo": is_virtual_cargo,
            "is_dynamic": is_dynamic,
            "remove_min": wall_time_to_sim_min(str(cargo["remove_time"])),
            "window": _load_window_minutes(cargo),
            "cost_time_min": parse_cost_time_to_minutes(cargo),
            "price": float(cargo.get("price", 0.0)),
        }
        if not is_dynamic:
            start_pos = _cargo_start(cargo)
            end_pos = _cargo_end(cargo)
            static["start_pos"] = start_pos
            static["end_pos"] = end_pos
            static["haul_km"] = _point_distance_km(start_pos, end_pos)
        return static
    except (KeyError, TypeError, ValueError):
        return {"ref": cargo, "ok": False}


class CargoGraph:
    """Maintain a driver's observed cargo pool and derive feasible take-order hops.

    ``cleanup`` removes cargos whose ``remove_time`` is strictly earlier than the
    current simulation minute, cargos whose loading window has already ended,
    and malformed cargos. Reachability is checked per hop in ``feasible_edges_from``
    so multi-hop routes do not lose cargos that are unreachable directly from
    the driver's current position but could be reachable after an intermediate
    order.
    """

    def __init__(self) -> None:
        self._edges: dict[str, dict[str, Any]] = {}
        self._taken: set[str] = set()

    def add_observations(self, items: list[dict[str, Any]], current_time_min: int) -> dict[str, int]:
        """Merge ``query_cargo`` items into the graph and return merge stats."""
        stats = {"new_edges": 0, "duplicate": 0, "ignored_taken": 0, "invalid": 0}
        for item in items:
            if not isinstance(item, dict):
                stats["invalid"] += 1
                continue
            cargo = item.get("cargo")
            if not isinstance(cargo, dict):
                stats["invalid"] += 1
                continue
            cargo_id = str(cargo.get("cargo_id", "")).strip()
            if not cargo_id:
                stats["invalid"] += 1
                continue
            try:
                _validate_cargo(cargo)
            except (KeyError, TypeError, ValueError):
                stats["invalid"] += 1
                continue
            if cargo_id in self._taken:
                stats["ignored_taken"] += 1
                continue
            if cargo_id in self._edges:
                self._edges[cargo_id]["last_seen_min"] = int(current_time_min)
                stats["duplicate"] += 1
            else:
                self._edges[cargo_id] = {
                    "cargo": cargo,
                    "first_seen_min": int(current_time_min),
                    "last_seen_min": int(current_time_min),
                }
                stats["new_edges"] += 1
        return stats

    def mark_taken(self, cargo_id: str) -> None:
        """Mark a cargo as already selected so it cannot be recommended again."""
        cid = str(cargo_id).strip()
        if not cid:
            return
        self._edges.pop(cid, None)
        self._taken.add(cid)

    def cargo_validity_snapshot(self, cargo_id: str) -> dict[str, Any] | None:
        """诊断专用（observability，不改任何状态）：取某货当前在图里的有效性字段快照——装货窗 / remove_min /
        首末观测步。供 loop 在乐观 mark_taken 前留底，下一步若 server 判『已失效』据此定位 模型 vs server 失配
        真因（窗缺失 has_load_time=False？陈货 last_seen<now？窗刚过 load_end vs arrival？remove 已过？）。
        已删/未知 cargo → None。"""
        entry = self._edges.get(str(cargo_id).strip())
        if entry is None:
            return None
        cargo = entry.get("cargo") or {}
        try:
            window = _load_window_minutes(cargo)
        except (KeyError, TypeError, ValueError):
            window = None
        try:
            remove_min = wall_time_to_sim_min(str(cargo.get("remove_time")))
        except (KeyError, TypeError, ValueError):
            remove_min = None
        return {
            "has_load_time": cargo.get("load_time") is not None,
            "load_window": list(window) if window is not None else None,
            "remove_min": remove_min,
            "first_seen_min": entry.get("first_seen_min"),
            "last_seen_min": entry.get("last_seen_min"),
        }

    def cleanup(self, current_time_min: int, current_pos: Position, speed_km_h: float) -> int:
        """Remove expired, stale-window, or malformed cargos and return the removed count."""
        to_remove: list[str] = []
        for cargo_id, entry in self._edges.items():
            cargo = entry["cargo"]
            try:
                remove_min = wall_time_to_sim_min(str(cargo["remove_time"]))
                window = _load_window_minutes(cargo)
                _cargo_start(cargo)
            except (KeyError, TypeError, ValueError):
                to_remove.append(cargo_id)
                continue
            if remove_min < int(current_time_min):
                to_remove.append(cargo_id)
                continue

            if window is None:
                continue
            _, load_end_min = window
            if load_end_min < int(current_time_min):
                to_remove.append(cargo_id)

        for cargo_id in to_remove:
            del self._edges[cargo_id]
        return len(to_remove)

    def feasible_edges_from(
        self,
        pos: Position,
        time_min: int,
        speed_km_h: float,
        cost_per_km: float,
        horizon_min: int,
        scan_margin_min: int = 0,
        cargo_yield_delta_fn: "Callable[[dict[str, Any], float, float, int], float] | None" = None,
    ) -> list[dict[str, Any]]:
        """Return feasible take-order hops from ``pos`` at ``time_min``.

        A hop is excluded when pickup arrival is strictly after ``load_time`` end
        or when a real cargo's projected finish minute reaches/exceeds
        ``horizon_min``.

        ``cargo_yield_delta_fn(cargo, deadhead_km, haul_km, finish_min) -> float`` (optional):
        a策略场 cargo_modifier hook. When provided, its delta is FOLDED INTO each
        REAL cargo's ``net_yield`` *during* the search, so a偏好-boosted-but-low-base
        cargo (e.g. a quota品类 the driver is behind on) survives into the candidate
        pool instead of being pruned before the post-hoc modifier could ever see it
        (the original bug: modifiers re-ranked an already-pruned top-k → 0 命中 for
        low-yield目标货). ``None`` → byte-identical to the pre-hook behavior (the
        agent_lixian copy and the MD5-pinned region-value tests call without it).

        ``scan_margin_min`` is added BEFORE the deadhead when computing arrival: each
        decision step first spends a fixed ``query_cargo`` scan (~15 min) before the
        vehicle can move, so the real pickup arrival is ``time + scan + deadhead``.
        Folding it in here keeps reachability honest — without it a far / marginal cargo
        whose window closes within the scan window looks feasible at planning but its
        load window has expired by the time the vehicle actually arrives (the observed
        "空驶 87km 去接、到了过期" failed take that stranded D001). Default 0 keeps the
        old behavior for callers that do not pass it (the agent_lixian copy is left as-is).

        The simulation month is treated as a half-open interval
        ``[0, horizon_min)``: minute ``horizon_min`` is already 2026-04-01
        00:00 for a 31-day March run, so orders finishing there are not
        income-eligible for March. Synthetic preference/compliance cargos are
        exempt because some of them intentionally lock until the month boundary
        to satisfy a no-order day.
        """
        result: list[dict[str, Any]] = []
        time_min_i = int(time_min)
        horizon_i = int(horizon_min)
        for cargo_id, entry in self._edges.items():
            cargo = entry["cargo"]
            # Query-invariant fields are parsed once and memoised on the entry,
            # keyed by cargo identity so a replaced cargo recomputes. This is the
            # hot path: a single plan_paths DFS calls this for every node.
            static = entry.get("_static")
            if static is None or static.get("ref") is not cargo:
                static = _parse_static_cargo(cargo)
                entry["_static"] = static
            if not static["ok"]:
                continue
            if static["remove_min"] < time_min_i:
                continue

            is_dynamic_position_virtual = static["is_dynamic"]
            is_virtual_cargo = static["is_virtual_cargo"]
            is_virtual_rest = static["is_virtual_rest"]
            cost_time_min = static["cost_time_min"]
            window = static["window"]
            if is_dynamic_position_virtual:
                start_pos = pos
                end_pos = pos
                haul_km = 0.0
            else:
                start_pos = static["start_pos"]
                end_pos = static["end_pos"]
                haul_km = static["haul_km"]

            deadhead_km = _point_distance_km(pos, start_pos)
            deadhead_min = _travel_minutes(deadhead_km, speed_km_h)
            # +scan_margin: the per-step query_cargo scan runs before the vehicle moves,
            # so the real arrival is time + scan + deadhead (see method docstring).
            arrival_min = time_min_i + int(scan_margin_min) + deadhead_min

            # remove_time 也按**到达时刻**判可达：remove 是该货从在线池撤走的时刻，司机必须在到达装货前它仍未撤。
            # 上面 static["remove_min"] < time_min_i 只滤掉"决策时已撤"的，漏掉"决策时在池、但 now+scan+deadhead
            # 到达时已撤"的——这正是实测 8/8『已失效』的成因(remove_min∈[now, arrival]：server 在 15min 扫描后那刻
            # 判货已撤)。与上面的 load_end 窗检查同口径(都用 arrival)。real 货才查;虚拟单 remove 由 manager 控、不受此限。
            if not is_virtual_cargo and static["remove_min"] < arrival_min:
                continue

            if window is not None:
                load_start_min, load_end_min = window
                if arrival_min > load_end_min:
                    continue
                ready_min = max(arrival_min, load_start_min)
            else:
                ready_min = arrival_min
            load_wait_min = ready_min - arrival_min

            finish_min = ready_min + cost_time_min
            if finish_min >= horizon_i and not is_virtual_cargo:
                continue

            price = static["price"]
            if is_virtual_rest and finish_min // 1440 != time_min_i // 1440:
                price = 0.0
            cost = float(cost_per_km) * (deadhead_km + haul_km)
            net_yield = price - cost
            # 策略场 cargo_modifier：把偏好奖惩折进真实货 net_yield（**搜索期**，非选完再叠加）。
            # 这样被升值的低基价目标货（落后的配额品类）能存活进候选池；禁品类/区域的负 delta
            # 也在此驱负、随 drop_negative 被剔。虚拟货（rest/waypoint/commit）不计。fn=None → 不变。
            if cargo_yield_delta_fn is not None and not is_virtual_cargo:
                try:
                    net_yield += float(cargo_yield_delta_fn(cargo, deadhead_km, haul_km, finish_min) or 0.0)
                except Exception:  # noqa: BLE001 — 谓词/系数异常绝不打断搜索
                    pass
            total_time_min = deadhead_min + load_wait_min + cost_time_min
            yield_per_min = net_yield / total_time_min if total_time_min > 0 else 0.0

            result.append(
                {
                    "cargo_id": cargo_id,
                    "cargo": _cargo_at_position(cargo, pos) if is_dynamic_position_virtual else cargo,
                    "deadhead_min": deadhead_min,
                    "deadhead_km": deadhead_km,
                    "load_wait_min": load_wait_min,
                    "cost_time_min": cost_time_min,
                    "total_time_min": total_time_min,
                    "action_start_min": int(time_min),
                    "arrival_min": arrival_min,
                    "finish_min": finish_min,
                    "haul_km": haul_km,
                    "price": price,
                    "cost": cost,
                    "net_yield": net_yield,
                    "yield_per_min": yield_per_min,
                    "end_pos": end_pos,
                }
            )
        return result

    def size(self) -> int:
        """Return the number of available graph edges."""
        return len(self._edges)

    def taken_count(self) -> int:
        """Return the number of cargos selected by this driver's graph policy."""
        return len(self._taken)

    def size_within_radius(self, pos: Position, radius_km: float) -> int:
        """Count available cargo pickups within ``radius_km`` of ``pos``."""
        return sum(
            1
            for entry in self._edges.values()
            if _point_distance_km(pos, _cargo_start(entry["cargo"])) <= float(radius_km)
        )

    def minutes_since_last_observation(
        self,
        pos: Position,
        radius_km: float,
        current_time_min: int,
    ) -> int | None:
        """Return minutes since the freshest observation in a local region."""
        relevant = [
            int(entry["last_seen_min"])
            for entry in self._edges.values()
            if _point_distance_km(pos, _cargo_start(entry["cargo"])) <= float(radius_km)
        ]
        if not relevant:
            return None
        return int(current_time_min) - max(relevant)

    def iter_cargos(self) -> list[dict[str, Any]]:
        """Return a snapshot of graph cargo dictionaries for planner statistics."""
        return [entry["cargo"] for entry in self._edges.values()]


def cargo_haul_km(cargo: dict[str, Any]) -> float:
    """Return the haul distance for a cargo dictionary."""
    return _point_distance_km(_cargo_start(cargo), _cargo_end(cargo))
