"""CargoMemory：司机级货源记忆模块。

存储司机历史 query_cargo 中见过的单个货源，按 cargo_id 去重。
在后续决策中，将记忆中仍有效（未下架/未过装货窗）的货源与当前 query 结果合并，
突破单次 100 条查询限制，减少因查询范围有限导致的好单遗漏。
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

SIM_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
WALL_FMT = "%Y-%m-%d %H:%M:%S"


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, l1 = math.radians(lat1), math.radians(lng1)
    p2, l2 = math.radians(lat2), math.radians(lng2)
    dp, dl = p2 - p1, l2 - l1
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * r * math.asin(math.sqrt(h))


def _wall_str_to_sim_min(text: str) -> int:
    dt = datetime.strptime(text.strip(), WALL_FMT)
    return int((dt - SIM_EPOCH).total_seconds() // 60)


class CargoMemory:
    """存储司机见过的单个货源，支持过期清理与按当前位置重新计算距离。"""

    def __init__(self, max_cargo: int = 2000) -> None:
        self._cargo: dict[str, dict[str, Any]] = {}
        self._max_cargo = max_cargo

    def observe(self, raw_items: list[dict[str, Any]]) -> None:
        """记录本次 query_cargo 返回的货源（按 cargo_id 去重保留最新）。"""
        for item in raw_items:
            cargo = item.get("cargo", {})
            if not cargo:
                continue
            cid = str(cargo.get("cargo_id", "")).strip()
            if not cid:
                continue
            self._cargo[cid] = dict(item)

        # 容量控制：按预估净收益保留最优
        if len(self._cargo) > self._max_cargo:
            def _net_score(it: dict[str, Any]) -> float:
                c = it.get("cargo", {})
                price = float(c.get("price", 0))
                dist = float(it.get("distance_km", 0))
                return price - dist * 1.5

            sorted_items = sorted(self._cargo.items(), key=lambda kv: _net_score(kv[1]), reverse=True)
            self._cargo = dict(sorted_items[:self._max_cargo])

    def _is_expired(self, item: dict[str, Any], current_minutes: int) -> bool:
        """判断货源是否已过接单窗口期。

        优先使用 load_time（装货窗口结束时间）判断；
        无 load_time 时兜底使用 remove_time（货源下架时间）。
        """
        cargo = item.get("cargo", {})

        load_time = cargo.get("load_time")
        if load_time is not None and isinstance(load_time, list) and len(load_time) >= 2:
            try:
                load_end = _wall_str_to_sim_min(str(load_time[1]).strip())
                if current_minutes > load_end:
                    return True
            except (ValueError, IndexError):
                pass

        remove_time = cargo.get("remove_time")
        if not remove_time:
            return False
        try:
            return current_minutes > _wall_str_to_sim_min(str(remove_time))
        except (ValueError, IndexError):
            return False

    def _update_distance(self, item: dict[str, Any], cur_lat: float, cur_lng: float) -> dict[str, Any]:
        """返回一份副本，其中 distance_km 按当前司机位置重新计算。"""
        item = dict(item)
        cargo = item.get("cargo", {})
        start = cargo.get("start", {})
        s_lat = float(start.get("lat", 0))
        s_lng = float(start.get("lng", 0))
        item["distance_km"] = round(_haversine_km(cur_lat, cur_lng, s_lat, s_lng), 2)
        return item

    def recall(
        self,
        current_lat: float,
        current_lng: float,
        current_minutes: int,
    ) -> list[dict[str, Any]]:
        """召回记忆中仍有效且未过期的货源，并按当前位置更新 distance_km。"""
        recalled: list[dict[str, Any]] = []
        expired_ids: list[str] = []

        for cid, item in self._cargo.items():
            if self._is_expired(item, current_minutes):
                expired_ids.append(cid)
                continue
            recalled.append(self._update_distance(item, current_lat, current_lng))

        for cid in expired_ids:
            del self._cargo[cid]

        return recalled

    @property
    def size(self) -> int:
        return len(self._cargo)

    def clear(self) -> None:
        self._cargo.clear()
