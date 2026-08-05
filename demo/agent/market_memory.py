"""MarketMemory：司机私有市场记忆模块。

网格参数：
- GRID_DEG = 0.2°（约 22km × 22km）
- BUCKET_HOURS = 1（hour_of_day 0-23，跨天聚合解决稀疏性）

核心机制：价值扩散——每条货源的全局净利润（price - haul_cost）
从装货地向周围网格扩散，邻居价值 = 全局净利润 - 行驶成本。
查询时做 3×3 邻域平滑，邻居价值扣除到网格边界的行驶成本。
"""

from __future__ import annotations

import heapq
import math
import re
from collections import defaultdict
from typing import Any

GRID_DEG = 0.2
BUCKET_HOURS = 1
_MAX_PER_CELL = 500
_TRIM_THRESHOLD = 700
_DECAY_HALF_LIFE_MINUTES = 720  # 12h -> 37%，24h -> 13.5%


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, l1 = math.radians(lat1), math.radians(lng1)
    p2, l2 = math.radians(lat2), math.radians(lng2)
    dp, dl = p2 - p1, l2 - l1
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * r * math.asin(math.sqrt(h))


def _decay_factor(delta_minutes: int) -> float:
    if delta_minutes <= 0:
        return 1.0
    return math.exp(-delta_minutes / _DECAY_HALF_LIFE_MINUTES)


def _maybe_trim(cell: dict[str, list[tuple[float, int]]], current_minutes: int) -> None:
    for key in ("nets", "adjusted_nets"):
        arr = cell.get(key, [])
        if len(arr) > _TRIM_THRESHOLD:
            cell[key] = heapq.nlargest(
                _MAX_PER_CELL, arr, key=lambda x: x[0] * _decay_factor(current_minutes - x[1])
            )


def _grid(lat: float, lng: float) -> tuple[int, int]:
    return (math.floor(lat / GRID_DEG), math.floor(lng / GRID_DEG))


def _grid_center(gx: int, gy: int) -> tuple[float, float]:
    return (gx * GRID_DEG + GRID_DEG / 2, gy * GRID_DEG + GRID_DEG / 2)


def _hour(minutes: int) -> int:
    return (minutes % 1440) // (BUCKET_HOURS * 60)


class MarketMemory:
    """司机级市场记忆。只记录该司机实际 query 到的货源信息。

    内部结构：
        _raw[(gx, gy, hour)] = {
            "nets": [(value, obs_time), ...],
            "adjusted_nets": [(adj_value, obs_time), ...],
        }
    """

    def __init__(self) -> None:
        self._raw: dict[
            tuple[int, int, int], dict[str, list[tuple[float, int]]]
        ] = defaultdict(lambda: {"nets": [], "adjusted_nets": []})
        self._cost_per_km: float = 1.5

    def set_cost_per_km(self, cost_per_km: float) -> None:
        self._cost_per_km = float(cost_per_km)

    # ── 写入 ──────────────────────────────────────────────

    def observe_query_result(
        self,
        items: list[dict[str, Any]],
        current_minutes: int,
        cost_per_km: float,
    ) -> None:
        """记录一次 query_cargo 的结果，做价值扩散。

        1. 计算订单全局净利润 gross_net = price - haul_cost（不含 pickup）
        2. 以装货地为中心向周围网格扩散：邻居价值 = gross_net - 行驶成本
        3. 扩散半径 = min(gross_net / cost_per_km / km_per_cell, 15)
        """
        if not items:
            return

        hour = _hour(current_minutes)
        km_per_cell = GRID_DEG * 111.0
        touched: set[tuple[int, int, int]] = set()

        for item in items:
            cargo = item.get("cargo", {})
            if not cargo:
                continue
            start = cargo.get("start", {})
            if not start:
                continue

            s_lat = float(start.get("lat", 0))
            s_lng = float(start.get("lng", 0))
            price = float(cargo.get("price", 0))

            end = cargo.get("end", {})
            e_lat = float(end.get("lat", s_lat))
            e_lng = float(end.get("lng", s_lng))
            haul_km = _haversine_km(s_lat, s_lng, e_lat, e_lng)
            haul_cost = haul_km * cost_per_km

            gross_net = price - haul_cost
            if gross_net <= 0:
                continue

            profit_radius = int(gross_net / cost_per_km / km_per_cell) + 1
            max_radius = min(profit_radius, 5)

            a_gx, a_gy = _grid(s_lat, s_lng)

            for dx in range(-max_radius, max_radius + 1):
                for dy in range(-max_radius, max_radius + 1):
                    ngx, ngy = a_gx + dx, a_gy + dy
                    n_lat, n_lng = _grid_center(ngx, ngy)
                    travel_km = _haversine_km(n_lat, n_lng, s_lat, s_lng)
                    travel_cost = travel_km * cost_per_km
                    diffused_net = gross_net - travel_cost
                    if diffused_net > 0:
                        key = (ngx, ngy, hour)
                        self._raw[key]["nets"].append((diffused_net, current_minutes))
                        touched.add(key)

        for key in touched:
            _maybe_trim(self._raw[key], current_minutes)

    def observe_filtered_cargo(
        self,
        filtered_cargo: list[dict[str, Any]],
        current_minutes: int,
    ) -> None:
        """记录预过滤后司机真正可接的货源，按装货地聚合扣除罚金后的 adjusted_net。"""
        if not filtered_cargo:
            return

        hour = _hour(current_minutes)

        for c in filtered_cargo:
            s_lat = float(c.get("start_lat", 0))
            s_lng = float(c.get("start_lng", 0))
            gx, gy = _grid(s_lat, s_lng)

            net = float(c.get("estimated_net_profit", 0))

            penalty_total = 0.0
            for flag in c.get("preference_flags", []):
                m = re.search(r"罚(\d+(?:\.\d+)?)元", str(flag))
                if m:
                    penalty_total += float(m.group(1))

            adjusted_net = net - penalty_total
            if adjusted_net > 0:
                key = (gx, gy, hour)
                self._raw[key]["adjusted_nets"].append((adjusted_net, current_minutes))
                _maybe_trim(self._raw[key], current_minutes)

    def observe_completed_order(
        self,
        pickup_lat: float,
        pickup_lng: float,
        dest_lat: float,
        dest_lng: float,
        actual_net_income: float,
        completion_minutes: int,
        cost_per_km: float,
    ) -> None:
        """订单完成后，把实际净收益作为高置信度观测扩散到装货地周围网格。

        这相当于对 MarketMemory 做简化版 TD 校准：用真实收益修正装货地价值估计。
        存入 adjusted_nets（权重高于普通 query 观测），并通过价值扩散传播到邻居。
        """
        if actual_net_income <= 0:
            return

        hour = _hour(completion_minutes)
        km_per_cell = GRID_DEG * 111.0
        profit_radius = int(actual_net_income / max(cost_per_km, 0.1) / km_per_cell) + 1
        max_radius = min(profit_radius, 3)  # 比 query 扩散更保守，避免过拟合

        a_gx, a_gy = _grid(pickup_lat, pickup_lng)
        touched: set[tuple[int, int, int]] = set()

        for dx in range(-max_radius, max_radius + 1):
            for dy in range(-max_radius, max_radius + 1):
                ngx, ngy = a_gx + dx, a_gy + dy
                n_lat, n_lng = _grid_center(ngx, ngy)
                travel_km = _haversine_km(n_lat, n_lng, pickup_lat, pickup_lng)
                travel_cost = travel_km * cost_per_km
                diffused_net = actual_net_income - travel_cost
                if diffused_net > 0:
                    key = (ngx, ngy, hour)
                    # 存储两次以提高权重（相当于该观测更可信）
                    self._raw[key]["adjusted_nets"].append((diffused_net, completion_minutes))
                    self._raw[key]["adjusted_nets"].append((diffused_net, completion_minutes))
                    touched.add(key)

        for key in touched:
            _maybe_trim(self._raw[key], completion_minutes)

    # ── 查询（实时 3×3 邻域平滑） ──────────────────────────

    def _cell_metric(
        self,
        gx: int,
        gy: int,
        hour: int,
        metric: str,
        current_minutes: int,
    ) -> float | None:
        cell = self._raw.get((gx, gy, hour))
        if cell is None:
            return None

        raw_items = cell["adjusted_nets"] or cell["nets"]
        if not raw_items:
            return None

        nets: list[float] = []
        for value, obs_minutes in raw_items:
            decayed = value * _decay_factor(current_minutes - obs_minutes)
            if decayed > 0:
                nets.append(decayed)

        if not nets:
            return None

        if len(nets) > _MAX_PER_CELL:
            nets = heapq.nlargest(_MAX_PER_CELL, nets)

        if metric in ("avg", "avg_adjusted_net"):
            return sum(nets) / len(nets)
        if metric in ("max", "max_adjusted_net"):
            return max(nets)
        if metric == "top30_avg":
            s = sorted(nets)
            idx = int(len(s) * 0.70)
            return sum(s[idx:]) / len(s[idx:])
        if metric == "median":
            s = sorted(nets)
            return s[len(s) // 2]
        if metric == "count":
            return float(len(nets))
        return None

    def _smoothed_value(
        self,
        lat: float,
        lng: float,
        gx: int,
        gy: int,
        hour: int,
        metric: str = "avg",
        current_minutes: int = 0,
    ) -> float | None:
        neighbors: list[tuple[float, int]] = []

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cell = self._raw.get((gx + dx, gy + dy, hour))
                if cell is None:
                    continue
                raw_items = cell["adjusted_nets"] or cell["nets"]
                if not raw_items:
                    continue
                v = self._cell_metric(gx + dx, gy + dy, hour, metric, current_minutes)
                if v is None:
                    continue

                ngx, ngy = gx + dx, gy + dy
                lat_min = ngx * GRID_DEG
                lat_max = (ngx + 1) * GRID_DEG
                lng_min = ngy * GRID_DEG
                lng_max = (ngy + 1) * GRID_DEG
                nearest_lat = max(lat_min, min(lat, lat_max))
                nearest_lng = max(lng_min, min(lng, lng_max))
                dist = _haversine_km(lat, lng, nearest_lat, nearest_lng)
                travel_cost = dist * self._cost_per_km
                adjusted_v = v - travel_cost
                neighbors.append((adjusted_v, len(raw_items)))

        if not neighbors:
            return None

        TEMP = 5.0
        raw_weights = [n[1] for n in neighbors]
        max_w = max(raw_weights)
        exp_weights = [math.exp((w - max_w) / TEMP) for w in raw_weights]
        sum_exp = sum(exp_weights)
        weights = [w / sum_exp for w in exp_weights]

        vals = [n[0] for n in neighbors]
        return sum(v * w for v, w in zip(vals, weights))

    def query(
        self,
        lat: float,
        lng: float,
        minutes: int,
        metric: str = "top30_avg",
        density_discount_denominator: float | None = None,
    ) -> float | None:
        gx, gy = _grid(lat, lng)
        hour = _hour(minutes)

        internal_metric = "top30_avg"
        if metric in ("max_adjusted_net", "max_net", "max"):
            internal_metric = "max"
        elif metric in ("avg_adjusted_net", "avg_net", "avg", "blended"):
            internal_metric = "avg"

        return self._smoothed_value(lat, lng, gx, gy, hour, internal_metric, minutes)

    def query_stats(
        self,
        lat: float,
        lng: float,
        minutes: int,
        metric: str = "top30_avg",
    ) -> dict[str, Any] | None:
        gx, gy = _grid(lat, lng)
        hour = _hour(minutes)

        internal_metric = "top30_avg"
        if metric in ("max_adjusted_net", "max_net", "max"):
            internal_metric = "max"
        elif metric in ("avg_adjusted_net", "avg_net", "avg", "blended"):
            internal_metric = "avg"

        cell = self._raw.get((gx, gy, hour))
        if cell is None:
            return None
        raw_items = cell["adjusted_nets"] or cell["nets"]
        if not raw_items:
            return None

        effective_count = sum(
            1 for v, t in raw_items if v * _decay_factor(minutes - t) > 0
        )

        v_raw = self._cell_metric(gx, gy, hour, internal_metric, minutes)
        if v_raw is None:
            return None

        v_smooth = self._smoothed_value(lat, lng, gx, gy, hour, internal_metric, minutes)
        if v_smooth is None:
            v_smooth = v_raw

        n = len(raw_items)
        credibility = min(1.0, effective_count / 50.0)

        return {
            "value": round(v_smooth, 2),
            "raw_value": round(v_raw, 2),
            "total_weight": n,
            "sample_count": n,
            "credibility": round(credibility, 2),
        }

    def has_data(self, lat: float, lng: float, minutes: int) -> bool:
        return self.query(lat, lng, minutes) is not None

    @property
    def size(self) -> int:
        return sum(
            len(v["nets"]) + len(v["adjusted_nets"])
            for v in self._raw.values()
        )
