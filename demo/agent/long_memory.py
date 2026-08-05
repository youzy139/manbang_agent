"""每个司机一份的"长期区域记忆"。

只通过 SimulationApiPort.query_cargo 返回的货源观测填充, 不读原始数据文件。
按 0.3° lat/lng 网格 (~33 km) 划分 cell, 每 cell 记录两套累积量:

- 衰减版 (4 小时半衰期 EMA): 反映近期热度
- 不衰减版 (long-term): 反映长期富矿事实, 避免久未观测就被判死

读取时按"信心权重"合成两套, 信心 = 1/(1 + age/half_life), 久未观测时 long-term
权重升高。region_value 公式:

    region_value = combined_avg_yield_per_min × log(1 + count_total)

线性正比于质量, 对数正比于密度, 避免"100 条 1 元 = 10 条 10 元"陷阱。
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .cargo_graph import wall_time_to_sim_min
from simkit.simulation_actions import haversine_km, parse_cost_time_to_minutes

LM_TRACE_LOGGER_NAME = "agent.long_memory.trace"
_lm_trace_handler_installed = False


def get_lm_trace_logger() -> logging.Logger:
    """Return the dedicated logger for LongMemory traces.

    On first call we attach a FileHandler writing to ``logs/long_memory_trace.log``
    under the project root ``logs/`` directory. Set the
    env var ``AGENT_LM_TRACE_FILE=0`` to skip the file handler (lines still go
    through the standard logging chain).
    """
    global _lm_trace_handler_installed
    logger = logging.getLogger(LM_TRACE_LOGGER_NAME)
    if _lm_trace_handler_installed:
        return logger
    _lm_trace_handler_installed = True
    if os.environ.get("AGENT_LM_TRACE_FILE", "1").strip() in {"0", "false", "False"}:
        return logger
    if logger.handlers:
        return logger
    try:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / "long_memory_trace.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    except OSError:
        pass
    return logger

CELL_SIZE_DEG = 0.2
HALF_LIFE_MIN = 240.0
LONG_HALF_LIFE_MIN = 5760.0  # 4 days
MAX_CELLS = 2000
MAX_CATEGORIES_PER_CELL = 20
MAX_OUTBOUND_PER_CELL = 10
MAX_LOAD_WINDOWS_PER_CELL = 50
MAX_SEEN_CARGO_IDS = 50000
REGION_QUERY_RADIUS_KM = 75.0
FLOW_VALUE_WEIGHT = 0.3
FLOW_CONFIDENCE_SAMPLE = 5.0
EXPLORATION_BONUS_CAP = 0.25
EXPLORATION_ALPHA_MAX = 0.30
EXPLORATION_ALPHA_MIN = 0.02
EXPLORATION_DECAY_MIN = 7 * 1440.0
MIN_CELLS_FOR_EXPLORATION = 5
EXPLORATORY_PRIOR_NORM = 0.05
# (已删 SOURCE_CONFIDENCE_TARGET_DRIVERS：跨司机共享池的置信曲线已移除，区域价值每司机独立、不共享)
BUCKET_COUNT = 4
BUCKET_WIDTH_MIN = 1440 // BUCKET_COUNT
BUCKET_SHRINKAGE_KAPPA = 5.0
# Tile size for the spatial index. Must be larger than CELL_SIZE_DEG so cells
# fit into discrete tiles, and large enough that a 3x3 tile neighbourhood
# strictly covers REGION_QUERY_RADIUS_KM. 1.0deg ~ 111km at the equator and
# ~102km at lat=23 (Pearl River Delta), comfortably above the 75km radius.
TILE_SIZE_DEG = 1.0


def _bucket_index(sim_min: int) -> int:
    return max(0, min(BUCKET_COUNT - 1, (int(sim_min) % 1440) // BUCKET_WIDTH_MIN))


def _tile_key(lat: float, lng: float) -> tuple[int, int]:
    return int(math.floor(lat / TILE_SIZE_DEG)), int(math.floor(lng / TILE_SIZE_DEG))


@dataclass
class CellStat:
    last_seen_min: int = 0
    count_recent: float = 0.0
    price_sum_recent: float = 0.0
    haul_km_sum_recent: float = 0.0
    cost_time_sum_recent: float = 0.0
    count_total: float = 0.0
    price_sum_total: float = 0.0
    haul_km_sum_total: float = 0.0
    cost_time_sum_total: float = 0.0
    categories: dict[str, int] = field(default_factory=dict)
    outbound: dict[tuple[int, int], float] = field(default_factory=dict)
    load_window_widths: list[int] = field(default_factory=list)
    load_lead_times: list[int] = field(default_factory=list)
    source_driver_ids: set[str] = field(default_factory=set)
    bucket_count_total: list[float] = field(default_factory=lambda: [0.0] * BUCKET_COUNT)
    bucket_price_sum_total: list[float] = field(default_factory=lambda: [0.0] * BUCKET_COUNT)
    bucket_haul_km_sum_total: list[float] = field(default_factory=lambda: [0.0] * BUCKET_COUNT)
    bucket_cost_time_sum_total: list[float] = field(default_factory=lambda: [0.0] * BUCKET_COUNT)


@dataclass(frozen=True)
class MemoryEvent:
    cargo_id: str
    observed_min: int
    source_driver_id: str
    start_key: tuple[int, int]
    end_key: tuple[int, int]
    price: float
    haul_km: float
    cost_time_min: int
    category: str
    load_window_width: int | None
    load_lead_time: int | None


def _cell_key(lat: float, lng: float) -> tuple[int, int]:
    return int(math.floor(lat / CELL_SIZE_DEG)), int(math.floor(lng / CELL_SIZE_DEG))


def _cell_center(key: tuple[int, int]) -> tuple[float, float]:
    return ((key[0] + 0.5) * CELL_SIZE_DEG, (key[1] + 0.5) * CELL_SIZE_DEG)


def _decay_factor(elapsed_min: float) -> float:
    if elapsed_min <= 0:
        return 1.0
    return 0.5 ** (elapsed_min / HALF_LIFE_MIN)


def _long_decay_factor(elapsed_min: float) -> float:
    if elapsed_min <= 0:
        return 1.0
    return 0.5 ** (elapsed_min / LONG_HALF_LIFE_MIN)


def _confidence(age_min: float) -> float:
    if age_min < 0:
        age_min = 0.0
    return 1.0 / (1.0 + age_min / HALF_LIFE_MIN)


def _trim_dict(d: dict[Any, int], max_keys: int) -> None:
    if len(d) <= max_keys:
        return
    sorted_items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    keep = dict(sorted_items[:max_keys])
    d.clear()
    d.update(keep)


def _trim_list(values: list[int], max_len: int) -> None:
    if len(values) > max_len:
        del values[: len(values) - max_len]


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2:
        return float(sorted_vals[mid])
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


class LongMemory:
    """Run-scoped shared market memory with causal query-time visibility."""

    def __init__(self, cost_per_km: float = 1.5, query_radius_km: float | None = None) -> None:
        self._cost_per_km = float(cost_per_km)
        self._cells: dict[tuple[int, int], CellStat] = {}
        self._seen_cargo_ids: set[str] = set()
        self._events: list[MemoryEvent] = []
        self._event_keys: set[tuple[str, int, str]] = set()
        self._events_version = 0
        self._max_seen_min = -1
        self._visible_cache: tuple[
            int,
            int,
            dict[tuple[int, int], CellStat],
            dict[tuple[int, int], list[tuple[int, int]]],
        ] | None = None
        self._max_value_cache: tuple[int, float] | None = None
        self._query_radius_km = float(query_radius_km) if query_radius_km is not None else REGION_QUERY_RADIUS_KM

    # ----- write path -----

    def set_cost_per_km(self, cost_per_km: float) -> None:
        self._cost_per_km = float(cost_per_km)
        self._max_value_cache = None

    def ingest(
        self,
        items: Iterable[dict[str, Any]],
        current_time_min: int,
        source_driver_id: str | None = None,
    ) -> dict[str, int]:
        added = 0
        skipped_dup = 0
        invalid = 0
        source = str(source_driver_id or "(unknown)")
        for item in items or []:
            if not isinstance(item, dict):
                invalid += 1
                continue
            cargo = item.get("cargo")
            if not isinstance(cargo, dict):
                invalid += 1
                continue
            cid = str(cargo.get("cargo_id", "")).strip()
            if not cid:
                invalid += 1
                continue
            event_key = (cid, int(current_time_min), source)
            if event_key in self._event_keys:
                skipped_dup += 1
                continue
            try:
                start = cargo["start"]
                end = cargo["end"]
                lat_start = float(start["lat"])
                lng_start = float(start["lng"])
                lat_end = float(end["lat"])
                lng_end = float(end["lng"])
                price = float(cargo.get("price", 0.0))
                cost_time_min = int(parse_cost_time_to_minutes(cargo))
                window_width, lead_time = self._parse_load_window(cargo, current_time_min)
            except (KeyError, TypeError, ValueError):
                invalid += 1
                continue
            haul_km = haversine_km(lat_start, lng_start, lat_end, lng_end)
            cat = str(cargo.get("cargo_name", "")).strip() or "(unknown)"
            self._seen_cargo_ids.add(cid)
            if len(self._seen_cargo_ids) > MAX_SEEN_CARGO_IDS:
                # cheap eviction: keep the most recently added by dropping a quarter
                dropped = len(self._seen_cargo_ids) // 4
                self._seen_cargo_ids = set(list(self._seen_cargo_ids)[dropped:])
            start_key = _cell_key(lat_start, lng_start)
            end_key = _cell_key(lat_end, lng_end)
            event = MemoryEvent(
                cargo_id=cid,
                observed_min=int(current_time_min),
                source_driver_id=source,
                start_key=start_key,
                end_key=end_key,
                price=price,
                haul_km=haul_km,
                cost_time_min=cost_time_min,
                category=cat,
                load_window_width=window_width,
                load_lead_time=lead_time,
            )
            self._events.append(event)
            self._event_keys.add(event_key)
            self._events_version += 1
            self._max_seen_min = max(self._max_seen_min, int(current_time_min))
            self._update_cell(
                key=start_key,
                current_time_min=current_time_min,
                price=price,
                haul_km=haul_km,
                cost_time_min=cost_time_min,
                category=cat,
                outbound_key=end_key,
                load_window_width=window_width,
                load_lead_time=lead_time,
                source_driver_id=source,
            )
            added += 1
        if added:
            self._visible_cache = None
            self._max_value_cache = None
            self._evict_if_needed(current_time_min)
        return {
            "added": added,
            "skipped_duplicate": skipped_dup,
            "invalid": invalid,
            "cells": self.size(current_time_min),
            "seen_total": len(self._seen_cargo_ids),
        }

    def _update_cell(
        self,
        *,
        key: tuple[int, int],
        current_time_min: int,
        price: float,
        haul_km: float,
        cost_time_min: int,
        category: str,
        outbound_key: tuple[int, int],
        load_window_width: int | None,
        load_lead_time: int | None,
        source_driver_id: str | None = None,
    ) -> None:
        self._update_cell_map(
            self._cells,
            key=key,
            current_time_min=current_time_min,
            price=price,
            haul_km=haul_km,
            cost_time_min=cost_time_min,
            category=category,
            outbound_key=outbound_key,
            load_window_width=load_window_width,
            load_lead_time=load_lead_time,
            source_driver_id=source_driver_id,
        )

    def _update_cell_map(
        self,
        cells: dict[tuple[int, int], CellStat],
        *,
        key: tuple[int, int],
        current_time_min: int,
        price: float,
        haul_km: float,
        cost_time_min: int,
        category: str,
        outbound_key: tuple[int, int],
        load_window_width: int | None,
        load_lead_time: int | None,
        source_driver_id: str | None = None,
    ) -> None:
        cell = cells.get(key)
        if cell is None:
            cell = CellStat(last_seen_min=int(current_time_min))
            cells[key] = cell
        elapsed = max(0, int(current_time_min) - int(cell.last_seen_min))
        decay = _decay_factor(elapsed)
        decay_long = _long_decay_factor(elapsed)
        cell.count_recent = cell.count_recent * decay + 1.0
        cell.price_sum_recent = cell.price_sum_recent * decay + price
        cell.haul_km_sum_recent = cell.haul_km_sum_recent * decay + haul_km
        cell.cost_time_sum_recent = cell.cost_time_sum_recent * decay + cost_time_min
        cell.count_total = cell.count_total * decay_long + 1.0
        cell.price_sum_total = cell.price_sum_total * decay_long + price
        cell.haul_km_sum_total = cell.haul_km_sum_total * decay_long + haul_km
        cell.cost_time_sum_total = cell.cost_time_sum_total * decay_long + cost_time_min
        for i in range(BUCKET_COUNT):
            cell.bucket_count_total[i] *= decay_long
            cell.bucket_price_sum_total[i] *= decay_long
            cell.bucket_haul_km_sum_total[i] *= decay_long
            cell.bucket_cost_time_sum_total[i] *= decay_long
        bucket = _bucket_index(current_time_min)
        cell.bucket_count_total[bucket] += 1.0
        cell.bucket_price_sum_total[bucket] += price
        cell.bucket_haul_km_sum_total[bucket] += haul_km
        cell.bucket_cost_time_sum_total[bucket] += cost_time_min
        cell.last_seen_min = int(current_time_min)
        if source_driver_id:
            cell.source_driver_ids.add(str(source_driver_id))
        cell.categories[category] = cell.categories.get(category, 0) + 1
        _trim_dict(cell.categories, MAX_CATEGORIES_PER_CELL)
        if cell.outbound:
            for ob_key in list(cell.outbound):
                decayed_count = cell.outbound[ob_key] * decay_long
                if decayed_count < 0.01:
                    cell.outbound.pop(ob_key, None)
                else:
                    cell.outbound[ob_key] = decayed_count
        if outbound_key != key:
            cell.outbound[outbound_key] = cell.outbound.get(outbound_key, 0) + 1
            _trim_dict(cell.outbound, MAX_OUTBOUND_PER_CELL)
        if load_window_width is not None:
            cell.load_window_widths.append(int(load_window_width))
            _trim_list(cell.load_window_widths, MAX_LOAD_WINDOWS_PER_CELL)
        if load_lead_time is not None:
            cell.load_lead_times.append(int(load_lead_time))
            _trim_list(cell.load_lead_times, MAX_LOAD_WINDOWS_PER_CELL)

    def _parse_load_window(self, cargo: dict[str, Any], current_time_min: int) -> tuple[int | None, int | None]:
        raw = cargo.get("load_time")
        if raw is None:
            return None, None
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"cargo load_time must be a 2-item list: {raw!r}")
        start_min = wall_time_to_sim_min(str(raw[0]))
        end_min = wall_time_to_sim_min(str(raw[1]))
        if end_min < start_min:
            raise ValueError(f"cargo load_time ends before it starts: {raw!r}")
        width = end_min - start_min
        lead = max(0, start_min - int(current_time_min))
        return width, lead

    def _evict_if_needed(self, current_time_min: int) -> None:
        if len(self._cells) <= MAX_CELLS:
            return
        # evict cells with lowest combined freshness × count
        scored = []
        for key, cell in self._cells.items():
            age = max(0, int(current_time_min) - int(cell.last_seen_min))
            freshness = _decay_factor(age)
            scored.append((key, freshness * (1 + self._decayed_total_count(cell, current_time_min))))
        scored.sort(key=lambda kv: kv[1])
        drop = len(self._cells) - MAX_CELLS
        for key, _score in scored[:drop]:
            self._cells.pop(key, None)

    # ----- read path -----

    def size(self, current_time_min: int | None = None) -> int:
        if current_time_min is None:
            return len(self._cells)
        return len(self._visible_cells(current_time_min))

    def visible_event_count(self, current_time_min: int) -> int:
        seen_cargo: set[str] = set()
        cutoff = int(current_time_min)
        for event in self._events:
            if event.observed_min <= cutoff:
                seen_cargo.add(event.cargo_id)
        return len(seen_cargo)

    def _visible_cells(self, current_time_min: int) -> dict[tuple[int, int], CellStat]:
        """Return a causally visible aggregate for ``current_time_min``.

        A shared LongMemory may contain events from another driver that happened
        later in simulation time. Query-time aggregation prevents those future
        observations from influencing an earlier driver's decisions.
        """
        return self._visible_view(current_time_min)[0]

    def _visible_view(
        self, current_time_min: int
    ) -> tuple[dict[tuple[int, int], CellStat], dict[tuple[int, int], list[tuple[int, int]]]]:
        cache_key = int(current_time_min)
        if (
            self._visible_cache is not None
            and self._visible_cache[0] == cache_key
            and self._visible_cache[1] == self._events_version
        ):
            return self._visible_cache[2], self._visible_cache[3]
        cells: dict[tuple[int, int], CellStat] = {}
        seen_cargo: set[str] = set()
        for event in sorted(self._events, key=lambda e: e.observed_min):
            if event.observed_min > cache_key:
                continue
            if event.cargo_id in seen_cargo:
                existing = cells.get(event.start_key)
                if existing is not None:
                    existing.source_driver_ids.add(event.source_driver_id)
                continue
            seen_cargo.add(event.cargo_id)
            self._update_cell_map(
                cells,
                key=event.start_key,
                current_time_min=event.observed_min,
                price=event.price,
                haul_km=event.haul_km,
                cost_time_min=event.cost_time_min,
                category=event.category,
                outbound_key=event.end_key,
                load_window_width=event.load_window_width,
                load_lead_time=event.load_lead_time,
                source_driver_id=event.source_driver_id,
            )
        tile_index: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for cell_key in cells:
            center = _cell_center(cell_key)
            tile_index.setdefault(_tile_key(center[0], center[1]), []).append(cell_key)
        self._visible_cache = (cache_key, self._events_version, cells, tile_index)
        return cells, tile_index

    def _iter_cells_in_radius(
        self, lat: float, lng: float, current_time_min: int
    ):
        cells, tile_index = self._visible_view(current_time_min)
        radius = self._query_radius_km
        base = _tile_key(lat, lng)
        for dt_lat in (-1, 0, 1):
            for dt_lng in (-1, 0, 1):
                bucket = tile_index.get((base[0] + dt_lat, base[1] + dt_lng))
                if not bucket:
                    continue
                for cell_key in bucket:
                    center = _cell_center(cell_key)
                    dist = haversine_km(lat, lng, center[0], center[1])
                    if dist >= radius:
                        continue
                    yield cell_key, cells[cell_key], center, dist, cells

    def value_distribution(self, current_time_min: int) -> dict[str, float]:
        """Snapshot the aggregated region_value distribution at each cell center."""
        values: list[float] = []
        ages: list[int] = []
        total_counts: list[int] = []
        cells = self._visible_cells(current_time_min)
        for key, cell in cells.items():
            center = _cell_center(key)
            val = self.region_value(center[0], center[1], current_time_min)
            values.append(val)
            ages.append(max(0, int(current_time_min) - int(cell.last_seen_min)))
            total_counts.append(self._decayed_total_count(cell, current_time_min))
        if not values:
            return {
                "cells": 0,
                "max": 0.0,
                "min": 0.0,
                "mean": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "mean_age_min": 0.0,
                "median_count_total": 0.0,
            }
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "cells": n,
            "max": round(sorted_vals[-1], 3),
            "min": round(sorted_vals[0], 3),
            "mean": round(sum(sorted_vals) / n, 3),
            "p50": round(sorted_vals[n // 2], 3),
            "p90": round(sorted_vals[min(n - 1, int(n * 0.9))], 3),
            "mean_age_min": round(sum(ages) / n, 1),
            "median_count_total": _median(total_counts),
        }

    def region_value(self, lat: float, lng: float, current_time_min: int) -> float:
        """Return aggregated region value at (lat, lng), summed over neighbors in radius."""
        total, _coverage, _local_obs = self._aggregate_in_radius(lat, lng, current_time_min)
        return total

    def region_coverage_cells(self, lat: float, lng: float, current_time_min: int | None = None) -> int:
        """Return how many LM cells have center within the query radius of (lat, lng)."""
        if current_time_min is None:
            coverage = 0
            for key in self._cells:
                center = _cell_center(key)
                if haversine_km(lat, lng, center[0], center[1]) < self._query_radius_km:
                    coverage += 1
            return coverage
        return sum(1 for _ in self._iter_cells_in_radius(lat, lng, current_time_min))

    def _aggregate_in_radius(self, lat: float, lng: float, current_time_min: int) -> tuple[float, int, float]:
        """Aggregate cell values within radius, returning (sum, coverage_count, local_observations)."""
        total = 0.0
        coverage = 0
        local_obs = 0.0
        radius = self._query_radius_km
        for _key, cell, _center, dist, cells in self._iter_cells_in_radius(lat, lng, current_time_min):
            coverage += 1
            weight = 1.0 - dist / radius
            total += weight * self._compute_region_value(cell, current_time_min, cells)
            local_obs += weight * self._decayed_total_count(cell, current_time_min)
        return total, coverage, local_obs

    def _source_driver_ids_in_radius(
        self, lat: float, lng: float, current_time_min: int
    ) -> set[str]:
        source_ids: set[str] = set()
        for _key, cell, _center, _dist, _cells in self._iter_cells_in_radius(lat, lng, current_time_min):
            source_ids.update(cell.source_driver_ids)
        return source_ids

    def _source_confidence(
        self,
        source_ids: set[str] | int,
        asking_driver_id: str | None = None,
    ) -> float:
        """**每司机一份独立 LongMemory、绝不跨司机共享区域价值**（结构保证，非运行期偶然）。
        置信二元化：本格被第一手观测过 → 满信 1.0；没观测过 → 0.0（raw_norm 退回探索先验）。
        每司机各自一个 LongMemory（见 loop._driver_state）、只 ingest 自己的观测，故 ``source_ids``
        只可能是空集或 {本司机}——**不存在"多司机佐证"，已删除原跨司机共享池的对数置信曲线**，
        杜绝任何跨司机区域价值串味。``asking_driver_id`` 仅留作签名兼容、不再参与计算；
        ``source_ids`` 给 int 是 legacy 调用路径。"""
        if isinstance(source_ids, set):
            return 1.0 if source_ids else 0.0
        return 1.0 if int(source_ids or 0) > 0 else 0.0

    def region_value_normalized(
        self,
        lat: float,
        lng: float,
        current_time_min: int,
        asking_driver_id: str | None = None,
    ) -> float:
        """Normalized [0,1]. Uses UCB style bonus for exploration when intrinsic value is similar or uncertain.
        """
        max_value = self._max_region_value(current_time_min)
        raw, coverage, local_obs = self._aggregate_in_radius(lat, lng, current_time_min)

        raw_norm = 0.0
        if max_value > 0 and raw > 0:
            raw_norm = raw / max_value
        source_ids = self._source_driver_ids_in_radius(lat, lng, current_time_min)
        source_conf = self._source_confidence(source_ids, asking_driver_id)
        raw_norm = source_conf * raw_norm + (1.0 - source_conf) * EXPLORATORY_PRIOR_NORM

        bonus = 0.0
        cells = self._visible_cells(current_time_min)
        if len(cells) >= MIN_CELLS_FOR_EXPLORATION:
            total_obs = sum(self._decayed_total_count(c, current_time_min) for c in cells.values())
            # Exploration is useful mainly during the first week. After that,
            # keep only a tiny residual bonus so proven regions dominate.
            progress = min(1.0, current_time_min / EXPLORATION_DECAY_MIN)
            alpha = EXPLORATION_ALPHA_MIN + (EXPLORATION_ALPHA_MAX - EXPLORATION_ALPHA_MIN) * (1.0 - progress)

            bonus = alpha * math.sqrt(math.log(total_obs + 1.0) / (local_obs + 1.0))
            bonus = min(EXPLORATION_BONUS_CAP, bonus)

        return min(1.0, max(EXPLORATORY_PRIOR_NORM, raw_norm + bonus))

    def hot_zones(
        self,
        current_time_min: int,
        n: int = 10,
        asking_driver_id: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cells = self._visible_cells(current_time_min)
        for key, cell in cells.items():
            center = _cell_center(key)
            aggregated = self.region_value(center[0], center[1], current_time_min)
            normalized = self.region_value_normalized(
                center[0], center[1], current_time_min, asking_driver_id=asking_driver_id
            )
            if aggregated <= 0:
                continue
            age = max(0, int(current_time_min) - int(cell.last_seen_min))
            # Decayed cargo count observed in the CURRENT 6h time-of-day bucket (00-06 / 06-12 / 12-18
            # / 18-24). region_value_norm shrinks a sparse bucket back to the cell-wide (day-dominated)
            # average, so a day-hot / night-DEAD zone still reads "hot" at night — a useless dead-zone
            # escape target. Exposing the bucket count lets the escape DOWNWEIGHT a zone with little/no
            # cargo at THIS time of day (晚上要去晚上有货的区).
            _bucket_cnt = cell.bucket_count_total[_bucket_index(current_time_min)] * _long_decay_factor(age)
            rows.append(
                {
                    "cell": list(key),
                    "center": {"lat": round(center[0], 4), "lng": round(center[1], 4)},
                    "region_value": round(aggregated, 3),
                    "region_value_norm": round(normalized, 4),
                    "count_total": round(self._decayed_total_count(cell, current_time_min), 2),
                    "bucket_count": round(_bucket_cnt, 2),
                    "source_driver_count": len(cell.source_driver_ids),
                    "last_seen_min": cell.last_seen_min,
                    "age_min": age,
                    "avg_haul_only_yield_per_min": round(self._avg_yield_per_min(cell, current_time_min), 3),
                }
            )
        rows.sort(key=lambda row: (row["region_value_norm"], row["region_value"]), reverse=True)
        return rows[: max(1, n)]

    def region_detail(
        self,
        lat: float,
        lng: float,
        current_time_min: int,
        asking_driver_id: str | None = None,
    ) -> dict[str, Any]:
        key = _cell_key(lat, lng)
        cells = self._visible_cells(current_time_min)
        cell = cells.get(key)
        if cell is None:
            return {
                "cell": list(key),
                "known": False,
                "center": {"lat": round((key[0] + 0.5) * CELL_SIZE_DEG, 4), "lng": round((key[1] + 0.5) * CELL_SIZE_DEG, 4)},
            }
        center = _cell_center(key)
        aggregated = self.region_value(center[0], center[1], current_time_min)
        normalized = self.region_value_normalized(
            center[0], center[1], current_time_min, asking_driver_id=asking_driver_id
        )
        intrinsic = self._compute_region_value(cell, current_time_min, cells)
        age = max(0, int(current_time_min) - int(cell.last_seen_min))
        decay = _decay_factor(age)
        recent_count = cell.count_recent * decay
        decayed_count = self._decayed_total_count(cell, current_time_min)
        decayed_price = self._decayed_total_sum(cell, current_time_min, "price_sum_total")
        decayed_haul = self._decayed_total_sum(cell, current_time_min, "haul_km_sum_total")
        decayed_cost_time = self._decayed_total_sum(cell, current_time_min, "cost_time_sum_total")
        avg_denominator = decayed_count if decayed_count > 1e-9 else 1.0
        outbound_decay = _long_decay_factor(age)
        outbound_summary = []
        for ob_key, cnt in sorted(cell.outbound.items(), key=lambda kv: kv[1], reverse=True)[:5]:
            decayed_outbound = float(cnt) * outbound_decay
            if decayed_outbound < 0.01:
                continue
            ob_center = _cell_center(ob_key)
            outbound_summary.append(
                {
                    "cell": list(ob_key),
                    "center": {"lat": round(ob_center[0], 4), "lng": round(ob_center[1], 4)},
                    "count": round(decayed_outbound, 2),
                }
            )
        return {
            "cell": list(key),
            "known": True,
            "center": {"lat": round(center[0], 4), "lng": round(center[1], 4)},
            "region_value": round(aggregated, 3),
            "region_value_norm": round(normalized, 4),
            "region_value_intrinsic": round(intrinsic, 3),
            "count_total": round(decayed_count, 2),
            "source_driver_count": len(cell.source_driver_ids),
            "count_recent_decayed": round(recent_count, 2),
            "avg_haul_only_yield_per_min": round(self._avg_yield_per_min(cell, current_time_min), 3),
            "avg_price": round(decayed_price / avg_denominator, 2),
            "avg_haul_km": round(decayed_haul / avg_denominator, 2),
            "avg_cost_time_min": round(decayed_cost_time / avg_denominator, 1),
            "median_load_window_width_min": round(_median(cell.load_window_widths), 1),
            "median_load_lead_time_min": round(_median(cell.load_lead_times), 1),
            "last_seen_min": cell.last_seen_min,
            "age_min": age,
            "confidence": round(_confidence(age), 3),
            "top_categories": dict(sorted(cell.categories.items(), key=lambda kv: kv[1], reverse=True)[:8]),
            "top_outbound": outbound_summary,
        }

    def category_locations(
        self,
        category: str,
        current_time_min: int,
        n: int = 10,
        asking_driver_id: str | None = None,
    ) -> list[dict[str, Any]]:
        target = str(category).strip()
        if not target:
            return []
        rows: list[dict[str, Any]] = []
        cells = self._visible_cells(current_time_min)
        for key, cell in cells.items():
            cnt = cell.categories.get(target, 0)
            if cnt <= 0:
                continue
            center = _cell_center(key)
            aggregated = self.region_value(center[0], center[1], current_time_min)
            normalized = self.region_value_normalized(
                center[0], center[1], current_time_min, asking_driver_id=asking_driver_id
            )
            rows.append(
                {
                    "cell": list(key),
                    "center": {"lat": round(center[0], 4), "lng": round(center[1], 4)},
                    "category_count": cnt,
                    "region_value_norm": round(normalized, 4),
                    "last_seen_min": cell.last_seen_min,
                    "age_min": max(0, int(current_time_min) - int(cell.last_seen_min)),
                }
            )
        rows.sort(key=lambda row: (row["category_count"], row["region_value_norm"]), reverse=True)
        return rows[: max(1, n)]

    # ----- internal -----

    def _long_age_decay(self, cell: CellStat, current_time_min: int) -> float:
        age = max(0, int(current_time_min) - int(cell.last_seen_min))
        return _long_decay_factor(age)

    def _decayed_total_count(self, cell: CellStat, current_time_min: int) -> float:
        return cell.count_total * self._long_age_decay(cell, current_time_min)

    def _decayed_total_sum(self, cell: CellStat, current_time_min: int, field_name: str) -> float:
        return float(getattr(cell, field_name, 0.0)) * self._long_age_decay(cell, current_time_min)

    def _compute_local_region_value(self, cell: CellStat, current_time_min: int) -> float:
        count = self._decayed_total_count(cell, current_time_min)
        if count <= 0:
            return 0.0
        avg_ypm = self._avg_yield_per_min(cell, current_time_min)
        if avg_ypm <= 0:
            return 0.0
        return avg_ypm * math.log(1.0 + count)

    def _compute_region_value(
        self,
        cell: CellStat,
        current_time_min: int,
        cells: dict[tuple[int, int], CellStat] | None = None,
    ) -> float:
        base_value = self._compute_local_region_value(cell, current_time_min)
        if base_value <= 0:
            return 0.0
        visible_cells = cells if cells is not None else self._visible_cells(current_time_min)

        # Outbound value integration (Simplified PageRank style)
        # 考虑从这个地方发出去通常会把司机送到哪里的价值，从而防范死胡同
        outbound_bonus = 0.0
        if cell.outbound:
            outbound_decay = self._long_age_decay(cell, current_time_min)
            decayed_outbound = {
                ob_key: max(0.0, float(cnt) * outbound_decay)
                for ob_key, cnt in cell.outbound.items()
            }
            total_ob = sum(decayed_outbound.values())
            sum_ob_val = 0.0
            for ob_key, cnt in decayed_outbound.items():
                ob_cell = visible_cells.get(ob_key)
                if ob_cell is not None:
                    sum_ob_val += cnt * self._compute_local_region_value(ob_cell, current_time_min)
            if total_ob > 0:
                flow_confidence = min(1.0, total_ob / FLOW_CONFIDENCE_SAMPLE)
                outbound_bonus = FLOW_VALUE_WEIGHT * flow_confidence * (sum_ob_val / total_ob)

        return base_value + outbound_bonus

    def _avg_yield_per_min(self, cell: CellStat, current_time_min: int) -> float:
        age = max(0, int(current_time_min) - int(cell.last_seen_min))
        decay = _decay_factor(age)
        cost_time_recent = cell.cost_time_sum_recent * decay
        if cost_time_recent > 0:
            ypm_recent = (
                cell.price_sum_recent * decay - self._cost_per_km * cell.haul_km_sum_recent * decay
            ) / cost_time_recent
        else:
            ypm_recent = 0.0
        decay_long = _long_decay_factor(age)
        cost_time_total = cell.cost_time_sum_total * decay_long
        if cost_time_total > 0:
            ypm_total_cell = (
                cell.price_sum_total * decay_long
                - self._cost_per_km * cell.haul_km_sum_total * decay_long
            ) / cost_time_total
        else:
            ypm_total_cell = 0.0
        # Bayesian shrinkage to the time-of-day bucket active at query time.
        # Empty bucket → w=0, fall back to the cell-wide average (no harm under sparsity);
        # well-observed bucket → w→1, intra-day signal dominates.
        bucket = _bucket_index(current_time_min)
        bucket_count = cell.bucket_count_total[bucket] * decay_long
        if bucket_count > 0:
            bucket_cost_time = cell.bucket_cost_time_sum_total[bucket] * decay_long
            if bucket_cost_time > 0:
                bucket_ypm = (
                    cell.bucket_price_sum_total[bucket] * decay_long
                    - self._cost_per_km * cell.bucket_haul_km_sum_total[bucket] * decay_long
                ) / bucket_cost_time
            else:
                bucket_ypm = 0.0
            w = bucket_count / (bucket_count + BUCKET_SHRINKAGE_KAPPA)
        else:
            bucket_ypm = 0.0
            w = 0.0
        ypm_total = w * bucket_ypm + (1.0 - w) * ypm_total_cell
        conf = _confidence(age)
        return conf * ypm_recent + (1.0 - conf) * ypm_total

    def _max_region_value(self, current_time_min: int) -> float:
        """Max aggregated region_value over all cell centers, used for normalization."""
        if (
            self._max_value_cache is not None
            and self._max_value_cache[0] == int(current_time_min)
        ):
            return self._max_value_cache[1]
        max_val = 0.0
        cells = self._visible_cells(current_time_min)
        for key in cells:
            center = _cell_center(key)
            val = self.region_value(center[0], center[1], current_time_min)
            if val > max_val:
                max_val = val
        self._max_value_cache = (int(current_time_min), max_val)
        return max_val
