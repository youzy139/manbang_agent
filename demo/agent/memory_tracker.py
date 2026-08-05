"""司机运行时记忆：在 agent 侧基于决策历史累计收入、里程与单量。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, l1 = math.radians(lat1), math.radians(lng1)
    p2, l2 = math.radians(lat2), math.radians(lng2)
    dp, dl = p2 - p1, l2 - l1
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * r * math.asin(math.sqrt(h))


class DriverMemory:
    def __init__(self) -> None:
        self.total_income_yuan = 0.0
        self.total_deadhead_km = 0.0
        self.total_haul_km = 0.0
        self.total_reposition_km = 0.0
        self.daily_orders: dict[str, int] = {}
        self._last_processed_step = 0
        self.last_rest_progress_minutes = 0
        self.wait_count = 0
        # 每日最长连续 wait 时长（分钟）
        self.daily_max_rest_minutes: dict[str, int] = {}
        # 每日首单完成时间（分钟），用于 first_order_before 违规计算
        self.daily_first_order_minutes: dict[str, int] = {}
        # Cap-aware constraint tracking（由 populate_violation_counts 填充）
        self.haul_violation_count = 0
        self.pickup_violation_count = 0
        self.deadhead_violation_km = 0.0
        self.boundary_violation_count = 0
        self.first_order_violation_count = 0
        self.daily_rest_violation_count = 0
        # 鲁棒优化：按类型分类统计近期罚金（rest/kpi/long_haul/time/geo）
        self.penalty_by_type: dict[str, float] = {
            "rest": 0.0,
            "kpi": 0.0,
            "long_haul": 0.0,
            "time": 0.0,
            "geo": 0.0,
        }

    def get_summary(self, today_date: str, current_progress_minutes: int) -> dict[str, Any]:
        today_order_count = self.daily_orders.get(today_date, 0)
        monthly_order_count = sum(self.daily_orders.values())
        continuous_work = max(0, current_progress_minutes - self.last_rest_progress_minutes)
        return {
            "today_date": today_date,
            "today_order_count": today_order_count,
            "monthly_order_count": monthly_order_count,
            "total_income_yuan": round(self.total_income_yuan, 2),
            "total_deadhead_km": round(self.total_deadhead_km, 2),
            "total_haul_km": round(self.total_haul_km, 2),
            "total_reposition_km": round(self.total_reposition_km, 2),
            "total_all_deadhead_km": round(self.total_deadhead_km + self.total_reposition_km, 2),
            "continuous_work_minutes": continuous_work,
            "minutes_since_last_rest": continuous_work,
            "wait_count_this_month": self.wait_count,
            "daily_max_rest_minutes": self.daily_max_rest_minutes.get(today_date, 0),
            "daily_first_order_minutes": self.daily_first_order_minutes.get(today_date),
            "haul_violation_count": self.haul_violation_count,
            "pickup_violation_count": self.pickup_violation_count,
            "deadhead_violation_km": round(self.deadhead_violation_km, 2),
            "boundary_violation_count": self.boundary_violation_count,
            "first_order_violation_count": self.first_order_violation_count,
            "daily_rest_violation_count": self.daily_rest_violation_count,
        }


class MemoryTracker:
    """基于 query_decision_history 的累计记忆；每次只增量处理新增记录。"""

    def __init__(self, cargo_price_map: dict[str, float] | None = None) -> None:
        self._memories: dict[str, DriverMemory] = {}
        self._cargo_price_map = cargo_price_map or {}

    def update(self, driver_id: str, records: list[dict[str, Any]]) -> None:
        mem = self._memories.setdefault(driver_id, DriverMemory())
        if not records:
            return
        # 增量：只处理 step > last_processed_step 的记录
        for rec in records:
            step = rec.get("step")
            if not isinstance(step, int) or step <= mem._last_processed_step:
                continue
            mem._last_processed_step = step
            action = rec.get("action", {})
            result = rec.get("result", {})
            action_name = str(action.get("action", "")).strip().lower()
            # 记录 reposition 空驶里程（用于月度空驶约束预算）
            if action_name == "reposition":
                pos_before = rec.get("position_before", {})
                pos_after = rec.get("position_after", {})
                if pos_before and pos_after:
                    mem.total_reposition_km += _haversine_km(
                        float(pos_before.get("lat", 0)), float(pos_before.get("lng", 0)),
                        float(pos_after.get("lat", 0)), float(pos_after.get("lng", 0)),
                    )
                continue
            # 记录休息节点
            if action_name == "wait":
                progress = int(result.get("simulation_progress_minutes", 0))
                mem.last_rest_progress_minutes = progress
                mem.wait_count += 1
                # 按天拆分跨天休息；post-hoc 端以日历日拆分区间，
                # 若这里记录完整 wait_dur 到起始日，会导致 simulation 认为该日休息充足，
                # 实际 post-hoc 因午夜拆分仅计部分时长 → 误判违规。
                wait_dur = int(action.get("params", {}).get("duration_minutes", 0))
                start_progress = max(0, progress - wait_dur)
                cursor = start_progress
                while cursor < progress:
                    day_end = ((cursor // 1440) + 1) * 1440
                    seg_end = min(progress, day_end)
                    seg_dur = seg_end - cursor
                    today = (datetime(2026, 3, 1) + timedelta(minutes=cursor)).strftime("%Y-%m-%d")
                    prev_max = mem.daily_max_rest_minutes.get(today, 0)
                    mem.daily_max_rest_minutes[today] = max(prev_max, seg_dur)
                    cursor = seg_end
                continue
            # 只累计成功接单
            if result.get("accepted") is not True:
                continue
            if action_name != "take_order":
                continue
            params = action.get("params", {})
            cargo_id = str(params.get("cargo_id", "")).strip()
            price = self._cargo_price_map.get(cargo_id, 0.0)
            mem.total_income_yuan += price
            mem.total_deadhead_km += float(result.get("pickup_deadhead_km", 0.0))
            mem.total_haul_km += float(result.get("haul_distance_km", 0.0))
            wall_time = str(result.get("simulation_wall_time", "")).strip()
            today = wall_time.split()[0] if wall_time else ""
            if today:
                mem.daily_orders[today] = mem.daily_orders.get(today, 0) + 1
            # 追踪每单的违反信息（由外部调用 populate_violation_counts 补充计算）
            action_progress = int(result.get("simulation_progress_minutes", 0))
            _track_daily_first_order(mem, today, action_progress)

    def get_summary(self, driver_id: str, today_date: str, current_progress_minutes: int) -> dict[str, Any]:
        mem = self._memories.get(driver_id)
        if mem is None:
            return {
                "today_date": today_date,
                "today_order_count": 0,
                "monthly_order_count": 0,
                "total_income_yuan": 0.0,
                "total_deadhead_km": 0.0,
                "total_haul_km": 0.0,
                "continuous_work_minutes": current_progress_minutes,
                "minutes_since_last_rest": current_progress_minutes,
                "wait_count_this_month": 0,
            }
        return mem.get_summary(today_date, current_progress_minutes)

    def populate_violation_counts(
        self,
        driver_id: str,
        persona: dict[str, Any],
    ) -> None:
        """基于画像约束与累计统计，推算当前违规次数（用于 cap 预算感知）。"""
        mem = self._memories.get(driver_id)
        if mem is None:
            return

        # haul_violation: 每单运费距离超过 max_haul_km 的次数
        haul_rules = persona.get("max_haul_km", [])
        if haul_rules:
            max_haul = haul_rules[0].get("max_km") or float("inf")
            # 估算方式：用总运输距离 / 月单量 反推平均运距，
            # 结合典型运距分布粗略估计违规单占比。
            # 精确做法是在 update 时逐单记录并比对，这里用保守近似：
            # 违规次数 ≈ 月运距违规率 × 总单数，当且仅当月度运距超过阈值时显著。
            monthly_orders = sum(mem.daily_orders.values())
            if monthly_orders > 0:
                avg_haul = mem.total_haul_km / monthly_orders
                if avg_haul > max_haul * 0.8:
                    # 平均运距接近阈值时，估计约 40% 订单超限
                    mem.haul_violation_count = max(0, int(monthly_orders * 0.4))
        if not haul_rules:
            mem.haul_violation_count = 0

        # pickup_violation: 每单空驶距离超过 max_pickup_deadhead_km 的次数
        pickup_rules = persona.get("max_pickup_deadhead_km", [])
        if pickup_rules:
            max_pickup = pickup_rules[0].get("max_km") or float("inf")
            monthly_orders = sum(mem.daily_orders.values())
            if monthly_orders > 0:
                avg_pickup = mem.total_deadhead_km / monthly_orders
                if avg_pickup > max_pickup * 0.8:
                    mem.pickup_violation_count = max(0, int(monthly_orders * 0.35))
        if not pickup_rules:
            mem.pickup_violation_count = 0

        # deadhead_violation_km: 月度总空驶超过 max_deadhead_km 的超额公里数
        deadhead_rules = persona.get("max_deadhead_km", [])
        if deadhead_rules:
            max_dh = deadhead_rules[0].get("max_km") or float("inf")
            total_dh = mem.total_deadhead_km + mem.total_reposition_km
            mem.deadhead_violation_km = max(0.0, total_dh - max_dh)

        # daily_rest_violation: 休息不足的天数（支持 days_of_week 筛选）
        rest_rules = persona.get("daily_rest_min_hours", [])
        if rest_rules:
            for day_str, rest_min in mem.daily_max_rest_minutes.items():
                day_dt = datetime.strptime(day_str, "%Y-%m-%d")
                weekday = day_dt.weekday()  # 0=周一 ... 6=周日
                matched_rule = None
                for rule in rest_rules:
                    dow = rule.get("days_of_week")
                    if dow is None or weekday in dow:
                        matched_rule = rule
                        break
                if matched_rule:
                    min_hours = matched_rule.get("min_hours") or 0
                    min_minutes = min_hours * 60
                    if rest_min < min_minutes:
                        mem.daily_rest_violation_count += 1

        # first_order_violation: 首单时间超过 deadline 的天数
        first_order_rules = persona.get("first_order_before", [])
        if first_order_rules:
            deadline_hour = first_order_rules[0].get("hour") or 12
            deadline_minutes = deadline_hour * 60
            for day, first_min in mem.daily_first_order_minutes.items():
                first_mod = first_min % 1440
                if first_mod > deadline_minutes:
                    mem.first_order_violation_count += 1

        # boundary_violation: geo_boundary 违反次数（从每日位置推算，保守估计）
        # 由于 MemoryTracker 不存储完整位置历史，该计数由 ModelDecisionService
        # 在扫描决策历史时单独计算并直接赋值。
        boundary_rules = persona.get("geo_boundary", [])
        if not boundary_rules:
            mem.boundary_violation_count = 0


def _track_daily_first_order(mem: DriverMemory, today: str, progress_minutes: int) -> None:
    """记录每日首单完成时间（用于 first_order_before 违规计算）。"""
    if not today:
        return
    if today in mem.daily_first_order_minutes:
        return
    mem.daily_first_order_minutes[today] = progress_minutes
