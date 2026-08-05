"""单司机仿真周期内的模型 token 累计与上限判定。"""

from __future__ import annotations

from typing import Any


class DriverTokenBudget:
    DEFAULT_LIMIT = 5_000_000

    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("driver_max_total_tokens 必须为正整数")
        self._limit = int(limit)
        self._cumulative: dict[str, int] = {}
        self._limit_stopped: dict[str, bool] = {}

    @property
    def limit(self) -> int:
        return self._limit

    def record_step(self, driver_id: str, token_usage: dict[str, Any]) -> int:
        step_total = max(0, int(token_usage.get("total_tokens", 0)))
        new_total = self._cumulative.get(driver_id, 0) + step_total
        self._cumulative[driver_id] = new_total
        if new_total > self._limit:
            self._limit_stopped[driver_id] = True
        return new_total

    def is_over_limit(self, driver_id: str) -> bool:
        return self._cumulative.get(driver_id, 0) > self._limit

    def was_stopped_by_limit(self, driver_id: str) -> bool:
        return self._limit_stopped.get(driver_id, False)

    def cumulative(self, driver_id: str) -> int:
        return self._cumulative.get(driver_id, 0)

    def cumulative_snapshot(self) -> dict[str, int]:
        return dict(self._cumulative)

    def limit_stopped_snapshot(self) -> dict[str, bool]:
        return dict(self._limit_stopped)
