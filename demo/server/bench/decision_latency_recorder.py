"""记录 agent decide 接口墙钟耗时（内部毫秒累积，输出秒）。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _DriverLatencyAccumulator:
    count: int = 0
    total_ms: float = 0.0

    def add(self, elapsed_ms: float) -> None:
        self.count += 1
        self.total_ms += float(elapsed_ms)


class DriverDecisionLatencyRecorder:
    """按司机累积 decide 耗时（毫秒），输出 count / total_seconds / avg_seconds。"""

    def __init__(self) -> None:
        self._by_driver: dict[str, _DriverLatencyAccumulator] = {}

    def record(self, driver_id: str, elapsed_ms: float) -> None:
        acc = self._by_driver.setdefault(driver_id, _DriverLatencyAccumulator())
        acc.add(elapsed_ms)

    def build_summary(self) -> dict[str, dict]:
        summary: dict[str, dict] = {}
        for driver_id, acc in self._by_driver.items():
            total_seconds = round(acc.total_ms / 1000.0, 2)
            avg_seconds = round(acc.total_ms / acc.count / 1000.0, 2) if acc.count else 0.0
            summary[driver_id] = {
                "count": acc.count,
                "total_seconds": total_seconds,
                "avg_seconds": avg_seconds,
            }
        return summary
