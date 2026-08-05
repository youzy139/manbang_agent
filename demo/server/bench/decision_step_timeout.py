"""单步 agent 决策墙钟超时（含选手代码与模型调用）。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable


class DecisionStepTimeoutError(TimeoutError):
    """单步决策超过裁判配置的上限。"""

    def __init__(self, driver_id: str, timeout_seconds: float) -> None:
        self.driver_id = driver_id
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"司机 {driver_id} 单步决策超过 {timeout_seconds:g}s 上限，"
            "已终止该司机仿真并保留此前已完成的步数"
        )


class DecisionStepGuard:
    """在独立线程中执行 decide，超时则抛出 DecisionStepTimeoutError。"""

    def __init__(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("decision_step_timeout_seconds 必须为正数")
        self._timeout_seconds = float(timeout_seconds)

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    def call(self, decide: Callable[[str], dict[str, Any]], driver_id: str) -> dict[str, Any]:
        # 不用 with：默认 shutdown(wait=True) 会在超时后仍阻塞等待卡死的 decide 线程
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(decide, driver_id)
        try:
            return future.result(timeout=self._timeout_seconds)
        except FuturesTimeoutError as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            raise DecisionStepTimeoutError(driver_id, self._timeout_seconds) from exc
        else:
            executor.shutdown(wait=False)
