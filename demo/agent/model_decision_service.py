"""模型决策服务：依赖 `simkit.ports.SimulationApiPort`，由评测进程注入具体环境。

本版本把当前 DriverPersona 画像提取结果接入参考实现的 StrategyFieldEngine（虚拟单 + 货源调价 + 单位时间收益规划）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from simkit.ports import SimulationApiPort

from agent._helpers import _load_reposition_speed_km_per_hour
from agent._persona_adapter import persona_to_parsed_preferences
from agent.driver_persona import DriverPersona
from agent.llm_persona_extractor import LLMPersonaConfig
from agent.loop import StrategyFieldEngine
from agent import time_tools


class ModelDecisionService:
    """评测主循环固定调用的对外门面：内部是策略场引擎。"""

    def __init__(
        self,
        api: SimulationApiPort,
        llm_config: LLMPersonaConfig | None = None,
    ) -> None:
        self._api = api
        self._llm_config = llm_config
        self._logger = logging.getLogger("agent.decision_service")
        self._persona_token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
        self._driver_personas: dict[str, DriverPersona] = {}
        self._persona_token_reported: dict[str, bool] = {}

        self._engine = StrategyFieldEngine(
            api,
            reposition_speed_km_per_hour=_load_reposition_speed_km_per_hour(),
            simulation_duration_days=time_tools.DURATION_DAYS,
            persona_builder=self._build_parsed_preferences,
        )

    def _ensure_persona(self, driver_id: str) -> DriverPersona:
        """通过 ApiPort 获取司机偏好，懒加载构建 DriverPersona；偏好变化时重建。"""
        status = self._api.get_driver_status(driver_id)
        prefs = status.get("preferences", [])
        current_key = self._prefs_fingerprint(prefs)

        existing = self._driver_personas.get(driver_id)
        if existing is not None:
            cached_key = getattr(existing, "_pref_fingerprint", None)
            if cached_key == current_key:
                return existing
            self._logger.info(
                "rebuilding persona driver_id=%s (prefs changed)",
                driver_id,
            )

        cost_per_km = float(status.get("cost_per_km", 1.5))
        persona = DriverPersona(
            driver_id,
            prefs,
            cost_per_km=cost_per_km,
            llm_config=self._llm_config,
            chat_func=self._api.model_chat_completion,
            token_accumulator=self._persona_token_usage,
        )
        persona._pref_fingerprint = current_key
        self._driver_personas[driver_id] = persona
        self._logger.info(
            "built persona driver_id=%s prefs=%s cost_per_km=%s",
            driver_id, len(prefs), cost_per_km,
        )
        return persona

    @staticmethod
    def _prefs_fingerprint(prefs: list[Any]) -> str:
        """偏好指纹：仅基于 content 文本，忽略元数据。"""
        contents: list[str] = []
        for p in prefs:
            if isinstance(p, str):
                contents.append(p)
            elif isinstance(p, dict):
                contents.append(p.get("content") or p.get("text", "") or "")
        return json.dumps(contents, ensure_ascii=False, sort_keys=True)

    def _build_parsed_preferences(
        self,
        driver_id: str,
        raw_prefs: list[dict[str, Any]],
    ) -> list[Any]:
        """画像提取结果 -> ParsedPreference 列表（供 StrategyFieldEngine 消费）。"""
        persona_obj = self._ensure_persona(driver_id)
        persona = persona_obj.to_dict(sparse=True) if hasattr(persona_obj, "to_dict") else {}
        parsed = persona_to_parsed_preferences(raw_prefs, persona)
        self._logger.info(
            "persona_adapter driver_id=%s raw_prefs=%s parsed=%s",
            driver_id, len(raw_prefs), len(parsed),
        )
        return parsed

    def decide(self, driver_id: str) -> dict[str, Any]:
        return self._engine.decide(driver_id)

    def parsed_prefs_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """转发引擎的已解析偏好缓存，供编排器落盘 per-run 偏好文件。"""
        return self._engine.parsed_prefs_snapshot()