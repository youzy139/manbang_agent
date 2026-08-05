"""司机画像：从 drivers.json 的 preferences 中解析结构化约束与偏好。"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable

from agent.llm_persona_extractor import LLMPersonaConfig, extract_with_llm

_LOGGER = logging.getLogger(__name__)


# 常见品类关键词（用于 cargo_name -> 品类映射）
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "建材": ["瓷砖", "水泥", "石材", "钢筋", "木材", "板材", "玻璃", "卫浴", "地板", "油漆涂料"],
    "家具家电": ["家具", "沙发", "床垫", "家电", "冰箱", "洗衣机", "空调", "电视", "灯具", "家居"],
    "食品饮料": ["食品", "饮料", "酒水", "生鲜", "果蔬", "冷冻", "速冻", "乳制品", "调味品", "水果", "蔬菜"],
    "服装纺织": ["服装", "纺织", "皮革", "鞋帽", "箱包", "面料", "纱线", "成衣"],
    "电子数码": ["电子", "数码", "手机", "电脑", "芯片", "元器件", "电路板", "显示器"],
    "化工塑料": ["化工", "塑料", "橡胶", "化肥", "农药", "涂料", "油墨", "树脂", "聚合物"],
    "机械设备": ["机械", "设备", "机床", "模具", "零部件", "五金", "工具", "配件", "仪表", "车辆"],
    "纸品印刷": ["纸", "纸箱", "包装", "印刷", "书刊", "标签", "纸盒"],
    "医药保健": ["医药", "药品", "医疗器械", "保健品", "药材", "疫苗", "试剂"],
    "汽车配件": ["汽配", "轮胎", "发动机", "变速箱", "刹车", "滤清器", "车灯"],
    "农产品": ["粮食", "饲料", "种子", "棉花", "油料", "糖料", "茶叶", "烟草", "农产品", "农用物资", "经济作物", "玉米", "谷物"],
    "钢铁金属": ["钢铁", "金属", "铜", "铝", "锌", "合金", "线材", "型材", "钢材"],
    "日用百货": ["日用", "百货", "洗化", "玩具", "文具", "礼品", "餐具", "厨具", "办公", "体育", "设施"],
    "快递包裹": ["快递", "包裹", "快件", "邮包", "物流件"],
    "煤炭矿产": ["煤炭", "矿产", "煤", "矿"],
    "废品废料": ["废品", "废料", "废旧", "回收"],
    "鲜活农产品": ["活禽", "活畜", "活虫", "鲜活", "水产品"],
}


def _normalize_pref_dicts(raw_prefs: list[Any]) -> list[dict[str, Any]]:
    """将混合格式的偏好列表统一为 dict 列表（确保 content/key 存在）。"""
    out: list[dict[str, Any]] = []
    for p in raw_prefs:
        if isinstance(p, dict):
            p = dict(p)  # 浅拷贝，不修改原始 dict
            # 统一 content 字段（兼容 "text"/"preference_text" 等别名）
            content = p.get("content") or p.get("text") or p.get("preference_text", "") or ""
            p["content"] = str(content)
            p.setdefault("penalty_amount", 0)
            p.setdefault("penalty_cap", None)
            out.append(p)
        elif isinstance(p, str):
            out.append({"content": p, "penalty_amount": 0, "penalty_cap": None})
    return out




def _post_process_by_driver_id(driver_id: str, persona: dict[str, Any]) -> dict[str, Any]:
    """按司机 ID 做后处理特例调整。"""
    return persona


def _guess_category(cargo_name: str) -> str | None:
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in cargo_name:
                return category
    return None


def _extract_coords_from_prefs(raw_preferences: list[Any]) -> dict[str, dict[str, float]]:
    """从原始偏好文本中提取 (纬度, 经度) 坐标及其关联名称，用于补充 known_locations。"""
    locations: dict[str, dict[str, float]] = {}
    for p in raw_preferences:
        text = str(p.get("content", p)) if isinstance(p, dict) else str(p)
        for m in re.finditer(r"[（(]\s*(\d+\.\d+)\s*[，,]\s*(\d+\.\d+)\s*[）)]", text):
            lat, lng = float(m.group(1)), float(m.group(2))
            # 向前查找地名
            prefix = text[:m.start()]
            name_m = re.search(r"([一-鿿]{2,8}(?:区|市|县|镇|街道|乡|村|一带|附近|家里|附近))[^）)]*$", prefix)
            if name_m:
                name = name_m.group(1)
                if name not in locations and len(name) >= 2:
                    locations[name] = {"lat": lat, "lng": lng}
            # 如果没有找到地名，用坐标本身作为 key
            key = f"coord_{lat}_{lng}"
            if key not in locations:
                locations[key] = {"lat": lat, "lng": lng}
    return locations


def _chinese_number_to_int(text: str) -> int | None:
    """将常见中文数字（1-99）转为整数，失败返回 None。"""
    if not text:
        return None
    if text.isdigit():
        return int(text)
    direct = {
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
        "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
        "二十一": 21, "二十二": 22, "二十三": 23, "二十四": 24, "二十五": 25,
        "二十六": 26, "二十七": 27, "二十八": 28, "二十九": 29, "三十": 30,
        "三十一": 31,
    }
    if text in direct:
        return direct[text]
    m = re.match(r"([一二三四五六七八九十])([一二三四五六七八九])", text)
    if m:
        tens = {"一": 10, "二": 20, "三": 30, "四": 40, "五": 50,
                "六": 60, "七": 70, "八": 80, "九": 90}
        ones = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9}
        return tens.get(m.group(1), 0) + ones.get(m.group(2), 0)
    return None


def _extract_special_events_from_texts(raw_preferences: list[Any]) -> list[dict[str, Any]]:
    """从原始偏好文本中提取多步骤特殊事件（如寿宴：先到A，再到B）。

    识别形如：
      三月三十一号舅公做寿，上午得先过增城区档口捎上寿礼，
      中午十二点前赶到四会县城（23.32，112.83）赴宴到下午两点。
    输出 special_events 条目，供 _apply_upcoming_event_proximity_bonus 使用。
    """
    events: list[dict[str, Any]] = []
    event_keywords = ("寿宴", "做寿", "赴宴", "婚礼", "宴席", "喜宴", "宴会", "聚餐", "聚会")
    for p in raw_preferences:
        text = str(p.get("content", p)) if isinstance(p, dict) else str(p)
        if not any(kw in text for kw in event_keywords):
            continue

        m_date = re.search(r"([一二三四五六七八九十百]+)月\s*([一二三四五六七八九十百]+)[号日]", text)
        if not m_date:
            continue
        month = _chinese_number_to_int(m_date.group(1))
        day = _chinese_number_to_int(m_date.group(2))
        if month is None or day is None:
            continue

        # 提取所有 地名 + 坐标
        locations: list[dict[str, Any]] = []
        for m in re.finditer(
            r"([一-鿿]{2,8}(?:区|市|县|镇|街道|乡|村))[^（(]*[（(]\s*(\d+\.\d+)\s*[，,]\s*(\d+\.\d+)\s*[）)]",
            text,
        ):
            locations.append({
                "name": m.group(1),
                "lat": float(m.group(2)),
                "lng": float(m.group(3)),
                "pos": m.start(),
            })

        if not locations:
            continue

        # 判断顺序：查找"先...再/然后..."结构
        primary = locations[0]
        secondary = None
        if len(locations) >= 2:
            first_pos = text.find("先")
            second_pos = -1
            for marker in ("再", "然后", "接着"):
                pos = text.find(marker)
                if pos != -1 and (second_pos == -1 or pos < second_pos):
                    second_pos = pos
            if first_pos != -1 and second_pos != -1 and first_pos < second_pos:
                primary_candidates = [loc for loc in locations if first_pos < loc["pos"] < second_pos]
                secondary_candidates = [loc for loc in locations if loc["pos"] > second_pos]
                if primary_candidates and secondary_candidates:
                    primary = min(primary_candidates, key=lambda x: x["pos"])
                    secondary = min(secondary_candidates, key=lambda x: x["pos"])
                else:
                    primary, secondary = locations[0], locations[1]
            else:
                primary, secondary = locations[0], locations[1]

        # 提取 secondary 截止时间，默认中午12点
        deadline_hour = 12
        m_time = re.search(r"(?:中午|上午|下午)?\s*(\d{1,2}|[一二三四五六七八九十]+)\s*点\s*前", text)
        if m_time:
            parsed_hour = _chinese_number_to_int(m_time.group(1))
            if parsed_hour is not None:
                deadline_hour = parsed_hour

        event: dict[str, Any] = {
            "type": "special_events",
            "month": month,
            "day": day,
            "multi_step": len(locations) >= 2,
            "prepare_day_before": True,
            "primary_name": primary["name"],
            "target_name": primary["name"],
            "target_lat": primary["lat"],
            "target_lng": primary["lng"],
            "secondary_lat": secondary["lat"] if secondary else None,
            "secondary_lng": secondary["lng"] if secondary else None,
            "secondary_radius_km": 30.0,
            "secondary_deadline_hour": deadline_hour,
            "secondary_stay_until_hour": 14,
            "wait_minutes": 120,
            "radius_km": 30.0,
            "date": f"2026-{month:02d}-{day:02d}",
            "description": text[:80],
            "penalty": 5000,
            "cap": None,
            "raw": text,
        }
        events.append(event)
    return events


# ================================================================
# 以下函数从 _prefs.py 迁移而来，作为 DriverPersona 的补充提取器
# ================================================================


class DriverPersona:
    """司机画像：解析自然语言偏好为结构化约束。"""

    def __init__(self, driver_id: str, raw_preferences: list[Any],
                 cost_per_km: float = 1.5,
                 llm_config: LLMPersonaConfig | None = None,
                 chat_func: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
                 token_accumulator: dict[str, int] | None = None) -> None:
        self.driver_id = driver_id
        # 防御性过滤：事件触发型偏好（带 trigger 结构）不进 persona LLM——trigger 模型看不到，
        # content 里"触发后再也不…"类措辞会被误读成立即生效。由 event_watcher 确定性处理。
        self.raw_preferences = [
            p for p in raw_preferences
            if not (isinstance(p, dict) and (isinstance(p.get("trigger"), dict)
                                             or str(p.get("type") or "").strip() == "事件触发型"))
        ]
        self.cost_per_km = float(cost_per_km)
        self._llm_config = llm_config
        self._chat_func = chat_func
        self._token_accumulator = token_accumulator
        self.persona_token_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0,
                                                      "reasoning_tokens": 0, "total_tokens": 0}
        self._parsed = self._parse()

    def _build_enhanced_messages(self) -> list[dict[str, str]]:
        """构建增强版 LLM 消息，复用 llm_persona_extractor 的 _build_messages。"""
        from agent.llm_persona_extractor import _build_messages
        pref_dicts: list[dict[str, Any]] = []
        for p in self.raw_preferences:
            if isinstance(p, str):
                pref_dicts.append({"content": p, "penalty_amount": 0, "penalty_cap": None, "start_time": "", "end_time": ""})
            elif isinstance(p, dict):
                pref_dicts.append({
                    "content": p.get("content") or p.get("text", ""),
                    "penalty_amount": p.get("penalty_amount", 0),
                    "penalty_cap": p.get("penalty_cap"),
                    "start_time": p.get("start_time", ""),
                    "end_time": p.get("end_time", ""),
                })
        return _build_messages(pref_dicts)

    def _parse(self) -> dict[str, Any]:
        text_prefs: list[str] = []
        pref_dicts: list[dict[str, Any]] = _normalize_pref_dicts(self.raw_preferences)
        for p in self.raw_preferences:
            if isinstance(p, str):
                text_prefs.append(p)
            elif isinstance(p, dict):
                content = p.get("content") or p.get("text")
                if content:
                    text_prefs.append(str(content))

        # ============================================================
        # 第一步：初始化默认值（画像提取全权交给 LLM）
        # ============================================================
        regex_persona: dict[str, Any] = {
            "forbidden_hours": [],
            "max_pickup_deadhead_km": [],
            "max_haul_km": [],
            "max_deadhead_km": [],
            "cargo_avoidance": [],
            "monthly_rest_days": [],
            "daily_order_limit": [],
            "first_order_before": [],
            "geo_boundary": [],
            "forbidden_zone": [],
            "preferred_order_regions": [],
            "must_visit": [],
            "must_take": [],
            "special_events": [],
            "home_event": [],
            "known_locations": {"home": {"lat": None, "lng": None}},
            "fixed_stationary_window": None,
            "daily_continuous_rest_minutes": None,
            "monthly_kpi": None,
            "monthly_long_haul_cap": None,
            "driving_limits": None,
            "sequence_constraints": [],
            "activation_guard": None,
        }

        # ============================================================
        # 第二步：LLM 主力提取（覆盖默认值）
        # ============================================================
        llm_available = self._chat_func is not None or (self._llm_config and self._llm_config.api_key)
        llm_succeeded = False
        persona = dict(regex_persona)  # 默认使用默认值

        if llm_available:
            for attempt in range(1, 3):
                attempt_accum: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0,
                                                  "reasoning_tokens": 0, "total_tokens": 0}
                try:
                    enhanced_messages = self._build_enhanced_messages()
                    llm_result = extract_with_llm(self.raw_preferences, config=self._llm_config,
                                                  chat_func=self._chat_func,
                                                  token_accumulator=attempt_accum,
                                                  custom_messages=enhanced_messages)
                    if llm_result is not None:
                        # 记录 token 用量
                        for k in self.persona_token_usage:
                            self.persona_token_usage[k] += attempt_accum.get(k, 0)
                        if self._token_accumulator is not None and attempt_accum.get("total_tokens", 0) > 0:
                            for k in self._token_accumulator:
                                self._token_accumulator[k] += attempt_accum.get(k, 0)

                        # LLM 结果覆盖所有字段，null/空字段保留默认值
                        for key in regex_persona:
                            llm_val = llm_result.get(key)
                            if llm_val is not None and llm_val != "" and llm_val != [] and llm_val != {}:
                                persona[key] = llm_val
                            # 否则保留默认值

                        # 列表字段：需要 normalize
                        list_fields = [
                            "forbidden_hours", "geo_boundary", "forbidden_zone",
                            "max_haul_km", "max_pickup_deadhead_km", "max_deadhead_km",
                            "monthly_rest_days", "daily_order_limit", "first_order_before",
                            "special_events", "home_event", "preferred_order_regions",
                            "must_take", "must_visit", "sequence_constraints",
                        ]
                        for field in list_fields:
                            raw_val = llm_result.get(field)
                            if isinstance(raw_val, list) and raw_val:
                                persona[field] = self._normalize_llm_field_list(field, raw_val)
                            # 否则保留默认值（已在 persona 中）

                        # 字典字段
                        dict_fields = ["known_locations", "driving_limits", "activation_guard"]
                        for field in dict_fields:
                            raw_val = llm_result.get(field)
                            if isinstance(raw_val, dict) and raw_val:
                                llm_dict = dict(raw_val)
                                # 合并坐标到 known_locations
                                if field == "known_locations":
                                    coord_locs = _extract_coords_from_prefs(self.raw_preferences)
                                    for name, coords in coord_locs.items():
                                        if name not in llm_dict:
                                            llm_dict[name] = coords
                                persona[field] = llm_dict
                            # driving_limits/activation_guard 允许为空 dict，但保留默认值

                        persona = self._filter_null_list_entries(persona)
                        llm_succeeded = True
                        _LOGGER.info("LLM persona OK driver_id=%s", self.driver_id)
                        break
                except Exception as exc:
                    _LOGGER.warning("LLM persona attempt %s/2 failed for %s: %s",
                                    attempt, self.driver_id, exc)
                    if attempt < 2:
                        time.sleep(3)

        if not llm_succeeded and llm_available:
            _LOGGER.warning("LLM persona failed for %s after 2 attempts, using regex fallback",
                            self.driver_id)

        # 本地正则补充 special_events（即使 LLM 失败也能覆盖寿宴类事件）
        if not persona.get("special_events"):
            fallback_events = _extract_special_events_from_texts(self.raw_preferences)
            if fallback_events:
                persona["special_events"] = fallback_events
                _LOGGER.info("regex fallback special_events driver_id=%s count=%s",
                             self.driver_id, len(fallback_events))

        persona = _post_process_by_driver_id(self.driver_id, persona)
        return persona

    _FIELD_NORMALIZERS: dict[str, tuple[list[str], dict[str, str]]] = {
        "forbidden_hours": (
            ["type", "start_hour", "start_min", "end_hour", "end_min",
             "no_order", "no_reposition", "penalty", "cap", "raw"],
            {"start": "start_hour", "end": "end_hour"},
        ),
        "geo_boundary": (
            ["type", "lat_min", "lat_max", "lng_min", "lng_max", "penalty", "cap", "raw"],
            {},
        ),
        "forbidden_zone": (
            ["type", "center_lat", "center_lng", "radius_km", "penalty", "cap", "raw"],
            {},
        ),
        "max_haul_km": (
            ["type", "max_km", "penalty", "cap", "raw"],
            {},
        ),
        "max_pickup_deadhead_km": (
            ["type", "max_km", "penalty", "cap", "raw"],
            {},
        ),
        "max_deadhead_km": (
            ["type", "max_km", "penalty", "cap", "raw"],
            {},
        ),
        "monthly_rest_days": (
            ["type", "min_days", "penalty", "cap", "raw"],
            {},
        ),
        "daily_order_limit": (
            ["type", "max_orders", "penalty", "cap", "raw"],
            {},
        ),
        "first_order_before": (
            ["type", "hour", "minute", "penalty", "raw"],
            {},
        ),
        "special_events": (
            ["type", "month", "day", "multi_step", "prepare_day_before",
             "primary_name", "target_name", "target_lat", "target_lng",
             "secondary_lat", "secondary_lng", "secondary_radius_km",
             "secondary_deadline_hour", "secondary_stay_until_hour",
             "wait_minutes", "radius_km", "date", "description",
             "penalty", "cap", "raw"],
            {
                "deadline_hour": "secondary_deadline_hour",
                "stay_until_hour": "secondary_stay_until_hour",
                "primary_location": "primary_name",
                "first_place": "primary_name",
                "first_stop": "primary_name",
                "pickup_place": "primary_name",
                "pickup_name": "primary_name",
                "intermediate_name": "primary_name",
                "intermediate_loc": "primary_name",
                "first_location": "primary_name",
            },
        ),
        "home_event": (
            ["type", "spouse_lat", "spouse_lng", "home_lat", "home_lng",
             "stay_minutes", "penalty_per_min", "deadline", "event_end",
             "start_time", "penalty", "cap", "raw"],
            {
                # deadline 相关别名
                "home_arrival_deadline": "deadline", "home_deadline": "deadline",
                # event_end 相关别名
                "stay_until": "event_end", "stay_end": "event_end",
                "min_stay_end": "event_end",
                # start_time 相关别名
                "pickup_deadline": "start_time", "stay_start": "start_time",
                "min_stay_start": "start_time", "event_start": "start_time",
                # spouse 坐标别名
                "pickup_lat": "spouse_lat", "pickup_lng": "spouse_lng",
                # penalty 别名
                "penalty_per_minute": "penalty_per_min",
                "penalty_violation": "penalty", "penalty_miss": "penalty",
                # cap 别名
                "penalty_cap": "cap",
            },
        ),
        "preferred_order_regions": (
            ["type", "region_keyword", "min_days", "penalty", "cap", "raw"],
            {},
        ),
        "must_take": (
            ["type", "cargo_id", "pickup_lat", "pickup_lng", "available_at",
             "start_time", "end_time", "penalty", "cap", "raw"],
            {},
        ),
        "must_visit": (
            ["type", "target_lat", "target_lng", "min_days", "radius_km",
             "penalty", "cap", "raw"],
            {},
        ),
        "sequence_constraints": (
            ["type", "relation", "distinct_key", "max_run", "category",
             "antecedent", "consequent", "window_n", "comparator", "value",
             "penalty_fn", "penalty", "cap", "raw"],
            {},
        ),
    }

    @staticmethod
    def _normalize_llm_field_list(field: str, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将 LLM 格式的规则列表转换成决策引擎期望的格式。"""
        info = DriverPersona._FIELD_NORMALIZERS.get(field)
        if not info:
            return rules
        expected_keys, key_map = info

        result: list[dict[str, Any]] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            normalized: dict[str, Any] = {"type": field, "raw": ""}
            for llm_key, val in rule.items():
                target = key_map.get(llm_key, llm_key)
                # LLM 有时返回列表而非标量（如 "cap": [74400]），
                # 展开为首个元素以避免下游 TypeError（列表 <= int）。
                if isinstance(val, list):
                    val = val[0] if val else None
                normalized[target] = val
            if field == "forbidden_hours":
                normalized.setdefault("start_min", 0)
                normalized.setdefault("end_min", 0)
                normalized.setdefault("no_order", True)
                normalized.setdefault("no_reposition", False)
            if field == "first_order_before":
                normalized.setdefault("minute", 0)
            if field == "home_event":
                normalized.setdefault("stay_minutes", 10)
                normalized.setdefault("penalty_per_min", 5)
                # LLM 可能输出 null/None 日期字段，替换为默认值
                for date_key, default_val in (("deadline", ""), ("event_end", ""), ("start_time", "")):
                    val = normalized.get(date_key)
                    if val is None or val == "":
                        normalized[date_key] = default_val
                    elif isinstance(val, str) and "T" in val:
                        normalized[date_key] = val.replace("T", " ")
            result.append(normalized)
        return result

    @staticmethod
    def _filter_null_list_entries(persona: dict[str, Any]) -> dict[str, Any]:
        """过滤列表类型字段中所有关键字段都为 null 的占位条目。

        LLM 有时会输出 `[{"hour": null, "home_lat": null, ...}]` 占位，
        此方法过滤掉这些无效条目，避免下游崩溃。
        """
        _KEY_FIELDS: dict[str, list[str]] = {
            "forbidden_hours": ["start_hour", "end_hour"],
            "max_pickup_deadhead_km": ["max_km"],
            "max_haul_km": ["max_km"],
            "max_deadhead_km": ["max_km"],
            "monthly_rest_days": ["min_days"],
            "daily_order_limit": ["max_orders"],
            "first_order_before": ["hour"],
            "geo_boundary": ["lat_min", "lat_max", "lng_min", "lng_max"],
            "forbidden_zone": ["center_lat", "center_lng", "radius_km"],
            "preferred_order_regions": ["region_keyword"],
            "must_visit": ["target_lat", "target_lng"],
            "must_take": ["cargo_id"],
            "special_events": ["month", "day"],
            "home_event": ["spouse_lat", "spouse_lng", "home_lat", "home_lng"],
            "sequence_constraints": ["relation"],
        }
        # 取值范围校验：字段 → {key: (min, max)}
        _VALUE_RANGES: dict[str, dict[str, tuple[int, int]]] = {
            "forbidden_hours": {"start_hour": (0, 23), "end_hour": (0, 23)},
            "first_order_before": {"hour": (0, 23)},
            "special_events": {"month": (1, 12), "day": (1, 31)},
        }
        result = dict(persona)
        for field, key_fields in _KEY_FIELDS.items():
            val = result.get(field)
            if not isinstance(val, list):
                continue
            ranges = _VALUE_RANGES.get(field, {})
            filtered = []
            for entry in val:
                if not isinstance(entry, dict):
                    filtered.append(entry)
                    continue
                # 跳过所有关键字段都为 null/"" 的占位条目
                if not any(entry.get(k) is not None and entry.get(k) != "" for k in key_fields):
                    continue
                # 取值范围校验：超出范围的条目丢弃
                in_range = True
                for k, (lo, hi) in ranges.items():
                    v = entry.get(k)
                    if v is not None and not (isinstance(v, (int, float)) and lo <= v <= hi):
                        in_range = False
                        break
                if in_range:
                    filtered.append(entry)
            result[field] = filtered
        return result

    @staticmethod
    def _strip_empty(obj: Any) -> Any:
        """递归删除 None、空列表、空字典。"""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                stripped = DriverPersona._strip_empty(v)
                if stripped is None:
                    continue
                if isinstance(stripped, (list, dict)) and len(stripped) == 0:
                    continue
                result[k] = stripped
            return result
        if isinstance(obj, list):
            result = [DriverPersona._strip_empty(i) for i in obj if i is not None]
            return [i for i in result if not (isinstance(i, (list, dict)) and len(i) == 0)] or None
        return obj

    @staticmethod
    def _clean_entry(entry: dict[str, Any]) -> dict[str, Any]:
        """清理单个约束条目：去掉 raw、type 等内部噪声。"""
        entry.pop("raw", None)
        entry.pop("type", None)
        return entry

    def to_dict(self, sparse: bool = False) -> dict[str, Any]:
        d = dict(self._parsed)
        d["cost_per_km"] = self.cost_per_km
        d["raw_preferences"] = self.raw_preferences
        if sparse:
            # 清理列表条目中的内部字段
            for key, val in d.items():
                if isinstance(val, list):
                    d[key] = [self._clean_entry(e) for e in val if isinstance(e, dict)]
            d = self._strip_empty(d)
        return d

    def is_cargo_blacklisted(self, cargo_name: str) -> bool:
        blacklist = self._parsed.get("cargo_avoidance", [])
        return any(bl in cargo_name for bl in blacklist)

