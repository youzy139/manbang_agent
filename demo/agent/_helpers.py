"""模型决策服务：依赖 `simkit.ports.SimulationApiPort`，由评测进程注入具体环境。

本版本整合 v0509 的多阶段评分管线与参考版的约束提取能力。
"""

from __future__ import annotations

import collections.abc
import hashlib
import math
import os
import re
from datetime import datetime, timedelta
from typing import Any

SIM_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
WALL_FMT = "%Y-%m-%d %H:%M:%S"

# ================================================================
# v0509 移植：全局常量
# ================================================================
_SIMULATION_EPOCH = SIM_EPOCH
_SIMULATION_HORIZON_MINUTES = 92 * 24 * 60
_SIMULATION_END = _SIMULATION_EPOCH + timedelta(minutes=_SIMULATION_HORIZON_MINUTES)
_WALL_TIME_FMT = WALL_FMT
_TIME_FMT = "%H:%M:%S"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


_DEFAULT_WAIT_MINUTES = _env_int("MDS_DEFAULT_WAIT_MINUTES", 60)
_CARGO_LIMIT_PER_STEP = _env_int("MDS_CARGO_LIMIT_PER_STEP", 200)
_ACTIVE_TASK_LOOKAHEAD_DAYS = _env_int("MDS_ACTIVE_TASK_LOOKAHEAD_DAYS", 3)

_ACTIONS = {"wait", "reposition", "take_order"}
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")
_CN_DT_RE = re.compile(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日\s*(\d{1,2})(?:[:点](\d{1,2}))?")

_HARD_RULE_VISIT_SLACK_DAYS = _env_int("MDS_HARD_RULE_VISIT_SLACK_DAYS", 4)
_HARD_RULE_ENDPOINT_SEARCH_SLACK_DAYS = _env_int("MDS_HARD_RULE_ENDPOINT_SEARCH_SLACK_DAYS", 8)
_HARD_RULE_ENDPOINT_SEARCH_VALUE_MARGIN = _env_float("MDS_HARD_RULE_ENDPOINT_SEARCH_VALUE_MARGIN", 1.5)
_MONTHLY_OFF_DAY_RESERVATION_SLACK_DAYS = _env_int("MDS_MONTHLY_OFF_DAY_RESERVATION_SLACK_DAYS", 4)
_MONTHLY_OFF_DAY_REFRESH_MINUTES = _env_int("MDS_MONTHLY_OFF_DAY_REFRESH_MINUTES", 60)
_DEFER_PROACTIVE_REPOSITION_FOR_FIXED_WAIT = _env_int("MDS_DEFER_PROACTIVE_REPOSITION_FOR_FIXED_WAIT", 0) != 0
_MARKET_GRID_DEGREES = _env_float("MDS_MARKET_GRID_DEGREES", 0.1)
_MARKET_FALLBACK_MIN_QUERIES = _env_int("MDS_MARKET_FALLBACK_MIN_QUERIES", 2)
_MARKET_FALLBACK_MIN_GAIN_PER_HOUR = _env_float("MDS_MARKET_FALLBACK_MIN_GAIN_PER_HOUR", 8.0)
_MARKET_FALLBACK_MAX_REPOSITION_KM = _env_float("MDS_MARKET_FALLBACK_MAX_REPOSITION_KM", 80.0)
_MARKET_FALLBACK_EVALUATION_HOURS = _env_float("MDS_MARKET_FALLBACK_EVALUATION_HOURS", 2.0)
_MARKET_DESTINATION_VALUE_HOURS = _env_float("MDS_MARKET_DESTINATION_VALUE_HOURS", 1.0)
_MARKET_DESTINATION_VALUE_MAX_BONUS = _env_float("MDS_MARKET_DESTINATION_VALUE_MAX_BONUS", 80.0)
_MARKET_QUERY_TOP_K = _env_int("MDS_MARKET_QUERY_TOP_K", 5)
_MARKET_TOP_OPPORTUNITY_WEIGHT = _env_float("MDS_MARKET_TOP_OPPORTUNITY_WEIGHT", 0.0)
_MARKET_PICKUP_FUTURE_VALUE_HOURS = _env_float("MDS_MARKET_PICKUP_FUTURE_VALUE_HOURS", 0.0)
_MARKET_PICKUP_FUTURE_VALUE_MAX_BONUS = _env_float("MDS_MARKET_PICKUP_FUTURE_VALUE_MAX_BONUS", 0.0)
_MARKET_FALLBACK_PICKUP_OPPORTUNITY_WEIGHT = _env_float("MDS_MARKET_FALLBACK_PICKUP_OPPORTUNITY_WEIGHT", 0.0)
_MARKET_FALLBACK_PICKUP_MIN_OBSERVATIONS = _env_int("MDS_MARKET_FALLBACK_PICKUP_MIN_OBSERVATIONS", 3)
_MARKET_EXPLORATION_UCB_WEIGHT = _env_float("MDS_MARKET_EXPLORATION_UCB_WEIGHT", 3.8)
_MARKET_EXPLORATION_MAX_BONUS = _env_float("MDS_MARKET_EXPLORATION_MAX_BONUS", 10.0)
_MARKET_EXPLORATION_JITTER = _env_float("MDS_MARKET_EXPLORATION_JITTER", 0.0)
_MARKET_EXPLORATION_SEED = _env_int("MDS_MARKET_EXPLORATION_SEED", 0)
_MARKET_OPPORTUNITY_MEMORY_MAX_AREAS = _env_int("MDS_MARKET_OPPORTUNITY_MEMORY_MAX_AREAS", 96)
_INFORMATION_REFRESH_TARGET_MINUTES = _env_int("MDS_INFORMATION_REFRESH_TARGET_MINUTES", 240)
_INFORMATION_REFRESH_COST_PER_EXCESS_MINUTE = _env_float("MDS_INFORMATION_REFRESH_COST_PER_EXCESS_MINUTE", 0.8)
_STATUS_REFRESH_CHECKPOINT_MINUTES = _env_int("MDS_STATUS_REFRESH_CHECKPOINT_MINUTES", 6 * 60)
_STATUS_REFRESH_CHECKPOINT_COST = _env_float("MDS_STATUS_REFRESH_CHECKPOINT_COST", 20.0)
_STATUS_REFRESH_DELAY_COST_PER_MINUTE = _env_float("MDS_STATUS_REFRESH_DELAY_COST_PER_MINUTE", 0.0)
_PENDING_WAIT_REFRESH_MAX_MINUTES = _env_int("MDS_PENDING_WAIT_REFRESH_MAX_MINUTES", 480)
_FIXED_WINDOW_TASK_START_BUFFER_MINUTES = _env_int("MDS_FIXED_WINDOW_TASK_START_BUFFER_MINUTES", 20)
_TASK_QUERY_SAFETY_MINUTES = _env_int("MDS_TASK_QUERY_SAFETY_MINUTES", 10)
_DAILY_WAIT_SLACK_TARGET_MINUTES = _env_int("MDS_DAILY_WAIT_SLACK_TARGET_MINUTES", 120)
_DAILY_WAIT_SLACK_SCORE_WEIGHT = _env_float("MDS_DAILY_WAIT_SLACK_SCORE_WEIGHT", 0.0)
_CANDIDATE_INCOME_WEIGHT = _env_float("MDS_CANDIDATE_INCOME_WEIGHT", 0.6)
_CANDIDATE_EFFICIENCY_WEIGHT = _env_float("MDS_CANDIDATE_EFFICIENCY_WEIGHT", 0.4)
_PEL_SCORE_MULTIPLIER = _env_float("MDS_PEL_SCORE_MULTIPLIER", 5.0)
_STRICT_PEL_SCORE_PENALTY = _env_float("MDS_STRICT_PEL_SCORE_PENALTY", 5000.0)
_STRICT_PEL_DROP_WHEN_ALTERNATIVE = _env_int("MDS_STRICT_PEL_DROP_WHEN_ALTERNATIVE", 1) != 0
_STRICT_PEL_WAIT_WHEN_ONLY_VIOLATIONS = _env_int("MDS_STRICT_PEL_WAIT_WHEN_ONLY_VIOLATIONS", 1) != 0
_NUMERIC_HARD_RULE_DROP_WHEN_ALTERNATIVE = _env_int("MDS_NUMERIC_HARD_RULE_DROP_WHEN_ALTERNATIVE", 0) != 0
_DAILY_ORDER_RULES_ENABLED = _env_int("MDS_DAILY_ORDER_RULES_ENABLED", 0) != 0
_NUMERIC_HARD_RULE_SCORE_MULTIPLIER = _env_float("MDS_NUMERIC_HARD_RULE_SCORE_MULTIPLIER", 1.0)
_NUMERIC_HARD_RULE_EXCESS_KM_SCORE = _env_float("MDS_NUMERIC_HARD_RULE_EXCESS_KM_SCORE", 0.0)
_FEWSHOT_CASE_LIMIT = _env_int("MDS_FEWSHOT_CASE_LIMIT", 100)

# ================================================================
# 鲁棒优化：三层架构配置常量
# ================================================================
_DAILY_KPI_TARGET_BUFFER = _env_float("MDS_DAILY_KPI_TARGET_BUFFER", 1.2)
_LONG_HAUL_QUOTA_MULTIPLIER = _env_float("MDS_LONG_HAUL_QUOTA_MULTIPLIER", 1.5)
_REST_PROTECTION_HOUR = _env_int("MDS_REST_PROTECTION_HOUR", 18)
_REST_PROTECTION_GAP_HOURS = _env_float("MDS_REST_PROTECTION_GAP_HOURS", 2.0)
_DESERT_CREDIBILITY_THRESHOLD = _env_float("MDS_DESERT_CREDIBILITY_THRESHOLD", 0.2)
_STRUCTURE_FILTER_ENABLED = _env_int("MDS_STRUCTURE_FILTER_ENABLED", 0) != 0
_UNCERTAINTY_GAMMA = _env_float("MDS_UNCERTAINTY_GAMMA", 0.5)
_PENALTY_ROOT_CAUSE_WINDOW = _env_int("MDS_PENALTY_ROOT_CAUSE_WINDOW", 50)
_PENALTY_ROOT_CAUSE_THRESHOLD = _env_float("MDS_PENALTY_ROOT_CAUSE_THRESHOLD", 0.30)


def _load_reposition_speed_km_per_hour() -> float:
    return 60.0


# ================================================================
# v0509 移植：辅助函数
# ================================================================
def _simulation_minutes_to_dt(minutes: int) -> datetime:
    """把仿真分钟数转成墙钟 datetime。"""
    return _SIMULATION_EPOCH + timedelta(minutes=int(minutes))


def _standardize_time_text(value: Any, keep_date: bool | None = None) -> str:
    """标准化模型返回的时间。"""
    text = str(value or "").strip()
    if not text:
        return ""

    cn = _CN_DT_RE.search(text)
    if cn:
        y, m, d, h, minute = cn.groups()
        dt = datetime(int(y), int(m), int(d), int(h), int(minute or 0), 0)
        return dt.strftime(_WALL_TIME_FMT) if keep_date is not False else dt.strftime(_TIME_FMT)

    if text in {"24:00", "24:00:00"}:
        return "24:00:00" if keep_date is not True else ""

    m = re.fullmatch(
        r"(\d{4})[-/:](\d{1,2})[-/:](\d{1,2})[ T:](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?",
        text,
    )
    if m:
        y, mo, d, h, mi, s = m.groups()
        dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s or 0))
        return dt.strftime(_WALL_TIME_FMT) if keep_date is not False else dt.strftime(_TIME_FMT)

    if _TIME_RE.fullmatch(text):
        h, mi, *sec = text.split(":")
        tod = f"{int(h):02d}:{int(mi):02d}:{int(sec[0]) if sec else 0:02d}"
        return "" if keep_date is True else tod

    return ""


def _parse_dt(value: Any, base_date: datetime | None = None) -> datetime | None:
    """把完整时间或日内时间转成 datetime。"""
    text = str(value or "").strip()
    if not text:
        return None

    full = _standardize_time_text(text, keep_date=True)
    if full:
        return datetime.strptime(full, _WALL_TIME_FMT)

    tod = _standardize_time_text(text, keep_date=False)
    if not tod or base_date is None:
        return None

    if tod == "24:00:00":
        return datetime.combine(base_date.date(), datetime.min.time()) + timedelta(days=1)

    return datetime.combine(base_date.date(), datetime.strptime(tod, _TIME_FMT).time())


def _clamp_recurring_window(start_text: str, end_text: str) -> tuple[str, str]:
    """Keep recurring rules inside the simulation horizon when the model invents dates."""
    start_dt = _parse_dt(start_text)
    end_dt = _parse_dt(end_text)
    horizon_end = _SIMULATION_END - timedelta(seconds=1)

    if (
        start_dt is None
        or end_dt is None
        or end_dt < _SIMULATION_EPOCH
        or start_dt >= _SIMULATION_END
        or end_dt < start_dt
    ):
        return (
            _SIMULATION_EPOCH.strftime(_WALL_TIME_FMT),
            horizon_end.strftime(_WALL_TIME_FMT),
        )

    return (
        max(start_dt, _SIMULATION_EPOCH).strftime(_WALL_TIME_FMT),
        min(end_dt, horizon_end).strftime(_WALL_TIME_FMT),
    )


def _anchor_single_time_to_simulation(value: Any) -> str:
    """Map one explicit single-task timestamp into the active simulation calendar."""
    text = _standardize_time_text(value, keep_date=True)
    dt = _parse_dt(text)
    if dt is None:
        return ""

    if _SIMULATION_EPOCH <= dt < _SIMULATION_END:
        return dt.strftime(_WALL_TIME_FMT)

    try:
        anchored = dt.replace(year=_SIMULATION_EPOCH.year)
    except ValueError:
        return ""

    if _SIMULATION_EPOCH <= anchored < _SIMULATION_END:
        return anchored.strftime(_WALL_TIME_FMT)
    return ""


def _new_preference_ref(source_key: str, index: int) -> str:
    """Build an opaque batch-local reference so context entries cannot be mistaken for new inputs."""
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:12]
    return f"new_{index}_{digest}"


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """计算两点球面距离。"""
    radius_km = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def _distance_to_minutes(distance_km: float, speed_km_per_hour: float) -> int:
    """距离转分钟；正距离至少 1 分钟，0 距离不推进时间。"""
    if distance_km <= 1e-6:
        return 0
    return max(1, math.ceil(distance_km / speed_km_per_hour * 60))


# ================================================================
# 时间感知工具：月份/周末/日检测
# ================================================================

_SIMULATION_MONTH_BOUNDARIES: list[tuple[int, int, int]] | None = None


def _get_month(minutes: int) -> int:
    """Return month (3, 4, or 5) for a given simulation minute count (0-based from March 1)."""
    dt = _simulation_minutes_to_dt(minutes)
    return dt.month


def _is_weekend(minutes: int) -> bool:
    """Return True if the simulation minute falls on Saturday or Sunday."""
    dt = _simulation_minutes_to_dt(minutes)
    return dt.weekday() >= 5


def _is_weekend_dt(dt: datetime) -> bool:
    """Return True if the datetime falls on Saturday or Sunday."""
    return dt.weekday() >= 5


def _get_day_of_month(minutes: int) -> int:
    """Return day of month (1-31) for a given simulation minute."""
    dt = _simulation_minutes_to_dt(minutes)
    return dt.day


def _get_hour(minutes: int) -> int:
    """Return hour (0-23) for a given simulation minute."""
    dt = _simulation_minutes_to_dt(minutes)
    return dt.hour


def _get_month_boundaries() -> list[tuple[int, int, int]]:
    """Return [(month, start_minute, end_minute_exclusive), ...] for Mar-May 2026."""
    global _SIMULATION_MONTH_BOUNDARIES
    if _SIMULATION_MONTH_BOUNDARIES is not None:
        return _SIMULATION_MONTH_BOUNDARIES
    ep = _SIMULATION_EPOCH  # 2026-03-01
    boundaries: list[tuple[int, int, int]] = []
    for m in (3, 4, 5):
        start_dt = datetime(ep.year, m, 1, 0, 0, 0)
        if m == 5:
            end_dt = datetime(ep.year, 6, 1, 0, 0, 0)
        elif m == 4:
            end_dt = datetime(ep.year, 5, 1, 0, 0, 0)
        else:
            end_dt = datetime(ep.year, 4, 1, 0, 0, 0)
        start_min = int((start_dt - ep).total_seconds() // 60)
        end_min = int((end_dt - ep).total_seconds() // 60)
        boundaries.append((m, start_min, end_min))
    _SIMULATION_MONTH_BOUNDARIES = boundaries
    return boundaries


def _get_current_month_range(minutes: int) -> tuple[int, int, int]:
    """Return (month, start_minute, end_minute_exclusive) for the month containing this minute."""
    for m, start_min, end_min in _get_month_boundaries():
        if start_min <= minutes < end_min:
            return (m, start_min, end_min)
    # Fallback: last month
    return _get_month_boundaries()[-1]


# ================================================================
# 空间感知工具：城市/区县层级解析
# ================================================================

# 中国省份/城市层级：省份名 → 城市集合
_PROVINCE_CITY_MAP: dict[str, set[str]] | None = None
# 城市名 → 省份名（反向映射，用于去重）
_CITY_TO_PROVINCE: dict[str, str] | None = None


def _init_region_map() -> None:
    """Initialize region hierarchy from commonly known Chinese administrative divisions."""
    global _PROVINCE_CITY_MAP, _CITY_TO_PROVINCE
    if _PROVINCE_CITY_MAP is not None:
        return
    _PROVINCE_CITY_MAP = {}
    _CITY_TO_PROVINCE = {}
    # 广东省
    guangdong_cities = {"广州市", "深圳市", "珠海市", "汕头市", "佛山市", "韶关市",
                        "湛江市", "肇庆市", "江门市", "茂名市", "惠州市", "梅州市",
                        "汕尾市", "河源市", "阳江市", "清远市", "东莞市", "中山市",
                        "潮州市", "揭阳市", "云浮市"}
    _PROVINCE_CITY_MAP["广东省"] = guangdong_cities
    for city in guangdong_cities:
        _CITY_TO_PROVINCE[city] = "广东省"


def _parse_city(city_str: str) -> dict[str, str]:
    """Parse a Chinese city string like '广东省广州市白云区' into components.

    Returns {'province': '广东省', 'city': '广州市', 'district': '白云区', 'full': '广东省广州市白云区'}.
    Missing components are empty strings.
    """
    _init_region_map()
    text = str(city_str or "").strip()
    if not text:
        return {"province": "", "city": "", "district": "", "full": ""}

    province = ""
    city = ""
    district = ""

    # Extract province (ends with 省/自治区/直辖市)
    prov_match = re.match(r"(.+?(?:省|自治区|市))", text)
    if prov_match:
        province = prov_match.group(1)
        rest = text[len(province):]
    else:
        rest = text

    # Extract city (ends with 市, or the known city name from the region map)
    city_match = re.match(r"(.+?市)", rest)
    if city_match:
        city = city_match.group(1)
        rest = rest[len(city):]
    else:
        # If no "市" suffix, check if the remaining text itself is a known city
        for known_city in _CITY_TO_PROVINCE or {}:
            if rest.startswith(known_city.rstrip("市")):
                city = known_city
                rest = rest[len(known_city.rstrip("市")):]
                break
        if not city and rest:
            # Take the first 2 characters as fallback
            city = rest[:2] + "市" if len(rest) >= 2 else rest
            rest = rest[len(city.rstrip("市")):] if city else rest

    # Remaining is district / town / street
    district = rest.strip()

    return {
        "province": province,
        "city": city,
        "district": district,
        "full": text,
    }


def _get_city_name(city_str: str) -> str:
    """Extract just the city name (e.g., '广州市') from a full city string."""
    return _parse_city(city_str)["city"]


def _get_district_name(city_str: str) -> str:
    """Extract just the district name (e.g., '白云区') from a full city string."""
    return _parse_city(city_str)["district"]


def _same_city(city_str1: str, city_str2: str) -> bool:
    """Return True if both city strings refer to the same city."""
    if not city_str1 or not city_str2:
        return False
    city1 = _get_city_name(city_str1)
    city2 = _get_city_name(city_str2)
    return bool(city1 and city2 and city1 == city2)


def _same_district(city_str1: str, city_str2: str) -> bool:
    """Return True if both city strings refer to the same district (within same city)."""
    if not city_str1 or not city_str2:
        return False
    p1 = _parse_city(city_str1)
    p2 = _parse_city(city_str2)
    if not p1["city"] or not p2["city"] or p1["city"] != p2["city"]:
        return False
    return bool(p1["district"] and p2["district"] and p1["district"] == p2["district"])


def _county_level_city(city_str: str) -> bool:
    """Return True if the city is a county-level city (县级市), which often means shorter haul."""
    # County-level cities typically don't have districts and are relatively small
    parsed = _parse_city(city_str)
    city = parsed["city"].rstrip("市") if parsed["city"] else ""
    # Known county-level cities in Guangdong
    county_level = {"英德", "连州", "兴宁", "陆丰", "阳春", "雷州", "廉江", "吴川",
                    "高州", "化州", "信宜", "四会", "开平", "台山", "鹤山", "恩平",
                    "普宁", "南雄"}
    return city in county_level


# ================================================================
# 参考版原有辅助函数（保留兼容）
# ================================================================
def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, l1 = math.radians(lat1), math.radians(lng1)
    p2, l2 = math.radians(lat2), math.radians(lng2)
    dp, dl = p2 - p1, l2 - l1
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * r * math.asin(math.sqrt(h))


def _sim_min_to_wall(minutes: int) -> datetime:
    return SIM_EPOCH + timedelta(minutes=int(minutes))


def _wall_str_to_sim_min(text: str) -> int:
    dt = datetime.strptime(text.strip(), WALL_FMT)
    return int((dt - SIM_EPOCH).total_seconds() // 60)


def _parse_dt_flexible(text: str) -> datetime | None:
    """灵活解析日期时间字符串，处理 LLM 可能输出的非标准格式。"""
    text = str(text).strip()
    for fmt in (WALL_FMT, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%-m-%-d %H:%M:%S",
                "%Y-%-m-%-d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except (ValueError, AttributeError):
            continue
    try:
        parts = text.replace("T", " ").split()
        date_part = parts[0]
        time_part = parts[1] if len(parts) > 1 else "00:00:00"
        y, m, d = date_part.split("-")
        hh_mm_ss = time_part.split(":")
        return datetime(int(y), int(m), int(d), int(hh_mm_ss[0]), int(hh_mm_ss[1]),
                        int(hh_mm_ss[2]) if len(hh_mm_ss) > 2 else 0)
    except (ValueError, IndexError, TypeError):
        pass
    try:
        import re as _re
        m = _re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
        if m:
            y, mth, d, hh, mm = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
            ss = int(m.group(6)) if m.group(6) else 0
            return datetime(y, mth, d, hh, mm, ss)
    except (ValueError, IndexError, TypeError):
        pass
    return None


def _in_zone(lat: float, lng: float, c_lat: float | None, c_lng: float | None, radius_km: float | None) -> bool:
    if c_lat is None or c_lng is None or radius_km is None:
        return False
    return _haversine_km(lat, lng, c_lat, c_lng) <= radius_km


def _scalar(val: Any, default: Any = None) -> Any:
    """LLM 有时返回列表而非标量（如 cap=[74400]），展开为首个元素。"""
    if isinstance(val, collections.abc.Sequence) and not isinstance(val, (str, bytes)):
        return val[0] if val else default
    return val






def _estimate_active_penalty(
    cargo: dict[str, Any], dist_km: float, persona: dict[str, Any],
    constraint_budget: dict[str, Any] | None = None,
) -> float:
    """估算货源违反「活跃」约束的预期罚金（用于排序时成本扣除）。

    活跃约束 = 无上限约束 + 有上限但尚未封顶的约束。
    已封顶的约束边际罚金为 0，不计入排序。
    预算稀缺时有放大效应：剩余违规次数越少，罚金权重越高（节省预算给更好的单）。
    """
    penalty = 0.0
    budget = constraint_budget or {}
    start = cargo.get("start", {})
    end = cargo.get("end", {})
    start_lat = float(start.get("lat", 0))
    start_lng = float(start.get("lng", 0))
    end_lat = float(end.get("lat", 0))
    end_lng = float(end.get("lng", 0))

    def _scarcity_multiplier(remaining: int | None) -> float:
        """剩余违规次数越少，惩罚倍数越高（节省稀缺预算）。"""
        if remaining is None:
            return 1.0
        if remaining <= 1:
            return 5.0
        if remaining <= 2:
            return 3.0
        if remaining <= 3:
            return 2.0
        return 1.0

    # max_haul_km
    for rule in persona.get("max_haul_km", []):
        cap = _scalar(rule.get("cap"))
        if cap is not None and budget.get("haul_cap_reached"):
            continue
        haul = _haversine_km(start_lat, start_lng, end_lat, end_lng)
        if haul > (_scalar(rule.get("max_km")) or float("inf")):
            base = float(_scalar(rule.get("penalty", 0)))
            mult = _scarcity_multiplier(budget.get("haul_remaining")) if cap is not None else 1.0
            penalty += base * mult

    # max_pickup_deadhead_km
    for rule in persona.get("max_pickup_deadhead_km", []):
        cap = _scalar(rule.get("cap"))
        if cap is not None and budget.get("pickup_cap_reached"):
            continue
        if dist_km > (_scalar(rule.get("max_km")) or float("inf")):
            base = float(_scalar(rule.get("penalty", 0)))
            mult = _scarcity_multiplier(budget.get("pickup_remaining")) if cap is not None else 1.0
            penalty += base * mult

    # daily_rest: 休息不足的严重性随剩余预算减少而放大
    # （虽然不能从 cargo 直接判断，但通过 _is_cargo_safe 的 cap-aware 逻辑已覆盖）

    return penalty


def _spans_forbidden_hours(
    start_min: int, end_min: int, rules: list[dict[str, Any]]
) -> bool:
    """检查时间段 [start_min, end_min) 是否与 forbidden_hours 有交集。"""
    start_dt = _sim_min_to_wall(start_min)
    end_dt = _sim_min_to_wall(end_min)
    for rule in rules:
        sh = rule.get("start_hour")
        if sh is None:
            continue
        sm = rule.get("start_min", 0)
        eh = rule.get("end_hour")
        if eh is None:
            continue
        em = rule.get("end_min", 0)
        if sh <= eh:
            day_start = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            while day_start <= end_dt:
                f_start = day_start + timedelta(hours=sh, minutes=sm)
                f_end = day_start + timedelta(hours=eh, minutes=em)
                if start_dt < f_end and end_dt > f_start:
                    return True
                day_start += timedelta(days=1)
        else:
            day_start = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            while day_start <= end_dt:
                f_start = day_start + timedelta(hours=sh, minutes=sm)
                f_end = day_start + timedelta(days=1, hours=eh, minutes=em)
                if start_dt < f_end and end_dt > f_start:
                    return True
                day_start += timedelta(days=1)
    return False


def _compute_rest_days(persona: dict[str, Any]) -> list[int]:
    """从画像的 monthly_rest_days 动态计算固定休息日，均匀分布。

    仅当约束明确要求"完全歇息"（不接单+不跑车）时才强制全天休息。
    "不接单"类约束（D002, D007）跳过——自然周转即可满足。
    """
    rules = persona.get("monthly_rest_days", [])
    if not rules:
        return []
    # 仅对"完全歇息"类约束强制休息（type="full_rest" 表示不接单+不跑车）
    full_rest_rules = [
        r for r in rules
        if r.get("type") == "full_rest"
        or ("也" in r.get("raw", "") or "完全歇" in r.get("raw", "")
            or "停驶" in r.get("raw", "") or "检修" in r.get("raw", ""))
    ]
    if not full_rest_rules:
        return []
    min_days = max((r.get("min_days") or 0) for r in full_rest_rules)
    if min_days <= 0:
        return []
    gap = 31 // (min_days + 1)
    return [gap * (i + 1) for i in range(min_days)]


def _compute_free_from_orders_days(persona: dict[str, Any]) -> list[int]:
    """从画像的 monthly_rest_days 计算"不接单"日（允许空跑，禁止接单）。

    区别于 _compute_rest_days 的"完全歇息"（不接单+不跑车）。
    用于 D007 "放空一整天不接单" 类约束。
    """
    rules = persona.get("monthly_rest_days", [])
    if not rules:
        return []
    free_from_orders = [
        r for r in rules
        if r.get("type") == "no_orders"
        or (r.get("type") not in ("full_rest",)
            and "也" not in r.get("raw", "")
            and "完全歇" not in r.get("raw", ""))
    ]
    if not free_from_orders:
        return []
    min_days = max((r.get("min_days") or 0) for r in free_from_orders)
    gap = 31 // (min_days + 1)
    return [gap * (i + 1) for i in range(min_days)]
