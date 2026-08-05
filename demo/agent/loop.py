"""策略场引擎（StrategyFieldEngine · 新架构 C5）。

LLM 不直接决策动作 —— 它（Virtual Manager）维护"策略场"（虚拟单 + 货源奖惩），
plan_route 在这张场里做**确定性最优选择**，执行 r0 第一跳。每步：

  ① 获取新货源（preplan 之前；cleanup→query_cargo，触发同旧）
  ② memory/ledger 更新（从 history 派生 pref_status / ledger_facts）
  ③ Virtual Manager LLM 检查并维护虚拟单（输出 patch → registry.apply_patch）
  ④ plan_route 生成候选（注 active waypoint 为合成边；**所有路必含全部 active waypoint**，删不含的）
  ⑤ harness 复核 top route 第一跳（虚拟单/硬约束/pricing；违反→回 ③ 给原因，有界）
  ⑥ 执行 r0 第一跳 + ledger 落账

plan_route 的排序/计价/区域价值**照搬** `plan_route.py`（不改、不简化）。偏好→定价经 Virtual Manager
注 cargo_modifier 实现（"货源种类奖惩直接作用于受影响货源"），取代旧的 typed-pref 罚款管道。
公共门面 ``ModelDecisionService`` 不变（server 不能改）。
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from simkit.ports import SimulationApiPort
from simkit.simulation_actions import distance_to_minutes, haversine_km

from . import harness, preference_parser, time_tools, virtual_manager
from .cargo_graph import CargoGraph, sim_min_to_wall
from .log_color import _clock, _line_end, _tag, _tokens, _total
from .long_memory import LongMemory
from .path_planner import region_bonus_for_path
from .plan_route import (
    append_unique_paths,
    build_rest_single_paths,
    drop_negative_yield_real_cargo_paths,
    plan_route,
)
from .virtual_registry import (
    CANCELED,
    CARGO_MODIFIER,
    CONSUMED,
    DEADHEAD,
    REST,
    WAYPOINT_KINDS,
    VirtualRegistry,
    apply_modifiers_to_yield,
)

_log = logging.getLogger("llm_agent.loop")

DEFAULT_COST_PER_KM = 1.5
REFRESH_CARGO_DEFAULT_K = 150
REST_WINDOW_REFRESH_K = 600  # 长休窗内预热查询条数：进窗喂长期记忆 + 出窗前 1h 规划，均用更宽视野（窗内查货不算违规）
REST_LONG_WINDOW_MIN = 360   # 休息窗长 ≥ 6h 才启用「拆两段 + 窗内预热」（用户口径：时长≥6h、含整 6h 的休息）
REST_PREWAKE_LEAD_MIN = 60   # 出窗前提前 1h 唤醒查货做出窗规划；醒来仍在窗内 → wait 到窗口结束才行动
PLAN_BEAM_WIDTH = 50         # plan_route 自适应 beam：可达货>触发阈值时每节点只展开 top-N(benchmark:候选池逐字节同·4.5x提速)
PLAN_BEAM_TRIGGER = 200      # 可达货 > 此值才启用 beam(白天货密会指数爆炸)；≤此值全枚举(深夜/货稀·admissible 零漏)
QUERY_SCAN_MARGIN_MIN = 15
REGION_VALUE_WEIGHT = 1.0
LM_WARMUP_THRESHOLD = 8
HARNESS_MAX_REPLANS = 2
LONG_HAUL_MIN = 8 * 60  # >8h 纯运货 = 长途（与偏好口径一致）
_REST_BREAK_MIN = 30  # ≥此时长的 wait 视为一次"休息"、重置连续驾驶计时（区分真休息 vs 1min reposition wait）
# manager 日内复评段数（skip-gate baseline 唤醒节奏）：2=上午/下午各一次。旧版用 4h 时间桶，但单步 sim 时钟
# 常跳 >4h（中位 251min、>240 占 53%）→ 几乎每步跨桶→manager 每步被唤醒(实测 skip 仅 19/374≈5%)→
# 每次 ~14k token × ~355 次烧穿 5M 预算、仿真 day75 腰斩。改成"日界 + 固定 day_part"稀疏唤醒：manager 在
# 日界注当晚 rest + 当日策略，执行（rest 等窗/配额定价/plan_route 选路）确定性、无须每 4h 复唤。2 段≈2 唤醒/天
# ×92 天×14k≈2.6M，留足余量。需更高响应可调 3（=每 8h）。
MANAGER_DAYPARTS = 2
# 地名→坐标 参考表（geocoder 缺位的兜底）：真实货端点只有坐标无地名文本，region 子串谓词命不中；当偏好原文
# 只给城市名(如"不要去深圳")而无坐标时，manager 用此表查中心坐标→编 geo 谓词。**这是地理参考数据、非偏好规则**，
# 仅覆盖仿真域(珠三角/广东)主要城市；不在表内的地名 manager 只能退回 region 子串(对真实货不可靠)。可按需扩。
GEO_HINTS: dict[str, tuple[float, float]] = {
    "广州": (23.13, 113.26), "深圳": (22.54, 114.06), "佛山": (23.02, 113.12), "东莞": (23.02, 113.75),
    "惠州": (23.11, 114.42), "珠海": (22.27, 113.58), "中山": (22.52, 113.39), "江门": (22.58, 113.08),
    "肇庆": (23.05, 112.47), "清远": (23.68, 113.05), "韶关": (24.81, 113.60), "汕头": (23.35, 116.68),
    "湛江": (21.27, 110.36), "茂名": (21.66, 110.93), "阳江": (21.86, 111.98), "云浮": (22.92, 112.04),
    "揭阳": (23.55, 116.37), "潮州": (23.66, 116.62), "汕尾": (22.79, 115.38), "河源": (23.74, 114.70),
    "梅州": (24.29, 116.12),
}
_WP_PREFIX = "__wp_"          # 合成 waypoint 图边 cargo_id 前缀
_DEFAULT_REST_VALUE = 1800.0  # rest 虚拟单未填 value 时的默认（≈夜休避免罚款额）
_DEADHEAD_DWELL_MIN = 1       # deadhead 到点后的最小 dwell（耗时主要是赴点空驶，由 plan_route 算）
_DISPLAY_REAL_CAP = 12
_RECENT_ORDERS_CAP = 20  # ledger_facts.recent_orders 限长（最近在前），喂 manager 判时序/到访/序列
# 窗前等待**不设上限**（用户令）：选中单跨休息窗时一步直接等到窗起/窗尾，绝不分段砍
# （实测旧 cap=30 把一晚等待炸成 6-7 步、82 步/月，且每次重查都空手）。


def _skip_gate_on() -> bool:
    """token skip-gate 总开关（默认开；STRATEGY_SKIP_GATE=0/false/no/off 关，便于 A/B）。"""
    return os.environ.get("STRATEGY_SKIP_GATE", "1").strip().lower() not in ("0", "false", "no", "off")


def _harness_on() -> bool:
    """harness-LLM 复核总开关。**当前默认关**（用户令：暂时关闭——实测它误报打回烧 362 次调用/月且
    2 次打回后照样执行；确定性墙(窗前罚+pre-rest guard+waypoint过滤)已承重）。设 STRATEGY_HARNESS=1 重新启用。"""
    return os.environ.get("STRATEGY_HARNESS", "0").strip().lower() in ("1", "true", "yes", "on")


def _long_rest_split_on() -> bool:
    """长休(窗长≥6h)『拆两段 + 出窗前1h醒来查600规划』总开关（默认开）。关 → 所有 rest 一觉睡到窗口结束、
    每步默认 k=150（精确还原改动前行为），仅供 A/B 基线对比用：AGENT_LONG_REST_SPLIT=0/false/no/off。"""
    return os.environ.get("AGENT_LONG_REST_SPLIT", "1").strip().lower() not in ("0", "false", "no", "off")


def _plan_beam_on() -> bool:
    """plan_route 自适应 beam 总开关（默认开）。关 → 全枚举(admissible 基线)，供 A/B 对比：
    AGENT_PLAN_BEAM=0/false/no/off。只在可达货 > PLAN_BEAM_TRIGGER 的步生效(白天货密)，深夜/货稀不受影响。"""
    return os.environ.get("AGENT_PLAN_BEAM", "1").strip().lower() not in ("0", "false", "no", "off")

_CN_WALL_RE = re.compile(
    r"^\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*(?:(\d{1,2})\s*[:时点]\s*(\d{1,2})?\s*分?)?\s*$"
)
# 裸钟点『HH:MM』(无日期前缀) → 锚到 now 当天，而非 next-occurrence（见 _resolve_walltime）。
_BARE_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def _endpoint_flat_text(node: Any) -> str:
    if isinstance(node, dict):
        for k in ("address", "name", "city", "district"):
            v = node.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _is_wp_cargo_id(cid: Any) -> bool:
    return str(cid or "").startswith(_WP_PREFIX)


def _as_float(x: Any) -> float | None:
    """容错转 float；bool/None/不可转 → None。"""
    if isinstance(x, bool) or x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _match_cargo_for_hop(hop: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    """逐跳谓词匹配用的货视图：坐标/品类来自嵌套 cargo（+ cargo_info_by_id 兜底 ``info``），
    数值来自 hop（plan_route 算好的 net_yield/haul_km/price）。供 cargo_modifier 的
    category/geo/region/attribute 谓词命中（真实货端点只有坐标 → geo 走坐标）。"""
    cargo = hop.get("cargo") if isinstance(hop.get("cargo"), dict) else {}
    s = cargo.get("start") if isinstance(cargo.get("start"), dict) else {}
    e = cargo.get("end") if isinstance(cargo.get("end"), dict) else {}
    return {
        # cargo_id 必带：否则 cargo_id 谓词在诊断/命中统计路径恒不命中(pool_hits=0)→ 误判 dud(已修回归)。
        "cargo_id": hop.get("cargo_id") or cargo.get("cargo_id"),
        "cargo_name": cargo.get("cargo_name") or cargo.get("name") or info.get("cargo_name"),
        "start": {"lat": s.get("lat", info.get("start_lat")), "lng": s.get("lng", info.get("start_lng"))},
        "end": {"lat": e.get("lat", info.get("end_lat")), "lng": e.get("lng", info.get("end_lng"))},
        "start_address": info.get("start_address", ""),
        "end_address": info.get("end_address", ""),
        "gross_yield": hop.get("price", cargo.get("price")),
        "net_yield": hop.get("net_yield"),
        "haul_km": hop.get("haul_km"),
        "cost_time_minutes": cargo.get("cost_time_minutes") or info.get("cost_time_minutes"),
        "finish_min": hop.get("finish_min"),  # finish_min 谓词诊断用
    }


def day_meters_for_date(history: list[dict[str, Any]], info_by_id: dict[str, Any], date_str: str,
                        spd: float = 60.0) -> dict[str, Any]:
    """查询**任意指定日期**（'YYYY-MM-DD'）的日级米表（完单/空驶km/载货km/载货分钟/纯空驶km/驾驶分钟/长途数）。
    与 _build_ledger_facts 的 day_meters 同口径（按完成时刻归日，drive_minutes 含载货+赴装空驶+纯 reposition 空驶）；
    context 只带 today+近7天，更早的日期用这个函数按需查（_diag/工具层）。日期非法 → 返回全 0。"""
    out = {"date": date_str, "orders": 0, "deadhead_km": 0.0, "haul_km": 0.0, "haul_minutes": 0,
           "reposition_km": 0.0, "drive_minutes": 0, "long_haul": 0, "gross": 0.0}
    try:
        day_idx = int(time_tools.sim_min_of(date_str, "00:00")) // time_tools.MINUTES_PER_DAY
    except Exception:  # noqa: BLE001 — 日期解析失败按无数据
        return out
    spd = max(1.0, float(spd or 60.0))
    prev_pos = None  # 上一记录终点（reposition 缺 position_before 时兜底算里程）
    for rec in history or []:
        if not isinstance(rec, dict):
            continue
        act = rec.get("action") if isinstance(rec.get("action"), dict) else {}
        action_name = str(act.get("action") or "")
        res = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        pa = rec.get("position_after") if isinstance(rec.get("position_after"), dict) else None
        if action_name in ("reposition", "move"):  # 纯空驶：按本步完成时刻归日（"每日驾驶≤Yh"含空驶）
            t = res.get("simulation_progress_minutes")
            in_day = (t is not None) and (int(t) // time_tools.MINUTES_PER_DAY == day_idx)
            pb = rec.get("position_before") if isinstance(rec.get("position_before"), dict) else None
            try:
                if pb and pb.get("lat") is not None and pa and pa.get("lat") is not None:
                    rep_km = haversine_km(float(pb["lat"]), float(pb["lng"]), float(pa["lat"]), float(pa["lng"]))
                elif prev_pos is not None and pa and pa.get("lat") is not None:
                    rep_km = haversine_km(prev_pos[0], prev_pos[1], float(pa["lat"]), float(pa["lng"]))
                else:
                    rep_km = 0.0
            except (TypeError, ValueError, KeyError):
                rep_km = 0.0
            if in_day:
                out["reposition_km"] = round(out["reposition_km"] + rep_km, 1)
            if pa and pa.get("lat") is not None:
                prev_pos = (float(pa["lat"]), float(pa["lng"]))
            continue
        if action_name != "take_order":
            continue
        if not res.get("accepted"):
            continue
        t = res.get("simulation_progress_minutes")
        if pa and pa.get("lat") is not None:
            prev_pos = (float(pa["lat"]), float(pa["lng"]))  # 卸货点=下一段 reposition 起点
        if t is None or int(t) // time_tools.MINUTES_PER_DAY != day_idx:
            continue
        cid = str(res.get("cargo_id") or (act.get("params") or {}).get("cargo_id") or "").strip()
        if not cid or _is_wp_cargo_id(cid):
            continue
        meta = (info_by_id or {}).get(cid, {})
        out["orders"] += 1
        dead_km = _as_float(res.get("pickup_deadhead_km"))
        haul_km = _as_float(res.get("haul_distance_km"))
        haul_min = meta.get("cost_time_minutes")
        gross = _as_float(meta.get("price"))
        if gross is not None and res.get("income_eligible", True):
            out["gross"] = round(out["gross"] + gross, 1)
        if dead_km is not None:
            out["deadhead_km"] = round(out["deadhead_km"] + dead_km, 1)
        if haul_km is not None:
            out["haul_km"] = round(out["haul_km"] + haul_km, 1)
        if haul_min is not None:
            out["haul_minutes"] += int(haul_min)
            if float(haul_min) > LONG_HAUL_MIN:
                out["long_haul"] += 1
    out["drive_minutes"] = int(out["haul_minutes"] + round((out["deadhead_km"] + out["reposition_km"]) / spd * 60.0))
    return out


def _market_cargo_view(cargo: dict[str, Any], pos) -> dict[str, Any]:
    """市场存量扫描（_collect_modifier_effects）用的货视图：原始图货顶层没有 haul_km/deadhead_km，
    补算端点球面距 + 当前位置→装点空驶，attribute 谓词才能在市场货上命中。坐标缺失时保持 None（不命中）。"""
    s = cargo.get("start") if isinstance(cargo.get("start"), dict) else {}
    e = cargo.get("end") if isinstance(cargo.get("end"), dict) else {}
    view = dict(cargo)
    view["_deadhead_km"] = None
    try:
        if s.get("lat") is not None and e.get("lat") is not None and "haul_km" not in view:
            view["haul_km"] = haversine_km(float(s["lat"]), float(s["lng"]), float(e["lat"]), float(e["lng"]))
        if s.get("lat") is not None and pos is not None:
            view["_deadhead_km"] = haversine_km(float(pos[0]), float(pos[1]), float(s["lat"]), float(s["lng"]))
    except (TypeError, ValueError, KeyError):
        pass
    return view


class StrategyFieldEngine:
    """策略场引擎：Virtual Manager 维护场 → plan_route 确定性选 r0 → harness 复核 → 执行首跳。"""

    def __init__(
        self,
        api: SimulationApiPort,
        *,
        reposition_speed_km_per_hour: float = 60.0,
        simulation_duration_days: int = time_tools.DURATION_DAYS,
        persona_builder: "Callable[[str, list[dict[str, Any]]], list[Any]] | None" = None,
    ) -> None:
        self._api = api
        self._speed = float(reposition_speed_km_per_hour)
        self._duration_days = int(simulation_duration_days)
        self._persona_builder = persona_builder
        self._state: dict[str, dict[str, Any]] = {}
        self._log = _log

    # ================================================================= 公共入口
    def decide(self, driver_id: str) -> dict[str, Any]:
        """顶层兜底：六步流任何环节抛异常（API/纯计算遇脏数据等）→ 不穿透给 server，返回安全 wait。
        所有 LLM 触点已各自降级，这层只兜未保护的 API/计算异常，保证 decide 永远返回合法动作。"""
        try:
            return self._decide(driver_id)
        except Exception:  # noqa: BLE001
            self._log.exception("decide_failed driver=%s — 兜底 wait", driver_id)
            try:  # 兜底路径也要推进步号，否则 step 冻结、与后续 now_min 脱节
                self._driver_state(driver_id)["step"] += 1
            except Exception:  # noqa: BLE001
                pass
            return {"action": "wait", "params": {"duration_minutes": 10}, "reason_brief": "decide exception fallback"}

    def _decide(self, driver_id: str) -> dict[str, Any]:
        if hasattr(self._api, "reset_last_model_usage"):
            self._api.reset_last_model_usage()
        status = self._normalized_status(self._api.get_driver_status(driver_id))
        pos = (status["current_lat"], status["current_lng"])
        now_min = int(status["simulation_progress_minutes"])
        cost_per_km = float(status.get("cost_per_km") or DEFAULT_COST_PER_KM)

        st = self._driver_state(driver_id)
        st["pos"] = pos
        st["driver_id"] = driver_id
        st["_last_rejected"] = None  # 每步起清空，避免上一步的 rejected 串味本步 attempt 0 的 context
        self._maybe_init(st, list(status.get("preferences", []) or []))

        wall = status.get("simulation_wall_time") or f"{time_tools.now(now_min)['date']} {time_tools.now(now_min)['hhmm']}"
        self._log.info("%s driver=%s step=%s clock=%s loc=(%.5f,%.5f)%s",
                       _tag("STEP_BEGIN"), driver_id, st["step"], _clock("STEP_BEGIN", wall), pos[0], pos[1], _line_end())

        # ① 获取新货源（preplan 之前）
        self._refresh_cargo(st, driver_id, now_min, pos, cost_per_km)
        history = self._history(driver_id)
        self._diagnose_prev_take_failure(st, history)  # 上一步接单若 server 判已失效 → 打印失配诊断

        # ② memory/ledger 更新
        ledger_facts = self._build_ledger_facts(st, history, now_min)
        pref_status = self._build_pref_status(st, ledger_facts, now_min)
        st["_pref_status_cache"] = pref_status  # harness ctx 取它的 canonical_text（此前从不写→恒走兜底）

        reg: VirtualRegistry = st["registry"]
        reg.refresh(now_min=now_min)

        # ③④⑤ Virtual Manager → plan_route → harness（有界回环 + 确定性 skip-gate 省 token）
        sig = self._field_signature(st, now_min, pref_status, ledger_facts)
        manager_skipped = False
        last_patch: dict[str, Any] = {}
        routes: list[dict[str, Any]] = []
        feedback: list[dict[str, Any]] = []
        verdict: dict[str, Any] = {"ok": True}
        for attempt in range(HARNESS_MAX_REPLANS + 1):
            # manager skip-gate：场签名未变 + 首轮 + 无 harness 回打 + 无上轮 rejected → 跳过整次 LLM 调用
            if self._should_call_manager(st, sig, feedback, attempt, now_min):
                mgr_ctx = self._build_manager_context(st, now_min, pos, pref_status, ledger_facts, feedback)
                patch = virtual_manager.maintain(self._api, mgr_ctx, self._log)
                self._apply_manager_patch(st, patch, now_min)
                st["_last_called_sig"] = sig
                last_patch = patch
                manager_changed = bool((patch.get("add") or patch.get("cancel") or patch.get("update")))
            else:
                manager_skipped = True
                manager_changed = False
            routes = self._plan_strategy_field(st, driver_id, now_min, pos, cost_per_km, status)
            st["route_cache"] = {f"r{i}": r for i, r in enumerate(routes)}
            top = routes[0] if routes else None
            # harness：总开关默认关(用户令,见 _harness_on)；开了再走 skip-gate(虚拟单/wait/无路免审)
            if _harness_on() and self._should_review_harness(st, top, now_min, manager_changed):
                verdict = harness.review(self._api, self._build_harness_context(st, now_min, pos, top), self._log)
            else:
                verdict = {"ok": True, "_skipped": True}
            if verdict.get("ok") or attempt >= HARNESS_MAX_REPLANS:
                break
            feedback = list(verdict.get("violations") or [])
        # 复核回合耗尽仍未 ok 却照样执行 → 显式标记，便于事后归因（不是静默放行）
        if not verdict.get("ok"):
            self._log.warning("%s driver=%s step=%s EXECUTE_DESPITE_VIOLATION violations=%s%s",
                              _tag("HARNESS"), driver_id, st["step"],
                              str(verdict.get("violations"))[:200], _line_end())

        # ⑥ 执行 r0 第一跳
        top = routes[0] if routes else None
        action = self._route_first_hop_action(st, top, now_min, pos)
        # 窗前穿窗硬兜底：选中真实单若跨越(active/pending)休息窗 → 改 bounded wait 等窗，绝不起飞穿窗
        action = self._guard_pre_rest_wait(st, action, top, now_min)
        # ledger 落账：真实货 take_order → mark_taken（合成 waypoint 绝不 mark_taken）
        if action.get("action") == "take_order":
            cid = str((action.get("params") or {}).get("cargo_id", "")).strip()
            if cid and not _is_wp_cargo_id(cid):
                # 诊断留底（observability）：mark_taken 是乐观删除，删前快照该货有效性。下一步若 server 判
                # 『已失效』，_diagnose_prev_take_failure 据此打印失配真因（窗缺失/陈货/窗刚过/remove已过）。
                h0 = (top.get("hops") or [{}])[0] if top else {}
                st["_last_take_dbg"] = {"cid": cid, "now_min": int(now_min),
                                        "arrival_min": h0.get("arrival_min"), "deadhead_km": h0.get("deadhead_km"),
                                        "finish_min": h0.get("finish_min"),
                                        "snap": st["graph"].cargo_validity_snapshot(cid)}
                st["graph"].mark_taken(cid)

        self._log.info("%s driver=%s step=%s action=%s %s%s", _tag("DECISION"), driver_id, st["step"],
                       action.get("action"),
                       _tokens("DECISION", top=("r0" if top else "none"),
                               reason=str(action.get("reason_brief", ""))[:50]), _line_end())
        self._log_tokens(driver_id, st)
        # per-step 审计落盘（env STRATEGY_AUDIT_FILE）：决策溯源，供真实跑后逐步对账/三方归因
        first0 = (top.get("hops") or [{}])[0] if top else {}
        self._audit_jsonl(st, driver_id, now_min, wall, pos, {
            "sim_day": int(now_min) // 1440 + 1,
            "manager_skipped": manager_skipped,
            "manager_patch": ({"add": len(last_patch.get("add") or []), "cancel": len(last_patch.get("cancel") or []),
                               "update": len(last_patch.get("update") or []), "reason": str(last_patch.get("reason", ""))[:80]}
                              if last_patch else None),
            "rejected": st.get("_last_rejected") or None,
            "harness": ("skipped" if verdict.get("_skipped") else ("ok" if verdict.get("ok") else "violation")),
            "violations": verdict.get("violations") if not verdict.get("ok") else None,
            "n_candidates": len(routes),
            "top_first": {"cid": str(first0.get("cargo_id") or ""),
                          "adjusted": (top or {}).get("preference_adjusted_yield")} if top else None,
            "action": action.get("action"),
            "action_params": action.get("params"),
            "reason": action.get("reason_brief"),
            # 虚拟单带 state+kind（非仅 id）：事后能分清 active rest vs pending、是否 consumed
            "virtuals_live": [{"id": v.get("id"), "kind": v.get("kind"), "state": v.get("state")}
                              for v in st["registry"].items if v.get("state") in ("active", "pending")],
        })
        st["step"] += 1
        return action

    @staticmethod
    def _rest_doable_in_place(v: dict[str, Any], pos, reg: "VirtualRegistry", now_min: int) -> bool:
        """该 rest 是否能『在当前位置原地完成』——pre-rest 硬等窗只对这类成立。
        排除：① params.lat/lng 距当前 pos >1km 的远端守候 rest(正解=去 venue);
        ② combo 下游(seq2)其更小 seq 兄弟尚未 consumed(前置赴点没做完,该 rest 不在当前点)。"""
        params = v.get("params") if isinstance(v.get("params"), dict) else {}
        lat, lng = params.get("lat"), params.get("lng")
        if lat is not None and lng is not None and pos is not None:
            try:
                if haversine_km(pos[0], pos[1], float(lat), float(lng)) >= 1.0:
                    return False  # 远端 rest：原地等窗错误，应先 deadhead 过去
            except (TypeError, ValueError):
                pass
        cid, seq = v.get("combo_id"), v.get("combo_seq")
        if cid is not None and seq is not None:
            for s in reg.selectable(now_min=now_min):
                if (s.get("combo_id") == cid and s.get("combo_seq") is not None
                        and int(s.get("combo_seq")) < int(seq)):
                    return False  # 前置 combo 步骤还在(未 consumed)→ 别原地等这步
        return True

    def _guard_pre_rest_wait(self, st, action, top, now_min: int) -> dict[str, Any]:
        """硬兜底：选中的真实 take_order 若其首跳完成时刻落进某 active/pending 休息窗(即跨窗作业)，
        改为 wait **一步等到窗起**(窗已开则等到窗尾)，**不设上限**(用户令)——绝不 30min 分段砍步。
        窗口罚已在定价层 de-rank crosser；此层是『绝不穿窗』的确定性保证，覆盖 all-cross 被 least-bad 兜回的残留。"""
        if action.get("action") != "take_order" or not top:
            return action
        reg: VirtualRegistry = st["registry"]
        rests = [v for v in reg.selectable(now_min=now_min) if v.get("kind") == REST]
        if not rests:
            return action
        # 只对『能在当前位置原地完成』的 rest 兜底等窗(夜休/宵禁)。**远端守候 rest**(combo seq2:先 deadhead
        # 到事件点再守候,params.lat/lng≠当前 pos)的正解是去 venue 而非原地空等;**前置未消耗的 combo 下游**同理。
        # 误把它们当原地窗 → 司机原地空等不赴约/livelock 整个事件窗(历史『家长会6.5h空等』同形)。
        rsp = top.get("route_start_pos") or []
        pos = (float(rsp[0]), float(rsp[1])) if len(rsp) >= 2 else None
        rests = [v for v in rests if self._rest_doable_in_place(v, pos, reg, now_min)]
        if not rests:
            return action
        hops = top.get("hops") or []
        first = hops[0] if hops and isinstance(hops[0], dict) else None
        fm = first.get("finish_min") if isinstance(first, dict) else None
        if fm is None:
            return action
        now, fm = int(now_min), int(fm)
        for v in rests:
            ws = int(v.get("start_min") or now)
            we = v.get("end_min") if v.get("end_min") is not None else v.get("expire_min")
            if we is None:
                continue
            if now < int(we) and fm > ws:  # 该单执行区间跨越休息窗
                # 窗未开(ws>now) → 一步等到窗起(下一步 rest 激活接管)；窗已开 → 一步等到窗尾。
                target = ws if ws > now else int(we)
                wait_min = max(1, target - now)
                self._log.warning("%s driver=%s 选中单跨休息窗(finish=%s,窗[%s,%s])→改 wait %s 等窗%s",
                                  _tag("DECISION"), st.get("driver_id"), fm, ws, int(we), wait_min, _line_end())
                return self._wait_action(wait_min, "pre-rest window guard")
        return action

    # ================================================================= skip-gate（省 token）
    @staticmethod
    def _field_signature(st, now_min: int, pref_status, ledger_facts) -> tuple:
        """策略场签名：囊括所有"会让 manager 想动手"的输入。两步签名相同 → 场未变 → manager 跳过。
        **稀疏唤醒版**(旧版每步跨 4h 桶→几乎每步唤醒→烧穿 5M token、仿真 day75 腰斩)。含：
          - 仿真日 + day_part(默认 2 段/天)：baseline 复评节奏。**去掉旧 4h 时间桶**——单步常跳 >4h、每步跨桶。
            manager 在日界注当晚 rest + 当日策略，执行(等窗/定价/选路)确定性、无须每 4h 复唤。
          - **非 REST** active+pending 虚拟单 id+state：rest 生命周期(pending→active→consumed)是确定性执行、
            注入后无须复唤；deadhead/combo/modifier 的跃迁(combo 推进/expire/被拒重注)才需 follow-up。
          - 进度维度**按偏好相关性门控**(_pref_dims)：只有真有"连续驾驶/每日驾驶/序列"偏好才纳入对应计数，
            否则无关计数(每接一单就变)会把 manager 拖成每步唤醒。月度配额/长途cap/月度空驶等聚合类**不纳入**
            (无日内紧迫性：超额不罚、压低持久；每个 day_part 复评配额进度足矣)。
          - dud digest + 偏好签名。"""
        reg: VirtualRegistry = st["registry"]
        day = int(now_min) // 1440
        day_part = (int(now_min) % 1440) * MANAGER_DAYPARTS // 1440
        virtuals = tuple(sorted((str(v.get("id")), str(v.get("state"))) for v in reg.items
                                if v.get("state") in ("active", "pending") and v.get("kind") != REST))
        dims = st.get("_pref_dims") or {}
        prog: list[Any] = []
        # 工时桶用 1h（非 0.5h）：限值多 ≥4h，1h 粒度逼近上限时仍有 ≥3-4 次唤醒注 rest 的机会，而重驾驶日
        # 桶跨越次数减半（连续/每日驾驶司机最坏情况下 token 不至失控）。注好 rest 后多余跨桶=no-op 复唤，故宜粗。
        if dims.get("continuous_driving"):  # 连续驾驶≤Xh必歇：逼近上限须及时，1h 桶
            prog.append(("cd", int((ledger_facts.get("continuous_driving") or {}).get("minutes", 0) or 0) // 60))
        if dims.get("daily_drive"):         # 每日驾驶≤Yh：当日驾驶分钟 1h 桶
            prog.append(("dd", int((((ledger_facts.get("day_meters") or {}).get("today") or {}).get("drive_minutes", 0)) or 0) // 60))
        if dims.get("sequence"):            # 序列/相邻：上一单(序列偏好需实时判相邻)
            prog.append(("lr", str((ledger_facts.get("last_real_order") or {}).get("cargo_id"))))
        if dims.get("daily_meter"):         # 当日收入/单数门：跨阈须唤醒(注收工 rest/翻转 modifier),收入 500 桶
            _td = ((ledger_facts.get("day_meters") or {}).get("today") or {})
            prog.append(("dm", int(_td.get("orders", 0) or 0), int(float(_td.get("gross", 0.0) or 0.0) // 500)))
        if dims.get("clock_rest"):          # 钟点休息安全网：在册 rest 被异常移除→present 翻 False→唤醒补注(否则整夜穿窗)
            prog.append(("rp", any(v.get("state") in ("active", "pending") and v.get("kind") == REST for v in reg.items)))
        # **dud digest**（#8）：上一步哑火 modifier 的 (id, value_delta桶)（pool_hits=0 且 market>0）。
        # digest **变化**才改签名→唤醒 manager 修一次。带上 value_delta：manager 加码后该单仍 dud（力度还不够）→
        # value 变→digest 变→下一步**再次**唤醒可继续加码（避免"一次没修够就永久沉默");manager 停手(value 稳定)→
        # digest 稳定→恢复 skip。既能逐步逼出可命中力度、又在不可达/已尽力时收敛、不每步烧 token。
        dud = tuple(sorted((str(p.get("id")), int(round((p.get("value_delta") or 0.0))))
                           for p in ((st.get("_modifier_fx") or {}).get("per_modifier") or [])
                           if int(p.get("pool_hits", 0) or 0) == 0 and int(p.get("market_count", 0) or 0) > 0))
        return (day, day_part, virtuals, tuple(prog), dud, st.get("pref_sig"))

    def _should_call_manager(self, st, sig: tuple, feedback, attempt: int, now_min: int) -> bool:
        # 休息窗内(main 进窗 + tail 填窗 + 补偿)是确定性自动执行(r0 必选 rest 单跳、plan_route 也跳过)，
        # **整个休息窗内绝不唤醒 manager**——该 day_part 的场维护自然落到出窗步(窗外 now≥end)。优先级最高，
        # 连 harness 回打/上轮 rejected 都不在窗内重调(窗内是确定性执行、无需维护场/复核)。
        # 用户口径:长休拆段是写好的自动执行程序、中间不参杂任何 LLM 调用。
        if self._rest_window_active(now_min, st["registry"].active_waypoints(now_min=now_min)):
            self._log.info("%s driver=%s step=%s manager skip(休息窗内纯执行)%s", _tag("VMGR"),
                           st.get("driver_id"), st["step"], _line_end())
            return False
        if not _skip_gate_on():
            return True
        if attempt > 0 or feedback or st.get("_last_rejected"):
            return True  # harness 回打 / 上轮被拒 → 必须让 manager 重新处理
        if st.get("_last_called_sig") != sig:
            # 场签名变了（日界/day_part/非rest虚拟单跃迁/门控的工时·序列维度/**dud变化**/偏好变）→ 调。
            # 稀疏唤醒：月度配额/长途cap/纯时钟推进**不再**改签名（旧版 4h 桶每步唤醒烧穿 token）；manager 在
            # 日界 + day_part 复评策略足矣，dud 持续不变则签名稳定→仍 skip（避免稳态死循环烧 token）。
            return True
        self._log.info("%s driver=%s step=%s manager skip(场未变)%s", _tag("VMGR"),
                       st.get("driver_id"), st["step"], _line_end())
        return False

    def _should_review_harness(self, st, top, now_min: int, manager_changed: bool) -> bool:
        """harness skip-gate（保守版）：只对**确定安全**的首跳免审——虚拟单首跳(manager 自己的意图)、
        wait/无路(无动作)。**所有真实 take_order 一律复核**：默认配置下偏好全是 freeform、靠 manager 编码进
        modifier/waypoint，若谓词失配(如 geo 坐标漂移)则"干净真实单"实则违规——故不按 clean/命中 跳过真实单，
        避免漏判。省下的是 rest/reposition/wait 这些步的复核（+manager 的 skip 才是大头）。"""
        if not _skip_gate_on():
            return True
        if not top or not top.get("hops"):
            return False  # 无路/wait → 无需复核
        first = top["hops"][0] if isinstance(top["hops"][0], dict) else {}
        cid = str(first.get("cargo_id") or "")
        if _is_wp_cargo_id(cid):
            return False  # 首跳是虚拟单（rest/deadhead，manager 自己的意图）→ 免审
        return True  # 真实单 → 一律复核

    def _audit_jsonl(self, st, driver_id: str, now_min: int, wall: str, pos, sig_inputs: dict[str, Any]) -> None:
        """每步审计落盘（env STRATEGY_AUDIT_FILE 设了才写）：一行 JSONL 记决策溯源，供事后逐步对账 + drift 检测。"""
        path = os.environ.get("STRATEGY_AUDIT_FILE", "").strip()
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"driver": driver_id, "step": st.get("step"), "now_min": int(now_min),
                                    "wall": wall, "pos": [pos[0], pos[1]], **sig_inputs}, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 — 审计落盘绝不打断决策；失败只告警一次（避免每步刷屏）
            if not st.get("_audit_warned"):
                self._log.warning("audit_jsonl_write_failed path=%s（后续不再重复告警）", path, exc_info=True)
                st["_audit_warned"] = True

    def parsed_prefs_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for did, st in self._state.items():
            out[did] = [p.to_dict() for p in st.get("parsed_prefs", []) if hasattr(p, "to_dict")]
        return out

    # ================================================================= 状态 / 初始化
    def _driver_state(self, driver_id: str) -> dict[str, Any]:
        if driver_id not in self._state:
            self._state[driver_id] = {
                "graph": CargoGraph(),
                # **每司机一份独立 LongMemory**（不变量）：区域价值/热区只由该司机自己的观测构成，
                # 绝不跨司机共享（无共享实例、无磁盘持久化池）。换司机=换一份全新记忆。
                "long_memory": LongMemory(cost_per_km=DEFAULT_COST_PER_KM),
                "registry": VirtualRegistry(),
                "cargo_info_by_id": {},
                "parsed_prefs": [],
                "pref_sig": None,
                "route_cache": {},
                "step": 0,
            }
        return self._state[driver_id]

    def _maybe_init(self, st: dict[str, Any], raw_prefs: list[dict[str, Any]]) -> None:
        sig = tuple(str(p.get("content", "")) for p in raw_prefs)
        if st["pref_sig"] == sig:
            return
        try:
            if self._persona_builder is not None:
                parsed = self._persona_builder(st.get("driver_id", ""), raw_prefs)
            else:
                parsed = preference_parser.parse_driver_preferences(self._api, raw_prefs, self._log)
            st["parsed_prefs"] = list(parsed or [])
            st["pref_sig"] = sig
            st["_pref_dims"] = self._compute_pref_dims(st["parsed_prefs"])  # 签名维度门控（见 _field_signature）
        except Exception:  # noqa: BLE001 — 解析失败不打断；下一步重试
            self._log.exception("parse_driver_preferences_failed")

    @staticmethod
    def _compute_pref_dims(parsed_prefs: list[Any]) -> dict[str, bool]:
        """从结构化偏好探测"哪些 ledger 进度维度真有偏好需要"——只有需要的维度才进 _field_signature，
        否则无关计数(每接一单就变)会把 manager 拖成每步唤醒。只读可靠的结构字段、不猜原文。
          - continuous_driving：driving_limits.max_continuous_drive_min（连续驾驶到点强制歇）。
          - daily_drive：driving_limits.max_daily_drive_min（每日驾驶≤Yh）。
          - sequence：sequence_constraints 或 max_idle_gap_between_orders_min（序列/相邻关系需看上一单）。
          - daily_meter：日级聚合/激活门（"当日收入超X收工"/"当日单数到N解禁冷链"）——触发是**当日米表跨阈**、
            发生在日内任意时刻，须纳入签名才能在跨阈时唤醒 manager 注"收工 rest"/翻转 modifier。
          - clock_rest：钟点锚休息（fixed_window 夜休/宵禁 或 continuous_daily 每日连续休）——这类**理应全天都有一张
            当晚/当日 rest 在册**(清晨注好、pending 到窗起)。签名带"是否存在 active/pending REST"布尔：正常恒 True、
            ≈0 额外唤醒(consumed→False 恰与日界唤醒重合)；若 rest 被异常移除(误 cancel/bug)→False→唤醒 manager 补注，
            否则没有 rest 在册时 _guard_pre_rest_wait 无栅栏可拦、整夜穿窗(=本次 17 夜违规同形)。
        月度配额/长途cap/月度空驶 等**月度聚合不纳入**——无日内紧迫性(超额不罚、压低持久)，每个 day_part 复评足矣。"""
        dims = {"continuous_driving": False, "daily_drive": False, "sequence": False,
                "daily_meter": False, "clock_rest": False}
        for p in parsed_prefs or []:
            dl = getattr(p, "driving_limits", None)
            if isinstance(dl, dict):
                if dl.get("max_continuous_drive_min"):
                    dims["continuous_driving"] = True
                if dl.get("max_daily_drive_min"):
                    dims["daily_drive"] = True
            if getattr(p, "sequence_constraints", None) or getattr(p, "max_idle_gap_between_orders_min", None) is not None:
                dims["sequence"] = True
            for agg in (getattr(p, "aggregate_constraints", None) or []):
                if isinstance(agg, dict) and str(agg.get("window") or "").lower() in ("natural_day", "daily", "day"):
                    dims["daily_meter"] = True
            guard = getattr(p, "activation_guard", None)
            if isinstance(guard, dict) and str(guard.get("metric") or "").lower().startswith("daily"):
                dims["daily_meter"] = True
            # 钟点锚休息：fixed_window(夜休/宵禁) / continuous_daily(每日连续休)。monthly_days 不算(月度账，日界处理)。
            if (str(getattr(p, "rest_type", "") or "") in ("fixed_window", "continuous_daily")
                    or getattr(p, "rest_window_start_hour", None) is not None):
                dims["clock_rest"] = True
        return dims

    @staticmethod
    def _normalized_status(status: dict[str, Any]) -> dict[str, Any]:
        def pick(*keys, default=None):
            for k in keys:
                if k in status and status[k] is not None:
                    return status[k]
            return default
        out = dict(status)
        out["current_lat"] = float(pick("current_lat", "latitude", "lat", default=0.0))
        out["current_lng"] = float(pick("current_lng", "longitude", "lng", default=0.0))
        out["simulation_progress_minutes"] = int(pick("simulation_progress_minutes", "sim_min", "progress_minutes", default=0))
        return out

    def _history(self, driver_id: str) -> list[dict[str, Any]]:
        try:
            resp = self._api.query_decision_history(driver_id, -1)
            return list(resp.get("records", []) or []) if isinstance(resp, dict) else []
        except Exception:  # noqa: BLE001
            return []

    def _diagnose_prev_take_failure(self, st: dict[str, Any], history: list[dict[str, Any]]) -> None:
        """诊断（observability，不改行为）：上一步乐观 mark_taken 前留了该货有效性快照(_last_take_dbg)；若本步
        history 显示那笔 take_order 被 server 判『已失效』，打印 模型 vs server 失配字段，定位真因：
          · has_load_time=False        → 窗缺失绕过(agent 无窗检查、server 有窗)；
          · last_seen<attempt_now      → 陈货(累积图里早先观测、server 池已轮换掉)；
          · load_end 略小于 arrival     → 窗刚过(可达性裕度差一点)；
          · remove_min<attempt_now     → server remove 已过而 agent 解析更晚。
        实测本批次 ~13% take_order 已失效；此日志为下一次重跑定位用，不做任何修复动作。"""
        dbg = st.get("_last_take_dbg")
        st["_last_take_dbg"] = None
        if not dbg or not history:
            return
        cid = dbg.get("cid")
        for rec in reversed(history):  # 找最近一条该 cid 的 take_order 结果（通常就是末条）
            if not isinstance(rec, dict):
                continue
            act = rec.get("action") if isinstance(rec.get("action"), dict) else {}
            if str(act.get("action") or "") != "take_order":
                continue
            res = rec.get("result") if isinstance(rec.get("result"), dict) else {}
            rcid = str(res.get("cargo_id") or (act.get("params") or {}).get("cargo_id") or "").strip()
            if rcid != str(cid):
                continue
            if (not res.get("accepted")) and ("失效" in str(res.get("detail") or "")):
                snap = dbg.get("snap") or {}
                lw = snap.get("load_window")
                self._log.warning(
                    "%s 接单已失效 cid=%s attempt_now=%s arrival_min=%s deadhead_km=%s finish_min=%s | "
                    "has_load_time=%s load_window=%s remove_min=%s first_seen=%s last_seen=%s | "
                    "fresh=%s load_end_vs_arrival=%s remove_vs_now=%s%s",
                    _tag("CARGO_DBG"), cid, dbg.get("now_min"), dbg.get("arrival_min"),
                    dbg.get("deadhead_km"), dbg.get("finish_min"),
                    snap.get("has_load_time"), lw, snap.get("remove_min"),
                    snap.get("first_seen_min"), snap.get("last_seen_min"),
                    (snap.get("last_seen_min") == dbg.get("now_min")),
                    ((lw[1] - dbg["arrival_min"]) if (lw and dbg.get("arrival_min") is not None) else None),
                    ((snap.get("remove_min") - dbg["now_min"]) if snap.get("remove_min") is not None else None),
                    _line_end())
            return  # 只看最近一条匹配

    # ================================================================= ① 货源刷新
    def _refresh_cargo(self, st: dict[str, Any], driver_id: str, now_min: int, pos, cost_per_km: float) -> None:
        graph: CargoGraph = st["graph"]
        graph.cleanup(now_min, pos, self._speed)
        st["long_memory"].set_cost_per_km(cost_per_km)
        # 落在长休(窗长≥6h)窗内、距窗口结束≥1h 的被唤醒步(进窗 / 出窗前1h) → 查 600 做记忆/出窗规划。
        # query_cargo 本身消耗 sim 时间(scan≈实际返回 items/10)；记下本步耗时供长休 rest 的 wait 扣除，
        # 否则查询耗时会把休息推过窗口（=之前"一觉睡到窗口结束、出窗前没醒"的真因）。
        wide = _long_rest_split_on() and self._long_rest_prewake_due(st, now_min)
        items = self._observe_and_ingest(st, driver_id, now_min, pos,
                                         REST_WINDOW_REFRESH_K if wide else REFRESH_CARGO_DEFAULT_K)
        st["_last_scan_min"] = len(items) / 10.0

    def _long_rest_prewake_due(self, st: dict[str, Any], now_min: int) -> bool:
        """now 是否落在某长休(窗长≥6h)rest 窗内、且距窗口结束 ≥1h —— 即"进窗"或"出窗前1h"这两个被唤醒步
        (中途一觉睡过、不被唤醒)。这两步查 600(进窗喂长期记忆 / 出窗前规划);误差补偿的残余步(距结束<1h)
        走默认 150、不再查满 600 把时间又推过窗口。按 start≤now<end 时间窗判(不依赖可能 stale 的 state),
        仅排除 consumed/canceled 终态(长休 main/tail 不 consume,故仍会被命中)。"""
        now = int(now_min)
        for v in st["registry"].items:
            if str(v.get("kind") or "") != REST or v.get("state") in (CONSUMED, CANCELED):
                continue
            start = v.get("start_min")
            end = v.get("end_min") if v.get("end_min") is not None else v.get("expire_min")
            if start is None or end is None:
                continue
            start, end = int(start), int(end)
            if (end - start) >= REST_LONG_WINDOW_MIN and start <= now < end and (end - now) >= REST_PREWAKE_LEAD_MIN:
                return True
        return False

    def _observe_and_ingest(self, st: dict[str, Any], driver_id: str, now_min: int, pos, k: int) -> list[dict[str, Any]]:
        """单点 query_cargo(k) → 入图 + 喂 long_memory + 累积 cargo_info_by_id（不含 cleanup，调用方负责）。
        返回观测到的 items（调用方据 len 估算本步 scan_cost≈items/10、用于长休 rest 的 wait 扣除）。"""
        observed = self._api.query_cargo(driver_id, pos[0], pos[1], k=k)
        items = observed.get("items", []) if isinstance(observed, dict) else []
        st["graph"].add_observations(items, now_min)
        st["long_memory"].ingest(items, now_min, source_driver_id=driver_id)
        info = st["cargo_info_by_id"]
        for it in items:
            cargo = it.get("cargo") if isinstance(it, dict) else None
            if not isinstance(cargo, dict):
                continue
            cid = str(cargo.get("cargo_id", "")).strip()
            if not cid or cid in info:
                continue
            s = cargo.get("start") if isinstance(cargo.get("start"), dict) else {}
            e = cargo.get("end") if isinstance(cargo.get("end"), dict) else {}
            info[cid] = {
                "cargo_name": str(cargo.get("cargo_name", "") or "").strip(),
                "start_address": _endpoint_flat_text(cargo.get("start")),
                "end_address": _endpoint_flat_text(cargo.get("end")),
                "start_lat": s.get("lat"), "start_lng": s.get("lng"),
                "end_lat": e.get("lat"), "end_lng": e.get("lng"),
                "price": cargo.get("price"), "cost_time_minutes": cargo.get("cost_time_minutes"),
            }
        return items

    # ================================================================= ② 账本 / 偏好状态
    def _build_ledger_facts(self, st: dict[str, Any], history: list[dict[str, Any]], now_min: int) -> dict[str, Any]:
        """从权威 history 派生本月进度事实（manager 据此判 due/behind）。observation-only。

        记录是**嵌套**结构（编排器每步追加，见 simkit）：
          rec['action']  = {'action':'take_order','params':{...}}   ← 动作名在 action.action
          rec['result']  = {'accepted':bool,'cargo_id':..,'simulation_progress_minutes':完成时刻,
                            'pickup_deadhead_km':赴装空驶,'haul_distance_km':载货里程,'income_eligible':bool}
          rec['position_after'] = {'lat','lng'}  ← take_order 即卸货点坐标
        产出：本月品类计数 / 长途数 / **累计空驶 km（月度空驶上限类靠它）** / 累计载货 km /
        last_real_order（最近一单，时序偏好用）/ recent_orders（限长，最近在前，供时序/到访/序列判定）。"""
        info = st["cargo_info_by_id"]
        _, _, m_start_day, m_end_day, _ = time_tools.month_window_of_min(now_min)
        mpd = time_tools.MINUTES_PER_DAY
        # m_end_day 是 **exclusive** 上界（=下月第 1 天的全局日索引，见 time_tools 文档）→ 直接 *mpd 即月末刻，
        # 不能 +1（否则把次月第 1 天整天误计入本月）。
        m_lo, m_hi = m_start_day * mpd, m_end_day * mpd
        # 上一个日历月窗口（跨月补欠额的原料：偏好会引用上月指标"四月没完成的五月补"，
        # manager 需要上月各品类实际完成数才能算欠额）。首月无上月 → 空。
        prev_lo = prev_hi = None
        prev_label = None
        if m_start_day > 0:
            py, pm, p_start, p_end, _ = time_tools.month_window_of_day(m_start_day - 1)
            prev_lo, prev_hi = p_start * mpd, p_end * mpd
            prev_label = f"{py}-{pm:02d}"
        cat_counts: dict[str, int] = {}
        prev_cat_counts: dict[str, int] = {}
        long_haul = 0
        completed = 0
        month_deadhead_km = 0.0
        month_haul_km = 0.0
        day_agg: dict[int, dict[str, Any]] = {}  # 日级米表：day_idx(全局) → 当日累计（日上限类偏好用）
        orders: list[dict[str, Any]] = []  # 全部已接真实单（后面排序/截断；last_real/recent 跨月也要看）
        cont_drive_min = 0.0  # 自上次"足够长休息"以来的连续驾驶分钟（haul + deadhead + 纯空驶 reposition）——工时触发类用
        spd = max(1.0, float(getattr(self, "_speed", 60.0) or 60.0))
        prev_pos = None  # 上一记录的 position_after（算 reposition 纯空驶里程用）
        for rec in history:  # 历史按时序（旧→新）：连续驾驶累加、遇 wait≥阈值重置
            if not isinstance(rec, dict):
                continue
            act = rec.get("action") if isinstance(rec.get("action"), dict) else {}
            action_name = str(act.get("action") or "")
            pa = rec.get("position_after") if isinstance(rec.get("position_after"), dict) else None
            if action_name in ("reposition", "move"):  # 纯空驶移动也是驾驶：累加但不重置（reposition 非休息）
                pb = rec.get("position_before") if isinstance(rec.get("position_before"), dict) else None
                rep_km = 0.0
                try:  # 本段空驶里程：优先本记录 position_before→after（最准），缺则退上一记录终点
                    if pb and pb.get("lat") is not None and pa and pa.get("lat") is not None:
                        rep_km = haversine_km(float(pb["lat"]), float(pb["lng"]),
                                              float(pa["lat"]), float(pa["lng"]))
                    elif prev_pos is not None and pa and pa.get("lat") is not None:
                        rep_km = haversine_km(prev_pos[0], prev_pos[1], float(pa["lat"]), float(pa["lng"]))
                except (TypeError, ValueError, KeyError):
                    rep_km = 0.0
                cont_drive_min += rep_km / spd * 60.0
                # 纯空驶也是驾驶/也是"非零工作日"：按本步完成时刻归日写日级米表
                # ——"每日驾驶≤Yh"含空驶 + rest_days 不把纯 reposition 日当休息日（对齐 calc _eval_off_days：
                # 零工作日=既无 take_order 也无 reposition）。缺完成时刻则只累连续驾驶、不归日。
                res_r = rec.get("result") if isinstance(rec.get("result"), dict) else {}
                t_rep = res_r.get("simulation_progress_minutes")
                if t_rep is not None:
                    dd = day_agg.setdefault(int(t_rep) // mpd, {"orders": 0, "deadhead_km": 0.0, "haul_km": 0.0,
                                                               "haul_minutes": 0, "long_haul": 0, "gross": 0.0,
                                                               "reposition_km": 0.0})
                    dd["reposition_km"] = dd.get("reposition_km", 0.0) + rep_km
                if pa and pa.get("lat") is not None:
                    prev_pos = (float(pa["lat"]), float(pa["lng"]))
                continue
            if action_name == "wait":  # 一次足够长的 wait（夜休/守候/中途歇）= 重置连续驾驶
                wmin = _as_float((act.get("params") or {}).get("duration_minutes")) or 0.0
                if wmin >= _REST_BREAK_MIN:
                    cont_drive_min = 0.0
                continue
            if action_name != "take_order":
                continue
            res = rec.get("result") if isinstance(rec.get("result"), dict) else {}
            if not res.get("accepted"):
                continue
            cid = str(res.get("cargo_id") or (act.get("params") or {}).get("cargo_id") or "").strip()
            if not cid or _is_wp_cargo_id(cid):
                continue
            meta = info.get(cid, {})
            cat = str(meta.get("cargo_name") or "").strip()
            t = res.get("simulation_progress_minutes")
            in_month = (t is not None) and (m_lo <= int(t) < m_hi)  # 缺完成时刻 → 不武断计入本月
            haul_min = meta.get("cost_time_minutes")
            dead_km = _as_float(res.get("pickup_deadhead_km"))
            haul_km = _as_float(res.get("haul_distance_km"))
            gross = _as_float(meta.get("price"))  # 该单货值（simkit 已 /100 转元）→ 日级收入米表
            drop = rec.get("position_after") if isinstance(rec.get("position_after"), dict) else {}
            # 连续驾驶累加：载货耗时(缺则用 haul_km/spd 兜底) + 赴装空驶分钟（空驶也是驾驶）。遇 wait≥阈值已在上面重置。
            load_drive = (float(haul_min) if haul_min is not None
                          else (float(haul_km) / spd * 60.0 if haul_km is not None else 0.0))
            cont_drive_min += load_drive + (float(dead_km) / spd * 60.0 if dead_km is not None else 0.0)
            if drop.get("lat") is not None:
                prev_pos = (float(drop["lat"]), float(drop["lng"]))  # 卸货点=下一段 reposition 的起点
            if in_month:
                completed += 1  # completed_orders = 本月已接（月度进度口径，与品类/里程计数一致）
                if cat:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
                if haul_min is not None and float(haul_min) > LONG_HAUL_MIN:
                    long_haul += 1
                if dead_km is not None:
                    month_deadhead_km += dead_km
                if haul_km is not None:
                    month_haul_km += haul_km
            elif (prev_lo is not None and t is not None and prev_lo <= int(t) < prev_hi and cat):
                prev_cat_counts[cat] = prev_cat_counts.get(cat, 0) + 1  # 上月品类完成数（补欠额用）
            if t is not None:  # 日级米表（按完成时刻归日，与月度口径一致）
                d = day_agg.setdefault(int(t) // mpd, {"orders": 0, "deadhead_km": 0.0, "haul_km": 0.0,
                                                       "haul_minutes": 0, "long_haul": 0, "gross": 0.0,
                                                       "reposition_km": 0.0})
                d["orders"] += 1
                if dead_km is not None:
                    d["deadhead_km"] += dead_km
                if haul_km is not None:
                    d["haul_km"] += haul_km
                if gross is not None and res.get("income_eligible", True):
                    d["gross"] += gross  # 与 calc 口径对齐：horizon 外完成的单不计收入
                if haul_min is not None:
                    d["haul_minutes"] += int(haul_min)
                    if float(haul_min) > LONG_HAUL_MIN:
                        d["long_haul"] += 1
            orders.append({
                "cargo_id": cid,
                "category": cat or None,
                "end_min": (int(t) if t is not None else None),
                "day": (int(t) // mpd + 1 if t is not None else None),
                "haul_km": (round(haul_km, 1) if haul_km is not None else None),
                "haul_minutes": (int(haul_min) if haul_min is not None else None),
                "deadhead_km": (round(dead_km, 1) if dead_km is not None else None),
                "gross": (round(gross, 1) if gross is not None else None),
                "income_eligible": bool(res.get("income_eligible", True)),
                "pickup": ({"lat": meta.get("start_lat"), "lng": meta.get("start_lng")}
                           if meta.get("start_lat") is not None else None),  # 起点(序列/区域偏好用)
                "drop": ({"lat": drop.get("lat"), "lng": drop.get("lng")}
                         if drop.get("lat") is not None else None),
            })
        # 最近在前（无完成时刻的排末尾）
        orders.sort(key=lambda o: (o["end_min"] is None, -(o["end_min"] or 0)))
        last_real = orders[0] if orders else None
        # 日级米表视图：today + 近 7 天 by_date（按日期可查；任意更早日期用模块级 day_meters_for_date）。
        today_idx = int(now_min) // mpd
        def _day_row(idx: int) -> dict[str, Any]:
            d = day_agg.get(idx) or {"orders": 0, "deadhead_km": 0.0, "haul_km": 0.0,
                                     "haul_minutes": 0, "long_haul": 0, "gross": 0.0, "reposition_km": 0.0}
            # drive_minutes = 载货耗时 + (赴装空驶 + 纯空驶 reposition) 折算分钟（"每日驾驶≤Yh"用这个，
            # 别用只含载货的 haul_minutes；纯空驶日 deadhead/haul 皆 0，全靠 reposition_km）。
            drive_min = int(d["haul_minutes"] + round((d["deadhead_km"] + d.get("reposition_km", 0.0)) / spd * 60.0))
            return {"orders": d["orders"], "deadhead_km": round(d["deadhead_km"], 1),
                    "haul_km": round(d["haul_km"], 1), "haul_minutes": d["haul_minutes"],
                    "drive_minutes": drive_min, "long_haul": d["long_haul"],
                    "gross": round(d.get("gross", 0.0), 1)}
        def _date_of(idx: int) -> str:
            dt = time_tools.sim_min_to_dt(idx * mpd)
            return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
        by_date = {_date_of(i): _day_row(i) for i in range(max(0, today_idx - 6), today_idx + 1)
                   if i in day_agg or i == today_idx}
        day_meters = {"today": {"date": _date_of(today_idx), **_day_row(today_idx)}, "by_date": by_date}
        # pacing 显式信号：配额节奏判断（manager 可自算，但显式给不易错）。月度配额"是否落后"=
        # 比较 进度比例(已完成/目标) 与 时间比例(month_elapsed_frac)：前者<后者 → 落后、需加力。
        din = max(1, int(m_end_day) - int(m_start_day))                 # 本月自然天数
        dom = today_idx - int(m_start_day) + 1                          # 本月第几天(1-based)
        days_left = int(m_end_day) - today_idx                          # 含今天的剩余天数
        pacing = {"day_of_month": dom, "days_in_month": din, "days_left": days_left,
                  "month_elapsed_frac": round(min(1.0, max(0.0, dom / din)), 2)}
        # 休息天数信号（"每月/每周/每N天休M天"类用）：零工作(零接单)日。
        # month_zero_work_days=本月口径(每月休M天用)；last_zero/days_since=**全局**回看(每周/每N天 跨月也准)。
        # worked = 有 take_order **或** reposition 的自然日（day_agg 现含纯空驶日）→ 与 calc _eval_off_days
        # 的"零工作日=既无接单也无空驶"口径一致，纯 reposition 日不再被误当休息日。
        worked = set(day_agg.keys())
        month_zero = sum(1 for i in range(int(m_start_day), today_idx) if i not in worked)  # 不含今天(未过完)
        # last_zero/days_since：**全局**回看到首个有活动的日（开工前的空白日不算"休息"，故下界=first_active
        # 而非 0；上界也不再 60 天封顶——92 天仿真里上次休息>60 天前仍要能定位到，"每周/每N天休M天"跨月才准）。
        first_active = min(day_agg.keys()) if day_agg else today_idx
        last_zero = next((i for i in range(today_idx - 1, first_active - 1, -1) if i not in worked), None)
        rest_days = {"month_zero_work_days": month_zero,
                     "last_zero_work_date": (_date_of(last_zero) if last_zero is not None else None),
                     "days_since_last_zero_work": (today_idx - last_zero if last_zero is not None else None)}
        # 连续驾驶信号（"连续驾驶≤Xh必须休息"类用）：自上次 wait≥阈值以来的累计驾驶分钟。
        cont = int(round(cont_drive_min))
        continuous_driving = {"minutes": cont, "text": f"{cont // 60}h{cont % 60}min",
                              "break_reset_min": _REST_BREAK_MIN}
        return {
            "month_day_range": [m_start_day, m_end_day],
            "pacing": pacing,  # {day_of_month, days_in_month, days_left, month_elapsed_frac} 配额节奏锚
            "completed_orders": completed,
            "month_category_counts": cat_counts,
            "prev_month": prev_label,                       # 上一个日历月（如 "2026-04"），首月为 None
            "prev_month_category_counts": prev_cat_counts,  # 上月品类完成数 —— 跨月补欠额(偏好联动)用
            "month_long_haul_count": long_haul,
            "month_deadhead_km": round(month_deadhead_km, 1),
            "month_haul_km": round(month_haul_km, 1),
            "day_meters": day_meters,  # 日级米表：today + 近7天 by_date（含 gross 收入）；日上限类偏好用
            "continuous_driving": continuous_driving,  # 自上次休息(wait≥阈值)的连续驾驶分钟 —— 工时触发休息类用
            "rest_days": rest_days,                     # 本月零工作日数+最近 —— "每月/周/N天休M天"类用
            "last_real_order": last_real,
            "recent_orders": orders[:_RECENT_ORDERS_CAP],
        }

    def _build_pref_status(self, st: dict[str, Any], ledger_facts: dict[str, Any], now_min: int) -> list[dict[str, Any]]:
        """偏好状态视图（manager 输入）：**raw_text(司机原话/最终判准) + canonical_text(去歧义清楚稿) + 结构化 facts**。
        rewrite-only 默认开 → canonical_text 是改写稿;改写器有改坏闭源偏好的前科(丢条件变体/过解读),故把 raw_text
        **显式抬到头部**与 canonical_text 并列、提示词令冲突时以 raw_text 为准,而非把原文埋进 facts 一个键里。
        facts **只留非空结构字段**（剔除 raw_content/clarified_text 这两个文本字段——已在 raw_text/canonical_text 显式给出，
        留在 facts 里既冗余又把"文本"混进"结构"）：ParsedPreference 有 40+ 槽，默认 passthrough 下多为 None/[]，全量 dump
        是纯噪声且误导（模型看到一堆 null 会以为"该偏好没结构"），过滤后只剩真正有值的几项。"""
        out: list[dict[str, Any]] = []
        for i, p in enumerate(st.get("parsed_prefs", [])):
            raw = str(getattr(p, "raw_content", "") or "")
            canonical = str(getattr(p, "clarified_text", None) or getattr(p, "canonical_text", None) or raw)
            full = p.to_dict() if hasattr(p, "to_dict") else {}
            facts = {k: v for k, v in full.items() if v is not None and v != "" and v != [] and v != {}
                     and k not in ("raw_content", "clarified_text", "canonical_text")}
            pid = str(getattr(p, "pref_id", None) or full.get("pref_id") or f"pref_{i}")
            out.append({"id": pid, "raw_text": raw, "canonical_text": canonical, "facts": facts})
        return out

    # ================================================================= ③ 应用 patch
    def _apply_manager_patch(self, st: dict[str, Any], patch: dict[str, Any], now_min: int) -> None:
        """把 manager 的 patch（walltime 制）转成 registry 规范（sim_min）后应用。
        rejected 存入 st 供下一轮 manager context 回喂（否则模型不知 add 失败、每步静默重发烧 token）。"""
        reg: VirtualRegistry = st["registry"]
        norm = {"add": [self._spec_walltime_to_min(s, now_min) for s in (patch.get("add") or [])],
                "cancel": patch.get("cancel") or [],
                "update": [self._spec_walltime_to_min(s, now_min) for s in (patch.get("update") or [])]}
        try:
            stats = reg.apply_patch(norm, now_min=now_min)
            st["_last_rejected"] = stats.get("rejected") or []
            if any(stats.get(k) for k in ("added", "canceled", "updated", "rejected")):
                self._log.info("%s patch stats=%s reason=%s%s", _tag("VMGR"),
                               stats, str(patch.get("reason", ""))[:80], _line_end())
        except Exception:  # noqa: BLE001 — 应用 patch 失败不打断
            self._log.exception("apply_manager_patch_failed")

    def _spec_walltime_to_min(self, spec: dict[str, Any], now_min: int) -> dict[str, Any]:
        if not isinstance(spec, dict):
            return {}
        out = dict(spec)
        for wt, mn in (("start_walltime", "start_min"), ("end_walltime", "end_min"),
                       ("expire_walltime", "expire_min")):
            if out.get(wt) is not None and out.get(mn) is None:
                m = self._resolve_walltime(out.get(wt), now_min)
                if m is not None:
                    out[mn] = int(m)
        # cargo_modifier 无结束 → 默认钳到本月末（而非永久；否则 manager 忘 cancel = 永久扭曲定价）。
        if str(out.get("kind") or "") == "cargo_modifier" and out.get("end_min") is None:
            out["end_min"] = self._month_end_min(now_min)
        return out

    @staticmethod
    def _month_end_min(now_min: int) -> int:
        """本月最后一刻 sim_min（cargo_modifier 默认有效期上限）。m_end_day 是 exclusive 上界
        （=下月第 1 天），故月末刻 = m_end_day*mpd − 1，不能再 +1（否则越界到次月第 1 天）。"""
        _, _, _m_start_day, m_end_day, _ = time_tools.month_window_of_min(int(now_min))
        return int(m_end_day) * time_tools.MINUTES_PER_DAY - 1

    def _resolve_walltime(self, walltime: Any, now_min: int) -> int | None:
        """绝对墙钟 → sim_min。接受「M月D日HH:MM」/「YYYY-MM-DD HH:MM」/「HH:MM」。"""
        if walltime is None or str(walltime).strip() == "":
            return None
        text = str(walltime).strip()
        cn = _CN_WALL_RE.match(text)
        if cn:
            try:
                mo, day = int(cn.group(1)), int(cn.group(2))
                hh = int(cn.group(3)) if cn.group(3) is not None else 0
                mm = int(cn.group(4)) if cn.group(4) is not None else 0
                year = time_tools.sim_min_to_dt(int(now_min)).year
                return time_tools.sim_min_of(f"{year:04d}-{mo:02d}-{day:02d}", f"{hh:02d}:{mm:02d}")
            except Exception:  # noqa: BLE001
                return None
        text = text.replace("T", " ").replace("t", " ").strip().rstrip("Zz").strip()
        try:
            if "-" in text:
                date_part, _, time_part = text.partition(" ")
                return time_tools.sim_min_of(date_part, time_part.strip() or "00:00")
            # 裸『HH:MM』(无日期)：锚到 **now 当天** 该钟点(允许 ≤now)，而非 next-occurrence。否则在 rest 应
            # 开始之后才注入(now 已过 start，如 23:30 注夜休『23:00→06:00』)会把 start/end 双双推到次日 →
            # 跨午夜窗被整体右移一天、当晚无 active rest → 整夜无休/穿窗违规。跨午夜成对窗由 normalize 的
            # end<=start→+1440 兜正。bare 'HH:MM' 在事件/deadline 语境表达的就是『今天该钟点』。
            bare = _BARE_CLOCK_RE.match(text)
            if bare:
                dt = time_tools.sim_min_to_dt(int(now_min))
                return time_tools.sim_min_of(f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}",
                                             f"{int(bare.group(1)):02d}:{int(bare.group(2)):02d}")
            return time_tools.walltime_to_sim_min(now_min, text)
        except Exception:  # noqa: BLE001
            return None

    # ================================================================= ④ 策略场规划
    @staticmethod
    def _rest_window_active(now_min: int, waypoints) -> bool:
        """当前是否落在某 active rest 的休息窗内(now in [start,end))。此时窗内接真实货必违规(穿窗罚 +
        '所有路必含 active waypoint' 过滤剔除)，r0 必是 rest 单跳 → 可跳过 plan_route 的真实货枚举、只构造
        waypoint 单跳，省算力。waypoints = active_waypoints()(已过 active + combo-gating)。"""
        now = int(now_min)
        for v in waypoints or []:
            if str(v.get("kind") or "") != REST:
                continue
            start = v.get("start_min")
            end = v.get("end_min") if v.get("end_min") is not None else v.get("expire_min")
            if start is None or end is None:
                continue
            if int(start) <= now < int(end):
                return True
        return False

    def _plan_strategy_field(self, st, driver_id, now_min, pos, cost_per_km, status) -> list[dict[str, Any]]:
        graph: CargoGraph = st["graph"]
        reg: VirtualRegistry = st["registry"]
        lm: LongMemory = st["long_memory"]
        waypoints = reg.active_waypoints(now_min=now_min)
        wp_ids = self._inject_virtual_waypoints(graph, waypoints, now_min, pos)

        region_weight = 0.0
        if self._rest_window_active(now_min, waypoints):
            # 休息窗内 rest 必选：窗内接真实货必被穿窗罚 + "所有路必含 active waypoint" 过滤剔除，r0 必是 waypoint
            # 单跳。**跳过 plan_route 的真实货枚举**(尤其 tail 步刚查 600 货、再 depth4 枚举它们最慢、纯浪费)，
            # 只构造 waypoint 单跳，大幅省算力——用户口径：休息窗虚拟单就是必选的、没必要 plan_route 那么多。
            try:
                routes = build_rest_single_paths(graph, wp_ids, now_min, pos, self._speed, cost_per_km)
            except Exception:  # noqa: BLE001
                self._log.exception("build_rest_single_paths_failed"); routes = []
            self._log.info("%s driver=%s step=%s rest-locked: skip plan_route(虚拟单必选)%s",
                           _tag("PREPLAN"), driver_id, st["step"], _line_end())
        else:
            completed = int(status.get("completed_order_count", 0) or 0)
            region_active = completed >= LM_WARMUP_THRESHOLD
            region_weight = REGION_VALUE_WEIGHT if region_active else 0.0
            region_value_fn = (lambda lat, lng: lm.region_value_normalized(lat, lng, now_min, asking_driver_id=driver_id)) if region_active else None
            region_coverage_fn = (lambda lat, lng: lm.region_coverage_cells(lat, lng, now_min)) if region_active else None
            # 策略场 cargo_modifier 折进**搜索期** net_yield（见 feasible_edges_from）：被升值的低基价目标货
            # （落后的配额品类等）才能存活进候选池——否则升值只是在 plan_route 选完的 top-k 上重排，target 货早被
            # 按毛价剪掉、永远 0 命中（这是实测 4 月水果升值整月哑火的真因）。fn=None / bonus=0 → plan_route 行为不变。
            cargo_yield_delta_fn, max_cargo_yield_bonus = self._build_cargo_yield_delta_fn(reg, now_min)
            try:
                routes = plan_route(
                    graph, pos, now_min, cost_per_km, self._speed,
                    deadline_time_min=None, destination=None, top_k=10, max_depth=4,
                    region_value_fn=region_value_fn, region_coverage_fn=region_coverage_fn,
                    region_value_weight=region_weight, scan_margin_min=QUERY_SCAN_MARGIN_MIN,
                    cargo_yield_delta_fn=cargo_yield_delta_fn, max_cargo_yield_bonus=max_cargo_yield_bonus,
                    beam_width=(PLAN_BEAM_WIDTH if _plan_beam_on() else None), beam_trigger=PLAN_BEAM_TRIGGER,
                )
            except Exception:  # noqa: BLE001
                self._log.exception("plan_route_failed")
                routes = []
            # 保证 waypoint 单跳路survive top_k 砍
            try:
                # scan_margin 故意为 0(默认):rest waypoint 的装货窗是零宽 [now,now](原地立刻休),正是这条 bypass
                # 用 scan=0 才能让它穿过主搜索(主搜索 +15 会判它不可达)进池——这是 rest 单跳兜底的承重设计。
                # (deadhead 紧 expire 的可达口径漂移属 LOW/窄,不值得为它破坏 rest 机制,暂留。)
                routes = append_unique_paths(routes, build_rest_single_paths(graph, wp_ids, now_min, pos, self._speed, cost_per_km))
            except Exception:  # noqa: BLE001
                self._log.exception("build_waypoint_single_paths_failed")
        for r in routes:
            r["route_start_pos"] = [pos[0], pos[1]]
            r["route_start_time_min"] = int(now_min)
        # cargo_modifier 奖惩（偏好→定价的新通道，取代旧 typed-pref 罚款管道）
        self._apply_cargo_modifiers(routes, reg, now_min, st.get("cargo_info_by_id") or {})
        # 窗前穿窗确定性防护：真实首跳执行区间与任何 active+pending waypoint 窗口重叠 → 减该 waypoint value
        # （写 adjusted_net_yield → drop_negative 剔）。这是确定性地基：pending rest(如 23:00,现在 22:00)
        # 对 active_waypoints 不可见，靠这层把"现在起飞、跨夜窗"的长单驱负，不全靠 LLM/harness。
        self._apply_window_crossing_penalty(routes, reg, now_min, driver_id, st["step"])
        # **所有路必含全部 active waypoint**：删不含的（least-missing 兜底）
        routes = self._filter_routes_by_waypoints(routes, set(wp_ids), driver_id, st["step"])
        # drop_negative（照搬）：违规负收益真实货起手的路被剔，保留 least-bad
        routes, _ = drop_negative_yield_real_cargo_paths(routes)
        # 最终排序（照搬口径：per-min + 区域 bonus + 负收益护栏）
        def _score(r: dict[str, Any]) -> float:
            adj = float(r.get("preference_adjusted_yield", r.get("total_net_yield", 0.0)) or 0.0)
            eff = max(1, int(r.get("total_time_min", 0) or 0))
            if adj < 0:
                return adj / eff
            return (adj + region_bonus_for_path(r, region_weight)) / eff
        routes.sort(key=_score, reverse=True)
        # display cap（保所有 waypoint 单 + top N 真实）
        kept, n_real = [], 0
        for r in routes:
            hops = r.get("hops") or []
            cid = str((hops[0].get("cargo_id") if hops and isinstance(hops[0], dict) else "") or "")
            if cid and not _is_wp_cargo_id(cid):
                if n_real >= _DISPLAY_REAL_CAP:
                    continue
                n_real += 1
            kept.append(r)
        self._log.info("%s driver=%s step=%s candidates=%s waypoints=%s%s", _tag("PREPLAN"),
                       driver_id, st["step"], len(kept), len(wp_ids), _line_end())
        # 效果回喂（观测）：本步场的实测效果（命中/市场存量/顶部标尺）存 st，下一步 manager 据实调场
        self._collect_modifier_effects(st, reg, graph, kept, st.get("cargo_info_by_id") or {}, pos, now_min)
        return kept

    def _inject_virtual_waypoints(self, graph: CargoGraph, waypoints, now_min: int, pos) -> list[str]:
        """active rest/deadhead → 合成图边（cargo_id=__wp_<vid>），plan_route 枚举时穿过。
        rest: 原地静止 cost_time=休息时长；deadhead: start=end=目标坐标（planner 自动算 pos→目标空驶）。
        耗时统一由 plan_route 算（deadhead 的赴点里程）。返回注入的 cargo_id 列表。"""
        cur = int(now_min)
        # Stale-clear：先删掉所有上一步遗留的 __wp_ 合成边，再注当前 active 的。否则已 consumed/不再 active 的
        # waypoint 边滞留图中 → 被 plan_route 重新枚举甚至选中（对不存在的 registry 项 mark_consumed）。
        for _k in [k for k in graph._edges if str(k).startswith(_WP_PREFIX)]:
            del graph._edges[_k]
        out: list[str] = []
        for v in waypoints:
            vid = str(v.get("id"))
            cid = f"{_WP_PREFIX}{vid}"
            kind = v.get("kind")
            value = v.get("value")
            value = float(value) if value is not None else (_DEFAULT_REST_VALUE if kind == REST else 1500.0)
            start_min = int(v.get("start_min") or cur)
            load_at = max(start_min, cur)
            if kind == REST:
                end_min = v.get("end_min") or v.get("expire_min")
                if end_min is None:
                    self._log.warning("%s skip rest vid=%s 缺 end/expire（应已被 normalize 拒）%s",
                                      _tag("VMGR"), vid, _line_end())
                    continue
                # rest 钉目标坐标（回家=deadhead 到家 + rest 在家休）：有 params.lat/lng 用之，否则当前 pos。
                params = v.get("params") or {}
                plat, plng = params.get("lat"), params.get("lng")
                lat = float(plat) if plat is not None else pos[0]
                lng = float(plng) if plng is not None else pos[1]
                # 休息时长 = 结束 − 实际开始（pending 被 combo 门控提前注时按 start，否则按 now）。
                rest_from = max(start_min, cur)
                dur = int(end_min) - rest_from
                if dur <= 0:
                    self._log.warning("%s skip rest vid=%s 时长≤0(end=%s,from=%s)%s",
                                      _tag("VMGR"), vid, end_min, rest_from, _line_end())
                    continue
                remove_min = int(end_min)
                # 长休(窗长≥6h)拆两段：main 先睡到「窗口结束前 lead」且下游不 consume（rest 保持 active 到窗口
                # 结束、靠"所有路必含 active waypoint"挡住窗内接单）；tail(剩余≤lead)睡到窗口结束。窗长由 end−start
                # 定(不随 now 变)→ main 段不 consume、下步重注时 dur=lead 自动落入 tail。短休(<6h)维持一觉到底。
                window_len = int(end_min) - int(start_min)
                # 末日跨界守卫：rest 结束跨过仿真名义末尾(duration_days×1440)→ main 睡到 end-60 后仿真即到时长上限停、
                # tail(填窗最后1h)来不及执行→末晚整休窗最后1h无 wait→rest_ok 假违规(实测 day91 唯一夜休罚)。这类 rest
                # 不拆、走 short 一觉睡到 end，单步覆盖整窗(=改前行为，实测改前 day91 零违规)。
                sim_end_min = self._duration_days * time_tools.MINUTES_PER_DAY
                split = _long_rest_split_on() and int(end_min) <= sim_end_min
                if split and window_len >= REST_LONG_WINDOW_MIN and dur > REST_PREWAKE_LEAD_MIN:
                    rest_phase = "main"
                    dur -= REST_PREWAKE_LEAD_MIN
                elif split and window_len >= REST_LONG_WINDOW_MIN:
                    rest_phase = "tail"
                else:
                    rest_phase = "short"
                cost_time = dur
                load_end = load_at  # rest 原地静止：load=start（finish=end=remove，镜像旧 _inject_fixed_window_rest）
            elif kind == DEADHEAD:
                rest_phase = None
                params = v.get("params") or {}
                lat, lng = params.get("lat"), params.get("lng")
                if lat is None or lng is None:
                    self._log.warning("%s skip deadhead vid=%s 缺 params.lat/lng%s",
                                      _tag("VMGR"), vid, _line_end())
                    continue
                remove_min = int(v.get("expire_min") or (cur + 24 * 60))
                cost_time = _DEADHEAD_DWELL_MIN
                # 装货窗放宽到 expire：deadhead 要赴点(里程→分钟由 plan_route 算)，到达在 now 之后才装，
                # 否则零宽窗 → 不可行 → 没路含它 → 过滤退化全留。
                load_end = remove_min
            else:
                continue
            cargo = {
                "cargo_id": cid,
                "cargo_name": f"__virtual_{kind}__",
                "start": {"lat": float(lat), "lng": float(lng)},
                "end": {"lat": float(lat), "lng": float(lng)},
                "price": value,
                "cost_time_minutes": int(cost_time),
                "load_time": [sim_min_to_wall(load_at), sim_min_to_wall(load_end)],
                "remove_time": sim_min_to_wall(remove_min),
                "_virtual_waypoint": True,
                "_virtual_kind": kind,
                "_virtual_id": vid,
                "_virtual_target": [float(lat), float(lng)],
                "_virtual_rest_min": int(cost_time) if kind == REST else None,
                "_virtual_rest_phase": rest_phase,
            }
            graph._edges[cid] = {"cargo": cargo, "first_seen_min": cur, "last_seen_min": cur}
            out.append(cid)
        return out

    def _build_cargo_yield_delta_fn(self, reg: VirtualRegistry, now_min: int):
        """构造传给 plan_route 的边级 cargo_modifier hook + 正向奖励上界（剪枝可采纳性用）。

        返回 ``(delta_fn, max_bonus)``：
          - ``delta_fn(cargo, deadhead_km, haul_km, finish_min=None) -> float``：该真实货命中的 active modifier 之
            value_delta 之和（复用 apply_modifiers_to_yield，base=0 → 直接得 delta）。category/geo/cargo_id 在裸 cargo
            上即可命中；attribute 谓词需 hop 上下文（haul_km/finish_min，cargo 顶层没有）→ 仅当存在 attribute 类
            modifier 时才浅拷贝注入这些字段（避开热路径无谓拷贝）。finish_min 支持"首单 T 前完成"类。
          - ``max_bonus``：所有正向 delta 之和（任一跳最多获得的奖励上界），plan_paths 据此抬高乐观剪枝上界。
        无 active modifier → ``(None, 0.0)``，plan_route 行为与改造前字节一致。"""
        mods = reg.active_modifiers(now_min=now_min)
        if not mods:
            return None, 0.0
        max_bonus = sum(max(0.0, float(m.get("value_delta") or 0.0))
                        for m in mods if m.get("kind") == CARGO_MODIFIER)
        needs_hop_ctx = any(isinstance(m.get("predicate"), dict) and "attribute" in m["predicate"] for m in mods)

        def _delta_fn(cargo: dict[str, Any], deadhead_km: float, haul_km: float, finish_min=None) -> float:
            view = cargo
            if needs_hop_ctx:  # attribute 谓词要 hop 上下文(haul_km/finish_min)；只在需要时浅拷贝
                view = {**cargo}
                if haul_km is not None and "haul_km" not in cargo:
                    view["haul_km"] = haul_km
                if finish_min is not None:
                    view["finish_min"] = finish_min
            return apply_modifiers_to_yield(0.0, view, mods, deadhead_km=deadhead_km)

        return _delta_fn, float(max_bonus)

    def _apply_cargo_modifiers(self, routes, reg: VirtualRegistry, now_min: int, info_by_id) -> None:
        """passthrough 落账：cargo_modifier 已在 plan_route **搜索期**折进每跳 net_yield（见 _build_cargo_yield_delta_fn
        / feasible_edges_from），这里 **不再二次叠加**（否则双计）。只把结果落到下游读取的字段：
        ``preference_adjusted_yield = total_net_yield``；真实 hop ``adjusted_net_yield = net_yield``
        （drop_negative / 窗前穿窗 / 最终排序读它们）。

        命中诊断/效果统计已移到 _collect_modifier_effects（同时回喂 manager），这里只做纯 passthrough。"""
        mods = reg.active_modifiers(now_min=now_min)
        for r in routes:
            r["preference_adjusted_yield"] = round(float(r.get("total_net_yield", 0.0) or 0.0), 2)
        if not mods:
            # 无 active modifier → 不写 hop.adjusted_net_yield，drop_negative 回退 path_adj，
            # 与边级注入改造前字节一致（无配额偏好的司机 / agent_lixian 副本路径不受影响）。
            return
        for r in routes:
            for hop in (r.get("hops") or []):
                if not isinstance(hop, dict):
                    continue
                cid = str(hop.get("cargo_id", "") or "")
                if _is_wp_cargo_id(cid):
                    continue
                hop["adjusted_net_yield"] = float(hop.get("net_yield", 0.0) or 0.0)

    def _collect_modifier_effects(self, st, reg: VirtualRegistry, graph: CargoGraph, routes,
                                  info_by_id, pos, now_min: int) -> None:
        """**modifier 效果回喂**（观测层，绝不改值）：每张 active modifier 本步的实测效果，存 st 喂给
        **下一步** manager（manager 在 plan 之前跑，所以它看到的是上一步规划的事实）：
          - ``pool_hits``：本步最终候选路里命中该谓词的真实跳数（0=目标货没进池/谓词失配）。
          - ``market_count``：当前货源图里满足该谓词的真实货数（含被剪枝/不可达的）——区分『市场上根本没有
            目标货（加码无意义）』vs『有货但没进池（力度不够/谓词错）』，这是治盲调抖动的关键事实。
          - ``top_route_yields``：本步前 5 条候选路的 preference_adjusted_yield —— 给升值校准提供"要赢过谁"的标尺。
        market 扫描对原始货补算 haul_km/deadhead_km（顶层没有），attribute 谓词才能命中。失败静默（观测层不打断主循环）。"""
        mods = reg.active_modifiers(now_min=now_min)
        if not mods:
            st["_modifier_fx"] = None
            return
        pool: dict[str, int] = {str(m.get("id")): 0 for m in mods}
        for r in routes:
            for hop in (r.get("hops") or []):
                if not isinstance(hop, dict):
                    continue
                cid = str(hop.get("cargo_id", "") or "")
                if _is_wp_cargo_id(cid):
                    continue
                mc = _match_cargo_for_hop(hop, info_by_id.get(cid) or {})
                for m in mods:
                    if apply_modifiers_to_yield(0.0, mc, [m], deadhead_km=hop.get("deadhead_km")):
                        pool[str(m.get("id"))] += 1
        market: dict[str, int] = {str(m.get("id")): 0 for m in mods}
        try:
            for cargo in graph.iter_cargos():
                cid = str(cargo.get("cargo_id", "") or "")
                if not cid or cid.startswith("__"):
                    continue  # 合成 waypoint 边不算市场货
                view = _market_cargo_view(cargo, pos)
                for m in mods:
                    if apply_modifiers_to_yield(0.0, view, [m], deadhead_km=view.get("_deadhead_km")):
                        market[str(m.get("id"))] += 1
        except Exception:  # noqa: BLE001 — 观测层
            pass
        tops = [round(float(r.get("preference_adjusted_yield", r.get("total_net_yield", 0.0)) or 0.0), 1)
                for r in routes[:5]]
        st["_modifier_fx"] = {
            "from_step": st.get("step"),
            "per_modifier": [{"id": str(m.get("id")), "pool_hits": pool[str(m.get("id"))],
                              "market_count": market[str(m.get("id"))],
                              "value_delta": _as_float(m.get("value_delta"))} for m in mods],
            "top_route_yields": tops,
        }
        if routes and all(v == 0 for v in pool.values()):
            self._log.info("%s %d active modifier 对 %d 候选路 0 命中（market=%s；target 没进池或谓词失配）%s",
                           _tag("VMGR"), len(mods), len(routes),
                           {k: v for k, v in market.items()}, _line_end())

    def _apply_window_crossing_penalty(self, routes, reg: VirtualRegistry, now_min: int,
                                       driver_id: str, step: int) -> None:
        """真实货首跳执行区间 [now, finish_min] 与任何 active+pending **REST** 窗口 [start, end/expire]
        重叠 → 该候选 adjusted_yield 额外减去该 rest 的 value（同写首跳 adjusted_net_yield → 能驱负 →
        drop_negative 剔）。占着 rest/事件守候窗干活 = 与该 waypoint 冲突，代价=放弃其价值。多窗各减。
        **只罚 REST**：穿窗罚的承重语义只对『需保留的静止/守候窗』成立；deadhead 是一次性移动、无守候语义，
        把『真实货执行区间与一个 pending deadhead 的 [start,expire] 重叠』当穿窗并扣 value 在语义上不成立
        （会误罚本该顺路先接的真实货）——deadhead 不产穿窗罚。
        **pending 也算**：22:00 决策时 23:00 的 rest 尚 pending、对 active_waypoints 不可见，靠这层兜。"""
        wps = [v for v in reg.selectable(now_min=now_min) if v.get("kind") == REST]
        if not routes or not wps:
            return
        now = int(now_min)
        windows: list[tuple[int, int, float]] = []
        for v in wps:
            ws = int(v.get("start_min") or now)
            we = v.get("end_min") if v.get("end_min") is not None else v.get("expire_min")
            if we is None:
                continue
            # value 只在缺省(None)时套默认；显式 value=0 被尊重（与注入侧 price=0 口径一致，不凭空罚）。
            raw = v.get("value")
            val = float(raw) if raw is not None else _DEFAULT_REST_VALUE
            windows.append((ws, int(we), val))
        if not windows:
            return
        for r in routes:
            hops = r.get("hops") or []
            first = hops[0] if hops and isinstance(hops[0], dict) else None
            if first is None:
                continue
            cid = str(first.get("cargo_id") or "")
            if _is_wp_cargo_id(cid):
                continue  # 虚拟单首跳不罚（它就是要被执行的 waypoint）
            fm = first.get("finish_min", r.get("finish_min"))
            if fm is None:
                continue
            fm = int(fm)
            pen = sum(val for ws, we, val in windows if now < we and fm > ws)  # 区间重叠
            if pen:
                base = float(r.get("preference_adjusted_yield", r.get("total_net_yield", 0.0)) or 0.0)
                r["preference_adjusted_yield"] = round(base - pen, 2)
                first["adjusted_net_yield"] = round(
                    float(first.get("adjusted_net_yield", first.get("net_yield", 0.0)) or 0.0) - pen, 2)
                r.setdefault("preference_reasons", []).append({"type": "window_crossing", "penalty": round(pen, 2)})

    def _filter_routes_by_waypoints(self, routes, wp_ids: set[str], driver_id: str, step: int) -> list[dict[str, Any]]:
        """所有路必含全部 active waypoint：删不含的。空集兜底→含最多 waypoint 的那些（least-missing）。
        least-missing 降级会悄悄放弃"全含"铁律 → 打日志记缺了哪些 vid，便于事后归因（违规是过滤放行还是别处）。"""
        if not wp_ids:
            return routes

        def covered(r) -> set[str]:
            return {str(h.get("cargo_id")) for h in (r.get("hops") or []) if isinstance(h, dict)} & wp_ids

        full = [r for r in routes if covered(r) >= wp_ids]
        if full:
            return full
        if not routes:
            return routes
        best = max(len(covered(r)) for r in routes)
        missing = wp_ids - max((covered(r) for r in routes), key=len, default=set())
        self._log.warning("%s driver=%s step=%s least-missing 降级：无路全含 waypoint，缺=%s（保留含最多者 cov=%s）%s",
                           _tag("PREPLAN"), driver_id, step, sorted(missing), best, _line_end())
        return [r for r in routes if len(covered(r)) == best]

    # ================================================================= ⑤ harness ctx
    def _build_manager_context(self, st, now_min, pos, pref_status, ledger_facts, feedback) -> dict[str, Any]:
        nw = time_tools.now(now_min)
        sim_day = int(now_min) // time_tools.MINUTES_PER_DAY + 1  # 全局 1-based 仿真日（首日豁免用）
        virtuals = self._virtuals_view(st, now_min)
        self._annotate_modifier_quota(virtuals, ledger_facts)  # 给 category 升值单绑本月实接数(防误读/凭记忆)
        return {
            "now": {"date": nw.get("date"), "weekday": nw.get("weekday_cn", nw.get("weekday")),
                    "is_weekend": nw.get("is_weekend"), "hhmm": nw.get("hhmm"), "day": sim_day,
                    "sim_min": int(now_min)},
            "pos": [pos[0], pos[1]],
            "pref_status": pref_status,
            "ledger_facts": ledger_facts,
            "hot_zones": self._hot_zones(st, now_min),
            "virtuals": virtuals,
            "modifier_effects": st.get("_modifier_fx"),  # 上一步规划实测：per_modifier 命中/市场存量 + top yields
            "geo_hints": self._geo_hints_for(pref_status),  # 偏好里提到的地名→中心坐标(编 geo 谓词用,无 geocoder 兜底)
            "rejected_last": st.get("_last_rejected") or None,  # 上轮 add 被拒+原因 → 据此修正，别重发
            "harness_feedback": feedback or None,
        }

    @staticmethod
    def _annotate_modifier_quota(virtuals: list[dict[str, Any]], ledger_facts: dict[str, Any]) -> None:
        """给每张 **category 谓词**的 cargo_modifier 直接绑上该品类的**本月**实接数(+上月)——确定性来自 ledger，
        manager 判配额达标只读这个钉死在单上的数、不再凭记忆估或把上月当本月误读。实测真因:建材本月已 59/目标12,
        manager 时而读对(59→应撤却没撤)、时而读成 6(保持升值)→升值一直不撤→过度接 74 单、挤掉更值钱的货=丢钱。
        绑数后 manager 看到的是『mod_x 建材升值: 本月已 59』,配合 prompt 铁律(达标必落 cancel)即可稳定撤奖。
        只标 category 谓词单(配额/禁品类);geo/attribute/cargo_id 谓词无品类计数可绑、跳过。原地改 virtuals 行。"""
        mcc = ledger_facts.get("month_category_counts") or {}
        pmcc = ledger_facts.get("prev_month_category_counts") or {}
        for r in virtuals or []:
            if not isinstance(r, dict) or str(r.get("kind") or "") != "cargo_modifier":
                continue
            pred = r.get("predicate") if isinstance(r.get("predicate"), dict) else {}
            cat = str(pred.get("category") or "").strip()
            if cat:
                r["category_count_this_month"] = int(mcc.get(cat, 0) or 0)
                r["category_count_prev_month"] = int(pmcc.get(cat, 0) or 0)

    @staticmethod
    def _geo_hints_for(pref_status) -> dict[str, list[float]]:
        """从偏好原话里挑出 GEO_HINTS 表命中的地名 → {地名:[lat,lng]}，喂给 manager 编 geo 谓词。
        只带偏好真正提到的城市（保持 context 精简）；无命中 → 空。"""
        blob = " ".join(str((p or {}).get("raw_text") or "") + " " + str((p or {}).get("canonical_text") or "")
                        for p in (pref_status or []))
        return {name: [lat, lng] for name, (lat, lng) in GEO_HINTS.items() if name in blob}

    @staticmethod
    def _virtuals_view(st, now_min: int) -> list[dict[str, Any]]:
        """registry.view + 给每个 *_min 配一份**墙钟字符串**（start_wall/end_wall/expire_wall）。
        裸 sim_min（如 20880）让 LLM 心算极易翻车（曾把绝对 sim_min 误读成钟点改坏寿宴时间），
        墙钟与 prompt 里的钟点同一口径，去掉这个事故源。registry 保持 dep-free，转换在 loop 侧做。"""
        rows = st["registry"].view(now_min=now_min)
        for r in rows:
            for mn, wk in (("start_min", "start_wall"), ("end_min", "end_wall"), ("expire_min", "expire_wall")):
                if r.get(mn) is not None:
                    r[wk] = sim_min_to_wall(int(r[mn]))
        return rows

    def _hot_zones(self, st: dict[str, Any], now_min: int, n: int = 8) -> list[dict[str, Any]]:
        """LongMemory 观测出的热区中心坐标（top-N），供 manager 注 deadhead 目标——避免它凭空编坐标。
        只暴露中心坐标 + 归一价值 + 当前时段货量计数（晚上去晚上有货的区）。空/未热身 → []。"""
        lm: LongMemory = st.get("long_memory")
        if lm is None:
            return []
        try:
            rows = lm.hot_zones(int(now_min), n=n, asking_driver_id=st.get("driver_id"))
        except Exception:  # noqa: BLE001 — 热区只是建议，绝不打断
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            c = row.get("center") or {}
            out.append({"lat": c.get("lat"), "lng": c.get("lng"),
                        "region_value_norm": row.get("region_value_norm"),
                        "bucket_count": row.get("bucket_count")})
        return out

    def _build_harness_context(self, st, now_min, pos, top_route) -> dict[str, Any]:
        nw = time_tools.now(now_min)
        sim_day = int(now_min) // time_tools.MINUTES_PER_DAY + 1
        first = None
        if top_route and top_route.get("hops"):
            h0 = top_route["hops"][0]
            cargo = h0.get("cargo") if isinstance(h0, dict) else {}
            cid = str(h0.get("cargo_id", "") or "")
            af = top_route.get("finish_pos") or pos
            first = {
                "cargo_id": cid,
                "kind": ("rest/deadhead虚拟单" if _is_wp_cargo_id(cid) else "real"),
                "category": str(cargo.get("cargo_name", "") or ""),
                "after_first_order": {"wall": sim_min_to_wall(int(top_route.get("finish_min", now_min))),
                                      "lat": af[0] if isinstance(af, (list, tuple)) else None,
                                      "lng": af[1] if isinstance(af, (list, tuple)) else None},
                "adjusted_yield": top_route.get("preference_adjusted_yield"),
                "total_time_min": top_route.get("total_time_min"),
            }
        return {
            "now": {"date": nw.get("date"), "weekday": nw.get("weekday_cn", nw.get("weekday")),
                    "is_weekend": nw.get("is_weekend"), "hhmm": nw.get("hhmm"), "day": sim_day},
            "pos": [pos[0], pos[1]],
            "top_route": {"first_hop": first} if first else {"first_hop": None},
            "virtuals": self._virtuals_view(st, now_min),
            # 复核判准用**原话**(raw_text);改写稿可能曲解,违规判定以司机原文为准。
            "pref_texts": [(s.get("raw_text") or s.get("canonical_text")) for s in (st.get("_pref_status_cache") or [])]
                          or [str(getattr(p, "raw_content", "") or getattr(p, "clarified_text", None) or "")
                              for p in st.get("parsed_prefs", [])],
        }

    # ================================================================= ⑥ 首跳 → 动作
    def _route_first_hop_action(self, st, top_route, now_min: int, pos) -> dict[str, Any]:
        if not top_route or not top_route.get("hops"):
            return self._wait_action(10, "no_route")
        h0 = top_route["hops"][0]
        cid = str(h0.get("cargo_id", "") or "")
        if not cid:
            return self._wait_action(10, "no_first_hop")
        if _is_wp_cargo_id(cid):
            return self._waypoint_action(st, cid, now_min, pos)
        return {"action": "take_order", "params": {"cargo_id": cid}, "reason_brief": "r0 real first-hop"}

    def _waypoint_action(self, st, cid: str, now_min: int, pos) -> dict[str, Any]:
        """合成 waypoint 第一跳 → 真实动作。rest→wait(休息时长)；deadhead→reposition(目标);到点即 consumed。"""
        reg: VirtualRegistry = st["registry"]
        edge = st["graph"]._edges.get(cid) or {}
        cargo = edge.get("cargo") or {}
        vid = str(cargo.get("_virtual_id") or "")
        kind = cargo.get("_virtual_kind")
        if kind == REST:
            phase = str(cargo.get("_virtual_rest_phase") or "short")
            mins = int(cargo.get("_virtual_rest_min") or cargo.get("cost_time_minutes") or 1)
            if phase == "short":
                reg.mark_consumed(vid)  # 短休(<6h)一觉到底，维持原样
                return self._wait_action(max(1, mins), "rest virtual")
            # 长休 main/tail：本步 ① 已按需查 600(进窗喂记忆 / 出窗前规划)，查询耗 scan(≈items/10)分钟;
            # wait 扣掉 scan，让「查询耗时 + wait」精确落到目标(main→窗口结束前1h；tail→窗口结束)。
            # main/tail 均**不 consume**：rest 保持 active 到 end_min 自然过期——既让出窗前1h能再被唤醒走 tail，
            # 又靠"所有路必含 active waypoint"在窗内(含误差补偿的残余步)全程挡住接单。不足 600 时 scan 自动变小、
            # wait 相应变大(=用户口径"不足600→查询后仍在窗内→wait到窗口结束")；残余误差靠 rest 仍 active 下轮补。
            scan = float(st.get("_last_scan_min") or 0.0)
            wait = max(1, int(round(mins - scan)))
            tag = "rest long-main: sleep to end-1h" if phase == "main" else "rest long-tail: fill to end"
            return self._wait_action(wait, tag)
        if kind == DEADHEAD:
            tgt = cargo.get("_virtual_target") or [None, None]
            lat, lng = tgt[0], tgt[1]
            if lat is None or lng is None:
                return self._wait_action(1, "deadhead missing target")
            # 已在目标点(≈1km) → 完成；否则空驶过去（reposition 一步到位 → 标 consumed）
            try:
                if haversine_km(float(pos[0]), float(pos[1]), float(lat), float(lng)) < 1.0:
                    reg.mark_consumed(vid)
                    return self._wait_action(1, "deadhead arrived")
            except (TypeError, ValueError):
                pass
            reg.mark_consumed(vid)
            return {"action": "reposition", "params": {"latitude": float(lat), "longitude": float(lng)},
                    "reason_brief": "deadhead virtual"}
        return self._wait_action(1, "unknown waypoint")

    @staticmethod
    def _wait_action(minutes: int, reason: str) -> dict[str, Any]:
        return {"action": "wait", "params": {"duration_minutes": max(1, int(minutes))}, "reason_brief": reason[:120]}

    # ================================================================= 日志
    def _log_tokens(self, driver_id: str, st: dict[str, Any]) -> None:
        try:
            usage = self._api.get_last_model_usage() if hasattr(self._api, "get_last_model_usage") else None
        except Exception:  # noqa: BLE001
            usage = None
        if not isinstance(usage, dict):
            return
        tot = st.setdefault("_tok", {"total": 0})
        step_total = int(usage.get("total_tokens", 0) or 0)
        tot["total"] += step_total
        self._log.info("%s driver=%s step=%s step_total=%s cumulative=%s%s", _tag("TOKENS"),
                       driver_id, st["step"], step_total, _total("TOKENS", tot["total"]), _line_end())


# 向后兼容别名（旧引用 LLMLedEngine 的测试/脚本仍可用名字找到入口；行为=新引擎）。
LLMLedEngine = StrategyFieldEngine
