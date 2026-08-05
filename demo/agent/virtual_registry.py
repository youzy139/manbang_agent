"""虚拟单注册表 — 多维策略场核心（新架构 C1）。

Virtual Manager LLM 经 patch（add/cancel/update/no_change）增删，系统确定性应用。三类：

  - ``rest``           原地静止/休息（对应休息偏好）。窗口 {start_min, end_min(休息止), expire_min}, value。
  - ``deadhead``       空驶到经纬度（死区脱困 / 事件 / 吸引去热区）。{params{lat,lng}, start_min, expire_min}, value。
  - ``cargo_modifier`` 货源奖惩，直接改受影响真实货 adjusted_yield。
        {predicate{category|geo|region|attribute}, value_delta, start/end_min}。
        真实货端点只有坐标无地名 → 区域/禁区/到访类用 ``geo``（circle/bbox 坐标几何），非 region 地名子串。

``rest`` / ``deadhead`` = **waypoint**：plan_route 的候选路必须包含全部 active waypoint（§7 过滤规则），
确保被执行、且多虚拟单不打架而是被统一编排。``cargo_modifier`` 非 waypoint，全局作用于命中真实货定价。

**组合 combo**：多步义务（回家 = deadhead 到家 + rest 到家后休息；特殊事件 = deadhead 到点 + rest/dwell 守候）
用 ``combo_id`` 串成有序组，``combo_seq`` 定序。

铁律：
  - 时间一律绝对 ``sim_min``；**耗时（赴点里程→分钟、休息时长）交给 plan_route 统一算**，注册表只存窗口 + 目标。
  - 状态确定性派生（refresh）：start 未到 → pending；过 expire/end → expired；cancel → canceled；
    否则 active。**consumed** 由 caller 显式标（对应动作执行）。
  - 注入上限 MAX_ACTIVE 防 runaway。与硬偏好冲突由 harness 质检兜（本模块只登记）。
确定性、dep-free、可独立单测；不调 LLM、不碰 plan_route。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

# ---- 三类 kind ----
REST = "rest"
DEADHEAD = "deadhead"
CARGO_MODIFIER = "cargo_modifier"
KINDS = {REST, DEADHEAD, CARGO_MODIFIER}
WAYPOINT_KINDS = {REST, DEADHEAD}  # 参与"所有路必含"过滤

# ---- 状态 ----
ACTIVE, PENDING, CONSUMED, EXPIRED, CANCELED = "active", "pending", "consumed", "expired", "canceled"
TERMINAL = {CONSUMED, EXPIRED, CANCELED}

MAX_ACTIVE = 24  # 每司机活跃虚拟单 + 修正器上限，防 runaway

# cargo_modifier 的 attribute 谓词支持的比较算子
_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}


def _num(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _wp_window(v: dict[str, Any]) -> tuple[int, int]:
    """waypoint 的时间窗 [start, end]：rest 用 end_min/expire_min；deadhead 用 expire_min。缺省给点宽度。"""
    start = int(v.get("start_min") or 0)
    end = v.get("end_min") if v.get("end_min") is not None else v.get("expire_min")
    return start, (int(end) if end is not None else start + 1)


def _windows_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a0, a1 = _wp_window(a)
    b0, b1 = _wp_window(b)
    return a0 < b1 and b0 < a1


class VirtualRegistry:
    """包裹 ``st['virtuals']`` 列表（可由 memory_store 持久化为内存态）。"""

    def __init__(self, items: Optional[list[dict[str, Any]]] = None) -> None:
        self.items: list[dict[str, Any]] = items if isinstance(items, list) else []
        self._seq = max([int(v.get("_seq", 0)) for v in self.items] or [0])

    # ----------------------------------------------------------------- 校验 / 规范化
    def normalize(self, spec: Any, *, now_min: int) -> Optional[dict[str, Any]]:
        """通用 schema 校验 → 规范化对象，或 None（非法）。id 缺失自动生成。"""
        if not isinstance(spec, dict):
            return None
        kind = str(spec.get("kind") or "").strip()
        if kind not in KINDS:
            return None
        now = int(now_min)
        start_min = _int(spec.get("start_min"))
        start_min = now if start_min is None else start_min
        base: dict[str, Any] = {
            "id": str(spec.get("id") or "").strip() or self._auto_id(kind),
            "kind": kind,
            "created_min": now,
            "start_min": start_min,
            "note": str(spec.get("note") or "")[:200],
            "pref_keys": [str(k) for k in (spec.get("pref_keys") or []) if str(k).strip()],
            "combo_id": (str(spec.get("combo_id")).strip() or None) if spec.get("combo_id") is not None else None,
            "combo_seq": _int(spec.get("combo_seq")),
        }
        if kind == CARGO_MODIFIER:
            pred = self._norm_predicate(spec.get("predicate"))
            delta = _num(spec.get("value_delta"))
            if pred is None or delta is None:
                return None  # cargo_modifier 必须有可命中的 predicate + value_delta
            end_min = _int(spec.get("end_min"))
            base.update(
                predicate=pred, value_delta=delta, end_min=end_min,
                expire_min=None, permanent=(end_min is None),
            )
        else:  # rest / deadhead = waypoint
            base["value"] = _num(spec.get("value"))  # 可为 None → 由 manager/默认补
            base["expire_min"] = _int(spec.get("expire_min"))
            base["end_min"] = _int(spec.get("end_min"))  # rest=休息止(until)；deadhead 一般 None
            if kind == REST and base["end_min"] is None and base["expire_min"] is None:
                return None  # rest 必须有结束钟点(end 或 expire)，否则注成"永不过期僵尸单"占位、manager 永不补
            # 跨午夜 rest 防御：正常应注绝对墙钟(22:00→次日06:00 = end_min>start_min)。若 end_min<=start_min
            # （manager 误用同日裸钟点，如 start=22:00/end=06:00 解析成同一天）→ 视为跨午夜，end/expire 各 +1 天。
            if kind == REST and base["start_min"] is not None:
                if base["end_min"] is not None and base["end_min"] <= base["start_min"]:
                    base["end_min"] += 1440
                if base["expire_min"] is not None and base["expire_min"] <= base["start_min"]:
                    base["expire_min"] += 1440
            # rest 一律一次性单（无周期机器）：测评休息偏好不保证每天同钟点，固定钟点的 daily-recurring
            # 是错误抽象——改由 manager 每天按当天原话/星期几注当晚的 rest（spec 里的 recurring 字段忽略）。
            params = spec.get("params") if isinstance(spec.get("params"), dict) else {}
            if kind == DEADHEAD:
                lat = _num(params.get("lat", params.get("latitude")))
                lng = _num(params.get("lng", params.get("longitude")))
                if lat is None or lng is None:
                    return None  # deadhead 必带经纬度
                params = {**params, "lat": lat, "lng": lng}
            base["params"] = params
        base["state"] = PENDING if start_min > now else ACTIVE
        return base

    @staticmethod
    def _norm_predicate(pred: Any) -> Optional[dict[str, Any]]:
        """命中维度（可组合，全部满足才命中）：
          - ``cargo_id``  按货 id 精确命中（单个或列表）——"必须接 240646 这票货"类**指定货源**偏好用；
          - ``category``  按 cargo_name 全等；
          - ``geo``       按起卸**坐标**几何（circle/bbox）—— 真实货端点只有 lat/lng、无地名文本，
                          故区域/禁区/到访类**必须**用 geo 坐标，``region`` 地名子串对真实货恒不命中；
          - ``region``    按起卸地名子串（仅当端点带地名文本时有效，真实货一般无 → 仅作兜底）；
          - ``attribute`` 数值字段 op 比较（haul_km/deadhead_km/gross_yield/net_yield/cost_time_minutes/finish_min）。
        至少命中一种维度才有效，否则 None（拒绝"登记成功但永不生效"的静默错觉）。"""
        if not isinstance(pred, dict):
            return None
        out: dict[str, Any] = {}
        cat = str(pred.get("category") or "").strip()
        reg = str(pred.get("region") or "").strip()
        cids_raw = pred.get("cargo_id", pred.get("cargo_ids"))
        if cids_raw is not None:
            cids = [str(c).strip() for c in (cids_raw if isinstance(cids_raw, (list, tuple, set)) else [cids_raw])
                    if str(c).strip()]
            if cids:
                out["cargo_id"] = cids
        if cat:
            out["category"] = cat
        if reg:
            out["region"] = reg
        geo = VirtualRegistry._norm_geo(pred.get("geo"))
        if geo is not None:
            out["geo"] = geo
        attr = pred.get("attribute")
        if isinstance(attr, dict):
            a = str(attr.get("attribute") or attr.get("attr") or "").strip()
            op = str(attr.get("op") or "").strip()
            v = _num(attr.get("value"))
            if a and op in _OPS and v is not None:
                out["attribute"] = {"attribute": a, "op": op, "value": v}
        return out or None

    @staticmethod
    def _norm_geo(geo: Any) -> Optional[dict[str, Any]]:
        """规范化 geo 谓词。两种形状：
          circle: {shape:"circle", lat, lng, radius_km, side?}
          bbox  : {shape:"bbox", min_lat, max_lat, min_lng, max_lng, side?}
        ``side`` ∈ {pickup, drop, any}（默认 any：起或卸任一端落入即命中）。非法 → None。"""
        if not isinstance(geo, dict):
            return None
        side = str(geo.get("side") or "any").strip().lower()
        if side not in ("pickup", "drop", "any"):
            side = "any"
        shape = str(geo.get("shape") or geo.get("type") or "").strip().lower()
        if shape in ("circle", ""):  # 默认 circle（带 lat/lng/radius 即认）
            lat = _num(geo.get("lat", geo.get("center_lat")))
            lng = _num(geo.get("lng", geo.get("center_lng")))
            rad = _num(geo.get("radius_km", geo.get("radius")))
            if lat is None or lng is None or rad is None or rad <= 0:
                return None
            return {"shape": "circle", "lat": lat, "lng": lng, "radius_km": rad, "side": side}
        if shape == "bbox":
            lo_la, hi_la = _num(geo.get("min_lat")), _num(geo.get("max_lat"))
            lo_ln, hi_ln = _num(geo.get("min_lng")), _num(geo.get("max_lng"))
            if None in (lo_la, hi_la, lo_ln, hi_ln):
                return None
            return {"shape": "bbox", "min_lat": min(lo_la, hi_la), "max_lat": max(lo_la, hi_la),
                    "min_lng": min(lo_ln, hi_ln), "max_lng": max(lo_ln, hi_ln), "side": side}
        return None

    # ----------------------------------------------------------------- 增删改
    def add(self, spec: Any, *, now_min: int) -> Optional[dict[str, Any]]:
        obj = self.normalize(spec, now_min=now_min)
        if obj is None:
            return None
        self.refresh(now_min=now_min)
        # 终态不复活：同 id 已 consumed/expired/canceled → 拒绝（否则物理删除终态记录、丢"本周期已完成"信号）。
        if any(v.get("id") == obj["id"] and v.get("state") in TERMINAL for v in self.items):
            return None
        # 幂等去重：已有等价的 active/pending 虚拟单 → no-op 返回它（防 LLM 不带 id 重发把同窗 rest 攒满 MAX_ACTIVE）。
        dup = self._find_duplicate(obj)
        if dup is not None:
            return dup
        live = [v for v in self.items if v.get("state") in (ACTIVE, PENDING)]
        if len(live) >= MAX_ACTIVE and obj["id"] not in {v["id"] for v in self.items}:
            return None  # 超上限
        self.items = [v for v in self.items if v.get("id") != obj["id"]]  # 覆盖同 id（仅非终态走到这）
        obj["_seq"] = self._next_seq()
        self.items.append(obj)
        return obj

    def _find_duplicate(self, obj: dict[str, Any]) -> Optional[dict[str, Any]]:
        """已有等价虚拟单？命中 → 返回旧的（去重，不重复登记）。判重维度按 kind：
          - cargo_modifier：(predicate 全等) AND (有效期窗口重叠) —— 仅比 predicate+delta 会把『同品类两
            义务期(本月指标 vs 本周补欠)、各自 end 不同』错并成一张，吞掉更紧的截止窗。
          - waypoint：(同 combo_id, 同 seq) AND 窗口重叠；deadhead 额外要求**目标坐标近似相等**
            (否则两个去不同地点的 deadhead 因同窗被错并、丢一个赴点义务)。一次性 rest 不同晚窗口不重叠 →
            各自登记（manager 每天注当晚的 rest，互不去重）；同晚重发 → 窗口重叠去重(no-op)。
          - combo 成员：同 (combo_id, seq) 若已有任一**非 canceled** 实例(含 consumed/expired)→ 视为本周期
            已实例化,返回它(防 manager 重发整个 combo 时新 auto-id 的 seq1 把已解锁的 seq2 重新压回 pending)。"""
        kind = obj.get("kind")
        has_combo = obj.get("combo_id") is not None and obj.get("combo_seq") is not None
        for v in self.items:
            if v.get("kind") != kind or v.get("id") == obj.get("id"):
                continue
            state = v.get("state")
            # 一般只与 active/pending 比；但 combo 成员需把终态(非 canceled)也纳入，
            # 否则去重盲区(consumed/已过期)让同一 combo 步骤被重复实例化。
            terminal_relevant = (has_combo and v.get("combo_id") == obj.get("combo_id")
                                 and v.get("combo_seq") == obj.get("combo_seq"))
            if state not in (ACTIVE, PENDING) and not (terminal_relevant and state != CANCELED):
                continue
            if kind == CARGO_MODIFIER:
                if (v.get("predicate") == obj.get("predicate")
                        and v.get("value_delta") == obj.get("value_delta")
                        and _windows_overlap(v, obj)):
                    return v
                continue
            # waypoint：同 combo、**同 seq** 才比；不同 seq 是 combo 的不同步骤，绝不去重
            if v.get("combo_id") != obj.get("combo_id"):
                continue
            if v.get("combo_seq") != obj.get("combo_seq"):
                continue
            # combo 成员：同 (combo_id, seq) 已有非 canceled 实例（含 consumed/expired）→ 本周期已实例化
            if has_combo and state != CANCELED:
                return v
            # deadhead 的本质区分维度是『去哪』(params.lat/lng)——两个去不同坐标的 deadhead 是两件事，
            # 仅窗口重叠不构成重复；坐标近似相等(<1km)才算同一赴点。rest(原地静止)params 多为 None，仍按窗口判。
            if kind == DEADHEAD and not self._same_target(v, obj):
                continue
            if _windows_overlap(v, obj):
                return v
        return None

    @staticmethod
    def _same_target(a: dict[str, Any], b: dict[str, Any]) -> bool:
        """两张 waypoint 的目标坐标是否近似相等(<1km)。两者都无坐标(原地)视为相等。"""
        pa = a.get("params") if isinstance(a.get("params"), dict) else {}
        pb = b.get("params") if isinstance(b.get("params"), dict) else {}
        la, na = _num(pa.get("lat")), _num(pa.get("lng"))
        lb, nb = _num(pb.get("lat")), _num(pb.get("lng"))
        if la is None or na is None or lb is None or nb is None:
            return (la is None or na is None) and (lb is None or nb is None)
        return _haversine_km(la, na, lb, nb) < 1.0

    @staticmethod
    def reject_reason(spec: Any, *, now_min: int) -> str:
        """add 失败的简短原因（回喂 manager，使其据此修正而非每步静默重发）。"""
        if not isinstance(spec, dict):
            return "spec 非 dict"
        kind = str(spec.get("kind") or "").strip()
        if kind not in KINDS:
            return f"kind 非法/缺失({kind or '空'})；应为 rest/deadhead/cargo_modifier"
        if kind == CARGO_MODIFIER:
            if VirtualRegistry._norm_predicate(spec.get("predicate")) is None:
                return "cargo_modifier 缺可命中 predicate(cargo_id/category/geo/region/attribute 至少一)"
            if _num(spec.get("value_delta")) is None:
                return "cargo_modifier 缺 value_delta"
        if kind == REST and _int(spec.get("end_min")) is None and _int(spec.get("expire_min")) is None:
            return "rest 缺结束钟点(end_walltime 或 expire_walltime)"
        if kind == DEADHEAD:
            params = spec.get("params") if isinstance(spec.get("params"), dict) else {}
            if _num(params.get("lat", params.get("latitude"))) is None or _num(params.get("lng", params.get("longitude"))) is None:
                return "deadhead 缺 params.lat/lng"
        return "已存在终态同 id（不可复活）或超出活跃上限 MAX_ACTIVE"

    def update(self, vid: str, patch: dict[str, Any], *, now_min: int) -> Optional[dict[str, Any]]:
        """就地更新一张非终态虚拟单（合并字段后重新规范化）。保留 _seq/created_min。"""
        for v in self.items:
            if v.get("id") == vid and v.get("state") not in TERMINAL:
                merged = {**v, **{k: val for k, val in (patch or {}).items() if k != "id"}, "id": vid}
                norm = self.normalize(merged, now_min=now_min)
                if norm is None:
                    return None
                norm["_seq"] = v.get("_seq", self._next_seq())
                norm["created_min"] = v.get("created_min", int(now_min))
                self.items = [x for x in self.items if x.get("id") != vid]
                self.items.append(norm)
                return norm
        return None

    def cancel(self, vid: str) -> bool:
        for v in self.items:
            if v.get("id") == vid and v.get("state") not in TERMINAL:
                v["state"] = CANCELED
                return True
        return False

    def mark_consumed(self, vid: str) -> bool:
        for v in self.items:
            if v.get("id") == vid and v.get("state") not in TERMINAL:
                v["state"] = CONSUMED
                return True
        return False

    def apply_patch(self, patch: dict[str, Any], *, now_min: int) -> dict[str, Any]:
        """应用 Virtual Manager 的 patch：{add:[spec...], cancel:[id...], update:[{id,...}...]}。
        返回操作统计，供日志/harness 复核。``no_change`` 不需动作。"""
        stats: dict[str, list[Any]] = {"added": [], "canceled": [], "updated": [], "rejected": []}
        for spec in (patch.get("add") or []):
            obj = self.add(spec, now_min=now_min)
            if obj:
                stats["added"].append(obj["id"])
            else:
                stats["rejected"].append({"spec": spec, "reason": self.reject_reason(spec, now_min=now_min)})
        for vid in (patch.get("cancel") or []):
            # cancel 权完全交给 manager（无周期机器后没有"误删整月夜休"的级联风险——删的只是这一张
            # 一次性单）。cancel=永久移除、不会自动重生；改窗优先 update（原子，无空窗间隙）。
            if self.cancel(str(vid)):
                stats["canceled"].append(str(vid))
        for upd in (patch.get("update") or []):
            if isinstance(upd, dict) and upd.get("id"):
                obj = self.update(str(upd["id"]), upd, now_min=now_min)
                if obj:
                    stats["updated"].append(upd.get("id"))
                else:
                    stats["rejected"].append({"id": upd.get("id"), "reason": "update 目标不存在/已终态/改后非法"})
        return stats

    # ----------------------------------------------------------------- 状态派生
    def refresh(self, *, now_min: int) -> None:
        now = int(now_min)
        for v in self.items:
            if v.get("state") in TERMINAL:
                continue
            exp = v.get("expire_min")
            if exp is None and v.get("kind") == REST:
                exp = v.get("end_min")  # 休息未消耗到结束钟点即过期
            end = v.get("end_min")
            if exp is not None and now >= int(exp):
                v["state"] = EXPIRED
            elif v.get("kind") == CARGO_MODIFIER and end is not None and now >= int(end):
                v["state"] = EXPIRED
            elif int(v.get("start_min", now)) > now:
                v["state"] = PENDING
            else:
                v["state"] = ACTIVE
        self._apply_combo_gating()

    def _apply_combo_gating(self) -> None:
        """combo 顺序门控（按 seq 升序处理，使终态沿链级联）。同 ``combo_id`` 内对每个未结算成员 v(seq=k)，
        看其所有更小 seq 兄弟的状态：
          - 任一更小 seq 兄弟 **EXPIRED/CANCELED**（前置步骤失败/被撤，**非** consumed）→ 该步前提已不成立
            （没赶到事件点就别"在事件点守候"、没回到家就别"在家休息"）→ 把 v **级联置 EXPIRED**，整条 combo 失效。
          - 否则任一更小 seq 兄弟仍 **ACTIVE/PENDING**（前置还没做完）→ v 压回 **PENDING** 等待；这一步同时
            **救回**被 derive 误判为 EXPIRED 的下游：seq1 赴点途中 now 推过了 seq2 守候窗 → derive 先标 EXPIRED，
            但此刻 seq1 未 consumed，本应继续等而非静默丢失，故拉回 pending（待 seq1 真到点后再按实际到达定 active/过期）。
          - 所有更小 seq 兄弟都 **CONSUMED**（前置真完成）→ 不干预，保留 derive 派生的 active/pending/expired。
        deterministic 保证 deadhead(seq1)→rest(seq2) 按序，且前置失败不放行幻影后继。无 combo_id/seq 的不受约束。"""
        groups: dict[str, list[dict[str, Any]]] = {}
        for v in self.items:
            cid = v.get("combo_id")
            if cid and v.get("combo_seq") is not None:
                groups.setdefault(str(cid), []).append(v)
        for members in groups.values():
            for v in sorted(members, key=lambda m: int(m.get("combo_seq") or 0)):
                if v.get("state") in (CONSUMED, CANCELED):
                    continue  # 已结算，不动
                seq = int(v.get("combo_seq") or 0)
                smaller = [s for s in members if s is not v and int(s.get("combo_seq") or 0) < seq]
                if any(s.get("state") in (EXPIRED, CANCELED) for s in smaller):
                    v["state"] = EXPIRED      # 前置步骤失败 → 整链失效（级联，升序传播）
                elif any(s.get("state") in (ACTIVE, PENDING) for s in smaller):
                    v["state"] = PENDING      # 前置未完成 → 等待（并救回被 derive 误 EXPIRED 的下游）

    # ----------------------------------------------------------------- 查询
    def active(self, *, now_min: int, kinds: Optional[set[str]] = None) -> list[dict[str, Any]]:
        self.refresh(now_min=now_min)
        return [v for v in self.items
                if v.get("state") == ACTIVE and (kinds is None or v.get("kind") in kinds)]

    def active_waypoints(self, *, now_min: int) -> list[dict[str, Any]]:
        """active rest/deadhead —— plan_route 必须全含的强制途经点（按 combo_seq/start 排序）。"""
        wps = self.active(now_min=now_min, kinds=WAYPOINT_KINDS)
        return sorted(wps, key=lambda v: (int(v.get("start_min") or 0),
                                          v.get("combo_seq") if v.get("combo_seq") is not None else 0))

    def active_modifiers(self, *, now_min: int) -> list[dict[str, Any]]:
        return self.active(now_min=now_min, kinds={CARGO_MODIFIER})

    def selectable(self, *, now_min: int) -> list[dict[str, Any]]:
        """active + pending（决策上下文展示用：active 可执行，pending 到点执行）。"""
        self.refresh(now_min=now_min)
        return [v for v in self.items if v.get("state") in (ACTIVE, PENDING)]

    def view(self, *, now_min: int, recent_terminal: int = 3) -> list[dict[str, Any]]:
        """给 Virtual Manager / harness 看的紧凑视图：全部 active+pending + 最近 N 张 terminal
        （让 LLM 分清『今晚已睡过=consumed』与『还没注』）。"""
        self.refresh(now_min=now_min)
        out: list[dict[str, Any]] = []
        terminals: list[dict[str, Any]] = []
        for v in self.items:
            (out if v.get("state") in (ACTIVE, PENDING) else terminals).append(v)
        out = [self._row(v) for v in out]
        for v in sorted(terminals, key=lambda x: int(x.get("created_min", 0) or 0), reverse=True)[:recent_terminal]:
            out.append(self._row(v))
        return out

    @staticmethod
    def _row(v: dict[str, Any]) -> dict[str, Any]:
        row = {k: v.get(k) for k in ("id", "kind", "state", "note", "start_min", "expire_min",
                                     "end_min", "combo_id", "combo_seq", "created_min")}
        if v.get("kind") == CARGO_MODIFIER:
            row["predicate"] = v.get("predicate")
            row["value_delta"] = v.get("value_delta")
        else:
            row["value"] = v.get("value")
            row["params"] = v.get("params")
        return row

    # ----------------------------------------------------------------- 内部
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _auto_id(self, kind: str) -> str:
        prefix = {REST: "rest", DEADHEAD: "dh", CARGO_MODIFIER: "mod"}.get(kind, "vo")
        return f"{prefix}_{self._seq + 1:04d}"


# ===================================================================== 定价
def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """dep-free 球面距离（与 simkit.haversine_km 同口径，半径 6371km）。"""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def cargo_endpoints(cargo: dict[str, Any]) -> dict[str, Optional[tuple[float, float]]]:
    """取一条货的起/卸坐标 (lat,lng)。兼容嵌套 ``start/end`` 节点与扁平 ``*_lat/*_lng`` 键。"""
    def _pt(node_key: str, flat_lat: str, flat_lng: str) -> Optional[tuple[float, float]]:
        node = cargo.get(node_key)
        if isinstance(node, dict):
            la, ln = _num(node.get("lat", node.get("latitude"))), _num(node.get("lng", node.get("longitude")))
            if la is not None and ln is not None:
                return (la, ln)
        la, ln = _num(cargo.get(flat_lat)), _num(cargo.get(flat_lng))
        return (la, ln) if la is not None and ln is not None else None
    return {"pickup": _pt("start", "start_lat", "start_lng"),
            "drop": _pt("end", "end_lat", "end_lng")}


def _geo_contains(geo: dict[str, Any], pt: tuple[float, float]) -> bool:
    lat, lng = pt
    if geo.get("shape") == "circle":
        return _haversine_km(geo["lat"], geo["lng"], lat, lng) <= float(geo["radius_km"])
    if geo.get("shape") == "bbox":
        return (geo["min_lat"] <= lat <= geo["max_lat"]) and (geo["min_lng"] <= lng <= geo["max_lng"])
    return False


def _geo_hit(geo: dict[str, Any], cargo: dict[str, Any]) -> bool:
    pts = cargo_endpoints(cargo)
    side = geo.get("side", "any")
    sides = ("pickup", "drop") if side == "any" else (side,)
    for s in sides:
        pt = pts.get(s)
        if pt is not None and _geo_contains(geo, pt):
            return True
    return False


def cargo_attribute_value(cargo: dict[str, Any], attribute: str,
                          deadhead_km: Optional[float] = None) -> Optional[float]:
    """从一条真实货（或其 hop 上下文）取 attribute 的数值，供 cargo_modifier 的 attribute 谓词比较。
    ``deadhead_km``（赴装空驶）由 plan_route 在 hop 上下文传入（cargo 本身不带）。"""
    a = str(attribute or "").strip()
    if a in ("deadhead_km", "pickup_deadhead_km", "空驶距离"):
        return _num(deadhead_km)
    g = _num(cargo.get("gross_yield"))
    if g is None:
        g = _num(cargo.get("price"))
    mapping = {
        "gross_yield": g, "gross": g, "price": g, "货值": g,
        "net_yield": _num(cargo.get("net_yield")),
        "haul_km": _num(cargo.get("haul_km") or cargo.get("haul_distance_km")), "运距": _num(cargo.get("haul_km")),
        "haul_minutes": _num(cargo.get("haul_minutes")),
        "cost_time_minutes": _num(cargo.get("cost_time_minutes")), "耗时": _num(cargo.get("cost_time_minutes")),
        # finish_min=该单**若现在接**的绝对完工 sim_min（hop 上下文，由 plan_route 搜索期算好注入 view）。
        # "首单 T 前完成/finish-before" 用 attribute{finish_min,">",T} 压低会晚于 T 完工的候选。
        "finish_min": _num(cargo.get("finish_min")), "完工时刻": _num(cargo.get("finish_min")),
    }
    return mapping.get(a)


def apply_modifiers_to_yield(base_yield: float, cargo: dict[str, Any],
                             active_modifiers: list[dict[str, Any]],
                             *, deadhead_km: Optional[float] = None) -> float:
    """把命中 predicate 的 active cargo_modifier 的 value_delta 叠加到某真实货 adjusted_yield。
    确定性纯函数。命中维度（全部满足才命中）：category=cargo_name 全等；geo=起卸**坐标**几何
    （circle/bbox，真实货端点只有坐标 → 区域/禁区类靠这个）；region=起卸地名子串（端点带地名才有效）；
    attribute=数值字段 op 比较。``cargo`` 兼容嵌套 start/end 与扁平字段。无 modifier → 返回 base_yield。"""
    y = float(base_yield)
    name = str(cargo.get("cargo_name") or cargo.get("name") or "").strip()
    cid = str(cargo.get("cargo_id") or "").strip()
    cities = " ".join(str(cargo.get(k) or "") for k in
                      ("start_city", "end_city", "start_district", "end_district",
                       "start_address", "end_address"))
    for m in active_modifiers or []:
        if m.get("kind") != CARGO_MODIFIER:
            continue
        pred = m.get("predicate") or {}
        hit = True
        if "cargo_id" in pred and cid not in pred["cargo_id"]:
            hit = False
        if hit and "category" in pred and name != pred["category"]:
            hit = False
        if hit and "geo" in pred and not _geo_hit(pred["geo"], cargo):
            hit = False
        if hit and "region" in pred and pred["region"] not in cities:
            hit = False
        if hit and "attribute" in pred:
            attr = pred["attribute"]
            val = cargo_attribute_value(cargo, attr["attribute"], deadhead_km)
            hit = val is not None and _OPS[attr["op"]](val, attr["value"])
        if hit:
            y += float(m.get("value_delta") or 0.0)
    return y
