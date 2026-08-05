"""事件触发型偏好监听器 — 确定性、不调 LLM、可独立单测。

黑盒复赛数据新形态：preferences 项可带 ``"type":"事件触发型"`` + 结构化 ``trigger`` dict，例如

  - ``{"event":"on_date","date":"2026-04-10","reference_cargo":"last_completed_before_date",
       "return_within_days":7}`` —— 到 date 触发：须 return_within_days 内回到"该日前最后完成单
       的装货地"并停留（时长在 content 原文里，如"到了至少停一个小时"），否则罚 penalty_amount。
  - ``{"event":"first_take_order_touch_city","city":"上海市"}`` —— 首次接到装/卸货地涉该市的单
       触发：之后装卸地涉该市的货每接一次罚一次。

这类义务的生效时刻由**运行时事件**决定，不能靠 Virtual Manager LLM 自觉注单（skip-gate 会连续
跳过、"首次接单触城"发生在任意一步）。本模块每步在 ledger 更新后被调（``loop._decide`` ②→③ 之间）：

  1. 确定性检测触发条件；
  2. 触发后向 VirtualRegistry **幂等 ensure** 现成虚拟单（机制零新增）：
     - on_date → ``deadhead(赴参考单装货地) + rest(到点停留 dwell)`` 的 combo（combo_seq 定序）；
     - touch_city → ``cargo_modifier{predicate:{region:城市}, value_delta:-penalty}``
       （货源端点带 city 文本，region 子串可命中；首单本身不罚——"以后"才罚）。

停留 rest 的时间窗每步随 now 前滑（start=now, end=now+dwell）：combo 门控把 seq2 压 pending 到
seq1 赴点 consumed，滑窗保证"到点后实际 wait 满 dwell"（rest 耗时=end−max(start,now)，见
``loop._inject_virtual_waypoints``）。manager 误 cancel 注入单时升 rev 重注（终态同 id 不可复活，
且 combo 门控会被 canceled 前置级联毒化 → 整组换 combo_id 重注）。

未知 event 类型：只登记状态，经 manager context 的 ``event_preferences`` 段交 LLM 按 content
原话处理（见 ``loop._build_manager_context``）。

铁律与 registry 相同：时间一律绝对 sim_min；耗时（赴点里程、停留）交 plan_route 统一算。
状态存 ``st["event_state"]``（内存态，与 registry 同生命周期）。dep-free，不 import loop。
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from . import time_tools
from .virtual_registry import CANCELED, CONSUMED, EXPIRED

# 注入单 id / combo 前缀（manager prompt 约定：pref_keys 含 "evt:" 的单不要 cancel）
_ID_PREFIX = "evt_"

# content 原文里的中文/阿拉伯数字（"停一个小时" / "一周内" / "3天内"）
_CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_NUM_RE = r"([0-9]+(?:\.[0-9]+)?|一|二|两|三|四|五|六|七|八|九|十|半)"
_DWELL_ANCHOR = r"(?:停|待|歇|呆|守|住|休息|静止|停留)[^。；，,\n]{0,10}?"


def _txt_num(s: Any) -> float | None:
    s = str(s or "").strip()
    if s == "半":
        return 0.5
    if s in _CN_NUM:
        return float(_CN_NUM[s])
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def dwell_minutes_from_text(text: str, default: int = 60) -> int:
    """从 content 原文提取"到点至少停多久"（锚定 停/待/歇 等动词，防误抓其它数字）。缺省 60 分钟。"""
    t = str(text or "")
    m = re.search(_DWELL_ANCHOR + _NUM_RE + r"\s*个?\s*小时", t)
    if m:
        v = _txt_num(m.group(1))
        if v:
            return max(1, int(round(v * 60)))
    m = re.search(_DWELL_ANCHOR + _NUM_RE + r"\s*分钟", t)
    if m:
        v = _txt_num(m.group(1))
        if v:
            return max(1, int(round(v)))
    return default


def days_from_text(text: str) -> int | None:
    """从 content 原文提取"N 天内/一周内"的回访期限（trigger 缺 return_within_days 时的回退）。"""
    t = str(text or "")
    m = re.search(_NUM_RE + r"\s*个?\s*(?:星期|周)", t)
    if m:
        v = _txt_num(m.group(1))
        if v:
            return max(1, int(round(v * 7)))
    m = re.search(_NUM_RE + r"\s*天", t)
    if m:
        v = _txt_num(m.group(1))
        if v:
            return max(1, int(round(v)))
    return None


def _pid_of(p: Any, idx: int) -> str:
    """事件偏好的稳定 id：pref_id（若有）否则 raw_content md5（跨重建稳定）。"""
    pid = getattr(p, "pref_id", None)
    if pid:
        return str(pid)
    raw = str(getattr(p, "raw_content", "") or "")
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12] if raw else f"pref_{idx}"


def has_event_prefs(parsed_prefs: list[Any]) -> bool:
    return any(getattr(p, "event_trigger", None) for p in (parsed_prefs or []))


# --------------------------------------------------------------------- 主入口
def ensure(st: dict[str, Any], ledger_facts: dict[str, Any], history: list[dict[str, Any]],
           now_min: int, log: logging.Logger) -> None:
    """每步调用（ledger 更新后）：检测触发、幂等 ensure 注入单。绝不抛异常打断决策。"""
    parsed = st.get("parsed_prefs") or []
    events = [(i, p) for i, p in enumerate(parsed) if getattr(p, "event_trigger", None)]
    if not events:
        return
    reg = st.get("registry")
    if reg is None:
        return
    es = st.setdefault("event_state", {})
    for idx, p in events:
        pid = _pid_of(p, idx)
        trig = dict(getattr(p, "event_trigger", None) or {})
        ev = str(trig.get("event") or "").strip()
        st_ev = es.setdefault(pid, {"fired": False, "done": False, "rev": 0, "note": ""})
        try:
            if ev == "on_date":
                _ensure_on_date(st, reg, st_ev, pid, p, trig, ledger_facts, int(now_min), log)
            elif ev == "first_take_order_touch_city":
                _ensure_touch_city(st, reg, st_ev, pid, p, trig, history, int(now_min), log)
            else:
                # 未知事件类型：交 Virtual Manager（context event_preferences 段）按 content 处理
                st_ev["note"] = st_ev.get("note") or "unknown_event_type: 交 Virtual Manager 按原文处理"
        except Exception:  # noqa: BLE001 — 单条事件失败不影响其它偏好/决策
            log.exception("event_watcher ensure failed pid=%s event=%s", pid, ev)


def _combo_members(reg: Any, combo_id: str) -> dict[int, dict[str, Any]]:
    """combo 内 seq→成员（同 seq 取最新 _seq 的——rev 重注后旧 canceled 组整体不再用）。"""
    out: dict[int, dict[str, Any]] = {}
    for v in reg.items:
        if str(v.get("combo_id") or "") != combo_id:
            continue
        seq = int(v.get("combo_seq") or 0)
        cur = out.get(seq)
        if cur is None or int(v.get("_seq", 0)) >= int(cur.get("_seq", 0)):
            out[seq] = v
    return out


def _modifier_alive(reg: Any, pid: str) -> dict[str, Any] | None:
    """该事件已注入且仍 active/pending 的 cargo_modifier（按 id 前缀找）。"""
    for v in reg.items:
        if (str(v.get("id") or "").startswith(f"{_ID_PREFIX}{pid}_mod")
                and v.get("state") in ("active", "pending")):
            return v
    return None


# --------------------------------------------------------------------- on_date
def _ensure_on_date(st: dict[str, Any], reg: Any, st_ev: dict[str, Any], pid: str,
                    p: Any, trig: dict[str, Any], ledger_facts: dict[str, Any],
                    now_min: int, log: logging.Logger) -> None:
    if st_ev.get("done"):
        return
    date_str = str(trig.get("date") or "").strip()
    if not date_str:
        st_ev["note"] = "on_date 缺 date 字段"
        st_ev["done"] = True
        return
    date_min = time_tools.sim_min_of(date_str)
    if now_min < date_min:
        return  # 未到触发日（事件偏好从仿真起点就可见，静候即可）
    st_ev["fired"] = True

    rev = int(st_ev.get("rev") or 0)
    combo = f"{_ID_PREFIX}{pid}" + (f"#{rev}" if rev else "")
    members = _combo_members(reg, combo)
    go, dwell = members.get(1), members.get(2)

    # 履约/失败终态判定
    if (go is not None and go.get("state") == CONSUMED
            and dwell is not None and dwell.get("state") == CONSUMED):
        st_ev["done"] = True
        st_ev["note"] = "fulfilled: 已回装货地并停留"
        log.info("[EVENT] pid=%s on_date fulfilled (回到 %s 装货地并停留)", pid, date_str)
        return
    if go is not None and go.get("state") == EXPIRED:
        st_ev["done"] = True
        st_ev["note"] = "回访窗口已过未履约（罚分由赛方结算判定）"
        log.warning("[EVENT] pid=%s on_date expired unfulfilled", pid)
        return

    # 解析履约点（参考货源装货地）——解析一次即缓存（"该日前最后完成单"是历史事实，不变）
    tgt = st_ev.get("target")
    if tgt is None:
        ref = str(trig.get("reference_cargo") or "").strip()
        if ref == "last_completed_before_date":
            for o in ledger_facts.get("recent_orders") or []:  # 最近在前，首个 end<date 即"最后完成单"
                pu = o.get("pickup") or {}
                if (o.get("end_min") is not None and int(o["end_min"]) < date_min
                        and pu.get("lat") is not None and pu.get("lng") is not None):
                    tgt = {"lat": float(pu["lat"]), "lng": float(pu["lng"]), "cargo_id": o.get("cargo_id")}
                    break
        if tgt is None:
            st_ev["done"] = True
            st_ev["note"] = f"无参考货源({ref or '未指定'})，无法定位履约点"
            log.warning("[EVENT] pid=%s on_date no reference cargo (ref=%s)", pid, ref)
            return
        st_ev["target"] = tgt
        log.info("[EVENT] pid=%s on_date fired: 回 %s 前最后完成单 %s 装货地 (%.4f,%.4f)",
                 pid, date_str, tgt.get("cargo_id"), tgt["lat"], tgt["lng"])

    # 期限：trigger.return_within_days > content"一周内/N天内" > 仿真末尾
    days = trig.get("return_within_days")
    days = int(days) if isinstance(days, (int, float)) and days else days_from_text(
        str(getattr(p, "raw_content", "") or ""))
    deadline = date_min + days * time_tools.MINUTES_PER_DAY if days else (
        time_tools.DURATION_DAYS * time_tools.MINUTES_PER_DAY - 1)
    dwell_min = dwell_minutes_from_text(str(getattr(p, "raw_content", "") or ""))
    pen = float(getattr(p, "penalty_amount", 0.0) or 0.0)

    # manager 误 cancel 任一成员 → 整组升 rev 重注（canceled 前置会级联毒化 combo 门控）
    if (go is not None and go.get("state") == CANCELED) or (dwell is not None and dwell.get("state") == CANCELED):
        st_ev["rev"] = rev + 1
        log.warning("[EVENT] pid=%s combo 被 cancel，升 rev=%d 重注", pid, rev + 1)
        _inject_on_date_combo(reg, st_ev, pid, tgt, deadline, dwell_min, pen, now_min, log)
        return

    if go is None:
        _inject_on_date_combo(reg, st_ev, pid, tgt, deadline, dwell_min, pen, now_min, log)
        return

    # dwell rest 时间窗随 now 前滑：到点前被 combo 门控压 pending，到点后实际 wait 满 dwell
    if dwell is not None and dwell.get("state") in ("active", "pending"):
        reg.update(str(dwell["id"]),
                   {"start_min": now_min, "end_min": now_min + dwell_min, "expire_min": deadline},
                   now_min=now_min)


def _inject_on_date_combo(reg: Any, st_ev: dict[str, Any], pid: str, tgt: dict[str, Any],
                          deadline: int, dwell_min: int, pen: float,
                          now_min: int, log: logging.Logger) -> None:
    rev = int(st_ev.get("rev") or 0)
    suf = f"#{rev}" if rev else ""
    combo = f"{_ID_PREFIX}{pid}{suf}"
    go = reg.add({
        "id": f"{_ID_PREFIX}{pid}_go{suf}", "kind": "deadhead",
        "params": {"lat": tgt["lat"], "lng": tgt["lng"]},
        "value": pen or None, "start_min": now_min, "expire_min": deadline,
        "combo_id": combo, "combo_seq": 1, "pref_keys": [f"evt:{pid}"],
        "note": f"事件义务:回单 {tgt.get('cargo_id')} 装货地",
    }, now_min=now_min)
    dwell = reg.add({
        "id": f"{_ID_PREFIX}{pid}_dwell{suf}", "kind": "rest",
        "params": {"lat": tgt["lat"], "lng": tgt["lng"]},
        "value": 0.0, "start_min": now_min, "end_min": now_min + dwell_min, "expire_min": deadline,
        "combo_id": combo, "combo_seq": 2, "pref_keys": [f"evt:{pid}"],
        "note": f"事件义务:装货地停留≥{dwell_min}分钟",
    }, now_min=now_min)
    if go and dwell:
        log.info("[EVENT] pid=%s 注入回访 combo=%s (deadhead→dwell %dmin, deadline=%s)",
                 pid, combo, dwell_min, time_tools.now(deadline)["date"])
    else:
        log.warning("[EVENT] pid=%s 注入回访 combo 被拒 go=%s dwell=%s", pid, bool(go), bool(dwell))


# --------------------------------------------------------------------- touch_city
def _ensure_touch_city(st: dict[str, Any], reg: Any, st_ev: dict[str, Any], pid: str,
                       p: Any, trig: dict[str, Any], history: list[dict[str, Any]],
                       now_min: int, log: logging.Logger) -> None:
    city = str(trig.get("city") or "").strip()
    if not city:
        st_ev["note"] = "first_take_order_touch_city 缺 city 字段"
        st_ev["done"] = True
        return
    if not st_ev.get("fired"):
        info = st.get("cargo_info_by_id") or {}
        for rec in history or []:
            act = rec.get("action") if isinstance(rec.get("action"), dict) else {}
            if act.get("action") != "take_order":
                continue
            res = rec.get("result") if isinstance(rec.get("result"), dict) else {}
            if not res.get("accepted"):
                continue
            cid = str(res.get("cargo_id") or (act.get("params") or {}).get("cargo_id") or "")
            meta = info.get(cid) or {}
            blob = f"{meta.get('start_address', '')} {meta.get('end_address', '')}"
            if city in blob:
                st_ev["fired"] = True
                st_ev["note"] = f"首单触{city}(cargo_id={cid})"
                log.info("[EVENT] pid=%s touch_city fired: cargo=%s 触 %s → 注入禁令 modifier",
                         pid, cid, city)
                break
        if not st_ev.get("fired"):
            return
    if _modifier_alive(reg, pid) is not None:
        return
    # 被 manager 误撤(canceled 终态同 id 不可复活) → 升 rev 重注；expired 只会在仿真末尾出现，视为结束
    old = next((v for v in reg.items if str(v.get("id") or "").startswith(f"{_ID_PREFIX}{pid}_mod")), None)
    if old is not None and old.get("state") == EXPIRED:
        st_ev["done"] = True
        return
    if old is not None and old.get("state") == CANCELED:
        st_ev["rev"] = int(st_ev.get("rev") or 0) + 1
        log.warning("[EVENT] pid=%s modifier 被 cancel，升 rev=%d 重注", pid, st_ev["rev"])
    suf = f"#{int(st_ev.get('rev') or 0)}" if st_ev.get("rev") else ""
    pen = abs(float(getattr(p, "penalty_amount", 0.0) or 0.0))
    obj = reg.add({
        "id": f"{_ID_PREFIX}{pid}_mod{suf}", "kind": "cargo_modifier",
        "predicate": {"region": city}, "value_delta": -pen,
        "start_min": now_min, "end_min": time_tools.DURATION_DAYS * time_tools.MINUTES_PER_DAY - 1,
        "pref_keys": [f"evt:{pid}"],
        "note": f"事件禁令:{city}装卸货每接一次罚{pen:g}",
    }, now_min=now_min)
    if obj:
        log.info("[EVENT] pid=%s 注入 %s 禁令 modifier (value_delta=-%g)", pid, city, pen)
    else:
        log.warning("[EVENT] pid=%s 禁令 modifier 注入被拒", pid)


# --------------------------------------------------------------------- 对外视图
def signature(st: dict[str, Any]) -> tuple:
    """事件状态签名（fired/done/rev 位）：变化 → loop._field_signature 变 → 唤醒 manager 编排路线。"""
    es = st.get("event_state") or {}
    return tuple(sorted((str(k), bool(v.get("fired")), bool(v.get("done")), int(v.get("rev") or 0))
                        for k, v in es.items()))


def context_view(st: dict[str, Any]) -> list[dict[str, Any]]:
    """manager context 的 event_preferences 段：trigger 结构 + content 原文 + watcher 状态 + 在册注入单。
    未知事件类型靠这段交 LLM 处理；已知类型也让 manager 看见义务全貌（便于编排路线、不要误 cancel）。"""
    parsed = st.get("parsed_prefs") or []
    es = st.get("event_state") or {}
    reg = st.get("registry")
    out: list[dict[str, Any]] = []
    for idx, p in enumerate(parsed):
        trig = getattr(p, "event_trigger", None)
        if not trig:
            continue
        pid = _pid_of(p, idx)
        st_ev = es.get(pid) or {}
        live = [str(v.get("id")) for v in (reg.items if reg is not None else [])
                if str(v.get("id") or "").startswith(f"{_ID_PREFIX}{pid}")
                and v.get("state") in ("active", "pending")]
        out.append({
            "id": pid,
            "event": str(trig.get("event") or ""),
            "trigger": dict(trig),
            "content": str(getattr(p, "raw_content", "") or ""),
            "penalty_amount": float(getattr(p, "penalty_amount", 0.0) or 0.0),
            "watcher_state": {"fired": bool(st_ev.get("fired")), "done": bool(st_ev.get("done")),
                              "note": str(st_ev.get("note") or "")},
            "injected_live_ids": live,
        })
    return out
