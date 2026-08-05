"""Deadline- and destination-bounded route planning for the LLM-led agent.

Thin wrapper over the copied multi-hop ``plan_paths`` search. It adds the two
parameters the decision-LLM steers with, agreed in the architecture discussion:

  deadline_time_min : only keep routes that finish by this absolute sim minute
                      (the model sets it to the next hard boundary, e.g. a rest
                      window or an event deadline).
  destination       : the point the driver must be at by the deadline. When set,
                      the final empty drive from a route's finish position to the
                      destination is priced in (time + cost) and infeasible routes
                      (can't reach the destination before the deadline) are dropped.

The underlying ``plan_paths`` is left untouched; all deadline/destination logic
lives here as post-processing + a bounded search horizon.
"""

from __future__ import annotations

from typing import Any

from simkit.simulation_actions import distance_to_minutes, haversine_km

from . import time_tools
from .cargo_graph import CargoGraph, sim_min_to_wall
from .monthly_quota_plan import build_monthly_quota_plan
from .path_planner import plan_paths, region_bonus_for_path
from .preference_parser import (
    completed_itinerary_event_ids,
    effective_penalty_amount,
    penalty_accounting,
    remaining_itinerary_events,
    resolve_itinerary_event_locations,
)

Position = tuple[float, float]

# feasible_edges_from uses an ABSOLUTE sim-minute horizon (the whole-run end): it
# drops a cargo whose absolute finish_min reaches it. With the 3-month span
# DURATION_DAYS=92, so 92*1440 = 2026-06-01 00:00 (March 1 + 92 days). plan_route must
# resolve an absolute horizon, never a budget. (Name kept for back-compat; it is the
# run-end horizon, not a calendar month end.)
MONTH_END_MIN = time_tools.DURATION_DAYS * time_tools.MINUTES_PER_DAY

# Virtual (synthetic, non-cargo) rest order-id prefixes — kept in sync with
# agent/tools.py so the action translator + monthly override recognise a rest "虚拟单".
VIRTUAL_REST_PREFIX = "__rest_v_"
VIRTUAL_REST_WINDOW_PREFIX = "__rest_window_v_"
VIRTUAL_MONTHLY_REST_PREFIX = "__monthly_rest_v_"
# Itinerary commitment (event: 盘库 / 寿宴) + required-pickup virtual prefixes — kept in
# sync with agent_lixian/tools.py so the action translator + event override recognise a
# commitment "虚拟单".
VIRTUAL_COMMIT_PREFIX = "__commit_v_"
VIRTUAL_PICKUP_PREFIX = "__pickup_v_"
# Only inject a fixed-window rest virtual within this lead before the window opens (or
# while inside it); past that the wait would span the whole intervening day and burn a
# work day (agent/tools.py _REST_WINDOW_INJECT_LEAD_MIN).
_REST_WINDOW_INJECT_LEAD_MIN = 4 * 60
# How many routes plan_route hands back to the caller (loop._plan). It must be WIDER than the
# caller's display top_k: preference penalties (banned 品类/禁区/穿休息窗) are applied AFTER this by
# the caller, then negatives dropped + re-ranked by ADJUSTED yield. A narrow RAW top_k cut here would
# let a cluster of RAW-loud but rule-breaking orders crowd out a compliant lower-RAW order BEFORE it
# is ever priced → the caller sees "no good order" (a false dead zone). A wide pool lets the compliant
# order reach the caller's adjusted ranking; the caller caps the DISPLAY.
_WIDE_RETURN_POOL = 50
# The commit virtual is surfaced ONLY once the driver is within this radius of the event
# point — it is the "dwell now that you're here" option. Reaching the event is handled by
# event-mode destination planning (loop._plan) + the deterministic override, NOT by a
# far-away high-value single (which magneted the model in half a day early — the 820min
# over-idle). Mirrors loop._TCT_NEAR_RADIUS_KM.
_COMMIT_NEAR_RADIUS_KM = 1.0


def _cargo_location_texts(cargo: dict[str, Any]) -> list[str]:
    """All location text fields on a cargo (start/end are 拼接地名串 like
    '广东省深圳市南山区', so a substring covers province/city/district at any level)."""
    texts: list[str] = []
    for side in ("start", "end"):
        node = cargo.get(side)
        if isinstance(node, dict):
            for key in ("city", "address", "province", "district"):
                val = node.get(key)
                if val:
                    texts.append(str(val))
    return texts


def _cargo_in_excluded_region(cargo: dict[str, Any], regions: list[str]) -> bool:
    texts = _cargo_location_texts(cargo)
    return any(region and any(region in t for t in texts) for region in regions)


def _filtered_graph(graph: "CargoGraph", regions: list[str], current_time_min: int) -> "CargoGraph":
    """Transient planning graph with any cargo touching an excluded region removed
    (matched on either endpoint)."""
    fg = CargoGraph()
    items = [
        {"cargo": cargo}
        for cargo in graph.iter_cargos()
        if not _cargo_in_excluded_region(cargo, regions)
    ]
    fg.add_observations(items, int(current_time_min))
    return fg


def plan_route(
    graph,
    start_pos: Position,
    start_time_min: int,
    cost_per_km: float,
    speed_km_h: float,
    *,
    deadline_time_min: int | None = None,
    destination: Position | None = None,
    exclude_region: str | list[str] | None = None,
    horizon_min: int | None = None,
    top_k: int = 5,
    max_depth: int = 3,
    **planner_kwargs: Any,
) -> list[dict[str, Any]]:
    """Return up to ``top_k`` routes maximizing yield within the deadline/destination box.

    The planning horizon handed to ``plan_paths`` is ABSOLUTE (it has to be —
    ``feasible_edges_from`` compares each cargo's absolute finish minute against
    it): ``deadline_time_min`` when given, otherwise the month end. ``horizon_min``
    is an optional extra look-ahead cap in minutes-from-now; neither is required.
    ``exclude_region`` (a name or list of names at any level — district/city/province)
    drops every cargo whose start or end location text contains it, before planning.
    Each returned path carries, in destination mode:
        destination_deadhead_km / _min, return_eta_min, deadline_feasible,
        post_arrival_adjusted_yield.
    """
    regions = [exclude_region] if isinstance(exclude_region, str) else list(exclude_region or [])
    regions = [str(r).strip() for r in regions if str(r).strip()]
    if regions:
        graph = _filtered_graph(graph, regions, start_time_min)

    # Resolve an ABSOLUTE planning horizon (see the module note + docstring). The
    # old code passed `deadline - now` (a relative budget) as the horizon, which
    # feasible_edges_from then compared against absolute finish minutes, silently
    # dropping every reachable cargo and returning [].
    planning_horizon = MONTH_END_MIN
    if deadline_time_min is not None:
        if int(deadline_time_min) <= int(start_time_min):
            return []
        planning_horizon = min(planning_horizon, int(deadline_time_min))
    if horizon_min is not None:
        planning_horizon = min(planning_horizon, int(start_time_min) + int(horizon_min))
    if planning_horizon <= int(start_time_min):
        return []

    # Over-fetch so deadline/destination filtering still leaves a full top_k.
    # Candidate-pool breadth aligned with the live ReactAgent (tools.plan_routes):
    # a wide planner pool plus per-depth retention (preserve_extra_depths) so the
    # deadline/destination filter and final ranking choose from the same breadth of
    # routes the ReactAgent searches, instead of an already-narrowed top-k. With the
    # caller's top_k=10 this is planner_top_k=100 / per_depth=40, matching agent.
    planner_top_k = min(200, max(top_k * 10, 100))
    per_depth_pool_k = min(80, max(top_k * 4, 40))
    extra_depth_top_k = {d: per_depth_pool_k for d in range(1, max_depth + 1)}
    raw = plan_paths(
        graph,
        start_pos,
        int(start_time_min),
        cost_per_km,
        speed_km_h,
        int(planning_horizon),
        max_depth=max_depth,
        top_k=planner_top_k,
        extra_depth_top_k=extra_depth_top_k,
        preserve_extra_depths=True,
        **planner_kwargs,
    )

    # The region-value weight (if any) was passed through to plan_paths inside
    # ``planner_kwargs``; read it here WITHOUT consuming it so this deadline/
    # destination top_k cut ranks WITH region value, mirroring the planner's own
    # candidate-pool ranking. Otherwise a rich-region route with slightly lower
    # net yield is dropped before loop._plan's final ranking ever sees it.
    region_value_weight_l2 = float(planner_kwargs.get("region_value_weight", 0.0) or 0.0)
    scored: list[tuple[float, dict[str, Any]]] = []
    for path in raw:
        finish_min = int(path.get("finish_min", start_time_min))
        if deadline_time_min is not None and finish_min > int(deadline_time_min):
            continue

        if destination is None:
            adjusted = float(path.get("total_net_yield", 0.0) or 0.0)
            eff_time = max(1, int(path.get("total_time_min", 0) or 0))
            path["destination_deadhead_km"] = 0.0
            path["destination_deadhead_min"] = 0
            path["return_eta_min"] = finish_min
            path["deadline_feasible"] = True
            path["post_arrival_adjusted_yield"] = round(adjusted, 2)
        else:
            finish_pos = path.get("finish_pos") or start_pos
            km = haversine_km(finish_pos[0], finish_pos[1], destination[0], destination[1])
            dh_min = int(distance_to_minutes(km, speed_km_h))
            eta = finish_min + dh_min
            feasible = deadline_time_min is None or eta <= int(deadline_time_min)
            if not feasible:
                continue
            adjusted = float(path.get("total_net_yield", 0.0) or 0.0) - km * float(cost_per_km)
            eff_time = max(1, int(path.get("total_time_min", 0) or 0) + dh_min)
            path["destination_deadhead_km"] = round(km, 2)
            path["destination_deadhead_min"] = dh_min
            path["return_eta_min"] = eta
            path["deadline_feasible"] = True
            path["post_arrival_adjusted_yield"] = round(adjusted, 2)

        # Rank by adjusted yield per effective minute, mirroring plan_paths'
        # per-minute ranking but net of the drive to the destination. Fold in the
        # region bonus with the SAME formula the planner uses for total_score, so
        # high-value regions survive this top_k cut and reach loop._plan's final
        # preference-adjusted ranking (where the negative-yield guard lives).
        region_bonus = region_bonus_for_path(path, region_value_weight_l2)
        scored.append(((adjusted + region_bonus) / eff_time, path))

    scored.sort(key=lambda item: item[0], reverse=True)
    # Hand back a WIDE pool, not just top_k. This RAW-yield cut is BEFORE the caller applies the
    # preference penalties + drops negatives + re-ranks by ADJUSTED yield; a narrow cut here would
    # silently discard a compliant low-RAW order under RAW-louder rule-breakers (forbidden/banned/
    # rest-crossing) and fabricate a dead zone. The caller (loop._plan) trims the DISPLAY after the
    # adjusted re-rank. (The single-hop survival the medium-risk note worries about rides on this too.)
    return [path for _score, path in scored[: max(int(top_k), _WIDE_RETURN_POOL)]]


def drop_negative_yield_real_cargo_paths(
    paths: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Remove paths that START with a real cargo whose penalty-adjusted CASH yield is
    negative — i.e. taking that order now loses money. Ported verbatim (logic) from
    agent/tools._drop_negative_yield_real_cargo_paths: this is the load-bearing
    "rest wins" mechanism. An order that plows straight through a fixed rest window
    (e.g. 0-6) eats the full rest penalty, so its ``preference_adjusted_yield`` is
    deeply negative even though the raw price looks fine; dropping it leaves only the
    compliant rest virtual (positive per-min) or repositioning, so the model cannot
    rationalise the raw gross and pick a money-loser instead of resting.

    Scope: only paths whose FIRST hop is a REAL cargo (id not starting "__").
    Compliance virtuals (__rest_*/__home_*/__commit_*/__pickup_*/__monthly_*) are
    never dropped — they ARE how the driver complies and are scored apart. Drop test keys on the
    FIRST hop (the only one executed before the model re-plans): ``first_hop adjusted_net_yield`` < 0
    drops a money-losing first hop. A MULTI-hop path that is negative ONLY because of a later hop /
    path-level obligation penalty is KEPT — its compliant first hop is still takeable. SINGLE-hop
    paths additionally test the whole-path ``preference_adjusted_yield`` < 0 (identical to the first
    hop there, but it also carries a 穿休息窗 path-level rest penalty), so the "rest wins" drop is
    unchanged for them.

    Fallback: never strand the model on an EMPTY real-cargo set — when EVERY real
    path loses money, keep the single LEAST-bad one (highest first-hop
    adjusted_net_yield) so it can still weigh "take the least-bad order vs reposition
    vs rest" instead of spinning on an empty candidate set.

    Returns (kept_paths, drop_count).
    """
    if not paths:
        return paths, 0
    kept: list[dict[str, Any]] = []
    drop = 0
    kept_has_real = False
    best_dropped: tuple[float, dict[str, Any]] | None = None
    for path in paths:
        hops = path.get("hops") or []
        first = hops[0] if hops and isinstance(hops[0], dict) else None
        first_cargo = (first.get("cargo") if isinstance(first, dict) else None) or {}
        first_id = str((first or {}).get("cargo_id") or first_cargo.get("cargo_id") or "").strip()
        is_real_first = bool(first_id) and not first_id.startswith("__")
        if is_real_first:
            path_adj = float(path.get("preference_adjusted_yield", 0.0) or 0.0)
            first_hop_adj = float((first or {}).get("adjusted_net_yield", path_adj) or 0.0)
            n_hops = sum(1 for h in hops if isinstance(h, dict))
            # Judge the FIRST hop — the ONLY hop executed before the model re-plans. A MULTI-hop path
            # whose first hop is profitable but whose later hop / path-level obligation penalty drives
            # the WHOLE path negative is KEPT: its compliant first hop is takeable (the model commits
            # only that, then re-plans). SINGLE-hop paths still test path_adj (== that one hop's full
            # cost, which carries a 穿休息窗 path-level rest penalty), so the load-bearing "rest wins"
            # drop is byte-for-byte unchanged for them.
            if first_hop_adj < 0.0 or (n_hops <= 1 and path_adj < 0.0):
                drop += 1
                # Rank fallback candidates by the cash realized on the hop the model
                # would actually commit to NOW (first hop), so a profitable later hop
                # can't make a money-losing first hop the fallback.
                if best_dropped is None or first_hop_adj > best_dropped[0]:
                    best_dropped = (first_hop_adj, path)
                continue
            kept_has_real = True
        kept.append(path)
    if not kept_has_real and best_dropped is not None:
        kept.append(best_dropped[1])
        drop -= 1
    return kept, drop


# --------------------------------------------------------------------------- #
# Rest "虚拟单" injection — light port of agent/tools._inject_generic_maintain_state +
# the force-keep single-path machinery, built DIRECTLY from typed rest fields (no
# constraint kernel). The injected synthetic cargo carries the same _virtual_rest /
# _virtual_rest_window flags + fields the (verbatim) cargo_graph already consumes, so
# pricing / ranking / translation behave identically to the live agent.
# --------------------------------------------------------------------------- #
def _iter_history_with_time_min(records: list[dict[str, Any]]):
    """Yield (record, step_start, action_start, action_end, step_end) minute spans —
    verbatim from agent/tools.py, reconstructed from each record's step_elapsed /
    query_scan / action_exec minutes."""
    sim_cursor = 0
    for record in records or []:
        step_elapsed = max(0, int(record.get("step_elapsed_minutes", 0) or 0))
        query_scan = max(0, int(record.get("query_scan_cost_minutes", 0) or 0))
        action_exec = max(0, int(record.get("action_exec_cost_minutes", 0) or 0))
        step_start = sim_cursor
        action_start = step_start + query_scan
        action_end = action_start + action_exec
        step_end = step_start + step_elapsed
        sim_cursor = step_end
        yield record, step_start, action_start, action_end, step_end


def _window_has_working_activity(records: list[dict[str, Any]], lo_min: int, hi_min: int) -> bool:
    """True if any take_order / reposition overlaps [lo_min, hi_min). The scorer only
    flags ACTIVITY inside a fixed rest window, so pure idleness since the window opened
    is still salvageable; a working action inside it means tonight is lost — roll the
    rest virtual to the next day's window (agent/tools._window_has_working_activity)."""
    if hi_min <= lo_min:
        return False
    for record, _ss, a_start, a_end, _se in _iter_history_with_time_min(records):
        action = record.get("action") if isinstance(record.get("action"), dict) else {}
        name = str(action.get("action", "")).strip().lower()
        if name in {"take_order", "reposition"} and a_end > lo_min and a_start < hi_min:
            return True
    return False


def _rest_virtual_price(pref: Any, next_cost: float | None = None) -> float:
    """Value of complying with this rest pref = the penalty it avoids. Aligned with
    agent/tools._generic_obligation_price's non-amortized branch (rest is not a
    distinct-day reward, so no /N amortization):
      1. PREFER the cap-aware ``next_violation_cost`` (from penalty_accounting) when
         it is positive — so a partially-consumed penalty_cap correctly LOWERS the
         reward (previously this was dropped, over-pricing rest near an exhausted cap;
         the divergence flagged in the audit). next_cost==0 / None falls through, same
         as agent: a missing/zero accounting value snaps back to the nominal price.
      2. else cap-clamped penalty_amount, 3. else effective_penalty_amount."""
    if next_cost is not None and float(next_cost) > 0:
        return float(next_cost)
    cap = getattr(pref, "penalty_cap", None)
    if cap is not None:
        return max(0.0, min(float(getattr(pref, "penalty_amount", 0.0) or 0.0), float(cap)))
    return effective_penalty_amount(pref)


def _inject_fixed_window_rest(graph, pref, idx, price, history, now_min, pos, horizon_min, today_idx):
    """Inject a ``__rest_window_v_`` stationary virtual for a fixed-window rest (e.g.
    0-6 静止). Verbatim logic of agent/tools._inject_generic_maintain_state's
    daily_start/daily_end branch, reading the typed window hours directly."""
    daily_start = getattr(pref, "rest_window_start_hour", None)
    daily_end = getattr(pref, "rest_window_end_hour", None)
    if daily_start is None or daily_end is None:
        return None
    win_start = int(daily_start)
    win_end = int(daily_end)
    # Crosses midnight is derived from the hours (start > end), NOT the parsed flag
    # (memory fixed-rest-window-overrest-bug: the flag was sometimes mis-tagged).
    crosses = win_start > win_end
    window_duration_min = (win_end + 24 - win_start) * 60 if crosses else (win_end - win_start) * 60
    if window_duration_min <= 0:
        return None
    mpd = time_tools.MINUTES_PER_DAY
    window_day_idx = today_idx
    window_start_min = window_day_idx * mpd + win_start * 60
    window_end_min = window_start_min + window_duration_min
    cur = int(now_min)
    inside = window_start_min <= cur < window_end_min
    # Roll to the NEXT day's window only when this day's is fully over, OR the driver
    # already worked inside it tonight (window lost). A few idle minutes past the start
    # keep tonight's window so the driver can rest the REMAINING time.
    if cur >= window_end_min or (inside and _window_has_working_activity(history, window_start_min, cur)):
        window_day_idx += 1
        window_start_min = window_day_idx * mpd + win_start * 60
        window_end_min = window_start_min + window_duration_min
    if cur < window_start_min - _REST_WINDOW_INJECT_LEAD_MIN:
        return None
    load_at_min = max(window_start_min, cur)
    # The wait that this virtual translates to STARTS at ``cur`` (when the model picks it), so its
    # length must run UNTIL ``window_end`` — i.e. ``window_end - cur`` — NOT ``window_end - load_at_min``
    # (the window's own duration). When picked a few minutes BEFORE the window opens (cur <
    # window_start, the 4h proactive-lead case), the old form rested only the window-length from cur and
    # so ENDED that many minutes BEFORE window_end, dropping the truck into a "rested but still inside
    # the 静止 window" gap (D002 22:14→07:14 vs a 23:00–08:00 宵禁): the next step then tries to move
    # at 07:14 and trips the curfew (2000) — the morning churn. Ending exactly at window_end removes the
    # gap. For cur INSIDE the window load_at_min == cur, so this is unchanged there.
    effective_duration_min = window_end_min - cur
    if effective_duration_min <= 0 or window_start_min >= int(horizon_min):
        return None
    cid = f"{VIRTUAL_REST_WINDOW_PREFIX}g_{idx}_d{window_day_idx}"
    if cid in graph._edges:
        return None
    cargo = {
        "cargo_id": cid,
        "cargo_name": f"__virtual_generic_rest_window_{win_start}_{win_end}__",
        "start": {"lat": pos[0], "lng": pos[1]},
        "end": {"lat": pos[0], "lng": pos[1]},
        "price": float(price),
        "cost_time_minutes": int(effective_duration_min),
        "load_time": [sim_min_to_wall(load_at_min), sim_min_to_wall(load_at_min)],
        "remove_time": sim_min_to_wall(window_end_min),
        "_virtual_rest_window": True,
        "_virtual_generic_obligation": True,
        "_virtual_rest_window_pref_idx": idx,
        "_virtual_rest_window_day_idx": window_day_idx,
        "_virtual_rest_window_start_min": window_start_min,
        "_virtual_rest_window_end_min": window_end_min,
        "_virtual_rest_window_duration_min": int(effective_duration_min),
    }
    graph._edges[cid] = {"cargo": cargo, "first_seen_min": cur, "last_seen_min": cur}
    return cid


def _inject_continuous_rest(graph, pref, idx, price, now_min, pos, horizon_min, today_idx,
                            today_longest_static_min=None):
    """Inject a ``__rest_v_`` stationary virtual for a continuous-daily rest (e.g. 每天
    连续休息≥8h). Verbatim logic of the duration_min branch."""
    hours = getattr(pref, "rest_continuous_hours", None)
    if not hours:
        return None
    duration_min = int(round(float(hours) * 60))
    if duration_min <= 0:
        return None
    # ALREADY-SATISFIED GATE. Mirrors the mature engine's obligation scheduler
    # (agent_lixian/constraint_obligation_scheduler.build_next_obligations: a
    # maintain_state obligation is dropped once evaluate_constraint reports
    # ``ev.satisfied``, so _inject_generic_maintain_state is never called for it). The
    # light port injects straight from typed fields and skipped that upstream gate, so
    # WITHOUT this it re-injects today's continuous-rest virtual at full price on EVERY
    # step even after the driver already rested the required block today — the model then
    # rests twice a day (e.g. 00:00 then 14:44). today_longest_static_min is today's
    # longest continuous static block (day_activity.static_day_report), the SAME quantity
    # note.reconcile_static_obligations uses to flag daily_rest today_satisfied, so this
    # stays drift-free and consistent with the snapshot the model already sees.
    if today_longest_static_min is not None and int(today_longest_static_min) >= duration_min:
        return None
    mpd = time_tools.MINUTES_PER_DAY
    today_start = today_idx * mpd
    today_end = today_start + mpd
    latest_start = today_end - duration_min
    earliest = max(int(now_min), today_start)
    if earliest > latest_start:
        return None
    cid = f"{VIRTUAL_REST_PREFIX}g_{idx}_d{today_idx}"
    if cid in graph._edges:
        return None
    cargo = {
        "cargo_id": cid,
        "cargo_name": f"__virtual_generic_rest_{duration_min}m__",
        "start": {"lat": pos[0], "lng": pos[1]},
        "end": {"lat": pos[0], "lng": pos[1]},
        "price": float(price),
        "cost_time_minutes": int(duration_min),
        "load_time": [sim_min_to_wall(earliest), sim_min_to_wall(latest_start)],
        "remove_time": sim_min_to_wall(today_end),
        "_virtual_rest": True,
        "_virtual_generic_obligation": True,
        "_virtual_rest_minutes": int(duration_min),
        "_virtual_rest_pref_idx": idx,
    }
    graph._edges[cid] = {"cargo": cargo, "first_seen_min": int(now_min), "last_seen_min": int(now_min)}
    return cid


def _monthly_rest_price(pref: Any) -> float:
    """Per-day value of one off-day toward a monthly "N整天" quota. Unlike agent (which
    reads the cap-aware next-violation cost), the light port amortizes the total penalty
    over the target N so an all-or-nothing cap (~10000) does not dominate the per-minute
    ranking on every recommended day — a full-day stop priced at total/N stays moderate
    (~total/N over a day), so a profitable order can still out-rank it on a busy day."""
    target = int(getattr(pref, "rest_monthly_days", 0) or 0)
    amt = float(getattr(pref, "penalty_amount", 0.0) or 0.0)
    cap = getattr(pref, "penalty_cap", None)
    total = amt if amt > 0 else (float(cap) if cap is not None else 0.0)
    if total <= 0:
        return 0.0
    return total / target if target > 0 else total


def _inject_monthly_rest(graph, pref, idx, price, history, now_min, pos, month_days, blackout_days):
    """Inject a ``__monthly_rest_v_`` full-day stationary stop so the model can
    VOLUNTARILY take a monthly off-day, mirroring agent/tools._inject_generic_maintain_state's
    monthly branch. Gated by the calendar plan (build_monthly_quota_plan): only on a day
    that is still a legal rest candidate (today_available) AND is recommended (spaced) or
    forced — so monthly rest spreads across the month instead of clustering, while the
    deterministic override still guarantees the count if the model skips it."""
    if month_days is None:
        return None
    try:
        plan = build_monthly_quota_plan(
            pref, idx, history or [], int(now_min), int(month_days), blackout_days=blackout_days
        )
    except Exception:  # noqa: BLE001 — planning must never break injection
        return None
    if plan is None or not plan.today_available:
        return None
    if not (plan.recommended_today or plan.forced_today):
        return None
    mpd = time_tools.MINUTES_PER_DAY
    today_idx = int(now_min) // mpd
    day_end = (today_idx + 1) * mpd
    if int(now_min) >= day_end:
        return None
    remaining = max(1, day_end - int(now_min))
    cid = f"{VIRTUAL_MONTHLY_REST_PREFIX}g_{idx}_d{today_idx}"
    if cid in graph._edges:
        return None
    cargo = {
        "cargo_id": cid,
        "cargo_name": f"__virtual_generic_full_stop_d{today_idx}__",
        "start": {"lat": pos[0], "lng": pos[1]},
        "end": {"lat": pos[0], "lng": pos[1]},
        "price": float(price),
        "cost_time_minutes": int(remaining),
        "load_time": [sim_min_to_wall(int(now_min)), sim_min_to_wall(int(now_min))],
        "remove_time": sim_min_to_wall(day_end),
        "_virtual_monthly_rest": True,
        "_virtual_generic_obligation": True,
        "_virtual_monthly_rest_full_stop": True,
        "_virtual_monthly_rest_pref_idx": idx,
        "_virtual_monthly_rest_day_idx": today_idx,
        "_virtual_monthly_rest_forced_today": bool(plan.forced_today),
        "_virtual_monthly_rest_recommended_today": bool(plan.recommended_today),
    }
    graph._edges[cid] = {"cargo": cargo, "first_seen_min": int(now_min), "last_seen_min": int(now_min)}
    return cid


def inject_rest_virtuals(
    graph, rest_prefs, history_records, now_min, pos, *,
    horizon_min=MONTH_END_MIN, month_days=None, blackout_days=None,
    today_longest_static_min=None,
):
    """Inject stationary rest "虚拟单" so "go rest now" competes in plan_route's
    per-minute ranking against real cargo. Built directly from typed rest fields (light
    port of agent/tools._inject_generic_maintain_state — no constraint kernel):
      fixed_window     -> __rest_window_v_ for the upcoming window (4h lead),
      continuous_daily -> __rest_v_ for today's required continuous block, ONLY while
                          today is not yet satisfied (today_longest_static_min < block);
                          once today's block is rested it is NOT re-injected (else the
                          model rests twice a day),
      monthly_days     -> __monthly_rest_v_ full-day stop, only on a recommended/forced
                          day per build_monthly_quota_plan (needs month_days; the
                          deterministic override is the backstop).
    On a day the monthly full-rest is on the calendar, the daily / fixed-window single is
    SUPPRESSED (the full-day stop already covers it — avoids resting twice in one day);
    safe because a dirtied day re-enables the daily single and a clean day is static anyway.
    Stale rest virtuals from a previous step/position are removed first so the synthetic
    order tracks the CURRENT position. Returns the injected cargo ids."""
    if graph is None or not rest_prefs:
        return []
    stale = [
        cid for cid, e in list(getattr(graph, "_edges", {}).items())
        if isinstance(e, dict) and isinstance(e.get("cargo"), dict)
        and (e["cargo"].get("_virtual_rest") or e["cargo"].get("_virtual_rest_window")
             or e["cargo"].get("_virtual_monthly_rest"))
    ]
    for cid in stale:
        graph._edges.pop(cid, None)
    today_idx = int(now_min) // time_tools.MINUTES_PER_DAY
    # Cap-aware next-violation cost per rest pref, so a partially/fully consumed
    # penalty_cap LOWERS the rest reward (aligns _rest_virtual_price with
    # agent/tools._generic_obligation_price <- _next_effective_violation_costs, which
    # prices off penalty_accounting). Indexed by position in rest_prefs (== idx below,
    # since we pass rest_prefs). Best-effort: any failure -> {} so pricing falls back
    # to the cap-clamp / effective branch (no regression, never breaks injection).
    try:
        next_cost_by_idx = {
            int(it["index"]): float(it.get("next_violation_cost", 0.0) or 0.0)
            for it in penalty_accounting(rest_prefs, history_records or [])
            if it.get("index") is not None
        }
    except Exception:  # noqa: BLE001 — pricing must never break injection
        next_cost_by_idx = {}
    # Is a monthly full-rest on today's calendar (clean day, recommended or forced)? If so
    # SUPPRESS today's daily / fixed-window rest single: the monthly full-day stop (or the
    # forced override) already holds the vehicle static all day, which satisfies the daily /
    # window rest — a separate daily single just makes the model rest TWICE (e.g. an 8h daily
    # block THEN a full day, the observed d25 redundancy). Same gate as _inject_monthly_rest,
    # so the monthly single IS being offered whenever this is True. SAFE against a missed
    # daily rest: (a) if the day is later dirtied by work, the monthly stops being a candidate
    # next step and the daily single returns; (b) a day left clean is static all day, which
    # itself satisfies the continuous / window rest (and the satisfied-gate); (c) forced days
    # are guaranteed a full stop by the deterministic override.
    monthly_rest_today = False
    if month_days is not None:
        for _i, _p in enumerate(rest_prefs):
            if getattr(_p, "rest_type", None) != "monthly_days":
                continue
            try:
                _mplan = build_monthly_quota_plan(
                    _p, _i, history_records or [], int(now_min), int(month_days), blackout_days=blackout_days
                )
            except Exception:  # noqa: BLE001 — never break injection
                _mplan = None
            if _mplan is not None and _mplan.today_available and (_mplan.recommended_today or _mplan.forced_today):
                monthly_rest_today = True
                break
    injected: list[str] = []
    for idx, pref in enumerate(rest_prefs):
        rtype = getattr(pref, "rest_type", None)
        cid = None
        if rtype == "fixed_window":
            if monthly_rest_today:
                continue  # today's monthly full-day stop covers the window rest — no double-rest
            price = _rest_virtual_price(pref, next_cost_by_idx.get(idx))
            if price > 0:
                cid = _inject_fixed_window_rest(graph, pref, idx, price, history_records, now_min, pos, horizon_min, today_idx)
        elif rtype == "continuous_daily":
            if monthly_rest_today:
                continue  # today's monthly full-day stop covers the daily rest — no double-rest
            price = _rest_virtual_price(pref, next_cost_by_idx.get(idx))
            if price > 0:
                cid = _inject_continuous_rest(
                    graph, pref, idx, price, now_min, pos, horizon_min, today_idx,
                    today_longest_static_min=today_longest_static_min,
                )
        elif rtype == "monthly_days":
            price = _monthly_rest_price(pref)
            if price > 0:
                cid = _inject_monthly_rest(graph, pref, idx, price, history_records, now_min, pos, month_days, blackout_days)
        if cid:
            injected.append(cid)
    return injected


def inject_commitment_virtuals(
    graph, event_prefs, history_records, now_min, pos, *,
    horizon_min=MONTH_END_MIN,
):
    """Inject ``__commit_v_`` stationary "go to the event point and dwell" virtual singles
    for the NEXT pending itinerary event per pref (盘库 / 寿宴), so "attend the event"
    competes in plan_route's per-minute ranking — letting the model VOLUNTARILY schedule
    it instead of only being force-repositioned at the last moment. Light port of
    agent/tools._inject_generic_commitment, built straight from the typed
    ``itinerary_commitment`` events (resolve location -> drop completed -> next pending).
    The deterministic loop._apply_itinerary_commitment_override is the backstop guarantee;
    this is the proactive channel. Stale commit virtuals are cleared first (track current
    position / pending set). Returns the injected cargo ids."""
    if graph is None or not event_prefs:
        return []
    stale = [
        cid for cid, e in list(getattr(graph, "_edges", {}).items())
        if isinstance(e, dict) and isinstance(e.get("cargo"), dict)
        and (e["cargo"].get("_virtual_commit") or e["cargo"].get("_virtual_required_pickup"))
    ]
    for cid in stale:
        graph._edges.pop(cid, None)
    cargos = list(graph.iter_cargos())
    cur = int(now_min)
    # Cap-aware penalty-avoided price per event pref (same source as the rest virtuals).
    try:
        next_cost_by_idx = {
            int(it["index"]): float(it.get("next_violation_cost", 0.0) or 0.0)
            for it in penalty_accounting(event_prefs, history_records or [])
            if it.get("index") is not None
        }
    except Exception:  # noqa: BLE001
        next_cost_by_idx = {}
    injected: list[str] = []
    for idx, pref in enumerate(event_prefs):
        events = getattr(pref, "itinerary_commitment", None) or []
        if not events:
            continue
        resolve_itinerary_event_locations(events, cargos)
        completed = completed_itinerary_event_ids(events, history_records or [])
        remaining = remaining_itinerary_events(events, completed)
        if not remaining:
            continue
        event = remaining[0]
        try:
            lat = float(event["lat"])
            lng = float(event["lng"])
        except (KeyError, TypeError, ValueError):
            continue
        dwell = max(1, int(event.get("dwell_min") or 1))
        nb = event.get("not_before_min")
        deadline = event.get("must_complete_before_min")
        until_min = event.get("until_min")
        event_type = str(event.get("type") or ("stay" if dwell > 1 else "visit"))
        # NEAR-ONLY: surface the commit virtual only once the driver has (about) reached the
        # event point — it is the "dwell now that you're here" option. Getting TO the event is
        # handled by event-mode destination planning (loop._plan) + the deterministic override,
        # NOT by a far-away high-value single that would magnet the model in early (the old
        # over-idle). Far -> skip; destination planning guides the approach.
        if haversine_km(float(pos[0]), float(pos[1]), lat, lng) > _COMMIT_NEAR_RADIUS_KM:
            continue
        load_after = max(cur, int(nb) if nb is not None else cur)
        dl = int(deadline) if deadline is not None else (int(until_min) if until_min is not None else None)
        if dl is not None and load_after > dl:
            continue  # window already closed — the override handles any overdue case
        latest_start = max(load_after, dl - dwell) if dl is not None else load_after + time_tools.MINUTES_PER_DAY
        if load_after >= int(horizon_min):
            continue
        # Penalty avoided = the same generic price the rest virtuals use (cap-aware).
        price = _rest_virtual_price(pref, next_cost_by_idx.get(idx))
        if price <= 0:
            continue
        cid = f"{VIRTUAL_COMMIT_PREFIX}g_{idx}_{event.get('event_id') or idx}"
        if cid in graph._edges:
            continue
        cargo = {
            "cargo_id": cid,
            "cargo_name": f"__virtual_commit_{idx}__",
            "start": {"lat": lat, "lng": lng},
            "end": {"lat": lat, "lng": lng},
            "price": float(price),
            "cost_time_minutes": int(dwell),
            "load_time": [sim_min_to_wall(load_after), sim_min_to_wall(latest_start)],
            "remove_time": sim_min_to_wall((dl + dwell) if dl is not None else (latest_start + dwell)),
            "_virtual_commit": True,
            "_virtual_generic_obligation": True,
            "_virtual_commit_pref_idx": idx,
            "_virtual_commit_event_id": str(event.get("event_id") or idx),
            "_virtual_commit_event_type": event_type,
            "_virtual_commit_load_after_min": int(load_after),
            "_virtual_commit_not_before_min": int(nb) if nb is not None else None,
            "_virtual_commit_until_min": int(until_min) if until_min is not None else None,
            "_virtual_commit_deadline_min": dl,
            "_virtual_commit_dwell_min": int(dwell),
        }
        graph._edges[cid] = {"cargo": cargo, "first_seen_min": cur, "last_seen_min": cur}
        injected.append(cid)
    return injected


def build_rest_single_paths(graph, cargo_ids, now_min, pos, speed_km_h, cost_per_km, horizon_min=MONTH_END_MIN):
    """Explicit single-hop paths for the injected rest virtuals, tagged
    ``_force_include_virtual_rest``, so they reach loop._plan's route_cache even if
    plan_route's internal top_k cut dropped them. Mirrors
    agent/tools._build_virtual_rest_single_paths (region bonus omitted: the rest single
    is stationary at the current position, so destination region value is moot).

    Calls feasible_edges_from with ``scan_margin_min=0`` (default) ON PURPOSE: an in-place
    rest's load window is zero-width [now, now], so any scan margin would push arrival past
    it and make the rest unreachable — this bypass exists precisely to admit that rest."""
    if not cargo_ids:
        return []
    wanted = {str(c) for c in cargo_ids}
    out: list[dict[str, Any]] = []
    for hop in graph.feasible_edges_from(pos, int(now_min), speed_km_h, cost_per_km, int(horizon_min)):
        if str(hop.get("cargo_id")) not in wanted:
            continue
        tny = float(hop.get("net_yield", 0.0) or 0.0)
        ttm = int(hop.get("total_time_min", 0) or 0)
        out.append(
            {
                "hops": [hop],
                "total_net_yield": tny,
                "total_time_min": ttm,
                "total_load_wait_min": int(hop.get("load_wait_min", 0) or 0),
                "yield_per_min": tny / ttm if ttm > 0 else 0.0,
                "finish_min": int(hop.get("finish_min", now_min) or now_min),
                "finish_pos": hop.get("end_pos", pos),
                "depth": 1,
                "total_score": round(tny, 2),
                "_force_include_virtual_rest": True,
            }
        )
    return out


def append_unique_paths(paths: list[dict[str, Any]], extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append paths from ``extra`` not already present (keyed by hop cargo_id tuple).
    Verbatim from agent/tools._append_unique_paths."""
    seen = {tuple(str(h.get("cargo_id")) for h in (p.get("hops") or [])) for p in paths}
    for p in extra:
        key = tuple(str(h.get("cargo_id")) for h in (p.get("hops") or []))
        if key in seen:
            continue
        paths.append(p)
        seen.add(key)
    return paths
