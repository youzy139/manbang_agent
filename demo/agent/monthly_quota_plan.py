"""Rolling plan for monthly "N整天休息" obligations.

The monthly full-rest quota (e.g. "三月至少3个整天完全停驶") is an
all-or-nothing penalty: missing the count by one day forfeits the whole
amount. A purely reactive injector (offer a rest single each day, let the
model pick it, priority = penalty / minutes_left) defers the rest until the
priority spikes at month end — by then the remaining days are too few, or the
day is already dirtied, and the quota is missed.

This module replaces the ad-hoc spacing valve with a *calendar-aware* rolling
plan that is recomputed (stateless) from history every decision step:

* ``forced_today``     — skipping today would make the month infeasible, so
                         today MUST be a rest day regardless of route ranking.
* ``protected_days``   — future rest-capable days we cannot afford to lose;
                         an order spilling into one of them must be blocked,
                         otherwise it silently contaminates a needed day (the
                         same root cause as a haul plowing through the 0-6
                         sleep window).
* ``recommended_days`` — the soft plan: which days to *prefer* resting on
                         (latest-safe by default, or min opportunity-cost when
                         a ``day_cost_fn`` is supplied), used only to nudge
                         priority, never to force.

Feasibility is computed by *construction* — "if today is not a rest day, how
many legal rest days are still placeable?" — instead of a closed-form
``remaining_days == 2*remaining - 1`` that breaks on blackout days,
already-completed days, or month-end shortage.

The scoring rule credits ANY zero-activity calendar day toward the quota with
NO spacing requirement (two consecutive full-rest days each count), so the HARD
gate is purely count-based — enforcing spacing there could self-inflict a false
"infeasible" and miss the count. Spacing (the "陪二老 / 别连休" intent) lives only
in the SOFT recommendation layer, which prefers non-adjacent days and degrades
gracefully to consecutive when the remaining target cannot otherwise fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# Ported verbatim from agent/monthly_quota_plan.py to keep llm_agent's monthly-rest
# planning identical to the live engine. Two kernel deps are inlined so this module
# does NOT drag in the constraint compiler/IR: (1) _monthly_activity_level
# unconditionally returns ActivityLevel.STATIONARY (agent/constraint_compiler.py:1182-1197
# — monthly_days is the settlement full-rest type = a ZERO-activity natural day), so
# `stationary` is hardcoded True in build_monthly_quota_plan; (2) ActivityLevel was used
# only by that constant. REST_TYPE_MONTHLY_DAYS + ParsedPreference already exist here.
from . import time_tools
from .preference_parser import REST_TYPE_MONTHLY_DAYS, ParsedPreference

MIN_PER_DAY = 1440


@dataclass
class MonthlyQuotaPlan:
    """Calendar state for one monthly full-rest preference, at one instant."""

    pref_index: int | None
    target_days: int
    stationary: bool
    month_days: int
    today: int
    completed_days: set[int]
    dirty_days: set[int]
    blackout_days: set[int]
    available_days: list[int]
    remaining_needed: int
    today_available: bool
    forced_today: bool
    recommended_days: set[int]
    recommended_today: bool
    protected_days: set[int]
    latest_feasible_day: int | None
    feasible: bool

    def order_hits_protected_day(self, action_start_min: int, action_end_min: int) -> bool:
        """True if an order executed over [start, end) would put activity inside
        a protected day (and therefore must not be accepted)."""
        if not self.protected_days:
            return False
        start = int(action_start_min)
        end = int(action_end_min)
        if end <= start:
            return False
        for day in self.protected_days:
            day_lo = day * MIN_PER_DAY
            day_hi = day_lo + MIN_PER_DAY
            if start < day_hi and end > day_lo:
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pref_index": self.pref_index,
            "target_days": self.target_days,
            "stationary": self.stationary,
            "today": self.today,
            "completed_days": sorted(self.completed_days),
            "dirty_days": sorted(self.dirty_days),
            "blackout_days": sorted(self.blackout_days),
            "available_days": list(self.available_days),
            "remaining_needed": self.remaining_needed,
            "today_available": self.today_available,
            "forced_today": self.forced_today,
            "recommended_days": sorted(self.recommended_days),
            "recommended_today": self.recommended_today,
            "protected_days": sorted(self.protected_days),
            "latest_feasible_day": self.latest_feasible_day,
            "feasible": self.feasible,
        }


def is_monthly_quota_pref(pref: ParsedPreference) -> bool:
    return (
        getattr(pref, "rest_type", None) == REST_TYPE_MONTHLY_DAYS
        and bool(getattr(pref, "rest_monthly_days", None))
    )


def _max_nonadjacent(days: Iterable[int]) -> int:
    """Maximum count of pairwise non-adjacent (|d_i - d_j| >= 2) days.

    Greedy earliest-first is optimal for maximum independent set on a line.
    The input need not be contiguous (blackouts punch holes); adjacency is
    literal calendar adjacency, so two days separated by a hole are both
    selectable.
    """
    count = 0
    prev = -(10**9)
    for day in sorted(days):
        if day - prev >= 2:
            count += 1
            prev = day
    return count


def _select_days(days: list[int], k: int, day_cost_fn: Callable[[int], float] | None) -> set[int]:
    """Choose exactly ``k`` pairwise non-adjacent days from ``days``.

    With a ``day_cost_fn`` -> weighted-min-cost independent set via DP. Without
    one -> the *latest-safe* set (prefer resting as late as still feasible, per
    the "越晚锁死越好" principle), via greedy from the end. If ``k`` cannot be
    placed, returns the largest achievable set.
    """
    days = sorted(days)
    if k <= 0 or not days:
        return set()

    if day_cost_fn is None:
        chosen: list[int] = []
        prev = 10**9
        for day in reversed(days):
            if prev - day >= 2:
                chosen.append(day)
                prev = day
            if len(chosen) >= k:
                break
        return set(chosen)

    # Weighted min-cost selection of exactly k non-adjacent days.
    # dp[i][j] = min cost using days[i:] choosing j, where the first pick (if
    # any) is >= days[i]; transitions skip day i or take it then skip i+1.
    n = len(days)
    INF = float("inf")
    # dp indexed by (start_index, remaining_k)
    from functools import lru_cache

    costs = [float(day_cost_fn(d)) for d in days]

    @lru_cache(maxsize=None)
    def best(i: int, j: int) -> float:
        if j == 0:
            return 0.0
        if i >= n:
            return INF
        skip = best(i + 1, j)
        # take days[i]; next pick must be a day >= days[i]+2
        nxt = i + 1
        while nxt < n and days[nxt] - days[i] < 2:
            nxt += 1
        take = costs[i] + best(nxt, j - 1)
        return min(skip, take)

    # Reconstruct
    chosen_set: set[int] = set()
    i, j = 0, k
    # If k infeasible, fall back to max achievable
    achievable = _max_nonadjacent(days)
    j = min(j, achievable)
    while j > 0 and i < n:
        skip = best(i + 1, j)
        nxt = i + 1
        while nxt < n and days[nxt] - days[i] < 2:
            nxt += 1
        take = costs[i] + best(nxt, j - 1)
        if take <= skip:
            chosen_set.add(days[i])
            i, j = nxt, j - 1
        else:
            i += 1
    best.cache_clear()
    return chosen_set


def monthly_blackout_days(
    prefs: Iterable[ParsedPreference] | None, month_days: int
) -> set[int]:
    """Natural days carrying a mandatory same-day commitment, which therefore
    cannot be a zero-activity rest day.

    Commitment days are derived generically from each preference's structured
    ``itinerary_commitment`` events (their ``not_before`` / ``must_complete_before``
    / ``until`` minutes) — nothing here is keyed on a specific date or event text,
    so any dated visit/stay the parser produces is handled the same way. Only ever
    *removes* candidate days from the rest plan, so an over-estimate is
    safe-but-conservative and an under-estimate just falls back to today's
    behaviour. Recurring daily rules (8h rest, 0-6 window, home-return) are NOT
    blackouts — a parked full-stop day satisfies them.
    """
    days: set[int] = set()
    month_days = max(1, int(month_days))
    for pref in prefs or []:
        for ev in getattr(pref, "itinerary_commitment", None) or []:
            if not isinstance(ev, dict):
                continue
            anchor = ev.get("must_complete_before_min")
            if anchor is None:
                anchor = ev.get("until_min")
            if anchor is None:
                anchor = ev.get("not_before_min")
            if anchor is None:
                continue
            start = ev.get("not_before_min")
            end = ev.get("until_min")
            lo = int(start) if start is not None else int(anchor)
            hi = int(end) if end is not None else int(anchor)
            if hi < lo:
                lo, hi = hi, lo
            for d in range(lo // MIN_PER_DAY, hi // MIN_PER_DAY + 1):
                if 0 <= d < month_days:
                    days.add(d)
    return days


def build_monthly_quota_plan(
    pref: ParsedPreference,
    pref_index: int | None,
    history_records: Iterable[dict[str, Any]] | None,
    current_time_min: int,
    month_days: int,
    *,
    blackout_days: Iterable[int] | None = None,
    end_buffer_days: int = 2,
    day_cost_fn: Callable[[int], float] | None = None,
) -> MonthlyQuotaPlan | None:
    """Build the rolling plan for ``pref`` at ``current_time_min``.

    ``blackout_days``   — days that cannot be rest days (mandatory commitments).
    ``end_buffer_days`` — reserve the last N days of the month: the forced rest
                          must land BEFORE them, so any month-end special
                          preference is never clobbered by a forced full-stop
                          (a generic guard, not tied to any specific date or
                          event). The buffer days are simply
                          excluded from the rest-candidate set; a voluntary rest
                          there is still credited (it shows up in history), the
                          plan just never *plans* or *forces* a rest onto them.

    Returns ``None`` if ``pref`` is not a monthly full-rest preference.
    """

    if not is_monthly_quota_pref(pref):
        return None

    target = int(pref.rest_monthly_days or 0)
    # _monthly_activity_level(pref) returns ActivityLevel.STATIONARY unconditionally
    # in agent/ (constraint_compiler.py:1197): a monthly off-day is credited only on a
    # ZERO-activity natural day (no take_order AND no reposition), mirroring
    # calc_monthly_income._eval_off_days. Inlined here to avoid the kernel dependency.
    stationary = True
    month_days = max(1, int(month_days))
    today = int(current_time_min) // MIN_PER_DAY
    # Per-CALENDAR-MONTH planning window: the quota re-arms each calendar month (3/4/5 月各自
    # 要 N 天), the month-end safety valve fires at THIS month's end (not only the whole-run end),
    # and spacing is judged within the month. Derived from the fixed calendar anchored at
    # SIM_EPOCH (duration-agnostic): a 31-day run stays entirely in March, so this collapses to
    # the original single-month [0, month_days) behaviour.
    _y, _m, m_start, m_end_cal, _dim = time_tools.month_window_of_day(today, month_days)
    m_end = min(int(m_end_cal), month_days)  # clamp to the horizon (safety)
    blackout = {int(d) for d in (blackout_days or [])}
    buffer_cutoff = m_end - max(0, int(end_buffer_days))  # days >= cutoff (this month) are reserved
    excluded = set(blackout) | {d for d in range(buffer_cutoff, m_end)}

    # --- Parse history into per-day activity. ---
    # stationary: any non-wait action (take_order / reposition) marks the day.
    # no_accepted_order: only an accepted take_order marks the day.
    activity_days: set[int] = set()  # completed days with activity (cannot be rest)
    today_clean = today not in blackout
    sim_cursor = 0
    for record in history_records or []:
        step_elapsed = max(0, int(record.get("step_elapsed_minutes", 0) or 0))
        query_scan = max(0, int(record.get("query_scan_cost_minutes", 0) or 0))
        action_exec = max(0, int(record.get("action_exec_cost_minutes", 0) or 0))
        step_start = sim_cursor
        action_start = step_start + query_scan
        action_end = action_start + action_exec
        sim_cursor = step_start + step_elapsed

        action = record.get("action") if isinstance(record.get("action"), dict) else {}
        action_name = str(action.get("action", "")).strip().lower()
        if action_name == "wait":
            continue
        breaks_day = False
        if stationary:
            breaks_day = action_name in {"take_order", "reposition"}
        else:
            if action_name == "take_order":
                result = record.get("result") if isinstance(record.get("result"), dict) else {}
                breaks_day = bool(result.get("accepted", False))
        if not breaks_day:
            continue
        start_day = action_start // MIN_PER_DAY
        end_day = max(action_start, action_end - 1) // MIN_PER_DAY
        for day in range(start_day, end_day + 1):
            if day < today:
                activity_days.add(day)
            elif day == today:
                today_clean = False
            else:
                # spill from an already-accepted order contaminates a future day
                activity_days.add(day)

    # Completed = any fully-elapsed zero-activity day (scoring credits these with
    # no spacing/blackout caveat — a rest taken on a buffer/commitment day still
    # counts, so do not subtract those here).
    # Completed rest days count ONLY within the current calendar month (the quota is per-month).
    completed_days = {d for d in range(m_start, today) if d not in activity_days}
    dirty_days = {d for d in activity_days if today <= d < m_end}

    remaining_needed = max(0, target - len(completed_days))

    # --- Candidate (available) future days. ---
    # IMPORTANT: the scoring rule (calc_monthly_income._eval_off_days) counts
    # ANY zero-activity day toward the quota with NO spacing requirement — two
    # consecutive full-rest days each count. So the HARD feasibility gate must
    # be purely count-based; enforcing spacing here could self-inflict a false
    # "infeasible" and miss the count. A day is available if it is in
    # [today, month_days), not blackout, not dirtied by committed activity/spill,
    # and (for today) still clean.
    available: list[int] = []
    today_available = False
    for d in range(max(today, m_start), m_end):
        if d in excluded or d in dirty_days:
            continue
        if d == today and not today_clean:
            continue
        available.append(d)
        if d == today:
            today_available = True

    # --- Feasibility by construction (count-only). ---
    # Max placeable = number of available days (consecutive allowed). Forced /
    # protected fall out of "would removing this day drop the count below what we
    # still need?".
    feasible = len(available) >= remaining_needed
    n_skip_today = len([d for d in available if d != today])
    forced_today = (
        remaining_needed > 0 and today_available and n_skip_today < remaining_needed
    )

    # Spacing ("别连休 / 陪二老") is a SOFT preference only: the hard availability gate
    # stays purely count-based (per the module contract — enforcing spacing here can
    # self-inflict a false "infeasible" and miss the count). Adjacency avoidance lives
    # entirely in the recommended_days selection below, which drops days adjacent to a
    # completed rest day and prefers non-adjacent picks, degrading gracefully when the
    # remaining target cannot otherwise fit. (A previous first-half hard gate that set
    # today_available=False after a completed rest day violated this contract and is
    # removed; the soft layer + region/geo penalties handle the "don't rest consecutively
    # in a bad spot" intent.)

    # Protected days: any future available day whose loss makes the count
    # infeasible. Removing one day drops the count by one, so once slack is gone
    # (len(available) <= remaining_needed) every remaining future day is needed.
    protected_days: set[int] = set()
    if remaining_needed > 0 and len(available) <= remaining_needed:
        protected_days = {d for d in available if d != today}

    # --- Soft plan: which days to recommend resting on. ---
    # Spacing lives ONLY here (the "陪二老 / 昨天刚休今天别连休" intent): the soft
    # candidate set drops days adjacent to an already-completed rest day, and the
    # selection prefers non-adjacent days. If spacing cannot fit the remaining
    # target it degrades gracefully — the hard gate above still guarantees count.
    soft_candidates = [
        d for d in available
        if (d - 1) not in completed_days and (d + 1) not in completed_days
    ]
    recommended_days = _select_days(soft_candidates, remaining_needed, day_cost_fn)
    recommended_today = today in recommended_days

    # PROACTIVE commitment ("提前规划，不是当天再规划"). The calendar-aware ``recommended_days`` (spaced
    # out, low-cost, 陪二老不连休) are PROMOTED to protected + forced from the START — not only once
    # month-end slack runs out. The slack-only gate fired the force so late that a NO-CURFEW driver
    # (D001, works around the clock) had already dirtied every morning by then (today_clean=False →
    # forced_today never fired → 0 off-days); a CURFEW driver (D002) kept its 00-08 static so the late
    # force still landed — the exact asymmetry observed. Committing the calendar days up front protects
    # each from spill (it stays full-day clean) and forces it the moment it arrives, which is what makes
    # 歇满 reliable regardless of a rest-window anchor. The original slack-based force/protect stays as
    # the backstop (the `or`/union), so a missed/dirtied recommended day still gets caught at month end.
    protected_days = protected_days | {d for d in recommended_days if d != today}
    forced_today = forced_today or (recommended_today and today_available)

    latest_feasible_day = max(available) if available else None

    return MonthlyQuotaPlan(
        pref_index=pref_index,
        target_days=target,
        stationary=stationary,
        month_days=month_days,
        today=today,
        completed_days=completed_days,
        dirty_days=dirty_days,
        blackout_days=blackout,
        available_days=available,
        remaining_needed=remaining_needed,
        today_available=today_available,
        forced_today=forced_today,
        recommended_days=recommended_days,
        recommended_today=recommended_today,
        protected_days=protected_days,
        latest_feasible_day=latest_feasible_day,
        feasible=feasible,
    )
