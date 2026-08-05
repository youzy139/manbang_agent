"""Deterministic time / date helpers for the LLM-led agent experiment.

The two LLMs (preference brain + decision model) never do date or clock
arithmetic themselves — they either receive ``now()``'s output injected into
their context, or call these helpers through the decision tool layer. Keeping
all time math here is the single biggest correctness lever for the experiment:
LLMs are unreliable at "how many days until month end" / "minutes until 23:00".

Simulation convention (mirrors calc_monthly_income / cargo_graph):
    epoch  = 2026-03-01 00:00:00  (MUST be the first day of the first month)
    sim_min = whole minutes since epoch
    horizon = 92 days = March(31) + April(30) + May(31)
The run now spans THREE calendar months (2026-03 .. 2026-05). The absolute
``sim_min // 1440`` day index keeps climbing 0..91 and never resets, so a "day"
in this module always means a global 0-based simulation-day index, NOT a
day-of-month. ``now()['day_of_month']`` is the genuine calendar day-of-month
(1..31, resetting each month) and no longer equals the sim-day index past March
— use :func:`month_window_of_day` / :func:`calendar_month_of_day` whenever the
calendar month matters (the per-calendar-month "每月X" quota semantics).
"""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from typing import Union

SIM_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
MINUTES_PER_DAY = 1440
# 3-month simulation span: March(31) + April(30) + May(31) = 92 natural days.
# Single source of truth for the run length — every horizon / active-day range /
# month_days default derives from this so there is exactly ONE place to change it.
DURATION_DAYS = 92

Ymd = Union[str, tuple, list, datetime]


# --------------------------------------------------------------------------- #
# calendar-month decomposition (per-month "每月X" quota semantics)
# --------------------------------------------------------------------------- #
def calendar_months(duration_days: int | None = None) -> list[tuple[int, int, int, int, int]]:
    """Tile the simulation span ``[0, duration_days)`` by calendar month.

    Returns a list of ``(year, month, start_day, end_day_exclusive, days_in_month)``
    in global 0-based sim-day indices, anchored at :data:`SIM_EPOCH`. With the
    default 92-day span starting 2026-03-01 this is::

        [(2026, 3, 0, 31, 31), (2026, 4, 31, 61, 30), (2026, 5, 61, 92, 31)]

    ``SIM_EPOCH`` is the first of its month, so calendar boundaries align exactly
    on these day indices. ``end_day_exclusive`` is clamped to the span end (a
    partial trailing month keeps its true ``days_in_month`` but a shorter window).
    """
    total = int(DURATION_DAYS if duration_days is None else duration_days)
    out: list[tuple[int, int, int, int, int]] = []
    start = 0
    year, month = SIM_EPOCH.year, SIM_EPOCH.month
    while start < total:
        dim = calendar.monthrange(year, month)[1]
        end = start + dim  # full calendar-month length from its 1st
        out.append((year, month, start, min(end, total), dim))
        start = end
        month += 1
        if month > 12:
            month = 1
            year += 1
    return out


def month_window_of_day(sim_day: int, duration_days: int | None = None) -> tuple[int, int, int, int, int]:
    """``(year, month, month_start_day, month_end_day_exclusive, days_in_month)`` for the
    calendar month containing global sim-day index ``sim_day``. A day past the span end
    maps to the last month (conservative)."""
    months = calendar_months(duration_days)
    d = int(sim_day)
    for entry in months:
        _, _, start, end, _ = entry
        if start <= d < end:
            return entry
    return months[-1]


def month_window_of_min(sim_min: int, duration_days: int | None = None) -> tuple[int, int, int, int, int]:
    """Same as :func:`month_window_of_day` keyed on an absolute sim-minute."""
    return month_window_of_day(int(sim_min) // MINUTES_PER_DAY, duration_days)


def calendar_month_of_day(sim_day: int, duration_days: int | None = None) -> tuple[int, int]:
    """``(year, month)`` bucket key for a global sim-day index — the grouping key for
    per-calendar-month "每月X" aggregation."""
    y, m, *_ = month_window_of_day(sim_day, duration_days)
    return (y, m)


# --------------------------------------------------------------------------- #
# core conversions
# --------------------------------------------------------------------------- #
def sim_min_to_dt(sim_min: int) -> datetime:
    return SIM_EPOCH + timedelta(minutes=int(sim_min))


def dt_to_sim_min(dt: datetime) -> int:
    return int((dt - SIM_EPOCH).total_seconds() // 60)


def _parse_ymd(value: Ymd) -> datetime:
    """Accept 'YYYY-MM-DD', '2026/3/31', (y,m,d), or a datetime → midnight dt."""
    if isinstance(value, datetime):
        return value.replace(hour=0, minute=0, second=0, microsecond=0)
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return datetime(int(value[0]), int(value[1]), int(value[2]))
    text = str(value).strip().replace("/", "-")
    return datetime.strptime(text, "%Y-%m-%d")


def _parse_hhmm(hhmm: str) -> tuple[int, int]:
    # Tolerate "HH", "HH:MM", "HH:MM:SS" — models write all three; seconds ignored.
    parts = str(hhmm).strip().split(":")
    hh = int(parts[0]) if parts[0] else 0
    mm = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    return hh, mm


# --------------------------------------------------------------------------- #
# helpers exposed to the agent / injected into LLM context
# --------------------------------------------------------------------------- #
_WEEKDAY_CN = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")  # isoweekday 1..7


def now(sim_min: int) -> dict:
    """Snapshot the current sim time as plain fields the LLMs can read."""
    dt = sim_min_to_dt(sim_min)
    iso = dt.isoweekday()  # 1=Mon ... 7=Sun (real Gregorian calendar, anchored at SIM_EPOCH)
    return {
        "sim_min": int(sim_min),
        "date": dt.strftime("%Y-%m-%d"),
        "date_cn": dt.strftime("%Y-%m-%d") + " " + _WEEKDAY_CN[iso - 1],  # "2026-03-12 周四"
        "day_of_month": dt.day,          # calendar day-of-month (1..31, resets each month)
        "weekday": iso,                  # 1=Mon ... 7=Sun
        "weekday_cn": _WEEKDAY_CN[iso - 1],  # 周一..周日 (readable)
        "is_weekend": iso >= 6,          # 周六/周日 — drives weekend-conditional rules (D001 周末顺延)
        "hhmm": dt.strftime("%H:%M"),
    }


def day_of_month(sim_min: int) -> int:
    return sim_min_to_dt(sim_min).day


def interval_minutes(start_ymd: Ymd, end_ymd: Ymd) -> int:
    """Minutes between two calendar dates (the helper the model calls before rest).

    Both args are interpreted at 00:00 unless a datetime carrying a time is
    passed. Result can be negative if end precedes start.
    """
    return int((_parse_ymd(end_ymd) - _parse_ymd(start_ymd)).total_seconds() // 60)


def sim_min_of(ymd: Ymd, hhmm: str = "00:00") -> int:
    """Absolute sim_min for a specific date + wall time.

    Uses timedelta from midnight so '24:00' (and any hh>=24) rolls into the next
    day instead of crashing — models routinely write '24:00' for end-of-day.
    """
    base = _parse_ymd(ymd)  # midnight of that date
    hh, mm = _parse_hhmm(hhmm)
    return dt_to_sim_min(base + timedelta(hours=hh, minutes=mm))


def walltime_to_sim_min(current_sim_min: int, hhmm: str, *, on_day: int | None = None) -> int:
    """Resolve a wall-clock 'HH:MM' to an absolute sim_min.

    ``on_day`` (a calendar day-of-month) pins it to that day **within the calendar
    month of ``current_sim_min``** — so across the 3-month run it resolves in the
    month the driver is currently in (in March this is unchanged from the old
    March-only behaviour). Without it, resolve to the next occurrence: today if the
    time is still ahead of now, otherwise tomorrow. This is how a directive like
    "rest by 23:00" becomes a concrete deadline relative to the current step.
    """
    hh, mm = _parse_hhmm(hhmm)
    if on_day is not None:
        cur_dt = sim_min_to_dt(current_sim_min)
        dim = calendar.monthrange(cur_dt.year, cur_dt.month)[1]
        day = min(max(1, int(on_day)), dim)  # clamp into the current month's length
        return dt_to_sim_min(datetime(cur_dt.year, cur_dt.month, day) + timedelta(hours=hh, minutes=mm))
    cur = sim_min_to_dt(current_sim_min)
    midnight = cur.replace(hour=0, minute=0, second=0, microsecond=0)
    target = midnight + timedelta(hours=hh, minutes=mm)  # timedelta → '24:00' = next 00:00
    if target < cur:
        target += timedelta(days=1)
    return dt_to_sim_min(target)


def minutes_until(current_sim_min: int, hhmm: str, *, on_day: int | None = None) -> int:
    """Minutes from now until the resolved wall time (see walltime_to_sim_min)."""
    return walltime_to_sim_min(current_sim_min, hhmm, on_day=on_day) - int(current_sim_min)
