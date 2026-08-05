"""Multi-hop cargo path planner."""

from __future__ import annotations

import heapq
import itertools
from typing import Any, Callable, Iterable

from .cargo_graph import CargoGraph, cargo_haul_km, wall_time_to_sim_min

Path = dict[str, Any]
RegionValueFn = Callable[[float, float], float]
RegionCoverageFn = Callable[[float, float], int]

DEFAULT_REGION_VALUE_SCALE = 300.0
DEFAULT_FINAL_WEIGHT = 0.5
DEFAULT_INTERMEDIATE_WEIGHT = 0.1


def plan_paths(
    graph: CargoGraph,
    start_pos: tuple[float, float],
    start_time_min: int,
    cost_per_km: float,
    speed_km_h: float,
    horizon_min: int,
    max_depth: int = 3,
    top_k: int = 5,
    extra_depth_top_k: dict[int, int] | None = None,
    preserve_extra_depths: bool = False,
    enable_pruning: bool = True,
    excluded_cargo_categories: Iterable[str] | None = None,
    required_cargo_categories: Iterable[str] | None = None,
    allowed_destination_regions: Iterable[dict[str, Any]] | None = None,
    forbidden_destination_regions: Iterable[dict[str, Any]] | None = None,
    cargo_load_after_min: int | None = None,
    cargo_load_before_min: int | None = None,
    candidate_pool_k: int | None = None,
    region_value_fn: RegionValueFn | None = None,
    region_coverage_fn: RegionCoverageFn | None = None,
    region_value_scale: float = DEFAULT_REGION_VALUE_SCALE,
    region_value_weight: float = 0.0,
    final_weight: float = DEFAULT_FINAL_WEIGHT,
    intermediate_weight: float = DEFAULT_INTERMEDIATE_WEIGHT,
    scan_margin_min: int = 0,
    cargo_yield_delta_fn: "Callable[[dict[str, Any], float, float, int], float] | None" = None,
    max_cargo_yield_bonus: float = 0.0,
    beam_width: int | None = None,
    beam_trigger: int = 200,
) -> list[Path]:
    """Search the graph for top-K paths ranked by score per minute.

    DFS records candidate paths by an admissible score estimate, then the final
    pool is ranked by ``total_score / total_time_min`` so short high-efficiency
    routes can beat longer routes with larger absolute yield. When
    ``region_value_fn`` and ``region_value_weight > 0`` are provided,
    ``total_score`` =
    total_net_yield + weight * scale *
    (final_w * dst_norm + inter_w * sum_intermediate_norm).
    Each returned path always carries ``destination_region_value_norm``,
    ``intermediate_region_values_norm`` and ``total_score`` so callers can
    audit the trade-off.
    """
    if top_k <= 0 or max_depth <= 0:
        return []

    pool_k = candidate_pool_k if candidate_pool_k and candidate_pool_k > 0 else max(top_k * 5, 50)
    pool_k = max(pool_k, top_k)
    pool_extra_depth = (
        {depth: max(limit, max(limit * 5, 30)) for depth, limit in (extra_depth_top_k or {}).items()}
        if extra_depth_top_k
        else None
    )
    excluded_categories = _normalize_category_filter(excluded_cargo_categories)
    required_categories = _normalize_category_filter(required_cargo_categories)
    allowed_regions = _normalize_geo_regions(allowed_destination_regions)
    forbidden_regions = _normalize_geo_regions(forbidden_destination_regions)
    extra_depth_limits = _normalize_depth_limits(pool_extra_depth, max_depth)
    max_hop_net_yield = _compute_max_hop_net_yield_upper_bound(graph, cost_per_km) if enable_pruning else float("inf")
    # 当 cargo_modifier 在搜索期抬升某些货的 net_yield（见 feasible_edges_from 的 cargo_yield_delta_fn），
    # 单跳实际收益可超过 base-price 上界 → 原可采纳剪枝会**误剪**被升值的目标货支路。把所有正向 delta 之和
    # （任一跳最多获得的奖励上界）加进乐观上界，保证 UB 永不低估 → 剪枝仍可采纳。max_cargo_yield_bonus=0 → 不变。
    _bonus = float(max_cargo_yield_bonus or 0.0)
    # Admissible lower bound on a single hop's total_time_min. Used to make the
    # per-min UB denominator (current_time + future_time) tighter — any future
    # hop must take at least min_hop_total_time minutes, so dividing by just
    # current_time over-states the achievable per-min ratio.
    min_hop_total_time = _compute_min_hop_total_time(graph) if enable_pruning else 0
    # Positive cargo yields sorted descending. We tighten the numerator UB from
    # ``remaining_hops × global_max_yield`` (always picks the same top cargo
    # however many times) to ``sum of top-K remaining yields`` (each cargo can
    # only be used once, so used_ids progressively shrinks the available top).
    sorted_positive_yields = _compute_sorted_positive_yields(graph, cost_per_km) if enable_pruning else []
    if enable_pruning and _bonus > 0.0:
        max_hop_net_yield += _bonus
        sorted_positive_yields = [(cid, y + _bonus) for cid, y in sorted_positive_yields]
    top_paths: list[tuple[float, int, Path]] = []
    depth_paths: dict[int, list[tuple[float, int, Path]]] = {depth: [] for depth in extra_depth_limits}
    tie_breaker = itertools.count()
    region_active = region_value_fn is not None and region_value_weight > 0

    # Memoise region_value_fn results by exact (lat, lng). Inside one DFS run
    # the same cargo end_pos is visited many times across recorded paths, so a
    # plain identity-keyed cache hits 90%+. Values are bit-for-bit identical
    # to the un-cached path.
    region_value_cache: dict[tuple[float, float], float] = {}

    def _region_at(pos: tuple[float, float]) -> float:
        try:
            key = (float(pos[0]), float(pos[1]))
        except (TypeError, ValueError):
            return 0.0
        cached = region_value_cache.get(key)
        if cached is not None:
            return cached
        try:
            v = max(0.0, min(1.0, float(region_value_fn(key[0], key[1]))))
        except (TypeError, ValueError):
            v = 0.0
        region_value_cache[key] = v
        return v

    # Tight admissible bound for region_value at any reachable end_pos. A hop's
    # end_pos is either a cargo end_pos (real or static virtual cargo) or — for
    # dynamic-position virtual cargos (rest_v / rest_window_v / monthly_rest_v)
    # — the driver's current DFS position, which is itself the previous hop's
    # end_pos or the initial start_pos. So scanning ``start_pos ∪ all cargo
    # end_pos`` covers every reachable point. The resulting max_rv is at most
    # 1.0 (the previous loose bound) but typically much smaller, tightening the
    # region_bonus UB by that factor.
    max_rv_in_graph = 0.0
    if region_active:
        max_rv_in_graph = _region_at(start_pos)
        for cargo in graph.iter_cargos():
            end = cargo.get("end") or {}
            try:
                rv = _region_at((float(end["lat"]), float(end["lng"])))
            except (KeyError, TypeError, ValueError):
                continue
            if rv > max_rv_in_graph:
                max_rv_in_graph = rv
        max_rv_in_graph = min(1.0, max(0.0, max_rv_in_graph))

    # Per-termination-depth region_bonus upper bound. Used by _can_prune to
    # evaluate the admissible max per_min across all termination depths
    # k+depth. ``depth_max_region_bonus`` mirrors the per-depth pool's exact
    # target depth.
    depth_max_region_bonus: dict[int, float] = {}
    region_bonus_at_depth: dict[int, float] = {}
    if region_active:
        for target_depth in extra_depth_limits:
            depth_max_region_bonus[target_depth] = region_value_weight * region_value_scale * (
                final_weight + max(0, target_depth - 1) * intermediate_weight
            ) * max_rv_in_graph
        for d in range(1, max_depth + 1):
            region_bonus_at_depth[d] = region_value_weight * region_value_scale * (
                final_weight + max(0, d - 1) * intermediate_weight
            ) * max_rv_in_graph

    def _score_estimate(hops: list[dict[str, Any]]) -> float:
        net = sum(float(h["net_yield"]) for h in hops)
        if not region_active or not hops:
            return net
        final_val = _region_at(hops[-1]["end_pos"])
        inter_sum = 0.0
        for h in hops[:-1]:
            inter_sum += _region_at(h["end_pos"])
        bonus = region_value_weight * region_value_scale * (
            final_weight * final_val + intermediate_weight * inter_sum
        )
        return net + bonus

    def _score_per_min_estimate(hops: list[dict[str, Any]]) -> float:
        total_time = max(1, sum(int(h["total_time_min"]) for h in hops))
        return _score_estimate(hops) / total_time

    def dfs(
        pos: tuple[float, float],
        time_min: int,
        path_so_far: list[dict[str, Any]],
        used_ids: set[str],
    ) -> None:
        depth = len(path_so_far)
        if depth >= 1 and _path_satisfies_required_categories(path_so_far, required_categories):
            score = _score_per_min_estimate(path_so_far)
            _record(top_paths, tie_breaker, path_so_far, pool_k, score)
            if depth in depth_paths:
                _record(depth_paths[depth], tie_breaker, path_so_far, extra_depth_limits[depth], score)

        if depth >= max_depth:
            return

        if enable_pruning and max_hop_net_yield >= 0:
            if _can_prune(
                top_paths,
                depth_paths,
                extra_depth_limits,
                path_so_far,
                depth,
                max_depth,
                pool_k,
                depth_max_region_bonus,
                min_hop_total_time,
                sorted_positive_yields,
                used_ids,
                region_bonus_at_depth,
            ):
                return

        feasible = graph.feasible_edges_from(pos, time_min, speed_km_h, cost_per_km, horizon_min, scan_margin_min,
                                             cargo_yield_delta_fn=cargo_yield_delta_fn)
        if region_active:
            # Heuristic edge order: include the would-be region_value contribution
            # converted into per_min units so high-final-score hops are explored
            # first, raising heap[0][0] quickly and enabling more pruning. Uses
            # ``final_weight`` (the larger of final/intermediate) so promising
            # terminal hops aren't deprioritised. Doesn't change the search tree
            # — only the order of exploration, so admissibility is preserved.
            def _edge_sort_key(hop: dict[str, Any]) -> float:
                try:
                    end_pos = hop["end_pos"]
                    rv = _region_at(end_pos)
                except (KeyError, TypeError, ValueError):
                    rv = 0.0
                total_time = max(1, int(hop["total_time_min"]))
                region_per_min = (
                    region_value_weight * region_value_scale * final_weight * rv
                ) / total_time
                return float(hop["yield_per_min"]) + region_per_min
            feasible.sort(key=_edge_sort_key, reverse=True)
        else:
            feasible.sort(key=lambda hop: float(hop["yield_per_min"]), reverse=True)

        # 自适应 beam：可达货 > beam_trigger 时(白天货密、DFS 指数爆炸)，每节点只展开 top-B 个**有效** hop;
        # ≤ trigger(深夜/货稀)不限、保持原 admissible 全枚举。feasible 已按 yield_per_min(+region)降序，top-B
        # 是最有希望的支路。beam_width=None → 永不启用(行为逐字节不变)。
        beam_limit = beam_width if (beam_width and len(feasible) > beam_trigger) else None
        expanded = 0
        for hop in feasible:
            cargo_id = str(hop["cargo_id"])
            if cargo_id in used_ids:
                continue
            if _hop_category(hop) in excluded_categories:
                continue
            if not _destination_satisfies_regions(hop["end_pos"], allowed_regions, forbidden_regions):
                continue
            if not _hop_within_time_window(hop, cargo_load_after_min, cargo_load_before_min):
                continue
            dfs(
                pos=hop["end_pos"],
                time_min=int(hop["finish_min"]),
                path_so_far=path_so_far + [hop],
                used_ids=used_ids | {cargo_id},
            )
            expanded += 1
            if beam_limit is not None and expanded >= beam_limit:
                break

    dfs(start_pos, int(start_time_min), [], set())
    pool = _merge_ranked_paths(top_paths, depth_paths)
    _annotate_region_value(pool, region_value_fn, region_coverage_fn)
    _annotate_total_score(
        pool,
        region_value_weight=region_value_weight,
        region_value_scale=region_value_scale,
        final_weight=final_weight,
        intermediate_weight=intermediate_weight,
    )
    def pool_rank_key(path: Path) -> tuple[float, float, float, float]:
        total_time = max(1, int(path.get("total_time_min", 0) or 0))
        total_score = float(path.get("total_score", path.get("total_net_yield", 0.0)) or 0.0)
        total_net_yield = float(path.get("total_net_yield", 0.0) or 0.0)
        yield_per_min = float(path.get("yield_per_min", 0.0) or 0.0)
        return (total_score / total_time, total_score, yield_per_min, total_net_yield)

    pool.sort(key=pool_rank_key, reverse=True)
    if preserve_extra_depths and extra_depth_limits:
        kept = list(pool[:top_k])
        seen = {tuple(str(hop.get("cargo_id")) for hop in path.get("hops", [])) for path in kept}
        for path in pool:
            if int(path.get("depth", 0) or 0) not in extra_depth_limits:
                continue
            key = tuple(str(hop.get("cargo_id")) for hop in path.get("hops", []))
            if key in seen:
                continue
            kept.append(path)
            seen.add(key)
        return kept
    return pool[:top_k]


def _annotate_region_value(
    paths: list[Path],
    region_value_fn: RegionValueFn | None,
    region_coverage_fn: RegionCoverageFn | None = None,
) -> None:
    for path in paths:
        hops = path.get("hops") or []
        final_val = 0.0
        final_coverage = 0
        intermediate: list[float] = []
        if region_value_fn is not None and hops:
            try:
                fp = path.get("finish_pos") or hops[-1]["end_pos"]
                final_val = max(0.0, min(1.0, float(region_value_fn(float(fp[0]), float(fp[1])))))
                if region_coverage_fn is not None:
                    final_coverage = int(region_coverage_fn(float(fp[0]), float(fp[1])))
            except (TypeError, ValueError):
                final_val = 0.0
                final_coverage = 0
            for hop in hops[:-1]:
                try:
                    pos = hop.get("end_pos")
                    val = max(0.0, min(1.0, float(region_value_fn(float(pos[0]), float(pos[1])))))
                except (TypeError, ValueError):
                    val = 0.0
                intermediate.append(val)
        path["destination_region_value_norm"] = round(final_val, 4)
        path["intermediate_region_values_norm"] = [round(v, 4) for v in intermediate]
        path["destination_coverage_cells"] = final_coverage
        path["destination_known"] = final_coverage > 0


def region_bonus_for_path(
    path: Path,
    region_value_weight: float,
    *,
    region_value_scale: float = DEFAULT_REGION_VALUE_SCALE,
    final_weight: float = DEFAULT_FINAL_WEIGHT,
    intermediate_weight: float = DEFAULT_INTERMEDIATE_WEIGHT,
) -> float:
    """Region-value bonus (in yield units) for an already-annotated path.

    Reuses the ``destination_region_value_norm`` / ``intermediate_region_values_norm``
    fields written by ``_annotate_region_value``. This is the single source of
    truth for the formula: the planner's own ``total_score`` (below) and any
    caller that ranks routes on top of a *preference-adjusted* yield (see
    ``loop._plan``) must add the SAME bonus, so region value stays consistent
    across the candidate-pool, deadline-filter, and final-decision rankings.
    Returns 0.0 when ``region_value_weight <= 0`` (feature off).
    """
    if region_value_weight <= 0:
        return 0.0
    final_val = float(path.get("destination_region_value_norm", 0.0) or 0.0)
    intermediate = path.get("intermediate_region_values_norm") or []
    weighted_sum = final_weight * final_val + intermediate_weight * sum(float(v) for v in intermediate)
    return region_value_weight * region_value_scale * weighted_sum


def _annotate_total_score(
    paths: list[Path],
    *,
    region_value_weight: float,
    region_value_scale: float,
    final_weight: float,
    intermediate_weight: float,
) -> None:
    for path in paths:
        base = float(path.get("total_net_yield", 0.0))
        bonus = region_bonus_for_path(
            path,
            region_value_weight,
            region_value_scale=region_value_scale,
            final_weight=final_weight,
            intermediate_weight=intermediate_weight,
        )
        path["total_score"] = round(base + bonus, 2)


def _record(
    top_paths_heap: list[tuple[float, int, Path]],
    tie_breaker: itertools.count,
    hops: list[dict[str, Any]],
    top_k: int,
    key_value: float,
) -> None:
    path = _build_path(hops)
    item = (float(key_value), next(tie_breaker), path)
    if len(top_paths_heap) < top_k:
        heapq.heappush(top_paths_heap, item)
    elif item[0] > top_paths_heap[0][0]:
        heapq.heapreplace(top_paths_heap, item)


def _can_prune(
    top_paths: list[tuple[float, int, Path]],
    depth_paths: dict[int, list[tuple[float, int, Path]]],
    extra_depth_limits: dict[int, int],
    path_so_far: list[dict[str, Any]],
    depth: int,
    max_depth: int,
    top_k: int,
    depth_max_region_bonus: dict[int, float],
    min_hop_total_time: int,
    sorted_positive_yields: list[tuple[str, float]],
    used_ids: set[str],
    region_bonus_at_depth: dict[int, float],
) -> bool:
    """Admissible pruning based on the maximum achievable per_min across any
    termination depth k in ``[depth, max_depth]``. For each k we use the actual
    top-k of currently-unused cargo yields (tighter than ``k × global_max``),
    a true lower bound on remaining hop time, and the region-value bonus that
    a path ending at exactly that depth would receive.
    """
    current_yield = sum(float(h["net_yield"]) for h in path_so_far)
    current_time = max(1, sum(int(h["total_time_min"]) for h in path_so_far))
    overall_full = len(top_paths) >= top_k
    if not overall_full:
        return False  # heap not yet full → no admissible bar to clear → can't prune
    overall_threshold = top_paths[0][0]
    overall_remaining_hops = max(0, max_depth - depth)
    min_hop_time = max(0, int(min_hop_total_time))
    top_yields = _remaining_top_k_yields(sorted_positive_yields, used_ids, overall_remaining_hops)

    # Best per_min reachable across termination depths k in [0, remaining_hops].
    # Short-circuit as soon as any k's UB clears the heap-min threshold — once
    # we know we won't prune, no need to keep computing tighter UB values.
    # k = 0 case: terminate here. Only meaningful when depth >= 1 (path already
    # has at least one hop and is itself a candidate).
    if depth >= 1 and (current_yield / current_time) >= overall_threshold:
        return False

    running_sum = 0.0
    for k in range(1, overall_remaining_hops + 1):
        if k <= len(top_yields):
            running_sum += top_yields[k - 1]
        bonus_k = region_bonus_at_depth.get(depth + k, 0.0)
        denom_k = max(1, current_time + k * min_hop_time)
        pm_k = (current_yield + running_sum + bonus_k) / denom_k
        if pm_k >= overall_threshold:
            return False  # UB already clears threshold; don't prune
    # If we got here, every k-UB was strictly below threshold → confirmed prune
    # at the overall pool. Still need to check per-depth pools below.

    # Per-target-depth pool: same construction but pinned to a single k.
    for target_depth, limit in extra_depth_limits.items():
        if target_depth <= depth:
            continue
        heap = depth_paths.get(target_depth, [])
        k_target = target_depth - depth
        yields_for_target = sum(top_yields[:k_target])
        target_bonus = depth_max_region_bonus.get(target_depth, 0.0)
        target_denom = max(1, current_time + k_target * min_hop_time)
        target_upper_per_min = (current_yield + yields_for_target + target_bonus) / target_denom
        if len(heap) < limit or target_upper_per_min >= heap[0][0]:
            return False
    return True


def _merge_ranked_paths(
    top_paths: list[tuple[float, int, Path]],
    depth_paths: dict[int, list[tuple[float, int, Path]]],
) -> list[Path]:
    merged: list[Path] = []
    seen: set[tuple[str, ...]] = set()

    def add_entries(entries: list[tuple[float, int, Path]]) -> None:
        for _score, _, path in sorted(
            entries,
            key=lambda item: (item[0], item[2]["yield_per_min"], item[2]["depth"]),
            reverse=True,
        ):
            key = tuple(str(hop.get("cargo_id")) for hop in path.get("hops", []))
            if key in seen:
                continue
            seen.add(key)
            merged.append(path)

    add_entries(top_paths)
    for depth in sorted(depth_paths.keys(), reverse=True):
        add_entries(depth_paths[depth])
    return merged


def _build_path(hops: list[dict[str, Any]]) -> Path:
    total_net_yield = sum(float(hop["net_yield"]) for hop in hops)
    total_time_min = sum(int(hop["total_time_min"]) for hop in hops)
    return {
        "hops": hops,
        "total_net_yield": total_net_yield,
        "total_time_min": total_time_min,
        "total_load_wait_min": sum(int(hop["load_wait_min"]) for hop in hops),
        "yield_per_min": total_net_yield / total_time_min if total_time_min > 0 else 0.0,
        "finish_min": int(hops[-1]["finish_min"]),
        "finish_pos": hops[-1]["end_pos"],
        "depth": len(hops),
    }


def _compute_max_hop_net_yield_upper_bound(graph: CargoGraph, cost_per_km: float) -> float:
    """Compute a conservative per-hop net-yield upper bound for pruning.

    Actual net yield also pays pickup deadhead cost, so ``price - haul_cost`` is
    an optimistic bound for any future hop using that cargo.
    """
    best = 0.0
    for cargo in graph.iter_cargos():
        price = float(cargo.get("price", 0.0))
        haul_cost = float(cost_per_km) * cargo_haul_km(cargo)
        best = max(best, price - haul_cost)
    return best


def _compute_min_hop_total_time(graph: CargoGraph) -> int:
    """Admissible lower bound on a single hop's total_time_min.

    A hop's total_time_min = deadhead_min + load_wait_min + cost_time_min.
    Deadhead and load_wait can be 0 at the optimistic limit, so the floor is
    the minimum cost_time_minutes across all cargos in the graph. Used to keep
    the per-min UB admissible without over-estimating future per-min ratios.
    """
    best: int | None = None
    for cargo in graph.iter_cargos():
        try:
            cost_time = int(cargo.get("cost_time_minutes", 0) or 0)
        except (TypeError, ValueError):
            continue
        if cost_time <= 0:
            continue
        if best is None or cost_time < best:
            best = cost_time
    return best or 0


def _compute_sorted_positive_yields(
    graph: CargoGraph, cost_per_km: float
) -> list[tuple[str, float]]:
    """Cargo (id, optimistic per-hop yield) sorted descending, dropping non-positives.

    A future hop using a cargo with non-positive ``price - haul_cost`` cannot
    raise an extension's net yield, so excluding them tightens the
    sum-of-top-K bound without losing admissibility. Cargos already used in
    the current DFS path are skipped at query time via ``used_ids``.
    """
    yields: list[tuple[str, float]] = []
    for cargo in graph.iter_cargos():
        cargo_id = str(cargo.get("cargo_id", ""))
        if not cargo_id:
            continue
        price = float(cargo.get("price", 0.0))
        haul_cost = float(cost_per_km) * cargo_haul_km(cargo)
        y = price - haul_cost
        if y > 0:
            yields.append((cargo_id, y))
    yields.sort(key=lambda kv: kv[1], reverse=True)
    return yields


def _remaining_top_k_yields(
    sorted_positive_yields: list[tuple[str, float]],
    used_ids: set[str],
    max_k: int,
) -> list[float]:
    """Top-K positive cargo yields among those not yet used in this DFS branch."""
    if max_k <= 0:
        return []
    out: list[float] = []
    for cargo_id, y in sorted_positive_yields:
        if cargo_id in used_ids:
            continue
        out.append(y)
        if len(out) >= max_k:
            break
    return out


def _normalize_category_filter(values: Iterable[str] | None) -> set[str]:
    if values is None:
        return set()
    result: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text:
            result.add(text)
    return result


def _normalize_depth_limits(value: dict[int, int] | None, max_depth: int) -> dict[int, int]:
    if not isinstance(value, dict):
        return {}
    limits: dict[int, int] = {}
    for raw_depth, raw_limit in value.items():
        try:
            depth = int(raw_depth)
            limit = int(raw_limit)
        except (TypeError, ValueError):
            continue
        if 1 <= depth <= max_depth and limit > 0:
            limits[depth] = limit
    return limits


def _path_satisfies_required_categories(hops: list[dict[str, Any]], required_categories: set[str]) -> bool:
    if not required_categories:
        return True
    return any(_hop_category(hop) in required_categories for hop in hops)


def _hop_category(hop: dict[str, Any]) -> str:
    cargo = hop.get("cargo") if isinstance(hop.get("cargo"), dict) else {}
    return str(cargo.get("cargo_name", "")).strip()


def _normalize_geo_regions(values: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if values is None:
        return []
    regions: list[dict[str, Any]] = []
    for raw in values:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label") or raw.get("name") or "").strip()
        region: dict[str, Any] = {"label": label}
        try:
            if all(key in raw for key in ("min_lat", "max_lat", "min_lng", "max_lng")):
                min_lat = float(raw["min_lat"])
                max_lat = float(raw["max_lat"])
                min_lng = float(raw["min_lng"])
                max_lng = float(raw["max_lng"])
                if min_lat > max_lat:
                    min_lat, max_lat = max_lat, min_lat
                if min_lng > max_lng:
                    min_lng, max_lng = max_lng, min_lng
                if -90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0 and -180.0 <= min_lng <= 180.0 and -180.0 <= max_lng <= 180.0:
                    region.update({"type": "bbox", "min_lat": min_lat, "max_lat": max_lat, "min_lng": min_lng, "max_lng": max_lng})
                    regions.append(region)
            elif all(key in raw for key in ("latitude", "longitude", "radius_km")):
                lat = float(raw["latitude"])
                lng = float(raw["longitude"])
                radius_km = float(raw["radius_km"])
                if -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0 and radius_km > 0:
                    region.update({"type": "circle", "latitude": lat, "longitude": lng, "radius_km": radius_km})
                    regions.append(region)
        except (TypeError, ValueError):
            continue
    return regions[:20]


def _hop_within_time_window(
    hop: dict[str, Any],
    cargo_load_after_min: int | None,
    cargo_load_before_min: int | None,
) -> bool:
    """Check that the hop's cargo loading window lies within ``[after, before]``."""
    if cargo_load_after_min is None and cargo_load_before_min is None:
        return True
    cargo = hop.get("cargo") if isinstance(hop.get("cargo"), dict) else {}
    raw = cargo.get("load_time")
    if not isinstance(raw, list) or len(raw) != 2:
        return True
    try:
        load_start = wall_time_to_sim_min(str(raw[0]))
        load_end = wall_time_to_sim_min(str(raw[1]))
    except (TypeError, ValueError):
        return False
    if load_end < load_start:
        return False
    if cargo_load_after_min is not None and load_start < int(cargo_load_after_min):
        return False
    if cargo_load_before_min is not None and load_end > int(cargo_load_before_min):
        return False
    return True


def _destination_satisfies_regions(
    destination: tuple[float, float],
    allowed_regions: list[dict[str, Any]],
    forbidden_regions: list[dict[str, Any]],
) -> bool:
    if forbidden_regions and any(_point_in_region(destination, region) for region in forbidden_regions):
        return False
    if allowed_regions and not any(_point_in_region(destination, region) for region in allowed_regions):
        return False
    return True


def _point_in_region(point: tuple[float, float], region: dict[str, Any]) -> bool:
    lat, lng = float(point[0]), float(point[1])
    if region.get("type") == "bbox":
        return (
            float(region["min_lat"]) <= lat <= float(region["max_lat"])
            and float(region["min_lng"]) <= lng <= float(region["max_lng"])
        )
    if region.get("type") == "circle":
        from simkit.simulation_actions import haversine_km

        return haversine_km(lat, lng, float(region["latitude"]), float(region["longitude"])) <= float(region["radius_km"])
    return False
