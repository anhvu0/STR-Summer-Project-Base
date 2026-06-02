from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np


@dataclass
class RouteCandidate:
    route_edges: List[str]
    features: np.ndarray
    route_index: int


_MAX_ROUTE_LEN = 40.0    # normalizer for junction count feature
# Feature layout:
#   0 length_norm, 1 eta_norm, 2 mean_density, 3 max_density,
#   4 first_edge_density, 5 edge_count_norm, 6 novelty_vs_previous_route,
#   7 eta_delta_vs_baseline, 8 length_delta_vs_baseline,
#   9 mean_density_relief_vs_baseline, 10 first_edge_relief_vs_baseline.
#
# The relative features are intentionally centered on the shortest-route
# baseline. The policy can then learn "sacrifice X seconds for Y congestion
# relief" instead of reacting to raw density in isolation.
ROUTE_FEATURE_DIM = 11
_ETA_DELTA_SCALE_S = 120.0
_LENGTH_DELTA_SCALE_M = 600.0

# Sentinel for the per-call density memo (see get_candidates). A bare ``None``
# default would be ambiguous because a density of ``0.0`` is a legitimate value.
_DENSITY_MISSING = object()


def pack_route_candidate_features(
    candidates: Sequence[RouteCandidate],
    route_k: int,
    route_feature_dim: int = ROUTE_FEATURE_DIM,
) -> np.ndarray:
    parts = []
    for candidate in list(candidates)[:int(route_k)]:
        parts.append(
            np.asarray(candidate.features, dtype=np.float32).reshape(int(route_feature_dim))
        )
    while len(parts) < int(route_k):
        parts.append(np.zeros(int(route_feature_dim), dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def filter_candidates_by_first_edges(
    candidates: Sequence[RouteCandidate],
    allowed_first_edges: Set[str],
) -> List[RouteCandidate]:
    return [
        candidate for candidate in candidates
        if len(candidate.route_edges) > 1 and candidate.route_edges[1] in allowed_first_edges
    ]


class RouteCandidateGenerator:
    """Generate k diverse candidate routes between two edges for route-level policy."""

    def __init__(
        self,
        connection_info,
        net=None,
        k_routes: int = 4,
        oversample: int = 8,
        max_route_length_m: float = 8000.0,
        lru_maxsize: int = 2048,
        density_deadband: float = 0.05,
    ):
        self.connection_info = connection_info
        self.net = net
        self.k_routes = int(k_routes)
        self.oversample = int(oversample)
        self.max_route_length_m = float(max_route_length_m)
        self.density_deadband = max(float(density_deadband), 0.0)

        # Build adjacency: edge_id -> list of next_edge_ids (unique)
        self._adjacency: Dict[str, List[str]] = {}
        for edge_id, dir_map in connection_info.outgoing_edges_dict.items():
            seen: Set[str] = set()
            neighbors: List[str] = []
            for nxt in dir_map.values():
                if nxt not in seen:
                    seen.add(nxt)
                    neighbors.append(nxt)
            self._adjacency[edge_id] = neighbors

        self._length: Dict[str, float] = {
            e: max(float(connection_info.edge_length_dict.get(e, 5.0)), 5.0)
            for e in connection_info.edge_list
        }

        # Speed dict for ETA feature — max speed in m/s per edge
        self._speed: Dict[str, float] = {}
        if net is not None:
            for edge_id in connection_info.edge_list:
                try:
                    self._speed[edge_id] = max(float(net.getEdge(edge_id).getSpeed()), 1.0)
                except Exception:
                    self._speed[edge_id] = 8.33  # ~30 km/h fallback
        else:
            for edge_id in connection_info.edge_list:
                self._speed[edge_id] = 8.33

        self._raw_cache: Dict = {}
        self._lru_maxsize = int(lru_maxsize)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_candidates(
        self,
        current_edge_id: str,
        destination_edge_id: str,
        edge_density_fn: Callable[[str], float],
        prev_route_edges: Optional[List[str]] = None,
        allowed_first_edges: Optional[Set[str]] = None,
    ) -> List[RouteCandidate]:
        """Return up to k_routes diverse candidates, including congestion-aware routes."""
        # Per-call density memo. Within a single call the live edge density is frozen
        # (the reservation field is seeded only *after* the action is chosen), yet the
        # same edge is read many times: the density-aware Dijkstra cost functions touch
        # it on every relaxation, and route metrics + the greedy diversity selection
        # re-score the same paths repeatedly. Caching the raw lookup is bit-identical to
        # calling ``edge_density_fn`` every time -- it only removes redundant work.
        _raw_density_fn = edge_density_fn
        _density_memo: Dict[str, float] = {}

        def edge_density_fn(edge_id):
            value = _density_memo.get(edge_id, _DENSITY_MISSING)
            if value is _DENSITY_MISSING:
                value = _raw_density_fn(edge_id)
                _density_memo[edge_id] = value
            return value

        structural_paths = self._get_raw_paths_cached(current_edge_id, destination_edge_id)
        density_paths = self._compute_density_aware_paths(
            current_edge_id,
            destination_edge_id,
            edge_density_fn,
            structural_paths,
        )
        first_edge_paths = self._compute_allowed_first_edge_paths(
            current_edge_id,
            destination_edge_id,
            edge_density_fn,
            allowed_first_edges or set(),
        )
        candidate_paths = self._select_paths(
            self._dedupe_paths([*structural_paths, *first_edge_paths, *density_paths]),
            edge_density_fn,
            self.k_routes,
        )
        if not candidate_paths:
            return []

        prev_set: Set[str] = set(prev_route_edges) if prev_route_edges else set()
        baseline_metrics = self._route_metrics(candidate_paths[0], edge_density_fn)
        candidates = []
        for i, route_edges in enumerate(candidate_paths[: self.k_routes]):
            metrics = self._route_metrics(route_edges, edge_density_fn)
            features = self._compute_features(
                route_edges,
                edge_density_fn,
                prev_set,
                baseline_metrics=baseline_metrics,
                route_metrics=metrics,
            )
            candidates.append(RouteCandidate(
                route_edges=route_edges,
                features=features,
                route_index=i,
            ))

        return candidates

    # ------------------------------------------------------------------
    # Cache layer for raw (unfeaturized) paths
    # ------------------------------------------------------------------

    def _get_raw_paths_cached(self, src: str, dst: str) -> List[List[str]]:
        key = (src, dst)
        cached = self._raw_cache.get(key)
        if cached is not None:
            return cached
        result = self._compute_raw_candidates_uncached(src, dst)
        # Simple LRU eviction: if over capacity, drop oldest (FIFO approximation)
        if len(self._raw_cache) >= self._lru_maxsize:
            oldest_key = next(iter(self._raw_cache))
            del self._raw_cache[oldest_key]
        self._raw_cache[key] = result
        return result

    def _compute_raw_candidates_uncached(self, src: str, dst: str) -> List[List[str]]:
        """Generate up to k diverse raw edge-ID paths from src to dst."""
        # Generate oversample candidate paths with penalized Dijkstra
        candidates: List[List[str]] = []
        excluded_segments: List[Set[str]] = [set()]   # first run: no exclusions

        baseline = self._dijkstra(src, dst, excluded_edges=set())
        if baseline is None:
            return []
        candidates.append(baseline)

        # For each subsequent attempt, exclude a rolling segment of the baseline
        # to force meaningfully different detours
        if len(baseline) > 2:
            segment_len = max(1, len(baseline) // max(self.oversample - 1, 1))
            for i in range(1, self.oversample):
                start = min((i - 1) * segment_len, len(baseline) - 2)
                end = min(start + segment_len, len(baseline) - 1)
                excluded = set(baseline[start:end])
                alt = self._dijkstra(src, dst, excluded_edges=excluded)
                if alt is not None:
                    candidates.append(alt)

        return self._select_diverse(self._dedupe_paths(candidates), self.k_routes)

    def _compute_allowed_first_edge_paths(
        self,
        src: str,
        dst: str,
        edge_density_fn: Callable[[str], float],
        allowed_first_edges: Set[str],
    ) -> List[List[str]]:
        """Generate candidate paths that deliberately cover feasible first turns."""
        if not allowed_first_edges or src == dst:
            return []
        outgoing = set(self._adjacency.get(src, []))
        paths: List[List[str]] = []

        def effective_density(edge_id: str) -> float:
            try:
                density = min(max(float(edge_density_fn(edge_id)), 0.0), 2.0)
            except Exception:
                density = 0.0
            return self._effective_density(density)

        def length_density_cost(edge_id: str) -> float:
            return self._length.get(edge_id, 5.0) * (1.0 + 2.2 * effective_density(edge_id))

        def eta_density_cost(edge_id: str) -> float:
            eta = self._length.get(edge_id, 5.0) / max(self._speed.get(edge_id, 8.33), 1.0)
            return eta * (1.0 + 1.8 * effective_density(edge_id))

        for first_edge in sorted(str(edge_id) for edge_id in allowed_first_edges):
            if first_edge not in outgoing:
                continue
            if first_edge == dst:
                paths.append([src, first_edge])
                continue
            for cost_fn in (None, length_density_cost, eta_density_cost):
                suffix = self._dijkstra(first_edge, dst, excluded_edges=set(), edge_cost_fn=cost_fn)
                if suffix:
                    paths.append([src, *suffix])
        return self._dedupe_paths(paths)

    # ------------------------------------------------------------------
    # Dijkstra with optional edge exclusion
    # ------------------------------------------------------------------

    def _dijkstra(
        self,
        src: str,
        dst: str,
        excluded_edges: Set[str],
        edge_cost_fn: Optional[Callable[[str], float]] = None,
    ) -> Optional[List[str]]:
        """Return shortest-cost path (list of edge IDs) from src to dst, or None."""
        if src == dst:
            return [src]
        if src not in self._adjacency:
            return None
        edge_cost = edge_cost_fn or (lambda edge_id: self._length.get(edge_id, 5.0))

        dist: Dict[str, float] = {src: 0.0}
        prev: Dict[str, Optional[str]] = {src: None}
        heap = [(0.0, src)]

        while heap:
            cost, u = heapq.heappop(heap)
            if cost > dist.get(u, math.inf):
                continue
            if u == dst:
                # Reconstruct path
                path: List[str] = []
                node: Optional[str] = dst
                while node is not None:
                    path.append(node)
                    node = prev[node]
                path.reverse()
                return path

            for v in self._adjacency.get(u, []):
                if v in excluded_edges:
                    continue
                step_cost = max(float(edge_cost(v)), 1.0e-6)
                new_cost = cost + step_cost
                if new_cost < dist.get(v, math.inf):
                    dist[v] = new_cost
                    prev[v] = u
                    heapq.heappush(heap, (new_cost, v))

        return None

    def _compute_density_aware_paths(
        self,
        src: str,
        dst: str,
        edge_density_fn: Callable[[str], float],
        structural_paths: Sequence[List[str]],
    ) -> List[List[str]]:
        if src == dst:
            return [[src]]

        def density(edge_id: str) -> float:
            try:
                raw = min(max(float(edge_density_fn(edge_id)), 0.0), 2.0)
            except Exception:
                raw = 0.0
            return self._effective_density(raw)

        def length_density_cost(edge_id: str) -> float:
            return self._length.get(edge_id, 5.0) * (1.0 + 2.2 * density(edge_id))

        def eta_density_cost(edge_id: str) -> float:
            eta = self._length.get(edge_id, 5.0) / max(self._speed.get(edge_id, 8.33), 1.0)
            return eta * (1.0 + 1.8 * density(edge_id))

        paths = [
            self._dijkstra(src, dst, excluded_edges=set(), edge_cost_fn=length_density_cost),
            self._dijkstra(src, dst, excluded_edges=set(), edge_cost_fn=eta_density_cost),
        ]

        if structural_paths:
            baseline = structural_paths[0]
            avoid_edges = sorted(
                baseline[1:-1],
                key=density,
                reverse=True,
            )[:2]
            for edge_id in avoid_edges:
                paths.append(
                    self._dijkstra(src, dst, excluded_edges={edge_id})
                )

        return [path for path in paths if path]

    @staticmethod
    def _dedupe_paths(paths: Sequence[List[str]]) -> List[List[str]]:
        seen_tuples: Set[tuple] = set()
        unique: List[List[str]] = []
        for path in paths:
            if not path:
                continue
            key = tuple(path)
            if key in seen_tuples:
                continue
            seen_tuples.add(key)
            unique.append(list(path))
        return unique

    def _path_dynamic_score(
        self,
        path: List[str],
        edge_density_fn: Callable[[str], float],
    ) -> float:
        if not path:
            return math.inf
        length = sum(self._length.get(edge_id, 5.0) for edge_id in path)
        eta = sum(
            self._length.get(edge_id, 5.0) / max(self._speed.get(edge_id, 8.33), 1.0)
            for edge_id in path
        )
        densities = []
        for edge_id in path:
            try:
                raw_density = min(max(float(edge_density_fn(edge_id)), 0.0), 2.0)
            except Exception:
                raw_density = 0.0
            densities.append(self._effective_density(raw_density))
        mean_density = float(np.mean(densities)) if densities else 0.0
        max_density = float(max(densities)) if densities else 0.0
        return (
            0.45 * min(length / max(self.max_route_length_m, 1.0), 2.0)
            + 0.35 * min(eta / 4000.0, 2.0)
            + 0.55 * mean_density
            + 0.25 * max_density
        )

    def _select_paths(
        self,
        paths: Sequence[List[str]],
        edge_density_fn: Callable[[str], float],
        k: int,
    ) -> List[List[str]]:
        remaining = self._dedupe_paths(paths)
        if len(remaining) <= int(k):
            return remaining

        # Keep the structural shortest route as candidate 0. Selfless routing
        # should compare detours against a stable selfish baseline rather than
        # letting small density noise reorder the whole action set.
        selected: List[List[str]] = [remaining.pop(0)]
        while remaining and len(selected) < int(k):
            def score(path: List[str]) -> float:
                diversity_bonus = 0.0
                if selected:
                    diversity_bonus = 0.18 * min(
                        self._jaccard_distance(path, chosen) for chosen in selected
                    )
                return self._path_dynamic_score(path, edge_density_fn) - diversity_bonus

            best_path = min(remaining, key=score)
            selected.append(best_path)
            remaining = [path for path in remaining if tuple(path) != tuple(best_path)]
        return selected

    # ------------------------------------------------------------------
    # Diversity selection (greedy max-Jaccard-distance)
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard_distance(a: List[str], b: List[str]) -> float:
        sa, sb = set(a), set(b)
        inter = len(sa & sb)
        union = len(sa | sb)
        return 1.0 - (inter / union if union > 0 else 1.0)

    def _select_diverse(self, paths: List[List[str]], k: int) -> List[List[str]]:
        """Greedy: keep the k most pairwise-diverse paths."""
        if len(paths) <= k:
            return paths
        selected = [paths[0]]   # start with shortest (paths[0] == baseline)
        remaining = paths[1:]
        while len(selected) < k and remaining:
            # Pick the path with maximum minimum distance to already-selected paths
            best_idx, best_score = 0, -1.0
            for i, p in enumerate(remaining):
                min_dist = min(self._jaccard_distance(p, s) for s in selected)
                if min_dist > best_score:
                    best_score = min_dist
                    best_idx = i
            selected.append(remaining.pop(best_idx))
        return selected

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _effective_density(self, density: float) -> float:
        return max(float(density) - self.density_deadband, 0.0)

    def _route_metrics(
        self,
        route_edges: List[str],
        edge_density_fn: Callable[[str], float],
    ) -> Dict[str, float]:
        total_length = sum(self._length.get(e, 5.0) for e in route_edges)
        total_eta = sum(
            self._length.get(e, 5.0) / self._speed.get(e, 8.33)
            for e in route_edges
        )
        densities = []
        for edge_id in route_edges:
            try:
                densities.append(min(max(float(edge_density_fn(edge_id)), 0.0), 1.0))
            except Exception:
                densities.append(0.0)
        mean_density = float(np.mean(densities)) if densities else 0.0
        max_density = float(max(densities)) if densities else 0.0
        first_edge_density = float(densities[1]) if len(densities) > 1 else mean_density
        return {
            "total_length": float(total_length),
            "total_eta": float(total_eta),
            "mean_density": float(mean_density),
            "max_density": float(max_density),
            "first_edge_density": float(first_edge_density),
        }

    def _compute_features(
        self,
        route_edges: List[str],
        edge_density_fn: Callable[[str], float],
        prev_set: Set[str],
        baseline_metrics: Optional[Dict[str, float]] = None,
        route_metrics: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        feats = np.zeros(ROUTE_FEATURE_DIM, dtype=np.float32)
        if not route_edges:
            return feats

        metrics = route_metrics or self._route_metrics(route_edges, edge_density_fn)
        baseline = baseline_metrics or metrics
        total_length = float(metrics["total_length"])
        total_eta = float(metrics["total_eta"])
        mean_density = float(metrics["mean_density"])
        max_density = float(metrics["max_density"])
        first_edge_density = float(metrics["first_edge_density"])

        # Jaccard distance from previous route (exploration novelty)
        if prev_set:
            route_set = set(route_edges)
            inter = len(route_set & prev_set)
            union = len(route_set | prev_set)
            diversity = 1.0 - (inter / union if union > 0 else 1.0)
        else:
            diversity = 0.0

        feats[0] = min(total_length / max(self.max_route_length_m, 1.0), 1.0)
        feats[1] = min(total_eta / 4000.0, 1.0)
        feats[2] = mean_density
        feats[3] = max_density
        feats[4] = first_edge_density
        feats[5] = min(len(route_edges) / _MAX_ROUTE_LEN, 1.0)
        feats[6] = float(diversity)
        feats[7] = min(max(
            (total_eta - float(baseline["total_eta"])) / _ETA_DELTA_SCALE_S,
            -1.0,
        ), 1.0)
        feats[8] = min(max(
            (total_length - float(baseline["total_length"])) / _LENGTH_DELTA_SCALE_M,
            -1.0,
        ), 1.0)
        feats[9] = min(max(
            float(baseline["mean_density"]) - mean_density,
            -1.0,
        ), 1.0)
        feats[10] = min(max(
            float(baseline["first_edge_density"]) - first_edge_density,
            -1.0,
        ), 1.0)
        return feats
