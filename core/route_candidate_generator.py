from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Set

import numpy as np


@dataclass
class RouteCandidate:
    route_edges: List[str]
    features: np.ndarray   # shape (5,)
    route_index: int


_MAX_ROUTE_LEN = 40.0    # normalizer for junction count feature


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
    ):
        self.connection_info = connection_info
        self.net = net
        self.k_routes = int(k_routes)
        self.oversample = int(oversample)
        self.max_route_length_m = float(max_route_length_m)

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

        # Wrap the raw-path cache in lru_cache on an instance method via closure
        raw_fn = self._compute_raw_candidates_uncached
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
    ) -> List[RouteCandidate]:
        """Return up to k_routes diverse RouteCandidate objects."""
        raw_paths = self._get_raw_paths_cached(current_edge_id, destination_edge_id)
        if not raw_paths:
            return []

        prev_set: Set[str] = set(prev_route_edges) if prev_route_edges else set()
        candidates = []
        for i, route_edges in enumerate(raw_paths[: self.k_routes]):
            features = self._compute_features(route_edges, edge_density_fn, prev_set)
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

        # Deduplicate
        seen_tuples: Set[tuple] = set()
        unique: List[List[str]] = []
        for p in candidates:
            t = tuple(p)
            if t not in seen_tuples:
                seen_tuples.add(t)
                unique.append(p)

        return self._select_diverse(unique, self.k_routes)

    # ------------------------------------------------------------------
    # Dijkstra with optional edge exclusion
    # ------------------------------------------------------------------

    def _dijkstra(
        self,
        src: str,
        dst: str,
        excluded_edges: Set[str],
    ) -> Optional[List[str]]:
        """Return shortest-cost path (list of edge IDs) from src to dst, or None."""
        if src == dst:
            return [src]
        if src not in self._adjacency:
            return None

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
                new_cost = cost + self._length.get(v, 5.0)
                if new_cost < dist.get(v, math.inf):
                    dist[v] = new_cost
                    prev[v] = u
                    heapq.heappush(heap, (new_cost, v))

        return None

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

    def _compute_features(
        self,
        route_edges: List[str],
        edge_density_fn: Callable[[str], float],
        prev_set: Set[str],
    ) -> np.ndarray:
        feats = np.zeros(5, dtype=np.float32)
        if not route_edges:
            return feats

        total_length = sum(self._length.get(e, 5.0) for e in route_edges)
        total_eta = sum(
            self._length.get(e, 5.0) / self._speed.get(e, 8.33)
            for e in route_edges
        )
        mean_density = float(np.mean([
            min(float(edge_density_fn(e)), 1.0) for e in route_edges
        ]))

        # Jaccard distance from previous route (exploration novelty)
        if prev_set:
            route_set = set(route_edges)
            inter = len(route_set & prev_set)
            union = len(route_set | prev_set)
            diversity = 1.0 - (inter / union if union > 0 else 1.0)
        else:
            diversity = 0.0

        feats[0] = min(total_length / max(self.max_route_length_m, 1.0), 1.0)
        feats[1] = min(len(route_edges) / _MAX_ROUTE_LEN, 1.0)
        feats[2] = mean_density
        feats[3] = min(total_eta / 4000.0, 1.0)  # normalized by MAX_SIMULATION_STEPS
        feats[4] = float(diversity)
        return feats
