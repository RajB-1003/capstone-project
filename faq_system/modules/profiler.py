"""
profiler.py — Phase 7: Latency Profiler

Responsibilities:
  - Measure wall-clock time for each pipeline stage in milliseconds
  - Provide a context manager interface for clean stage-level wrapping
  - Return a structured latency_ms dict compatible with the final response

Design:
  - LatencyProfiler is a lightweight, zero-dependency timer
  - Uses time.perf_counter() (highest-resolution clock available)
  - Context manager: `with profiler.measure("stage"): ...`
  - get_profile() returns all recorded stages + "total" in ms
  - No external dependencies (no numpy, no ML libraries)

Public API:
  LatencyProfiler              — profiler class
    .measure(label)            — context manager, records elapsed ms for label
    .elapsed_ms()              — total elapsed ms since profiler created
    .get_profile()             -> dict[str, float]   (stage → ms, plus "total")
"""

import time
from contextlib import contextmanager


class LatencyProfiler:
    """
    Lightweight wall-clock profiler for pipeline stage timing.

    Usage:
        profiler = LatencyProfiler()

        with profiler.measure("embedding"):
            vec = embed_single(query, model)

        with profiler.measure("semantic"):
            results = search_semantic(...)

        profile = profiler.get_profile()
        # {"embedding": 12.3, "semantic": 11.8, ..., "total": 58.4}

    Notes:
        - Stages are recorded in insertion order (Python 3.7+ dict).
        - "total" is always computed as wall-clock from __init__ to get_profile(),
          NOT the sum of individual stages (captures overhead between stages too).
        - If the same label is measured twice, the later measurement overwrites
          the earlier one to avoid confusion in repeat calls.
        - All times are in milliseconds (ms), rounded to 3 decimal places.
    """

    def __init__(self):
        self._start    = time.perf_counter()
        self._stages:  dict[str, float] = {}   # label → elapsed ms

    @contextmanager
    def measure(self, label: str):
        """
        Context manager: time the enclosed block and record it under `label`.

        Args:
            label: Stage name (e.g., "embedding", "semantic", "hybrid").
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._stages[label] = round(elapsed_ms, 3)

    def elapsed_ms(self) -> float:
        """Total wall-clock ms since profiler was created (snapshot)."""
        return round((time.perf_counter() - self._start) * 1000, 3)

    def get_profile(self) -> dict[str, float]:
        """
        Return all recorded stage timings plus the total wall-clock time.

        Returns:
            dict: {stage_label: ms, ..., "total": ms}
            Keys (in order): embedding, router, semantic, keyword, hybrid,
                             explainability, total (plus any custom labels).
        """
        profile = dict(self._stages)
        profile["total"] = self.elapsed_ms()
        return profile

    def __repr__(self) -> str:
        stages = ", ".join(f"{k}={v:.1f}ms" for k, v in self._stages.items())
        return f"LatencyProfiler({stages}, elapsed={self.elapsed_ms():.1f}ms)"
