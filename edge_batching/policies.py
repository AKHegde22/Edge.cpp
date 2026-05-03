from __future__ import annotations

from .engine import BatchEngine
from .models import DeviceProfile
from .scheduler import AdaptiveBatchScheduler, _clamp


class FixedBatchScheduler(AdaptiveBatchScheduler):
    """Baseline scheduler with constant batch limit."""

    def __init__(
        self,
        device: DeviceProfile,
        engine: BatchEngine,
        fixed_batch_size: int = 4,
        **kwargs,
    ):
        super().__init__(device=device, engine=engine, **kwargs)
        self.fixed_batch_size = max(device.min_batch_size, fixed_batch_size)

    def _select_batch_limit_locked(self, now: float) -> int:
        total_q = len(self._realtime_q) + len(self._background_q)
        if total_q <= 0:
            return 0
        cap = self.device.effective_batch_capacity()
        fixed = _clamp(self.fixed_batch_size, self.device.min_batch_size, cap)
        return min(fixed, total_q)


class ThroughputFirstScheduler(AdaptiveBatchScheduler):
    """Baseline scheduler that maximizes batch size for throughput."""

    def _select_batch_limit_locked(self, now: float) -> int:
        total_q = len(self._realtime_q) + len(self._background_q)
        if total_q <= 0:
            return 0
        cap = self.device.effective_batch_capacity()
        return min(cap, total_q)


class RealtimeSingleScheduler(AdaptiveBatchScheduler):
    """Latency-first baseline with no batching (batch size=1)."""

    def _select_batch_limit_locked(self, now: float) -> int:
        total_q = len(self._realtime_q) + len(self._background_q)
        if total_q <= 0:
            return 0
        return 1
