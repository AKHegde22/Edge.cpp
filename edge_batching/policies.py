from __future__ import annotations

from typing import List

from .engine import LlamaCppIterationEngine
from .models import DeviceProfile, GenerationResult, WorkloadType
from .scheduler import AdaptiveBatchScheduler


class FixedBatchScheduler(AdaptiveBatchScheduler):
    """
    Real baseline scheduler that enforces a static batch size limit.
    Useful for comparing continuous batching with/without adaptive sizing.
    """

    def __init__(
        self,
        device: DeviceProfile,
        engine: LlamaCppIterationEngine,
        fixed_limit: int = 4,
        **kwargs,
    ):
        super().__init__(device=device, engine=engine, **kwargs)
        self.fixed_limit = max(1, fixed_limit)

    def run_step(self) -> List[GenerationResult]:
        """Override to use a fixed batch limit instead of the device max."""
        with self._cv:
            self._active_batch = [r for r in self._active_batch if not r.is_finished]
            
            # Injection with fixed limit
            while len(self._active_batch) < self.fixed_limit:
                if self._realtime_q:
                    self._active_batch.append(self._realtime_q.popleft())
                elif self._background_q:
                    # Even in fixed mode, we still honor thermal state for safety
                    if self.device.thermal_state in ["serious", "critical"]:
                        break
                    self._active_batch.append(self._background_q.popleft())
                else:
                    break

            if not self._active_batch:
                return []

        # Rest of logic is the same as parent
        return super().run_step()


class ThroughputFirstScheduler(AdaptiveBatchScheduler):
    """
    Real baseline scheduler that ignores latency targets and always tries 
    to fill the hardware to its maximum batch capacity.
    """

    def run_step(self) -> List[GenerationResult]:
        """Same as Adaptive but without the preemption or thermal throttling of background."""
        with self._cv:
            self._active_batch = [r for r in self._active_batch if not r.is_finished]
            
            while len(self._active_batch) < self.device.max_batch_size:
                if self._realtime_q:
                    self._active_batch.append(self._realtime_q.popleft())
                elif self._background_q:
                    self._active_batch.append(self._background_q.popleft())
                else:
                    break
                    
            if not self._active_batch:
                return []
                
        return super().run_step()


class RealtimeSingleScheduler(AdaptiveBatchScheduler):
    """
    Real baseline scheduler that processes only one request at a time (batch size = 1).
    Ensures minimum possible latency for that single request but zero throughput gain.
    """

    def run_step(self) -> List[GenerationResult]:
        with self._cv:
            self._active_batch = [r for r in self._active_batch if not r.is_finished]
            
            if not self._active_batch:
                if self._realtime_q:
                    self._active_batch.append(self._realtime_q.popleft())
                elif self._background_q:
                    self._active_batch.append(self._background_q.popleft())
            
            if not self._active_batch:
                return []
                
        return super().run_step()
