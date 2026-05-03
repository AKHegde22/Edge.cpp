from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, List, Optional

from .engine import LlamaCppIterationEngine
from .models import (
    DeviceProfile,
    GenerationRequest,
    GenerationResult,
    SchedulerSnapshot,
    WorkloadType,
)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


class AdaptiveBatchScheduler:
    """
    Production-grade hybrid scheduler for edge LLM deployments:
    - Token-level continuous batching (iteration-level)
    - In-flight request preemption (Realtime vs Background)
    - Thermal-aware background throttling
    """

    def __init__(
        self,
        device: DeviceProfile,
        engine: LlamaCppIterationEngine,
        time_source: Optional[Callable[[], float]] = None,
    ):
        self.device = device
        self.engine = engine
        self._time_source = time_source or time.monotonic

        self._realtime_q: deque[GenerationRequest] = deque()
        self._background_q: deque[GenerationRequest] = deque()
        self._active_batch: List[GenerationRequest] = []
        
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        # Statistics tracking
        self._completed_requests = 0
        self._completed_batches = 0
        self._total_generated_tokens = 0
        self._start_time = self._time_source()
        
        self._stats_alpha = 0.1
        self._avg_batch_runtime_ms = 0.0
        self._avg_tokens_per_second = 0.0

    def submit(self, request: GenerationRequest) -> None:
        """Submit a new generation request to the scheduler."""
        with self._cv:
            if request.workload == WorkloadType.BACKGROUND:
                self._background_q.append(request)
            else:
                self._realtime_q.append(request)
            self._cv.notify()

    def pending(self) -> int:
        """Return the total number of requests in the system (queued or active)."""
        with self._lock:
            return len(self._realtime_q) + len(self._background_q) + len(self._active_batch)

    def run_step(self) -> List[GenerationResult]:
        """
        Performs ONE iteration of continuous batching across all active requests.
        Handles injection of new requests and preemption logic.
        """
        with self._cv:
            # 1. Cleanup finished requests from active batch
            self._active_batch = [r for r in self._active_batch if not r.is_finished]
            
            # 2. Injection & Capacity Management
            # Fill remaining slots in the batch
            while len(self._active_batch) < self.device.max_batch_size:
                if self._realtime_q:
                    self._active_batch.append(self._realtime_q.popleft())
                elif self._background_q:
                    # Power-Aware Throttling: Skip background if hardware is stressed
                    if self.device.thermal_state in ["serious", "critical"] or self.device.is_low_power_mode:
                        break
                    self._active_batch.append(self._background_q.popleft())
                else:
                    break
            
            # 3. Forced Preemption Logic
            # If urgent realtime requests are waiting but batch is saturated with background tasks,
            # swap the last background task for the oldest realtime task.
            if self._realtime_q and all(r.workload == WorkloadType.BACKGROUND for r in self._active_batch):
                if self._active_batch:
                    preempted = self._active_batch.pop()
                    self._background_q.appendleft(preempted)
                    self._active_batch.append(self._realtime_q.popleft())

            if not self._active_batch:
                return []

        # 4. Execute token decode step via engine
        start_ts = self._time_source()
        finished_results = self.engine.step(self._active_batch)
        end_ts = self._time_source()
        
        # 5. Update Statistics
        runtime_ms = (end_ts - start_ts) * 1000.0
        with self._lock:
            self._completed_batches += 1
            if finished_results:
                self._completed_requests += len(finished_results)
                tokens_in_finished = sum(r.total_tokens for r in finished_results)
                self._total_generated_tokens += tokens_in_finished
            
            # Moving average of batch runtime
            if self._completed_batches == 1:
                self._avg_batch_runtime_ms = runtime_ms
            else:
                self._avg_batch_runtime_ms = (
                    (1 - self._stats_alpha) * self._avg_batch_runtime_ms + 
                    self._stats_alpha * runtime_ms
                )
            
            # Throughput calculation
            elapsed = end_ts - self._start_time
            if elapsed > 0:
                self._avg_tokens_per_second = self._total_generated_tokens / elapsed

        return finished_results

    def drain(self, timeout_s: Optional[float] = None) -> List[GenerationResult]:
        """
        Block and process all pending requests until both queues and active batch are empty.
        Useful for benchmarks or CLI simulations.
        """
        all_finished = []
        start_drain = self._time_source()
        
        while self.pending() > 0:
            if timeout_s and (self._time_source() - start_drain) > timeout_s:
                break
            results = self.run_step()
            all_finished.extend(results)
            if not results and self.pending() > 0:
                # Avoid busy loop if engine is waiting for something
                time.sleep(0.001)
                
        return all_finished

    def snapshot(self) -> SchedulerSnapshot:
        """Capture a point-in-time state of the scheduler."""
        with self._lock:
            return SchedulerSnapshot(
                queued_realtime=len(self._realtime_q),
                queued_background=len(self._background_q),
                completed_requests=self._completed_requests,
                completed_batches=self._completed_batches,
                current_batch_limit=self.device.max_batch_size,
                avg_batch_runtime_ms=self._avg_batch_runtime_ms,
                avg_tokens_per_second=self._avg_tokens_per_second,
            )
