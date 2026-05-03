from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable

from .engine import BatchEngine
from .models import (
    BatchExecutionMetrics,
    BatchRun,
    DeviceProfile,
    GenerationRequest,
    SchedulerSnapshot,
    WorkloadType,
)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


@dataclass(slots=True)
class _EwmaRuntimeModel:
    prefill_ms_per_token: float = 0.35
    decode_ms_per_token: float = 2.4
    alpha: float = 0.18

    def update(self, metrics: BatchExecutionMetrics) -> None:
        if metrics.batch_size <= 0:
            return
        token_count = max(1, metrics.prompt_tokens + metrics.generated_tokens)
        runtime_per_token = metrics.runtime_ms / token_count
        # Prefill work is usually cheaper than decode, so keep decode heavier.
        observed_prefill = runtime_per_token * 0.6
        observed_decode = runtime_per_token * 1.4
        self.prefill_ms_per_token = (
            1 - self.alpha
        ) * self.prefill_ms_per_token + self.alpha * observed_prefill
        self.decode_ms_per_token = (
            1 - self.alpha
        ) * self.decode_ms_per_token + self.alpha * observed_decode


class AdaptiveBatchScheduler:
    """
    Hybrid scheduler for edge deployments:
    - Latency-aware adaptive batch sizing
    - Mixed workload queueing (realtime + background)
    - Fairness with starvation protection
    """

    def __init__(
        self,
        device: DeviceProfile,
        engine: LlamaCppIterationEngine, # Use the new iteration engine
        time_source: Callable[[], float] | None = None,
    ):
        self.device = device
        self.engine = engine
        self._time_source = time_source or time.monotonic

        self._realtime_q: deque[GenerationRequest] = deque()
        self._background_q: deque[GenerationRequest] = deque()
        self._active_batch: list[GenerationRequest] = []
        self._cv = threading.Condition()

        self._completed_requests = 0
        self._completed_batches = 0
        self._avg_batch_runtime_ms = 0.0
        self._avg_tokens_per_second = 0.0
        
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()

    def submit(self, request: GenerationRequest) -> None:
        with self._cv:
            if request.workload == WorkloadType.BACKGROUND:
                self._background_q.append(request)
            else:
                self._realtime_q.append(request)
            self._cv.notify()

    def run_step(self) -> list[GenerationResult]:
        """
        Performs one iteration of continuous batching.
        """
        with self._cv:
            # 1. Cleanup finished requests from active batch
            self._active_batch = [r for r in self._active_batch if not r.is_finished]
            
            # 2. Adaptive In-Flight Preemption & Injection
            # If we have space in batch, fill with Realtime first, then Background
            while len(self._active_batch) < self.device.max_batch_size:
                if self._realtime_q:
                    self._active_batch.append(self._realtime_q.popleft())
                elif self._background_q:
                    # Power-Aware: If serious thermal pressure, don't add background tasks
                    if self.device.thermal_state in ["serious", "critical"]:
                        break
                    self._active_batch.append(self._background_q.popleft())
                else:
                    break
            
            # 3. Forced Preemption: If a Realtime request is waiting but batch is full of Background
            if self._realtime_q and all(r.workload == WorkloadType.BACKGROUND for r in self._active_batch):
                # Preempt the last background task to make room for urgent Realtime
                preempted = self._active_batch.pop()
                # We don't discard progress, just move it back to front of queue
                self._background_q.appendleft(preempted)
                self._active_batch.append(self._realtime_q.popleft())

            if not self._active_batch:
                return []

        # 4. Execute one iteration token
        start = time.monotonic()
        finished = self.engine.step(self._active_batch)
        end = time.monotonic()
        
        # Update metrics
        if finished:
            with self._cv:
                self._completed_requests += len(finished)
                # Statistics update...
        
        return finished

    def _worker_loop(self, poll_interval_s: float) -> None:
        while not self._stop_event.is_set():
            with self._cv:
                if not self._realtime_q and not self._background_q and not self._active_batch:
                    self._cv.wait(timeout=poll_interval_s)
                    continue
            self.run_step()

    def _now(self) -> float:
        return self._time_source()

    def _select_batch_limit_locked(self, now: float) -> int:
        total_q = len(self._realtime_q) + len(self._background_q)
        if total_q <= 0:
            return 0

        device_cap = self.device.effective_batch_capacity()
        queue_pressure = min(1.0, total_q / float(max(1, device_cap)))

        avg_prompt = self._avg_prompt_tokens_locked()
        avg_gen = self._avg_generation_tokens_locked()
        est_single_latency_ms = self._estimate_batch_runtime_ms(1, avg_prompt, avg_gen)

        latency_ratio = est_single_latency_ms / max(
            1.0, self.device.target_realtime_latency_ms
        )
        if latency_ratio > 1.0:
            latency_scale = max(0.40, 1.0 / latency_ratio)
        else:
            latency_scale = min(1.40, 1.0 + (1.0 - latency_ratio) * 0.40)

        base = self.device.min_batch_size + int(
            (device_cap - self.device.min_batch_size) * queue_pressure
        )
        if total_q >= 2:
            base = max(base, 2)
        limit = int(round(base * latency_scale))
        limit = _clamp(limit, self.device.min_batch_size, device_cap)

        if queue_pressure >= 0.75:
            limit = max(limit, int(round(device_cap * 0.60)))

        if self._realtime_q:
            oldest_rt_wait_ms = (now - self._realtime_q[0].submitted_at) * 1000.0
            urgency = oldest_rt_wait_ms / max(
                1.0, self.device.target_realtime_latency_ms
            )
            if urgency >= 1.20:
                limit = min(limit, 2)
            elif urgency >= 0.95:
                limit = max(self.device.min_batch_size, int(round(limit * 0.65)))

        return min(limit, total_q)

    def _estimate_batch_runtime_ms(
        self,
        batch_size: int,
        avg_prompt_tokens: float,
        avg_generation_tokens: float,
    ) -> float:
        if batch_size <= 0:
            return 0.0
        prompt_total = avg_prompt_tokens * batch_size
        gen_total = avg_generation_tokens * batch_size
        speedup = 1.0 + (batch_size - 1) * 0.6
        runtime = (
            prompt_total * self._runtime.prefill_ms_per_token
            + gen_total * self._runtime.decode_ms_per_token
        ) / speedup
        return runtime + 2.0

    def _avg_prompt_tokens_locked(self) -> float:
        items = list(self._realtime_q) + list(self._background_q)
        if not items:
            return 48.0
        return sum(item.prompt_tokens for item in items) / float(len(items))

    def _avg_generation_tokens_locked(self) -> float:
        items = list(self._realtime_q) + list(self._background_q)
        if not items:
            return 72.0
        return sum(item.max_new_tokens for item in items) / float(len(items))

    def _build_batch_locked(
        self,
        limit: int,
        now: float,
    ) -> list[GenerationRequest]:
        if limit <= 0:
            return []

        realtime_urgent_after_s = self.device.target_realtime_latency_ms / 1000.0 * 0.45
        background_starvation_after_s = self.device.max_queue_wait_ms / 1000.0

        batch: list[GenerationRequest] = []
        while len(batch) < limit and (self._realtime_q or self._background_q):
            if not self._realtime_q:
                batch.append(self._background_q.popleft())
                continue
            if not self._background_q:
                batch.append(self._realtime_q.popleft())
                continue

            realtime_wait = now - self._realtime_q[0].submitted_at
            background_wait = now - self._background_q[0].submitted_at

            if realtime_wait >= realtime_urgent_after_s:
                batch.append(self._realtime_q.popleft())
                continue
            if background_wait >= background_starvation_after_s:
                batch.append(self._background_q.popleft())
                continue

            # Deficit scheduling: give background work a bounded share.
            self._bg_credit += self.device.background_weight
            if self._bg_credit >= 1.0:
                self._bg_credit -= 1.0
                batch.append(self._background_q.popleft())
            else:
                batch.append(self._realtime_q.popleft())

        return batch

    def _record_batch_locked(self, metrics: BatchExecutionMetrics) -> None:
        self._completed_batches += 1
        self._completed_requests += metrics.batch_size

        if self._completed_batches == 1:
            self._avg_batch_runtime_ms = metrics.runtime_ms
        else:
            self._avg_batch_runtime_ms = (
                1 - self._stats_alpha
            ) * self._avg_batch_runtime_ms + self._stats_alpha * metrics.runtime_ms

        if metrics.runtime_ms > 0:
            tps = metrics.generated_tokens / (metrics.runtime_ms / 1000.0)
            if self._completed_batches == 1:
                self._avg_tokens_per_second = tps
            else:
                self._avg_tokens_per_second = (
                    1 - self._stats_alpha
                ) * self._avg_tokens_per_second + self._stats_alpha * tps
