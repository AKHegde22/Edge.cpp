from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any


class WorkloadType(str, Enum):
    REALTIME = "realtime"
    BACKGROUND = "background"


@dataclass
class GenerationRequest:
    request_id: str
    prompt: str
    prompt_tokens: int
    max_new_tokens: int
    workload: WorkloadType = WorkloadType.REALTIME
    submitted_at: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Phase 2: State tracking for continuous batching
    tokens_generated: int = 0
    is_finished: bool = False
    generated_text: str = ""
    token_ids: list[int] = field(default_factory=list)
    kv_cache_seq_id: int = -1


@dataclass(slots=True)
class GenerationResult:
    request_id: str
    workload: WorkloadType
    output_text: str
    queue_wait_ms: float
    end_to_end_latency_ms: float
    total_tokens: int = 0


@dataclass(slots=True)
class BatchExecutionMetrics:
    batch_size: int
    runtime_ms: float
    prompt_tokens: int
    generated_tokens: int


@dataclass(slots=True)
class DeviceProfile:
    """
    Hardware hints used by the scheduler.
    """

    name: str
    memory_gb: float
    compute_score: float
    max_batch_size: int
    min_batch_size: int = 1
    target_realtime_latency_ms: float = 250.0
    max_queue_wait_ms: float = 1200.0
    background_weight: float = 0.25
    
    # Phase 2: Hardware Health
    thermal_state: str = "nominal"  # nominal, fair, serious, critical
    battery_level: float = 100.0   # 0.0 to 100.0
    is_low_power_mode: bool = False

    def effective_batch_capacity(self) -> int:
        memory_bound = max(1, int(self.memory_gb * 1.6))
        compute_bound = max(1, int(self.compute_score * 2.2))
        derived = memory_bound + compute_bound
        return max(self.min_batch_size, min(self.max_batch_size, derived))


@dataclass(slots=True)
class BatchRun:
    batch_limit: int
    requests: list[GenerationRequest]
    results: list[GenerationResult]
    metrics: BatchExecutionMetrics


@dataclass(slots=True)
class SchedulerSnapshot:
    queued_realtime: int
    queued_background: int
    completed_requests: int
    completed_batches: int
    current_batch_limit: int
    avg_batch_runtime_ms: float
    avg_tokens_per_second: float
