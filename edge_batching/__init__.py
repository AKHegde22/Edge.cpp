"""Adaptive batching primitives for edge-oriented llama.cpp serving."""

from .benchmark import PolicyBenchmarkResult, WorkloadSpec, run_benchmark_suite
from .engine import BatchEngine, LlamaCppIterationEngine, MockLlamaCppEngine
from .models import (
    BatchExecutionMetrics,
    BatchRun,
    DeviceProfile,
    GenerationRequest,
    GenerationResult,
    SchedulerSnapshot,
    WorkloadType,
)
from .policies import (
    FixedBatchScheduler,
    RealtimeSingleScheduler,
    ThroughputFirstScheduler,
)
from .scheduler import AdaptiveBatchScheduler
from .service import EdgeBatchingService

__all__ = [
    "AdaptiveBatchScheduler",
    "BatchEngine",
    "BatchExecutionMetrics",
    "BatchRun",
    "DeviceProfile",
    "EdgeBatchingService",
    "FixedBatchScheduler",
    "GenerationRequest",
    "GenerationResult",
    "LlamaCppIterationEngine",
    "MockLlamaCppEngine",
    "PolicyBenchmarkResult",
    "RealtimeSingleScheduler",
    "SchedulerSnapshot",
    "ThroughputFirstScheduler",
    "WorkloadType",
    "WorkloadSpec",
    "run_benchmark_suite",
]
