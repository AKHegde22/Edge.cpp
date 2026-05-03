"""Adaptive batching primitives for edge-oriented llama.cpp serving."""

from .benchmark import PolicyBenchmarkResult, WorkloadSpec, run_benchmark_suite
from .engine import LlamaCppIterationEngine
from .hardware_monitor import HardwareMonitor
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
    "BatchExecutionMetrics",
    "BatchRun",
    "DeviceProfile",
    "EdgeBatchingService",
    "FixedBatchScheduler",
    "GenerationRequest",
    "GenerationResult",
    "HardwareMonitor",
    "LlamaCppIterationEngine",
    "PolicyBenchmarkResult",
    "RealtimeSingleScheduler",
    "SchedulerSnapshot",
    "ThroughputFirstScheduler",
    "WorkloadType",
    "WorkloadSpec",
    "run_benchmark_suite",
]
