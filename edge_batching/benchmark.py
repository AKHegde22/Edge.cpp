from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean, pstdev
from typing import Callable

from .engine import MockLlamaCppEngine
from .models import DeviceProfile, GenerationRequest, WorkloadType
from .scheduler import AdaptiveBatchScheduler


@dataclass(slots=True)
class VirtualClock:
    now_s: float = 0.0

    def now(self) -> float:
        return self.now_s

    def set(self, value_s: float) -> None:
        self.now_s = max(self.now_s, value_s)

    def advance(self, delta_s: float) -> None:
        self.now_s += max(0.0, delta_s)


@dataclass(slots=True)
class WorkloadSpec:
    duration_s: float
    realtime_rps: float
    background_rps: float
    realtime_prompt_tokens: tuple[int, int] = (24, 96)
    realtime_gen_tokens: tuple[int, int] = (32, 128)
    background_prompt_tokens: tuple[int, int] = (96, 256)
    background_gen_tokens: tuple[int, int] = (128, 384)


@dataclass(slots=True)
class WorkloadEvent:
    arrival_s: float
    workload: WorkloadType
    prompt_tokens: int
    gen_tokens: int


@dataclass(slots=True)
class PolicyBenchmarkResult:
    policy_name: str
    seeds: list[int]
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]
    request_count: int
    realtime_count: int
    background_count: int


PolicyFactory = Callable[
    [DeviceProfile, MockLlamaCppEngine, Callable[[], float]],
    AdaptiveBatchScheduler,
]


def _sample_poisson_arrivals(
    rate_per_second: float,
    duration_s: float,
    rng: random.Random,
) -> list[float]:
    if rate_per_second <= 0:
        return []
    t = 0.0
    arrivals: list[float] = []
    while True:
        t += rng.expovariate(rate_per_second)
        if t > duration_s:
            break
        arrivals.append(t)
    return arrivals


def generate_workload_events(spec: WorkloadSpec, seed: int) -> list[WorkloadEvent]:
    rng = random.Random(seed)
    events: list[WorkloadEvent] = []

    for arrival in _sample_poisson_arrivals(spec.realtime_rps, spec.duration_s, rng):
        events.append(
            WorkloadEvent(
                arrival_s=arrival,
                workload=WorkloadType.REALTIME,
                prompt_tokens=rng.randint(*spec.realtime_prompt_tokens),
                gen_tokens=rng.randint(*spec.realtime_gen_tokens),
            )
        )

    for arrival in _sample_poisson_arrivals(spec.background_rps, spec.duration_s, rng):
        events.append(
            WorkloadEvent(
                arrival_s=arrival,
                workload=WorkloadType.BACKGROUND,
                prompt_tokens=rng.randint(*spec.background_prompt_tokens),
                gen_tokens=rng.randint(*spec.background_gen_tokens),
            )
        )

    events.sort(key=lambda item: item.arrival_s)
    return events


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    pos = (len(values) - 1) * p
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return values[lower]
    weight = pos - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def run_policy_trial(
    policy_name: str,
    policy_factory: PolicyFactory,
    device: DeviceProfile,
    spec: WorkloadSpec,
    seed: int,
) -> tuple[dict[str, float], dict[str, int]]:
    events = generate_workload_events(spec, seed)
    clock = VirtualClock()
    engine = MockLlamaCppEngine(
        sleep_for_runtime=False,
        jitter_fraction=0.0,
        time_source=clock.now,
    )
    scheduler = policy_factory(device, engine, clock.now)

    submitted = 0
    realtime_submitted = 0
    background_submitted = 0
    total_generated_tokens = 0
    batch_sizes: list[float] = []
    realtime_latency_ms: list[float] = []
    background_latency_ms: list[float] = []
    realtime_queue_ms: list[float] = []
    background_queue_ms: list[float] = []

    next_event_index = 0
    request_index = 0

    while next_event_index < len(events) or scheduler.pending() > 0:
        if scheduler.pending() == 0 and next_event_index < len(events):
            clock.set(events[next_event_index].arrival_s)

        while next_event_index < len(events):
            event = events[next_event_index]
            if event.arrival_s > clock.now():
                break
            req = GenerationRequest(
                request_id=f"{policy_name}-{seed}-{request_index}",
                prompt="synthetic",
                prompt_tokens=event.prompt_tokens,
                max_new_tokens=event.gen_tokens,
                workload=event.workload,
                submitted_at=event.arrival_s,
            )
            scheduler.submit(req)
            request_index += 1
            submitted += 1
            if event.workload == WorkloadType.REALTIME:
                realtime_submitted += 1
            else:
                background_submitted += 1
            next_event_index += 1

        run = scheduler.run_once()
        if run is None:
            if next_event_index < len(events):
                clock.set(events[next_event_index].arrival_s)
            continue

        batch_start_s = clock.now()
        batch_runtime_s = max(0.0, run.metrics.runtime_ms / 1000.0)
        batch_end_s = batch_start_s + batch_runtime_s

        total_generated_tokens += run.metrics.generated_tokens
        batch_sizes.append(float(run.metrics.batch_size))

        for req in run.requests:
            queue_ms = max(0.0, (batch_start_s - req.submitted_at) * 1000.0)
            latency_ms = max(0.0, (batch_end_s - req.submitted_at) * 1000.0)
            if req.workload == WorkloadType.REALTIME:
                realtime_queue_ms.append(queue_ms)
                realtime_latency_ms.append(latency_ms)
            else:
                background_queue_ms.append(queue_ms)
                background_latency_ms.append(latency_ms)

        clock.advance(batch_runtime_s)

    makespan_s = max(clock.now(), spec.duration_s)
    completed = len(realtime_latency_ms) + len(background_latency_ms)
    metrics = {
        "throughput_rps": completed / max(1e-9, makespan_s),
        "generated_tokens_per_second": total_generated_tokens / max(1e-9, makespan_s),
        "avg_batch_size": mean(batch_sizes) if batch_sizes else 0.0,
        "realtime_latency_p50_ms": _percentile(realtime_latency_ms, 0.50),
        "realtime_latency_p95_ms": _percentile(realtime_latency_ms, 0.95),
        "realtime_latency_p99_ms": _percentile(realtime_latency_ms, 0.99),
        "background_latency_p50_ms": _percentile(background_latency_ms, 0.50),
        "background_latency_p95_ms": _percentile(background_latency_ms, 0.95),
        "realtime_queue_p95_ms": _percentile(realtime_queue_ms, 0.95),
        "background_queue_p95_ms": _percentile(background_queue_ms, 0.95),
        "background_completion_rate": (
            len(background_latency_ms) / background_submitted
            if background_submitted
            else 1.0
        ),
    }

    counts = {
        "submitted": submitted,
        "realtime_submitted": realtime_submitted,
        "background_submitted": background_submitted,
    }
    return metrics, counts


def run_benchmark_suite(
    device: DeviceProfile,
    spec: WorkloadSpec,
    policies: dict[str, PolicyFactory],
    seeds: list[int],
) -> list[PolicyBenchmarkResult]:
    results: list[PolicyBenchmarkResult] = []
    for policy_name, policy_factory in policies.items():
        per_seed_metrics: list[dict[str, float]] = []
        counts_reference: dict[str, int] | None = None
        for seed in seeds:
            metrics, counts = run_policy_trial(
                policy_name=policy_name,
                policy_factory=policy_factory,
                device=device,
                spec=spec,
                seed=seed,
            )
            per_seed_metrics.append(metrics)
            if counts_reference is None:
                counts_reference = counts

        metric_names = per_seed_metrics[0].keys() if per_seed_metrics else []
        metrics_mean = {
            name: mean([item[name] for item in per_seed_metrics])
            for name in metric_names
        }
        metrics_std = {
            name: pstdev([item[name] for item in per_seed_metrics])
            for name in metric_names
        }
        counts_reference = counts_reference or {
            "submitted": 0,
            "realtime_submitted": 0,
            "background_submitted": 0,
        }
        results.append(
            PolicyBenchmarkResult(
                policy_name=policy_name,
                seeds=list(seeds),
                metrics_mean=metrics_mean,
                metrics_std=metrics_std,
                request_count=counts_reference["submitted"],
                realtime_count=counts_reference["realtime_submitted"],
                background_count=counts_reference["background_submitted"],
            )
        )
    return results
