from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, Dict, List, Tuple

from .engine import LlamaCppIterationEngine
from .models import DeviceProfile, GenerationRequest, WorkloadType
from .scheduler import AdaptiveBatchScheduler
from .service import EdgeBatchingService


@dataclass(slots=True)
class WorkloadSpec:
    """Defines the shape of a benchmarking workload."""
    duration_s: float
    realtime_rps: float
    background_rps: float
    realtime_prompt_tokens: Tuple[int, int] = (24, 96)
    realtime_gen_tokens: Tuple[int, int] = (32, 128)
    background_prompt_tokens: Tuple[int, int] = (96, 256)
    background_gen_tokens: Tuple[int, int] = (128, 384)


@dataclass(slots=True)
class WorkloadEvent:
    arrival_s: float
    workload: WorkloadType
    prompt_tokens: int
    gen_tokens: int


@dataclass(slots=True)
class PolicyBenchmarkResult:
    policy_name: str
    seeds: List[int]
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    request_count: int
    realtime_count: int
    background_count: int


def _sample_poisson_arrivals(
    rate_per_second: float,
    duration_s: float,
    rng: random.Random,
) -> List[float]:
    if rate_per_second <= 0:
        return []
    t = 0.0
    arrivals: List[float] = []
    while True:
        t += rng.expovariate(rate_per_second)
        if t > duration_s:
            break
        arrivals.append(t)
    return arrivals


def generate_workload_events(spec: WorkloadSpec, seed: int) -> List[WorkloadEvent]:
    rng = random.Random(seed)
    events: List[WorkloadEvent] = []

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


def _percentile(values: List[float], p: float) -> float:
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
    policy_class: type[AdaptiveBatchScheduler],
    engine: LlamaCppIterationEngine,
    device: DeviceProfile,
    spec: WorkloadSpec,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Runs a real-time benchmarking trial using a real model and engine.
    """
    events = generate_workload_events(spec, seed)
    scheduler = policy_class(device=device, engine=engine)
    service = EdgeBatchingService(scheduler)
    service.start()

    print(f"  Starting trial: policy={policy_name} seed={seed} events={len(events)}")
    
    start_time = time.monotonic()
    futures = []
    
    # Process events in real-time
    for event in events:
        now = time.monotonic()
        wait_time = (start_time + event.arrival_s) - now
        if wait_time > 0:
            time.sleep(wait_time)
            
        req = GenerationRequest(
            request_id=f"{policy_name}-{seed}-{len(futures)}",
            prompt="Tell me a short story about edge computing."[:event.prompt_tokens], # Truncated for token count approximation
            prompt_tokens=event.prompt_tokens,
            max_new_tokens=event.gen_tokens,
            workload=event.workload,
        )
        futures.append((event, service.submit(req)))

    # Wait for all to finish
    results = []
    for event, future in futures:
        try:
            res = future.result(timeout=120) # 2 min timeout per request
            results.append((event, res))
        except Exception as e:
            print(f"    Request failed: {e}")

    service.stop()
    end_time = time.monotonic()
    
    # Calculate metrics
    realtime_latencies = [r.end_to_end_latency_ms for e, r in results if e.workload == WorkloadType.REALTIME]
    background_latencies = [r.end_to_end_latency_ms for e, r in results if e.workload == WorkloadType.BACKGROUND]
    
    total_tokens = sum(r.total_tokens for e, r in results)
    makespan = end_time - start_time
    
    metrics = {
        "throughput_rps": len(results) / max(1e-9, makespan),
        "generated_tokens_per_second": total_tokens / max(1e-9, makespan),
        "realtime_latency_p50_ms": _percentile(realtime_latencies, 0.50),
        "realtime_latency_p95_ms": _percentile(realtime_latencies, 0.95),
        "background_latency_p50_ms": _percentile(background_latencies, 0.50),
        "background_latency_p95_ms": _percentile(background_latencies, 0.95),
    }

    counts = {
        "submitted": len(events),
        "completed": len(results),
        "realtime_submitted": sum(1 for e in events if e.workload == WorkloadType.REALTIME),
        "background_submitted": sum(1 for e in events if e.workload == WorkloadType.BACKGROUND),
    }
    
    return metrics, counts


def run_benchmark_suite(
    engine: LlamaCppIterationEngine,
    device: DeviceProfile,
    spec: WorkloadSpec,
    policies: Dict[str, type[AdaptiveBatchScheduler]],
    seeds: List[int],
) -> List[PolicyBenchmarkResult]:
    results: List[PolicyBenchmarkResult] = []
    for policy_name, policy_class in policies.items():
        per_seed_metrics = []
        counts_ref = None
        for seed in seeds:
            metrics, counts = run_policy_trial(
                policy_name=policy_name,
                policy_class=policy_class,
                engine=engine,
                device=device,
                spec=spec,
                seed=seed
            )
            per_seed_metrics.append(metrics)
            counts_ref = counts

        if not per_seed_metrics:
            continue

        metric_names = per_seed_metrics[0].keys()
        metrics_mean = {name: mean([m[name] for m in per_seed_metrics]) for name in metric_names}
        metrics_std = {name: pstdev([m[name] for m in per_seed_metrics]) for name in metric_names}
        
        results.append(
            PolicyBenchmarkResult(
                policy_name=policy_name,
                seeds=seeds,
                metrics_mean=metrics_mean,
                metrics_std=metrics_std,
                request_count=counts_ref["submitted"],
                realtime_count=counts_ref["realtime_submitted"],
                background_count=counts_ref["background_submitted"]
            )
        )
    return results
