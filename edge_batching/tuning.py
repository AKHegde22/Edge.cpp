from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .benchmark import WorkloadSpec, run_policy_trial
from .models import DeviceProfile
from .scheduler import AdaptiveBatchScheduler


@dataclass(slots=True)
class AdaptiveTuneResult:
    target_realtime_latency_ms: float
    background_weight: float
    max_queue_wait_ms: float
    objective: float
    realtime_latency_p95_ms: float
    background_latency_p95_ms: float
    throughput_rps: float
    generated_tokens_per_second: float


def _adaptive_factory(device: DeviceProfile):
    return lambda dev, engine, clock: AdaptiveBatchScheduler(
        device=dev,
        engine=engine,
        time_source=clock,
    )


def tune_adaptive(
    base_device: DeviceProfile,
    workload: WorkloadSpec,
    seeds: list[int],
    latency_targets_ms: list[float],
    background_weights: list[float],
    max_queue_waits_ms: list[float],
) -> list[AdaptiveTuneResult]:
    rows: list[AdaptiveTuneResult] = []
    factory = _adaptive_factory(base_device)
    for target in latency_targets_ms:
        for bg_weight in background_weights:
            for max_wait in max_queue_waits_ms:
                device = DeviceProfile(
                    name=base_device.name,
                    memory_gb=base_device.memory_gb,
                    compute_score=base_device.compute_score,
                    max_batch_size=base_device.max_batch_size,
                    min_batch_size=base_device.min_batch_size,
                    target_realtime_latency_ms=target,
                    max_queue_wait_ms=max_wait,
                    background_weight=bg_weight,
                )
                metrics = []
                for seed in seeds:
                    run_metrics, _ = run_policy_trial(
                        policy_name="adaptive_hybrid",
                        policy_factory=factory,
                        device=device,
                        spec=workload,
                        seed=seed,
                    )
                    metrics.append(run_metrics)

                rt_p95 = mean(item["realtime_latency_p95_ms"] for item in metrics)
                bg_p95 = mean(item["background_latency_p95_ms"] for item in metrics)
                throughput = mean(item["throughput_rps"] for item in metrics)
                tok_s = mean(item["generated_tokens_per_second"] for item in metrics)
                objective = rt_p95 + (0.25 * bg_p95) - (100.0 * throughput)

                rows.append(
                    AdaptiveTuneResult(
                        target_realtime_latency_ms=target,
                        background_weight=bg_weight,
                        max_queue_wait_ms=max_wait,
                        objective=objective,
                        realtime_latency_p95_ms=rt_p95,
                        background_latency_p95_ms=bg_p95,
                        throughput_rps=throughput,
                        generated_tokens_per_second=tok_s,
                    )
                )

    rows.sort(key=lambda item: item.objective)
    return rows


def _write_csv(path: Path, rows: list[AdaptiveTuneResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "objective",
                "target_realtime_latency_ms",
                "background_weight",
                "max_queue_wait_ms",
                "realtime_latency_p95_ms",
                "background_latency_p95_ms",
                "throughput_rps",
                "generated_tokens_per_second",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": idx,
                    "objective": row.objective,
                    "target_realtime_latency_ms": row.target_realtime_latency_ms,
                    "background_weight": row.background_weight,
                    "max_queue_wait_ms": row.max_queue_wait_ms,
                    "realtime_latency_p95_ms": row.realtime_latency_p95_ms,
                    "background_latency_p95_ms": row.background_latency_p95_ms,
                    "throughput_rps": row.throughput_rps,
                    "generated_tokens_per_second": row.generated_tokens_per_second,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search adaptive scheduler parameters for a target edge workload."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("research_outputs/adaptive_tuning.csv"),
    )
    parser.add_argument("--seeds", type=str, default="101,202,303")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]

    device = DeviceProfile(
        name="jetson-like",
        memory_gb=8.0,
        compute_score=4.0,
        max_batch_size=16,
        target_realtime_latency_ms=220.0,
        max_queue_wait_ms=900.0,
        background_weight=0.30,
    )
    workload = WorkloadSpec(duration_s=120.0, realtime_rps=3.0, background_rps=1.0)
    results = tune_adaptive(
        base_device=device,
        workload=workload,
        seeds=seeds,
        latency_targets_ms=[180.0, 220.0, 260.0, 300.0],
        background_weights=[0.20, 0.30, 0.40],
        max_queue_waits_ms=[500.0, 900.0, 1300.0],
    )
    _write_csv(args.output_csv, results)
    print(f"wrote_tuning_csv={args.output_csv}")
    if results:
        best = results[0]
        print(
            "best objective=%.2f target_ms=%.1f bg_weight=%.2f max_wait_ms=%.1f "
            "rt_p95=%.1f bg_p95=%.1f throughput=%.2f"
            % (
                best.objective,
                best.target_realtime_latency_ms,
                best.background_weight,
                best.max_queue_wait_ms,
                best.realtime_latency_p95_ms,
                best.background_latency_p95_ms,
                best.throughput_rps,
            )
        )


if __name__ == "__main__":
    main()
