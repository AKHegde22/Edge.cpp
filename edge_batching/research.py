from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from .benchmark import PolicyFactory, WorkloadSpec, run_benchmark_suite
from .models import DeviceProfile
from .policies import (
    FixedBatchScheduler,
    RealtimeSingleScheduler,
    ThroughputFirstScheduler,
)
from .scheduler import AdaptiveBatchScheduler


@dataclass(slots=True)
class ResearchScenario:
    name: str
    device: DeviceProfile
    workload: WorkloadSpec


def _policy_factories() -> dict[str, PolicyFactory]:
    return {
        "adaptive_hybrid": lambda device, engine, clock: AdaptiveBatchScheduler(
            device=device,
            engine=engine,
            time_source=clock,
        ),
        "fixed_batch_4": lambda device, engine, clock: FixedBatchScheduler(
            device=device,
            engine=engine,
            time_source=clock,
            fixed_batch_size=4,
        ),
        "throughput_first": lambda device, engine, clock: ThroughputFirstScheduler(
            device=device,
            engine=engine,
            time_source=clock,
        ),
        "single_request": lambda device, engine, clock: RealtimeSingleScheduler(
            device=device,
            engine=engine,
            time_source=clock,
        ),
    }


def _scenarios() -> list[ResearchScenario]:
    jetson_like = DeviceProfile(
        name="jetson-like",
        memory_gb=8.0,
        compute_score=4.0,
        max_batch_size=16,
        target_realtime_latency_ms=220.0,
        max_queue_wait_ms=900.0,
        background_weight=0.30,
    )
    small_cpu = DeviceProfile(
        name="small-cpu",
        memory_gb=4.0,
        compute_score=1.4,
        max_batch_size=8,
        target_realtime_latency_ms=260.0,
        max_queue_wait_ms=700.0,
        background_weight=0.25,
    )
    return [
        ResearchScenario(
            name="balanced_load",
            device=jetson_like,
            workload=WorkloadSpec(
                duration_s=120.0, realtime_rps=2.2, background_rps=0.9
            ),
        ),
        ResearchScenario(
            name="chat_spike",
            device=jetson_like,
            workload=WorkloadSpec(
                duration_s=120.0, realtime_rps=3.4, background_rps=0.5
            ),
        ),
        ResearchScenario(
            name="background_heavy",
            device=small_cpu,
            workload=WorkloadSpec(
                duration_s=120.0, realtime_rps=1.4, background_rps=1.8
            ),
        ),
    ]


def _write_csv(
    path: Path,
    rows: list[dict[str, str | float | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_delta(value: float, reference: float, lower_is_better: bool) -> str:
    if reference == 0:
        return "n/a"
    delta = (value - reference) / abs(reference) * 100.0
    if lower_is_better:
        # lower is better: negative delta is improvement
        return f"{delta:+.1f}%"
    return f"{delta:+.1f}%"


def _build_markdown_report(
    scenario_results: dict[str, list],
) -> str:
    lines: list[str] = []
    lines.append("# Edge Dynamic Batching Research Results")
    lines.append("")
    lines.append(
        "This report compares adaptive hybrid batching against fixed and baseline schedulers "
        "for mixed realtime/background edge workloads."
    )
    lines.append("")

    for scenario_name, results in scenario_results.items():
        lines.append(f"## Scenario: {scenario_name}")
        lines.append("")
        lines.append(
            "| Policy | RT p95 (ms) | BG p95 (ms) | Throughput (req/s) | Gen tok/s | Avg batch |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for result in results:
            m = result.metrics_mean
            lines.append(
                "| %s | %.1f | %.1f | %.2f | %.1f | %.2f |"
                % (
                    result.policy_name,
                    m["realtime_latency_p95_ms"],
                    m["background_latency_p95_ms"],
                    m["throughput_rps"],
                    m["generated_tokens_per_second"],
                    m["avg_batch_size"],
                )
            )
        lines.append("")

        by_name = {item.policy_name: item for item in results}
        adaptive = by_name.get("adaptive_hybrid")
        if adaptive:
            a = adaptive.metrics_mean
            lines.append("Adaptive comparison vs baselines:")
            for other_name in ("fixed_batch_4", "throughput_first", "single_request"):
                if other_name not in by_name:
                    continue
                b = by_name[other_name].metrics_mean
                rt_delta = _format_delta(
                    a["realtime_latency_p95_ms"],
                    b["realtime_latency_p95_ms"],
                    lower_is_better=True,
                )
                tp_delta = _format_delta(
                    a["throughput_rps"],
                    b["throughput_rps"],
                    lower_is_better=False,
                )
                lines.append(
                    "- vs `%s`: RT p95 %s, throughput %s"
                    % (other_name, rt_delta, tp_delta)
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_research(output_dir: Path, seeds: list[int]) -> tuple[Path, Path]:
    policies = _policy_factories()
    scenarios = _scenarios()
    flat_rows: list[dict[str, str | float | int]] = []
    scenario_results = {}

    for scenario in scenarios:
        results = run_benchmark_suite(
            device=scenario.device,
            spec=scenario.workload,
            policies=policies,
            seeds=seeds,
        )
        scenario_results[scenario.name] = results
        for result in results:
            row: dict[str, str | float | int] = {
                "scenario": scenario.name,
                "policy": result.policy_name,
                "seeds": ",".join(str(seed) for seed in result.seeds),
                "request_count": result.request_count,
                "realtime_count": result.realtime_count,
                "background_count": result.background_count,
            }
            for key, value in result.metrics_mean.items():
                row[f"{key}_mean"] = value
            for key, value in result.metrics_std.items():
                row[f"{key}_std"] = value
            flat_rows.append(row)

    csv_path = output_dir / "benchmark_results.csv"
    report_path = output_dir / "findings.md"
    _write_csv(csv_path, flat_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_build_markdown_report(scenario_results), encoding="utf-8")
    return csv_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run edge batching research experiments and emit report artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_outputs"),
        help="Directory where CSV and markdown findings are written.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="101,202,303",
        help="Comma-separated random seeds for repeated trials.",
    )
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    csv_path, report_path = run_research(args.output_dir, seeds)
    print(f"wrote_csv={csv_path}")
    print(f"wrote_report={report_path}")


if __name__ == "__main__":
    main()
