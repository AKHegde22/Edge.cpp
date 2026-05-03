from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Type

from .benchmark import PolicyBenchmarkResult, WorkloadSpec, run_benchmark_suite
from .engine import LlamaCppIterationEngine
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


def _policy_classes() -> Dict[str, Type[AdaptiveBatchScheduler]]:
    return {
        "adaptive_hybrid": AdaptiveBatchScheduler,
        "fixed_batch_4": FixedBatchScheduler,
        "throughput_first": ThroughputFirstScheduler,
        "single_request": RealtimeSingleScheduler,
    }


def _scenarios() -> List[ResearchScenario]:
    # Real-world device profiles
    macbook_m2 = DeviceProfile(
        name="macbook-m2",
        memory_gb=16.0,
        compute_score=5.0,
        max_batch_size=8,
        target_realtime_latency_ms=250.0,
        background_weight=0.30,
    )
    
    return [
        ResearchScenario(
            name="balanced_load",
            device=macbook_m2,
            workload=WorkloadSpec(
                duration_s=60.0, realtime_rps=1.0, background_rps=0.5
            ),
        ),
        ResearchScenario(
            name="realtime_spike",
            device=macbook_m2,
            workload=WorkloadSpec(
                duration_s=60.0, realtime_rps=2.0, background_rps=0.2
            ),
        ),
    ]


def _write_csv(
    path: Path,
    rows: List[Dict[str, str | float | int]],
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


def _build_markdown_report(
    scenario_results: Dict[str, List[PolicyBenchmarkResult]],
) -> str:
    lines = ["# Edge Dynamic Batching — Real Inference Report", ""]
    lines.append("Comparison of adaptive scheduling vs baselines using real model execution.")
    lines.append("")

    for scenario_name, results in scenario_results.items():
        lines.append(f"## Scenario: {scenario_name}")
        lines.append("")
        lines.append("| Policy | RT p95 (ms) | BG p95 (ms) | Throughput (req/s) | Gen tok/s |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in results:
            m = r.metrics_mean
            lines.append(
                f"| {r.policy_name} | {m['realtime_latency_p95_ms']:.1f} | {m['background_latency_p95_ms']:.1f} | {m['throughput_rps']:.2f} | {m['generated_tokens_per_second']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_research(model_path: str, output_dir: Path, seeds: List[int]) -> Tuple[Path, Path]:
    engine = LlamaCppIterationEngine(model_path=model_path)
    policies = _policy_classes()
    scenarios = _scenarios()
    
    flat_rows = []
    scenario_results = {}

    for scenario in scenarios:
        print(f"\nRunning Research Scenario: {scenario.name}")
        results = run_benchmark_suite(
            engine=engine,
            device=scenario.device,
            spec=scenario.workload,
            policies=policies,
            seeds=seeds,
        )
        scenario_results[scenario.name] = results
        for result in results:
            row = {
                "scenario": scenario.name,
                "policy": result.policy_name,
                "seeds": ",".join(str(s) for s in result.seeds),
            }
            row.update({f"{k}_mean": v for k, v in result.metrics_mean.items()})
            flat_rows.append(row)

    csv_path = output_dir / "real_benchmark_results.csv"
    report_path = output_dir / "real_findings.md"
    _write_csv(csv_path, flat_rows)
    report_path.write_text(_build_markdown_report(scenario_results), encoding="utf-8")
    return csv_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-inference edge batching research.")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model")
    parser.add_argument("--output-dir", type=Path, default=Path("research_outputs"))
    parser.add_argument("--seeds", type=str, default="101")
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    csv_path, report_path = run_research(args.model, args.output_dir, seeds)
    print(f"\nResults written to:\n  CSV: {csv_path}\n  MD:  {report_path}")


if __name__ == "__main__":
    main()
