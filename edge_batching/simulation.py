from __future__ import annotations

import argparse
import random
import time

from .engine import MockLlamaCppEngine
from .models import DeviceProfile, GenerationRequest, WorkloadType
from .scheduler import AdaptiveBatchScheduler


def run_simulation(
    rounds: int,
    realtime_per_round: int,
    background_per_round: int,
    sleep_between_rounds_s: float,
) -> None:
    device = DeviceProfile(
        name="jetson-like",
        memory_gb=8.0,
        compute_score=4.0,
        max_batch_size=16,
        target_realtime_latency_ms=220.0,
        background_weight=0.30,
    )
    engine = MockLlamaCppEngine(sleep_for_runtime=True)
    scheduler = AdaptiveBatchScheduler(device=device, engine=engine)

    req_index = 0
    for round_idx in range(1, rounds + 1):
        for _ in range(realtime_per_round):
            req = GenerationRequest(
                request_id=f"rt-{req_index}",
                prompt="User chat request",
                prompt_tokens=random.randint(24, 96),
                max_new_tokens=random.randint(32, 128),
                workload=WorkloadType.REALTIME,
            )
            scheduler.submit(req)
            req_index += 1

        for _ in range(background_per_round):
            req = GenerationRequest(
                request_id=f"bg-{req_index}",
                prompt="Background summarization task",
                prompt_tokens=random.randint(64, 192),
                max_new_tokens=random.randint(96, 256),
                workload=WorkloadType.BACKGROUND,
            )
            scheduler.submit(req)
            req_index += 1

        runs = scheduler.drain(max_batches=3)
        if not runs:
            print(f"round={round_idx} no work")
        else:
            last = runs[-1]
            realtime_count = sum(
                1 for item in last.requests if item.workload == WorkloadType.REALTIME
            )
            background_count = len(last.requests) - realtime_count
            print(
                "round=%d batches=%d last_limit=%d batch_size=%d realtime=%d background=%d "
                "runtime_ms=%.2f"
                % (
                    round_idx,
                    len(runs),
                    last.batch_limit,
                    last.metrics.batch_size,
                    realtime_count,
                    background_count,
                    last.metrics.runtime_ms,
                )
            )
        time.sleep(max(0.0, sleep_between_rounds_s))

    remaining_runs = scheduler.drain()
    if remaining_runs:
        print(f"drained_remaining_batches={len(remaining_runs)}")

    snap = scheduler.snapshot()
    print(
        "completed_requests=%d completed_batches=%d avg_batch_ms=%.2f avg_tps=%.2f"
        % (
            snap.completed_requests,
            snap.completed_batches,
            snap.avg_batch_runtime_ms,
            snap.avg_tokens_per_second,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate adaptive edge batching behavior under mixed workloads."
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--realtime-per-round", type=int, default=2)
    parser.add_argument("--background-per-round", type=int, default=1)
    parser.add_argument("--sleep-between-rounds", type=float, default=0.05)
    args = parser.parse_args()

    run_simulation(
        rounds=args.rounds,
        realtime_per_round=args.realtime_per_round,
        background_per_round=args.background_per_round,
        sleep_between_rounds_s=args.sleep_between_rounds,
    )


if __name__ == "__main__":
    main()
