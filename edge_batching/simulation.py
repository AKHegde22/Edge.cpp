from __future__ import annotations

import argparse
import random
import time

from .engine import LlamaCppIterationEngine
from .models import DeviceProfile, GenerationRequest, WorkloadType
from .scheduler import AdaptiveBatchScheduler
from .service import EdgeBatchingService


def run_simulation(
    model_path: str,
    rounds: int,
    realtime_per_round: int,
    background_per_round: int,
    sleep_between_rounds_s: float,
) -> None:
    print(f"Initializing real-time simulation with model: {model_path}")
    
    device = DeviceProfile(
        name="local-device",
        memory_gb=16.0,
        compute_score=5.0,
        max_batch_size=8,
        target_realtime_latency_ms=250.0,
        background_weight=0.30,
    )
    
    engine = LlamaCppIterationEngine(model_path=model_path)
    scheduler = AdaptiveBatchScheduler(device=device, engine=engine)
    service = EdgeBatchingService(scheduler)
    service.start()

    req_index = 0
    try:
        for round_idx in range(1, rounds + 1):
            print(f"\n--- Round {round_idx} ---")
            
            # Inject realtime requests
            for _ in range(realtime_per_round):
                req = GenerationRequest(
                    request_id=f"rt-{req_index}",
                    prompt="Summarize the benefits of edge computing.",
                    prompt_tokens=32,
                    max_new_tokens=random.randint(32, 64),
                    workload=WorkloadType.REALTIME,
                )
                service.submit(req)
                print(f"  Submitted Realtime: {req.request_id}")
                req_index += 1

            # Inject background requests
            for _ in range(background_per_round):
                req = GenerationRequest(
                    request_id=f"bg-{req_index}",
                    prompt="Explain the difference between Llama 3 and Mistral.",
                    prompt_tokens=32,
                    max_new_tokens=random.randint(64, 128),
                    workload=WorkloadType.BACKGROUND,
                )
                service.submit(req)
                print(f"  Submitted Background: {req.request_id}")
                req_index += 1

            # Monitor progress
            time.sleep(sleep_between_rounds_s)
            snap = scheduler.snapshot()
            print(f"  Status: RT_Queued={snap.queued_realtime} BG_Queued={snap.queued_background} TPS={snap.avg_tokens_per_second:.1f}")

        print("\nAll rounds submitted. Draining remaining requests...")
        while scheduler.pending() > 0:
            snap = scheduler.snapshot()
            print(f"  Draining: Pending={scheduler.pending()} TPS={snap.avg_tokens_per_second:.1f}", end="\r")
            time.sleep(1.0)
            
    finally:
        service.stop()
        print("\nSimulation complete.")

    snap = scheduler.snapshot()
    print("\nFinal Statistics:")
    print(f"  Completed Requests: {snap.completed_requests}")
    print(f"  Completed Batches:  {snap.completed_batches}")
    print(f"  Avg Batch Latency:  {snap.avg_batch_runtime_ms:.2f} ms")
    print(f"  Avg Throughput:     {snap.avg_tokens_per_second:.2f} tokens/s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate adaptive edge batching behavior with a real LLM model."
    )
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model file")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--realtime-per-round", type=int, default=1)
    parser.add_argument("--background-per-round", type=int, default=1)
    parser.add_argument("--sleep-between-rounds", type=float, default=5.0)
    args = parser.parse_args()

    run_simulation(
        model_path=args.model,
        rounds=args.rounds,
        realtime_per_round=args.realtime_per_round,
        background_per_round=args.background_per_round,
        sleep_between_rounds_s=args.sleep_between_rounds,
    )


if __name__ == "__main__":
    main()
