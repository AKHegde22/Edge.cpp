# Adaptive Edge Inference — Continuous Batching with Thermal-Aware Scheduling

> A research framework for low-latency LLM inference on edge devices, featuring iteration-level continuous batching, in-flight request preemption, and power-aware thermal scheduling over llama.cpp.

---

## Highlights

| Capability | Description |
| :--- | :--- |
| **Continuous Batching** | Token-level iteration loop via the llama.cpp low-level API — new requests are injected between decode steps, not between batches. |
| **In-Flight Preemption** | Realtime (interactive) requests instantly preempt background tasks without waiting for a batch to finish. |
| **Power-Aware Thermal Scheduling** | A hardware monitor feeds thermal pressure and battery level into the scheduler, which throttles background work under thermal stress. |
| **Baseline Comparison Suite** | Built-in FIFO, Fixed-Batch, Throughput-First, and Single-Request baselines for controlled evaluation. |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                EdgeBatchingService                   │
│  ┌───────────┐   ┌──────────────────────────────┐   │
│  │  Hardware  │──▶│   AdaptiveBatchScheduler      │   │
│  │  Monitor   │   │                              │   │
│  │ (thermal,  │   │  ┌─────────┐  ┌──────────┐  │   │
│  │  battery)  │   │  │Realtime │  │Background│  │   │
│  └───────────┘   │  │  Queue  │  │  Queue   │  │   │
│                  │  └────┬────┘  └────┬─────┘  │   │
│                  │       │   Preemption│        │   │
│                  │       ▼            ▼        │   │
│                  │  ┌──────────────────────┐   │   │
│                  │  │   Active Batch       │   │   │
│                  │  │  (token-level step)  │   │   │
│                  │  └──────────┬───────────┘   │   │
│                  └─────────────┼───────────────┘   │
│                               ▼                    │
│                  ┌─────────────────────────┐       │
│                  │ LlamaCppIterationEngine  │       │
│                  │  (llama_decode per token) │       │
│                  └─────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
edge_batching/
├── models.py              # Data models — GenerationRequest, DeviceProfile, WorkloadType
├── engine.py              # Inference engines — LlamaCppIterationEngine, MockEngine
├── scheduler.py           # Core scheduler — continuous batching + preemption logic
├── service.py             # Async service layer — Future-based API + hardware monitor
├── hardware_monitor.py    # Cross-platform thermal pressure & battery monitoring (macOS/Linux/Windows)
├── policies.py            # Baseline schedulers — Fixed, Throughput-First, Single
├── benchmark.py           # Synthetic workload generator + policy comparison framework
├── research.py            # Multi-seed experiment runner with CSV/Markdown output
├── simulation.py          # Interactive CLI simulation for quick validation
└── tuning.py              # Grid-search parameter tuner for scheduler knobs

download_model.py          # Downloads Qwen-0.5B GGUF for local testing
RESULTS.md                 # Full empirical results with raw metrics
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install llama-cpp-python numpy huggingface-hub
```

### 2. Download the Test Model

```bash
python download_model.py
```

### 3. Run the Synthetic Benchmark Suite

```bash
python -m edge_batching.research --output-dir research_outputs --seeds 101,202,303
```

This generates:
- `research_outputs/benchmark_results.csv` — raw metrics per policy per seed
- `research_outputs/findings.md` — formatted comparison tables

### 4. Run the Parameter Tuner

```bash
python -m edge_batching.tuning --output-csv research_outputs/adaptive_tuning.csv
```

---

## Integration Example

```python
from edge_batching import (
    AdaptiveBatchScheduler,
    DeviceProfile,
    EdgeBatchingService,
    GenerationRequest,
    LlamaCppIterationEngine,
    WorkloadType,
)

engine = LlamaCppIterationEngine(model_path="models/qwen1_5-0_5b-chat-q4_k_m.gguf")

device = DeviceProfile(
    name="macbook-m2",
    memory_gb=16.0,
    compute_score=5.0,
    max_batch_size=4,
    target_realtime_latency_ms=250.0,
    background_weight=0.30,
)

scheduler = AdaptiveBatchScheduler(device=device, engine=engine)
service = EdgeBatchingService(scheduler)
service.start()

# Submit a realtime request — gets priority over any background work
future = service.submit(
    GenerationRequest(
        request_id="chat-1",
        prompt="What is the capital of France?",
        prompt_tokens=8,
        max_new_tokens=32,
        workload=WorkloadType.REALTIME,
    )
)

result = future.result(timeout=10.0)
print(result.output_text)
service.stop()
```

---

## Key Research Results

Full results with raw metrics are in [`RESULTS.md`](RESULTS.md).

| Experiment | Key Finding |
| :--- | :--- |
| **Adaptive vs. FIFO** | 2.1× faster realtime latency under 10-task saturation (40s vs 84s) |
| **Llama-cpp vs. MLX** | 38% higher throughput at saturation; 17% faster interruption response |
| **SLA Sweep** | Adaptive scheduler meets tighter SLA targets than MLX across all thresholds |
| **Continuous Batching** | TTFT drops from ~10s (batch-level) to <100ms (iteration-level) |
| **Thermal Scheduling** | Background tasks auto-throttled under thermal pressure; battery-aware mode |

---

## Tuning Knobs

The `DeviceProfile` is the primary control surface:

| Parameter | Effect |
| :--- | :--- |
| `target_realtime_latency_ms` | Lower → more aggressive latency protection |
| `background_weight` | Higher → more throughput share for background tasks |
| `max_queue_wait_ms` | Lower → earlier anti-starvation trigger |
| `max_batch_size` | Hardware safety limit for concurrent sequences |
| `thermal_state` | Auto-updated by hardware monitor; affects batch admission |
| `battery_level` | Auto-updated; triggers low-power mode below 20% |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
