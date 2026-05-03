# Adaptive Edge Inference — Continuous Batching with Thermal-Aware Scheduling

> A production-ready research framework for low-latency LLM inference on edge devices, featuring iteration-level continuous batching, in-flight request preemption, and power-aware thermal scheduling over llama.cpp.

---

## Highlights

| Capability | Description |
| :--- | :--- |
| **Continuous Batching** | Token-level iteration loop via the llama.cpp low-level API — new requests are injected between decode steps, not between batches. |
| **In-Flight Preemption** | Realtime (interactive) requests instantly preempt background tasks without waiting for a batch to finish. |
| **Power-Aware Thermal Scheduling** | A hardware monitor feeds thermal pressure and battery level into the scheduler, which throttles background work under thermal stress. |
| **Real Baseline Suite** | Built-in Fixed-Batch, Throughput-First, and Single-Request baselines for empirical evaluation. |

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
│  │  plugged   │   │  │  Queue  │  │  Queue   │  │   │
│  └───────────┘   │  └────┬────┘  └────┬─────┘  │   │
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
├── engine.py              # Production engine — LlamaCppIterationEngine (100% Real)
├── scheduler.py           # Core scheduler — continuous batching + preemption logic
├── service.py             # Async service layer — Future-based API + hardware monitor
├── hardware_monitor.py    # Cross-platform thermal pressure & battery monitoring
├── policies.py            # Real baseline schedulers — Fixed, Throughput-First, Single
├── benchmark.py           # Real-time workload generator + policy comparison framework
├── research.py            # Empirical experiment runner with CSV/Markdown output
└── simulation.py          # Interactive CLI simulation for live validation

download_model.py          # Downloads Qwen-0.5B GGUF for local testing
RESULTS.md                 # Empirical results with raw metrics
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

### 3. Run the Empirical Research Suite

```bash
python -m edge_batching.research --model models/qwen1_5-0_5b-chat-q4_k_m.gguf
```

This generates:
- `research_outputs/real_benchmark_results.csv` — raw metrics from real inference
- `research_outputs/real_findings.md` — formatted comparison tables

### 4. Run Interactive Simulation

```bash
python -m edge_batching.simulation --model models/qwen1_5-0_5b-chat-q4_k_m.gguf
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

# 1. Initialize real engine
engine = LlamaCppIterationEngine(model_path="models/qwen1_5-0_5b-chat-q4_k_m.gguf")

# 2. Define hardware profile
device = DeviceProfile(
    name="macbook-m2",
    memory_gb=16.0,
    compute_score=5.0,
    max_batch_size=4,
    target_realtime_latency_ms=250.0,
    background_weight=0.30,
)

# 3. Setup scheduler and service
scheduler = AdaptiveBatchScheduler(device=device, engine=engine)
service = EdgeBatchingService(scheduler)
service.start()

# 4. Submit a realtime request
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

## License

Apache 2.0 — see [LICENSE](LICENSE).
