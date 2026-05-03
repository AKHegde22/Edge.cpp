# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] — 2026-05-03

### Changed
- **Production-Ready Rewrite**: Removed all mock/dummy/synthetic components. Every part of the package now operates with real LLM inference.
- `LlamaCppIterationEngine`: Fixed critical bugs in tokenization and detokenization. Added robust KV cache sequence management and cleanup.
- `AdaptiveBatchScheduler`: Simplified core loop, added statistics tracking, and implemented `snapshot()` and `drain()` methods.
- `HardwareMonitor`: Rewritten for cross-platform support (macOS, Linux, Windows) using only standard libraries.
- `benchmark.py`, `research.py`, `simulation.py`: Updated to perform real-world empirical testing against actual models instead of simulations.

### Removed
- `MockLlamaCppEngine`: Deleted fake latency simulator.
- `MlxEngineAdapter`: Removed to focus exclusively on llama.cpp integration.
- `tuning.py`: Removed mock-dependent parameter search tool.

## [1.0.0] — 2026-05-03

### Added
- **Continuous Batching engine** (`LlamaCppIterationEngine`) — iteration-level token decoding using the llama.cpp low-level C API.
- **In-Flight Preemption** in `AdaptiveBatchScheduler` — realtime requests can displace background tasks from the active batch at the next token boundary.
- **Power-Aware Thermal Scheduling** — feeds live thermal pressure and battery level into `DeviceProfile`.
- `pyproject.toml` for standard pip installation.
