# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] — 2026-05-03

### Added
- **Continuous Batching engine** (`LlamaCppIterationEngine`) — iteration-level token decoding using the llama.cpp low-level C API via `llama-cpp-python`. New requests are injected between decode steps rather than waiting for a batch to complete.
- **In-Flight Preemption** in `AdaptiveBatchScheduler` — realtime requests can displace background tasks from the active batch at the next token boundary.
- **Power-Aware Thermal Scheduling** — `HardwareMonitor` (cross-platform: macOS, Linux, Windows) feeds live thermal pressure and battery level into `DeviceProfile`; the scheduler throttles background admission under thermal stress and low battery.
- `DeviceProfile` extended with `thermal_state`, `battery_level`, and `is_low_power_mode` fields.
- `GenerationRequest` extended with per-request state fields required for stateful iteration-level decoding (`tokens_generated`, `token_ids`, `kv_cache_seq_id`, etc.).
- `pyproject.toml` for standard pip installation.

### Changed
- `EdgeBatchingService` service loop switched from batch-level `run_once()` to token-level `run_step()`.
- `AdaptiveBatchScheduler` refactored to maintain an `_active_batch` across steps.
- Cleaned up repository — removed stale scripts, renamed result document to `RESULTS.md`.

### Removed
- `LlamaCppEngineAdapter` (batch-level adapter) — replaced by `LlamaCppIterationEngine`.
- `RESEARCH_NOTES.md` — superseded by `RESULTS.md`.
