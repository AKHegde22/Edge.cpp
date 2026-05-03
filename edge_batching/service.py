from __future__ import annotations

from concurrent.futures import Future
import threading
import time

from .models import GenerationRequest, GenerationResult
from .scheduler import AdaptiveBatchScheduler


class EdgeBatchingService:
    """
    Thin orchestration layer over AdaptiveBatchScheduler.

    - `submit` returns a Future for per-request async handling.
    - Background loop repeatedly runs scheduler batches and resolves futures.
    """

    def __init__(
        self,
        scheduler: AdaptiveBatchScheduler,
        idle_sleep_s: float = 0.001, # Faster loop for continuous batching
    ):
        self.scheduler = scheduler
        self.idle_sleep_s = idle_sleep_s

        self._futures: dict[str, Future[GenerationResult]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        
        # Phase 2: Hardware Monitor (cross-platform)
        from .hardware_monitor import HardwareMonitor
        self._hw_monitor = HardwareMonitor(self._on_hardware_update)

    def _on_hardware_update(self, thermal: str, battery: float):
        self.scheduler.device.thermal_state = thermal
        self.scheduler.device.battery_level = battery
        if battery < 20.0 or thermal in ["serious", "critical"]:
            self.scheduler.device.is_low_power_mode = True
        else:
            self.scheduler.device.is_low_power_mode = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._hw_monitor.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, wait: bool = True) -> None:
        self._stop_event.set()
        self._hw_monitor.stop()
        if wait and self._thread:
            self._thread.join(timeout=2.0)

    def submit(self, request: GenerationRequest) -> Future[GenerationResult]:
        future: Future[GenerationResult] = Future()
        with self._lock:
            self._futures[request.request_id] = future
        self.scheduler.submit(request)
        return future

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            # In Continuous Batching, we run one token step
            finished_results = self.scheduler.run_step()
            
            if not finished_results:
                time.sleep(self.idle_sleep_s)
                continue
                
            for result in finished_results:
                self._resolve(result)

    def _resolve(self, result: GenerationResult) -> None:
        with self._lock:
            future = self._futures.pop(result.request_id, None)
        if not future:
            return
        if not future.done():
            future.set_result(result)

    def _fail_missing_results(self, run) -> None:
        result_ids = {result.request_id for result in run.results}
        missing = [
            req.request_id for req in run.requests if req.request_id not in result_ids
        ]
        if not missing:
            return
        with self._lock:
            for req_id in missing:
                future = self._futures.pop(req_id, None)
                if future and not future.done():
                    future.set_exception(
                        RuntimeError(f"Missing result for request: {req_id}")
                    )
