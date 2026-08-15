"""
AquaROV AI - Inference Worker

Dependency-free inference worker interface.

This module provides the threading and message-flow foundation for the
future Axelera Metis inference backend. The actual accelerator-specific
implementation can be connected later without changing the DTO layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, Callable, Optional, Protocol, Sequence

from aquarov.core.dto import Detection


@dataclass
class InferenceFrame:
    """A frame submitted to the inference pipeline."""

    camera_id: str
    frame_id: int
    timestamp: float
    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result returned by an inference backend."""

    camera_id: str
    frame_id: int
    timestamp: float
    detections: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceBackend(Protocol):
    """Interface implemented by an AI inference backend."""

    def infer(self, frame: InferenceFrame) -> InferenceResult:
        """Run inference for one frame."""
        ...

    def start(self) -> None:
        """Initialize the backend."""
        ...

    def stop(self) -> None:
        """Release backend resources."""
        ...


ResultCallback = Callable[[InferenceResult], None]
ErrorCallback = Callable[[Exception], None]


class InferenceWorker:
    """
    Background worker for asynchronous AI inference.

    The worker is deliberately independent of a GUI framework and of any
    particular accelerator SDK. A future Axelera Metis backend can implement
    InferenceBackend and be injected here.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        queue_size: int = 4,
        on_result: Optional[ResultCallback] = None,
        on_error: Optional[ErrorCallback] = None,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")

        self._backend = backend
        self._queue: Queue[InferenceFrame] = Queue(maxsize=queue_size)
        self._on_result = on_result
        self._on_error = on_error

        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._state_lock = Lock()
        self._running = False

    @property
    def is_running(self) -> bool:
        """Return True when the worker thread is active."""
        with self._state_lock:
            return self._running

    def start(self) -> None:
        """Start the inference backend and worker thread."""
        with self._state_lock:
            if self._running:
                return

            self._backend.start()
            self._stop_event.clear()
            self._thread = Thread(
                target=self._run,
                name="AquaROV-InferenceWorker",
                daemon=True,
            )
            self._running = True
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the worker and release backend resources."""
        with self._state_lock:
            if not self._running:
                return

            self._stop_event.set()
            thread = self._thread

        if thread is not None:
            thread.join(timeout=max(0.0, timeout))

        with self._state_lock:
            self._running = False
            self._thread = None

        self._backend.stop()

    def submit(self, frame: InferenceFrame, timeout: float = 0.0) -> bool:
        """
        Submit a frame for inference.

        Returns False when the queue is full or the worker is not running.
        """
        if not self.is_running:
            return False

        try:
            self._queue.put(frame, timeout=max(0.0, timeout))
            return True
        except Exception:
            return False

    def queue_size(self) -> int:
        """Return the number of frames currently waiting for inference."""
        return self._queue.qsize()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                started = monotonic()
                result = self._backend.infer(frame)

                if result.inference_time_ms <= 0.0:
                    result.inference_time_ms = (monotonic() - started) * 1000.0

                if self._on_result is not None:
                    self._on_result(result)

            except Exception as exc:
                if self._on_error is not None:
                    self._on_error(exc)
            finally:
                self._queue.task_done()


class NullInferenceBackend:
    """
    Safe placeholder backend for development and GUI integration.

    It returns an empty detection list and performs no AI inference.
    Replace it with the Axelera Metis backend when accelerator integration
    is implemented.
    """

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def infer(self, frame: InferenceFrame) -> InferenceResult:
        return InferenceResult(
            camera_id=frame.camera_id,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            detections=[],
            inference_time_ms=0.0,
            metadata={"backend": "null"},
        )


__all__ = [
    "InferenceFrame",
    "InferenceResult",
    "InferenceBackend",
    "InferenceWorker",
    "NullInferenceBackend",
]
