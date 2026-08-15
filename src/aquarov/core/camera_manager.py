"""AquaROV AI - Camera Manager.

Framework-independent camera abstraction.

The manager handles camera channels, frame acquisition, and frame delivery
to the inference pipeline. Hardware-specific camera implementations can be
plugged in later without changing the rest of the application.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import time
from typing import Any, Callable, Optional, Protocol

from aquarov.core.dto import CameraChannel
from aquarov.core.inference_worker import InferenceFrame, InferenceWorker


class CameraBackend(Protocol):
    """Interface for a physical or virtual camera source."""

    def open(self) -> None:
        """Open the camera source."""
        ...

    def close(self) -> None:
        """Close the camera source."""
        ...

    def read(self) -> Any:
        """Read and return the next frame."""
        ...


@dataclass
class CameraState:
    """Runtime state for one camera channel."""

    channel: CameraChannel
    connected: bool = False
    frame_count: int = 0
    last_timestamp: float = 0.0
    last_error: Optional[str] = None


FrameCallback = Callable[[InferenceFrame], None]
ErrorCallback = Callable[[str, Exception], None]


class CameraManager:
    """Manage multiple camera channels.

    This class intentionally does not depend on OpenCV, Qt, GStreamer, or
    a specific ROV camera SDK. Those integrations can be implemented as
    CameraBackend classes later.
    """

    def __init__(
        self,
        inference_worker: Optional[InferenceWorker] = None,
        *,
        on_frame: Optional[FrameCallback] = None,
        on_error: Optional[ErrorCallback] = None,
    ) -> None:
        self._inference_worker = inference_worker
        self._on_frame = on_frame
        self._on_error = on_error

        self._channels: dict[str, CameraState] = {}
        self._backends: dict[str, CameraBackend] = {}
        self._lock = Lock()

    def register_camera(
        self,
        channel: CameraChannel,
        backend: CameraBackend,
    ) -> None:
        """Register a camera channel and its backend."""
        camera_id = channel.id

        with self._lock:
            if camera_id in self._channels:
                raise ValueError(f"Camera already registered: {camera_id}")

            self._channels[camera_id] = CameraState(channel=channel)
            self._backends[camera_id] = backend

    def unregister_camera(self, camera_id: str) -> None:
        """Close and remove a registered camera."""
        self.close_camera(camera_id)

        with self._lock:
            self._backends.pop(camera_id, None)
            self._channels.pop(camera_id, None)

    def open_camera(self, camera_id: str) -> None:
        """Open a registered camera."""
        backend = self._get_backend(camera_id)

        try:
            backend.open()
            with self._lock:
                state = self._channels[camera_id]
                state.connected = True
                state.last_error = None
        except Exception as exc:
            self._set_error(camera_id, exc)
            raise

    def close_camera(self, camera_id: str) -> None:
        """Close a registered camera."""
        backend = self._get_backend(camera_id)

        try:
            backend.close()
        finally:
            with self._lock:
                state = self._channels.get(camera_id)
                if state is not None:
                    state.connected = False

    def read_frame(self, camera_id: str) -> Optional[InferenceFrame]:
        """Read one frame and optionally submit it to the inference worker."""
        backend = self._get_backend(camera_id)

        with self._lock:
            state = self._channels[camera_id]
            if not state.connected:
                raise RuntimeError(f"Camera is not connected: {camera_id}")

            frame_id = state.frame_count
            state.frame_count += 1

        try:
            image = backend.read()
            timestamp = time()

            frame = InferenceFrame(
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp=timestamp,
                image=image,
                metadata={
                    "camera_name": state.channel.name,
                    "source_type": state.channel.source_type.value,
                },
            )

            with self._lock:
                state.last_timestamp = timestamp
                state.last_error = None

            if self._inference_worker is not None:
                self._inference_worker.submit(frame)

            if self._on_frame is not None:
                self._on_frame(frame)

            return frame

        except Exception as exc:
            self._set_error(camera_id, exc)

            if self._on_error is not None:
                self._on_error(camera_id, exc)

            return None

    def get_state(self, camera_id: str) -> CameraState:
        """Return a snapshot of the current camera state."""
        with self._lock:
            state = self._channels.get(camera_id)
            if state is None:
                raise KeyError(f"Unknown camera: {camera_id}")

            return CameraState(
                channel=state.channel,
                connected=state.connected,
                frame_count=state.frame_count,
                last_timestamp=state.last_timestamp,
                last_error=state.last_error,
            )

    def list_cameras(self) -> list[CameraChannel]:
        """Return all registered camera channels."""
        with self._lock:
            return [state.channel for state in self._channels.values()]

    def connected_cameras(self) -> list[str]:
        """Return IDs of currently connected cameras."""
        with self._lock:
            return [
                camera_id
                for camera_id, state in self._channels.items()
                if state.connected
            ]

    def _get_backend(self, camera_id: str) -> CameraBackend:
        with self._lock:
            backend = self._backends.get(camera_id)

        if backend is None:
            raise KeyError(f"Unknown camera: {camera_id}")

        return backend

    def _set_error(self, camera_id: str, exc: Exception) -> None:
        with self._lock:
            state = self._channels.get(camera_id)
            if state is not None:
                state.last_error = str(exc)


class MemoryCameraBackend:
    """Simple in-memory camera backend for development and testing.

    A real ROV deployment can replace this with an RTSP, USB, CSI, GStreamer,
    or vendor-specific camera backend.
    """

    def __init__(self) -> None:
        self._opened = False
        self._frames: list[Any] = []
        self._index = 0

    def open(self) -> None:
        self._opened = True
        self._index = 0

    def close(self) -> None:
        self._opened = False

    def add_frame(self, frame: Any) -> None:
        """Add a test frame to the backend."""
        self._frames.append(frame)

    def read(self) -> Any:
        if not self._opened:
            raise RuntimeError("Memory camera is not open")

        if not self._frames:
            raise RuntimeError("No frames available")

        frame = self._frames[self._index % len(self._frames)]
        self._index += 1
        return frame


__all__ = [
    "CameraBackend",
    "CameraState",
    "CameraManager",
    "MemoryCameraBackend",
]
