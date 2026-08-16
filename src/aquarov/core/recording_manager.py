"""
AquaROV AI — Recording Manager.

Framework-independent recording lifecycle manager.

The manager controls recording sessions and uses the shared
RecordingState DTO defined in the core DTO layer.

Actual video encoding is delegated to the camera/video backend.
Runtime recording files must remain outside the Git repository.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import time

from aquarov.core.dto import RecordingState


class RecordingManager:
    """Manage recording sessions independently from the video backend."""

    def __init__(self, recording_dir: str | Path) -> None:
        self._recording_dir = Path(recording_dir)
        self._lock = Lock()
        self._state = RecordingState()

    @property
    def recording_dir(self) -> Path:
        """Return the runtime recording directory."""
        return self._recording_dir

    @property
    def state(self) -> RecordingState:
        """Return the current shared recording state."""
        with self._lock:
            return self._state

    @property
    def is_recording(self) -> bool:
        """Return True when a recording session is active."""
        with self._lock:
            return self._state.is_recording

    def start(
        self,
        camera_id: str,
        filename: str | None = None,
    ) -> RecordingState:
        """
        Start a recording session.

        This method prepares the destination path and updates the shared
        RecordingState. Actual video encoding is handled elsewhere.
        """
        with self._lock:
            if self._state.is_recording:
                raise RuntimeError("A recording is already active.")

            self._recording_dir.mkdir(parents=True, exist_ok=True)

            started_at = time()

            if filename is None:
                filename = f"camera_{camera_id}_{int(started_at)}.mp4"

            file_path = self._recording_dir / filename

            try:
                file_path.resolve().relative_to(
                    self._recording_dir.resolve()
                )
            except ValueError as exc:
                raise ValueError(
                    "Recording filename must remain inside "
                    "the recording directory."
                ) from exc

            self._state = RecordingState(
                is_recording=True,
                filename=filename,
                duration=0.0,
                storage_path=str(file_path),
                camera_id=camera_id,
                status="recording",
                file_size_mb=0.0,
            )

            return self._state

    def stop(self) -> RecordingState:
        """
        Stop the active recording session.

        The manager updates lifecycle state only. Actual media finalization
        remains the responsibility of the recording backend.
        """
        with self._lock:
            if not self._state.is_recording:
                raise RuntimeError("No recording is currently active.")

            self._state = RecordingState(
                is_recording=False,
                filename=self._state.filename,
                duration=self._state.duration,
                storage_path=self._state.storage_path,
                camera_id=self._state.camera_id,
                status="stopped",
                file_size_mb=self._state.file_size_mb,
            )

            return self._state

    def cancel(self) -> RecordingState:
        """
        Cancel the active recording session.

        This clears the active state without deleting any partially
        created media file.
        """
        with self._lock:
            if not self._state.is_recording:
                raise RuntimeError("No recording is currently active.")

            self._state = RecordingState(
                is_recording=False,
                filename=self._state.filename,
                duration=self._state.duration,
                storage_path=self._state.storage_path,
                camera_id=self._state.camera_id,
                status="stopped",
                file_size_mb=self._state.file_size_mb,
            )

            return self._state
