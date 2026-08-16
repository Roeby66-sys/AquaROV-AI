"""
AquaROV AI — Recording Manager.

Framework-independent recording lifecycle manager.

The manager controls recording sessions and uses the shared
RecordingState DTO defined in the core DTO layer.

Actual video encoding is delegated to the camera/video backend.
Runtime recording files must remain outside the Git repository.
"""

from __future__ import annotations

from dataclasses import replace
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
        self._started_at: float | None = None

    @property
    def recording_dir(self) -> Path:
        """Return the runtime recording directory."""
        return self._recording_dir

    @property
    def state(self) -> RecordingState:
        """Return the current recording state."""
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
        Start a new recording session.

        The manager prepares the destination path and updates the shared
        RecordingState. Actual video encoding is handled elsewhere.
        """
        with self._lock:
            if self._state.is_recording:
                raise RuntimeError("A recording is already active.")

            if not camera_id.strip():
                raise ValueError("camera_id must not be empty.")

            self._recording_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            started_at = time()

            if filename is None:
                filename = (
                    f"camera_{camera_id}_{int(started_at)}.mp4"
                )

            safe_filename = Path(filename).name

            if not safe_filename:
                raise ValueError("filename must not be empty.")

            file_path = self._recording_dir / safe_filename

            self._started_at = started_at

            self._state = RecordingState(
                is_recording=True,
                filename=safe_filename,
                duration=0.0,
                storage_path=str(file_path),
                camera_id=camera_id,
                status="recording",
                file_size_mb=0.0,
            )

            return self._state

    def update(self) -> RecordingState:
        """
        Update recording duration and current file size.

        Actual video encoding remains the responsibility of the backend.
        """
        with self._lock:
            if not self._state.is_recording:
                return self._state

            duration = self._state.duration

            if self._started_at is not None:
                duration = max(0.0, time() - self._started_at)

            file_size_mb = self._state.file_size_mb
            storage_path = self._state.storage_path

            if storage_path:
                file_path = Path(storage_path)

                try:
                    if file_path.is_file():
                        file_size_mb = (
                            file_path.stat().st_size / (1024 * 1024)
                        )
                except OSError:
                    pass

            self._state = replace(
                self._state,
                duration=duration,
                file_size_mb=file_size_mb,
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

            self._update_state_locked()

            self._state = replace(
                self._state,
                is_recording=False,
                status="stopped",
            )

            self._started_at = None

            return self._state

    def cancel(self) -> RecordingState:
        """
        Cancel the active recording session.

        This clears the active recording state without deleting any
        partially created media file.
        """
        with self._lock:
            if not self._state.is_recording:
                raise RuntimeError("No recording is currently active.")

            self._state = replace(
                self._state,
                is_recording=False,
                status="cancelled",
            )

            self._started_at = None

            return self._state

    def _update_state_locked(self) -> None:
        """Update duration and file size while the lock is held."""
        duration = self._state.duration

        if self._started_at is not None:
            duration = max(0.0, time() - self._started_at)

        file_size_mb = self._state.file_size_mb
        storage_path = self._state.storage_path

        if storage_path:
            file_path = Path(storage_path)

            try:
                if file_path.is_file():
                    file_size_mb = (
                        file_path.stat().st_size / (1024 * 1024)
                    )
            except OSError:
                pass

        self._state = replace(
            self._state,
            duration=duration,
            file_size_mb=file_size_mb,
        )


__all__ = [
    "RecordingManager",
]
