"""
AquaROV AI — Recording Manager.

Framework-independent recording lifecycle manager.

The manager controls recording sessions and keeps runtime recording
files outside the Git repository.

Project: AquaROV AI - Underwater ROV Inspection System
Target: Axelera Metis + Voyager SDK deployments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import time
from typing import Optional


@dataclass(frozen=True, slots=True)
class RecordingInfo:
    """Information about the active or completed recording."""

    camera_id: str
    file_path: Path
    started_at: float
    ended_at: Optional[float] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Return recording duration when the recording has ended."""
        if self.ended_at is None:
            return None

        return max(0.0, self.ended_at - self.started_at)


class RecordingManager:
    """Manage recording sessions independently from the video backend."""

    def __init__(self, recording_dir: str | Path) -> None:
        self._recording_dir = Path(recording_dir)
        self._lock = Lock()
        self._active_recording: Optional[RecordingInfo] = None

    @property
    def recording_dir(self) -> Path:
        """Return the runtime recording directory."""
        return self._recording_dir

    @property
    def is_recording(self) -> bool:
        """Return True when a recording session is active."""
        with self._lock:
            return self._active_recording is not None

    @property
    def active_recording(self) -> Optional[RecordingInfo]:
        """Return the active recording information, if any."""
        with self._lock:
            return self._active_recording

    def start(self, camera_id: str, filename: str | None = None) -> RecordingInfo:
        """
        Start a recording session.

        This manager creates the recording path and tracks the session.
        Actual video encoding is intentionally delegated to the camera/
        video backend.
        """
        with self._lock:
            if self._active_recording is not None:
                raise RuntimeError("A recording is already active.")

            self._recording_dir.mkdir(parents=True, exist_ok=True)

            started_at = time()

            if filename is None:
                filename = f"camera_{camera_id}_{int(started_at)}.mp4"

            file_path = self._recording_dir / filename

            recording = RecordingInfo(
                camera_id=camera_id,
                file_path=file_path,
                started_at=started_at,
            )

            self._active_recording = recording
            return recording

    def stop(self) -> RecordingInfo:
        """Stop the active recording and return its final metadata."""
        with self._lock:
            if self._active_recording is None:
                raise RuntimeError("No recording is currently active.")

            active = self._active_recording

            completed = RecordingInfo(
                camera_id=active.camera_id,
                file_path=active.file_path,
                started_at=active.started_at,
                ended_at=time(),
            )

            self._active_recording = None
            return completed

    def cancel(self) -> Optional[RecordingInfo]:
        """
        Cancel the active recording session.

        The manager only clears its session state. Removing or deleting
        a partially created media file is intentionally left to the
        recording backend/application policy.
        """
        with self._lock:
            recording = self._active_recording
            self._active_recording = None
            return recording
