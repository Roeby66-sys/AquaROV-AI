"""
AquaROV AI — Snapshot Manager.

Framework-independent snapshot lifecycle manager.

The manager creates snapshot metadata and prepares safe output paths.
Actual image capture and encoding are delegated to the camera/video backend.
Runtime snapshot files must remain outside the Git repository.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import time

from aquarov.core.dto import SnapshotInfo


class SnapshotManager:
    """Manage snapshot metadata and runtime output paths."""

    def __init__(self, snapshot_dir: str | Path) -> None:
        self._snapshot_dir = Path(snapshot_dir)
        self._lock = Lock()

    @property
    def snapshot_dir(self) -> Path:
        """Return the runtime snapshot directory."""
        return self._snapshot_dir

    def create_snapshot_info(
        self,
        camera_id: str,
        filename: str | None = None,
    ) -> SnapshotInfo:
        """
        Create snapshot metadata and prepare a safe output path.

        Actual image capture is handled by the camera/video backend.
        """
        with self._lock:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time()

            if filename is None:
                filename = f"camera_{camera_id}_{int(timestamp)}.jpg"

            file_path = self._snapshot_dir / filename

            try:
                file_path.resolve().relative_to(self._snapshot_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    "Snapshot filename must remain inside the snapshot directory."
                ) from exc

            return SnapshotInfo(
                camera_id=camera_id,
                image_path=str(file_path),
                timestamp=timestamp,
            )

    def build_path(
        self,
        camera_id: str,
        filename: str | None = None,
    ) -> Path:
        """Build a safe runtime path for a snapshot."""
        snapshot = self.create_snapshot_info(camera_id, filename)
        return Path(snapshot.image_path)
