"""
AquaROV AI — Snapshot Manager.

Framework-independent snapshot lifecycle manager.

The manager creates snapshot paths and tracks snapshot metadata.
Actual image capture is delegated to the camera/video backend.

Project: AquaROV AI - Underwater ROV Inspection System
Target: Axelera Metis + Voyager SDK deployments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import time


DEFAULT_IMAGE_EXTENSION = ".jpg"


@dataclass(frozen=True, slots=True)
class SnapshotInfo:
    """Information describing a captured snapshot."""

    camera_id: str
    file_path: Path
    timestamp: float


class SnapshotManager:
    """Manage snapshot paths and metadata independently of the camera backend."""

    def __init__(
        self,
        snapshot_dir: str | Path,
        image_extension: str = DEFAULT_IMAGE_EXTENSION,
    ) -> None:
        self._snapshot_dir = Path(snapshot_dir)
        self._image_extension = self._normalize_extension(image_extension)
        self._lock = Lock()

    @property
    def snapshot_dir(self) -> Path:
        """Return the runtime snapshot directory."""
        return self._snapshot_dir

    @property
    def image_extension(self) -> str:
        """Return the configured snapshot image extension."""
        return self._image_extension

    def create_snapshot_info(
        self,
        camera_id: str,
        filename: str | None = None,
    ) -> SnapshotInfo:
        """
        Create metadata for a new snapshot.

        This method creates the destination directory and returns the
        path that the camera/video backend should use when saving the image.
        It does not capture or write image data itself.
        """
        with self._lock:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time()

            if filename is None:
                filename = (
                    f"camera_{camera_id}_{int(timestamp * 1000)}"
                    f"{self._image_extension}"
                )
            elif Path(filename).suffix == "":
                filename = f"{filename}{self._image_extension}"

            file_path = self._snapshot_dir / filename

            return SnapshotInfo(
                camera_id=camera_id,
                file_path=file_path,
                timestamp=timestamp,
            )

    def build_path(
        self,
        camera_id: str,
        filename: str | None = None,
    ) -> Path:
        """Return a snapshot destination path without creating image data."""
        return self.create_snapshot_info(camera_id, filename).file_path

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        """Normalize an image extension to include a leading dot."""
        extension = extension.strip()

        if not extension:
            return DEFAULT_IMAGE_EXTENSION

        if not extension.startswith("."):
            extension = f".{extension}"

        return extension.lower()
