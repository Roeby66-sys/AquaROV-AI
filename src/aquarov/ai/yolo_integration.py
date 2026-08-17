"""AquaROV AI — YOLO Integration.

YOLO model integration layer for the AquaROV AI inference pipeline.

This module provides a framework-independent adapter that converts YOLO
predictions into the shared AquaROV Detection and DetectionSet DTOs.

The YOLO runtime is imported lazily so the rest of the application can be
imported and tested without requiring Ultralytics to be installed.

Pipeline:

    Camera -> InferenceWorker -> YOLODetector -> DetectionSet -> Overlay

The adapter is intentionally isolated from the UI and core inference
orchestration so the backend can later be replaced by Axelera Voyager/Metis
without changing the application DTO contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

import numpy as np

from aquarov.core.dto import (
    BoundingBox,
    Detection,
    DetectionClass,
    DetectionSet,
)


@dataclass(slots=True, frozen=True)
class YOLOConfig:
    """Configuration for a YOLO detector."""

    model_path: str
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "auto"
    classes: tuple[int, ...] | None = None


class YOLOIntegrationError(RuntimeError):
    """Raised when YOLO integration cannot be initialized or executed."""


class YOLODetector:
    """YOLO inference adapter for AquaROV AI.

    The detector keeps the YOLO implementation behind a small interface so
    the rest of AquaROV AI does not depend directly on Ultralytics.

    Parameters
    ----------
    config:
        YOLO detector configuration.
    """

    def __init__(self, config: YOLOConfig) -> None:
        self.config = config
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        """Return True when the YOLO model has been loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the configured YOLO model lazily."""
        model_path = Path(self.config.model_path)

        if not model_path.exists():
            raise YOLOIntegrationError(
                f"YOLO model not found: {model_path}"
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise YOLOIntegrationError(
                "Ultralytics is not installed. "
                "Install the YOLO runtime before loading the model."
            ) from exc

        try:
            self._model = YOLO(str(model_path))
        except Exception as exc:
            raise YOLOIntegrationError(
                f"Failed to load YOLO model: {model_path}"
            ) from exc

    def predict(
        self,
        frame: np.ndarray,
        *,
        frame_index: int = 0,
        source_camera: str = "",
        timestamp: float | None = None,
    ) -> DetectionSet:
        """Run YOLO inference on a single image frame.

        Parameters
        ----------
        frame:
            BGR image represented as a NumPy array.

        frame_index:
            Sequential frame number supplied by the caller.

        source_camera:
            Logical AquaROV camera identifier.

        timestamp:
            Frame timestamp. If omitted, the current Unix timestamp is used.

        Returns
        -------
        DetectionSet
            AquaROV detection DTOs generated from the YOLO predictions.
        """
        if self._model is None:
            self.load()

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")

        if frame.ndim not in (2, 3):
            raise ValueError(
                "frame must contain either a 2-D grayscale image "
                "or a 3-D color image"
            )

        if timestamp is None:
            timestamp = time()

        predict_kwargs: dict[str, Any] = {
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
            "verbose": False,
        }

        if self.config.device != "auto":
            predict_kwargs["device"] = self.config.device

        if self.config.classes is not None:
            predict_kwargs["classes"] = list(self.config.classes)

        try:
            results = self._model.predict(
                source=frame,
                **predict_kwargs,
            )
        except Exception as exc:
            raise YOLOIntegrationError(
                "YOLO inference failed"
            ) from exc

        if not results:
            return DetectionSet(
                frame_index=frame_index,
                timestamp=timestamp,
                source_camera=source_camera,
                detections=(),
                tracks=(),
            )

        return self._convert_result(
            results[0],
            frame_index=frame_index,
            timestamp=timestamp,
            source_camera=source_camera,
        )

    def _convert_result(
        self,
        result: Any,
        *,
        frame_index: int,
        timestamp: float,
        source_camera: str,
    ) -> DetectionSet:
        """Convert one YOLO result into the AquaROV DetectionSet DTO."""

        detections: list[Detection] = []

        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}

        if boxes is None:
            return DetectionSet(
                frame_index=frame_index,
                timestamp=timestamp,
                source_camera=source_camera,
                detections=(),
                tracks=(),
            )

        xyxy = self._to_numpy(boxes.xyxy)
        confidences = self._to_numpy(boxes.conf)
        class_ids = self._to_numpy(boxes.cls)

        for index in range(len(xyxy)):
            x1, y1, x2, y2 = (
                float(value) for value in xyxy[index]
            )

            confidence = float(confidences[index])
            class_id = int(class_ids[index])

            label = self._resolve_label(
                names=names,
                class_id=class_id,
            )

            detection_class = self._map_detection_class(label)

            detection = Detection(
                object_id=index,
                class_name=detection_class,
                confidence=confidence,
                bounding_box=BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                ),
                timestamp=timestamp,
                source_camera=source_camera,
            )

            detections.append(detection)

        return DetectionSet(
            frame_index=frame_index,
            timestamp=timestamp,
            source_camera=source_camera,
            detections=tuple(detections),
            tracks=(),
        )

    @staticmethod
    def _resolve_label(
        *,
        names: Any,
        class_id: int,
    ) -> str:
        """Resolve a YOLO class ID to its textual label."""

        if isinstance(names, dict):
            value = names.get(class_id, class_id)
        elif isinstance(names, (list, tuple)):
            if 0 <= class_id < len(names):
                value = names[class_id]
            else:
                value = class_id
        else:
            value = class_id

        return str(value).strip().lower()

    @staticmethod
    def _map_detection_class(label: str) -> DetectionClass:
        """Map a YOLO label to the AquaROV DetectionClass enum.

        Unknown model labels intentionally fall back to UNKNOWN instead of
        raising an exception. This allows different YOLO models to be used
        without breaking the application DTO contract.
        """

        normalized = label.strip().lower().replace("-", "_").replace(" ", "_")

        aliases: dict[str, DetectionClass] = {
            "fish": DetectionClass.FISH,
            "jellyfish": DetectionClass.JELLYFISH,
            "jelly_fish": DetectionClass.JELLYFISH,
            "starfish": DetectionClass.STARFISH,
            "star_fish": DetectionClass.STARFISH,
            "debris": DetectionClass.DEBRIS,
            "marine_debris": DetectionClass.DEBRIS,
            "trash": DetectionClass.DEBRIS,
            "plastic": DetectionClass.DEBRIS,
            "net_damage": DetectionClass.NET_DAMAGE,
            "netdamage": DetectionClass.NET_DAMAGE,
            "net_defect": DetectionClass.NET_DAMAGE,
            "infra_defect": DetectionClass.INFRA_DEFECT,
            "infrastructure_defect": DetectionClass.INFRA_DEFECT,
        }

        return aliases.get(normalized, DetectionClass.UNKNOWN)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Convert a YOLO tensor-like object to NumPy."""

        if hasattr(value, "detach"):
            value = value.detach()

        if hasattr(value, "cpu"):
            value = value.cpu()

        if hasattr(value, "numpy"):
            value = value.numpy()

        return np.asarray(value)

    def unload(self) -> None:
        """Release the loaded YOLO model."""

        self._model = None


__all__ = [
    "YOLOConfig",
    "YOLOIntegrationError",
    "YOLODetector",
        ]
