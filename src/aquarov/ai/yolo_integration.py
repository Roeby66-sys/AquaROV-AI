"""AquaROV AI — YOLO Integration.

YOLO model integration layer for the AquaROV AI inference pipeline.

This module provides a framework-independent interface for loading a YOLO
model and converting its predictions into AquaROV detection objects.

The actual model backend is loaded lazily so the rest of the application can
be imported and tested without requiring the YOLO runtime to be installed.

Designed to support the AquaROV AI pipeline:

    Camera -> InferenceWorker -> YOLODetector -> DetectionSet -> Overlay

The backend can later be replaced or extended for Axelera Voyager/Metis
without changing the UI layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aquarov.core.dto import BoundingBox, Detection, DetectionSet


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

    The detector intentionally keeps the YOLO implementation behind a small
    interface so the rest of AquaROV AI does not depend directly on a
    particular YOLO package.

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
        """Load the configured YOLO model.

        The Ultralytics package is imported lazily. This prevents the entire
        AquaROV application from failing to import when YOLO dependencies
        are not installed.
        """
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

    def predict(self, frame: np.ndarray) -> DetectionSet:
        """Run YOLO inference on a single image frame.

        Parameters
        ----------
        frame:
            BGR image represented as a NumPy array.

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
            return DetectionSet(detections=[])

        return self._convert_result(results[0])

    def _convert_result(self, result: Any) -> DetectionSet:
        """Convert a YOLO result into AquaROV DetectionSet."""

        detections: list[Detection] = []

        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}

        if boxes is None:
            return DetectionSet(detections=detections)

        xyxy = self._to_numpy(boxes.xyxy)
        confidences = self._to_numpy(boxes.conf)
        class_ids = self._to_numpy(boxes.cls)

        for index in range(len(xyxy)):
            x1, y1, x2, y2 = (
                float(value) for value in xyxy[index]
            )

            confidence = float(confidences[index])
            class_id = int(class_ids[index])

            label = str(names.get(class_id, class_id))

            bbox = BoundingBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )

            detection = Detection(
                class_id=class_id,
                label=label,
                confidence=confidence,
                bbox=bbox,
            )

            detections.append(detection)

        return DetectionSet(detections=detections)

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
