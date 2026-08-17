"""AquaROV AI — Fish Activity Analysis.

Fish activity analytics layer for the AquaROV AI inference pipeline.

This module consumes DetectionSet objects produced by the inference layer and
converts fish detections/tracks into the shared FishMetrics DTO.

The analyzer is intentionally independent from YOLO, Voyager SDK, Qt, and
hardware-specific code.

Pipeline:

    DetectionSet -> FishActivityAnalyzer -> FishMetrics

Activity is estimated from the movement of fish bounding-box centers between
successive frames. When persistent Track objects are available, their
position_history is preferred. Otherwise, detections from consecutive frames
are associated using nearest-center matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from aquarov.core.dto import (
    ActivityLevel,
    Detection,
    DetectionClass,
    DetectionSet,
    FishMetrics,
    Track,
)


@dataclass(slots=True, frozen=True)
class FishActivityConfig:
    """Configuration for fish activity analysis.

    The activity index is normalized to the range 0.0-1.0.

    ``pixels_per_second_at_high_activity`` defines the image-space movement
    considered to represent an activity index of 1.0.
    """

    fish_classes: frozenset[DetectionClass] = frozenset(
        {
            DetectionClass.FISH,
        }
    )
    matching_distance_px: float = 120.0
    pixels_per_second_at_high_activity: float = 300.0
    low_threshold: float = 0.20
    normal_threshold: float = 0.50
    high_threshold: float = 0.75
    abnormal_threshold: float = 0.90


@dataclass(slots=True)
class _PreviousFrame:
    """Internal state used for frame-to-frame association."""

    timestamp: float
    centers: list[tuple[float, float]] = field(default_factory=list)


class FishActivityAnalyzer:
    """Analyze fish count and movement activity from DetectionSet frames."""

    def __init__(
        self,
        config: FishActivityConfig | None = None,
    ) -> None:
        self.config = config or FishActivityConfig()
        self._previous: dict[str, _PreviousFrame] = {}

    def analyze(self, detection_set: DetectionSet) -> FishMetrics:
        """Analyze one DetectionSet and return FishMetrics."""

        fish_detections = [
            detection
            for detection in detection_set.detections
            if detection.class_name in self.config.fish_classes
        ]

        count = len(fish_detections)
        average_size = self._average_size(fish_detections)

        activity_index = self._calculate_activity_index(
            detection_set=detection_set,
            fish_detections=fish_detections,
        )

        activity_level = self._activity_level(activity_index)

        return FishMetrics(
            count=count,
            average_size=average_size,
            activity_level=activity_level,
            activity_index=activity_index,
            abnormal_behavior=activity_index >= self.config.abnormal_threshold,
            timestamp=detection_set.timestamp,
            source_camera=detection_set.source_camera,
        )

    def reset(self, source_camera: str | None = None) -> None:
        """Reset stored frame history.

        If ``source_camera`` is provided, only that camera is reset.
        Otherwise all camera history is cleared.
        """

        if source_camera is None:
            self._previous.clear()
        else:
            self._previous.pop(source_camera, None)

    def _calculate_activity_index(
        self,
        *,
        detection_set: DetectionSet,
        fish_detections: list[Detection],
    ) -> float:
        """Calculate normalized movement activity for one camera."""

        previous = self._previous.get(detection_set.source_camera)

        current_centers = [
            detection.bounding_box.center
            for detection in fish_detections
        ]

        self._previous[detection_set.source_camera] = _PreviousFrame(
            timestamp=detection_set.timestamp,
            centers=current_centers,
        )

        if not current_centers:
            return 0.0

        track_speed = self._track_speed(
            detection_set.tracks,
            detection_set.timestamp,
        )

        if track_speed is not None:
            return self._normalize_speed(track_speed)

        if previous is None:
            return 0.0

        elapsed = detection_set.timestamp - previous.timestamp

        if elapsed <= 0.0:
            return 0.0

        movement = self._match_and_measure(
            previous.centers,
            current_centers,
        )

        if movement is None:
            return 0.0

        speed_px_s = movement / elapsed
        return self._normalize_speed(speed_px_s)

    def _track_speed(
        self,
        tracks: tuple[Track, ...],
        timestamp: float,
    ) -> float | None:
        """Return average fish speed from persistent tracks when available."""

        speeds: list[float] = []

        for track in tracks:
            if track.object_type not in self.config.fish_classes:
                continue

            history = track.position_history

            if len(history) < 2:
                continue

            elapsed = track.last_seen - track.first_seen

            if elapsed <= 0.0:
                continue

            distance = sum(
                hypot(
                    current[0] - previous[0],
                    current[1] - previous[1],
                )
                for previous, current in zip(history, history[1:])
            )

            if distance > 0.0:
                speeds.append(distance / elapsed)

        if not speeds:
            return None

        return sum(speeds) / len(speeds)

    def _match_and_measure(
        self,
        previous_centers: list[tuple[float, float]],
        current_centers: list[tuple[float, float]],
    ) -> float | None:
        """Match current fish to nearby previous fish and return mean movement."""

        remaining = set(range(len(previous_centers)))
        movements: list[float] = []

        for current in current_centers:
            best_index: int | None = None
            best_distance = self.config.matching_distance_px

            for index in remaining:
                previous = previous_centers[index]
                distance = hypot(
                    current[0] - previous[0],
                    current[1] - previous[1],
                )

                if distance <= best_distance:
                    best_distance = distance
                    best_index = index

            if best_index is not None:
                remaining.remove(best_index)
                movements.append(best_distance)

        if not movements:
            return None

        return sum(movements) / len(movements)

    def _normalize_speed(self, speed_px_s: float) -> float:
        """Convert image-space speed into a 0.0-1.0 activity index."""

        reference = self.config.pixels_per_second_at_high_activity

        if reference <= 0.0:
            return 0.0

        return max(0.0, min(1.0, speed_px_s / reference))

    @staticmethod
    def _average_size(
        fish_detections: list[Detection],
    ) -> float:
        """Return mean fish bounding-box area in square pixels.

        FishMetrics.average_size has no physical unit in the current DTO, so
        this implementation uses image-space bounding-box area rather than
        pretending that pixel dimensions represent real-world fish length.
        """

        if not fish_detections:
            return 0.0

        areas = [
            max(0.0, detection.bounding_box.width)
            * max(0.0, detection.bounding_box.height)
            for detection in fish_detections
        ]

        return sum(areas) / len(areas)

    def _activity_level(self, activity_index: float) -> ActivityLevel:
        """Map the normalized activity index to the shared enum."""

        if activity_index <= 0.0:
            return ActivityLevel.NONE

        if activity_index < self.config.low_threshold:
            return ActivityLevel.LOW

        if activity_index < self.config.normal_threshold:
            return ActivityLevel.NORMAL

        if activity_index < self.config.high_threshold:
            return ActivityLevel.HIGH

        return ActivityLevel.ABNORMAL


__all__ = [
    "FishActivityConfig",
    "FishActivityAnalyzer",
]
