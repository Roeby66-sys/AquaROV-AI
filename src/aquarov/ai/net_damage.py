"""AquaROV AI — Net Damage Detection.

Net damage analytics layer for the AquaROV AI inference pipeline.

This module consumes DetectionSet objects produced by the inference layer
and converts net-damage detections into operator Alert objects.

The analyzer is intentionally independent from YOLO, Voyager SDK, Qt,
and hardware-specific code.

Pipeline:

    DetectionSet -> NetDamageAnalyzer -> Alert
"""

from __future__ import annotations

from dataclasses import dataclass

from aquarov.core.dto import (
    Alert,
    AlertLevel,
    DetectionClass,
    DetectionSet,
)


@dataclass(slots=True, frozen=True)
class NetDamageConfig:
    """Configuration for net-damage analysis."""

    confidence_threshold: float = 0.50
    critical_confidence_threshold: float = 0.85
    alert_type: str = "net_damage"


class NetDamageAnalyzer:
    """Convert net-damage detections into operator alerts."""

    def __init__(
        self,
        config: NetDamageConfig | None = None,
    ) -> None:
        self.config = config or NetDamageConfig()

    def analyze(self, detection_set: DetectionSet) -> list[Alert]:
        """Analyze one DetectionSet and return net-damage alerts."""

        alerts: list[Alert] = []

        for detection in detection_set.detections:
            if detection.class_name is not DetectionClass.NET_DAMAGE:
                continue

            if detection.confidence < self.config.confidence_threshold:
                continue

            alerts.append(
                Alert(
                    alert_id=(
                        f"net-damage-{detection.source_camera}-"
                        f"{detection.object_id}-{detection.timestamp:.6f}"
                    ),
                    level=self._alert_level(detection.confidence),
                    message=self._message(detection.confidence),
                    timestamp=detection.timestamp,
                    related_object=str(detection.object_id),
                    source_camera=detection.source_camera,
                    alert_type=self.config.alert_type,
                )
            )

        return alerts

    def _alert_level(self, confidence: float) -> AlertLevel:
        """Map detection confidence to an operator alert level."""

        if confidence >= self.config.critical_confidence_threshold:
            return AlertLevel.CRITICAL

        return AlertLevel.WARNING

    @staticmethod
    def _message(confidence: float) -> str:
        """Create a concise operator-facing alert message."""

        return (
            "Possible net damage detected "
            f"(confidence: {confidence:.0%})"
        )


__all__ = [
    "NetDamageConfig",
    "NetDamageAnalyzer",
  ]
