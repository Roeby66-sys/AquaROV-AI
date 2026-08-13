"""
AquaROV AI — Shared Data Transfer Objects (DTO layer).

Project: AquaROV AI - Underwater ROV Inspection System
Target: Axelera Metis + Voyager SDK deployments.

This module is the central, dependency-free data contract shared by:
- GUI (Qt/QML operator console)
- AI inference engine (Voyager SDK InferenceStream worker)
- Camera / stream manager
- Sensor integration
- Recording and snapshot managers
- Mission control
- MQTT communications

Design rules:
- Python 3.12 standard library only.
- No business logic.
- Dataclasses/enums only.
- JSON-safe to_dict()/from_dict() helpers.
- No imports from Qt, Voyager SDK, MQTT, database, or external packages.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class _SerializableMixin:
    """Provide JSON-safe dictionary serialization for DTO dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        def _convert(value: Any) -> Any:
            if isinstance(value, StrEnum):
                return value.value
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            return value

        return {
            key: _convert(value)
            for key, value in asdict(self).items()
        }


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CameraStatus(StrEnum):
    IDLE = "idle"
    CONNECTING = "connecting"
    LIVE = "live"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    EOS = "eos"
    DISABLED = "disabled"


class SourceType(StrEnum):
    RTSP = "rtsp"
    USB = "usb"
    FILE = "file"
    ROV = "rov"


class DetectionClass(StrEnum):
    FISH = "fish"
    JELLYFISH = "jellyfish"
    STARFISH = "starfish"
    DEBRIS = "debris"
    NET_DAMAGE = "net_damage"
    INFRA_DEFECT = "infra_defect"
    UNKNOWN = "unknown"


class TrackStatus(StrEnum):
    ACTIVE = "active"
    LOST = "lost"
    ENDED = "ended"


class AlertLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ActivityLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    ABNORMAL = "abnormal"


class RecordingStatus(StrEnum):
    STOPPED = "stopped"
    RECORDING = "recording"
    PAUSED = "paused"
    ERROR = "error"


class MissionStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class UnitSystem(StrEnum):
    METRIC = "metric"
    IMPERIAL = "imperial"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class BoundingBox(_SerializableMixin):
    """Axis-aligned pixel-space bounding box: (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundingBox":
        return cls(
            x1=data["x1"],
            y1=data["y1"],
            x2=data["x2"],
            y2=data["y2"],
        )


@dataclass(slots=True, frozen=True)
class GeoLocation(_SerializableMixin):
    """WGS84 geographic position with optional depth."""

    latitude: float
    longitude: float
    depth_m: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeoLocation":
        return cls(
            latitude=data["latitude"],
            longitude=data["longitude"],
            depth_m=data.get("depth_m"),
        )


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CameraChannel(_SerializableMixin):
    """Logical camera input managed by the camera/stream subsystem."""

    id: str
    name: str
    source_type: SourceType = SourceType.RTSP
    source_uri: str = ""
    resolution: tuple[int, int] = (1920, 1080)
    fps: float = 30.0
    status: CameraStatus = CameraStatus.IDLE
    network: str = ""
    recording: bool = False
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraChannel":
        return cls(
            id=data["id"],
            name=data["name"],
            source_type=SourceType(data.get("source_type", "rtsp")),
            source_uri=data.get("source_uri", ""),
            resolution=tuple(data.get("resolution", (1920, 1080))),
            fps=data.get("fps", 30.0),
            status=CameraStatus(data.get("status", "idle")),
            network=data.get("network", ""),
            recording=data.get("recording", False),
            enabled=data.get("enabled", True),
        )


# ---------------------------------------------------------------------------
# AI inference
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class Detection(_SerializableMixin):
    """Single object detection produced by the inference engine."""

    object_id: int
    class_name: DetectionClass
    confidence: float
    bounding_box: BoundingBox
    timestamp: float
    source_camera: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        return cls(
            object_id=data["object_id"],
            class_name=DetectionClass(data["class_name"]),
            confidence=data["confidence"],
            bounding_box=BoundingBox.from_dict(data["bounding_box"]),
            timestamp=data["timestamp"],
            source_camera=data["source_camera"],
        )


@dataclass(slots=True)
class Track(_SerializableMixin):
    """Persistent tracked object across inference frames."""

    track_id: int
    object_type: DetectionClass
    position_history: list[tuple[float, float]] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    status: TrackStatus = TrackStatus.ACTIVE
    source_camera: str = ""
    confidence: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Track":
        return cls(
            track_id=data["track_id"],
            object_type=DetectionClass(data["object_type"]),
            position_history=[
                tuple(position)
                for position in data.get("position_history", [])
            ],
            first_seen=data.get("first_seen", 0.0),
            last_seen=data.get("last_seen", 0.0),
            status=TrackStatus(data.get("status", "active")),
            source_camera=data.get("source_camera", ""),
            confidence=data.get("confidence", 0.0),
        )


@dataclass(slots=True)
class DetectionSet(_SerializableMixin):
    """Detections and tracks extracted from one processed frame."""

    frame_index: int
    timestamp: float
    source_camera: str
    detections: tuple[Detection, ...] = ()
    tracks: tuple[Track, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionSet":
        return cls(
            frame_index=data["frame_index"],
            timestamp=data["timestamp"],
            source_camera=data["source_camera"],
            detections=tuple(
                Detection.from_dict(item)
                for item in data.get("detections", [])
            ),
            tracks=tuple(
                Track.from_dict(item)
                for item in data.get("tracks", [])
            ),
        )


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Alert(_SerializableMixin):
    """Operator alert raised by the analytics/rules layer."""

    alert_id: str
    level: AlertLevel
    message: str
    timestamp: float
    related_object: str | None = None
    source_camera: str | None = None
    alert_type: str = ""
    snapshot_path: str | None = None
    acknowledged: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Alert":
        return cls(
            alert_id=data["alert_id"],
            level=AlertLevel(data.get("level", "info")),
            message=data["message"],
            timestamp=data["timestamp"],
            related_object=data.get("related_object"),
            source_camera=data.get("source_camera"),
            alert_type=data.get("alert_type", ""),
            snapshot_path=data.get("snapshot_path"),
            acknowledged=data.get("acknowledged", False),
        )


# ---------------------------------------------------------------------------
# Aquaculture analytics
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FishMetrics(_SerializableMixin):
    """Aggregated fish analytics for one camera/channel or cage zone."""

    count: int = 0
    average_size: float = 0.0
    activity_level: ActivityLevel = ActivityLevel.NONE
    activity_index: float = 0.0
    abnormal_behavior: bool = False
    timestamp: float = field(default_factory=time.time)
    source_camera: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FishMetrics":
        return cls(
            count=data.get("count", 0),
            average_size=data.get("average_size", 0.0),
            activity_level=ActivityLevel(
                data.get("activity_level", "none")
            ),
            activity_index=data.get("activity_index", 0.0),
            abnormal_behavior=data.get("abnormal_behavior", False),
            timestamp=data.get("timestamp", time.time()),
            source_camera=data.get("source_camera", ""),
        )


@dataclass(slots=True)
class WaterQuality(_SerializableMixin):
    """Water-quality sensor readings; values may be unavailable."""

    temperature: float | None = None
    ph: float | None = None
    dissolved_oxygen: float | None = None
    turbidity: float | None = None
    salinity: float | None = None
    timestamp: float = field(default_factory=time.time)
    sensor_id: str | None = None

    @property
    def is_available(self) -> bool:
        return any(
            value is not None
            for value in (
                self.temperature,
                self.ph,
                self.dissolved_oxygen,
                self.turbidity,
                self.salinity,
            )
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WaterQuality":
        return cls(
            temperature=data.get("temperature"),
            ph=data.get("ph"),
            dissolved_oxygen=data.get("dissolved_oxygen"),
            turbidity=data.get("turbidity"),
            salinity=data.get("salinity"),
            timestamp=data.get("timestamp", time.time()),
            sensor_id=data.get("sensor_id"),
        )


# ---------------------------------------------------------------------------
# System monitoring
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SystemMetrics(_SerializableMixin):
    """Host and Axelera Metis health/performance metrics."""

    cpu_usage: float = 0.0
    gpu_usage: float | None = None
    memory_usage: float = 0.0
    temperature: float = 0.0
    uptime: float = 0.0
    end_to_end_fps: float = 0.0
    disk_free_gb: float = 0.0
    mqtt_connected: bool = False
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemMetrics":
        return cls(
            cpu_usage=data.get("cpu_usage", 0.0),
            gpu_usage=data.get("gpu_usage"),
            memory_usage=data.get("memory_usage", 0.0),
            temperature=data.get("temperature", 0.0),
            uptime=data.get("uptime", 0.0),
            end_to_end_fps=data.get("end_to_end_fps", 0.0),
            disk_free_gb=data.get("disk_free_gb", 0.0),
            mqtt_connected=data.get("mqtt_connected", False),
            timestamp=data.get("timestamp", time.time()),
        )


# ---------------------------------------------------------------------------
# Recording and snapshots
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RecordingState(_SerializableMixin):
    """Recording-manager state for one camera channel."""

    is_recording: bool = False
    filename: str = ""
    duration: float = 0.0
    storage_path: str = ""
    camera_id: str = ""
    status: RecordingStatus = RecordingStatus.STOPPED
    file_size_mb: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecordingState":
        return cls(
            is_recording=data.get("is_recording", False),
            filename=data.get("filename", ""),
            duration=data.get("duration", 0.0),
            storage_path=data.get("storage_path", ""),
            camera_id=data.get("camera_id", ""),
            status=RecordingStatus(data.get("status", "stopped")),
            file_size_mb=data.get("file_size_mb", 0.0),
        )


@dataclass(slots=True, frozen=True)
class SnapshotInfo(_SerializableMixin):
    """Metadata for a captured snapshot image."""

    image_path: str
    timestamp: float
    camera_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotInfo":
        return cls(
            image_path=data["image_path"],
            timestamp=data["timestamp"],
            camera_id=data["camera_id"],
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Mission control
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MissionState(_SerializableMixin):
    """State of an autonomous or operator-assisted ROV inspection mission."""

    mission_id: str
    mission_name: str
    status: MissionStatus = MissionStatus.PLANNED
    start_time: float | None = None
    location: GeoLocation | None = None
    waypoints_total: int = 0
    waypoints_done: int = 0
    battery_pct: float | None = None
    notes: str = ""

    @property
    def progress(self) -> float:
        if self.waypoints_total <= 0:
            return 0.0
        return min(1.0, self.waypoints_done / self.waypoints_total)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MissionState":
        location_data = data.get("location")
        return cls(
            mission_id=data["mission_id"],
            mission_name=data["mission_name"],
            status=MissionStatus(
                data.get("status", "planned")
            ),
            start_time=data.get("start_time"),
            location=(
                GeoLocation.from_dict(location_data)
                if location_data is not None
                else None
            ),
            waypoints_total=data.get("waypoints_total", 0),
            waypoints_done=data.get("waypoints_done", 0),
            battery_pct=data.get("battery_pct"),
            notes=data.get("notes", ""),
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CameraConfig(_SerializableMixin):
    """Camera subsystem configuration."""

    channels: list[CameraChannel] = field(default_factory=list)
    reconnect_interval_s: float = 5.0
    default_resolution: tuple[int, int] = (1920, 1080)
    default_fps: float = 30.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraConfig":
        return cls(
            channels=[
                CameraChannel.from_dict(channel)
                for channel in data.get("channels", [])
            ],
            reconnect_interval_s=data.get("reconnect_interval_s", 5.0),
            default_resolution=tuple(
                data.get("default_resolution", (1920, 1080))
            ),
            default_fps=data.get("default_fps", 30.0),
        )


@dataclass(slots=True)
class AIConfig(_SerializableMixin):
    """Voyager SDK / Metis inference configuration."""

    network: str = "aquarov-marine-det"
    pipe_type: str = "gst"
    display_confidence: dict[str, float] = field(default_factory=dict)
    alert_confidence: dict[str, float] = field(default_factory=dict)
    tracker_algorithm: str = "oc-sort"
    build_root: str = "build"
    metrics_interval_s: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AIConfig":
        return cls(
            network=data.get("network", "aquarov-marine-det"),
            pipe_type=data.get("pipe_type", "gst"),
            display_confidence=dict(
                data.get("display_confidence", {})
            ),
            alert_confidence=dict(
                data.get("alert_confidence", {})
            ),
            tracker_algorithm=data.get(
                "tracker_algorithm", "oc-sort"
            ),
            build_root=data.get("build_root", "build"),
            metrics_interval_s=data.get("metrics_interval_s", 1.0),
        )


@dataclass(slots=True)
class SensorConfig(_SerializableMixin):
    """IoT water-quality sensor subsystem configuration."""

    enabled: bool = False
    mqtt_topic_prefix: str = ""
    poll_interval_s: float = 30.0
    stale_after_s: float = 120.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensorConfig":
        return cls(
            enabled=data.get("enabled", False),
            mqtt_topic_prefix=data.get("mqtt_topic_prefix", ""),
            poll_interval_s=data.get("poll_interval_s", 30.0),
            stale_after_s=data.get("stale_after_s", 120.0),
        )


@dataclass(slots=True)
class AppConfig(_SerializableMixin):
    """Top-level application configuration shared by all modules."""

    application_name: str = "AquaROV AI"
    version: str = "0.1.0"
    camera_config: CameraConfig = field(default_factory=CameraConfig)
    ai_config: AIConfig = field(default_factory=AIConfig)
    sensor_config: SensorConfig = field(default_factory=SensorConfig)
    site_name: str = ""
    unit_system: UnitSystem = UnitSystem.METRIC
    glove_mode: bool = False
    recording_root: str = "recordings"
    mqtt_topic_prefix: str = "aquarov"
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        return cls(
            application_name=data.get(
                "application_name", "AquaROV AI"
            ),
            version=data.get("version", "0.1.0"),
            camera_config=CameraConfig.from_dict(
                data.get("camera_config", {})
            ),
            ai_config=AIConfig.from_dict(
                data.get("ai_config", {})
            ),
            sensor_config=SensorConfig.from_dict(
                data.get("sensor_config", {})
            ),
            site_name=data.get("site_name", ""),
            unit_system=UnitSystem(
                data.get("unit_system", UnitSystem.METRIC.value)
            ),
            glove_mode=data.get("glove_mode", False),
            recording_root=data.get("recording_root", "recordings"),
            mqtt_topic_prefix=data.get(
                "mqtt_topic_prefix", "aquarov"
            ),
            extras=dict(data.get("extras", {})),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # enums
    "CameraStatus",
    "SourceType",
    "DetectionClass",
    "TrackStatus",
    "AlertLevel",
    "ActivityLevel",
    "RecordingStatus",
    "MissionStatus",
    "UnitSystem",
    # geometry
    "BoundingBox",
    "GeoLocation",
    # camera
    "CameraChannel",
    # inference
    "Detection",
    "Track",
    "DetectionSet",
    # alerts and analytics
    "Alert",
    "FishMetrics",
    "WaterQuality",
    # system
    "SystemMetrics",
    # recording and snapshots
    "RecordingState",
    "SnapshotInfo",
    # mission
    "MissionState",
    # configuration
    "CameraConfig",
    "AIConfig",
    "SensorConfig",
    "AppConfig",
]
