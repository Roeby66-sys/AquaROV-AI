"""AquaROV AI — Main Operator Console Window.

Qt GUI shell and integration boundary for AquaROV AI.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot, QObject
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from aquarov.core.camera_manager import CameraManager
from aquarov.core.dto import (
    Alert,
    AppConfig,
    CameraChannel,
    DetectionSet,
    MissionState,
    RecordingState,
    SystemMetrics,
)
from aquarov.core.inference_worker import (
    InferenceBackend,
    InferenceFrame,
    InferenceResult,
    InferenceWorker,
    NullInferenceBackend,
)
from aquarov.core.video_view import VideoView


class _GuiBridge(QObject):
    """Thread-safe Qt signal bridge for non-GUI callbacks."""

    frame_ready = Signal(str, object)
    detection_ready = Signal(object)
    inference_error = Signal(object)
    camera_error = Signal(str, object)


class MainWindow(QMainWindow):
    """Top-level AquaROV AI Operator Console window.

    Video rendering and detection overlays are owned by VideoView.
    This class only coordinates DTOs, camera/inference services, and Qt UI.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        inference_backend: InferenceBackend | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.config = config or AppConfig()
        self._backend = inference_backend or NullInferenceBackend()
        self._bridge = _GuiBridge()

        self._video_views: dict[str, VideoView] = {}
        self._last_detection: DetectionSet | None = None
        self._last_system_metrics: SystemMetrics | None = None
        self._last_alert: Alert | None = None
        self._last_recording_state: RecordingState | None = None
        self._last_mission_state: MissionState | None = None

        self._bridge.frame_ready.connect(
            self._on_frame_ready,
            Qt.ConnectionType.QueuedConnection,
        )
        self._bridge.detection_ready.connect(
            self._on_detection_ready,
            Qt.ConnectionType.QueuedConnection,
        )
        self._bridge.inference_error.connect(
            self._on_inference_error,
            Qt.ConnectionType.QueuedConnection,
        )
        self._bridge.camera_error.connect(
            self._on_camera_error,
            Qt.ConnectionType.QueuedConnection,
        )

        self.inference_worker = InferenceWorker(
            self._backend,
            on_result=self._handle_inference_result,
            on_error=self._handle_inference_error,
        )

        self.camera_manager = CameraManager(
            self.inference_worker,
            on_frame=self._handle_camera_frame,
            on_error=self._handle_camera_error,
        )

        self._build_ui()
        self._configure_cameras()

    def _build_ui(self) -> None:
        self.setWindowTitle(
            f"{self.config.application_name} — Operator Console"
        )
        self.resize(1280, 800)

        central = QWidget(self)
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.status_label = QLabel("SYSTEM: READY")
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        root.addWidget(self.status_label)

        self.video_grid = QWidget(central)
        self.video_layout = QGridLayout(self.video_grid)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_layout.setSpacing(6)
        root.addWidget(self.video_grid, 1)

        self.alert_label = QLabel("Alerts: none")
        self.recording_label = QLabel("Recording: stopped")
        self.mission_label = QLabel("Mission: idle")
        self.metrics_label = QLabel("Metrics: waiting")

        root.addWidget(self.alert_label)
        root.addWidget(self.recording_label)
        root.addWidget(self.mission_label)
        root.addWidget(self.metrics_label)

    def _configure_cameras(self) -> None:
        for index, channel in enumerate(
            c for c in self.config.camera_config.channels if c.enabled
        ):
            self._add_video_view(channel, index)

    def _add_video_view(
        self,
        channel: CameraChannel,
        index: int,
    ) -> VideoView:
        view = VideoView(channel=channel)
        view.apply_ai_config(self.config.ai_config)
        view.tileActivated.connect(self._on_tile_activated)

        self.video_layout.addWidget(
            view,
            index // 2,
            index % 2,
        )

        self._video_views[channel.id] = view
        return view

    # ------------------------------------------------------------------
    # Non-GUI callbacks -> Qt signal bridge
    # ------------------------------------------------------------------

    def _handle_camera_frame(
        self,
        frame: InferenceFrame,
    ) -> None:
        if isinstance(frame.image, QImage) and not frame.image.isNull():
            self._bridge.frame_ready.emit(
                frame.camera_id,
                frame.image,
            )

    def _handle_camera_error(
        self,
        camera_id: str,
        exc: Exception,
    ) -> None:
        self._bridge.camera_error.emit(
            camera_id,
            exc,
        )

    def _handle_inference_result(
        self,
        result: InferenceResult,
    ) -> None:
        detection_set = DetectionSet(
            frame_index=result.frame_id,
            timestamp=result.timestamp,
            source_camera=result.camera_id,
            detections=tuple(result.detections),
            tracks=(),
        )

        self._bridge.detection_ready.emit(detection_set)

    def _handle_inference_error(
        self,
        exc: Exception,
    ) -> None:
        self._bridge.inference_error.emit(exc)

    # ------------------------------------------------------------------
    # GUI-thread slots
    # ------------------------------------------------------------------

    @Slot(str, object)
    def _on_frame_ready(
        self,
        camera_id: str,
        image: object,
    ) -> None:
        if isinstance(image, QImage):
            view = self._video_views.get(camera_id)
            if view is not None:
                view.update_frame(
                    camera_id,
                    image,
                )

    @Slot(object)
    def _on_detection_ready(
        self,
        value: object,
    ) -> None:
        if not isinstance(value, DetectionSet):
            return

        self._last_detection = value

        view = self._video_views.get(value.source_camera)
        if view is not None:
            view.update_detections(value)

    @Slot(object)
    def _on_inference_error(
        self,
        exc: object,
    ) -> None:
        self.status_label.setText(
            f"SYSTEM: INFERENCE ERROR — {exc}"
        )

    @Slot(str, object)
    def _on_camera_error(
        self,
        camera_id: str,
        exc: object,
    ) -> None:
        view = self._video_views.get(camera_id)

        if view is not None:
            view.set_connected(False)

        self.status_label.setText(
            f"SYSTEM: CAMERA ERROR — {camera_id}: {exc}"
        )

    @Slot(str)
    def _on_tile_activated(
        self,
        camera_id: str,
    ) -> None:
        self.status_label.setText(
            f"SYSTEM: CAMERA FOCUS — {camera_id}"
        )

    @Slot(object)
    def update_system_metrics(
        self,
        metrics: SystemMetrics,
    ) -> None:
        self._last_system_metrics = metrics

        self.metrics_label.setText(
            f"Metrics: CPU {metrics.cpu_usage:.1f}% | "
            f"Memory {metrics.memory_usage:.1f}% | "
            f"FPS {metrics.end_to_end_fps:.1f}"
        )

    @Slot(object)
    def update_alert(
        self,
        alert: Alert,
    ) -> None:
        self._last_alert = alert

        self.alert_label.setText(
            f"Alert [{alert.level.value}]: {alert.message}"
        )

    @Slot(object)
    def update_recording_state(
        self,
        state: RecordingState,
    ) -> None:
        self._last_recording_state = state

        suffix = (
            f" — {state.filename}"
            if state.filename
            else ""
        )

        self.recording_label.setText(
            f"Recording: {state.status.value}{suffix}"
        )

    @Slot(object)
    def update_mission_state(
        self,
        state: MissionState,
    ) -> None:
        self._last_mission_state = state

        self.mission_label.setText(
            f"Mission: {state.mission_name} — "
            f"{state.status.value} — "
            f"{state.progress * 100.0:.0f}%"
        )

    def start_services(self) -> None:
        """Start the inference services."""
        self.inference_worker.start()
        self.status_label.setText(
            "SYSTEM: INFERENCE READY"
        )

    def stop_services(self) -> None:
        """Stop the inference services."""
        self.inference_worker.stop()
        self.status_label.setText(
            "SYSTEM: STOPPED"
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        """Stop services before closing the application."""
        self.stop_services()
        event.accept()


__all__ = ["MainWindow"]        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_layout.setSpacing(6)
        root.addWidget(self.video_grid, 1)

        self.alert_label = QLabel("Alerts: none")
        self.recording_label = QLabel("Recording: stopped")
        self.mission_label = QLabel("Mission: idle")
        self.metrics_label = QLabel("Metrics: waiting")
        root.addWidget(self.alert_label)
        root.addWidget(self.recording_label)
        root.addWidget(self.mission_label)
        root.addWidget(self.metrics_label)

    def _configure_cameras(self) -> None:
        for index, channel in enumerate(
            c for c in self.config.camera_config.channels if c.enabled
        ):
            self._add_video_view(channel, index)

    def _add_video_view(self, channel: CameraChannel, index: int) -> VideoView:
        view = VideoView(channel=channel)
        view.apply_ai_config(self.config.ai_config)
        view.tileActivated.connect(self._on_tile_activated)
        self.video_layout.addWidget(view, index // 2, index % 2)
        self._video_views[channel.id] = view
        return view

    # Non-GUI callbacks -> Qt signal bridge
    def _handle_camera_frame(self, frame: InferenceFrame) -> None:
        if isinstance(frame.image, QImage) and not frame.image.isNull():
            self._bridge.frame_ready.emit(frame.camera_id, frame.image)

    def _handle_camera_error(self, camera_id: str, exc: Exception) -> None:
        self._bridge.camera_error.emit(camera_id, exc)

    def _handle_inference_result(self, result: InferenceResult) -> None:
        detection_set = DetectionSet(
            frame_index=result.frame_id,
            timestamp=result.timestamp,
            source_camera=result.camera_id,
            detections=tuple(result.detections),
            tracks=(),
        )
        self._bridge.detection_ready.emit(detection_set)

    def _handle_inference_error(self, exc: Exception) -> None:
        self._bridge.inference_error.emit(exc)

    # GUI-thread slots
    @Slot(str, object)
    def _on_frame_ready(self, camera_id: str, image: object) -> None:
        if isinstance(image, QImage):
            view = self._video_views.get(camera_id)
            if view is not None:
                view.update_frame(camera_id, image)

    @Slot(object)
    def _on_detection_ready(self, value: object) -> None:
        if not isinstance(value, DetectionSet):
            return
        self._last_detection = value
        view = self._video_views.get(value.source_camera)
        if view is not None:
            view.update_detections(value)

    @Slot(object)
    def _on_inference_error(self, exc: object) -> None:
        self.status_label.setText(f"SYSTEM: INFERENCE ERROR — {exc}")

    @Slot(str, object)
    def _on_camera_error(self, camera_id: str, exc: object) -> None:
        view = self._video_views.get(camera_id)
        if view is not None:
            view.set_connected(False)
        self.status_label.setText(f"SYSTEM: CAMERA ERROR — {camera_id}: {exc}")

    @Slot(str)
    def _on_tile_activated(self, camera_id: str) -> None:
        self.status_label.setText(f"SYSTEM: CAMERA FOCUS — {camera_id}")

    @Slot(object)
    def update_system_metrics(self, metrics: SystemMetrics) -> None:
        self._last_system_metrics = metrics
        self.metrics_label.setText(
            f"Metrics: CPU {metrics.cpu_usage:.1f}% | "
            f"Memory {metrics.memory_usage:.1f}% | FPS {metrics.end_to_end_fps:.1f}"
        )

    @Slot(object)
    def update_alert(self, alert: Alert) -> None:
        self._last_alert = alert
        self.alert_label.setText(
            f"Alert [{alert.level.value}]: {alert.message}"
        )

    @Slot(object)
    def update_recording_state(self, state: RecordingState) -> None:
        self._last_recording_state = state
        suffix = f" — {state.filename}" if state.filename else ""
        self.recording_label.setText(
            f"Recording: {state.status.value}{suffix}"
        )

    @Slot(object)
    def update_mission_state(self, state: MissionState) -> None:
        self._last_mission_state = state
        self.mission_label.setText(
            f"Mission: {state.mission_name} — {state.status.value} — "
            f"{state.progress * 100.0:.0f}%"
        )

    def start_services(self) -> None:
        self.inference_worker.start()
        self.status_label.setText("SYSTEM: INFERENCE READY")

    def stop_services(self) -> None:
        self.inference_worker.stop()
        self.status_label.setText("SYSTEM: STOPPED")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_services()
        event.accept()


__all__ = ["MainWindow"]
