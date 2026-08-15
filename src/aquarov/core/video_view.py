"""AquaROV AI — Video View widget.

PySide6 video display widget for the AquaROV AI Operator Console.
Consumes approved DTOs only and keeps rendering on the Qt GUI thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QRectF, Qt, Signal, Slot
from PySide6.QtGui import (
    QColor, QFont, QFontMetricsF, QImage, QMouseEvent,
    QPainter, QPaintEvent, QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from aquarov.core.dto import (
    AIConfig, BoundingBox, CameraChannel, Detection,
    DetectionClass, DetectionSet,
)

__all__ = ["OverlayStyle", "VideoView", "image_from_rgb888"]


def image_from_rgb888(data: bytes, width: int, height: int) -> QImage:
    """Build a deep-copied QImage from packed RGB888 bytes."""
    if width <= 0 or height <= 0 or len(data) < width * height * 3:
        return QImage()
    img = QImage(
        data, width, height, width * 3, QImage.Format.Format_RGB888
    )
    return img.copy()


_DEFAULT_CLASS_COLORS: dict[DetectionClass, str] = {
    DetectionClass.FISH: "#38bdf8",
    DetectionClass.JELLYFISH: "#c084fc",
    DetectionClass.STARFISH: "#fb923c",
    DetectionClass.DEBRIS: "#facc15",
    DetectionClass.NET_DAMAGE: "#ef4444",
    DetectionClass.INFRA_DEFECT: "#f87171",
    DetectionClass.UNKNOWN: "#94a3b8",
}


@dataclass(slots=True)
class OverlayStyle:
    """Visual style for detection overlays."""

    class_colors: dict[DetectionClass, str] = field(
        default_factory=lambda: dict(_DEFAULT_CLASS_COLORS)
    )
    fallback_color: str = "#94a3b8"
    line_width: float = 2.0
    label_point_size: float = 11.0
    show_labels: bool = True
    show_confidence: bool = True
    label_background_alpha: int = 170
    placeholder_color: str = "#64748b"
    background_color: str = "#0b1220"

    def color_for(self, cls: DetectionClass) -> QColor:
        return QColor(self.class_colors.get(cls, self.fallback_color))


class VideoView(QWidget):
    """Video tile with AI detection overlays for one camera channel."""

    tileActivated = Signal(str)

    def __init__(
        self,
        channel: CameraChannel | None = None,
        style: OverlayStyle | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._style = style or OverlayStyle()
        self._channel: CameraChannel | None = None
        self._camera_id = ""
        self._camera_name = ""
        self._frame = QImage()
        self._frame_size: tuple[int, int] = (0, 0)
        self._detections: DetectionSet | None = None
        self._connected = True
        self._overlay_enabled = True
        self._class_visible = {cls: True for cls in DetectionClass}
        self._global_threshold = 0.0
        self._class_thresholds: dict[DetectionClass, float] = {}

        if channel is not None:
            self.set_channel(channel)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(160, 90)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    @Slot(object)
    def set_channel(self, channel: CameraChannel) -> None:
        """Bind this tile to a camera channel."""
        self._channel = channel
        self._camera_id = channel.id
        self._camera_name = channel.name
        self._detections = None
        self.update()

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @Slot(str, QImage)
    def update_frame(self, camera_id: str, frame: QImage) -> None:
        """Replace the displayed frame using latest-wins semantics."""
        if camera_id != self._camera_id:
            return
        if frame.isNull() or frame.width() <= 0 or frame.height() <= 0:
            return
        self._frame = frame.copy()
        self._frame_size = (frame.width(), frame.height())
        self.update()

    @Slot(object)
    def update_detections(self, detections: DetectionSet) -> None:
        """Replace the overlay detection set for this camera."""
        if detections.source_camera != self._camera_id:
            return
        self._detections = detections
        self.update()

    @Slot(bool)
    def set_connected(self, connected: bool) -> None:
        """Mark the camera as connected/disconnected."""
        if self._connected == connected:
            return
        self._connected = connected
        if not connected:
            self._detections = None
        self.update()

    @Slot()
    def clear(self) -> None:
        """Drop the current frame and detections."""
        self._frame = QImage()
        self._frame_size = (0, 0)
        self._detections = None
        self.update()

    @Slot(bool)
    def set_overlay_enabled(self, enabled: bool) -> None:
        self._overlay_enabled = enabled
        self.update()

    @Slot(object, bool)
    def set_class_visible(self, cls: DetectionClass, visible: bool) -> None:
        self._class_visible[cls] = visible
        self.update()

    @Slot(object, float)
    def set_confidence_threshold(
        self, cls: DetectionClass | None, threshold: float
    ) -> None:
        threshold = min(1.0, max(0.0, threshold))
        if cls is None:
            self._global_threshold = threshold
        else:
            self._class_thresholds[cls] = threshold
        self.update()

    @Slot(object)
    def apply_ai_config(self, config: AIConfig) -> None:
        """Load per-class display thresholds from AIConfig."""
        for name, value in config.display_confidence.items():
            try:
                cls = DetectionClass(name)
            except ValueError:
                continue
            self._class_thresholds[cls] = min(
                1.0, max(0.0, float(value))
            )
        self.update()

    @Slot(object)
    def set_style(self, style: OverlayStyle) -> None:
        self._style = style
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._camera_id:
            self.tileActivated.emit(self._camera_id)
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        try:
            painter.fillRect(
                self.rect(), QColor(self._style.background_color)
            )

            if self._frame.isNull():
                self._draw_placeholder(
                    painter,
                    "NO SIGNAL" if self._connected else "DISCONNECTED",
                )
                return

            target = self._letterbox_rect()
            painter.setRenderHint(
                QPainter.RenderHint.SmoothPixmapTransform
            )
            painter.drawImage(target, self._frame)

            if not self._connected:
                painter.fillRect(target, QColor(0, 0, 0, 140))
                self._draw_placeholder(painter, "DISCONNECTED")
                return

            if self._overlay_enabled and self._detections is not None:
                self._draw_detections(painter, target)

            self._draw_camera_label(painter)
        finally:
            painter.end()

    def _letterbox_rect(self) -> QRectF:
        fw, fh = self._frame_size
        if fw <= 0 or fh <= 0:
            return QRectF(self.rect())

        ww, wh = float(self.width()), float(self.height())
        scale = min(ww / fw, wh / fh)
        dw, dh = fw * scale, fh * scale

        return QRectF(
            (ww - dw) / 2.0, (wh - dh) / 2.0, dw, dh
        )

    def _is_drawable(self, det: Detection) -> bool:
        if not self._class_visible.get(det.class_name, True):
            return False
        threshold = self._class_thresholds.get(
            det.class_name, self._global_threshold
        )
        return det.confidence >= threshold

    def _draw_detections(
        self, painter: QPainter, target: QRectF
    ) -> None:
        if self._detections is None:
            return

        fw, fh = self._frame_size
        if fw <= 0 or fh <= 0:
            return

        sx, sy = target.width() / fw, target.height() / fh

        font = QFont(self.font())
        font.setPointSizeF(self._style.label_point_size)
        font.setBold(True)
        painter.setFont(font)
        metrics = QFontMetricsF(font)

        for det in self._detections.detections:
            if not self._is_drawable(det):
                continue

            rect = self._map_box(
                det.bounding_box, target, sx, sy
            )
            color = self._style.color_for(det.class_name)

            pen = QPen(color)
            pen.setWidthF(self._style.line_width)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

            if self._style.show_labels:
                self._draw_label(
                    painter, metrics, rect, det, color
                )

    @staticmethod
    def _map_box(
        box: BoundingBox,
        target: QRectF,
        sx: float,
        sy: float,
    ) -> QRectF:
        return QRectF(
            target.x() + box.x1 * sx,
            target.y() + box.y1 * sy,
            max(1.0, box.width * sx),
            max(1.0, box.height * sy),
        )

    def _draw_label(
        self,
        painter: QPainter,
        metrics: QFontMetricsF,
        rect: QRectF,
        det: Detection,
        color: QColor,
    ) -> None:
        text = det.class_name.value
        if self._style.show_confidence:
            text = f"{text} {det.confidence * 100.0:.0f}%"

        pad = 3.0
        tw = metrics.horizontalAdvance(text) + 2 * pad
        th = metrics.height() + 2 * pad
        x, y = rect.x(), rect.y() - th
        if y < 0:
            y = rect.y()

        chip = QRectF(x, y, tw, th)
        bg = QColor(color)
        bg.setAlpha(self._style.label_background_alpha)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bg)
        painter.drawRect(chip)
        painter.setPen(QColor("#0b1220"))
        painter.drawText(
            chip, Qt.AlignmentFlag.AlignCenter, text
        )

    def _draw_camera_label(self, painter: QPainter) -> None:
        if not self._camera_name:
            return

        font = QFont(self.font())
        font.setPointSizeF(self._style.label_point_size)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        pad = 4.0
        text = self._camera_name

        chip = QRectF(
            6.0,
            6.0,
            metrics.horizontalAdvance(text) + 2 * pad,
            metrics.height() + 2 * pad,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(11, 18, 32, 180))
        painter.drawRect(chip)
        painter.setPen(QColor("#e2e8f0"))
        painter.drawText(
            chip, Qt.AlignmentFlag.AlignCenter, text
        )

    def _draw_placeholder(
        self, painter: QPainter, message: str
    ) -> None:
        font = QFont(self.font())
        font.setPointSizeF(self._style.label_point_size + 4.0)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(self._style.placeholder_color))

        label = (
            f"{self._camera_name}  ·  {message}"
            if self._camera_name
            else message
        )
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            label,
        )
