"""AquaROV AI - Detection Overlay.

Qt overlay widget for drawing AI detections and tracking information
over the live video view.

The overlay consumes DTO objects from aquarov.core.dto and does not
contain inference or detection logic.
"""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from aquarov.core.dto import Detection, DetectionClass, Track


class DetectionOverlay(QWidget):
    """Transparent overlay that renders AI detection bounding boxes."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._detections: tuple[Detection, ...] = ()
        self._tracks: tuple[Track, ...] = ()

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def set_detections(
        self,
        detections: Iterable[Detection],
    ) -> None:
        """Replace the detections currently displayed by the overlay."""
        self._detections = tuple(detections)
        self.update()

    def set_tracks(
        self,
        tracks: Iterable[Track],
    ) -> None:
        """Replace the tracks currently displayed by the overlay."""
        self._tracks = tuple(tracks)
        self.update()

    def clear(self) -> None:
        """Clear all detection and tracking graphics."""
        self._detections = ()
        self._tracks = ()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        """Paint detection boxes, labels, and track IDs."""
        del event

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for detection in self._detections:
            self._draw_detection(painter, detection)

        for track in self._tracks:
            self._draw_track(painter, track)

        painter.end()

    def _draw_detection(
        self,
        painter: QPainter,
        detection: Detection,
    ) -> None:
        """Draw one detection bounding box and confidence label."""
        box = detection.bounding_box

        rect = QRectF(
            box.x1,
            box.y1,
            max(0.0, box.width),
            max(0.0, box.height),
        )

        pen = QPen(self._color_for_class(detection.class_name))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(rect)

        label = (
            f"{detection.class_name.value} "
            f"{detection.confidence * 100:.0f}%"
        )

        self._draw_label(
            painter,
            label,
            rect.topLeft(),
            pen.color(),
        )

    def _draw_track(
        self,
        painter: QPainter,
        track: Track,
    ) -> None:
        """Draw the latest position of a tracked object."""
        if not track.position_history:
            return

        x, y = track.position_history[-1]
        point = QPointF(x, y)

        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)

        painter.drawEllipse(point, 4.0, 4.0)

        label = f"ID {track.track_id}"

        self._draw_label(
            painter,
            label,
            QPointF(x + 6.0, y - 6.0),
            QColor(255, 255, 255),
        )

    def _draw_label(
        self,
        painter: QPainter,
        text: str,
        position: QPointF,
        color: QColor,
    ) -> None:
        """Draw a readable text label beside an overlay element."""
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)

        painter.setFont(font)

        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 8
        height = metrics.height() + 4

        background = QColor(color)
        background.setAlpha(190)

        background_rect = QRectF(
            position.x(),
            position.y() - height,
            width,
            height,
        )

        painter.fillRect(background_rect, background)

        painter.setPen(QColor(0, 0, 0))
        painter.drawText(
            background_rect,
            Qt.AlignmentFlag.AlignCenter,
            text,
        )

    @staticmethod
    def _color_for_class(
        detection_class: DetectionClass,
    ) -> QColor:
        """Return a consistent display color for each detection class."""
        colors = {
            DetectionClass.FISH: QColor(0, 220, 120),
            DetectionClass.JELLYFISH: QColor(180, 120, 255),
            DetectionClass.STARFISH: QColor(255, 170, 60),
            DetectionClass.DEBRIS: QColor(255, 220, 0),
            DetectionClass.NET_DAMAGE: QColor(255, 60, 60),
            DetectionClass.INFRA_DEFECT: QColor(255, 100, 200),
            DetectionClass.UNKNOWN: QColor(220, 220, 220),
        }

        return colors.get(
            detection_class,
            QColor(220, 220, 220),
        )


__all__ = ["DetectionOverlay"]
