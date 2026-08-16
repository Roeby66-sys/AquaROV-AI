"""
AquaROV AI — MQTT communication client.

Provides a small, framework-independent MQTT adapter for:
- water-quality sensor telemetry
- system metrics
- alerts
- application status

The module is intentionally independent from Qt, Voyager SDK, and
database/recording components.

MQTT implementation:
- Uses paho-mqtt when available.
- Keeps MQTT optional so the rest of AquaROV AI can still be imported
  and tested without an MQTT broker or MQTT package installed.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .dto import Alert, SystemMetrics, WaterQuality

logger = logging.getLogger(__name__)


MessageHandler = Callable[[str, dict[str, Any]], None]


@dataclass(slots=True, frozen=True)
class MQTTConfig:
    """Configuration for the AquaROV MQTT client."""

    host: str = "localhost"
    port: int = 1883
    username: str | None = None
    password: str | None = None
    topic_prefix: str = "aquarov"
    keepalive: int = 60
    reconnect_delay_s: float = 5.0
    client_id: str = "aquarov-ai"


class MQTTClient:
    """
    Lightweight MQTT client for AquaROV AI.

    The class does not require MQTT during construction. The paho-mqtt
    package is imported only when connect() is called.
    """

    def __init__(
        self,
        config: MQTTConfig | None = None,
        *,
        message_handler: MessageHandler | None = None,
    ) -> None:
        self.config = config or MQTTConfig()
        self.message_handler = message_handler

        self._client: Any | None = None
        self._connected = False
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        """Return True when the MQTT client is connected."""
        with self._lock:
            return self._connected

    @property
    def client(self) -> Any | None:
        """Return the underlying paho client, if initialized."""
        return self._client

    def connect(self) -> None:
        """
        Connect to the configured MQTT broker.

        Raises:
            RuntimeError: If paho-mqtt is not installed.
            ConnectionError: If the broker connection fails.
        """
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError(
                "paho-mqtt is required for MQTT connectivity. "
                "Install it with: pip install paho-mqtt"
            ) from exc

        with self._lock:
            if self._client is not None:
                return

            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.config.client_id,
            )

            if (
                self.config.username is not None
                and self.config.password is not None
            ):
                client.username_pw_set(
                    self.config.username,
                    self.config.password,
                )

            client.reconnect_delay_set(
                min_delay=max(1, int(self.config.reconnect_delay_s)),
                max_delay=max(
                    1,
                    int(self.config.reconnect_delay_s * 6),
                ),
            )

            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.on_message = self._on_message

            self._client = client

            try:
                client.connect(
                    self.config.host,
                    self.config.port,
                    self.config.keepalive,
                )
                client.loop_start()
            except Exception:
                self._client = None
                raise

    def disconnect(self) -> None:
        """Disconnect cleanly from the MQTT broker."""
        with self._lock:
            client = self._client

            if client is None:
                self._connected = False
                return

            try:
                client.loop_stop()
                client.disconnect()
            finally:
                self._connected = False
                self._client = None

    def subscribe(self, topic: str, qos: int = 0) -> None:
        """Subscribe to an MQTT topic."""
        client = self._require_client()
        client.subscribe(topic, qos=qos)

    def subscribe_sensor_topic(self, sensor_id: str) -> None:
        """Subscribe to telemetry from one water-quality sensor."""
        topic = self.topic("sensor", sensor_id, "water_quality")
        self.subscribe(topic)

    def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        qos: int = 0,
        retain: bool = False,
    ) -> None:
        """Publish a JSON payload to an MQTT topic."""
        client = self._require_client()

        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        result = client.publish(
            topic,
            encoded,
            qos=qos,
            retain=retain,
        )

        if result.rc != 0:
            raise ConnectionError(
                f"MQTT publish failed with return code {result.rc}"
            )

    def publish_water_quality(
        self,
        reading: WaterQuality,
    ) -> None:
        """Publish a WaterQuality DTO."""
        sensor_id = reading.sensor_id or "unknown"
        topic = self.topic(
            "sensor",
            sensor_id,
            "water_quality",
        )
        self.publish(topic, reading.to_dict())

    def publish_system_metrics(
        self,
        metrics: SystemMetrics,
    ) -> None:
        """Publish system health/performance metrics."""
        topic = self.topic("system", "metrics")
        self.publish(topic, metrics.to_dict())

    def publish_alert(self, alert: Alert) -> None:
        """Publish an operator alert."""
        topic = self.topic("alerts", alert.alert_type or "general")
        self.publish(topic, alert.to_dict())

    def publish_status(self, status: str) -> None:
        """Publish a simple application status message."""
        topic = self.topic("status")
        self.publish(
            topic,
            {
                "status": status,
            },
            retain=True,
        )

    def topic(self, *parts: str) -> str:
        """Build a topic under the configured AquaROV prefix."""
        clean_parts = [
            str(part).strip("/")
            for part in parts
            if str(part).strip("/")
        ]

        return "/".join(
            [self.config.topic_prefix.strip("/"), *clean_parts]
        )

    def set_message_handler(
        self,
        handler: MessageHandler | None,
    ) -> None:
        """Set or clear the incoming-message callback."""
        self.message_handler = handler

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError(
                "MQTT client is not initialized. Call connect() first."
            )

        if not self.connected:
            raise ConnectionError(
                "MQTT client is not connected to the broker."
            )

        return self._client

    def _on_connect(
        self,
        client: Any,
        userdata: Any,
        flags: Any,
        reason_code: Any,
        properties: Any = None,
    ) -> None:
        del client, userdata, flags, properties

        with self._lock:
            self._connected = reason_code == 0

        if self._connected:
            logger.info(
                "Connected to MQTT broker %s:%s",
                self.config.host,
                self.config.port,
            )
        else:
            logger.error(
                "MQTT connection failed: %s",
                reason_code,
            )

    def _on_disconnect(
        self,
        client: Any,
        userdata: Any,
        disconnect_flags: Any,
        reason_code: Any,
        properties: Any = None,
    ) -> None:
        del client, userdata, disconnect_flags, properties

        with self._lock:
            self._connected = False

        logger.info(
            "Disconnected from MQTT broker: %s",
            reason_code,
        )

    def _on_message(
        self,
        client: Any,
        userdata: Any,
        message: Any,
    ) -> None:
        del client, userdata

        try:
            payload = json.loads(
                message.payload.decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning(
                "Ignoring invalid MQTT JSON payload on %s",
                message.topic,
            )
            return

        if not isinstance(payload, dict):
            logger.warning(
                "Ignoring non-object MQTT payload on %s",
                message.topic,
            )
            return

        handler = self.message_handler

        if handler is not None:
            try:
                handler(message.topic, payload)
            except Exception:
                logger.exception(
                    "MQTT message handler failed for %s",
                    message.topic,
                )


__all__ = [
    "MQTTClient",
    "MQTTConfig",
    "MessageHandler",
]
