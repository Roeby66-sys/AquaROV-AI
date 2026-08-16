"""
AquaROV AI — Local database manager.

Provides a lightweight SQLite persistence layer for AquaROV AI.
The database file itself is runtime data and should NOT be committed
to the Git repository.

Project: AquaROV AI - Underwater ROV Inspection System
Target: Axelera Metis + Voyager SDK deployments.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterator


DEFAULT_DB_FILENAME = "aquarov.db"


class Database:
    """Thread-safe SQLite database manager for AquaROV AI."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._lock = Lock()

    @property
    def db_path(self) -> Path:
        """Return the configured database path."""
        return self._db_path

    def initialize(self) -> None:
        """Create the database and required base tables."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    x1 REAL NOT NULL,
                    y1 REAL NOT NULL,
                    x2 REAL NOT NULL,
                    y2 REAL NOT NULL,
                    timestamp REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    acknowledged INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS recordings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    duration_seconds REAL
                );

                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );
                """
            )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Provide a transactional SQLite connection."""
        with self._lock:
            connection = sqlite3.connect(
                self._db_path,
                timeout=30.0,
            )
            connection.row_factory = sqlite3.Row

            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def execute(
        self,
        sql: str,
        parameters: tuple[object, ...] = (),
    ) -> int:
        """Execute a single SQL statement and return the affected row count."""
        with self.connection() as connection:
            cursor = connection.execute(sql, parameters)
            return cursor.rowcount

    def fetch_all(
        self,
        sql: str,
        parameters: tuple[object, ...] = (),
    ) -> list[sqlite3.Row]:
        """Execute a query and return all rows."""
        with self.connection() as connection:
            cursor = connection.execute(sql, parameters)
            return cursor.fetchall()

    def fetch_one(
        self,
        sql: str,
        parameters: tuple[object, ...] = (),
    ) -> sqlite3.Row | None:
        """Execute a query and return one row, if available."""
        with self.connection() as connection:
            cursor = connection.execute(sql, parameters)
            return cursor.fetchone()


def create_database(db_path: str | Path) -> Database:
    """Create and initialize a Database instance."""
    database = Database(db_path)
    database.initialize()
    return database
