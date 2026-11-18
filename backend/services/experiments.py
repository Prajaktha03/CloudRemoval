import json
import sqlite3
from pathlib import Path
from typing import Any


class ExperimentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    metrics_json TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def add(self, kind: str, metrics: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO experiments (kind, metrics_json) VALUES (?, ?)",
                (kind, json.dumps(metrics, allow_nan=False)),
            )

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, kind, created_at, metrics_json FROM experiments ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "kind": row["kind"],
                "created_at": row["created_at"],
                "metrics": json.loads(row["metrics_json"]),
            }
            for row in rows
        ]
