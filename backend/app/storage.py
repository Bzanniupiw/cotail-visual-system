from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[2] / "storage" / "cotail_visual.db"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                workload TEXT NOT NULL,
                risk TEXT NOT NULL,
                recommended_policy TEXT NOT NULL,
                decision TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_diagnosis(payload: dict[str, Any]) -> dict:
    init_db()
    created_at = datetime.utcnow().isoformat() + "Z"
    workload = str(payload.get("workload") or "unknown")
    risk = str(payload.get("risk", {}).get("risk") or "UNKNOWN")
    recommended = str(payload.get("recommendation", {}).get("candidate_policy") or "none")
    decision = str(payload.get("final", {}).get("decision") or "NEEDS_VALIDATION")
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO diagnoses(created_at, workload, risk, recommended_policy, decision, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (created_at, workload, risk, recommended, decision, json.dumps(payload, ensure_ascii=False)),
        )
        conn.commit()
        payload = dict(payload)
        payload["id"] = cur.lastrowid
        payload["created_at"] = created_at
        return payload


def list_diagnoses(limit: int = 100) -> list[dict]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, created_at, workload, risk, recommended_policy, decision FROM diagnoses ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(row) for row in rows]


def get_diagnosis(diag_id: int) -> dict | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM diagnoses WHERE id = ?", (int(diag_id),)).fetchone()
        if not row:
            return None
        data = json.loads(row["payload_json"])
        data["id"] = row["id"]
        data["created_at"] = row["created_at"]
        return data

