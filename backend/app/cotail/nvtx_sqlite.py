from __future__ import annotations

import sqlite3
from pathlib import Path

from .constants import ALL_VLLM_STAGES


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({_quote(table)})")]


def _pick(cols: list[str], names: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for name in names:
        if name.lower() in low:
            return low[name.lower()]
    return None


def _find_nvtx_table(conn: sqlite3.Connection) -> str | None:
    preferred = ["NVTX_EVENTS", "NVTX_PUSHPOP_RANGE", "NVTX_PUSHPOP_RANGES", "NVTX_RANGE", "NVTX_RANGES"]
    lower = {t.lower(): t for t in _tables(conn)}
    for name in preferred:
        if name.lower() in lower:
            return lower[name.lower()]
    for table in lower.values():
        if "nvtx" in table.lower():
            return table
    return None


def _string_ids(conn: sqlite3.Connection) -> dict[int, str]:
    tables = {t.lower(): t for t in _tables(conn)}
    table = tables.get("stringids")
    if not table:
        return {}
    cols = _columns(conn, table)
    if "id" not in cols or "value" not in cols:
        return {}
    return {int(row[0]): str(row[1]) for row in conn.execute(f"SELECT id, value FROM {_quote(table)}")}


def _resolve(mapping: dict[int, str], value) -> str:
    if value is None:
        return ""
    try:
        return mapping.get(int(value), str(value))
    except Exception:
        return str(value)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    low = int(pos)
    high = min(low + 1, len(vals) - 1)
    frac = pos - low
    return vals[low] * (1 - frac) + vals[high] * frac


def summarize_nsys_sqlite(path: str | Path) -> dict:
    sqlite_path = Path(path)
    conn = sqlite3.connect(str(sqlite_path))
    try:
        table = _find_nvtx_table(conn)
        if not table:
            return {"ok": False, "error": "no NVTX table found", "stages": {}}
        cols = _columns(conn, table)
        start_col = _pick(cols, ["start", "startTime", "timestamp"])
        end_col = _pick(cols, ["end", "endTime"])
        text_col = _pick(cols, ["text", "message", "name", "range", "label"])
        text_id_col = _pick(cols, ["textId", "messageId", "nameId"])
        if not start_col or not end_col:
            return {"ok": False, "error": f"missing start/end columns in {table}", "stages": {}}
        select = [start_col, end_col]
        if text_col:
            select.append(text_col)
        elif text_id_col:
            select.append(text_id_col)
        else:
            return {"ok": False, "error": f"missing NVTX text column in {table}", "stages": {}}
        mapping = _string_ids(conn)
        sql = f"SELECT {', '.join(_quote(c) for c in select)} FROM {_quote(table)} WHERE {_quote(end_col)} > {_quote(start_col)}"
        per_stage: dict[str, list[float]] = {stage: [] for stage in ALL_VLLM_STAGES}
        total = 0
        for row in conn.execute(sql):
            start = float(row[0])
            end = float(row[1])
            name = str(row[2]) if text_col else _resolve(mapping, row[2])
            if "vllm." not in name:
                continue
            dur_us = (end - start) / 1000.0
            total += 1
            for stage in ALL_VLLM_STAGES:
                if stage in name:
                    per_stage[stage].append(dur_us)
        stages = {}
        for stage, vals in per_stage.items():
            stages[stage] = {
                "count": len(vals),
                "avg_us": sum(vals) / len(vals) if vals else 0.0,
                "p50_us": _percentile(vals, 0.50),
                "p95_us": _percentile(vals, 0.95),
                "p99_us": _percentile(vals, 0.99),
                "max_us": max(vals) if vals else 0.0,
            }
        return {"ok": True, "file": str(sqlite_path), "nvtx_table": table, "vllm_event_count": total, "stages": stages}
    finally:
        conn.close()

