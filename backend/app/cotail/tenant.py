from __future__ import annotations

from typing import Any

from .constants import WORKLOADS, normalize_policy


def _cmdline(proc) -> str:
    try:
        return " ".join(proc.cmdline())
    except Exception:
        return ""


def _detect_workload(text: str) -> str:
    lower = text.lower()
    aliases = {
        "stress-ng": ["stress-ng"],
        "kernel_build": ["make ", "ninja", "gcc", "g++", "clang", "kernel_build"],
        "zstd-compress": ["zstd"],
        "sqlite-txn": ["sqlite", "sqlite3"],
        "image-preprocess": ["pillow", "opencv", "cv2", "image-preprocess"],
        "text-search": ["ripgrep", " rg ", "grep", "text-search"],
    }
    for workload in WORKLOADS:
        if workload in lower:
            return workload
    for workload, keys in aliases.items():
        if any(key in lower for key in keys):
            return workload
    if "nginx" in lower:
        return "nginx"
    if "redis" in lower:
        return "redis"
    if "memcached" in lower:
        return "memcached"
    return "unknown"


def discover_cpu_tenants(limit: int = 30) -> dict[str, Any]:
    try:
        import psutil  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"psutil unavailable: {exc}", "tenants": []}

    rows = []
    for proc in psutil.process_iter(["pid", "name", "username", "memory_info", "cpu_percent"]):
        try:
            if int(proc.pid) <= 0:
                continue
            name = proc.info.get("name") or proc.name()
            if "idle process" in str(name).lower():
                continue
            text = f"{name} {_cmdline(proc)}"
            workload = _detect_workload(text)
            cpu = min(100.0 * max(1, psutil.cpu_count() or 1), float(proc.info.get("cpu_percent") or proc.cpu_percent(interval=None) or 0.0))
            rss_mb = float(getattr(proc.info.get("memory_info"), "rss", 0.0) or 0.0) / 1024**2
            if workload == "unknown" and cpu < 2.0:
                continue
            rows.append(
                {
                    "pid": int(proc.pid),
                    "user": proc.info.get("username") or "",
                    "name": name,
                    "workload": workload,
                    "cpu_percent": round(cpu, 3),
                    "rss_mb": round(rss_mb, 1),
                    "cmdline": text[:500],
                }
            )
        except Exception:
            continue
    rows.sort(key=lambda item: (item["workload"] == "unknown", -item["cpu_percent"], item["pid"]))
    return {"ok": True, "tenants": rows[: max(1, int(limit))]}


def estimate_cost(policy: str, tenants: list[dict[str, Any]], protected_pids: list[int] | None = None) -> dict[str, Any]:
    normalized = normalize_policy(policy)
    protected = set(int(pid) for pid in (protected_pids or []))
    active = [t for t in tenants if int(t.get("pid", -1)) not in protected]
    cpu_pressure = sum(float(t.get("cpu_percent") or 0.0) for t in active)
    known_count = sum(1 for t in active if t.get("workload") != "unknown")
    base = {
        "none": 0.0,
        "nice": 6.0,
        "cgroup": 9.0,
        "rt": 5.5,
        "numa": 3.0,
        "rt+numa": 7.5,
    }.get(normalized, 4.0)
    pressure_penalty = min(18.0, cpu_pressure / 40.0)
    tenant_penalty = min(8.0, known_count * 0.7)
    slowdown = base + pressure_penalty + tenant_penalty
    return {
        "policy": normalized,
        "tenant_count": len(active),
        "known_workload_count": known_count,
        "cpu_pressure_pct": round(cpu_pressure, 3),
        "estimated_cotenant_slowdown_pct": round(slowdown, 3),
        "cost_class": "HIGH" if slowdown >= 18 else "MEDIUM" if slowdown >= 8 else "LOW",
        "notes": [
            "RT protects EngineCore latency but may preempt CPU tenants.",
            "NUMA isolation usually has lower tenant cost when spare remote cores exist.",
            "The estimate is online heuristic; use tenant-side throughput probes for billing-grade numbers.",
        ],
    }


def diagnostic_overhead_summary(last_measurement: dict[str, Any] | None = None) -> dict[str, Any]:
    overhead = (last_measurement or {}).get("diagnostic_overhead") or {}
    canary = float(overhead.get("estimated_canary_overhead_pct") or 0.0)
    nsys = overhead.get("nsight_trace_overhead_pct")
    return {
        "canary_overhead_pct": round(canary, 3),
        "nsight_trace_overhead_pct": nsys,
        "recommended_trace_interval_s": 300 if nsys is None else 600,
        "risk": "HIGH" if canary >= 5.0 or (nsys is not None and nsys >= 5.0) else "LOW",
    }
