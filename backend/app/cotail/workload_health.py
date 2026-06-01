from __future__ import annotations

import shutil
import os
import time
from typing import Any

from .constants import WORKLOADS
from .workload_runner import build_workload_command


SIGNATURES: dict[str, dict[str, Any]] = {
    "stress-ng": {"keywords": ["--workload stress-ng", "stress-ng"], "cpu_weight": 3, "required_any": [["stress-ng"]]},
    "ffmpeg": {"keywords": ["--workload ffmpeg", "ffmpeg", "test_input_benchmark.mp4"], "cpu_weight": 3, "required_any": [["ffmpeg"]]},
    "7zip": {"keywords": ["--workload 7zip", "7z"], "cpu_weight": 3, "required_any": [["7z"]]},
    "redis": {"keywords": ["--workload redis", "redis-server", "redis-benchmark"], "cpu_weight": 2, "required_any": [["redis-server"], ["redis-benchmark"]]},
    "openssl": {"keywords": ["--workload openssl", "openssl", "aes-256-cbc"], "cpu_weight": 3, "required_any": [["openssl"]]},
    "kernel_build": {"keywords": ["--workload kernel_build", "gcc", "cc1", "_kbuild_", "_linux_src", "stress-ng"], "cpu_weight": 3, "required_any": [["gcc"], ["python3"]]},
    "memcached": {"keywords": ["--workload memcached", "memcached", "_mc_bench.py", "11299"], "cpu_weight": 2, "required_any": [["memcached"], ["python3"]]},
    "nginx": {"keywords": ["--workload nginx", "nginx", "wrk", "_nginx_www", "_nginx_bench.conf"], "cpu_weight": 2, "required_any": [["nginx"], ["wrk", "ab", "python3"]]},
    "zstd-compress": {"keywords": ["--workload zstd-compress", "zstd", "xz ", "gzip ", "cotail_heldout/compress"], "cpu_weight": 3, "required_any": [["python3"], ["zstd", "xz", "gzip", "openssl"]]},
    "sqlite-txn": {"keywords": ["--workload sqlite-txn", "cotail_heldout/sqlite", "_sqlite_txn_loop.py"], "cpu_weight": 2, "required_any": [["python3"]]},
    "image-preprocess": {"keywords": ["--workload image-preprocess", "image_preprocess_loop.py", "cotail_heldout/images"], "cpu_weight": 3, "required_any": [["python3"]]},
    "text-search": {"keywords": ["--workload text-search", "rg ", "grep ", "cotail_heldout/text"], "cpu_weight": 2, "required_any": [["python3"], ["rg", "grep"]]},
}


def _binary_group_status(groups: list[list[str]]) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        found = {name: shutil.which(name) for name in group}
        rows.append(
            {
                "any_of": group,
                "ok": any(found.values()),
                "found": {name: path for name, path in found.items() if path},
                "missing": [name for name, path in found.items() if not path],
            }
        )
    return rows


def workload_readiness(workload: str) -> dict[str, Any]:
    w = workload.strip().lower()
    command = build_workload_command(w, workers=60, duration_s=60)
    signature = SIGNATURES.get(w, {"keywords": [w], "required_any": [[name] for name in command.required_binaries]})
    groups = []
    for base_binary in ("bash", "taskset"):
        if base_binary in command.required_binaries:
            groups.append([base_binary])
    groups.extend(signature.get("required_any") or [[name] for name in command.required_binaries])
    binary_groups = _binary_group_status(groups)
    return {
        "workload": w,
        "ready": all(group["ok"] for group in binary_groups),
        "binary_groups": binary_groups,
        "command": command.to_dict(),
        "expected_keywords": signature.get("keywords", []),
        "notes": command.notes,
    }


def _safe_cmdline(proc: Any) -> str:
    try:
        cmdline = proc.cmdline()
        if cmdline:
            return " ".join(str(x) for x in cmdline)
    except Exception:
        pass
    try:
        return str(proc.name())
    except Exception:
        return ""


def _proc_cpu_time(proc: Any) -> float | None:
    try:
        times = proc.cpu_times()
        return float(getattr(times, "user", 0.0) or 0.0) + float(getattr(times, "system", 0.0) or 0.0)
    except Exception:
        return None


def _proc_row(proc: Any, cpu_percent: float = 0.0) -> dict[str, Any] | None:
    try:
        return {
            "pid": int(proc.pid),
            "name": proc.name(),
            "status": proc.status(),
            "cpu_percent": round(float(cpu_percent), 3),
            "cmdline": _safe_cmdline(proc)[:500],
        }
    except Exception:
        return None


def _job_processes(job: dict[str, Any] | None) -> list[Any]:
    if not job:
        return []
    try:
        import psutil  # type: ignore
    except Exception:
        return []
    procs = []
    seen: set[int] = set()
    for item in job.get("processes") or []:
        pid = item.get("pid")
        if not pid:
            continue
        try:
            root = psutil.Process(int(pid))
            candidates = [root] + root.children(recursive=True)
            for proc in candidates:
                if proc.pid not in seen:
                    procs.append(proc)
                    seen.add(proc.pid)
        except Exception:
            continue
    return procs


def _matching_processes(workload: str) -> list[Any]:
    try:
        import psutil  # type: ignore
    except Exception:
        return []
    keywords = [str(x).lower() for x in SIGNATURES.get(workload, {}).get("keywords", [workload])]
    rows = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.pid == os.getpid():
            continue
        text = ""
        try:
            name = proc.info.get("name") or ""
            cmdline = " ".join(proc.info.get("cmdline") or [])
            text = f"{name} {cmdline}".lower()
        except Exception:
            continue
        if any(key and key in text for key in keywords):
            rows.append(proc)
    return rows


def _counter_snapshot() -> dict[str, float]:
    try:
        import psutil  # type: ignore

        stats = psutil.cpu_stats()
        disk = psutil.disk_io_counters()
        net = psutil.net_io_counters()
        return {
            "ctx": float(getattr(stats, "ctx_switches", 0.0) or 0.0),
            "disk": float((getattr(disk, "read_bytes", 0.0) or 0.0) + (getattr(disk, "write_bytes", 0.0) or 0.0)),
            "net": float((getattr(net, "bytes_recv", 0.0) or 0.0) + (getattr(net, "bytes_sent", 0.0) or 0.0)),
        }
    except Exception:
        return {"ctx": 0.0, "disk": 0.0, "net": 0.0}


def _rate(after: dict[str, float], before: dict[str, float], key: str, elapsed: float) -> float:
    return max(0.0, float(after.get(key, 0.0)) - float(before.get(key, 0.0))) / max(elapsed, 1e-6)


def evaluate_workload(
    workload: str,
    *,
    job: dict[str, Any] | None = None,
    sample_seconds: float = 1.5,
) -> dict[str, Any]:
    w = workload.strip().lower()
    readiness = workload_readiness(w)
    signature = SIGNATURES.get(w, {"keywords": [w], "cpu_weight": 2})
    try:
        import psutil  # type: ignore
    except Exception as exc:
        return {
            **readiness,
            "ok": False,
            "active": False,
            "verdict": "psutil_missing",
            "message": f"psutil is required for workload health checks: {exc}",
        }

    job_procs = _job_processes(job)
    procs = job_procs or _matching_processes(w)
    before_proc_cpu = {proc.pid: value for proc in procs if (value := _proc_cpu_time(proc)) is not None}
    psutil.cpu_percent(interval=None)
    before = _counter_snapshot()
    start = time.perf_counter()
    time.sleep(max(0.05, float(sample_seconds)))
    system_cpu = float(psutil.cpu_percent(interval=None) or 0.0)
    after = _counter_snapshot()
    elapsed = max(time.perf_counter() - start, 1e-6)

    refreshed = _job_processes(job) if job else _matching_processes(w)
    process_rows = []
    total_proc_cpu = 0.0
    for proc in refreshed:
        after_cpu = _proc_cpu_time(proc)
        if after_cpu is None:
            proc_cpu_pct = 0.0
        elif proc.pid in before_proc_cpu:
            proc_cpu_pct = max(0.0, after_cpu - before_proc_cpu[proc.pid]) * 100.0 / elapsed
        else:
            try:
                age = max(elapsed, time.time() - float(proc.create_time()))
                proc_cpu_pct = max(0.0, after_cpu) * 100.0 / age
            except Exception:
                proc_cpu_pct = 0.0
        row = _proc_row(proc, proc_cpu_pct)
        if not row:
            continue
        total_proc_cpu += float(row.get("cpu_percent") or 0.0)
        process_rows.append(row)

    ctx_per_s = _rate(after, before, "ctx", elapsed)
    disk_mbps = _rate(after, before, "disk", elapsed) / 1024**2
    net_mbps = _rate(after, before, "net", elapsed) / 1024**2

    root_statuses = []
    if job:
        for item in job.get("processes") or []:
            if item.get("role") == "cpu_workload":
                root_statuses.append(
                    {
                        "pid": item.get("pid"),
                        "status": item.get("status"),
                        "return_code": item.get("return_code"),
                    }
                )

    process_present = bool(process_rows)
    exited = bool(root_statuses) and all(item.get("return_code") is not None or item.get("status") in {"exited", "failed_to_start"} for item in root_statuses)
    cpu_signal = total_proc_cpu >= 15.0 or system_cpu >= 20.0
    ctx_signal = ctx_per_s >= 50000.0
    io_signal = disk_mbps >= 2.0 or net_mbps >= 5.0

    score = 0
    if process_present:
        score += 1
    if cpu_signal:
        score += int(signature.get("cpu_weight", 2))
    if ctx_signal:
        score += 1
    if io_signal:
        score += 1

    missing_groups = [group for group in readiness["binary_groups"] if not group["ok"]]
    if missing_groups:
        verdict = "missing_dependency"
        active = False
        message = "Required binary group is missing."
    elif exited and not process_present:
        verdict = "exited"
        active = False
        message = "The managed workload process exited before or during the health sample."
    elif not process_present:
        verdict = "not_running"
        active = False
        message = "No matching workload process is visible."
    elif score >= 4:
        verdict = "strong"
        active = True
        message = "Workload is running and producing visible CPU/CTX/I/O pressure."
    elif score >= 2:
        verdict = "weak"
        active = True
        message = "Workload is running, but the observed pressure is weak."
    else:
        verdict = "idle"
        active = False
        message = "Processes exist, but no meaningful pressure was observed."

    hints = []
    if w == "nginx" and verdict in {"weak", "idle", "not_running"}:
        hints.append("nginx needs active clients; install wrk or ab, or use the built-in Python client fallback.")
    if w == "kernel_build" and verdict in {"weak", "idle", "not_running"}:
        hints.append("kernel_build now uses a self-contained gcc compile loop; make sure gcc exists and workers are high enough.")
    if w in {"ffmpeg", "zstd-compress", "text-search"} and verdict in {"weak", "idle", "not_running"}:
        hints.append("This workload generates its own input corpus on first run; the first probe may spend time preparing data.")
    if not readiness["ready"]:
        hints.append("Install the missing dependency group shown in binary_groups.")

    return {
        **readiness,
        "ok": active,
        "active": active,
        "verdict": verdict,
        "message": message,
        "hints": hints,
        "sample_seconds": round(elapsed, 3),
        "signals": {
            "process_count": len(process_rows),
            "process_cpu_percent_sum": round(total_proc_cpu, 3),
            "system_cpu_percent": round(system_cpu, 3),
            "ctx_switches_per_s": round(ctx_per_s, 3),
            "disk_MBps": round(disk_mbps, 3),
            "net_MBps": round(net_mbps, 3),
            "score": score,
        },
        "job": {
            "id": job.get("id") if job else None,
            "processes": root_statuses,
            "notes": job.get("notes") if job else [],
        },
        "processes": sorted(process_rows, key=lambda row: -float(row.get("cpu_percent") or 0.0))[:30],
    }


def readiness_matrix() -> dict[str, Any]:
    rows = [workload_readiness(workload) for workload in WORKLOADS]
    return {"ok": True, "workloads": rows}
