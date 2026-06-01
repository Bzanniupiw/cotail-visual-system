from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any


def run_command(cmd: list[str], timeout_s: float = 5.0) -> tuple[int | None, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:
        return None, "", str(exc)


@dataclass
class SystemSnapshot:
    hostname: str
    os: str
    kernel: str
    cpu_count: int
    loadavg: list[float]
    cpu_percent: float | None
    memory_total_gb: float | None
    memory_used_gb: float | None
    memory_percent: float | None
    disk_percent: float | None
    timestamp: float
    tools: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _psutil_snapshot() -> dict[str, Any]:
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "cpu_percent": float(psutil.cpu_percent(interval=0.05)),
            "memory_total_gb": round(mem.total / 1024**3, 3),
            "memory_used_gb": round(mem.used / 1024**3, 3),
            "memory_percent": float(mem.percent),
            "disk_percent": float(disk.percent),
        }
    except Exception:
        return {
            "cpu_percent": None,
            "memory_total_gb": None,
            "memory_used_gb": None,
            "memory_percent": None,
            "disk_percent": None,
        }


def system_snapshot() -> SystemSnapshot:
    try:
        loadavg = [float(x) for x in os.getloadavg()]
    except Exception:
        loadavg = []
    ps = _psutil_snapshot()
    tools = {
        "nvidia-smi": shutil.which("nvidia-smi") is not None,
        "numactl": shutil.which("numactl") is not None,
        "taskset": shutil.which("taskset") is not None,
        "chrt": shutil.which("chrt") is not None,
        "systemd-run": shutil.which("systemd-run") is not None,
        "perf": shutil.which("perf") is not None,
    }
    return SystemSnapshot(
        hostname=platform.node(),
        os=platform.system(),
        kernel=platform.release(),
        cpu_count=os.cpu_count() or 0,
        loadavg=loadavg,
        timestamp=time.time(),
        tools=tools,
        **ps,
    )

