from __future__ import annotations

import csv
import io
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .runtime import run_command


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in ("N/A", "[N/A]", "Not Supported", "[Not Supported]"):
        return None
    text = text.replace("MiB", "").replace("W", "").replace("%", "").strip()
    try:
        return float(text)
    except Exception:
        return None


def _int(value: str | None) -> int | None:
    num = _float(value)
    return None if num is None else int(round(num))


@dataclass
class GPUProcess:
    gpu_index: int | None
    gpu_uuid: str | None
    pid: int
    process_type: str
    process_name: str
    used_memory_mib: int | None
    username: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GPUStatus:
    index: int
    uuid: str | None
    name: str
    bus_id: str | None
    pstate: str | None
    temperature_c: int | None
    power_draw_w: float | None
    power_limit_w: float | None
    memory_used_mib: int
    memory_total_mib: int
    utilization_gpu_pct: int | None
    processes: list[GPUProcess] = field(default_factory=list)

    @property
    def memory_free_mib(self) -> int:
        return max(0, self.memory_total_mib - self.memory_used_mib)

    @property
    def compute_process_count(self) -> int:
        return sum(1 for p in self.processes if "C" in p.process_type.upper())

    @property
    def is_likely_idle(self) -> bool:
        util = self.utilization_gpu_pct or 0
        return util <= 10 and self.compute_process_count == 0 and self.memory_free_mib >= 18_000

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["memory_free_mib"] = self.memory_free_mib
        data["compute_process_count"] = self.compute_process_count
        data["is_likely_idle"] = self.is_likely_idle
        data["memory_used_pct"] = (
            round(100.0 * self.memory_used_mib / self.memory_total_mib, 2)
            if self.memory_total_mib
            else None
        )
        return data


GPU_QUERY_FIELDS = [
    "index",
    "uuid",
    "name",
    "pci.bus_id",
    "pstate",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "memory.used",
    "memory.total",
    "utilization.gpu",
]


def _parse_csv_lines(text: str) -> list[list[str]]:
    if not text.strip():
        return []
    return [row for row in csv.reader(io.StringIO(text), skipinitialspace=True)]


def parse_query_gpu_csv(text: str) -> list[GPUStatus]:
    gpus: list[GPUStatus] = []
    for row in _parse_csv_lines(text):
        if len(row) < len(GPU_QUERY_FIELDS):
            continue
        gpus.append(
            GPUStatus(
                index=int(row[0]),
                uuid=row[1].strip() or None,
                name=row[2].strip(),
                bus_id=row[3].strip() or None,
                pstate=row[4].strip() or None,
                temperature_c=_int(row[5]),
                power_draw_w=_float(row[6]),
                power_limit_w=_float(row[7]),
                memory_used_mib=_int(row[8]) or 0,
                memory_total_mib=_int(row[9]) or 0,
                utilization_gpu_pct=_int(row[10]),
            )
        )
    return gpus


def parse_compute_apps_csv(text: str, uuid_to_index: dict[str, int]) -> list[GPUProcess]:
    procs: list[GPUProcess] = []
    for row in _parse_csv_lines(text):
        if len(row) < 4:
            continue
        uuid = row[0].strip()
        pid = _int(row[1])
        if pid is None:
            continue
        procs.append(
            GPUProcess(
                gpu_index=uuid_to_index.get(uuid),
                gpu_uuid=uuid,
                pid=pid,
                process_type="C",
                process_name=row[2].strip(),
                used_memory_mib=_int(row[3]),
            )
        )
    return procs


PROCESS_LINE_RE = re.compile(
    r"^\|\s*(?P<gpu>\d+)\s+\S+\s+\S+\s+(?P<pid>\d+)\s+(?P<type>[CG+]+)\s+(?P<name>.*?)\s+(?P<mem>\d+)MiB\s*\|"
)


def parse_nvidia_smi_process_table(text: str) -> list[GPUProcess]:
    procs: list[GPUProcess] = []
    for line in text.splitlines():
        match = PROCESS_LINE_RE.match(line)
        if not match:
            continue
        procs.append(
            GPUProcess(
                gpu_index=int(match.group("gpu")),
                gpu_uuid=None,
                pid=int(match.group("pid")),
                process_type=match.group("type"),
                process_name=match.group("name").strip(),
                used_memory_mib=int(match.group("mem")),
            )
        )
    return procs


def _attach_usernames(processes: list[GPUProcess]) -> None:
    try:
        import psutil  # type: ignore
    except Exception:
        return
    for proc in processes:
        try:
            proc.username = psutil.Process(proc.pid).username()
        except Exception:
            pass


def collect_gpu_status() -> dict[str, Any]:
    query_cmd = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    rc, out, err = run_command(query_cmd, timeout_s=6)
    if rc != 0:
        return {
            "ok": False,
            "timestamp": time.time(),
            "error": (err or out or "nvidia-smi query failed").strip(),
            "gpus": [],
            "raw_process_table": "",
        }
    gpus = parse_query_gpu_csv(out)
    uuid_to_index = {gpu.uuid: gpu.index for gpu in gpus if gpu.uuid}

    procs: list[GPUProcess] = []
    rc_apps, apps_out, _ = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=6,
    )
    if rc_apps == 0:
        procs.extend(parse_compute_apps_csv(apps_out, uuid_to_index))

    rc_full, full_out, _ = run_command(["nvidia-smi"], timeout_s=6)
    if rc_full == 0:
        table_procs = parse_nvidia_smi_process_table(full_out)
        seen = {(p.gpu_index, p.pid, p.process_type) for p in procs}
        for proc in table_procs:
            key = (proc.gpu_index, proc.pid, proc.process_type)
            if key not in seen:
                procs.append(proc)
                seen.add(key)
    else:
        full_out = ""

    _attach_usernames(procs)
    by_index: dict[int, list[GPUProcess]] = {}
    for proc in procs:
        if proc.gpu_index is not None:
            by_index.setdefault(proc.gpu_index, []).append(proc)
    for gpu in gpus:
        gpu.processes = sorted(by_index.get(gpu.index, []), key=lambda p: (p.process_type, p.pid))

    return {
        "ok": True,
        "timestamp": time.time(),
        "gpus": [gpu.to_dict() for gpu in gpus],
        "raw_process_table": full_out,
    }
