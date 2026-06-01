from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, asdict

from .cpu import parse_cpu_range, format_cpu_range


@dataclass
class GpuNumaInfo:
    gpu_idx: int
    source: str
    numa_node: int | None
    cpus: list[int]
    cpu_range: str
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def discover_numa_cpu_nodes() -> dict[int, list[int]]:
    nodes: dict[int, list[int]] = {}
    root = "/sys/devices/system/node"
    try:
        for name in os.listdir(root):
            if not name.startswith("node") or not name[4:].isdigit():
                continue
            cpulist = os.path.join(root, name, "cpulist")
            if os.path.exists(cpulist):
                with open(cpulist, "r", encoding="utf-8", errors="replace") as f:
                    nodes[int(name[4:])] = parse_cpu_range(f.read().strip())
    except Exception:
        pass
    if nodes:
        return nodes

    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=8)
        for line in result.stdout.splitlines():
            m = re.match(r"NUMA node(\d+) CPU\(s\):\s*(.+)", line)
            if m:
                nodes[int(m.group(1))] = parse_cpu_range(m.group(2).strip())
    except Exception:
        pass
    return nodes


def _from_nvidia_smi_topo(gpu_idx: int) -> GpuNumaInfo | None:
    try:
        result = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return GpuNumaInfo(gpu_idx, "nvidia-smi-error", None, [], "", str(exc))
    if result.returncode != 0:
        return None
    target = f"GPU{gpu_idx}"
    for line in result.stdout.splitlines():
        if not line.strip().startswith(target):
            continue
        for part in line.split():
            if "-" in part and part[0].isdigit():
                try:
                    cpus = parse_cpu_range(part)
                except Exception:
                    continue
                if len(cpus) > 1:
                    return GpuNumaInfo(gpu_idx, "nvidia-smi topo", None, cpus, format_cpu_range(cpus), line.strip())
    return None


def _from_sysfs(gpu_idx: int) -> GpuNumaInfo | None:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        pci = pynvml.nvmlDeviceGetPciInfo(handle)
        bus_id = pci.busId.decode("utf-8") if isinstance(pci.busId, bytes) else str(pci.busId)
        candidates = [
            f"/sys/bus/pci/devices/{bus_id.lower()}/numa_node",
            f"/sys/bus/pci/devices/0000:{bus_id.lower().replace('0000:', '')}/numa_node",
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                numa_node = int(f.read().strip())
            if numa_node < 0:
                continue
            cpulist = f"/sys/devices/system/node/node{numa_node}/cpulist"
            with open(cpulist, "r", encoding="utf-8", errors="replace") as f:
                cpus = parse_cpu_range(f.read().strip())
            return GpuNumaInfo(gpu_idx, "sysfs", numa_node, cpus, format_cpu_range(cpus), bus_id)
    except Exception:
        return None
    return None


def _from_lscpu(gpu_idx: int) -> GpuNumaInfo | None:
    nodes = discover_numa_cpu_nodes()
    if not nodes:
        return None
    count = len(nodes)
    gpus_per_node = max(1, 8 // count)
    target_node = min(sorted(nodes)[-1], gpu_idx // gpus_per_node)
    cpus = nodes.get(target_node)
    if not cpus:
        return None
    return GpuNumaInfo(gpu_idx, "lscpu heuristic", target_node, cpus, format_cpu_range(cpus), f"{count} NUMA nodes")


def discover_gpu_numa(gpu_idx: int, configured_map: dict | None = None) -> GpuNumaInfo:
    configured_map = configured_map or {}
    if gpu_idx in configured_map or str(gpu_idx) in configured_map:
        item = configured_map.get(gpu_idx) or configured_map.get(str(gpu_idx))
        cpus = parse_cpu_range(item["cpus"])
        return GpuNumaInfo(gpu_idx, "configured", item.get("numa_node"), cpus, format_cpu_range(cpus))

    for fn in [_from_nvidia_smi_topo, _from_sysfs, _from_lscpu]:
        info = fn(gpu_idx)
        if info and info.cpus:
            return info

    total = os.cpu_count() or 1
    cpus = list(range(max(1, total // 2)))
    return GpuNumaInfo(gpu_idx, "fallback-first-half", None, cpus, format_cpu_range(cpus))


def infer_remote_numa_cpus(local_cpus: list[int]) -> list[int]:
    nodes = discover_numa_cpu_nodes()
    if not nodes:
        return []
    local = set(local_cpus)
    best_node = max(nodes, key=lambda node: len(local & set(nodes[node])))
    remote: list[int] = []
    for node, cpus in nodes.items():
        if node != best_node:
            remote.extend(cpus)
    return sorted(set(remote))

