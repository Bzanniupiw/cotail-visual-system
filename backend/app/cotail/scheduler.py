from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GpuSelectionPolicy:
    min_free_memory_mib: int = 18_000
    max_gpu_util_pct: int = 10
    max_compute_processes: int = 0
    allow_graphics_processes: bool = True
    exclude_gpu_ids: list[int] | None = None


def score_gpu(gpu: dict, policy: GpuSelectionPolicy) -> tuple[bool, float, list[str]]:
    reasons: list[str] = []
    index = int(gpu.get("index", -1))
    excluded = set(policy.exclude_gpu_ids or [])
    if index in excluded:
        reasons.append("excluded by policy")
    free_mem = int(gpu.get("memory_free_mib") or 0)
    util = int(gpu.get("utilization_gpu_pct") or 0)
    compute_count = int(gpu.get("compute_process_count") or 0)
    graphics_count = sum(1 for p in gpu.get("processes", []) if "G" in str(p.get("process_type", "")).upper())

    if free_mem < policy.min_free_memory_mib:
        reasons.append(f"free memory {free_mem} MiB < required {policy.min_free_memory_mib} MiB")
    if util > policy.max_gpu_util_pct:
        reasons.append(f"GPU util {util}% > allowed {policy.max_gpu_util_pct}%")
    if compute_count > policy.max_compute_processes:
        reasons.append(f"compute processes {compute_count} > allowed {policy.max_compute_processes}")
    if graphics_count and not policy.allow_graphics_processes:
        reasons.append("graphics process present")

    eligible = not reasons
    score = free_mem - util * 256 - compute_count * 50_000 - graphics_count * 100
    return eligible, float(score), reasons


def select_idle_gpu(gpus: list[dict], policy: GpuSelectionPolicy | None = None) -> dict[str, Any]:
    policy = policy or GpuSelectionPolicy()
    rows = []
    for gpu in gpus:
        eligible, score, reasons = score_gpu(gpu, policy)
        rows.append(
            {
                "gpu_index": int(gpu.get("index", -1)),
                "eligible": eligible,
                "score": score,
                "reasons": reasons,
                "free_memory_mib": int(gpu.get("memory_free_mib") or 0),
                "utilization_gpu_pct": int(gpu.get("utilization_gpu_pct") or 0),
                "compute_process_count": int(gpu.get("compute_process_count") or 0),
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    selected = max(eligible_rows, key=lambda row: row["score"]) if eligible_rows else None
    return {
        "selected_gpu": None if selected is None else selected["gpu_index"],
        "eligible_count": len(eligible_rows),
        "policy": asdict(policy),
        "candidates": sorted(rows, key=lambda row: row["score"], reverse=True),
    }
