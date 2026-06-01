from __future__ import annotations

import time
from typing import Any

from .gpu_monitor import collect_gpu_status
from .llm_client import benchmark_chat
from .runtime import system_snapshot


def _psutil_counters() -> dict[str, Any]:
    try:
        import psutil  # type: ignore

        stats = psutil.cpu_stats()
        disk = psutil.disk_io_counters()
        net = psutil.net_io_counters()
        return {
            "ok": True,
            "ctx_switches": float(getattr(stats, "ctx_switches", 0.0)),
            "disk_bytes": float((getattr(disk, "read_bytes", 0.0) or 0.0) + (getattr(disk, "write_bytes", 0.0) or 0.0)),
            "net_bytes": float((getattr(net, "bytes_recv", 0.0) or 0.0) + (getattr(net, "bytes_sent", 0.0) or 0.0)),
            "cpu_percent": float(psutil.cpu_percent(interval=0.05)),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _rate(after: dict, before: dict, key: str, elapsed: float) -> float:
    if not after.get("ok") or not before.get("ok"):
        return 0.0
    return max(0.0, float(after.get(key, 0.0)) - float(before.get(key, 0.0))) / max(elapsed, 1e-6)


def _metric_from_benchmark(benchmark: dict[str, Any]) -> dict[str, float | None]:
    return {
        "throughput_tok_s": benchmark.get("throughput_tok_s"),
        "ttft_avg_ms": benchmark.get("ttft_avg_ms"),
        "tpot_avg_ms": benchmark.get("tpot_avg_ms"),
        "request_total_avg_ms": benchmark.get("request_total_avg_ms"),
    }


def collect_canary_measurement(
    *,
    phase: str,
    workload: str,
    api_base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
    timeout_s: float,
    concurrency: int,
    total_requests: int,
    run_benchmark: bool = True,
    sample_seconds: float = 2.0,
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    started_at = time.time()
    before_sys = system_snapshot().to_dict()
    before = _psutil_counters()

    benchmark: dict[str, Any]
    if run_benchmark:
        benchmark = benchmark_chat(
            api_base=api_base,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            timeout_s=timeout_s,
            concurrency=concurrency,
            total_requests=total_requests,
        )
    else:
        time.sleep(max(0.1, float(sample_seconds)))
        benchmark = {"ok": False, "skipped": True, "message": "benchmark disabled"}

    after = _psutil_counters()
    after_sys = system_snapshot().to_dict()
    elapsed = max(time.perf_counter() - wall_start, 1e-6)

    ctx_switches_per_s = _rate(after, before, "ctx_switches", elapsed)
    disk_mbps = _rate(after, before, "disk_bytes", elapsed) / 1024**2
    net_mbps = _rate(after, before, "net_bytes", elapsed) / 1024**2
    cpu_before = float(before.get("cpu_percent") or before_sys.get("cpu_percent") or 0.0)
    cpu_after = float(after.get("cpu_percent") or after_sys.get("cpu_percent") or 0.0)
    cpu_delta = max(cpu_before, cpu_after)

    return {
        "ok": bool(benchmark.get("ok")) if run_benchmark else True,
        "phase": phase,
        "workload": workload,
        "started_at": started_at,
        "duration_s": elapsed,
        "api_base": api_base,
        "model": model,
        "benchmark": benchmark,
        "metric": _metric_from_benchmark(benchmark),
        "workload_profile": {
            "cpu_delta_pct": round(cpu_delta, 3),
            "ctx_switches_per_s": round(ctx_switches_per_s, 3),
            "disk_MBps": round(disk_mbps, 3),
            "loopback_MBps": round(net_mbps, 3),
            "llc_miss_pct": 0.0,
            "l1d_miss_pct": 0.0,
            "memcache_proxy_GBps": 0.0,
        },
        "system_before": before_sys,
        "system_after": after_sys,
        "gpu": collect_gpu_status(),
        "diagnostic_overhead": {
            "canary_wall_time_s": round(elapsed, 3),
            "canary_requests": int(total_requests) if run_benchmark else 0,
            "estimated_canary_overhead_pct": round(min(5.0, 0.15 * max(1, int(total_requests))), 3) if run_benchmark else 0.0,
            "nsight_trace_overhead_pct": None,
        },
    }
