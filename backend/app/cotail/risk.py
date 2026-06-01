from __future__ import annotations

from .constants import STAGE1_THRESHOLDS


def _value(profile: dict, *names: str) -> float:
    for name in names:
        if name in profile and profile[name] not in ("", None):
            try:
                return float(profile[name])
            except Exception:
                pass
    return 0.0


def screen_hardware_risk(profile: dict, thresholds: dict | None = None) -> dict:
    th = dict(STAGE1_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    cpu = _value(profile, "cpu_delta_pct", "cpu_high_delta_pct", "hw_cpu_usr_pct")
    ctx = _value(profile, "ctx_switches_per_s", "context_switches_per_s", "hw_context_switches_per_s")
    disk = _value(profile, "disk_MBps", "disk_mb_s", "hw_disk_read_MBps")
    loopback = _value(profile, "loopback_MBps", "loopback_mb_s", "net_loopback_MBps")
    llc = _value(profile, "llc_miss_pct", "hw_llc_load_miss_rate_pct")
    l1d = _value(profile, "l1d_miss_pct", "hw_l1_dcache_miss_rate_pct")
    memproxy = _value(profile, "memcache_proxy_GBps", "mem_bw_GBps", "hw_mem_total_GBps")

    signals = {
        "cpu_high": cpu >= th["cpu_high_delta_pct"],
        "ctx_high": ctx >= th["ctx_high_switches_per_s"],
        "disk_high": disk >= th["disk_high_MBps"],
        "loopback_high": loopback >= th["loopback_high_MBps"],
        "llc_high": llc >= th["llc_high_miss_pct"],
        "l1d_high": l1d >= th["l1d_high_miss_pct"],
        "memcache_proxy_high": memproxy >= th["memcache_high_proxy_GBps"],
    }
    signals["mem_cache_high"] = bool(signals["llc_high"] or signals["l1d_high"] or signals["memcache_proxy_high"])
    signals["io_high"] = bool(signals["disk_high"] or signals["loopback_high"])

    high_groups = sum(
        bool(x)
        for x in [
            signals["cpu_high"],
            signals["ctx_high"],
            signals["mem_cache_high"],
            signals["io_high"],
        ]
    )
    if signals["mem_cache_high"] and (signals["cpu_high"] or signals["ctx_high"] or signals["io_high"]):
        risk = "HIGH"
    elif signals["ctx_high"] and signals["cpu_high"]:
        risk = "HIGH"
    elif high_groups >= 2:
        risk = "HIGH"
    elif high_groups == 1:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    reasons = [name for name, enabled in signals.items() if enabled and name.endswith("_high")]
    return {
        "risk": risk,
        "signals": signals,
        "values": {
            "cpu_delta_pct": cpu,
            "ctx_switches_per_s": ctx,
            "disk_MBps": disk,
            "loopback_MBps": loopback,
            "llc_miss_pct": llc,
            "l1d_miss_pct": l1d,
            "memcache_proxy_GBps": memproxy,
        },
        "reasons": reasons,
        "thresholds": th,
    }


def ttft_dominated(macro: dict, ratio: float) -> bool:
    ttft = float(macro.get("TTFT_Increase_Pct") or 0.0)
    tpot = abs(float(macro.get("TPOT_Increase_Pct") or 0.0))
    thr = abs(float(macro.get("Throughput_Drop_Pct") or 0.0))
    denom = max(tpot, thr, 1.0)
    return ttft / denom >= ratio

