from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

from .constants import normalize_policy
from .macro import compute_macro_degradation
from .risk import screen_hardware_risk


def _to_float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _metric_from_csv_row(row: dict) -> dict:
    return {
        "throughput_tok_s": _to_float(row.get("throughput_tok_s") or row.get("throughput")),
        "ttft_avg_ms": _to_float(row.get("ttft_avg_ms")),
        "tpot_avg_ms": _to_float(row.get("tpot_avg_ms")),
    }


def _profile_from_csv_row(row: dict) -> dict:
    ctx = (_to_float(row.get("hw_voluntary_cswch_per_s")) or 0.0) + (_to_float(row.get("hw_nonvoluntary_cswch_per_s")) or 0.0)
    return {
        "cpu_delta_pct": _to_float(row.get("hw_cpu_usr_pct")) or 0.0,
        "llc_miss_pct": _to_float(row.get("hw_llc_load_miss_rate_pct")) or 0.0,
        "l1d_miss_pct": _to_float(row.get("hw_l1_dcache_miss_rate_pct")) or 0.0,
        "memcache_proxy_GBps": _to_float(row.get("hw_mem_total_GBps")) or 0.0,
        "ctx_switches_per_s": ctx,
        "disk_MBps": (_to_float(row.get("hw_disk_read_MBps")) or 0.0) + (_to_float(row.get("hw_disk_write_MBps")) or 0.0),
        "loopback_MBps": (_to_float(row.get("hw_net_rx_MBps")) or 0.0) + (_to_float(row.get("hw_net_tx_MBps")) or 0.0),
    }


def import_colocation_csv(path: str | Path) -> dict:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {"ok": False, "error": "empty CSV", "diagnoses": []}

    by_key: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        policy = normalize_policy(row.get("protection"))
        workload = str(row.get("interference") or row.get("workload") or "").strip()
        by_key.setdefault((policy, workload), []).append(row)

    baseline_row = None
    for key in [("none", "none"), ("none", "")]:
        if key in by_key:
            baseline_row = by_key[key][0]
            break
    if baseline_row is None:
        baseline_row = rows[0]
    baseline_metric = _metric_from_csv_row(baseline_row)

    diagnoses = []
    for (policy, workload), items in by_key.items():
        if workload in ("", "none"):
            continue
        metrics = [_metric_from_csv_row(item) for item in items]
        merged_metric = {
            "throughput_tok_s": mean([m["throughput_tok_s"] for m in metrics if m["throughput_tok_s"] is not None]) if any(m["throughput_tok_s"] is not None for m in metrics) else None,
            "ttft_avg_ms": mean([m["ttft_avg_ms"] for m in metrics if m["ttft_avg_ms"] is not None]) if any(m["ttft_avg_ms"] is not None for m in metrics) else None,
            "tpot_avg_ms": mean([m["tpot_avg_ms"] for m in metrics if m["tpot_avg_ms"] is not None]) if any(m["tpot_avg_ms"] is not None for m in metrics) else None,
        }
        macro = compute_macro_degradation(merged_metric, baseline_metric)
        profile = _profile_from_csv_row(items[0])
        diagnoses.append(
            {
                "workload": workload,
                "policy": policy,
                "macro": macro,
                "workload_profile": profile,
                "risk": screen_hardware_risk(profile),
                "raw_count": len(items),
            }
        )
    return {"ok": True, "source": str(path), "diagnoses": diagnoses, "baseline": baseline_metric}


def _metric_from_json_exp(exp: dict) -> dict:
    metrics = exp.get("metrics") or {}
    return {
        "throughput": metrics.get("throughput"),
        "ttft": metrics.get("ttft") or {},
        "tpot": metrics.get("tpot") or {},
    }


def import_colocation_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    experiments = []
    for round_obj in data.get("rounds", []):
        experiments.extend(round_obj.get("experiments", []))
    if not experiments:
        return {"ok": False, "error": "no experiments found", "diagnoses": []}

    baseline = None
    for exp in experiments:
        if normalize_policy(exp.get("protection")) == "none" and exp.get("interference") == "none":
            baseline = _metric_from_json_exp(exp)
            break
    baseline = baseline or _metric_from_json_exp(experiments[0])

    diagnoses = []
    for exp in experiments:
        workload = str(exp.get("interference") or "")
        if workload in ("", "none"):
            continue
        hw = exp.get("hw_metrics") or {}
        ctx = hw.get("context_switches") or {}
        cpu = hw.get("cpu_utilization") or {}
        mem = hw.get("memory_bandwidth") or {}
        sys_hw = hw.get("system") or {}
        disk = hw.get("disk_io") or {}
        net = hw.get("network_io") or {}
        profile = {
            "cpu_delta_pct": cpu.get("usr_pct") or 0.0,
            "llc_miss_pct": sys_hw.get("llc_load_miss_rate") or 0.0,
            "l1d_miss_pct": sys_hw.get("l1_dcache_miss_rate") or 0.0,
            "memcache_proxy_GBps": mem.get("total_bw_GBps") or mem.get("total_GBps") or 0.0,
            "ctx_switches_per_s": (ctx.get("voluntary_cswch_per_s") or 0.0) + (ctx.get("nonvoluntary_cswch_per_s") or 0.0),
            "disk_MBps": (disk.get("read_MBps") or 0.0) + (disk.get("write_MBps") or 0.0),
            "loopback_MBps": (net.get("rx_MBps") or 0.0) + (net.get("tx_MBps") or 0.0),
        }
        metric = _metric_from_json_exp(exp)
        diagnoses.append(
            {
                "workload": workload,
                "policy": normalize_policy(exp.get("protection")),
                "macro": compute_macro_degradation(metric, baseline),
                "workload_profile": profile,
                "risk": screen_hardware_risk(profile),
                "timestamp": exp.get("timestamp"),
            }
        )
    return {"ok": True, "source": str(path), "diagnoses": diagnoses, "baseline": baseline}


def import_any(path: str | Path) -> dict:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return import_colocation_csv(path)
    if suffix == ".json":
        return import_colocation_json(path)
    return {"ok": False, "error": f"unsupported suffix {suffix}", "diagnoses": []}

