from __future__ import annotations

EPS = 1e-12


def safe_pct_increase(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or abs(float(baseline)) <= EPS:
        return None
    return 100.0 * (float(current) - float(baseline)) / float(baseline)


def safe_pct_drop(current: float | None, baseline: float | None) -> float | None:
    inc = safe_pct_increase(current, baseline)
    return None if inc is None else -inc


def compute_macro_degradation(
    current: dict,
    baseline: dict,
) -> dict[str, float | None]:
    throughput = current.get("throughput") or current.get("throughput_tok_s")
    base_throughput = baseline.get("throughput") or baseline.get("throughput_tok_s")
    ttft = current.get("ttft_avg_ms") or current.get("ttft", {}).get("avg")
    base_ttft = baseline.get("ttft_avg_ms") or baseline.get("ttft", {}).get("avg")
    tpot = current.get("tpot_avg_ms") or current.get("tpot", {}).get("avg")
    base_tpot = baseline.get("tpot_avg_ms") or baseline.get("tpot", {}).get("avg")
    return {
        "Throughput_Drop_Pct": safe_pct_drop(throughput, base_throughput),
        "TTFT_Increase_Pct": safe_pct_increase(ttft, base_ttft),
        "TPOT_Increase_Pct": safe_pct_increase(tpot, base_tpot),
    }


def decode_slo_pass(macro: dict, throughput_drop_max_pct: float, tpot_increase_max_pct: float) -> bool:
    thr = macro.get("Throughput_Drop_Pct")
    tpot = macro.get("TPOT_Increase_Pct")
    if thr is None or tpot is None:
        return False
    return float(thr) <= throughput_drop_max_pct and float(tpot) <= tpot_increase_max_pct


def compute_tail_recovery_ratio(unprotected_macro: dict, protected_macro: dict | None) -> float | None:
    if not protected_macro:
        return None
    unprotected = unprotected_macro.get("TPOT_Increase_Pct")
    protected = protected_macro.get("TPOT_Increase_Pct")
    if unprotected is None or protected is None or float(unprotected) <= EPS:
        return None
    return 1.0 - max(0.0, float(protected)) / float(unprotected)
