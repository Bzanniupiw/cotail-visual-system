from __future__ import annotations

from .constants import PRIMARY_DECODE_SLO, STAGE2_THRESHOLDS
from .macro import decode_slo_pass
from .risk import ttft_dominated


CORE_POLICY_STAGES = {"scheduler.step", "batch.construct", "model.execute", "model.forward"}


def recommend_policy(
    risk_result: dict,
    cpti_result: dict | None,
    macro: dict | None = None,
    thresholds: dict | None = None,
) -> dict:
    th = dict(STAGE2_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    trace: list[str] = []
    risk = risk_result.get("risk", "LOW")
    signals = risk_result.get("signals", {})
    macro = macro or {}

    if risk == "LOW":
        trace.append("Stage-1 risk is LOW, direct co-location candidate.")
        return {
            "candidate_policy": "none",
            "tail_diagnosed": False,
            "diagnosis": "low_risk_direct_colocation",
            "trace": trace,
        }

    cpti = None
    dominant_stage = None
    if cpti_result:
        cpti = cpti_result.get("cpti_ratio")
        dominant_stage = cpti_result.get("dominant_stage")

    if cpti is None:
        trace.append("No CPTI is available; keep policy as none until profiling completes.")
        return {
            "candidate_policy": "none",
            "tail_diagnosed": False,
            "diagnosis": "needs_unprotected_profiling",
            "trace": trace,
        }

    if cpti <= th["cpti_low"]:
        trace.append(f"CPTI={cpti:.3f} <= low threshold {th['cpti_low']}; direct co-location candidate.")
        return {
            "candidate_policy": "none",
            "tail_diagnosed": False,
            "diagnosis": "low_cpti_direct_colocation",
            "trace": trace,
        }

    tail_diagnosed = cpti >= th["cpti_high"]
    if tail_diagnosed:
        trace.append(f"CPTI={cpti:.3f} >= high threshold {th['cpti_high']}; service-tail path diagnosed.")
    else:
        trace.append(f"CPTI={cpti:.3f} is between low and high thresholds; ambiguous case.")

    mem_or_topology = bool(signals.get("mem_cache_high") or signals.get("topology_high"))
    ttft_dom = ttft_dominated(macro, th["ttft_dominated_ratio"])

    if tail_diagnosed and mem_or_topology:
        trace.append("High CPTI plus mem/cache or topology signal; validate combined RT+NUMA.")
        return {
            "candidate_policy": "rt+numa",
            "tail_diagnosed": True,
            "diagnosis": "mixed_risk_combined_validation",
            "trace": trace,
        }

    if tail_diagnosed and dominant_stage in CORE_POLICY_STAGES:
        trace.append(f"Dominant stage is {dominant_stage}; recommend EngineCore-targeted RT.")
        return {
            "candidate_policy": "rt",
            "tail_diagnosed": True,
            "diagnosis": "cpu_service_tail_path",
            "trace": trace,
        }

    if mem_or_topology and ttft_dom:
        trace.append("Cache/topology signal and TTFT-dominated degradation; recommend NUMA.")
        return {
            "candidate_policy": "numa",
            "tail_diagnosed": False,
            "diagnosis": "locality_topology_path",
            "trace": trace,
        }

    trace.append("No strong service-tail or topology signal; keep none and validate macro SLO.")
    return {
        "candidate_policy": "none",
        "tail_diagnosed": tail_diagnosed,
        "diagnosis": "mild_or_ambiguous_risk",
        "trace": trace,
    }


def final_deployment_decision(
    recommendation: dict,
    validation_macro: dict | None,
    cts_ratio: float | None,
    slo: dict | None = None,
) -> dict:
    slo = dict(PRIMARY_DECODE_SLO if slo is None else slo)
    if validation_macro is None:
        return {
            "decision": "NEEDS_VALIDATION",
            "slo_pass": False,
            "cts_ok": None,
            "reason": "Candidate policy has not been validated yet.",
        }
    slo_ok = decode_slo_pass(
        validation_macro,
        slo["throughput_drop_max_pct"],
        slo["tpot_increase_max_pct"],
    )
    throughput_drop = validation_macro.get("Throughput_Drop_Pct")
    tpot_increase = validation_macro.get("TPOT_Increase_Pct")
    tail_slo_ok = tpot_increase is not None and float(tpot_increase) <= slo["tpot_increase_max_pct"]
    throughput_soft_ok = (
        throughput_drop is not None
        and float(throughput_drop) <= slo.get("throughput_soft_drop_max_pct", slo["throughput_drop_max_pct"])
    )
    tail_diagnosed = bool(recommendation.get("tail_diagnosed"))
    cts_ok = True if not tail_diagnosed else (cts_ratio is not None and cts_ratio > 0.0)
    if slo_ok and cts_ok:
        policy = recommendation.get("candidate_policy", "none")
        decision = "DIRECT_COLOCATE" if policy == "none" else "PROTECTED_COLOCATE"
        return {
            "decision": decision,
            "slo_pass": True,
            "cts_ok": cts_ok,
            "reason": "Common-baseline decode SLO is satisfied.",
        }
    policy = recommendation.get("candidate_policy", "none")
    if policy != "none" and cts_ok and tail_slo_ok and throughput_soft_ok:
        return {
            "decision": "PROTECTED_COLOCATE_WARN",
            "slo_pass": False,
            "cts_ok": cts_ok,
            "reason": (
                "Tail SLO and CTS are satisfied, but throughput is outside the strict SLO "
                f"({throughput_drop:.1f}% > {slo['throughput_drop_max_pct']:.1f}%)."
            ),
        }
    return {
        "decision": "REJECT_COLOCATION",
        "slo_pass": slo_ok,
        "cts_ok": cts_ok,
        "reason": "Validation failed SLO or CTS acceptance.",
    }
