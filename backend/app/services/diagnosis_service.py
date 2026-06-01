from __future__ import annotations

from typing import Any

from ..cotail.constants import PRIMARY_DECODE_SLO
from ..cotail.cpti import compute_cpti, compute_cts
from ..cotail.decision import final_deployment_decision, recommend_policy
from ..cotail.risk import screen_hardware_risk
from ..schemas import DiagnosisInput
from ..storage import save_diagnosis


def run_diagnosis(input_data: DiagnosisInput, persist: bool = True) -> dict[str, Any]:
    risk = screen_hardware_risk(input_data.workload_profile)
    cpti = None
    if input_data.stage_metrics and input_data.baseline_stage_metrics:
        cpti = compute_cpti(input_data.stage_metrics, input_data.baseline_stage_metrics).to_dict()

    recommendation = recommend_policy(risk, cpti, input_data.macro)

    protected_cpti = None
    cts_ratio = None
    cts_source = "cpti"
    if input_data.protected_stage_metrics and input_data.baseline_stage_metrics:
        protected_cpti = compute_cpti(input_data.protected_stage_metrics, input_data.baseline_stage_metrics).to_dict()
        cts_ratio = compute_cts(
            cpti.get("cpti_ratio") if cpti else None,
            protected_cpti.get("cpti_ratio") if protected_cpti else None,
        )

    final = final_deployment_decision(recommendation, input_data.validation_macro, cts_ratio, PRIMARY_DECODE_SLO)
    payload = {
        "workload": input_data.workload,
        "metadata": input_data.metadata,
        "risk": risk,
        "macro": input_data.macro,
        "cpti": cpti,
        "protected_cpti": protected_cpti,
        "cts_ratio": cts_ratio,
        "cts_pct": None if cts_ratio is None else 100.0 * cts_ratio,
        "cts_source": cts_source if cts_ratio is not None else None,
        "recommendation": recommendation,
        "final": final,
        "slo": PRIMARY_DECODE_SLO,
    }
    return save_diagnosis(payload) if persist else payload
