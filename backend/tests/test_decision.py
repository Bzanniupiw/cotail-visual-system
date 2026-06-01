from backend.app.cotail.cpti import compute_cpti
from backend.app.cotail.decision import final_deployment_decision, recommend_policy
from backend.app.cotail.risk import screen_hardware_risk


def test_high_cpti_recommends_rt():
    risk = screen_hardware_risk({"cpu_delta_pct": 40, "ctx_switches_per_s": 200000})
    base = {
        "vllm.scheduler.step": {"p95_us": 10, "p99_us": 10},
        "vllm.batch.construct": {"p95_us": 10, "p99_us": 10},
        "vllm.model.execute": {"p95_us": 10, "p99_us": 10},
        "vllm.model.forward": {"p95_us": 10, "p99_us": 10},
    }
    cur = {
        "vllm.scheduler.step": {"p95_us": 100, "p99_us": 100},
        "vllm.batch.construct": {"p95_us": 100, "p99_us": 100},
        "vllm.model.execute": {"p95_us": 100, "p99_us": 100},
        "vllm.model.forward": {"p95_us": 100, "p99_us": 100},
    }
    cpti = compute_cpti(cur, base).to_dict()
    rec = recommend_policy(risk, cpti)
    assert rec["candidate_policy"] == "rt"


def test_protected_colocate_warn_when_tail_recovers_but_throughput_is_soft_fail():
    rec = {"candidate_policy": "rt", "tail_diagnosed": True}
    validation_macro = {
        "Throughput_Drop_Pct": 13.3,
        "TPOT_Increase_Pct": 9.4,
    }
    final = final_deployment_decision(
        rec,
        validation_macro,
        cts_ratio=0.96,
        slo={
            "throughput_drop_max_pct": 10.0,
            "throughput_soft_drop_max_pct": 15.0,
            "tpot_increase_max_pct": 10.0,
        },
    )
    assert final["decision"] == "PROTECTED_COLOCATE_WARN"
