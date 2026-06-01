from __future__ import annotations

from ..schemas import DiagnosisInput


def sample_nginx_case() -> DiagnosisInput:
    baseline = {
        "vllm.scheduler.step": {"p95_us": 1000, "p99_us": 1500},
        "vllm.batch.construct": {"p95_us": 800, "p99_us": 1200},
        "vllm.model.execute": {"p95_us": 1300, "p99_us": 1600},
        "vllm.model.forward": {"p95_us": 1800, "p99_us": 2400},
    }
    unprotected = {
        "vllm.scheduler.step": {"p95_us": 8200, "p99_us": 13200},
        "vllm.batch.construct": {"p95_us": 10800, "p99_us": 17200},
        "vllm.model.execute": {"p95_us": 7600, "p99_us": 11400},
        "vllm.model.forward": {"p95_us": 5800, "p99_us": 7100},
    }
    protected = {
        "vllm.scheduler.step": {"p95_us": 1150, "p99_us": 1600},
        "vllm.batch.construct": {"p95_us": 900, "p99_us": 1350},
        "vllm.model.execute": {"p95_us": 1400, "p99_us": 1800},
        "vllm.model.forward": {"p95_us": 1900, "p99_us": 2500},
    }
    return DiagnosisInput(
        workload="nginx",
        workload_profile={
            "cpu_delta_pct": 36.8,
            "ctx_switches_per_s": 180000,
            "loopback_MBps": 830,
            "llc_miss_pct": 8,
            "l1d_miss_pct": 12,
            "memcache_proxy_GBps": 5,
        },
        macro={
            "Throughput_Drop_Pct": 78.8,
            "TTFT_Increase_Pct": 429.5,
            "TPOT_Increase_Pct": 362.4,
        },
        stage_metrics=unprotected,
        baseline_stage_metrics=baseline,
        protected_stage_metrics=protected,
        validation_macro={
            "Throughput_Drop_Pct": 6.8,
            "TTFT_Increase_Pct": 32.0,
            "TPOT_Increase_Pct": 4.0,
        },
        metadata={"source": "built-in demo based on local CoTail script schema"},
    )

