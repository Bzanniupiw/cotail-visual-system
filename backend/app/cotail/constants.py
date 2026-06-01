from __future__ import annotations

CORE_STAGES = {
    "scheduler.step": "vllm.scheduler.step",
    "batch.construct": "vllm.batch.construct",
    "model.execute": "vllm.model.execute",
    "model.forward": "vllm.model.forward",
}

ALL_VLLM_STAGES = [
    "vllm.request.received",
    "vllm.request.queued",
    "vllm.scheduler.schedule",
    "vllm.scheduler.step",
    "vllm.batch.construct",
    "vllm.model.execute",
    "vllm.model.forward",
    "vllm.sampling",
    "vllm.output.process",
    "vllm.response.send",
]

POLICY_ALIASES = {
    "none": "none",
    "nice": "nice",
    "cgroup": "cgroup",
    "rt": "rt",
    "rt_sched": "rt",
    "numa": "numa",
    "numa_isolate": "numa",
    "rt+numa": "rt+numa",
    "numa+rt": "rt+numa",
    "numa_isolate+rt_sched": "rt+numa",
    "rt_sched+numa_isolate": "rt+numa",
}

POLICY_ORDER = ["none", "nice", "cgroup", "rt", "numa", "rt+numa"]

SCRIPT_COMPAT = {
    "battle_cores": "0-60,128-188",
    "safe_cores": "61-63,189-191",
    "numa_vllm_cpus": "0-63,128-191",
    "numa_interference_cpus": "64-127,192-255",
    "stress_workers": 60,
    "warmup_time_s": 8.0,
    "benchmark_timeout_s": 180.0,
}

PRIMARY_DECODE_SLO = {
    "throughput_drop_max_pct": 10.0,
    "throughput_soft_drop_max_pct": 15.0,
    "tpot_increase_max_pct": 10.0,
}

STAGE1_THRESHOLDS = {
    "cpu_high_delta_pct": 12.5,
    "ctx_high_switches_per_s": 100000.0,
    "disk_high_MBps": 1.0,
    "loopback_high_MBps": 500.0,
    "llc_high_miss_pct": 30.0,
    "l1d_high_miss_pct": 30.0,
    "memcache_high_proxy_GBps": 40.0,
}

STAGE2_THRESHOLDS = {
    "cpti_low": 0.5,
    "cpti_high": 2.0,
    "dominant_stage_min_contribution_frac": 0.25,
    "ttft_dominated_ratio": 2.0,
}

WORKLOADS = [
    "stress-ng",
    "ffmpeg",
    "7zip",
    "redis",
    "openssl",
    "kernel_build",
    "memcached",
    "nginx",
    "zstd-compress",
    "sqlite-txn",
    "image-preprocess",
    "text-search",
]

FRAMEWORK_SPECS = {
    "vllm": {
        "display_name": "vLLM",
        "root_keywords": ["vllm"],
        "root_markers": ["serve"],
        "process_keywords": ["vllm", "enginecore", "vllm::enginecore"],
        "engine_markers": ["enginecore", "vllm::enginecore"],
    },
    "sglang": {
        "display_name": "SGLang",
        "root_keywords": ["sglang"],
        "root_markers": ["launch_server", "serve"],
        "process_keywords": [
            "sglang",
            "sglang.launch_server",
            "launch_server",
            "scheduler",
            "tokenizer_manager",
            "detokenizer_manager",
        ],
        "engine_markers": [
            "sglang",
            "scheduler",
            "tokenizer_manager",
            "detokenizer_manager",
        ],
    },
    "llama-server": {
        "display_name": "llama.cpp server",
        "root_keywords": ["llama-server"],
        "root_markers": ["llama-server", "--port", "--host"],
        "process_keywords": ["llama-server", "llama.cpp"],
        "engine_markers": ["llama-server"],
    },
}


def normalize_policy(policy: str | None) -> str:
    raw = str(policy or "none").strip().lower()
    return POLICY_ALIASES.get(raw, raw)
