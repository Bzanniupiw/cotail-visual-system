from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DiagnosisInput(BaseModel):
    workload: str = Field(default="unknown")
    workload_profile: dict[str, Any] = Field(default_factory=dict)
    macro: dict[str, Any] = Field(default_factory=dict)
    stage_metrics: dict[str, Any] = Field(default_factory=dict)
    baseline_stage_metrics: dict[str, Any] = Field(default_factory=dict)
    protected_stage_metrics: dict[str, Any] | None = None
    validation_macro: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImportPathRequest(BaseModel):
    path: str
    persist: bool = False


class PermissionProbeRequest(BaseModel):
    protections: list[str] = Field(default_factory=lambda: ["none", "nice", "cgroup", "rt", "numa"])
    interferences: list[str] = Field(default_factory=list)
    api_url: str = "http://127.0.0.1:8102/v1"
    gpu_idx: int = 0
    no_perf: bool = False


class ProtectionPlanRequest(BaseModel):
    policy: str
    vllm_pids: list[int] = Field(default_factory=list)
    engine_tid: int | None = None
    battle_cores: str = ""
    numa_vllm_cpus: str = ""
    numa_interference_cpus: str = ""
    cgroup_base: str = ""
    rt_priority: int = 50
    execute: bool = False


class ProcessDiscoveryRequest(BaseModel):
    framework: str = "vllm"
    ports: list[int] = Field(default_factory=lambda: [8101, 8102])
    model_hint: str = ""
    owner_user: str = ""


class GpuSelectionRequest(BaseModel):
    min_free_memory_mib: int = 18000
    max_gpu_util_pct: int = 10
    max_compute_processes: int = 0
    allow_graphics_processes: bool = True
    exclude_gpu_ids: list[int] = Field(default_factory=list)


class LaunchJobRequest(BaseModel):
    model: str
    backend: str = "vllm"
    gpu_index: int | None = None
    auto_select_gpu: bool = True
    host: str = "0.0.0.0"
    port: int = 8102
    gpu_memory_utilization: float | None = 0.7
    max_model_len: int | None = 8192
    enforce_eager: bool = True
    cpu_workload: str = "none"
    cpu_cores: str = ""
    workers: int = 60
    dry_run: bool = True
    serving_extra_args: str = ""
    serving_template: str = ""
    gpu_policy: GpuSelectionRequest = Field(default_factory=GpuSelectionRequest)


class LLMRequestInput(BaseModel):
    api_base: str = "http://127.0.0.1:8102/v1"
    model: str = "/home/liguowei/models/deepseek-7b"
    prompt: str = "请用三句话解释 CoTail 如何诊断 CPU co-location interference。"
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = True
    timeout_s: float = 120.0


class LLMBenchmarkInput(LLMRequestInput):
    concurrency: int = 32
    total_requests: int = 32


class CanaryMeasurementRequest(LLMBenchmarkInput):
    phase: str = "colocation"
    workload: str = "unknown"
    run_benchmark: bool = True
    sample_seconds: float = 2.0


class MeasurementDiagnosisRequest(BaseModel):
    workload: str = "unknown"
    baseline: dict[str, Any] = Field(default_factory=dict)
    colocation: dict[str, Any] = Field(default_factory=dict)
    protected: dict[str, Any] | None = None
    baseline_stage_metrics: dict[str, Any] = Field(default_factory=dict)
    stage_metrics: dict[str, Any] = Field(default_factory=dict)
    protected_stage_metrics: dict[str, Any] | None = None
    persist: bool = True


class TenantCostRequest(BaseModel):
    policy: str = "rt"
    protected_pids: list[int] = Field(default_factory=list)
    limit: int = 30


class WorkloadLaunchRequest(BaseModel):
    workload: str = "stress-ng"
    cpu_cores: str = ""
    workers: int = 60
    duration_s: int = 3600
    stress_cgroup_path: str = ""
    dry_run: bool = False


class WorkloadProbeAllRequest(BaseModel):
    workloads: list[str] = Field(default_factory=list)
    cpu_cores: str = ""
    workers: int = 60
    duration_s: int = 18
    sample_seconds: float = 2.0
    startup_wait_s: float = 8.0
    dry_run: bool = False


class WorkloadCleanupRequest(BaseModel):
    workloads: list[str] = Field(default_factory=list)
    current_user_only: bool = True
    dry_run: bool = True
    force: bool = True
