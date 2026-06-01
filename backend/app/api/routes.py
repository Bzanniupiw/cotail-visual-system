from __future__ import annotations

import tempfile
import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from ..cotail.gpu_monitor import collect_gpu_status
from ..cotail.constants import WORKLOADS
from ..cotail.importers import import_any
from ..cotail.job_manager import JOB_MANAGER
from ..cotail.llm_client import benchmark_chat, chat_completion, list_models
from ..cotail.macro import compute_macro_degradation
from ..cotail.measurement import collect_canary_measurement
from ..cotail.nvtx_sqlite import summarize_nsys_sqlite
from ..cotail.permissions import probe_permissions
from ..cotail.process_discovery import discover_service_processes, identify_busy_thread
from ..cotail.protections import build_protection_plan, execute_plan
from ..cotail.runtime import system_snapshot
from ..cotail.scheduler import GpuSelectionPolicy, select_idle_gpu
from ..cotail.tenant import diagnostic_overhead_summary, discover_cpu_tenants, estimate_cost
from ..cotail.topology import discover_gpu_numa, infer_remote_numa_cpus
from ..cotail.workload_cleanup import cleanup_cotail_workloads, list_cotail_workload_processes
from ..cotail.workload_health import evaluate_workload, readiness_matrix
from ..cotail.workload_runner import build_workload_command
from ..schemas import (
    DiagnosisInput,
    GpuSelectionRequest,
    CanaryMeasurementRequest,
    LLMBenchmarkInput,
    LLMRequestInput,
    ImportPathRequest,
    LaunchJobRequest,
    MeasurementDiagnosisRequest,
    PermissionProbeRequest,
    ProcessDiscoveryRequest,
    ProtectionPlanRequest,
    TenantCostRequest,
    WorkloadCleanupRequest,
    WorkloadLaunchRequest,
    WorkloadProbeAllRequest,
)
from ..services.diagnosis_service import run_diagnosis
from ..services.sample_data import sample_nginx_case
from ..storage import get_diagnosis, list_diagnoses

router = APIRouter(prefix="/api")

LAST_MEASUREMENT: dict | None = None


@router.get("/health")
def health() -> dict:
    return {"ok": True, "service": "cotail-visual-system"}


def _selection_policy(req: GpuSelectionRequest) -> GpuSelectionPolicy:
    return GpuSelectionPolicy(
        min_free_memory_mib=req.min_free_memory_mib,
        max_gpu_util_pct=req.max_gpu_util_pct,
        max_compute_processes=req.max_compute_processes,
        allow_graphics_processes=req.allow_graphics_processes,
        exclude_gpu_ids=req.exclude_gpu_ids,
    )


def _runtime_snapshot(req: GpuSelectionRequest | None = None) -> dict:
    gpu = collect_gpu_status()
    policy = _selection_policy(req or GpuSelectionRequest())
    selection = select_idle_gpu(gpu.get("gpus", []), policy)
    return {
        "system": system_snapshot().to_dict(),
        "gpu": gpu,
        "selection": selection,
        "jobs": JOB_MANAGER.list_jobs(),
    }


def _persist_imported_diagnoses(result: dict) -> dict:
    data = dict(result)
    saved = []
    if not data.get("ok"):
        data["persisted"] = False
        data["saved_count"] = 0
        data["saved_ids"] = []
        return data

    for item in data.get("diagnoses", []):
        metadata = dict(item.get("metadata") or {})
        metadata.update(
            {
                "source": data.get("source"),
                "import_policy": item.get("policy"),
                "raw_count": item.get("raw_count"),
                "timestamp": item.get("timestamp"),
            }
        )
        input_data = DiagnosisInput(
            workload=str(item.get("workload") or "unknown"),
            workload_profile=item.get("workload_profile") or {},
            macro=item.get("macro") or {},
            stage_metrics=item.get("stage_metrics") or {},
            baseline_stage_metrics=item.get("baseline_stage_metrics") or {},
            protected_stage_metrics=item.get("protected_stage_metrics"),
            validation_macro=item.get("validation_macro"),
            metadata=metadata,
        )
        saved.append(run_diagnosis(input_data, persist=True))

    data["persisted"] = True
    data["saved_count"] = len(saved)
    data["saved_ids"] = [item.get("id") for item in saved]
    data["saved"] = saved[:10]
    return data


@router.get("/runtime/snapshot")
def runtime_snapshot() -> dict:
    return _runtime_snapshot()


@router.post("/scheduler/select")
def scheduler_select(req: GpuSelectionRequest) -> dict:
    gpu = collect_gpu_status()
    return {
        "gpu": gpu,
        "selection": select_idle_gpu(gpu.get("gpus", []), _selection_policy(req)),
    }


@router.websocket("/runtime/ws")
async def runtime_ws(websocket: WebSocket, interval_s: float = 2.0) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(_runtime_snapshot())
            await asyncio.sleep(max(0.5, float(interval_s)))
    except WebSocketDisconnect:
        return


@router.get("/diagnoses")
def diagnoses(limit: int = 100) -> list[dict]:
    return list_diagnoses(limit)


@router.get("/diagnoses/latest")
def latest_diagnosis() -> dict:
    rows = list_diagnoses(1)
    if not rows:
        return {"ok": False, "message": "no diagnosis yet"}
    item = get_diagnosis(int(rows[0]["id"]))
    if item is None:
        return {"ok": False, "message": "latest diagnosis not found"}
    item["ok"] = True
    return item


@router.get("/diagnoses/{diag_id}")
def diagnosis_detail(diag_id: int) -> dict:
    item = get_diagnosis(diag_id)
    if item is None:
        raise HTTPException(status_code=404, detail="diagnosis not found")
    return item


@router.post("/diagnoses")
def create_diagnosis(input_data: DiagnosisInput) -> dict:
    return run_diagnosis(input_data, persist=True)


@router.post("/diagnoses/sample")
def create_sample_diagnosis() -> dict:
    return run_diagnosis(sample_nginx_case(), persist=True)


@router.post("/measure/canary")
def measure_canary(req: CanaryMeasurementRequest) -> dict:
    global LAST_MEASUREMENT
    LAST_MEASUREMENT = collect_canary_measurement(
        phase=req.phase,
        workload=req.workload,
        api_base=req.api_base,
        model=req.model,
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=req.stream,
        timeout_s=req.timeout_s,
        concurrency=req.concurrency,
        total_requests=req.total_requests,
        run_benchmark=req.run_benchmark,
        sample_seconds=req.sample_seconds,
    )
    return LAST_MEASUREMENT


def _metric(obj: dict) -> dict:
    metric = obj.get("metric") or {}
    if metric:
        return metric
    benchmark = obj.get("benchmark") or {}
    return {
        "throughput_tok_s": benchmark.get("throughput_tok_s"),
        "ttft_avg_ms": benchmark.get("ttft_avg_ms"),
        "tpot_avg_ms": benchmark.get("tpot_avg_ms"),
    }


@router.post("/diagnoses/from-measurements")
def create_diagnosis_from_measurements(req: MeasurementDiagnosisRequest) -> dict:
    if not req.baseline_stage_metrics or not req.stage_metrics:
        raise HTTPException(
            status_code=400,
            detail="CoTail diagnosis requires Baseline and Co-location micro stage metrics; collect macro canary data first, then parse Nsight traces in the CPTI view.",
        )
    baseline_metric = _metric(req.baseline)
    coloc_metric = _metric(req.colocation)
    protected_metric = _metric(req.protected or {}) if req.protected else {}
    protected_experiment = (req.protected or {}).get("experiment") or {}
    macro = compute_macro_degradation(coloc_metric, baseline_metric)
    validation_macro = compute_macro_degradation(protected_metric, baseline_metric) if protected_metric else None
    input_data = DiagnosisInput(
        workload=req.workload or str(req.colocation.get("workload") or "unknown"),
        workload_profile=req.colocation.get("workload_profile") or {},
        macro=macro,
        stage_metrics=req.stage_metrics,
        baseline_stage_metrics=req.baseline_stage_metrics,
        protected_stage_metrics=req.protected_stage_metrics,
        validation_macro=validation_macro,
        metadata={
            "source": "live_canary_measurements",
            "baseline_phase": req.baseline.get("phase"),
            "colocation_phase": req.colocation.get("phase"),
            "protected_phase": (req.protected or {}).get("phase"),
            "protected_policy": protected_experiment.get("protection_policy"),
            "protected_executed": protected_experiment.get("protection_executed"),
        },
    )
    payload = run_diagnosis(input_data, persist=req.persist)
    payload["measurements"] = {
        "baseline": req.baseline,
        "colocation": req.colocation,
        "protected": req.protected,
    }
    return payload


@router.post("/import/path")
def import_path(req: ImportPathRequest) -> dict:
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"path does not exist: {path}")
    result = import_any(path)
    return _persist_imported_diagnoses(result) if req.persist else result


@router.post("/import/upload")
async def import_upload(file: UploadFile = File(...), persist: bool = False) -> dict:
    suffix = Path(file.filename or "upload").suffix
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        result = import_any(tmp_path)
        if "source" in result:
            result["source"] = file.filename or result["source"]
        return _persist_imported_diagnoses(result) if persist else result
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass


@router.post("/nsight/parse")
def parse_nsight(req: ImportPathRequest) -> dict:
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"path does not exist: {path}")
    return summarize_nsys_sqlite(path)


@router.post("/permissions/probe")
def permissions_probe(req: PermissionProbeRequest) -> dict:
    return probe_permissions(
        protections=req.protections,
        interferences=req.interferences,
        api_url=req.api_url,
        gpu_idx=req.gpu_idx,
        no_perf=req.no_perf,
    )


@router.post("/process/discover")
def process_discover(req: ProcessDiscoveryRequest) -> dict:
    return discover_service_processes(
        framework=req.framework,
        ports=req.ports,
        model_hint=req.model_hint,
        owner_user=req.owner_user,
    )


@router.post("/process/busy-thread")
def process_busy_thread(pids: list[int], probe_seconds: float = 1.0) -> dict:
    return identify_busy_thread(pids, probe_seconds)


@router.post("/protection/plan")
def protection_plan(req: ProtectionPlanRequest) -> dict:
    plan = build_protection_plan(
        policy=req.policy,
        vllm_pids=req.vllm_pids,
        engine_tid=req.engine_tid,
        battle_cores=req.battle_cores,
        numa_vllm_cpus=req.numa_vllm_cpus,
        numa_interference_cpus=req.numa_interference_cpus,
        cgroup_base=req.cgroup_base,
        rt_priority=req.rt_priority,
        execute=req.execute,
    )
    execution = execute_plan(plan)
    return {"plan": plan.to_dict(), "execution": execution}


@router.get("/workloads/{workload}/command")
def workload_command(workload: str, cpus: str = "", workers: int = 60, duration_s: int = 60) -> dict:
    return build_workload_command(workload, cpus=cpus, workers=workers, duration_s=duration_s).to_dict()


@router.post("/workloads/launch")
def workload_launch(req: WorkloadLaunchRequest) -> dict:
    job = JOB_MANAGER.launch_workload(
        workload=req.workload,
        cpu_cores=req.cpu_cores,
        workers=req.workers,
        duration_s=req.duration_s,
        stress_cgroup_path=req.stress_cgroup_path,
        dry_run=req.dry_run,
    )
    message = (
        "dry-run only: workload command was generated but not started."
        if job.get("dry_run")
        else f"co-location workload {req.workload} launch requested; check managed jobs and logs."
    )
    return {"job": job, "message": message}


@router.post("/workloads/stop-managed")
def workloads_stop_managed() -> dict:
    return JOB_MANAGER.stop_cpu_workloads()


@router.get("/workloads/orphans")
def workloads_orphans(workloads: str = "", current_user_only: bool = True) -> dict:
    selected = [w.strip() for w in workloads.split(",") if w.strip()]
    rows = list_cotail_workload_processes(workloads=selected, current_user_only=current_user_only)
    return {"ok": True, "count": len(rows), "processes": rows}


@router.post("/workloads/cleanup")
def workloads_cleanup(req: WorkloadCleanupRequest) -> dict:
    result = cleanup_cotail_workloads(
        workloads=req.workloads,
        current_user_only=req.current_user_only,
        dry_run=req.dry_run,
        force=req.force,
    )
    result["managed_removed"] = JOB_MANAGER.prune_finished_cpu_workloads()
    return result


@router.get("/workloads/readiness")
def workloads_readiness() -> dict:
    return readiness_matrix()


@router.get("/workloads/health")
def workload_health(workload: str = "stress-ng", job_id: str = "", sample_seconds: float = 1.5) -> dict:
    job = JOB_MANAGER.get_job(job_id) if job_id else None
    return evaluate_workload(workload, job=job, sample_seconds=sample_seconds)


@router.post("/workloads/probe-all")
def workloads_probe_all(req: WorkloadProbeAllRequest) -> dict:
    selected = req.workloads or WORKLOADS
    rows = []
    for workload in selected:
        job = JOB_MANAGER.launch_workload(
            workload=workload,
            cpu_cores=req.cpu_cores,
            workers=req.workers,
            duration_s=max(int(req.duration_s), int(req.startup_wait_s + req.sample_seconds + 5)),
            dry_run=req.dry_run,
        )
        if req.dry_run:
            health = evaluate_workload(workload, job=job, sample_seconds=0.05)
            logs = {}
        else:
            warmup = max(0.2, float(req.startup_wait_s))
            if workload in {"kernel_build", "nginx", "memcached", "sqlite-txn", "image-preprocess", "text-search", "zstd-compress"}:
                warmup = max(warmup, 5.0)
            time.sleep(warmup)
            health = evaluate_workload(workload, job=JOB_MANAGER.get_job(str(job.get("id"))), sample_seconds=req.sample_seconds)
            logs = JOB_MANAGER.read_logs(str(job.get("id")), role="cpu_workload", tail_bytes=6000)
            JOB_MANAGER.stop(str(job.get("id")))
        rows.append({"workload": workload, "job_id": job.get("id"), "health": health, "logs": logs})
    return {
        "ok": True,
        "dry_run": req.dry_run,
        "count": len(rows),
        "active_count": sum(1 for row in rows if row.get("health", {}).get("active")),
        "results": rows,
    }


@router.get("/topology/gpu/{gpu_idx}")
def topology_gpu(gpu_idx: int) -> dict:
    info = discover_gpu_numa(gpu_idx)
    remote = infer_remote_numa_cpus(info.cpus)
    data = info.to_dict()
    data["remote_cpus"] = remote
    return data


@router.post("/tenants/snapshot")
def tenants_snapshot(req: TenantCostRequest) -> dict:
    tenants = discover_cpu_tenants(limit=req.limit)
    rows = tenants.get("tenants", [])
    return {
        **tenants,
        "cost": estimate_cost(req.policy, rows, req.protected_pids),
        "diagnostic_overhead": diagnostic_overhead_summary(LAST_MEASUREMENT),
    }


@router.get("/jobs")
def jobs() -> list[dict]:
    return JOB_MANAGER.list_jobs()


@router.post("/jobs/launch")
def launch_job(req: LaunchJobRequest) -> dict:
    selected_gpu = req.gpu_index
    selection = None
    if req.auto_select_gpu or selected_gpu is None:
        gpu = collect_gpu_status()
        selection = select_idle_gpu(gpu.get("gpus", []), _selection_policy(req.gpu_policy))
        selected_gpu = selection.get("selected_gpu")
        if selected_gpu is None:
            raise HTTPException(status_code=409, detail={"message": "no eligible idle GPU", "selection": selection})
    job = JOB_MANAGER.launch(
        gpu_index=int(selected_gpu),
        model=req.model,
        backend=req.backend,
        host=req.host,
        port=req.port,
        gpu_memory_utilization=req.gpu_memory_utilization,
        max_model_len=req.max_model_len,
        enforce_eager=req.enforce_eager,
        cpu_workload=req.cpu_workload,
        cpu_cores=req.cpu_cores,
        workers=req.workers,
        dry_run=req.dry_run,
        serving_extra_args=req.serving_extra_args,
        serving_template=req.serving_template,
    )
    message = (
        "dry-run only: no process was started. Uncheck dry-run to launch."
        if job.get("dry_run")
        else "launch requested; check job status and logs while the service starts."
    )
    return {"job": job, "selection": selection, "message": message}


@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict:
    return JOB_MANAGER.stop(job_id)


@router.get("/jobs/{job_id}/logs")
def job_logs(job_id: str, role: str = "llm_server", tail_bytes: int = 16000) -> dict:
    return JOB_MANAGER.read_logs(job_id, role=role, tail_bytes=tail_bytes)


@router.post("/llm/models")
def llm_models(req: LLMRequestInput) -> dict:
    return list_models(req.api_base, timeout_s=min(req.timeout_s, 10.0))


@router.post("/llm/request")
def llm_request(req: LLMRequestInput) -> dict:
    return chat_completion(
        api_base=req.api_base,
        model=req.model,
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=req.stream,
        timeout_s=req.timeout_s,
    )


@router.post("/llm/benchmark")
def llm_benchmark(req: LLMBenchmarkInput) -> dict:
    return benchmark_chat(
        api_base=req.api_base,
        model=req.model,
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=req.stream,
        timeout_s=req.timeout_s,
        concurrency=req.concurrency,
        total_requests=req.total_requests,
    )
