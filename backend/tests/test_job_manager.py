import time

from backend.app.cotail.job_manager import JobManager, ManagedJob, ManagedProcess


def test_vllm_command_matches_server_convention():
    cmd = JobManager.build_serving_command(
        backend="vllm",
        model="/home/liguowei/models/deepseek-7b",
        port=8102,
        gpu_memory_utilization=0.7,
        max_model_len=8192,
        enforce_eager=True,
    )
    assert cmd[:3] == ["vllm", "serve", "/home/liguowei/models/deepseek-7b"]
    assert ["--port", "8102"][0] in cmd
    assert "--gpu-memory-utilization" in cmd
    assert "0.7" in cmd
    assert "--max-model-len" in cmd
    assert "8192" in cmd
    assert "--enforce-eager" in cmd


def test_list_jobs_prunes_finished_cpu_workload_jobs():
    manager = JobManager()
    manager.jobs["cpu-done"] = ManagedJob(
        id="cpu-done",
        gpu_index=-1,
        model="",
        backend="cpu-workload",
        port=0,
        dry_run=False,
        created_at=time.time(),
        processes=[
            ManagedProcess(
                role="cpu_workload",
                pid=123,
                command=["stress-ng"],
                env={},
                started_at=time.time(),
                status="exited",
            )
        ],
    )
    manager.jobs["llm-stopped"] = ManagedJob(
        id="llm-stopped",
        gpu_index=1,
        model="/model",
        backend="vllm",
        port=8102,
        dry_run=False,
        created_at=time.time(),
        processes=[
            ManagedProcess(
                role="llm_server",
                pid=456,
                command=["vllm"],
                env={},
                started_at=time.time(),
                status="stopped",
            )
        ],
    )

    rows = manager.list_jobs()

    assert "cpu-done" not in manager.jobs
    assert {row["id"] for row in rows} == {"llm-stopped"}
