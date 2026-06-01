from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .runtime import run_command
from .workload_cleanup import _collect_process_targets, _terminate_targets
from .workload_runner import build_workload_command


@dataclass
class ManagedProcess:
    role: str
    pid: int | None
    command: list[str]
    env: dict[str, str]
    started_at: float | None
    status: str
    return_code: int | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None


@dataclass
class ManagedJob:
    id: str
    gpu_index: int
    model: str
    backend: str
    port: int
    dry_run: bool
    created_at: float
    processes: list[ManagedProcess] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class JobManager:
    _CPU_TERMINAL_STATUSES = {"stopped", "exited", "planned", "failed_to_start"}

    def __init__(self) -> None:
        self.jobs: dict[str, ManagedJob] = {}
        self._handles: dict[tuple[str, str], subprocess.Popen] = {}
        self.log_dir = Path(__file__).resolve().parents[3] / "storage" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _refresh_process_statuses(self) -> None:
        for key, proc in list(self._handles.items()):
            job_id, role = key
            rc = proc.poll()
            if rc is None:
                continue
            job = self.jobs.get(job_id)
            if not job:
                continue
            for item in job.processes:
                if item.role == role:
                    item.return_code = rc
                    item.status = "exited"

    @staticmethod
    def _is_cpu_workload_job(job: ManagedJob) -> bool:
        return job.backend == "cpu-workload" and all(process.role == "cpu_workload" for process in job.processes)

    def _is_finished_cpu_workload_job(self, job: ManagedJob) -> bool:
        return self._is_cpu_workload_job(job) and all(
            process.status in self._CPU_TERMINAL_STATUSES for process in job.processes
        )

    def prune_finished_cpu_workloads(self) -> int:
        self._refresh_process_statuses()
        removed = 0
        for job_id, job in list(self.jobs.items()):
            if not self._is_finished_cpu_workload_job(job):
                continue
            self.jobs.pop(job_id, None)
            for key in list(self._handles):
                if key[0] == job_id:
                    self._handles.pop(key, None)
            removed += 1
        return removed

    def list_jobs(self) -> list[dict[str, Any]]:
        self._refresh_process_statuses()
        self.prune_finished_cpu_workloads()
        rows = []
        for job in self.jobs.values():
            item = asdict(job)
            item["api_base"] = f"http://127.0.0.1:{job.port}/v1"
            item["api_port_open"] = self._port_open("127.0.0.1", job.port, timeout_s=0.15)
            rows.append(item)
        return rows

    @staticmethod
    def _port_open(host: str, port: int, timeout_s: float = 0.2) -> bool:
        try:
            with socket.create_connection((host, int(port)), timeout=timeout_s):
                return True
        except Exception:
            return False

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        self._refresh_process_statuses()
        job = self.jobs.get(job_id)
        return None if job is None else asdict(job)

    @staticmethod
    def build_serving_command(
        backend: str,
        model: str,
        port: int,
        host: str = "0.0.0.0",
        gpu_memory_utilization: float | None = 0.7,
        max_model_len: int | None = 8192,
        enforce_eager: bool = True,
        extra_args: str = "",
        template: str = "",
    ) -> list[str]:
        if template:
            rendered = template.format(model=shlex.quote(model), port=port, host=host)
            return ["bash", "-lc", rendered]
        backend = backend.lower()
        if backend == "vllm":
            cmd = [
                "vllm",
                "serve",
                model,
                "--host",
                host,
                "--port",
                str(port),
                "--tensor-parallel-size",
                "1",
            ]
            if gpu_memory_utilization is not None:
                cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
            if max_model_len is not None:
                cmd.extend(["--max-model-len", str(int(max_model_len))])
            if enforce_eager:
                cmd.append("--enforce-eager")
        elif backend == "sglang":
            cmd = ["python3", "-m", "sglang.launch_server", "--model-path", model, "--host", host, "--port", str(port)]
        elif backend == "llama-server":
            cmd = ["llama-server", "-m", model, "--host", host, "--port", str(port)]
        else:
            cmd = ["bash", "-lc", f"echo unknown backend {shlex.quote(backend)}; sleep 3"]
        if extra_args:
            cmd.extend(shlex.split(extra_args))
        return cmd

    def launch(
        self,
        gpu_index: int,
        model: str,
        backend: str = "vllm",
        host: str = "0.0.0.0",
        port: int = 8102,
        gpu_memory_utilization: float | None = 0.7,
        max_model_len: int | None = 8192,
        enforce_eager: bool = True,
        cpu_workload: str = "none",
        cpu_cores: str = "",
        workers: int = 60,
        stress_cgroup_path: str = "",
        dry_run: bool = True,
        serving_extra_args: str = "",
        serving_template: str = "",
    ) -> dict[str, Any]:
        job_id = str(uuid.uuid4())[:8]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))
        job = ManagedJob(
            id=job_id,
            gpu_index=int(gpu_index),
            model=model,
            backend=backend,
            port=int(port),
            dry_run=bool(dry_run),
            created_at=time.time(),
        )
        if dry_run:
            job.notes.append("dry-run: commands were generated but no process was started. Uncheck dry-run to launch.")

        serving_cmd = self.build_serving_command(
            backend=backend,
            model=model,
            port=port,
            host=host,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            extra_args=serving_extra_args,
            template=serving_template,
        )
        job.processes.append(
            ManagedProcess(
                role="llm_server",
                pid=None,
                command=serving_cmd,
                env={"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]},
                started_at=None,
                status="planned" if dry_run else "starting",
            )
        )

        if cpu_workload and cpu_workload != "none":
            workload = build_workload_command(
                cpu_workload,
                cpus=cpu_cores,
                workers=workers,
                duration_s=3600,
                stress_cgroup_path=stress_cgroup_path,
            )
            job.processes.append(
                ManagedProcess(
                    role="cpu_workload",
                    pid=None,
                    command=workload.command,
                    env={},
                    started_at=None,
                    status="planned" if dry_run else "starting",
                )
            )

        if not dry_run:
            for process in job.processes:
                proc_env = env if process.role == "llm_server" else os.environ.copy()
                stdout_log = self.log_dir / f"{job_id}_{process.role}.out.log"
                stderr_log = self.log_dir / f"{job_id}_{process.role}.err.log"
                out_f = stdout_log.open("ab")
                err_f = stderr_log.open("ab")
                process.stdout_log = str(stdout_log)
                process.stderr_log = str(stderr_log)
                try:
                    proc = subprocess.Popen(
                        process.command,
                        stdout=out_f,
                        stderr=err_f,
                        env=proc_env,
                        start_new_session=True,
                    )
                    process.pid = proc.pid
                    process.started_at = time.time()
                    process.status = "running"
                    self._handles[(job_id, process.role)] = proc
                except Exception as exc:
                    process.status = "failed_to_start"
                    process.return_code = -1
                    err_f.write((f"\n[CoTail launch error] {exc}\n").encode("utf-8", errors="replace"))
                    job.notes.append(f"{process.role} failed to start: {exc}")

        self.jobs[job_id] = job
        return asdict(job)

    def launch_workload(
        self,
        workload: str,
        cpu_cores: str = "",
        workers: int = 60,
        duration_s: int = 3600,
        stress_cgroup_path: str = "",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        job_id = str(uuid.uuid4())[:8]
        command = build_workload_command(
            workload,
            cpus=cpu_cores,
            workers=workers,
            duration_s=duration_s,
            stress_cgroup_path=stress_cgroup_path,
        )
        job = ManagedJob(
            id=job_id,
            gpu_index=-1,
            model="",
            backend="cpu-workload",
            port=0,
            dry_run=bool(dry_run),
            created_at=time.time(),
            notes=[command.notes] if command.notes else [],
        )
        if dry_run:
            job.notes.append("dry-run: workload command was generated but no process was started.")
        process = ManagedProcess(
            role="cpu_workload",
            pid=None,
            command=command.command,
            env={},
            started_at=None,
            status="planned" if dry_run else "starting",
        )
        job.processes.append(process)

        if not dry_run:
            stdout_log = self.log_dir / f"{job_id}_cpu_workload.out.log"
            stderr_log = self.log_dir / f"{job_id}_cpu_workload.err.log"
            out_f = stdout_log.open("ab")
            err_f = stderr_log.open("ab")
            process.stdout_log = str(stdout_log)
            process.stderr_log = str(stderr_log)
            try:
                proc = subprocess.Popen(
                    process.command,
                    stdout=out_f,
                    stderr=err_f,
                    env=os.environ.copy(),
                    start_new_session=True,
                )
                process.pid = proc.pid
                process.started_at = time.time()
                process.status = "running"
                self._handles[(job_id, process.role)] = proc
            except Exception as exc:
                process.status = "failed_to_start"
                process.return_code = -1
                err_f.write((f"\n[CoTail workload launch error] {exc}\n").encode("utf-8", errors="replace"))
                job.notes.append(f"cpu_workload failed to start: {exc}")

        self.jobs[job_id] = job
        data = asdict(job)
        data["workload_command"] = command.to_dict()
        return data

    def stop(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            return {"ok": False, "error": "job not found"}
        results = []
        for process in job.processes:
            if not process.pid:
                process.status = "stopped"
                continue
            try:
                if os.name != "nt":
                    try:
                        import psutil  # type: ignore

                        targets, pgids = _collect_process_targets(int(process.pid), psutil)
                        kill_result = _terminate_targets(targets, pgids, psutil, force=True, grace_s=6.0)
                    except Exception:
                        pgid = os.getpgid(process.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        time.sleep(5.0)
                        kill_result = {"pgids": [pgid], "fallback": True}
                    if kill_result.get("alive_after_kill"):
                        raise RuntimeError(f"processes still alive: {kill_result['alive_after_kill']}")
                    try:
                        os.killpg(os.getpgid(process.pid), 0)
                    except Exception:
                        pass
                else:
                    run_command(["taskkill", "/PID", str(process.pid), "/T", "/F"], timeout_s=5)
                process.status = "stopped"
                results.append({"role": process.role, "pid": process.pid, "ok": True})
            except Exception as exc:
                results.append({"role": process.role, "pid": process.pid, "ok": False, "error": str(exc)})
        job_data = asdict(job)
        removed = 0
        if self._is_finished_cpu_workload_job(job):
            self.jobs.pop(job_id, None)
            for key in list(self._handles):
                if key[0] == job_id:
                    self._handles.pop(key, None)
            removed = 1
        return {"ok": all(r["ok"] for r in results), "results": results, "job": job_data, "removed": removed}

    def stop_cpu_workloads(self) -> dict[str, Any]:
        self._refresh_process_statuses()
        rows = []
        for job_id, job in list(self.jobs.items()):
            has_cpu_workload = job.backend == "cpu-workload" or any(p.role == "cpu_workload" for p in job.processes)
            if not has_cpu_workload:
                continue
            if all(p.status in {"stopped", "exited", "planned"} for p in job.processes):
                continue
            rows.append(self.stop(job_id))
        return {
            "ok": all(item.get("ok") for item in rows) if rows else True,
            "count": len(rows),
            "results": rows,
            "removed": self.prune_finished_cpu_workloads(),
        }

    def read_logs(self, job_id: str, role: str = "llm_server", tail_bytes: int = 16000) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            return {"ok": False, "error": "job not found"}
        proc = next((p for p in job.processes if p.role == role), None)
        if not proc:
            return {"ok": False, "error": "role not found"}

        def tail(path: str | None) -> str:
            if not path:
                return ""
            p = Path(path)
            if not p.exists():
                return ""
            with p.open("rb") as f:
                if p.stat().st_size > tail_bytes:
                    f.seek(-tail_bytes, os.SEEK_END)
                return f.read().decode("utf-8", errors="replace")

        command_preview = " ".join(shlex.quote(x) for x in proc.command)
        stdout = tail(proc.stdout_log)
        stderr = tail(proc.stderr_log)
        message = ""
        if job.dry_run:
            message = "dry-run job: no process was started, so no runtime logs exist."
        elif not stdout and not stderr:
            message = "no log output yet; the process may still be starting or failed before writing logs."
        return {
            "ok": True,
            "job_id": job_id,
            "role": role,
            "status": proc.status,
            "pid": proc.pid,
            "command": proc.command,
            "command_preview": command_preview,
            "message": message,
            "stdout": stdout,
            "stderr": stderr,
        }


JOB_MANAGER = JobManager()
