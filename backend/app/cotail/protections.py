from __future__ import annotations

import os
import posixpath
import subprocess
from dataclasses import dataclass, asdict, field

from .constants import normalize_policy
from .cpu import format_cpu_range, parse_cpu_range


@dataclass
class ProtectionAction:
    kind: str
    target: str
    detail: str
    command: list[str] | None = None


@dataclass
class ProtectionPlan:
    policy: str
    execute: bool
    actions: list[ProtectionAction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _write_file(path: str, value: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(value))


def _probe_writable_cgroup_root(root: str) -> bool:
    if not root or not os.path.isdir(root):
        return False
    probe = os.path.join(root, f".cotail_probe_{os.getpid()}")
    try:
        os.mkdir(probe)
        os.rmdir(probe)
        return True
    except Exception:
        return False


def _find_delegated_cgroup_root(explicit: str = "") -> str:
    if explicit and _probe_writable_cgroup_root(explicit):
        return explicit
    env_root = os.environ.get("VLLM_CGROUP_BASE", "").strip()
    if env_root and _probe_writable_cgroup_root(env_root):
        return env_root
    candidates: list[str] = []
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as f:
            for line in f:
                if "::" not in line:
                    continue
                rel = line.strip().split("::", 1)[1]
                if not rel.startswith("/"):
                    rel = "/" + rel
                cur = os.path.normpath("/sys/fs/cgroup" + rel)
                while cur.startswith("/sys/fs/cgroup") and cur not in ("/sys/fs/cgroup", "/sys/fs"):
                    candidates.append(cur)
                    parent = os.path.dirname(cur)
                    if parent == cur:
                        break
                    cur = parent
                break
    except Exception:
        pass
    if hasattr(os, "getuid"):
        uid = os.getuid()
        candidates.append(f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service")
    for candidate in candidates:
        if _probe_writable_cgroup_root(candidate):
            return candidate
    return explicit or env_root or "/sys/fs/cgroup"


def _effective_cpuset(root: str, requested: str) -> str:
    effective = _read_file(os.path.join(root, "cpuset.cpus.effective")) or _read_file(os.path.join(root, "cpuset.cpus"))
    if not effective:
        return requested
    req = set(parse_cpu_range(requested))
    allowed = set(parse_cpu_range(effective))
    inter = sorted(req & allowed)
    return format_cpu_range(inter) if inter else effective


def _effective_mems(root: str) -> str:
    return _read_file(os.path.join(root, "cpuset.mems.effective")) or _read_file(os.path.join(root, "cpuset.mems")) or "0"


def build_protection_plan(
    policy: str,
    vllm_pids: list[int] | None = None,
    engine_tid: int | None = None,
    battle_cores: str = "",
    numa_vllm_cpus: str = "",
    numa_interference_cpus: str = "",
    cgroup_base: str = "",
    rt_priority: int = 50,
    execute: bool = False,
) -> ProtectionPlan:
    normalized = normalize_policy(policy)
    plan = ProtectionPlan(policy=normalized, execute=execute)
    invalid_pids = [int(p) for p in (vllm_pids or []) if int(p) <= 0]
    vllm_pids = [int(p) for p in (vllm_pids or []) if int(p) > 0]
    if invalid_pids:
        plan.warnings.append(f"Ignored invalid vLLM PIDs: {invalid_pids}.")

    if not vllm_pids:
        plan.warnings.append("No vLLM PIDs provided; affinity/nice/cgroup actions will be incomplete.")

    shared_cpus = parse_cpu_range(battle_cores) if battle_cores else []
    local_cpus = parse_cpu_range(numa_vllm_cpus) if numa_vllm_cpus else []

    if normalized in ("none", "rt", "nice", "cgroup"):
        if shared_cpus:
            for pid in vllm_pids:
                plan.actions.append(ProtectionAction("affinity", f"pid:{pid}", f"bind vLLM to battle_cores {battle_cores}"))
        else:
            plan.warnings.append("No battle cores provided; script-compatible vLLM baseline affinity cannot be changed.")

    if normalized == "none":
        plan.warnings.append("Policy none selected; script-compatible action only fixes vLLM on battle_cores.")

    if normalized == "rt":
        plan.warnings.append(
            "RT follows improved_real_stress_various_new.py: vLLM remains on battle_cores and only the selected EngineCore TID gets SCHED_FIFO."
        )

    if normalized in ("numa", "rt+numa"):
        vllm_cpu_text = numa_vllm_cpus
        if not local_cpus and shared_cpus:
            local_cpus = shared_cpus
            vllm_cpu_text = battle_cores
            plan.warnings.append("NUMA vLLM CPUs were empty; falling back to battle cores.")
        if local_cpus:
            for pid in vllm_pids:
                plan.actions.append(ProtectionAction("affinity", f"pid:{pid}", f"bind vLLM to GPU-local NUMA {vllm_cpu_text}"))
        else:
            plan.warnings.append("No NUMA vLLM CPU range provided; vLLM affinity cannot be changed.")
        if numa_interference_cpus:
            plan.actions.append(
                ProtectionAction("interference_affinity", "co-tenant workload", f"launch co-tenant on remote NUMA {numa_interference_cpus}")
            )

    if normalized == "nice":
        for pid in vllm_pids:
            plan.actions.append(ProtectionAction("nice", f"pid:{pid}", "set nice to -15"))

    if normalized == "cgroup":
        root = _find_delegated_cgroup_root(cgroup_base)
        suffix = f"{os.getuid() if hasattr(os, 'getuid') else 0}_{os.getpid()}"
        vllm_path = posixpath.join(root.rstrip("/"), f"vllm_protect_{suffix}")
        stress_path = posixpath.join(root.rstrip("/"), f"stress_limit_{suffix}")
        cpuset = _effective_cpuset(root, battle_cores) if battle_cores else ""
        mems = _effective_mems(root)
        if not _probe_writable_cgroup_root(root):
            plan.warnings.append(f"cgroup root may not be writable: {root}. Set VLLM_CGROUP_BASE to a delegated cgroup root.")
        plan.actions.append(ProtectionAction("cgroup", vllm_path, f"cpu.weight=5000 cpuset={cpuset} mems={mems}"))
        for pid in vllm_pids:
            plan.actions.append(ProtectionAction("cgroup_add_pid", f"pid:{pid}", f"move to {vllm_path}"))
        plan.actions.append(ProtectionAction("cgroup", stress_path, f"cpu.weight=50 cpuset={cpuset} mems={mems}"))

    if normalized in ("rt", "rt+numa"):
        if engine_tid:
            plan.actions.append(
                ProtectionAction(
                    "rt_sched",
                    f"tid:{engine_tid}",
                    f"set SCHED_FIFO priority={rt_priority}",
                    ["chrt", "-f", "-p", str(rt_priority), str(int(engine_tid))],
                )
            )
        else:
            plan.warnings.append("No EngineCore TID provided; RT action cannot be executed.")
    return plan


def execute_plan(plan: ProtectionPlan) -> dict:
    if not plan.execute:
        return {"executed": False, "ok": False, "results": [], "message": "dry-run only", "warnings": plan.warnings}

    results: list[dict] = []
    try:
        import psutil  # type: ignore
    except Exception as exc:
        return {"executed": False, "ok": False, "results": [], "error": f"psutil unavailable: {exc}", "warnings": plan.warnings}

    for action in plan.actions:
        try:
            if action.kind == "affinity":
                pid = int(action.target.split(":", 1)[1])
                cpus_text = action.detail.rsplit(" ", 1)[-1]
                cpus = parse_cpu_range(cpus_text)
                proc = psutil.Process(pid)
                proc.cpu_affinity(cpus)
                actual = proc.cpu_affinity()
                results.append({"action": asdict(action), "ok": bool(set(cpus) & set(actual)), "actual_affinity": format_cpu_range(actual)})
            elif action.kind == "nice":
                pid = int(action.target.split(":", 1)[1])
                os.setpriority(os.PRIO_PROCESS, pid, -15)
                results.append({"action": asdict(action), "ok": True, "actual_nice": os.getpriority(os.PRIO_PROCESS, pid)})
            elif action.kind == "rt_sched" and action.command:
                proc = subprocess.run(action.command, capture_output=True, text=True, timeout=5)
                verify = subprocess.run(["chrt", "-p", action.target.split(":", 1)[1]], capture_output=True, text=True, timeout=5)
                results.append(
                    {
                        "action": asdict(action),
                        "ok": proc.returncode == 0 and verify.returncode == 0 and "SCHED_FIFO" in verify.stdout,
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                        "verify_stdout": verify.stdout,
                        "verify_stderr": verify.stderr,
                    }
                )
            elif action.kind == "cgroup":
                path = action.target
                os.makedirs(path, exist_ok=True)
                parts = dict(part.split("=", 1) for part in action.detail.split() if "=" in part)
                weight = parts.get("cpu.weight", "100")
                cpuset = parts.get("cpuset", "")
                mems = parts.get("mems", "")
                cpuset_status = "skipped"
                if cpuset:
                    try:
                        if mems and os.path.exists(os.path.join(path, "cpuset.mems")):
                            _write_file(os.path.join(path, "cpuset.mems"), mems)
                        if os.path.exists(os.path.join(path, "cpuset.cpus")):
                            _write_file(os.path.join(path, "cpuset.cpus"), cpuset)
                        cpuset_status = "ok"
                    except Exception as exc:
                        cpuset_status = f"skipped: {exc}"
                weight_path = os.path.join(path, "cpu.weight")
                if os.path.exists(weight_path):
                    _write_file(weight_path, weight)
                results.append(
                    {
                        "action": asdict(action),
                        "ok": os.path.isdir(path),
                        "path": path,
                        "cpu_weight": weight,
                        "cpuset": cpuset,
                        "cpuset_status": cpuset_status,
                    }
                )
            elif action.kind == "cgroup_add_pid":
                pid = int(action.target.split(":", 1)[1])
                path = action.detail.rsplit(" ", 1)[-1]
                _write_file(os.path.join(path, "cgroup.procs"), str(pid))
                results.append({"action": asdict(action), "ok": True, "path": path})
            else:
                results.append({"action": asdict(action), "ok": False, "error": "execution not implemented for this action"})
        except Exception as exc:
            results.append({"action": asdict(action), "ok": False, "error": str(exc)})
    actionable = [r for r in results if r.get("action", {}).get("kind") != "interference_affinity"]
    ok = bool(actionable) and all(bool(r.get("ok")) for r in actionable)
    return {
        "executed": True,
        "ok": ok,
        "results": results,
        "warnings": plan.warnings,
        "summary": {
            "actions": len(results),
            "actionable": len(actionable),
            "ok": sum(1 for r in actionable if r.get("ok")),
            "failed": sum(1 for r in actionable if not r.get("ok")),
        },
    }
