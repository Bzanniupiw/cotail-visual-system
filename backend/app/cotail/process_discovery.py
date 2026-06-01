from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict

from .constants import FRAMEWORK_SPECS


@dataclass
class ProcessInfo:
    pid: int
    ppid: int | None
    pgid: int | None
    name: str
    cmdline: str
    is_engine: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _cmdline(proc) -> str:
    try:
        return " ".join(proc.cmdline())
    except Exception:
        return ""


def _pgid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except Exception:
        return None


def _thread_name(pid: int, tid: int) -> str:
    try:
        with open(f"/proc/{pid}/task/{tid}/comm", "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except Exception:
        return ""


def _is_engine_text(text: str) -> bool:
    lower = text.lower()
    return "enginecore" in lower or "vllm::enginecore" in lower


def find_listener_pids(ports: list[int]) -> list[int]:
    try:
        import psutil  # type: ignore
    except Exception:
        return []
    wanted = set(int(p) for p in ports)
    pids: set[int] = set()
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.status == "LISTEN" and conn.pid and getattr(conn.laddr, "port", None) in wanted:
                pids.add(int(conn.pid))
    except Exception:
        return []
    return sorted(pids)


def discover_service_processes(
    framework: str = "vllm",
    ports: list[int] | None = None,
    model_hint: str = "",
    owner_user: str = "",
) -> dict:
    try:
        import psutil  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"psutil unavailable: {exc}", "processes": []}

    spec = FRAMEWORK_SPECS.get(framework.lower(), FRAMEWORK_SPECS["vllm"])
    root_keywords = [str(x).lower() for x in spec["root_keywords"]]
    root_markers = [str(x).lower() for x in spec["root_markers"]]
    process_keywords = [str(x).lower() for x in spec["process_keywords"]]
    engine_markers = [str(x).lower() for x in spec["engine_markers"]]
    ports = ports or []
    listener_pids = set(find_listener_pids(ports))
    roots = []

    for proc in psutil.process_iter(["pid", "name", "username", "cmdline"]):
        try:
            if owner_user and proc.username() != owner_user:
                continue
            text = (proc.info.get("name") or "") + " " + _cmdline(proc)
            lower = text.lower()
            port_match = proc.pid in listener_pids
            model_match = bool(model_hint and model_hint in text)
            if (port_match or model_match) and all(any(k in lower for k in group) for group in [root_keywords, root_markers]):
                roots.append(proc)
        except Exception:
            continue

    if not roots:
        for proc in psutil.process_iter(["pid", "name", "username", "cmdline"]):
            try:
                if owner_user and proc.username() != owner_user:
                    continue
                lower = ((proc.info.get("name") or "") + " " + _cmdline(proc)).lower()
                if any(k in lower for k in process_keywords):
                    roots.append(proc)
            except Exception:
                continue
    roots = sorted({p.pid: p for p in roots}.values(), key=lambda p: p.pid)

    family: dict[int, object] = {}
    for root in roots[:3]:
        try:
            family[root.pid] = root
            for child in root.children(recursive=True):
                family[child.pid] = child
        except Exception:
            pass

    infos: list[ProcessInfo] = []
    for proc in family.values():
        try:
            text = (proc.name() + " " + _cmdline(proc)).lower()
            infos.append(
                ProcessInfo(
                    pid=int(proc.pid),
                    ppid=proc.ppid(),
                    pgid=_pgid(proc.pid),
                    name=proc.name(),
                    cmdline=_cmdline(proc),
                    is_engine=any(marker in text for marker in engine_markers),
                )
            )
        except Exception:
            continue

    return {
        "ok": bool(infos),
        "framework": framework,
        "root_pids": [p.pid for p in roots],
        "listener_pids": sorted(listener_pids),
        "processes": [p.to_dict() for p in infos],
    }


def identify_busy_thread(pids: list[int], probe_seconds: float = 1.0) -> dict:
    try:
        import psutil  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"psutil unavailable: {exc}"}

    def snap() -> dict[tuple[int, int], dict]:
        data: dict[tuple[int, int], dict] = {}
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                proc_text = f"{proc.name()} {_cmdline(proc)}"
                for thread in proc.threads():
                    tid = int(thread.id)
                    thread_name = _thread_name(pid, tid)
                    data[(pid, tid)] = {
                        "pid": int(pid),
                        "tid": tid,
                        "process_name": proc.name(),
                        "thread_name": thread_name,
                        "cmdline": _cmdline(proc),
                        "cpu_time_s": float(thread.user_time + thread.system_time),
                        "is_engine": _is_engine_text(f"{proc_text} {thread_name}"),
                    }
            except Exception:
                pass
        return data

    before = snap()
    time.sleep(max(0.05, probe_seconds))
    after = snap()
    candidates = []
    for key, item in after.items():
        previous = before.get(key, item)
        delta = float(item["cpu_time_s"]) - float(previous.get("cpu_time_s", item["cpu_time_s"]))
        if delta > 0:
            candidates.append(
                {
                    "pid": item["pid"],
                    "tid": item["tid"],
                    "cpu_delta_s": delta,
                    "process_name": item["process_name"],
                    "thread_name": item["thread_name"],
                    "is_engine": item["is_engine"],
                    "cmdline": item["cmdline"][:240],
                }
            )
    candidates.sort(key=lambda x: (not x["is_engine"], -x["cpu_delta_s"]))
    fallback_engine = [
        {
            "pid": item["pid"],
            "tid": item["tid"],
            "cpu_delta_s": 0.0,
            "process_name": item["process_name"],
            "thread_name": item["thread_name"],
            "is_engine": item["is_engine"],
            "cmdline": item["cmdline"][:240],
        }
        for item in after.values()
        if item["is_engine"]
    ]
    selected = candidates[0] if candidates else (fallback_engine[0] if fallback_engine else None)
    if selected:
        selected = {
            **selected,
            "selection_reason": "engine_thread_preferred" if selected.get("is_engine") else "busiest_thread_fallback",
        }
    return {
        "ok": bool(selected),
        "selected": selected,
        "candidates": candidates[:10],
        "engine_candidates": fallback_engine[:10],
        "note": "RT is applied to the selected Linux TID only; affinity/NUMA policies apply to the listed process PIDs.",
    }
