from __future__ import annotations

import os
import signal
import shutil
import time
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class WorkloadProcess:
    pid: int
    ppid: int | None
    name: str
    username: str
    workload: str
    reason: str
    cmdline: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cmdline(proc: Any) -> str:
    try:
        cmdline = proc.cmdline()
        return " ".join(str(x) for x in cmdline)
    except Exception:
        return ""


def _proc_user(proc: Any) -> str:
    try:
        return str(proc.username())
    except Exception:
        return ""


def _matches_cotail_workload(name: str, cmdline: str, workloads: set[str]) -> tuple[str, str] | None:
    text = f"{name} {cmdline}".lower()
    if "backend.app.cotail.script_compat_runner" in text and "--workload" in text:
        for workload in [
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
        ]:
            if f"--workload {workload}" in text and (not workloads or workload in workloads):
                return workload, "reference script workload launcher"

    candidates = [
        ("stress-ng", "_stress_ng" in text or "stress_ng_loop" in text),
        ("stress-ng", "stress-ng" in text and ("--cpu-method matrixprod" in text or "--vm-method" in text or "stress-ng-cpu" in text)),
        ("ffmpeg", "_ffmpeg_loop_" in text),
        ("ffmpeg", "ffmpeg" in text and ("testsrc2=size=1920x1080" in text or "test_input_benchmark.mp4" in text or "libx264" in text)),
        ("7zip", "_7zip_loop_" in text or "_7zip_testdata.bin" in text),
        ("7zip", (" 7z " in f" {text} " or "7z b" in text) and "-mmt" in text),
        ("redis", "_redis_bench_" in text),
        ("redis", ("redis-server" in text or "redis-benchmark" in text) and ("16389" in text or "6400" in text or "maxmemory 512mb" in text or "maxmemory 256mb" in text or "lpush,lpop,sadd" in text)),
        ("openssl", "_openssl_loop_" in text),
        ("openssl", "openssl" in text and " speed " in f" {text} " and ("rsa2048" in text or "aes-256-cbc" in text)),
        ("kernel_build", "_cotail_kernel_build" in text or "cotail_kernel_workers" in text or "_kbuild_" in text or "_linux_src" in text),
        ("memcached", "memcached" in text and (" -m 1024 " in f" {text} " or " -m 2048 " in f" {text} " or "11299" in text or "cotail_memcached" in text)),
        ("nginx", "_cotail_nginx" in text or "cotail_nginx_port" in text or "_nginx_www" in text or "_nginx_bench.conf" in text),
        ("nginx", "_nginx_www" in text or "_nginx_bench.conf" in text),
        ("zstd-compress", "_zstd_compress_loop_" in text),
        ("zstd-compress", "cotail_heldout/compress" in text),
        ("sqlite-txn", "_sqlite_txn_loop.py" in text),
        ("sqlite-txn", "cotail_heldout/sqlite" in text or "cotail_sqlite_workers" in text),
        ("image-preprocess", "_image_preprocess_loop.py" in text or "_image_preprocess_numpy_loop.py" in text),
        ("image-preprocess", "cotail_img_workers" in text or "image_preprocess" in text or "image_preprocess_loop.py" in text),
        ("text-search", "_text_search_loop_" in text),
        ("text-search", "cotail_heldout/text" in text),
    ]
    for workload, matched in candidates:
        if matched and (not workloads or workload in workloads):
            return workload, "cotail command fingerprint"
    return None


def _parent_can_own_workload(row: dict[str, Any], workload: str) -> bool:
    text = f"{row.get('name', '')} {row.get('cmdline', '')}".lower()
    name = str(row.get("name") or "").lower()
    if workload and workload in text:
        return True
    if "backend.app.cotail.script_compat_runner" in text:
        return True
    if "improved_real_stress_various_new.py" in text or "improved_real_stress_various_copy.py" in text:
        return True
    if name not in {"bash", "sh", "python", "python3"}:
        return False
    markers = {
        "stress-ng": ["stress_ng_loop", "stress-ng"],
        "ffmpeg": ["_ffmpeg_loop_", "test_input_benchmark.mp4"],
        "7zip": ["_7zip_loop_", "_7zip_testdata.bin"],
        "redis": ["_redis_bench_", "redis-benchmark", "redis-server"],
        "openssl": ["_openssl_loop_", "aes-256-cbc"],
        "kernel_build": ["_kbuild_", "_linux_src", "_cotail_kernel_build"],
        "memcached": ["_mc_bench.py", "11299", "memcached"],
        "nginx": ["_nginx_www", "_nginx_bench.conf", "cotail_nginx_port"],
        "zstd-compress": ["_zstd_compress_loop_", "cotail_heldout/compress"],
        "sqlite-txn": ["_sqlite_txn_loop.py", "cotail_heldout/sqlite"],
        "image-preprocess": ["_image_preprocess_loop.py", "_image_preprocess_numpy_loop.py", "cotail_heldout/images"],
        "text-search": ["_text_search_loop_", "cotail_heldout/text"],
    }
    return any(marker in text for marker in markers.get(workload, []))


def _maybe_cotail_name(name: str) -> bool:
    lower = str(name or "").lower()
    candidates = [
        "stress-ng",
        "stress-ng-cpu",
        "ffmpeg",
        "7z",
        "redis",
        "openssl",
        "gcc",
        "cc1",
        "memcached",
        "nginx",
        "zstd",
        "xz",
        "gzip",
        "sqlite",
        "rg",
        "grep",
        "bash",
        "sh",
        "python",
        "python3",
    ]
    return any(item in lower for item in candidates)


def _collect_process_targets(root_pid: int, psutil: Any) -> tuple[list[Any], list[int]]:
    try:
        root = psutil.Process(root_pid)
    except Exception:
        return [], []
    targets = []
    seen: set[int] = set()
    try:
        candidates = root.children(recursive=True) + [root]
    except Exception:
        candidates = [root]
    pgids: set[int] = set()
    for proc in candidates:
        try:
            pid = int(proc.pid)
            if pid in seen:
                continue
            seen.add(pid)
            targets.append(proc)
            if os.name != "nt":
                try:
                    pgids.add(os.getpgid(pid))
                except Exception:
                    pass
        except Exception:
            continue
    return targets, sorted(pgids)


def _terminate_targets(targets: list[Any], pgids: list[int], psutil: Any, *, force: bool, grace_s: float = 5.0) -> dict[str, Any]:
    results: dict[str, Any] = {"target_pids": [], "pgids": pgids, "alive_after_term": [], "alive_after_kill": []}
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass
    for target in targets:
        try:
            results["target_pids"].append(int(target.pid))
            target.terminate()
        except Exception:
            pass
    _, alive = psutil.wait_procs(targets, timeout=grace_s)
    results["alive_after_term"] = [int(proc.pid) for proc in alive if getattr(proc, "pid", None)]
    if force and alive:
        for pgid in pgids:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
        for target in alive:
            try:
                target.kill()
            except Exception:
                pass
        _, alive = psutil.wait_procs(alive, timeout=2.0)
        results["alive_after_kill"] = [int(proc.pid) for proc in alive if getattr(proc, "pid", None)]
    return results


def _cleanup_temp_artifacts() -> None:
    import glob

    for pattern in [
        "/tmp/_*_loop*.sh",
        "/tmp/_redis_bench_*.sh",
        "/tmp/_mc_bench.py",
        "/tmp/_nginx_bench.conf",
        "/tmp/_nginx_bench.pid",
        "/tmp/_kbuild_*.sh",
        "/tmp/_zstd_compress_loop_*.sh",
        "/tmp/_sqlite_txn_loop.py",
        "/tmp/_image_preprocess_loop.py",
        "/tmp/_image_preprocess_numpy_loop.py",
        "/tmp/_text_search_loop_*.sh",
        "/tmp/_7zip_out_*.7z",
        "/tmp/cotail_heldout/compress/*.zst",
        "/tmp/cotail_heldout/compress/*.xz",
        "/tmp/cotail_heldout/compress/*.gz",
        "/tmp/cotail_heldout/rg.*.out",
        "/tmp/cotail_heldout/grep.*.out",
    ]:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except Exception:
                pass
    for path in ["/tmp/_nginx_www", "/tmp/cotail_heldout/images_out"]:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass


def list_cotail_workload_processes(
    *,
    workloads: list[str] | None = None,
    current_user_only: bool = True,
) -> list[dict[str, Any]]:
    try:
        import psutil  # type: ignore
    except Exception:
        return []

    wanted = {w.strip().lower() for w in (workloads or []) if w.strip()}
    current_user = _proc_user(psutil.Process(os.getpid()))
    proc_rows: dict[int, dict[str, Any]] = {}
    children_by_ppid: dict[int, list[int]] = {}
    matches: dict[int, WorkloadProcess] = {}

    def add_match(pid: int, workload: str, reason: str) -> bool:
        row = proc_rows.get(pid)
        if not row or pid in matches:
            return False
        matches[pid] = WorkloadProcess(
            pid=pid,
            ppid=row["ppid"],
            name=row["name"],
            username=row["username"],
            workload=workload,
            reason=reason,
            cmdline=row["cmdline"][:800],
        )
        return True

    for proc in psutil.process_iter(["pid", "ppid", "name", "username"]):
        try:
            pid = int(proc.info.get("pid") or proc.pid)
            if pid == os.getpid():
                continue
            username = str(proc.info.get("username") or "")
            if current_user_only and current_user and username != current_user:
                continue
            ppid_raw = proc.info.get("ppid")
            ppid = int(ppid_raw) if ppid_raw else None
            name = str(proc.info.get("name") or "")
            cmdline = _cmdline(proc) if _maybe_cotail_name(name) else ""
            proc_rows[pid] = {
                "pid": pid,
                "ppid": ppid,
                "name": name,
                "username": username,
                "cmdline": cmdline,
            }
            if ppid is not None:
                children_by_ppid.setdefault(ppid, []).append(pid)
        except Exception:
            continue

    for pid, row in proc_rows.items():
        matched = _matches_cotail_workload(row["name"], row["cmdline"], wanted)
        if not matched:
            continue
        workload, reason = matched
        add_match(pid, workload, reason)

    changed = True
    while changed:
        changed = False
        for pid, item in list(matches.items()):
            parent_pid = proc_rows.get(pid, {}).get("ppid")
            depth = 0
            while parent_pid and parent_pid in proc_rows and depth < 8:
                parent = proc_rows[parent_pid]
                explicit = _matches_cotail_workload(parent["name"], parent["cmdline"], wanted)
                if explicit:
                    parent_workload, parent_reason = explicit
                elif _parent_can_own_workload(parent, item.workload):
                    parent_workload, parent_reason = item.workload, f"parent of matched {pid}"
                else:
                    break
                if add_match(int(parent_pid), parent_workload, parent_reason):
                    changed = True
                parent_pid = parent.get("ppid")
                depth += 1

    for pid, item in list(matches.items()):
        stack = list(children_by_ppid.get(pid, []))
        while stack:
            child_pid = stack.pop()
            if child_pid not in proc_rows:
                continue
            if add_match(child_pid, item.workload, f"child of matched {pid}"):
                stack.extend(children_by_ppid.get(child_pid, []))

    return [item.to_dict() for item in sorted(matches.values(), key=lambda row: (row.workload, row.pid))]


def cleanup_cotail_workloads(
    *,
    workloads: list[str] | None = None,
    current_user_only: bool = True,
    dry_run: bool = True,
    force: bool = True,
) -> dict[str, Any]:
    initial_rows = list_cotail_workload_processes(workloads=workloads, current_user_only=current_user_only)
    if dry_run:
        return {"ok": True, "dry_run": True, "count": len(initial_rows), "processes": initial_rows, "results": []}

    try:
        import psutil  # type: ignore
    except Exception as exc:
        return {"ok": False, "dry_run": False, "error": f"psutil unavailable: {exc}", "processes": rows, "results": []}

    results = []
    total_roots: set[int] = set()
    max_passes = 5
    rows = initial_rows
    for pass_idx in range(1, max_passes + 1):
        if not rows:
            break
        roots = []
        matched_pids = {int(row["pid"]) for row in rows}
        for row in rows:
            pid = int(row["pid"])
            ppid = row.get("ppid")
            if ppid not in matched_pids:
                roots.append(pid)
        total_roots.update(int(pid) for pid in roots)

        pass_results = []
        for pid in sorted(set(roots)):
            try:
                targets, pgids = _collect_process_targets(pid, psutil)
                result = _terminate_targets(targets, pgids, psutil, force=force, grace_s=5.0)
                pass_results.append({"pid": pid, "ok": not result["alive_after_kill"], **result})
            except Exception as exc:
                if os.name != "nt":
                    try:
                        try:
                            os.killpg(os.getpgid(pid), signal.SIGTERM)
                        except Exception:
                            os.kill(pid, signal.SIGTERM)
                        time.sleep(1.0)
                        if force:
                            try:
                                os.killpg(os.getpgid(pid), signal.SIGKILL)
                            except Exception:
                                os.kill(pid, signal.SIGKILL)
                        pass_results.append({"pid": pid, "ok": True, "fallback": "os.kill"})
                        continue
                    except Exception as kill_exc:
                        pass_results.append({"pid": pid, "ok": False, "error": f"{exc}; fallback={kill_exc}"})
                else:
                    pass_results.append({"pid": pid, "ok": False, "error": str(exc)})

        _cleanup_temp_artifacts()
        time.sleep(0.5)
        rows = list_cotail_workload_processes(workloads=workloads, current_user_only=current_user_only)
        results.append(
            {
                "pass": pass_idx,
                "matched_before": len(matched_pids),
                "root_count": len(set(roots)),
                "remaining_after": len(rows),
                "roots": pass_results,
            }
        )
        if not rows:
            break
        if pass_idx >= 2:
            previous = int(results[-2].get("remaining_after", 10**9))
            current = int(results[-1].get("remaining_after", 10**9))
            if current >= previous and not any(item.get("ok") for item in pass_results):
                break

    remaining = list_cotail_workload_processes(workloads=workloads, current_user_only=current_user_only)
    return {
        "ok": not remaining,
        "dry_run": False,
        "count": len(initial_rows),
        "root_count": len(total_roots),
        "passes": len(results),
        "processes": initial_rows,
        "remaining_count": len(remaining),
        "remaining_processes": remaining,
        "results": results,
    }
