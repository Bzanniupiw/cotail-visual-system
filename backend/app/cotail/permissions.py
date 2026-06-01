from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import urllib.request


def _run(cmd: list[str], timeout: float = 6.0) -> tuple[int | None, str, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return None, "", str(exc)


def _first(text: str, n: int = 180) -> str:
    return (text or "").splitlines()[0][:n] if text else ""


def _check(results: list[dict], name: str, ok: bool, detail: str = "", hint: str = "") -> None:
    results.append({"name": name, "ok": bool(ok), "detail": detail, "hint": hint})


def _have_module(name: str) -> tuple[bool, str]:
    try:
        __import__(name)
        return True, "import ok"
    except Exception as exc:
        return False, str(exc)


def probe_permissions(
    protections: list[str] | None = None,
    interferences: list[str] | None = None,
    api_url: str = "http://127.0.0.1:8102/v1",
    gpu_idx: int = 0,
    no_perf: bool = False,
) -> dict:
    protections = protections or ["none", "nice", "cgroup", "rt", "numa"]
    interferences = interferences or []
    results: list[dict] = []

    if os.name == "nt":
        _check(results, "linux_runtime", False, "running on Windows", "run probes on the Linux serving host")
        return {"ok": False, "results": results, "failed": results}

    uid = os.getuid() if hasattr(os, "getuid") else -1
    euid = os.geteuid() if hasattr(os, "geteuid") else -1
    _check(results, "runtime_user", True, f"uid={uid}, euid={euid}")

    try:
        fd, path = tempfile.mkstemp(prefix="cotail_probe_", dir="/tmp")
        os.write(fd, b"ok")
        os.close(fd)
        os.remove(path)
        _check(results, "write_tmp", True, "/tmp writable")
    except Exception as exc:
        _check(results, "write_tmp", False, str(exc), "need write permission on /tmp")

    for module in ["psutil", "pynvml", "openai", "numpy", "torch", "transformers"]:
        ok, detail = _have_module(module)
        _check(results, f"python_module:{module}", ok, detail, f"install {module}")

    binaries = {"base": ["bash", "python3", "taskset"]}
    if "rt" in protections or "rt_sched" in protections:
        binaries["rt"] = ["chrt"]
    workload_bins = {
        "stress-ng": ["stress-ng"],
        "ffmpeg": ["ffmpeg"],
        "7zip": ["7z", "dd"],
        "redis": ["redis-server", "redis-benchmark"],
        "openssl": ["openssl"],
        "kernel_build": ["make", "gcc"],
        "memcached": ["memcached"],
        "nginx": ["nginx", "wrk"],
        "zstd-compress": ["zstd"],
        "sqlite-txn": ["sqlite3"],
        "text-search": ["rg"],
    }
    for workload in interferences:
        if workload in workload_bins:
            binaries[workload] = workload_bins[workload]
    for group in binaries.values():
        for binary in group:
            path = shutil.which(binary)
            _check(results, f"binary:{binary}", bool(path), path or "not found", f"install {binary}")

    try:
        import psutil  # type: ignore

        proc = psutil.Process(os.getpid())
        affinity = proc.cpu_affinity()
        if affinity:
            proc.cpu_affinity([affinity[0]])
            proc.cpu_affinity(affinity)
            _check(results, "set_cpu_affinity", True, f"set+restore cpu={affinity[0]}")
        else:
            _check(results, "set_cpu_affinity", False, "empty affinity")
        try:
            conns = psutil.net_connections(kind="inet")
            _check(results, "psutil_net_connections", True, f"returned={len(conns)}")
        except Exception as exc:
            _check(results, "psutil_net_connections", False, str(exc), "need LISTEN PID visibility")
    except Exception as exc:
        _check(results, "psutil_basic", False, str(exc), "need psutil")

    if "nice" in protections:
        try:
            cur = os.getpriority(os.PRIO_PROCESS, os.getpid())
            os.setpriority(os.PRIO_PROCESS, os.getpid(), cur + 1)
            os.setpriority(os.PRIO_PROCESS, os.getpid(), cur)
            _check(results, "setpriority_relax", True, f"{cur}->{cur+1}->{cur}")
        except Exception as exc:
            _check(results, "setpriority_relax", False, str(exc))

    if "rt" in protections or "rt_sched" in protections:
        chrt = shutil.which("chrt")
        if chrt:
            tid = threading.get_native_id()
            rc, out, err = _run([chrt, "-f", "-p", "50", str(tid)], 5)
            if rc == 0:
                _run([chrt, "-o", "-p", "0", str(tid)], 5)
            _check(results, "rt_sched_fifo50", rc == 0, f"rc={rc}, err={_first(err)}", "need CAP_SYS_NICE")
        else:
            _check(results, "rt_sched_fifo50", False, "chrt not found")

    if "cgroup" in protections:
        cg_root = "/sys/fs/cgroup"
        cg2 = os.path.exists(os.path.join(cg_root, "cgroup.controllers"))
        _check(results, "cgroup_v2_available", cg2, "detected" if cg2 else "missing")
        if cg2:
            probe = os.path.join(cg_root, f"cotail_probe_{os.getpid()}")
            try:
                os.mkdir(probe)
                _check(results, "cgroup_create_subdir", True, probe)
                try:
                    with open(os.path.join(probe, "cpu.weight"), "w", encoding="utf-8") as f:
                        f.write("100")
                    _check(results, "cgroup_write_cpu_weight", True, "ok")
                except Exception as exc:
                    _check(results, "cgroup_write_cpu_weight", False, str(exc), "need delegated cpu controller")
                try:
                    os.rmdir(probe)
                except Exception:
                    pass
            except Exception as exc:
                _check(results, "cgroup_create_subdir", False, str(exc), "need delegated cgroup subtree")

    parsed = urllib.parse.urlparse(api_url)
    if parsed.hostname:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((parsed.hostname, port), timeout=3):
                _check(results, "api_tcp_connect", True, f"{parsed.hostname}:{port}")
            with urllib.request.urlopen(api_url.rstrip("/") + "/models", timeout=5) as resp:
                _check(results, "api_models_endpoint", 200 <= getattr(resp, "status", 200) < 300, f"status={getattr(resp, 'status', 200)}")
        except Exception as exc:
            _check(results, "api_connectivity", False, str(exc), "start serving endpoint or update api_url")

    if shutil.which("nvidia-smi"):
        rc, out, err = _run(["nvidia-smi", "topo", "-m"], 8)
        _check(results, "nvidia_smi_topology", rc == 0 and bool(out), f"rc={rc}, err={_first(err)}")

    if not no_perf:
        perf = shutil.which("perf")
        if perf:
            rc, out, err = _run([perf, "stat", "-a", "-e", "cycles", "--", "sleep", "0.2"], 8)
            _check(results, "perf_system_wide", rc == 0, f"rc={rc}, err={_first(err)}", "need CAP_PERFMON or lower perf_event_paranoid")
        else:
            _check(results, "perf_binary", False, "perf not found")

    failed = [r for r in results if not r["ok"]]
    return {"ok": not failed, "results": results, "failed": failed, "python": sys.version.split()[0]}
