from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import signal
import sys
from pathlib import Path


def _install_signal_handlers() -> None:
    def _raise_keyboard_interrupt(signum, frame):  # noqa: ANN001
        raise KeyboardInterrupt(f"received signal {signum}")

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is not None:
            try:
                signal.signal(sig, _raise_keyboard_interrupt)
            except Exception:
                pass


def _find_reference_script() -> Path:
    explicit = os.environ.get("COTAIL_REFERENCE_SCRIPT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    here = Path(__file__).resolve()
    candidates = [
        here.parents[4] / "improved_real_stress_various_new.py",
        here.parents[4] / "improved_real_stress_various_copy.py",
        Path.cwd().parent / "improved_real_stress_various_new.py",
        Path.cwd().parent / "improved_real_stress_various_copy.py",
        Path.cwd() / "improved_real_stress_various_new.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_reference(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"reference script not found: {path}. Set COTAIL_REFERENCE_SCRIPT to improved_real_stress_various_new.py"
        )
    spec = importlib.util.spec_from_file_location("cotail_reference_stress", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reference script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _CompatProtection:
    def __init__(self, stress_cgroup_path: str = "", numa_active: bool = False) -> None:
        self._stress_cgroup_path = stress_cgroup_path
        self._numa_active = numa_active

    def is_numa_isolate_active(self) -> bool:
        return self._numa_active

    def is_cgroup_active(self) -> bool:
        return bool(self._stress_cgroup_path)

    def is_cgroup_scope_fallback_active(self) -> bool:
        return False

    def get_scope_stress_cpu_weight(self) -> int:
        return 50

    def get_stress_cgroup_path(self) -> str:
        return self._stress_cgroup_path


def _configure_reference(module, cpus: str, workers: int) -> None:
    cfg = module.CONFIG
    cfg.setdefault("core_partition", {})
    cfg["core_partition"]["battle_cores"] = cpus
    cfg["core_partition"]["stress_workers"] = int(workers)
    cfg.setdefault("numa_isolate", {})
    cfg["numa_isolate"]["interference_cpus"] = cpus
    cfg["numa_isolate"]["stress_workers"] = int(workers)


def _disable_unrelated_ffmpeg_prepare(module, workload: str) -> None:
    if workload.strip().lower() == "ffmpeg":
        return
    manager_cls = getattr(module, "InterferenceManager", None)
    if manager_cls is None:
        return

    def _noop_ensure_ffmpeg_input(self):  # noqa: ANN001
        return None

    manager_cls._ensure_ffmpeg_input = _noop_ensure_ffmpeg_input


async def _run(args: argparse.Namespace) -> int:
    module = _load_reference(_find_reference_script())
    _configure_reference(module, args.cpus, args.workers)
    _disable_unrelated_ffmpeg_prepare(module, args.workload)
    protection = _CompatProtection(args.stress_cgroup_path, args.numa_active)
    numa_cpus = module.CPUTopology._parse_cpu_range(
        module.CONFIG.get("numa_isolate", {}).get("vllm_cpus", "")
    )
    manager = module.InterferenceManager(numa_cpus=numa_cpus, protection_mgr=protection)
    try:
        await manager.start_async(args.workload)
        await asyncio.sleep(max(1, int(args.duration)))
    finally:
        manager.stop()
    return 0


def main() -> int:
    _install_signal_handlers()
    parser = argparse.ArgumentParser(description="Run CoTail workloads through improved_real_stress_various_new.py")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--stress-cgroup-path", default="")
    parser.add_argument("--numa-active", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
