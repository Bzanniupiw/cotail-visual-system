from backend.app.cotail.protections import build_protection_plan
from backend.app.cotail.workload_runner import build_workload_command


def test_workload_delegates_to_reference_runner():
    cmd = build_workload_command("stress-ng", cpus="", workers=60, duration_s=18)
    assert cmd.command[:3] == ["python3", "-m", "backend.app.cotail.script_compat_runner"]
    assert "--workload" in cmd.command
    assert "stress-ng" in cmd.command
    assert "--cpus" in cmd.command
    assert "0-60,128-188" in cmd.command


def test_cgroup_plan_exposes_stress_limit_path():
    plan = build_protection_plan(
        "cgroup",
        vllm_pids=[123],
        battle_cores="0-60,128-188",
        cgroup_base="/tmp",
    )
    paths = [action.target for action in plan.actions if action.kind == "cgroup"]
    assert any("vllm_protect" in path for path in paths)
    assert any("stress_limit" in path for path in paths)
