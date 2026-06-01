from backend.app.cotail.gpu_monitor import parse_nvidia_smi_process_table, parse_query_gpu_csv
from backend.app.cotail.scheduler import GpuSelectionPolicy, select_idle_gpu


def test_parse_gpu_csv_and_select_idle():
    text = """0, GPU-0, NVIDIA GeForce RTX 4090, 00000000:01:00.0, P2, 63, 289, 450, 7485, 24564, 97
1, GPU-1, NVIDIA GeForce RTX 4090, 00000000:23:00.0, P8, 29, 30, 450, 19843, 24564, 0
4, GPU-4, NVIDIA GeForce RTX 4090, 00000000:81:00.0, P8, 26, 22, 450, 18, 24564, 0
"""
    gpus = [g.to_dict() for g in parse_query_gpu_csv(text)]
    gpus[1]["processes"] = [{"process_type": "C", "pid": 2383695}]
    gpus[1]["compute_process_count"] = 1
    selected = select_idle_gpu(gpus, GpuSelectionPolicy(min_free_memory_mib=18000))
    assert selected["selected_gpu"] == 4


def test_parse_process_table():
    text = """
|    5   N/A  N/A         1631522      C   VLLM::EngineCore                      23512MiB |
|    7   N/A  N/A         1868952      C   ...uild-cuda128/bin/llama-server      21504MiB |
"""
    procs = parse_nvidia_smi_process_table(text)
    assert len(procs) == 2
    assert procs[0].gpu_index == 5
    assert procs[0].process_name == "VLLM::EngineCore"

