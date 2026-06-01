# Source Logic Map

The new project was derived from the working local experiment scripts rather
than the artifact implementation.

## Files Scanned

The local scan found 23,339 Python files including vendored `vllm/` source.
For CoTail system logic, the project uses the 66 non-vendored, non-artifact
Python files in the workspace.  The main sources were:

- `improved_real_stress_various_copy.py`
  - framework-aware process discovery
  - GPU-to-NUMA CPU discovery
  - workload command launch logic
  - `none`, `nice`, `cgroup`, `rt_sched`, `numa_isolate`, `numa+rt` policies
  - benchmark macro metrics
  - hardware metric collection
- `improved_real_stress_various.py`, `real_stress_various.py`
  - earlier clean versions of topology, interference, and policy control
- `collect_cpu_load_hardware_metrics.py`
  - workload-only CPU/cache/context/I/O signal collection
  - workload command definitions, including held-out workloads
- `run_cotenant_alone_baseline.py`
  - co-tenant utility and workload command patterns
- `run_enginecore_schedstat_experiment.py`
  - EngineCore TID identification by light probe and CPU-time delta
  - schedstat wait/run accounting
- `run_delay_injection_experiment.py`
  - macro metric calculation and stage event loading helpers
- `nsys_colocation_gpu1_20260418_233356/extract_vllm_nvtx_stage_metrics.py`
  - Nsight SQLite NVTX table discovery and vLLM stage summaries
- `nsys_colocation_gpu1_20260418_233356/vllm_nvtx_stage_analysis_live/calculate_cpti.py`
  - CPTI/CTS definitions and stage decomposition
- `vllm_min_privilege_probe.py`
  - minimum privilege checks for RT, cgroup, perf, NVML, binaries, API
- `numa_isolate_check.py`
  - NUMA CPU range validation and affinity smoke tests
- `cgroup_subtree_delegate_check.py`
  - cgroup v2 delegation checks
- `systemd_scope_check.py`
  - user-scope cgroup fallback feasibility
- `permission_change_evidence_collector.py`
  - runtime evidence collection for cgroup, affinity, and scheduler state

## Design Choices

- The UI/API does not import the old scripts directly because several files
  contain host-specific constants, duplicated methods, or mojibake comments.
- The reusable logic has been rewritten into small modules with explicit inputs.
- Dangerous actions are represented as plans first; execution is opt-in.
- The decision logic follows CoTail Algorithm 1:
  hardware risk screen, unprotected profiling, CPTI/dominant stage diagnosis,
  policy selection, and SLO/CTS validation.

