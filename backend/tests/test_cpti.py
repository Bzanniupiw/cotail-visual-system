from backend.app.cotail.cpti import compute_cpti, compute_cts


def test_cpti_positive_tail_average():
    base = {
        "vllm.scheduler.step": {"p95_us": 10, "p99_us": 10},
        "vllm.batch.construct": {"p95_us": 10, "p99_us": 10},
        "vllm.model.execute": {"p95_us": 10, "p99_us": 10},
        "vllm.model.forward": {"p95_us": 10, "p99_us": 10},
    }
    cur = {
        "vllm.scheduler.step": {"p95_us": 20, "p99_us": 20},
        "vllm.batch.construct": {"p95_us": 10, "p99_us": 10},
        "vllm.model.execute": {"p95_us": 10, "p99_us": 10},
        "vllm.model.forward": {"p95_us": 10, "p99_us": 10},
    }
    res = compute_cpti(cur, base)
    assert round(res.cpti_ratio, 3) == 0.25
    assert compute_cts(1.0, 0.1) == 0.9

