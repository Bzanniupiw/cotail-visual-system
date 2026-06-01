from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import mean

from .constants import CORE_STAGES

EPS = 1e-12


@dataclass
class StageScore:
    stage: str
    p95_us: float | None
    p99_us: float | None
    base_p95_us: float | None
    base_p99_us: float | None
    delta_p95_ratio: float | None
    delta_p99_ratio: float | None
    score_ratio: float | None


@dataclass
class CPTIResult:
    cpti_ratio: float | None
    cpti_pct: float | None
    dominant_stage: str | None
    dominant_score_ratio: float | None
    stage_scores: list[StageScore]

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


def _num(value) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _stage_lookup(metrics: dict, stage_short: str, stage_full: str, quantile: str) -> float | None:
    candidates = [
        (stage_short, quantile),
        (stage_full, quantile),
        (stage_short, quantile.upper()),
        (stage_full, quantile.upper()),
    ]
    for stage, q in candidates:
        obj = metrics.get(stage)
        if isinstance(obj, dict):
            for key in [q, f"{q}_us", q.lower(), f"{q.lower()}_us"]:
                if key in obj:
                    return _num(obj[key])
        flat_keys = [
            f"{stage}_{q}_us",
            f"{stage}_{q.upper()}_us",
            f"{stage}_{q.lower()}_us",
            f"{stage}.{q}_us",
        ]
        for key in flat_keys:
            if key in metrics:
                return _num(metrics[key])
    return None


def positive_delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or baseline <= EPS:
        return None
    return max(0.0, (current - baseline) / baseline)


def compute_cpti(stage_metrics: dict, baseline_stage_metrics: dict) -> CPTIResult:
    stage_scores: list[StageScore] = []
    raw_scores: list[float] = []
    for stage_short, stage_full in CORE_STAGES.items():
        p95 = _stage_lookup(stage_metrics, stage_short, stage_full, "p95")
        p99 = _stage_lookup(stage_metrics, stage_short, stage_full, "p99")
        base_p95 = _stage_lookup(baseline_stage_metrics, stage_short, stage_full, "p95")
        base_p99 = _stage_lookup(baseline_stage_metrics, stage_short, stage_full, "p99")
        d95 = positive_delta(p95, base_p95)
        d99 = positive_delta(p99, base_p99)
        score = None if d95 is None or d99 is None else 0.5 * (d95 + d99)
        if score is not None:
            raw_scores.append(score)
        stage_scores.append(
            StageScore(
                stage=stage_short,
                p95_us=p95,
                p99_us=p99,
                base_p95_us=base_p95,
                base_p99_us=base_p99,
                delta_p95_ratio=d95,
                delta_p99_ratio=d99,
                score_ratio=score,
            )
        )
    if len(raw_scores) != len(CORE_STAGES):
        return CPTIResult(None, None, None, None, stage_scores)
    cpti = mean(raw_scores)
    dominant = max(stage_scores, key=lambda s: -1.0 if s.score_ratio is None else s.score_ratio)
    return CPTIResult(
        cpti_ratio=cpti,
        cpti_pct=100.0 * cpti,
        dominant_stage=dominant.stage,
        dominant_score_ratio=dominant.score_ratio,
        stage_scores=stage_scores,
    )


def compute_cts(unprotected_cpti: float | None, protected_cpti: float | None) -> float | None:
    if unprotected_cpti is None or protected_cpti is None or abs(float(unprotected_cpti)) <= EPS:
        return None
    return 1.0 - float(protected_cpti) / float(unprotected_cpti)

