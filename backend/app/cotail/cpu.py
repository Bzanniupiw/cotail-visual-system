from __future__ import annotations


def parse_cpu_range(cpu_range: str | None) -> list[int]:
    cpus: list[int] = []
    if not cpu_range:
        return cpus
    for part in str(cpu_range).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def format_cpu_range(cpus: list[int] | set[int] | tuple[int, ...]) -> str:
    vals = sorted(set(int(c) for c in cpus))
    if not vals:
        return ""
    spans: list[str] = []
    start = prev = vals[0]
    for cpu in vals[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        spans.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    spans.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(spans)


def intersection_as_range(left: str, right: str) -> str:
    return format_cpu_range(set(parse_cpu_range(left)) & set(parse_cpu_range(right)))


def first_n_as_range(cpu_range: str, n: int) -> str:
    return format_cpu_range(parse_cpu_range(cpu_range)[: max(0, int(n))])

