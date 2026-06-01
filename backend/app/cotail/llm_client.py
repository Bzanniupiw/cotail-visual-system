from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any

from .constants import SCRIPT_COMPAT
from .cpu import parse_cpu_range

TOKENIZER_CACHE: dict[str, Any] = {}
CLIENT_AFFINITY_APPLIED = False


def _apply_script_client_affinity() -> dict[str, Any] | None:
    global CLIENT_AFFINITY_APPLIED
    if CLIENT_AFFINITY_APPLIED or os.name == "nt":
        return None
    try:
        import psutil  # type: ignore

        safe_cores = parse_cpu_range(str(SCRIPT_COMPAT["safe_cores"]))
        proc = psutil.Process(os.getpid())
        proc.cpu_affinity(safe_cores)
        CLIENT_AFFINITY_APPLIED = True
        return {"ok": True, "pid": os.getpid(), "safe_cores": SCRIPT_COMPAT["safe_cores"]}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "safe_cores": SCRIPT_COMPAT["safe_cores"]}


def _url(api_base: str, path: str) -> str:
    return api_base.rstrip("/") + "/" + path.lstrip("/")


def _tokenizer_count(model: str, text: str) -> int | None:
    if not text:
        return 0
    if model not in TOKENIZER_CACHE:
        try:
            from transformers import AutoTokenizer  # type: ignore

            TOKENIZER_CACHE[model] = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        except Exception:
            TOKENIZER_CACHE[model] = None
    tokenizer = TOKENIZER_CACHE.get(model)
    if tokenizer is None:
        return None
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


def _approx_tokens(text: str, model: str = "") -> tuple[int, str]:
    tokenized = _tokenizer_count(model, text)
    if tokenized is not None:
        return max(1, int(tokenized)), "transformers_tokenizer"
    # Good enough for live operational monitoring when tokenizer is not loaded.
    return max(1, int(len(text) / 1.8)), "char_heuristic_len_div_1.8"


def list_models(api_base: str, timeout_s: float = 5.0) -> dict[str, Any]:
    start = time.time()
    try:
        with urllib.request.urlopen(_url(api_base, "models"), timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        return {"ok": True, "latency_ms": (time.time() - start) * 1000.0, "models": data.get("data", data)}
    except Exception as exc:
        return {"ok": False, "latency_ms": (time.time() - start) * 1000.0, "error": str(exc)}


def chat_completion(
    api_base: str,
    model: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    stream: bool = True,
    timeout_s: float = 120.0,
    include_usage: bool = True,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": bool(stream),
    }
    if stream and include_usage:
        payload["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        _url(api_base, "chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    start = time.time()
    first_token_t: float | None = None
    text_parts: list[str] = []
    usage_completion_tokens: int | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if stream:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    usage = obj.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        try:
                            usage_completion_tokens = int(usage.get("completion_tokens"))
                        except Exception:
                            pass
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        if first_token_t is None:
                            first_token_t = time.time()
                        text_parts.append(content)
            else:
                obj = json.loads(resp.read().decode("utf-8", errors="replace"))
                choices = obj.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    text_parts.append(msg.get("content") or "")
                    first_token_t = time.time()
                usage = obj.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    usage_completion_tokens = int(usage.get("completion_tokens"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        if stream and include_usage and exc.code in (400, 404, 422):
            retry = chat_completion(
                api_base=api_base,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                timeout_s=timeout_s,
                include_usage=False,
            )
            if retry.get("ok"):
                retry["usage_retry"] = "stream_options.include_usage unsupported; retried without usage"
                return retry
        return {"ok": False, "error": f"HTTP {exc.code}: {detail}", "total_ms": (time.time() - start) * 1000.0}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "total_ms": (time.time() - start) * 1000.0}

    end = time.time()
    output = "".join(text_parts)
    if usage_completion_tokens is not None:
        tokens = max(1, usage_completion_tokens)
        token_count_method = "api_usage.completion_tokens"
    else:
        tokens, token_count_method = _approx_tokens(output, model)
    total_ms = (end - start) * 1000.0
    ttft_ms = None if first_token_t is None else (first_token_t - start) * 1000.0
    decode_ms = None if first_token_t is None else max(0.0, (end - first_token_t) * 1000.0)
    tpot_ms = None if decode_ms is None else decode_ms / max(1, tokens - 1)
    return {
        "ok": True,
        "api_base": api_base,
        "model": model,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "total_ms": total_ms,
        "tokens": tokens,
        "token_count_method": token_count_method,
        "throughput_tok_s": tokens / max(1e-9, total_ms / 1000.0),
        "output_preview": output[:1000],
    }


def benchmark_chat(
    api_base: str,
    model: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    stream: bool = True,
    timeout_s: float = 120.0,
    concurrency: int = 4,
    total_requests: int = 8,
) -> dict[str, Any]:
    client_affinity = _apply_script_client_affinity()
    concurrency = max(1, int(concurrency))
    total_requests = max(1, int(total_requests))
    start = time.time()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                chat_completion,
                api_base,
                model,
                prompt,
                max_tokens,
                temperature,
                stream,
                timeout_s,
            )
            for _ in range(total_requests)
        ]
        for fut in as_completed(futures):
            rows.append(fut.result())
    wall_s = time.time() - start
    ok_rows = [r for r in rows if r.get("ok")]
    tokens = sum(int(r.get("tokens") or 0) for r in ok_rows)

    def avg(key: str) -> float | None:
        vals = [float(r[key]) for r in ok_rows if r.get(key) is not None]
        return mean(vals) if vals else None

    return {
        "ok": bool(ok_rows),
        "success_count": len(ok_rows),
        "total_requests": total_requests,
        "concurrency": concurrency,
        "wall_time_s": wall_s,
        "total_tokens": tokens,
        "token_count_methods": sorted(set(str(r.get("token_count_method") or "unknown") for r in ok_rows)),
        "client_affinity": client_affinity,
        "throughput_tok_s": tokens / max(1e-9, wall_s),
        "ttft_avg_ms": avg("ttft_ms"),
        "tpot_avg_ms": avg("tpot_ms"),
        "request_total_avg_ms": avg("total_ms"),
        "errors": [r.get("error") for r in rows if not r.get("ok")],
        "samples": rows[: min(5, len(rows))],
    }
