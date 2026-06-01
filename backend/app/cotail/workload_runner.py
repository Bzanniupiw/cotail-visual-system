from __future__ import annotations

import shlex
import os
from dataclasses import asdict, dataclass
from textwrap import dedent

from .constants import SCRIPT_COMPAT, WORKLOADS
from .cpu import format_cpu_range, parse_cpu_range


@dataclass
class WorkloadCommand:
    workload: str
    command: list[str]
    required_binaries: list[str]
    notes: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["shell_preview"] = " ".join(shlex.quote(x) for x in self.command)
        return data


def _prefix(cpus: str) -> list[str]:
    return ["taskset", "-c", cpus] if cpus else []


def _bash(script: str) -> list[str]:
    return ["bash", "-lc", dedent(script).strip()]


def _script_cpus(cpus: str) -> tuple[str, list[int]]:
    text = cpus.strip() or str(SCRIPT_COMPAT["battle_cores"])
    values = parse_cpu_range(text)
    return format_cpu_range(values), values


def build_workload_command(
    workload: str,
    cpus: str = "",
    workers: int = 60,
    duration_s: int = 60,
    stress_cgroup_path: str = "",
) -> WorkloadCommand:
    cpu_list_str, cpu_values = _script_cpus(cpus)
    prefix = _prefix(cpu_list_str)
    worker_count = max(1, int(workers or 1))
    duration = max(1, int(duration_s or 1))
    w = workload.strip().lower()

    if os.environ.get("COTAIL_USE_REFERENCE_WORKLOADS", "1").strip() != "0" and w in WORKLOADS:
        command = [
            "python3",
            "-m",
            "backend.app.cotail.script_compat_runner",
            "--workload",
            w,
            "--cpus",
            cpu_list_str,
            "--workers",
            str(worker_count),
            "--duration",
            str(duration),
        ]
        if stress_cgroup_path:
            command.extend(["--stress-cgroup-path", stress_cgroup_path])
        return WorkloadCommand(
            w,
            command,
            ["python3"],
            "reference-compatible: delegates workload launch to improved_real_stress_various_new.py",
        )

    if w == "stress-ng":
        return WorkloadCommand(
            w,
            prefix
            + [
                "stress-ng",
                "--cpu",
                str(worker_count),
                "--cpu-method",
                "matrixprod",
                "--timeout",
                f"{duration}s",
                "--metrics-brief",
            ],
            ["taskset", "stress-ng"],
            "direct CPU saturation workload",
        )

    if w == "ffmpeg":
        script = f"""
        set -e
        command -v ffmpeg >/dev/null
        ffmpeg -hide_banner -loglevel error \
          -f lavfi -i testsrc2=size=1920x1080:rate=60 \
          -threads {worker_count} -t {duration} \
          -vf scale=3840:2160,format=yuv420p \
          -f null - >/dev/null
        """
        return WorkloadCommand(w, prefix + _bash(script), ["taskset", "bash", "ffmpeg"], "self-contained synthetic video transcode")

    if w == "7zip":
        threads_per_instance = 8
        n_instances = max(1, worker_count // threads_per_instance)
        instance_specs = []
        for i in range(n_instances):
            selected = [str(cpu_values[(i * threads_per_instance + j) % len(cpu_values)]) for j in range(threads_per_instance)]
            instance_specs.append(",".join(selected))
        specs = " ".join(shlex.quote(x) for x in instance_specs)
        script = f"""
        set -e
        command -v 7z >/dev/null
        test_file=/tmp/_7zip_testdata.bin
        if [ ! -f "$test_file" ]; then
          dd if=/dev/urandom of="$test_file" bs=1M count=512 >/dev/null 2>&1
        fi
        cleanup() {{ pkill -P $$ >/dev/null 2>&1 || true; rm -f /tmp/_7zip_out_*.7z; }}
        trap cleanup EXIT
        end=$((SECONDS + {duration}))
        idx=0
        for cores in {specs}; do
          (
            out=/tmp/_7zip_out_$idx.7z
            while [ "$SECONDS" -lt "$end" ]; do
              taskset -c "$cores" 7z a -t7z -mx=9 -mmt={threads_per_instance} -y "$out" "$test_file" >/dev/null 2>&1 || true
              rm -f "$out"
            done
          ) &
          idx=$((idx + 1))
        done
        wait
        """
        return WorkloadCommand(w, _bash(script), ["taskset", "bash", "7z"], "script-compatible 7zip: 512MB file, instances x 8 threads")

    if w == "redis":
        threads = max(4, min(worker_count, 64))
        clients = max(64, min(worker_count * 4, 512))
        script = f"""
        set -e
        command -v redis-server >/dev/null
        command -v redis-benchmark >/dev/null
        port=$((16389 + $$ % 1000))
        cleanup() {{
          if [ -n "${{server_pid:-}}" ]; then kill "$server_pid" >/dev/null 2>&1 || true; fi
          pkill -P $$ >/dev/null 2>&1 || true
        }}
        trap cleanup EXIT
        redis-server --save '' --appendonly no --protected-mode no --bind 127.0.0.1 \
          --port "$port" --maxmemory 512mb --maxmemory-policy allkeys-lru >/dev/null 2>&1 &
        server_pid=$!
        sleep 2
        end=$((SECONDS + {duration}))
        while [ "$SECONDS" -lt "$end" ]; do
          redis-benchmark -h 127.0.0.1 -p "$port" -t set,get,lpush,lpop,sadd \
            -n 2000000 -d 1024 -c {clients} --threads {threads} -q >/dev/null 2>&1 || true
        done
        """
        return WorkloadCommand(w, prefix + _bash(script), ["taskset", "bash", "redis-server", "redis-benchmark"], "redis-server plus redis-benchmark clients")

    if w == "openssl":
        cpu_words = " ".join(str(cpu_values[i % len(cpu_values)]) for i in range(worker_count))
        script = f"""
        set -e
        command -v openssl >/dev/null
        cleanup() {{ pkill -P $$ >/dev/null 2>&1 || true; }}
        trap cleanup EXIT
        for cpu in {cpu_words}; do
          taskset -c "$cpu" openssl speed -seconds {duration} aes-256-cbc >/dev/null 2>&1 &
        done
        wait
        """
        return WorkloadCommand(w, _bash(script), ["taskset", "bash", "openssl"], "script-compatible OpenSSL: one aes-256-cbc process per worker/core")

    if w == "kernel_build":
        script = f"""
        set -e
        work=/tmp/_cotail_kernel_build
        mkdir -p "$work"
        if ! command -v python3 >/dev/null; then
          echo "python3 missing; using stress-ng/vm fallback when available" >&2
          if command -v stress-ng >/dev/null; then
            exec stress-ng --vm {max(1, worker_count // 2)} --vm-bytes 128M --vm-method all --timeout {duration}s
          fi
          exit 127
        fi
        if ! command -v gcc >/dev/null; then
          echo "gcc missing; using stress-ng/vm fallback when available" >&2
          if command -v stress-ng >/dev/null; then
            exec stress-ng --vm {max(1, worker_count // 2)} --vm-bytes 128M --vm-method all --timeout {duration}s
          fi
          COTAIL_KERNEL_WORKERS="{worker_count}" COTAIL_DURATION="{duration}" python3 - <<'PY'
import hashlib
import multiprocessing as mp
import os
import time

workers = int(os.environ["COTAIL_KERNEL_WORKERS"])
deadline = time.time() + float(os.environ["COTAIL_DURATION"])

def worker() -> None:
    data = os.urandom(1024 * 1024)
    while time.time() < deadline:
        data = hashlib.sha256(data).digest() * 32768

procs = [mp.Process(target=worker) for _ in range(workers)]
for proc in procs:
    proc.start()
for proc in procs:
    proc.join()
PY
          exit 0
        fi
        python3 - <<'PY'
from pathlib import Path
root = Path("/tmp/_cotail_kernel_build")
root.mkdir(parents=True, exist_ok=True)
for idx in range(64):
    path = root / f"unit_{{idx}}.c"
    if path.exists():
        continue
    lines = ["#include <stdint.h>\\n"]
    for fn in range(48):
        lines.append(f"uint64_t cotail_fn_{{idx}}_{{fn}}(uint64_t x) {{\\n")
        lines.append("  for (int i = 0; i < 2048; ++i) {{ x = (x * 2862933555777941757ULL + (uint64_t)i) ^ (x >> 17); }}\\n")
        lines.append("  return x;\\n}}\\n")
    path.write_text("".join(lines))
PY
        cleanup() {{
          pkill -P $$ >/dev/null 2>&1 || true
        }}
        trap cleanup EXIT
        if ! gcc -O2 -pipe -c "$work/unit_0.c" -o "$work/verify.o"; then
          echo "gcc compile loop verification failed; using stress-ng/python fallback" >&2
          rm -f "$work/verify.o"
          if command -v stress-ng >/dev/null; then
            exec stress-ng --vm {max(1, worker_count // 2)} --vm-bytes 128M --vm-method all --timeout {duration}s
          fi
          COTAIL_KERNEL_WORKERS="{worker_count}" COTAIL_DURATION="{duration}" python3 - <<'PY'
import hashlib
import multiprocessing as mp
import os
import time

workers = int(os.environ["COTAIL_KERNEL_WORKERS"])
deadline = time.time() + float(os.environ["COTAIL_DURATION"])

def worker() -> None:
    data = os.urandom(1024 * 1024)
    while time.time() < deadline:
        data = hashlib.sha256(data).digest() * 32768

procs = [mp.Process(target=worker) for _ in range(workers)]
for proc in procs:
    proc.start()
for proc in procs:
    proc.join()
PY
          exit 0
        fi
        rm -f "$work/verify.o"
        end=$((SECONDS + {duration}))
        for i in $(seq 0 {worker_count - 1}); do
          (
            n=$((i % 64))
            while [ "$SECONDS" -lt "$end" ]; do
              gcc -O2 -pipe -c "$work/unit_${{n}}.c" -o "$work/out_${{i}}.o" >/dev/null 2>&1 || true
              rm -f "$work/out_${{i}}.o"
            done
          ) &
        done
        wait
        """
        return WorkloadCommand(
            w,
            prefix + _bash(script),
            ["taskset", "bash", "python3", "gcc"],
            "self-contained gcc compile loop; falls back to stress-ng vm only if gcc is missing",
        )

    if w == "memcached":
        bench_workers = max(8, min(worker_count, 96))
        script = f"""
        set -e
        command -v memcached >/dev/null
        port=$((21212 + $$ % 1000))
        cleanup() {{
          if [ -n "${{server_pid:-}}" ]; then kill "$server_pid" >/dev/null 2>&1 || true; fi
          pkill -P $$ >/dev/null 2>&1 || true
        }}
        trap cleanup EXIT
        memcached -t {max(4, min(worker_count, 64))} -m 1024 -p "$port" -u nobody -l 127.0.0.1 >/dev/null 2>&1 &
        server_pid=$!
        sleep 2
        COTAIL_MEMCACHED_PORT="$port" COTAIL_MEMCACHED_WORKERS="{bench_workers}" COTAIL_DURATION="{duration}" python3 - <<'PY'
import multiprocessing as mp
import os
import random
import socket
import time

port = int(os.environ["COTAIL_MEMCACHED_PORT"])
workers = int(os.environ["COTAIL_MEMCACHED_WORKERS"])
deadline = time.time() + float(os.environ["COTAIL_DURATION"])

def worker(wid: int) -> None:
    rng = random.Random(os.getpid())
    value = b"x" * 512
    while time.time() < deadline:
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=2)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for i in range(20000):
                key = f"cotail_{{wid}}_{{rng.randint(0, 500000)}}".encode()
                sock.sendall(b"set " + key + b" 0 0 " + str(len(value)).encode() + b"\\r\\n" + value + b"\\r\\n")
                sock.recv(128)
                sock.sendall(b"get " + key + b"\\r\\n")
                sock.recv(2048)
                if time.time() >= deadline:
                    break
            sock.close()
        except Exception:
            time.sleep(0.05)

procs = [mp.Process(target=worker, args=(i,)) for i in range(workers)]
for proc in procs:
    proc.start()
for proc in procs:
    proc.join()
PY
        """
        return WorkloadCommand(w, prefix + _bash(script), ["taskset", "bash", "python3", "memcached"], "memcached server plus Python TCP clients")

    if w == "nginx":
        n_cores = worker_count
        n_nginx_workers = max(4, n_cores // 2)
        n_wrk = max(2, min(8, n_cores // 8))
        threads_per_wrk = max(2, n_cores // (n_wrk * 2))
        conns_per_wrk = 500
        python_clients = max(8, min(worker_count, 96))
        script = f"""
        set -e
        command -v nginx >/dev/null
        port=$((18088 + $$ % 1000))
        www=/tmp/_cotail_nginx_www
        conf=/tmp/_cotail_nginx_bench.conf
        pid=/tmp/_cotail_nginx_bench.pid
        mkdir -p "$www"
        python3 - <<'PY'
from pathlib import Path
root = Path('/tmp/_cotail_nginx_www')
root.mkdir(parents=True, exist_ok=True)
for name, size in [('file_1k.bin', 1024), ('file_16k.bin', 16384), ('file_256k.bin', 262144)]:
    path = root / name
    if not path.exists():
        path.write_bytes(b'x' * size)
index = root / 'index.html'
body = ''.join(f'<p>Benchmark paragraph {{i}}: Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet.</p>\\n' for i in range(20000))
index.write_text('<html><body>\\n' + body + '</body></html>\\n')
PY
        cat > "$conf" <<EOF
worker_processes {n_nginx_workers};
pid $pid;
error_log /dev/null crit;
events {{
    worker_connections 8192;
    multi_accept on;
}}
http {{
    access_log off;
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    gzip on;
    gzip_min_length 256;
    gzip_types text/html text/plain application/octet-stream;
    gzip_comp_level 6;
    server {{
        listen $port reuseport;
        root $www;
        location / {{ try_files $uri =404; }}
    }}
}}
EOF
        cleanup() {{
          nginx -c "$conf" -s quit >/dev/null 2>&1 || true
          pkill -P $$ >/dev/null 2>&1 || true
        }}
        trap cleanup EXIT
        if ! nginx -t -c "$conf"; then
          echo "nginx config test failed; using Python HTTP fallback" >&2
          COTAIL_NGINX_PORT="$port" COTAIL_NGINX_CLIENTS="{python_clients}" COTAIL_DURATION="{duration}" python3 - <<'PY'
import gzip
import multiprocessing as mp
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(os.environ["COTAIL_NGINX_PORT"])
clients = int(os.environ["COTAIL_NGINX_CLIENTS"])
deadline = time.time() + float(os.environ["COTAIL_DURATION"])
payload = ("<html><body>" + ("Benchmark paragraph. Lorem ipsum dolor sit amet. " * 12000) + "</body></html>").encode()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = gzip.compress(payload, compresslevel=6)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

def worker() -> None:
    url = f"http://127.0.0.1:{{port}}/index.html"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                resp.read(262144)
        except Exception:
            time.sleep(0.02)

procs = [mp.Process(target=worker) for _ in range(clients)]
for proc in procs:
    proc.start()
for proc in procs:
    proc.join()
server.shutdown()
PY
          exit 0
        fi
        nginx -c "$conf" -g 'daemon off;' &
        server_pid=$!
        sleep 2
        if ! kill -0 "$server_pid" >/dev/null 2>&1; then
          echo "nginx exited after startup; using Python HTTP fallback" >&2
          COTAIL_NGINX_PORT="$port" COTAIL_NGINX_CLIENTS="{python_clients}" COTAIL_DURATION="{duration}" python3 - <<'PY'
import gzip
import multiprocessing as mp
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(os.environ["COTAIL_NGINX_PORT"])
clients = int(os.environ["COTAIL_NGINX_CLIENTS"])
deadline = time.time() + float(os.environ["COTAIL_DURATION"])
payload = ("<html><body>" + ("Benchmark paragraph. Lorem ipsum dolor sit amet. " * 12000) + "</body></html>").encode()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = gzip.compress(payload, compresslevel=6)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

def worker() -> None:
    url = f"http://127.0.0.1:{{port}}/index.html"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                resp.read(262144)
        except Exception:
            time.sleep(0.02)

procs = [mp.Process(target=worker) for _ in range(clients)]
for proc in procs:
    proc.start()
for proc in procs:
    proc.join()
server.shutdown()
PY
          exit 0
        fi
        if command -v wrk >/dev/null; then
          for i in $(seq 1 {n_wrk}); do
            wrk --latency -t {threads_per_wrk} -c {conns_per_wrk} -d {duration}s http://127.0.0.1:$port/index.html >/dev/null 2>&1 &
          done
          wait
        elif command -v ab >/dev/null; then
          for i in $(seq 1 {n_wrk}); do
            (end=$((SECONDS + {duration})); while [ "$SECONDS" -lt "$end" ]; do ab -n 200000 -c 500 http://127.0.0.1:$port/index.html >/dev/null 2>&1 || true; done) &
          done
          wait
        else
          COTAIL_NGINX_PORT="$port" COTAIL_NGINX_CLIENTS="{python_clients}" COTAIL_DURATION="{duration}" python3 - <<'PY'
import multiprocessing as mp
import os
import time
import urllib.request

port = int(os.environ["COTAIL_NGINX_PORT"])
clients = int(os.environ["COTAIL_NGINX_CLIENTS"])
deadline = time.time() + float(os.environ["COTAIL_DURATION"])
url = f"http://127.0.0.1:{{port}}/index.html"

def worker() -> None:
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                resp.read(262144)
        except Exception:
            time.sleep(0.02)

procs = [mp.Process(target=worker) for _ in range(clients)]
for proc in procs:
    proc.start()
for proc in procs:
    proc.join()
PY
        fi
        """
        return WorkloadCommand(
            w,
            prefix + _bash(script),
            ["taskset", "bash", "python3", "nginx"],
            f"nginx plus wrk/ab/Python clients; {n_wrk} client groups, gzip=on",
        )

    if w == "zstd-compress":
        n_workers = max(1, min(worker_count, len(cpu_values)))
        cpu_words = " ".join(str(cpu_values[i % len(cpu_values)]) for i in range(n_workers))
        script = f"""
        set -e
        root=/tmp/cotail_heldout/compress
        input="$root/input.txt"
        mkdir -p "$root"
        python3 - <<'PY'
from pathlib import Path
import random
import string
path = Path("/tmp/cotail_heldout/compress/input.txt")
if path.exists() and path.stat().st_size >= 128 * 1024 * 1024:
    raise SystemExit(0)
rng = random.Random(42)
alphabet = string.ascii_letters + string.digits
with path.open("w") as f:
    for _ in range(1_200_000):
        f.write("CoTail compression workload line ")
        f.write("".join(rng.choices(alphabet, k=96)))
        f.write("\\n")
PY
        if command -v zstd >/dev/null; then
          tool=zstd
        elif command -v xz >/dev/null; then
          tool=xz
        elif command -v gzip >/dev/null; then
          tool=gzip
        else
          tool=openssl
        fi
        cleanup() {{ pkill -P $$ >/dev/null 2>&1 || true; }}
        trap cleanup EXIT
        end=$((SECONDS + {duration}))
        worker_id=0
        for cpu in {cpu_words}; do
          (
            while [ "$SECONDS" -lt "$end" ]; do
              if [ "$tool" = zstd ]; then
                taskset -c "$cpu" zstd -T1 -f "$input" -o "$root/out_${{worker_id}}.zst" >/dev/null 2>&1 || true
                rm -f "$root/out_${{worker_id}}.zst"
              elif [ "$tool" = xz ]; then
                taskset -c "$cpu" xz -T1 -kf -c "$input" >"$root/out_${{worker_id}}.xz" 2>/dev/null || true
                rm -f "$root/out_${{worker_id}}.xz"
              elif [ "$tool" = gzip ]; then
                taskset -c "$cpu" gzip -kf -c "$input" >"$root/out_${{worker_id}}.gz" 2>/dev/null || true
                rm -f "$root/out_${{worker_id}}.gz"
              else
                taskset -c "$cpu" openssl dgst -sha256 "$input" >/dev/null 2>&1 || true
              fi
            done
          ) &
          worker_id=$((worker_id + 1))
        done
        wait
        """
        return WorkloadCommand(w, _bash(script), ["taskset", "bash", "python3"], "script-compatible compression loop: one worker per core")

    if w == "sqlite-txn":
        sqlite_workers = max(32, min(worker_count, 64))
        script = f"""
        set -e
        COTAIL_SQLITE_WORKERS="{sqlite_workers}" COTAIL_DURATION="{duration}" python3 - <<'PY'
import multiprocessing as mp
import os
import random
import sqlite3
import time
from pathlib import Path

root = Path("/tmp/cotail_heldout/sqlite")
root.mkdir(parents=True, exist_ok=True)
workers = int(os.environ["COTAIL_SQLITE_WORKERS"])
deadline = time.time() + float(os.environ["COTAIL_DURATION"])

def worker(worker_id: int) -> None:
    rng = random.Random(os.getpid())
    conn = sqlite3.connect(str(root / f"worker_{{worker_id}}.db"), timeout=30)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("CREATE TABLE IF NOT EXISTS kv (k INTEGER PRIMARY KEY, v TEXT);")
    conn.commit()
    while time.time() < deadline:
        cur.execute("BEGIN;")
        for _ in range(200):
            key = rng.randint(0, 200000)
            val = f"value-{{worker_id}}-{{rng.random()}}"
            cur.execute("INSERT OR REPLACE INTO kv(k, v) VALUES (?, ?);", (key, val))
            if key % 7 == 0:
                cur.execute("SELECT v FROM kv WHERE k=?;", (key,))
                cur.fetchone()
        conn.commit()

procs = [mp.Process(target=worker, args=(i,)) for i in range(workers)]
for proc in procs:
    proc.start()
for proc in procs:
    proc.join()
PY
        """
        return WorkloadCommand(w, prefix + _bash(script), ["taskset", "bash", "python3"], "script-compatible sqlite transaction workers")

    if w == "image-preprocess":
        image_workers = max(32, min(worker_count, 64))
        script = f"""
        set -e
        COTAIL_IMG_WORKERS="{image_workers}" COTAIL_DURATION="{duration}" python3 - <<'PY'
import hashlib
import multiprocessing as mp
import os
import time
from pathlib import Path

workers = int(os.environ["COTAIL_IMG_WORKERS"])
deadline = time.time() + float(os.environ["COTAIL_DURATION"])

def pillow_worker(worker_id: int) -> None:
    import random
    from PIL import Image, ImageFilter
    import numpy as np
    img_dir = Path("/tmp/cotail_heldout/images")
    out_dir = Path("/tmp/cotail_heldout/images_out")
    img_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(list(img_dir.glob("*.jpg"))) < 1000:
        rng = np.random.default_rng(42)
        for i in range(2000):
            path = img_dir / f"img_{{i:05d}}.jpg"
            if path.exists():
                continue
            arr = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
            Image.fromarray(arr).save(path, quality=90)
    images = list(img_dir.glob("*.jpg"))
    rng = random.Random(worker_id)
    while time.time() < deadline:
        path = rng.choice(images)
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize((224, 224))
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            if rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(out_dir / f"worker_{{worker_id}}.jpg", quality=85)

def numpy_worker() -> None:
    import numpy as np
    rng = np.random.default_rng(os.getpid())
    while time.time() < deadline:
        arr = rng.integers(0, 255, (1024, 1024, 3), dtype=np.uint8).astype(np.float32)
        arr = arr[32:992, 32:992, :]
        arr = (arr[:-2, :-2] + arr[1:-1, :-2] + arr[2:, :-2] + arr[:-2, 1:-1] + arr[1:-1, 1:-1] + arr[2:, 1:-1] + arr[:-2, 2:] + arr[1:-1, 2:] + arr[2:, 2:]) / 9.0
        float(arr.mean())

def hash_worker() -> None:
    data = os.urandom(1024 * 1024)
    while time.time() < deadline:
        data = hashlib.sha256(data).digest() * 32768

try:
    import PIL  # noqa: F401
    import numpy  # noqa: F401
    target = pillow_worker
except Exception:
    try:
        import numpy  # noqa: F401
        target = numpy_worker
    except Exception:
        target = hash_worker

procs = [mp.Process(target=target) for _ in range(workers)]
for proc in procs:
    proc.start()
for proc in procs:
    proc.join()
PY
        """
        return WorkloadCommand(w, prefix + _bash(script), ["taskset", "bash", "python3"], "script-compatible Pillow image preprocess with NumPy/hash fallback")

    if w == "text-search":
        n_workers = max(1, min(worker_count, len(cpu_values)))
        cpu_words = " ".join(str(cpu_values[i % len(cpu_values)]) for i in range(n_workers))
        script = f"""
        set -e
        root=/tmp/cotail_heldout/text
        mkdir -p "$root"
        python3 - <<'PY'
from pathlib import Path
import random
import string
root = Path("/tmp/cotail_heldout/text")
root.mkdir(parents=True, exist_ok=True)
if len(list(root.glob("log_*.txt"))) >= 1000:
    raise SystemExit(0)
rng = random.Random(42)
keywords = ["error", "warning", "timeout", "cuda", "kernel", "request", "token"]
alphabet = string.ascii_lowercase
for i in range(2000):
    path = root / f"log_{{i:05d}}.txt"
    with path.open("w") as f:
        for _ in range(2000):
            words = ["".join(rng.choices(alphabet, k=8)) for _ in range(20)]
            if rng.random() < 0.1:
                words.append(rng.choice(keywords))
            f.write(" ".join(words) + "\\n")
PY
        if command -v rg >/dev/null; then
          search_cmd='rg -n "cuda|kernel|timeout|request|token|error" /tmp/cotail_heldout/text >/tmp/cotail_heldout/rg.$COTAIL_WORKER_ID.out'
        else
          search_cmd='grep -R -n -E "cuda|kernel|timeout|request|token|error" /tmp/cotail_heldout/text >/tmp/cotail_heldout/grep.$COTAIL_WORKER_ID.out'
        fi
        cleanup() {{ pkill -P $$ >/dev/null 2>&1 || true; }}
        trap cleanup EXIT
        end=$((SECONDS + {duration}))
        worker_id=0
        for cpu in {cpu_words}; do
          (
            export COTAIL_WORKER_ID="$worker_id"
            while [ "$SECONDS" -lt "$end" ]; do
              taskset -c "$cpu" bash -lc "$search_cmd" || true
            done
          ) &
          worker_id=$((worker_id + 1))
        done
        wait
        """
        return WorkloadCommand(w, _bash(script), ["taskset", "bash", "python3"], "script-compatible text-search: 2000x2000 corpus, one rg/grep loop per core")

    return WorkloadCommand(
        w,
        prefix + _bash(f"echo unknown workload {shlex.quote(w)}; sleep {duration}"),
        ["taskset", "bash"],
        "unknown workload placeholder",
    )
