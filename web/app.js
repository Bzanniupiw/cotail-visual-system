const state = {
  runtime: null,
  ws: null,
  current: null,
  measurements: { baseline: null, colocation: null, protected: null },
  stages: { baseline: null, current: null, protected: null },
  discovery: null,
  busyThread: null,
  selectedPolicy: "rt",
  workloadJobId: null,
  workloadHealth: null,
  workloadProbe: null,
};

const qs = (sel) => document.querySelector(sel);
const qsa = (sel) => Array.from(document.querySelectorAll(sel));

const LONG_PROMPT_BASE =
  "Transformer中的Self-Attention机制通过查询向量Q、键向量K和值向量V之间的相似度计算，实现对不同位置token的上下文建模。" +
  "在预填充阶段，模型需要一次性处理完整输入序列并建立KV Cache；在解码阶段，模型通常逐token生成，每一步会复用历史缓存。" +
  "Self-Attention的核心操作包括线性映射、注意力分数计算、缩放、Softmax归一化和加权求和。" +
  "对于长度为n、隐藏维度为d的序列，标准全注意力的时间复杂度通常为O(n^2·d)，空间复杂度也会随注意力矩阵规模增长。" +
  "在大语言模型推理系统中，请求排队、tokenization、prefill调度、batch合并、streaming返回以及引擎线程调度都会对端到端时延产生影响。";

const LONG_PROMPT_TAIL =
  "请基于以上技术材料，系统解释 Transformer 中 Self-Attention 的计算流程、Q/K/V 作用、缩放点积注意力、KV Cache、Prefill 与 Decode 的差异、时间复杂度与空间复杂度，并说明在长上下文场景下的瓶颈来源。";

const SCRIPT_COMPAT_WARMUP_MS = 8000;

async function api(path, options = {}) {
  const headers = options.body instanceof FormData ? options.headers || {} : { "Content-Type": "application/json", ...(options.headers || {}) };
  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function num(selector, fallback = 0) {
  const value = Number(qs(selector).value);
  return Number.isFinite(value) ? value : fallback;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function showJson(el, data) {
  el.textContent = JSON.stringify(data, null, 2);
}

function badge(text, cls = "") {
  return `<span class="badge ${cls}">${escapeHtml(text)}</span>`;
}

function setStatus(text, tone = "") {
  const el = qs("#global-status");
  el.className = `status-pill ${tone}`.trim();
  el.textContent = text;
}

function setNotice(selector, text, tone = "") {
  const el = qs(selector);
  el.className = `notice ${tone}`.trim();
  el.setAttribute("role", tone === "bad" ? "alert" : "status");
  el.textContent = text;
}

async function withButton(selector, text, fn) {
  const btn = qs(selector);
  const old = btn.textContent;
  btn.disabled = true;
  btn.classList.add("is-loading");
  btn.textContent = text;
  try {
    return await fn();
  } finally {
    btn.classList.remove("is-loading");
    btn.disabled = false;
    btn.textContent = old;
  }
}

function parseCsvInts(value) {
  return String(value || "")
    .split(",")
    .map((x) => Number(x.trim()))
    .filter((x) => Number.isFinite(x));
}

function parseCsvPositiveInts(value) {
  return parseCsvInts(value).filter((x) => Number.isInteger(x) && x > 0);
}

function formatCpuRange(values) {
  const sorted = [...new Set((values || []).map(Number).filter(Number.isFinite))].sort((a, b) => a - b);
  const out = [];
  let start = null;
  let prev = null;
  for (const value of sorted) {
    if (start === null) {
      start = value;
      prev = value;
    } else if (value === prev + 1) {
      prev = value;
    } else {
      out.push(start === prev ? String(start) : `${start}-${prev}`);
      start = value;
      prev = value;
    }
  }
  if (start !== null) out.push(start === prev ? String(start) : `${start}-${prev}`);
  return out.join(",");
}

function buildLongPrompt() {
  const repeat = Math.max(1, num("#measure-repeat-times", 30));
  const prompt =
    "下面给出一段关于大语言模型推理系统和Transformer机制的技术材料，请完整阅读并回答最后的问题。\n\n" +
    LONG_PROMPT_BASE.repeat(repeat) +
    "\n\n" +
    LONG_PROMPT_TAIL;
  qs("#measure-prompt").value = prompt;
  const approxTokens = Math.max(1, Math.round(prompt.length / 1.8));
  qs("#prompt-size").textContent = `${prompt.length} chars, approx ${approxTokens} tokens`;
}

function setView(name) {
  qsa(".nav-item").forEach((btn) => btn.classList.toggle("active", btn.dataset.view === name));
  qsa(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  const titles = {
    monitor: ["实时监控", "共享 8×4090 服务器上的 GPU、vLLM 服务、请求压测和托管作业。"],
    diagnosis: ["自动诊断", "自动采集 Baseline、Co-location 与 Protected 的宏观 canary 指标。"],
    micro: ["微观 CPTI", "按需计算核心服务阶段的 P95/P99 尾部放大，定位主导瓶颈。"],
    workflow: ["策略流", "Algorithm 1 状态机、CTS 验证盘和人工策略覆盖控制台。"],
    tenants: ["租户成本", "共置 CPU 负载画像、慢化估计和诊断开销墙。"],
  };
  qs("#page-title").textContent = titles[name][0];
  qs("#page-subtitle").textContent = titles[name][1];
}

function renderRuntime(data) {
  state.runtime = data;
  const sys = data.system || {};
  const gpuRoot = data.gpu || {};
  const gpus = gpuRoot.gpus || [];
  const selection = data.selection || {};
  const jobs = data.jobs || [];

  qs("#rt-host").textContent = sys.hostname || "-";
  qs("#rt-cpu").textContent = sys.cpu_percent == null ? `${sys.cpu_count || "-"} cores` : `${fmt(sys.cpu_percent, 1)}%`;
  qs("#rt-mem").textContent = sys.memory_percent == null ? "-" : `${fmt(sys.memory_percent, 1)}%`;
  qs("#rt-selected").textContent = selection.selected_gpu == null ? "None" : `GPU ${selection.selected_gpu}`;
  qs("#rt-jobs").textContent = String(jobs.length);
  qs("#rt-updated").textContent = gpuRoot.ok
    ? `updated ${new Date((gpuRoot.timestamp || Date.now() / 1000) * 1000).toLocaleTimeString()}`
    : `nvidia-smi unavailable: ${gpuRoot.error || "unknown"}`;
  setStatus(gpuRoot.ok ? "runtime live" : "runtime degraded", gpuRoot.ok ? "good" : "bad");

  const grid = qs("#gpu-grid");
  if (!gpus.length) {
    grid.innerHTML = `<div class="gpu-card">${escapeHtml(gpuRoot.error || "No GPU data. Run this on the Linux 8×4090 server with nvidia-smi in PATH.")}</div>`;
  } else {
    grid.innerHTML = gpus
      .map((gpu) => {
        const util = Number(gpu.utilization_gpu_pct || 0);
        const memPct = Number(gpu.memory_used_pct || 0);
        const idle = gpu.is_likely_idle || selection.selected_gpu === gpu.index;
        const statusCls = idle ? "good" : util > 80 || gpu.compute_process_count > 0 ? "bad" : "warn";
        const statusText = idle ? "IDLE" : gpu.compute_process_count > 0 ? "BUSY" : "WARM";
        return `
          <article class="gpu-card ${idle ? "idle" : "busy"}">
            <div class="gpu-title"><strong>GPU ${gpu.index}</strong>${badge(statusText, statusCls)}</div>
            <div class="muted">${escapeHtml(gpu.name || "")}</div>
            <div class="gpu-kv">
              <div><span>Util</span><strong>${util}%</strong></div>
              <div><span>Temp</span><strong>${gpu.temperature_c ?? "-"}C</strong></div>
              <div><span>Memory</span><strong>${gpu.memory_used_mib}/${gpu.memory_total_mib} MiB</strong></div>
              <div><span>Free</span><strong>${gpu.memory_free_mib} MiB</strong></div>
              <div><span>Power</span><strong>${fmt(gpu.power_draw_w, 0)}/${fmt(gpu.power_limit_w, 0)} W</strong></div>
              <div><span>Proc</span><strong>${gpu.compute_process_count} compute</strong></div>
            </div>
            <div class="mini-bar"><i class="${util > 80 ? "hot" : util > 30 ? "mid" : ""}" style="width:${Math.min(100, util)}%"></i></div>
            <div class="mini-bar"><i class="${memPct > 80 ? "hot" : memPct > 40 ? "mid" : ""}" style="width:${Math.min(100, memPct)}%"></i></div>
            <div class="muted">${escapeHtml(gpu.bus_id || "")} ${escapeHtml(gpu.pstate || "")}</div>
          </article>
        `;
      })
      .join("");
  }

  const processes = gpus.flatMap((gpu) => (gpu.processes || []).map((p) => ({ ...p, gpu_index: gpu.index })));
  qs("#gpu-process-table").innerHTML = processes.length
    ? `
      <table>
        <thead><tr><th>GPU</th><th>PID</th><th>Type</th><th>User</th><th>Memory</th><th>Process</th></tr></thead>
        <tbody>${processes
          .map((p) => `<tr><td>${p.gpu_index}</td><td>${p.pid}</td><td>${escapeHtml(p.process_type)}</td><td>${escapeHtml(p.username || "-")}</td><td>${p.used_memory_mib ?? "-"} MiB</td><td>${escapeHtml(p.process_name)}</td></tr>`)
          .join("")}</tbody>
      </table>`
    : "No GPU processes.";
  renderJobs(jobs);
}

function renderJobs(jobs) {
  qs("#jobs-table").innerHTML = jobs.length
    ? `
      <table>
        <thead><tr><th>ID</th><th>GPU</th><th>Backend</th><th>Port</th><th>API</th><th>Dry-run</th><th>Processes</th><th>Action</th></tr></thead>
        <tbody>${jobs
          .map((job) => {
            const procText = (job.processes || []).map((p) => `${escapeHtml(p.role)}:${p.pid || "-"}:${escapeHtml(p.status)}`).join("<br>");
            const apiState = job.api_port_open ? badge("OPEN", "good") : badge("CLOSED", "bad");
            const useButton = job.backend === "cpu-workload" ? "" : `<button data-use-job="${job.id}">Use</button>`;
            const logRole = job.backend === "cpu-workload" ? "cpu_workload" : "llm_server";
            return `<tr><td>${escapeHtml(job.id)}</td><td>${job.gpu_index}</td><td>${escapeHtml(job.backend)}</td><td>${job.port || "-"}</td><td>${job.backend === "cpu-workload" ? "-" : apiState}</td><td>${job.dry_run}</td><td>${procText}</td><td>${useButton} <button data-logs="${job.id}" data-log-role="${logRole}">Logs</button> <button data-stop="${job.id}">Stop</button></td></tr>`;
          })
          .join("")}</tbody>
      </table>`
    : "No managed jobs.";
  qs("#jobs-table").querySelectorAll("button[data-use-job]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const job = jobs.find((item) => item.id === btn.dataset.useJob);
      if (!job) return;
      const apiBase = job.api_base || `http://127.0.0.1:${job.port}/v1`;
      qs("#llm-api").value = apiBase;
      qs("#llm-model").value = job.model || qs("#llm-model").value;
      qs("#measure-api").value = apiBase;
      qs("#measure-model").value = job.model || qs("#measure-model").value;
      qs("#override-ports").value = String(job.port);
    });
  });
  qs("#jobs-table").querySelectorAll("button[data-logs]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const role = btn.dataset.logRole || "llm_server";
      const data = await api(`/api/jobs/${btn.dataset.logs}/logs?role=${encodeURIComponent(role)}`);
      showJson(qs("#job-log-output"), data);
    });
  });
  qs("#jobs-table").querySelectorAll("button[data-stop]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await api(`/api/jobs/${btn.dataset.stop}/stop`, { method: "POST", body: "{}" });
      await refreshRuntime();
    });
  });
}

async function refreshRuntime() {
  const data = await api("/api/runtime/snapshot");
  renderRuntime(data);
}

function connectRuntimeStream() {
  if (state.ws) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/runtime/ws?interval_s=2`);
  state.ws = ws;
  ws.onmessage = (event) => {
    try {
      renderRuntime(JSON.parse(event.data));
    } catch (err) {
      console.error(err);
    }
  };
  ws.onclose = () => {
    state.ws = null;
    setStatus("runtime reconnecting", "warn");
    setTimeout(() => {
      refreshRuntime().catch(console.error);
      connectRuntimeStream();
    }, 2000);
  };
}

function llmPayload() {
  return {
    api_base: qs("#llm-api").value,
    model: qs("#llm-model").value,
    prompt: qs("#llm-prompt").value,
    max_tokens: num("#llm-max-tokens", 512),
    temperature: num("#llm-temperature", 0.7),
    stream: true,
    timeout_s: 120,
  };
}

function renderRequestMetrics(data) {
  if (data.success_count !== undefined) {
    qs("#req-success").textContent = `${data.success_count}/${data.total_requests}`;
    qs("#req-ttft").textContent = data.ttft_avg_ms == null ? "-" : `${fmt(data.ttft_avg_ms, 1)} ms`;
    qs("#req-tpot").textContent = data.tpot_avg_ms == null ? "-" : `${fmt(data.tpot_avg_ms, 2)} ms`;
    qs("#req-throughput").textContent = `${fmt(data.throughput_tok_s, 1)} tok/s`;
    qs("#req-latency").textContent = data.request_total_avg_ms == null ? "-" : `${fmt(data.request_total_avg_ms, 1)} ms`;
  } else {
    qs("#req-success").textContent = data.ok ? "1/1" : "0/1";
    qs("#req-ttft").textContent = data.ttft_ms == null ? "-" : `${fmt(data.ttft_ms, 1)} ms`;
    qs("#req-tpot").textContent = data.tpot_ms == null ? "-" : `${fmt(data.tpot_ms, 2)} ms`;
    qs("#req-throughput").textContent = data.throughput_tok_s == null ? "-" : `${fmt(data.throughput_tok_s, 1)} tok/s`;
    qs("#req-latency").textContent = data.total_ms == null ? "-" : `${fmt(data.total_ms, 1)} ms`;
  }
}

async function llmHealth() {
  await withButton("#llm-health-btn", "检查中", async () => {
    const data = await api("/api/llm/models", { method: "POST", body: JSON.stringify(llmPayload()) });
    showJson(qs("#llm-output"), data);
  });
}

async function llmRequest() {
  await withButton("#llm-request-btn", "请求中", async () => {
    const data = await api("/api/llm/request", { method: "POST", body: JSON.stringify(llmPayload()) });
    renderRequestMetrics(data);
    showJson(qs("#llm-output"), data);
  });
}

async function llmBenchmark() {
  await withButton("#llm-bench-btn", "压测中", async () => {
    const payload = {
      ...llmPayload(),
      concurrency: num("#llm-concurrency", 32),
      total_requests: num("#llm-requests", 32),
    };
    const data = await api("/api/llm/benchmark", { method: "POST", body: JSON.stringify(payload) });
    renderRequestMetrics(data);
    showJson(qs("#llm-output"), data);
  });
}

async function launchJob() {
  await withButton("#launch-job-btn", "处理中", async () => {
    const manualGpuText = qs("#launch-gpu").value.trim();
    const manualGpu = manualGpuText === "" ? null : Number(manualGpuText);
    const autoSelect = qs("#launch-auto-gpu").checked || manualGpu === null || !Number.isFinite(manualGpu);
    const payload = {
      model: qs("#launch-model").value,
      backend: qs("#launch-backend").value,
      auto_select_gpu: autoSelect,
      gpu_index: autoSelect ? null : manualGpu,
      port: num("#launch-port", 8102),
      gpu_memory_utilization: num("#launch-gmem", 0.7),
      max_model_len: num("#launch-maxlen", 8192),
      enforce_eager: qs("#launch-eager").checked,
      cpu_workload: qs("#launch-workload").value,
      cpu_cores: qs("#launch-cpus").value,
      workers: num("#launch-workers", 60),
      dry_run: qs("#launch-dryrun").checked,
      serving_extra_args: qs("#launch-extra").value,
      gpu_policy: {
        min_free_memory_mib: num("#launch-minmem", 18000),
        max_gpu_util_pct: num("#launch-maxutil", 10),
        max_compute_processes: 0,
        allow_graphics_processes: true,
        exclude_gpu_ids: [],
      },
    };
    const data = await api("/api/jobs/launch", { method: "POST", body: JSON.stringify(payload) });
    showJson(qs("#launch-output"), data);
    setNotice("#launch-status", data.message || "launch request submitted", payload.dry_run ? "" : "good");
    if (data.job) {
      const apiBase = `http://127.0.0.1:${data.job.port}/v1`;
      qs("#llm-api").value = apiBase;
      qs("#measure-api").value = apiBase;
      qs("#measure-model").value = data.job.model;
      qs("#override-ports").value = String(data.job.port);
      qs("#measure-protection-ports").value = String(data.job.port);
    }
    await refreshRuntime();
  });
}

function measurePayload(phase) {
  return {
    phase,
    workload: qs("#measure-workload").value,
    api_base: qs("#measure-api").value,
    model: qs("#measure-model").value,
    prompt: qs("#measure-prompt").value,
    max_tokens: num("#measure-max-tokens", 512),
    temperature: 0.7,
    stream: true,
    concurrency: num("#measure-concurrency", 32),
    total_requests: num("#measure-requests", 32),
    run_benchmark: qs("#measure-run-benchmark").checked,
    sample_seconds: 2,
    timeout_s: 180,
  };
}

function workloadPayload(cpuCoreOverride = null, stressCgroupPath = "") {
  return {
    workload: qs("#measure-workload").value,
    cpu_cores: cpuCoreOverride || qs("#measure-workload-cores").value,
    workers: num("#measure-workload-workers", 60),
    duration_s: num("#measure-workload-duration", 3600),
    stress_cgroup_path: stressCgroupPath,
    dry_run: qs("#measure-workload-dryrun").checked,
  };
}

function healthBadge(verdict) {
  const tone = verdict === "strong" ? "good" : verdict === "weak" ? "warn" : "bad";
  return badge(verdict || "unknown", tone);
}

function renderWorkloadHealth(data) {
  state.workloadHealth = data;
  const el = qs("#workload-health-table");
  if (!el) return;
  const rows = data?.results
    ? data.results.map((item) => ({ workload: item.workload, ...(item.health || {}) }))
    : data
      ? [data]
      : [];
  el.innerHTML = rows.length
    ? `
      <table>
        <thead><tr><th>Workload</th><th>Verdict</th><th>CPU</th><th>Proc CPU</th><th>CTX</th><th>I/O</th><th>Processes</th><th>Message</th></tr></thead>
        <tbody>${rows
          .map((row) => {
            const sig = row.signals || {};
            const io = Number(sig.disk_MBps || 0) + Number(sig.net_MBps || 0);
            return `<tr>
              <td>${escapeHtml(row.workload)}</td>
              <td>${healthBadge(row.verdict)}</td>
              <td>${fmt(sig.system_cpu_percent, 1)}%</td>
              <td>${fmt(sig.process_cpu_percent_sum, 1)}%</td>
              <td>${fmt(sig.ctx_switches_per_s, 0)}/s</td>
              <td>${fmt(io, 1)} MB/s</td>
              <td>${sig.process_count ?? 0}</td>
              <td>${escapeHtml(row.message || (row.hints || []).join(" "))}</td>
            </tr>`;
          })
          .join("")}</tbody>
      </table>`
    : "No workload health data.";
  if (data?.verdict) {
    const tone = data.active ? (data.verdict === "strong" ? "good" : "") : "bad";
    setNotice("#workload-status", `${data.workload}: ${data.verdict} - ${data.message || ""}`, tone);
  } else if (data?.results) {
    const active = data.active_count || 0;
    setNotice("#workload-status", `Probe finished: ${active}/${data.count} workloads produced visible pressure.`, active === data.count ? "good" : "");
  }
}

async function checkCurrentWorkload(showNotice = true) {
  const workload = qs("#measure-workload").value;
  const params = new URLSearchParams({ workload, sample_seconds: "1.5" });
  if (state.workloadJobId) params.set("job_id", state.workloadJobId);
  if (showNotice) setNotice("#workload-status", `Checking ${workload} pressure...`, "");
  const data = await api(`/api/workloads/health?${params.toString()}`);
  renderWorkloadHealth(data);
  return data;
}

async function probeAllWorkloads() {
  await withButton("#probe-all-workloads-btn", "Probing...", async () => {
    const payload = {
      workloads: [],
      cpu_cores: qs("#measure-workload-cores").value,
      workers: num("#measure-workload-workers", 60),
      duration_s: 18,
      sample_seconds: 2,
      startup_wait_s: 8,
      dry_run: qs("#probe-all-dryrun").checked,
    };
    const data = await api("/api/workloads/probe-all", { method: "POST", body: JSON.stringify(payload) });
    state.workloadProbe = data;
    renderWorkloadHealth(data);
    showJson(qs("#workload-health-output"), data);
    await refreshRuntime();
  });
}

async function startSelectedWorkload() {
  await withButton("#start-workload-btn", "Starting...", async () => {
    const data = await launchColocationWorkload();
    showJson(qs("#workload-health-output"), data);
  });
}

async function launchColocationWorkload(cpuCoreOverride = null, stressCgroupPath = "") {
  const payload = workloadPayload(cpuCoreOverride, stressCgroupPath);
  const data = await api("/api/workloads/launch", { method: "POST", body: JSON.stringify(payload) });
  state.workloadJobId = data.job?.id || null;
  setNotice(
    "#workload-status",
    `${data.message || "workload launch submitted"}${state.workloadJobId ? ` job=${state.workloadJobId}` : ""}`,
    payload.dry_run ? "" : "good"
  );
  await refreshRuntime();
  if (!payload.dry_run && state.workloadJobId) {
    await sleep(SCRIPT_COMPAT_WARMUP_MS);
    await checkCurrentWorkload(false);
  }
  return data;
}

async function stopColocationWorkload() {
  if (!state.workloadJobId) {
    setNotice("#workload-status", "当前没有由自动诊断启动的负载 job。", "bad");
    return;
  }
  const data = await api(`/api/jobs/${state.workloadJobId}/stop`, { method: "POST", body: "{}" });
  setNotice("#workload-status", `已请求停止负载 job ${state.workloadJobId}。`, data.ok ? "good" : "bad");
  state.workloadJobId = null;
  await refreshRuntime();
}

async function stopAllManagedWorkloads() {
  await withButton("#stop-managed-workloads-btn", "Stopping...", async () => {
    const data = await api("/api/workloads/stop-managed", { method: "POST", body: "{}" });
    state.workloadJobId = null;
    setNotice("#workload-status", `已请求停止 ${data.count || 0} 个托管 CPU 负载。`, data.ok ? "good" : "bad");
    showJson(qs("#workload-health-output"), data);
    await refreshRuntime();
  });
}

function renderOrphanWorkloads(data) {
  const rows = data?.processes || [];
  const el = qs("#workload-health-table");
  if (!rows.length) {
    el.innerHTML = "没有发现 CoTail 指纹匹配的遗留 CPU 负载。";
    return;
  }
  el.innerHTML = `
    <table>
      <thead><tr><th>PID</th><th>PPID</th><th>Workload</th><th>User</th><th>Name</th><th>Reason</th><th>Command</th></tr></thead>
      <tbody>${rows
        .map((row) => `<tr>
          <td>${row.pid}</td>
          <td>${row.ppid ?? "-"}</td>
          <td>${escapeHtml(row.workload)}</td>
          <td>${escapeHtml(row.username)}</td>
          <td>${escapeHtml(row.name)}</td>
          <td>${escapeHtml(row.reason)}</td>
          <td>${escapeHtml(row.cmdline)}</td>
        </tr>`)
        .join("")}</tbody>
    </table>
  `;
}

async function scanOrphanWorkloads() {
  await withButton("#scan-orphan-workloads-btn", "Scanning...", async () => {
    const data = await api("/api/workloads/orphans?current_user_only=true");
    renderOrphanWorkloads(data);
    setNotice("#workload-status", `发现 ${data.count || 0} 个遗留 CoTail CPU 负载进程。`, data.count ? "bad" : "good");
    showJson(qs("#workload-health-output"), data);
  });
}

async function cleanupOrphanWorkloads() {
  await withButton("#cleanup-orphan-workloads-btn", "Cleaning...", async () => {
    const data = await api("/api/workloads/cleanup", {
      method: "POST",
      body: JSON.stringify({ dry_run: false, current_user_only: true, force: true, workloads: [] }),
    });
    const remaining = {
      ok: !data.remaining_count,
      count: data.remaining_count || 0,
      processes: data.remaining_processes || [],
    };
    renderOrphanWorkloads(remaining);
    setNotice("#workload-status", `已清理 ${data.root_count || 0} 个遗留负载根进程，匹配进程 ${data.count || 0} 个。`, data.ok ? "good" : "bad");
    showJson(qs("#workload-health-output"), data);
    setNotice("#workload-status", `Cleanup finished: roots=${data.root_count || 0}, matched_before=${data.count || 0}, managed_removed=${data.managed_removed || 0}, remaining=${data.remaining_count || 0}.`, data.remaining_count ? "bad" : "good");
    await refreshRuntime();
  });
}

function protectionPayload(execute) {
  return {
    policy: qs("#measure-protection-policy").value,
    vllm_pids: parseCsvPositiveInts(qs("#measure-protection-pids").value),
    engine_tid: Number(qs("#measure-protection-tid").value.trim()) || null,
    battle_cores: qs("#measure-protection-battle").value,
    numa_vllm_cpus: qs("#measure-protection-numa-vllm").value,
    numa_interference_cpus: qs("#measure-protection-numa-intf").value,
    rt_priority: num("#measure-protection-rt-priority", 50),
    execute,
  };
}

async function ensureServiceDiscovery(needTid = false) {
  const pids = parseCsvPositiveInts(qs("#measure-protection-pids").value);
  if (!pids.length) {
    await scanService("#measure-scan-service", "#measure-protection-ports");
  }
  if (needTid && !Number(qs("#measure-protection-tid").value.trim())) {
    await identifyTid("#measure-identify-tid", "#measure-scan-service", "#measure-protection-ports");
  }
}

async function applyControlledServicePlacement(phase) {
  const enabled = qs("#measure-control-placement")?.checked;
  if (!enabled) return null;
  await ensureServiceDiscovery(false);
  const pids = parseCsvPositiveInts(qs("#measure-protection-pids").value);
  qs("#measure-protection-pids").value = pids.join(",");
  if (!pids.length) {
    throw new Error("Cannot fix vLLM CPU placement: no valid vLLM PID was discovered. Click 扫描服务, or uncheck Fix vLLM CPU placement.");
  }
  const policy = phase === "protected" ? qs("#measure-protection-policy").value : "none";
  const serviceCores = policy.includes("numa")
    ? qs("#measure-protection-numa-vllm").value.trim()
    : qs("#measure-protection-battle").value.trim();
  if (!serviceCores) {
    throw new Error("Cannot fix vLLM CPU placement: script-compatible CPU range is empty.");
  }
  setNotice("#diagnosis-status", `${phase}: fixing vLLM CPU placement to ${serviceCores}.`, "");
  const data = await api("/api/protection/plan", {
    method: "POST",
    body: JSON.stringify({
      policy: policy.includes("numa") ? "numa" : "none",
      vllm_pids: pids,
      engine_tid: null,
      battle_cores: policy.includes("numa") ? "" : serviceCores,
      numa_vllm_cpus: policy.includes("numa") ? serviceCores : "",
      numa_interference_cpus: "",
      rt_priority: num("#measure-protection-rt-priority", 50),
      execute: true,
    }),
  });
  if (!data.execution?.ok) {
    const failed = (data.execution?.results || []).filter((item) => !item.ok);
    const reason = failed[0]?.error || failed[0]?.stderr || data.plan?.warnings?.[0] || "unknown placement failure";
    throw new Error(`vLLM CPU placement failed: ${reason}`);
  }
  data.service_cores = serviceCores;
  return data;
}

async function applyMeasureProtection() {
  const execute = qs("#measure-apply-protection").checked;
  if (execute) {
    const policy = qs("#measure-protection-policy").value;
    await ensureServiceDiscovery(policy.includes("rt"));
  }
  const payload = protectionPayload(execute);
  const data = await api("/api/protection/plan", { method: "POST", body: JSON.stringify(payload) });
  const execution = data.execution || {};
  const warnings = data.plan?.warnings || execution.warnings || [];
  const failed = (execution.results || []).filter((item) => !item.ok);
  let message = "";
  let tone = "";
  if (!execute) {
    message = `${payload.policy} 只是预演，没有真实修改进程。勾选 Protected 前执行策略后才会生效。`;
  } else if (execution.ok) {
    message = `${payload.policy} 已执行并验证通过。`;
    tone = "good";
  } else {
    const reason = failed[0]?.stderr || failed[0]?.error || warnings[0] || execution.error || "没有可验证的成功动作";
    message = `${payload.policy} 执行未生效：${reason}`;
    tone = "bad";
  }
  setNotice(
    "#protection-status",
    message,
    tone
  );
  renderPlan(data);
  state.selectedPolicy = payload.policy;
  return data;
}

function metricValue(m, key) {
  return m?.metric?.[key] ?? m?.benchmark?.[key] ?? null;
}

function renderPhaseCard(phase) {
  const data = state.measurements[phase];
  const el = qs(`#phase-${phase}`);
  const card = qs(`#phase-${phase}-card`);
  card.classList.toggle("ready", Boolean(data));
  if (!data) {
    el.textContent = "未采集";
    return;
  }
  const impact = phase === "baseline" ? null : phaseImpact(data);
  const overhead = data.diagnostic_overhead || {};
  el.innerHTML = `
    <div class="kv-list">
      <div><span>状态</span><strong>${data.ok ? "OK" : "FAILED"}</strong></div>
      <div><span>Throughput</span><strong>${fmt(metricValue(data, "throughput_tok_s"), 1)} tok/s</strong></div>
      <div><span>TTFT</span><strong>${fmt(metricValue(data, "ttft_avg_ms"), 1)} ms</strong></div>
      <div><span>TPOT</span><strong>${fmt(metricValue(data, "tpot_avg_ms"), 2)} ms</strong></div>
      <div><span>CPU</span><strong>${fmt(data.workload_profile?.cpu_delta_pct, 1)}%</strong></div>
      <div><span>CTX</span><strong>${fmt(data.workload_profile?.ctx_switches_per_s, 0)}/s</strong></div>
      <div><span>I/O</span><strong>${fmt((data.workload_profile?.disk_MBps || 0) + (data.workload_profile?.loopback_MBps || 0), 1)} MB/s</strong></div>
      <div><span>Load cores</span><strong>${escapeHtml(data.experiment?.workload_cores || "-")}</strong></div>
      <div><span>vLLM cores</span><strong>${escapeHtml(servicePlacementText(data))}</strong></div>
      <div><span>Placement</span><strong>${data.experiment?.service_placement_controlled ? "fixed" : "-"}</strong></div>
      <div><span>负载</span><strong>${escapeHtml(data.experiment?.workload || data.workload || "-")}</strong></div>
      <div><span>保护</span><strong>${escapeHtml(data.experiment?.protection_policy || "none")}</strong></div>
      <div><span>保护执行</span><strong>${phase === "protected" ? escapeHtml(protectionStatusText(data)) : "-"}</strong></div>
      <div><span>负载活跃度</span><strong>${data.workload_health ? `${escapeHtml(data.workload_health.verdict || "-")} / ${fmt(data.workload_health.signals?.system_cpu_percent, 1)}% CPU` : "-"}</strong></div>
      <div><span>LLM 影响</span><strong>${impact ? `TPOT ${fmt(impact.tpot, 1)}%, Thr ${fmt(impact.thr, 1)}%` : "-"}</strong></div>
      <div><span>诊断开销</span><strong>${fmt(overhead.estimated_canary_overhead_pct, 2)}% / ${fmt(overhead.canary_wall_time_s, 1)}s / ${overhead.canary_requests ?? 0} req</strong></div>
    </div>
  `;
}

function phaseImpact(data) {
  const base = state.measurements.baseline;
  if (!base) return null;
  return {
    tpot: pctIncrease(metricValue(data, "tpot_avg_ms"), metricValue(base, "tpot_avg_ms")),
    thr: pctDrop(metricValue(data, "throughput_tok_s"), metricValue(base, "throughput_tok_s")),
  };
}

function protectionStatusText(data) {
  const result = data.protection_result;
  if (!data.experiment?.protection_executed) return "dry-run / 未执行";
  if (!result) return "未知";
  if (result.execution?.ok) return "已验证";
  const failed = (result.execution?.results || []).filter((item) => !item.ok);
  const reason = failed[0]?.stderr || failed[0]?.error || result.plan?.warnings?.[0] || "失败";
  return `失败: ${String(reason).slice(0, 80)}`;
}

function stressCgroupPathFromProtection(result) {
  const rows = result?.execution?.results || [];
  const match = rows.find((item) => {
    const path = String(item.path || "");
    const weight = String(item.cpu_weight || "");
    return path.includes("stress_limit") || weight === "50";
  });
  return match?.path || "";
}

function protectionRequiresWorkloadRelaunch(policy) {
  const normalized = String(policy || "").toLowerCase();
  return normalized === "cgroup" || normalized.includes("numa");
}

function servicePlacementText(data) {
  const actual = (data.service_placement_result?.execution?.results || [])
    .map((item) => item.actual_affinity)
    .filter(Boolean);
  return actual.length ? actual.join("; ") : data.experiment?.service_cores || "-";
}

function renderAllPhaseCards() {
  ["baseline", "colocation", "protected"].forEach(renderPhaseCard);
  renderMacroBars();
}

async function collectPhase(phase, buttonSelector) {
  await withButton(buttonSelector, "采集中", async () => {
    let activeWorkloadCores = qs("#measure-workload-cores").value;
    if (phase === "baseline" && state.workloadJobId) {
      setNotice("#diagnosis-status", "Baseline 采集前停止本系统启动的共置负载，避免基线污染。", "");
      await stopColocationWorkload();
    }
    const protectedPolicy = phase === "protected" ? qs("#measure-protection-policy").value : "none";
    const shouldRelaunchWorkload = phase === "protected" && protectionRequiresWorkloadRelaunch(protectedPolicy);
    if (shouldRelaunchWorkload && state.workloadJobId) {
      setNotice("#diagnosis-status", `${protectedPolicy} needs a fresh co-tenant placement/cgroup; restarting the CPU workload before Protected.`, "");
      await stopColocationWorkload();
    }
    const placementResult = await applyControlledServicePlacement(phase);
    if (phase === "colocation" && qs("#measure-start-workload").checked) {
      setNotice("#diagnosis-status", `正在启动 ${qs("#measure-workload").value} 共置负载。`, "");
      await launchColocationWorkload();
    }
    if (phase === "protected") {
      const policy = protectedPolicy;
      const remoteCores = qs("#measure-protection-numa-intf").value.trim();
      setNotice("#diagnosis-status", `正在准备 ${qs("#measure-protection-policy").value} 保护策略。`, "");
      var protectionResult = await applyMeasureProtection();
      if (qs("#measure-start-workload").checked && (!state.workloadJobId || shouldRelaunchWorkload)) {
        const stressCgroupPath = policy === "cgroup" ? stressCgroupPathFromProtection(protectionResult) : "";
        const workloadCores = policy.includes("numa") && remoteCores ? remoteCores : qs("#measure-workload-cores").value;
        setNotice("#diagnosis-status", `Starting protected co-tenant workload on ${workloadCores || "default battle_cores"}.`, "");
        await launchColocationWorkload(workloadCores, stressCgroupPath);
        activeWorkloadCores = workloadCores;
      } else if (state.workloadJobId) {
        setNotice("#diagnosis-status", `${policy} keeps the existing co-tenant workload running; only vLLM protection is changed.`, "");
      }
    }
    let workloadHealth = null;
    if ((phase === "colocation" || phase === "protected") && state.workloadJobId) {
      workloadHealth = await checkCurrentWorkload(false);
    }
    setNotice("#diagnosis-status", `正在采集 ${phase}。`, "");
    const data = await api("/api/measure/canary", { method: "POST", body: JSON.stringify(measurePayload(phase)) });
    data.experiment = {
      workload: phase === "baseline" ? "none" : qs("#measure-workload").value,
      workload_job_id: phase === "colocation" || phase === "protected" ? state.workloadJobId : null,
      workload_cores: phase === "baseline" ? "" : activeWorkloadCores,
      service_cores: placementResult?.service_cores || "",
      service_placement_controlled: Boolean(placementResult),
      protection_policy: phase === "protected" ? qs("#measure-protection-policy").value : "none",
      protection_executed: phase === "protected" ? qs("#measure-apply-protection").checked : false,
    };
    data.workload_health = phase === "baseline" ? null : (workloadHealth || state.workloadHealth);
    data.service_placement_result = placementResult || null;
    if (phase === "protected") data.protection_result = protectionResult || null;
    state.measurements[phase] = data;
    renderAllPhaseCards();
    setNotice("#diagnosis-status", `${phase} 采集完成：${data.ok ? "canary 成功" : "canary 失败，仍保留系统采样"}`, data.ok ? "good" : "bad");
    await refreshTenants(false);
  });
}

function pctDrop(current, baseline) {
  if (current == null || baseline == null || Math.abs(Number(baseline)) < 1e-9) return null;
  return -100 * (Number(current) - Number(baseline)) / Number(baseline);
}

function pctIncrease(current, baseline) {
  if (current == null || baseline == null || Math.abs(Number(baseline)) < 1e-9) return null;
  return 100 * (Number(current) - Number(baseline)) / Number(baseline);
}

function renderMacroBars() {
  const base = state.measurements.baseline;
  const coloc = state.measurements.colocation;
  const prot = state.measurements.protected;
  const rows = [];
  if (base && coloc) {
    rows.push(["Throughput drop", pctDrop(metricValue(coloc, "throughput_tok_s"), metricValue(base, "throughput_tok_s")), pctDrop(metricValue(prot, "throughput_tok_s"), metricValue(base, "throughput_tok_s"))]);
    rows.push(["TTFT increase", pctIncrease(metricValue(coloc, "ttft_avg_ms"), metricValue(base, "ttft_avg_ms")), pctIncrease(metricValue(prot, "ttft_avg_ms"), metricValue(base, "ttft_avg_ms"))]);
    rows.push(["TPOT increase", pctIncrease(metricValue(coloc, "tpot_avg_ms"), metricValue(base, "tpot_avg_ms")), pctIncrease(metricValue(prot, "tpot_avg_ms"), metricValue(base, "tpot_avg_ms"))]);
  }
  const max = Math.max(10, ...rows.flatMap(([, a, b]) => [Math.abs(a || 0), Math.abs(b || 0)]));
  qs("#macro-bars").innerHTML = rows.length
    ? rows
        .map(([label, colocVal, protVal]) => `
          <div class="compare-row">
            <div class="bar-label">${escapeHtml(label)}</div>
            <div class="compare-bars">
              <div><span>Co-location</span><i class="hot" style="width:${Math.min(100, Math.abs(colocVal || 0) / max * 100)}%"></i><b>${fmt(colocVal, 1)}%</b></div>
              <div><span>Protected</span><i class="good" style="width:${Math.min(100, Math.abs(protVal || 0) / max * 100)}%"></i><b>${fmt(protVal, 1)}%</b></div>
            </div>
          </div>
        `)
        .join("")
    : "采集 Baseline 与 Co-location 后显示宏观性能变化。";
}

function stageMetrics(name) {
  return state.stages[name] || {};
}

async function buildDiagnosis(buttonSelector = "#rebuild-with-stage") {
  await withButton(buttonSelector, "诊断中", async () => {
    if (!state.measurements.baseline || !state.measurements.colocation) {
      setNotice("#micro-status", "生成 CoTail 诊断前，至少需要先在自动诊断页采集 Baseline 和 Co-location。", "bad");
      setView("diagnosis");
      return;
    }
    if (!Object.keys(stageMetrics("baseline")).length || !Object.keys(stageMetrics("current")).length) {
      setNotice("#micro-status", "CoTail 诊断需要 Baseline 和 Co-location 的微观阶段数据。请先解析 Nsight trace。", "bad");
      setView("micro");
      return;
    }
    const data = await api("/api/diagnoses/from-measurements", {
      method: "POST",
      body: JSON.stringify({
        workload: qs("#measure-workload").value,
        baseline: state.measurements.baseline,
        colocation: state.measurements.colocation,
        protected: state.measurements.protected,
        baseline_stage_metrics: stageMetrics("baseline"),
        stage_metrics: stageMetrics("current"),
        protected_stage_metrics: state.stages.protected,
        persist: true,
      }),
    });
    setCurrentDiagnosis(data);
    setNotice("#micro-status", `CoTail 诊断完成：${data.recommendation?.candidate_policy || "none"} / ${data.final?.decision || "UNKNOWN"} - ${data.final?.reason || ""}`, "good");
    setView("workflow");
  });
}

function setCurrentDiagnosis(data) {
  state.current = data;
  qs("#m-risk").textContent = data.risk?.risk || "-";
  qs("#m-cpti").textContent = data.cpti?.cpti_ratio == null ? "-" : fmt(data.cpti.cpti_ratio, 3);
  qs("#m-dominant").textContent = data.cpti?.dominant_stage || "-";
  qs("#m-policy").textContent = data.recommendation?.candidate_policy || "-";
  qs("#m-decision").textContent = data.final?.decision || "-";
  if (data.measurements) {
    state.measurements.baseline = data.measurements.baseline || state.measurements.baseline;
    state.measurements.colocation = data.measurements.colocation || state.measurements.colocation;
    state.measurements.protected = data.measurements.protected || state.measurements.protected;
    renderAllPhaseCards();
  }
  renderMicroDiagnosis();
  renderWorkflow();
  renderCtsGauge();
}

async function loadLatestDiagnosis() {
  try {
    const data = await api("/api/diagnoses/latest");
    if (data.ok) setCurrentDiagnosis(data);
  } catch (err) {
    console.error(err);
  }
}

function metricsFromNsight(data) {
  const stages = data.stages || {};
  return Object.fromEntries(
    Object.entries(stages)
      .filter(([, item]) => Number(item.count || 0) > 0)
      .map(([stage, item]) => [stage, { p95_us: item.p95_us, p99_us: item.p99_us }])
  );
}

function stageTrackHtml(kind, label, p95, p99, max) {
  const p95Width = Math.min(100, Math.max(0, Number(p95 || 0) / max * 100));
  const tailWidth = Math.min(100 - p95Width, Math.max(0, (Number(p99 || 0) - Number(p95 || 0)) / max * 100));
  return `
    <div class="stage-track ${kind}">
      <span>${escapeHtml(label)}</span>
      <div class="stage-bar-stack">
        <i style="width:${p95Width}%"></i>
        <i class="tail" style="width:${tailWidth}%"></i>
      </div>
      <b>${fmt(p95, 0)} / ${fmt(p99, 0)} us</b>
    </div>
  `;
}

function renderStageSources() {
  const rows = [
    ["Baseline", Object.keys(state.stages.baseline || {}).length],
    ["Co-location", Object.keys(state.stages.current || {}).length],
    ["Protected", Object.keys(state.stages.protected || {}).length],
  ];
  qs("#stage-source").innerHTML = rows.map(([name, count]) => `<div><span>${name}</span><strong>${count ? `${count} stages ready` : "missing"}</strong></div>`).join("");
}

async function parseStage(which, pathSelector, buttonSelector) {
  await withButton(buttonSelector, "解析中", async () => {
    const data = await api("/api/nsight/parse", { method: "POST", body: JSON.stringify({ path: qs(pathSelector).value }) });
    if (!data.ok) {
      setNotice("#micro-status", data.error || "解析失败", "bad");
      return;
    }
    state.stages[which] = metricsFromNsight(data);
    renderStageSources();
    setNotice("#micro-status", `${which} 阶段数据已解析：${data.vllm_event_count || 0} 个 vLLM NVTX 事件。`, "good");
  });
}

async function rebuildWithStage() {
  await buildDiagnosis("#rebuild-with-stage");
}

function renderMicroDiagnosis() {
  const data = state.current;
  renderStageSources();
  if (!data?.cpti?.stage_scores?.length) {
    qs("#stage-amplification").textContent = "还没有 CPTI 阶段数据。请解析 Nsight SQLite，或运行包含 stage_metrics 的诊断。";
    qs("#bottleneck-badge").className = "badge";
    qs("#bottleneck-badge").textContent = "UNKNOWN";
    return;
  }
  const dominant = data.cpti.dominant_stage;
  qs("#bottleneck-badge").className = "badge bad";
  qs("#bottleneck-badge").textContent = dominant || "UNKNOWN";
  const protectedByStage = Object.fromEntries((data.protected_cpti?.stage_scores || []).map((s) => [s.stage, s]));
  const max = Math.max(
    1,
    ...data.cpti.stage_scores.map((s) => Math.max(s.p99_us || 0, s.base_p99_us || 0, protectedByStage[s.stage]?.p99_us || 0))
  );
  qs("#stage-amplification").innerHTML = data.cpti.stage_scores
    .map((s) => {
      const isDominant = s.stage === dominant;
      const protectedStage = protectedByStage[s.stage];
      const explain = s.stage === "batch.construct" || s.stage === "scheduler.step"
        ? "调度器/队列构造受阻，优先考虑 EngineCore RT。"
        : s.stage === "model.forward"
          ? "模型前向尾部被拖慢，关注 NUMA/缓存/拓扑压力。"
          : "执行阶段受影响，检查 CPU 抢占与内存通路。";
      return `
        <article class="stage-row ${isDominant ? "dominant" : ""}">
          <div class="stage-title">
            <strong>${escapeHtml(s.stage)}</strong>
            ${isDominant ? badge("主导瓶颈", "bad") : ""}
          </div>
          <div class="stage-bars">
            ${stageTrackHtml("baseline", "Baseline P95/P99", s.base_p95_us, s.base_p99_us, max)}
            ${stageTrackHtml("coloc", "Co-location P95/P99", s.p95_us, s.p99_us, max)}
            ${
              protectedStage
                ? stageTrackHtml("protected", "Protected P95/P99", protectedStage.p95_us, protectedStage.p99_us, max)
                : `<div class="stage-track missing"><span>Protected P95/P99</span><div class="stage-bar-stack missing-bar"></div><b>等待 protected trace</b></div>`
            }
          </div>
          <div class="stage-foot"><span>CPTI score ${fmt(s.score_ratio, 3)}${protectedStage?.score_ratio != null ? ` / protected ${fmt(protectedStage.score_ratio, 3)}` : ""}</span><span>${escapeHtml(explain)}</span></div>
        </article>
      `;
    })
    .join("");
  setNotice("#micro-status", `CPTI=${fmt(data.cpti.cpti_ratio, 3)}，主导阶段：${dominant || "unknown"}。`, "good");
}

function renderWorkflow() {
  const data = state.current || {};
  const hasRisk = Boolean(data.risk);
  const hasCpti = Boolean(data.cpti?.cpti_ratio != null);
  const hasPolicy = Boolean(data.recommendation?.candidate_policy);
  const hasCts = Boolean(data.cts_ratio != null || data.final?.decision);
  const nodes = [
    ["硬件特征抓取", hasRisk, hasRisk ? data.risk.risk : "waiting"],
    ["风险评估", hasRisk, hasRisk ? data.risk.risk : "waiting"],
    ["NVTX Trace 与 CPTI", hasCpti, hasCpti ? `CPTI ${fmt(data.cpti.cpti_ratio, 2)}` : "need trace"],
    ["策略推荐", hasPolicy, hasPolicy ? data.recommendation.candidate_policy : "waiting"],
    ["SLO 与 CTS 验证", hasCts, hasCts ? data.final?.decision || `CTS ${fmt(data.cts_pct, 1)}%` : "waiting"],
  ];
  const currentIndex = nodes.findIndex(([, done]) => !done);
  qs("#workflow-current").textContent = currentIndex === -1 ? "流程完成" : `当前：${nodes[currentIndex][0]}`;
  qs("#workflow-graph").innerHTML = nodes
    .map(([label, done, detail], idx) => `<div class="workflow-node ${done ? "done" : idx === currentIndex ? "active" : ""}"><span>${idx + 1}</span><strong>${label}</strong><em>${escapeHtml(detail)}</em></div>`)
    .join("");
}

function renderCtsGauge() {
  const data = state.current || {};
  const cts = data.cts_pct == null ? null : Math.max(0, Math.min(100, Number(data.cts_pct)));
  qs("#cts-value").textContent = cts == null ? "-" : `${fmt(cts, 1)}%`;
  qs("#cts-gauge").style.setProperty("--value", String(cts || 0));
  qs("#cts-text").textContent = data.cts_ratio == null ? "等待 protected measurement" : `CTS ratio ${fmt(data.cts_ratio, 3)}`;
  const policy = data.recommendation?.candidate_policy || state.selectedPolicy;
  const dominant = data.cpti?.dominant_stage || "核心阶段";
  setNotice(
    "#cts-explain",
    cts == null
      ? "执行策略并采集 Protected 后显示核心尾部抑制率。"
      : `开启 ${policy} 后，${dominant} 的尾部放大被抑制 ${fmt(cts, 1)}%，用于判断 TPOT 是否恢复。`,
    cts != null && cts >= 80 ? "good" : ""
  );
}

function renderPlan(data) {
  const actions = data.plan?.actions || [];
  qs("#policy-actions").innerHTML = actions.length
    ? `
      <table>
        <thead><tr><th>Kind</th><th>Target</th><th>Detail</th><th>Command</th></tr></thead>
        <tbody>${actions.map((a) => `<tr><td>${escapeHtml(a.kind)}</td><td>${escapeHtml(a.target)}</td><td>${escapeHtml(a.detail)}</td><td>${escapeHtml((a.command || []).join(" "))}</td></tr>`).join("")}</tbody>
      </table>`
    : "没有生成动作。";
  showJson(qs("#policy-output"), data);
}

async function applyPolicy(execute) {
  const selector = execute ? "#execute-policy" : "#dry-run-policy";
  if (execute && !window.confirm("将真实修改进程 affinity/nice/RT 调度，确认执行？")) return;
  await withButton(selector, execute ? "执行中" : "生成中", async () => {
    const data = await api("/api/protection/plan", {
      method: "POST",
      body: JSON.stringify({
        policy: qs("#override-policy").value,
        vllm_pids: parseCsvPositiveInts(qs("#override-pids").value),
        engine_tid: Number(qs("#override-tid").value.trim()) || null,
        battle_cores: qs("#override-battle").value,
        numa_vllm_cpus: qs("#override-numa-vllm").value,
        numa_interference_cpus: qs("#override-numa-intf").value,
        rt_priority: num("#override-rt-priority", 50),
        execute,
      }),
    });
    renderPlan(data);
    state.selectedPolicy = qs("#override-policy").value;
    await refreshTenants(false);
  });
}

async function scanService(buttonSelector = "#scan-service-btn", portsSelector = "#override-ports") {
  await withButton(buttonSelector, "扫描中", async () => {
    const data = await api("/api/process/discover", {
      method: "POST",
      body: JSON.stringify({ framework: "vllm", ports: parseCsvInts(qs(portsSelector).value) }),
    });
    state.discovery = data;
    const pids = (data.processes || []).map((p) => Number(p.pid)).filter((pid) => Number.isInteger(pid) && pid > 0);
    qs("#override-pids").value = pids.join(",");
    qs("#measure-protection-pids").value = pids.join(",");
    qs("#tenant-protected-pids").value = pids.join(",");
    setNotice("#protection-status", pids.length ? `已发现服务 PID: ${pids.join(",")}` : "没有发现服务进程。", pids.length ? "good" : "bad");
    showJson(qs("#policy-output"), data);
  });
}

async function identifyTid(buttonSelector = "#identify-tid-btn", scanButtonSelector = "#scan-service-btn", portsSelector = "#override-ports") {
  if (!state.discovery?.processes?.length) await scanService(scanButtonSelector, portsSelector);
  const pids = (state.discovery?.processes || []).map((p) => Number(p.pid)).filter((pid) => Number.isInteger(pid) && pid > 0);
  if (!pids.length) return;
  await withButton(buttonSelector, "识别中", async () => {
    const data = await api("/api/process/busy-thread?probe_seconds=1.0", { method: "POST", body: JSON.stringify(pids) });
    state.busyThread = data;
    if (data.selected?.tid) {
      qs("#override-tid").value = data.selected.tid;
      qs("#measure-protection-tid").value = data.selected.tid;
    }
    const selectedText = data.selected?.tid
      ? `已识别 TID ${data.selected.tid} (${data.selected.selection_reason || "selected"}, ${data.selected.process_name || "-"} / ${data.selected.thread_name || "-"})`
      : "未识别到忙线程。";
    setNotice("#protection-status", selectedText, data.selected?.tid ? "good" : "bad");
    showJson(qs("#policy-output"), { discovery: state.discovery, busy_thread: data });
  });
}

async function refreshTenants(useButton = true) {
  const run = async () => {
    const data = await api("/api/tenants/snapshot", {
      method: "POST",
      body: JSON.stringify({
        policy: qs("#tenant-policy").value || state.selectedPolicy,
        protected_pids: parseCsvPositiveInts(qs("#tenant-protected-pids").value),
        limit: num("#tenant-limit", 30),
      }),
    });
    renderTenants(data);
  };
  if (useButton) await withButton("#tenant-refresh-btn", "刷新中", run);
  else await run();
}

function renderTenants(data) {
  const cost = data.cost || {};
  const overhead = data.diagnostic_overhead || {};
  qs("#tenant-cpu-pressure").textContent = `${fmt(cost.cpu_pressure_pct, 1)}%`;
  qs("#tenant-slowdown").textContent = `${fmt(cost.estimated_cotenant_slowdown_pct, 1)}%`;
  qs("#tenant-cost-class").textContent = cost.cost_class || "-";
  qs("#tenant-canary-overhead").textContent = `${fmt(overhead.canary_overhead_pct, 2)}%`;
  qs("#tenant-trace-interval").textContent = `${overhead.recommended_trace_interval_s || "-"}s`;
  qs("#tenant-count").textContent = `${(data.tenants || []).length} processes`;
  qs("#tenant-table").innerHTML = (data.tenants || []).length
    ? `
      <table>
        <thead><tr><th>PID</th><th>Workload</th><th>CPU</th><th>RSS</th><th>User</th><th>Command</th></tr></thead>
        <tbody>${data.tenants.map((t) => `<tr><td>${t.pid}</td><td>${escapeHtml(t.workload)}</td><td>${fmt(t.cpu_percent, 1)}%</td><td>${fmt(t.rss_mb, 1)} MB</td><td>${escapeHtml(t.user)}</td><td>${escapeHtml(t.cmdline)}</td></tr>`).join("")}</tbody>
      </table>`
    : "没有发现明显 CPU 共置负载。";
  qs("#overhead-wall").innerHTML = `
    <div class="wall-item"><span>Canary</span><strong>${fmt(overhead.canary_overhead_pct, 2)}%</strong></div>
    <div class="wall-item"><span>Nsight</span><strong>${overhead.nsight_trace_overhead_pct == null ? "on demand" : `${fmt(overhead.nsight_trace_overhead_pct, 2)}%`}</strong></div>
    <div class="wall-item"><span>Trace cadence</span><strong>${overhead.recommended_trace_interval_s || "-"}s</strong></div>
    <div class="wall-item"><span>Risk</span><strong>${escapeHtml(overhead.risk || "-")}</strong></div>
  `;
  showJson(qs("#tenant-output"), data);
}

function bind() {
  qsa(".nav-item").forEach((btn) => btn.addEventListener("click", () => setView(btn.dataset.view)));
  qs("#launch-job-btn").addEventListener("click", () => launchJob().catch((err) => {
    setNotice("#launch-status", err.message, "bad");
    showJson(qs("#launch-output"), { ok: false, error: err.message });
  }));
  qs("#refresh-jobs").addEventListener("click", () => refreshRuntime().catch(alert));
  qs("#llm-health-btn").addEventListener("click", () => llmHealth().catch((err) => showJson(qs("#llm-output"), { ok: false, error: err.message })));
  qs("#llm-request-btn").addEventListener("click", () => llmRequest().catch((err) => showJson(qs("#llm-output"), { ok: false, error: err.message })));
  qs("#llm-bench-btn").addEventListener("click", () => llmBenchmark().catch((err) => showJson(qs("#llm-output"), { ok: false, error: err.message })));

  qs("#collect-baseline").addEventListener("click", () => collectPhase("baseline", "#collect-baseline").catch((err) => setNotice("#diagnosis-status", err.message, "bad")));
  qs("#collect-colocation").addEventListener("click", () => collectPhase("colocation", "#collect-colocation").catch((err) => setNotice("#diagnosis-status", err.message, "bad")));
  qs("#collect-protected").addEventListener("click", () => collectPhase("protected", "#collect-protected").catch((err) => setNotice("#diagnosis-status", err.message, "bad")));
  qs("#build-long-prompt").addEventListener("click", buildLongPrompt);

  qs("#parse-baseline-stage").addEventListener("click", () => parseStage("baseline", "#stage-baseline-path", "#parse-baseline-stage").catch((err) => setNotice("#micro-status", err.message, "bad")));
  qs("#parse-current-stage").addEventListener("click", () => parseStage("current", "#stage-current-path", "#parse-current-stage").catch((err) => setNotice("#micro-status", err.message, "bad")));
  qs("#parse-protected-stage").addEventListener("click", () => parseStage("protected", "#stage-protected-path", "#parse-protected-stage").catch((err) => setNotice("#micro-status", err.message, "bad")));
  qs("#rebuild-with-stage").addEventListener("click", () => rebuildWithStage().catch((err) => setNotice("#micro-status", err.message, "bad")));
  qs("#refresh-micro-btn").addEventListener("click", renderMicroDiagnosis);

  qsa(".segmented button").forEach((btn) => {
    btn.addEventListener("click", () => {
      qsa(".segmented button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.selectedPolicy = btn.dataset.policy;
      qs("#override-policy").value = btn.dataset.policy;
      qs("#measure-protection-policy").value = btn.dataset.policy;
      qs("#tenant-policy").value = btn.dataset.policy;
      renderWorkflow();
    });
  });
  qs("#start-workload-btn").addEventListener("click", () => startSelectedWorkload().catch((err) => setNotice("#workload-status", err.message, "bad")));
  qs("#stop-workload-btn").addEventListener("click", () => stopColocationWorkload().catch((err) => setNotice("#workload-status", err.message, "bad")));
  qs("#stop-managed-workloads-btn").addEventListener("click", () => stopAllManagedWorkloads().catch((err) => setNotice("#workload-status", err.message, "bad")));
  qs("#scan-orphan-workloads-btn").addEventListener("click", () => scanOrphanWorkloads().catch((err) => setNotice("#workload-status", err.message, "bad")));
  qs("#cleanup-orphan-workloads-btn").addEventListener("click", () => cleanupOrphanWorkloads().catch((err) => setNotice("#workload-status", err.message, "bad")));
  qs("#check-workload-btn").addEventListener("click", () => checkCurrentWorkload(true).catch((err) => setNotice("#workload-status", err.message, "bad")));
  qs("#probe-all-workloads-btn").addEventListener("click", () => probeAllWorkloads().catch((err) => setNotice("#workload-status", err.message, "bad")));
  qs("#measure-scan-service").addEventListener("click", () => scanService("#measure-scan-service", "#measure-protection-ports").catch((err) => setNotice("#protection-status", err.message, "bad")));
  qs("#measure-identify-tid").addEventListener("click", () => identifyTid("#measure-identify-tid", "#measure-scan-service", "#measure-protection-ports").catch((err) => setNotice("#protection-status", err.message, "bad")));
  qs("#scan-service-btn").addEventListener("click", () => scanService().catch((err) => showJson(qs("#policy-output"), { ok: false, error: err.message })));
  qs("#identify-tid-btn").addEventListener("click", () => identifyTid().catch((err) => showJson(qs("#policy-output"), { ok: false, error: err.message })));
  qs("#dry-run-policy").addEventListener("click", () => applyPolicy(false).catch((err) => showJson(qs("#policy-output"), { ok: false, error: err.message })));
  qs("#execute-policy").addEventListener("click", () => applyPolicy(true).catch((err) => showJson(qs("#policy-output"), { ok: false, error: err.message })));

  qs("#tenant-refresh-btn").addEventListener("click", () => refreshTenants(true).catch((err) => showJson(qs("#tenant-output"), { ok: false, error: err.message })));
}

bind();
buildLongPrompt();
renderAllPhaseCards();
renderStageSources();
renderWorkflow();
renderCtsGauge();
connectRuntimeStream();
refreshRuntime().catch(console.error);
loadLatestDiagnosis().catch(console.error);
refreshTenants(false).catch(console.error);
