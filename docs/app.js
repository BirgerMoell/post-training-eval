const state = { data: null, filters: { model: "", capability: "", evidence: "", search: "" } };

const escapeHtml = value => String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]);
const shortModel = id => id.split("/").pop();
const shortSha = sha => sha ? sha.slice(0, 9) : "unpinned";
const label = value => String(value || "").replaceAll("-", " ").replace(/\b\w/g, c => c.toUpperCase());

function canonicalMetric(value) {
  const aliases = {
    "prompt_level_strict_acc": "prompt_level_strict_accuracy",
    "inst_level_strict_acc": "instruction_level_strict_accuracy",
    "inst_level_strict_accuracy": "instruction_level_strict_accuracy",
    "acc_norm": "accuracy_normalized",
    "acc": "accuracy",
    "exact_match,strict-match": "exact_match_strict",
    "exact_match,flexible-extract": "exact_match_flexible",
    "flexible_exact_match": "exact_match_flexible",
    "pass@1": "pass_at_1"
  };
  const raw = String(value || "");
  const normalized = raw.endsWith(",none") ? raw.slice(0, -5) : raw;
  return aliases[raw] || aliases[normalized] || normalized;
}

function lookupBenchmark(capability, benchmark) {
  const cap = state.data.capabilities.find(item => item.id === capability);
  return cap?.benchmarks.find(item => item.id === benchmark);
}

function benchmarkSources(benchmark) {
  if (Array.isArray(benchmark?.sources)) return benchmark.sources;
  return benchmark?.url ? [{ label: "Source", url: benchmark.url }] : [];
}

function benchmarkNameLink(benchmark, fallbackName = "") {
  const name = benchmark?.name || fallbackName;
  const primary = benchmarkSources(benchmark)[0];
  if (!primary) return escapeHtml(name);
  return `<a class="benchmark-name-link" href="${escapeHtml(primary.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(name)} <span aria-hidden="true">↗</span></a>`;
}

function benchmarkSourceLinks(benchmark) {
  const sources = benchmarkSources(benchmark);
  if (!sources.length) return "";
  return `<span class="benchmark-source-links">${sources.map(source => `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.label)} ↗</a>`).join("")}</span>`;
}

function modelCapability(model, capabilityId) {
  return model.capability_scores.find(item => item.id === capabilityId);
}

function scoreText(metric) {
  if (metric.scale === "percentage") return `${Number(metric.value).toFixed(metric.value % 1 ? 2 : 0)}%`;
  if (metric.scale === "rank") return `#${metric.value}`;
  if (metric.scale === "milliseconds") return `${metric.value} ms`;
  return String(metric.value);
}

function flatten() {
  return state.data.runs.flatMap(run => run.metrics.map(metric => ({ run, metric })));
}

function targetLabel(status) {
  return ({
    met: "On target",
    missed: "Off target",
    partial: "Partial coverage",
    "not-measured": "Not measured",
    "not-set": "No target"
  })[status] || "No target";
}

function targetBadge(status) {
  return `<span class="target-badge ${escapeHtml(status)}">${escapeHtml(targetLabel(status))}</span>`;
}

function targetThreshold(benchmark) {
  if (benchmark?.target == null) return "No numeric target";
  return `${benchmark.direction === "lower" ? "≤" : "≥"} ${benchmark.target}`;
}

function targetUnit(scale, benchmark) {
  if (scale === "rank" || benchmark?.metric === "rank") return "places";
  if (scale === "milliseconds" || benchmark?.metric === "milliseconds") return "ms";
  if (benchmark?.metric === "output_tokens_per_second") return "tok/s";
  if (["bleu", "chrf++", "aggregate_score", "primary_task_metric"].includes(benchmark?.metric)) return "pts";
  return "pp";
}

function targetValueText(value, scale, benchmark) {
  if (value == null) return "—";
  const numeric = Number(value);
  const rendered = numeric.toLocaleString(undefined, { maximumFractionDigits: 1 });
  if (scale === "rank" || benchmark?.metric === "rank") return `#${rendered}`;
  if (scale === "milliseconds" || benchmark?.metric === "milliseconds") return `${rendered} ms`;
  if (benchmark?.metric === "output_tokens_per_second") return `${rendered} tok/s`;
  if (["bleu", "chrf++", "aggregate_score", "primary_task_metric"].includes(benchmark?.metric)) return rendered;
  return `${rendered}%`;
}

function targetGap(value, benchmark) {
  if (benchmark?.target == null || !Number.isFinite(Number(value))) return null;
  return benchmark.direction === "lower" ? Number(benchmark.target) - Number(value) : Number(value) - Number(benchmark.target);
}

function targetGapText(gap, direction, scale, benchmark) {
  if (gap == null) return "";
  const absolute = Math.abs(Number(gap)).toLocaleString(undefined, { maximumFractionDigits: 1 });
  const unit = targetUnit(scale, benchmark);
  if (Number(gap) === 0) return "At target";
  if (direction === "lower") return gap > 0 ? `${absolute} ${unit} inside limit` : `${absolute} ${unit} over limit`;
  return gap > 0 ? `+${absolute} ${unit} above target` : `−${absolute} ${unit} below target`;
}

function targetGapBadge(gap, direction, scale, benchmark) {
  if (gap == null) return "";
  const status = gap >= 0 ? "met" : "missed";
  return `<span class="target-gap ${status}">${escapeHtml(targetGapText(gap, direction, scale, benchmark))}</span>`;
}

function metricTargetStatus(metric, benchmark) {
  if (benchmark?.target == null) return "not-set";
  if (canonicalMetric(metric.metric) !== canonicalMetric(benchmark.metric)) return "not-set";
  const met = benchmark.direction === "lower" ? metric.value <= benchmark.target : metric.value >= benchmark.target;
  return met ? "met" : "missed";
}

function evidenceCell(run) {
  const source = /^https?:\/\//.test(run.provenance.source)
    ? `<a class="source" href="${escapeHtml(run.provenance.source)}">source ↗</a>`
    : `<span class="source" title="${escapeHtml(run.provenance.source)}">archived raw artifact</span>`;
  const caveat = run.diagnostic
    ? `<span class="pill diagnostic" title="${escapeHtml((run.limitations || []).join(" "))}">Diagnostic</span>`
    : "";
  return `<span class="pill ${escapeHtml(run.provenance.kind)}">${escapeHtml(label(run.provenance.kind))}</span>${caveat}${source}`;
}

function renderScorecards() {
  const cards = state.data.models.map(model => {
    const aggregate = model.aggregate_score == null ? "—" : model.aggregate_score.toFixed(1);
    const capabilities = model.capability_scores.map(capability => {
      const score = capability.score == null ? "—" : capability.score.toFixed(1);
      return `<button class="mini-capability ${capability.score == null ? "missing" : ""} target-${escapeHtml(capability.target_status)}" type="button" data-model="${escapeHtml(model.id)}" data-capability="${escapeHtml(capability.id)}" title="Open ${escapeHtml(capability.name)} evidence for ${escapeHtml(shortModel(model.id))}"><span>${escapeHtml(capability.name)}</span><strong>${score}</strong></button>`;
    }).join("");
    const targetCopy = `${model.targets_met}/${model.targets_measured} measured targets met · ${model.targets_measured}/${model.target_benchmark_count} target evaluations measured`;
    return `<article class="model-scorecard">
      <div class="model-card-head"><div><p class="eyebrow">MODEL</p><h3>${escapeHtml(shortModel(model.id))}</h3><span class="model-revision">${escapeHtml(model.revisions.map(shortSha).join(" · "))}</span></div>${targetBadge(model.target_status)}</div>
      <div class="aggregate"><strong>${aggregate}</strong><span>capability index / 100<br>${model.scored_capability_count}/${model.total_capability_count} capabilities scored</span></div>
      <div class="aggregate-track"><span style="width:${model.aggregate_score || 0}%"></span></div>
      <p class="target-summary">${escapeHtml(targetCopy)}</p>
      <div class="mini-capability-grid">${capabilities}</div>
    </article>`;
  }).join("");
  document.querySelector("#model-scorecards").innerHTML = cards;
  document.querySelectorAll(".mini-capability").forEach(button => button.addEventListener("click", () => openCapability(button.dataset.capability, button.dataset.model)));
}

function renderGate() {
  const gate = state.data.gates?.at(-1);
  const target = document.querySelector("#gate-summary");
  if (!gate) { target.hidden = true; return; }
  const status = gate.status === "passed" ? "PASS" : "FAIL";
  const regressions = gate.regressions.length
    ? gate.regressions.map(item => `${label(item.key[1])}: ${item.improvement.toFixed(2)} pts`).join(" · ")
    : "No matched metric exceeded the tolerance.";
  target.innerHTML = `<p><span class="gate-status ${gate.status}">${status}</span><strong>Latest parent-retention gate</strong> · ${gate.comparison_count} matched measurements</p><span>${escapeHtml(regressions)}</span>`;
}

function renderSweepContract() {
  const profile = state.data.sweep_profile;
  if (!profile) return;
  const probes = profile.capability_probes || [];
  const total = probes.reduce((sum, probe) => sum + probe.benchmarks.length, 0);
  document.querySelector("#sweep-summary").innerHTML = `<div><strong>${probes.length}/9</strong><span>capabilities covered</span></div><div><strong>${total}</strong><span>benchmark probes</span></div><p>${escapeHtml(profile.coverage)}</p>`;
  document.querySelector("#sweep-probes").innerHTML = probes.map(probe => {
    const capability = state.data.capabilities.find(item => item.id === probe.capability);
    const links = probe.benchmarks.map(id => {
      const benchmark = lookupBenchmark(probe.capability, id);
      return `<li>${benchmarkNameLink(benchmark, label(id))}</li>`;
    }).join("");
    return `<article><span>${escapeHtml(capability?.name || label(probe.capability))}</span><strong>${probe.benchmarks.length}</strong><ul>${links}</ul></article>`;
  }).join("");
}

function formatDuration(seconds) {
  if (!Number.isFinite(Number(seconds))) return "—";
  const total = Math.round(Number(seconds));
  return `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, "0")}s`;
}

function renderFastDiagnostics() {
  const runs = state.data.runs.filter(run => run.diagnostic && run.profile === "fast");
  const section = document.querySelector("#fast-diagnostics");
  if (!runs.length) { section.hidden = true; return; }
  section.hidden = false;
  const newestByModel = new Map();
  runs.forEach(run => {
    const current = newestByModel.get(run.model.id);
    const runKey = `${run.finished_at || run.started_at || ""}\u0000${run.run_id || ""}`;
    const currentKey = current ? `${current.finished_at || current.started_at || ""}\u0000${current.run_id || ""}` : "";
    if (!current || runKey > currentKey) newestByModel.set(run.model.id, run);
  });
  const ordered = [...newestByModel.values()].sort((a, b) => a.model.id.localeCompare(b.model.id));
  document.querySelector("#fast-run-cards").innerHTML = ordered.map(run => {
    const summary = run.diagnostic_summary || {};
    const completed = summary.completed_tasks ?? run.task_statuses?.filter(item => item.status === "completed").length ?? 0;
    const scheduled = summary.scheduled_tasks ?? run.task_statuses?.length ?? 0;
    return `<article class="fast-run-card"><div><span class="pill diagnostic">Fast · n≤${escapeHtml(summary.example_limit ?? "?")}</span><h3>${escapeHtml(shortModel(run.model.id))}</h3><span class="model-revision">${escapeHtml(shortSha(run.model.revision))}</span></div><div class="fast-run-stat"><strong>${completed}/${scheduled}</strong><span>tasks available</span></div><div class="fast-run-stat"><strong>${escapeHtml(formatDuration(run.runtime_seconds))}</strong><span>active run time · ${escapeHtml(run.environment?.accelerator || "accelerator unrecorded")}</span></div></article>`;
  }).join("");

  const preferred = ["MATH500", "AIME24", "AIME25", "AMC23", "HumanEval", "LiveCodeBench", "GPQADiamond"];
  const statuses = ordered.flatMap(run => run.task_statuses || []);
  const tasks = [...new Set(statuses.map(item => item.task))].sort((a, b) => {
    const ai = preferred.indexOf(a), bi = preferred.indexOf(b);
    return (ai < 0 ? 999 : ai) - (bi < 0 ? 999 : bi) || a.localeCompare(b);
  });
  const header = `<div class="fast-row fast-head" style="--fast-model-count:${ordered.length}"><span>Benchmark</span>${ordered.map(run => `<span>${escapeHtml(shortModel(run.model.id))}</span>`).join("")}</div>`;
  const rows = tasks.map(task => {
    const definitionStatus = statuses.find(item => item.task === task);
    const benchmark = lookupBenchmark(
      state.data.capabilities.find(cap => cap.benchmarks.some(item => item.id === definitionStatus?.benchmark))?.id,
      definitionStatus?.benchmark,
    );
    const name = benchmark?.name || label(task);
    const cells = ordered.map(run => {
      const taskStatus = (run.task_statuses || []).find(item => item.task === task);
      if (!taskStatus || taskStatus.status !== "completed") {
        return `<div class="fast-score unavailable"><strong>Unavailable</strong><small>${escapeHtml(taskStatus?.reason || "not run")}</small></div>`;
      }
      const metrics = run.metrics.filter(metric => metric.benchmark === taskStatus.benchmark);
      if (!metrics.length) return `<div class="fast-score unavailable"><strong>—</strong><small>no saved score</small></div>`;
      const score = metrics.reduce((sum, metric) => sum + Number(metric.value), 0) / metrics.length;
      const sampleCounts = [...new Set(metrics.map(metric => metric.n).filter(value => value != null))];
      const slices = metrics.length > 1 ? `${metrics.length} slices · ` : "";
      const gap = targetGap(score, benchmark);
      return `<div class="fast-score"><strong>${escapeHtml(score.toFixed(score % 1 ? 1 : 0))}%</strong><small>${escapeHtml(slices)}${sampleCounts.length ? `n=${sampleCounts.join("/")}` : "sample count unavailable"}</small>${targetGapBadge(gap, benchmark?.direction, "percentage", benchmark)}</div>`;
    }).join("");
    return `<div class="fast-row" style="--fast-model-count:${ordered.length}"><div><strong>${benchmarkNameLink(benchmark, name)}</strong><small>${escapeHtml(benchmark?.metric || "score")} · target ${escapeHtml(targetThreshold(benchmark))}</small></div>${cells}</div>`;
  }).join("");
  document.querySelector("#fast-comparison").innerHTML = header + rows;
}

function benchmarkCell(model, capabilityId, benchmark) {
  const capability = modelCapability(model, capabilityId);
  const evidence = capability?.benchmarks.find(item => item.id === benchmark.id);
  const score = evidence?.target_value != null
    ? targetValueText(evidence.target_value, evidence.target_scale, benchmark)
    : (evidence?.score == null ? "—" : evidence.score.toFixed(1));
  const status = evidence?.target_status || (benchmark.target == null ? "not-set" : "not-measured");
  const gap = targetGapBadge(evidence?.target_gap, benchmark.direction, evidence?.target_scale, benchmark);
  return `<div class="benchmark-model-cell"><strong>${escapeHtml(score)}</strong>${targetBadge(status)}${gap}</div>`;
}

function renderCapabilityComparison() {
  const models = state.data.models;
  const header = `<div class="comparison-header comparison-columns" style="--model-count:${models.length}"><span>Capability</span>${models.map(model => `<span>${escapeHtml(shortModel(model.id))}</span>`).join("")}</div>`;
  const rows = state.data.capabilities.map(capability => {
    const modelSummaries = models.map(model => {
      const result = modelCapability(model, capability.id);
      const score = result?.score == null ? "—" : result.score.toFixed(1);
      return `<div class="capability-model-summary"><strong>${score}</strong><div class="score-track"><span style="width:${result?.score || 0}%"></span></div>${targetBadge(result?.target_status || "not-measured")}<small>${result?.benchmark_count || 0} scored evals</small></div>`;
    }).join("");
    const benchmarks = capability.benchmarks.map(benchmark => `<div class="benchmark-row comparison-columns" style="--model-count:${models.length}"><div><strong>${benchmarkNameLink(benchmark)}</strong><small>${escapeHtml(benchmark.metric)} · ${escapeHtml(targetThreshold(benchmark))}</small>${benchmarkSourceLinks(benchmark)}</div>${models.map(model => benchmarkCell(model, capability.id, benchmark)).join("")}</div>`).join("");
    return `<details class="capability-comparison-row" id="capability-${escapeHtml(capability.id)}"><summary class="comparison-columns" style="--model-count:${models.length}"><div class="capability-title"><span>${escapeHtml(capability.name)}</span><small>${capability.benchmarks.length} registered evaluations</small></div>${modelSummaries}</summary><div class="benchmark-list">${benchmarks}<button class="evidence-button" type="button" data-capability="${escapeHtml(capability.id)}">Open evidence ledger</button></div></details>`;
  }).join("");
  document.querySelector("#capability-comparison").innerHTML = header + rows;
  document.querySelectorAll(".evidence-button").forEach(button => button.addEventListener("click", () => openEvidence("", button.dataset.capability)));
}

function openCapability(capabilityId, modelId) {
  const detail = document.querySelector(`#capability-${CSS.escape(capabilityId)}`);
  if (detail) {
    detail.open = true;
    detail.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function openEvidence(modelId, capabilityId) {
  state.filters.model = modelId;
  state.filters.capability = capabilityId;
  document.querySelector("#model-filter").value = modelId;
  document.querySelector("#capability-filter").value = capabilityId;
  document.querySelector("#evidence-explorer").open = true;
  renderResults();
  document.querySelector("#evidence-explorer").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderCapabilities() {
  document.querySelector("#capabilities").innerHTML = state.data.capabilities.map((cap, index) => {
    const operational = cap.benchmarks.filter(item => item.status === "operational").length;
    const targeted = cap.benchmarks.filter(item => item.target != null).length;
    const benchmarks = cap.benchmarks.map(item => {
      const target = item.target == null ? "no numeric target" : `target ${item.direction === "lower" ? "≤" : "≥"}${item.target}`;
      return `<li><span>${benchmarkNameLink(item)}<small>${escapeHtml(item.metric)} · ${escapeHtml(target)}</small>${benchmarkSourceLinks(item)}</span><span class="status ${escapeHtml(item.status)}">${escapeHtml(label(item.status))}</span></li>`;
    }).join("");
    return `<article class="capability-card"><span class="number">${String(index + 1).padStart(2, "0")}</span><h3>${escapeHtml(cap.name)}</h3><p>${escapeHtml(cap.question)}</p><div class="counts"><span>${operational} operational</span><span>${targeted} targets</span></div><details><summary>Evaluation contract</summary><ul>${benchmarks}</ul></details></article>`;
  }).join("");
}

function populateFilters() {
  const model = document.querySelector("#model-filter");
  state.data.models.forEach(item => model.insertAdjacentHTML("beforeend", `<option value="${escapeHtml(item.id)}">${escapeHtml(shortModel(item.id))}</option>`));
  const capability = document.querySelector("#capability-filter");
  state.data.capabilities.forEach(item => capability.insertAdjacentHTML("beforeend", `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`));
  const evidence = document.querySelector("#evidence-filter");
  [...new Set(state.data.runs.map(run => run.provenance.kind))].sort().forEach(item => evidence.insertAdjacentHTML("beforeend", `<option value="${escapeHtml(item)}">${escapeHtml(label(item))}</option>`));
  [[model,"model"],[capability,"capability"],[evidence,"evidence"]].forEach(([element, key]) => element.addEventListener("change", event => { state.filters[key] = event.target.value; renderResults(); }));
  document.querySelector("#search-filter").addEventListener("input", event => { state.filters.search = event.target.value.toLowerCase().trim(); renderResults(); });
}

function renderResults() {
  const filtered = flatten().filter(({run, metric}) => {
    if (state.filters.model && run.model.id !== state.filters.model) return false;
    if (state.filters.capability && metric.capability !== state.filters.capability) return false;
    if (state.filters.evidence && run.provenance.kind !== state.filters.evidence) return false;
    const haystack = [run.model.id, metric.capability, metric.benchmark, metric.metric, metric.language, metric.slice].join(" ").toLowerCase();
    return !state.filters.search || haystack.includes(state.filters.search);
  });
  document.querySelector("#result-count").textContent = `${filtered.length} of ${flatten().length} measurements`;
  document.querySelector("#results").innerHTML = filtered.length ? filtered.map(({run, metric}) => {
    const benchmark = lookupBenchmark(metric.capability, metric.benchmark);
    const status = metricTargetStatus(metric, benchmark);
    const target = status === "not-set" ? "—" : targetThreshold(benchmark);
    const gap = status === "not-set" ? null : targetGap(metric.value, benchmark);
    const slice = [metric.language ? `lang: ${metric.language}` : "", metric.slice, metric.n ? `n=${metric.n}` : ""].filter(Boolean).join(" · ") || "—";
    return `<tr><td><span class="model-name">${escapeHtml(shortModel(run.model.id))}</span><span class="model-revision">${escapeHtml(shortSha(run.model.revision))}</span></td><td><span class="cap-label">${escapeHtml(label(metric.capability))}</span><span class="benchmark">${benchmarkNameLink(benchmark, label(metric.benchmark))}</span>${benchmarkSourceLinks(benchmark)}<span class="metric">${escapeHtml(metric.metric)}</span></td><td>${escapeHtml(slice)}</td><td><span class="score">${escapeHtml(scoreText(metric))}</span></td><td>${targetBadge(status)}<small class="target-threshold">${escapeHtml(target)}</small>${targetGapBadge(gap, benchmark?.direction, metric.scale, benchmark)}</td><td>${evidenceCell(run)}</td></tr>`;
  }).join("") : `<tr><td class="empty" colspan="6">No evidence matches these filters.</td></tr>`;
}

async function init() {
  const response = await fetch("data/index.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load data: ${response.status}`);
  state.data = await response.json();
  renderScorecards(); renderSweepContract(); renderFastDiagnostics(); renderGate(); renderCapabilityComparison(); renderCapabilities(); populateFilters(); renderResults();
  document.querySelector("#score-method").innerHTML = `<strong>${escapeHtml(state.data.score_method.name)}:</strong> ${escapeHtml(state.data.score_method.description)} <span>Excluded: ${escapeHtml(state.data.score_method.exclusions.join(", "))}.</span>`;
  document.querySelector("#target-method").innerHTML = `<strong>${escapeHtml(state.data.target_policy.name)}:</strong> ${escapeHtml(state.data.target_policy.description)} <span>${escapeHtml(state.data.target_policy.gap_definition)}</span>`;
  document.querySelector("#generated").textContent = `Last built ${new Date(state.data.generated_at).toLocaleString()}.`;
}

init().catch(error => { document.querySelector("main").innerHTML = `<section class="panel"><h2>Dashboard unavailable</h2><p>${escapeHtml(error.message)}</p></section>`; });
