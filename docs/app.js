const state = { data: null, filters: { model: "", capability: "", evidence: "", search: "" } };

const escapeHtml = value => String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]);
const shortModel = id => id.split("/").pop();
const shortSha = sha => sha ? sha.slice(0, 9) : "unpinned";
const label = value => String(value || "").replaceAll("-", " ").replace(/\b\w/g, c => c.toUpperCase());

function lookupBenchmark(capability, benchmark) {
  const cap = state.data.capabilities.find(item => item.id === capability);
  return cap?.benchmarks.find(item => item.id === benchmark);
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

function renderSummary() {
  const evidence = new Set(state.data.runs.map(run => run.provenance.kind));
  const fresh = state.data.runs.filter(run => run.provenance.kind === "fresh-reproduced").length;
  const values = [
    [state.data.models.length, "models registered"],
    [state.data.runs.length, "versioned evidence runs"],
    [flatten().length, "individual measurements"],
    [fresh, `fresh reproduced run${fresh === 1 ? "" : "s"} · ${evidence.size} evidence types`]
  ];
  document.querySelector("#summary").innerHTML = values.map(([value, text]) => `<div class="stat"><strong>${value}</strong><span>${escapeHtml(text)}</span></div>`).join("");
}

function renderGate() {
  const gate = state.data.gates?.at(-1);
  const target = document.querySelector("#gate-summary");
  if (!gate) { target.hidden = true; return; }
  const status = gate.status === "passed" ? "PASS" : "FAIL";
  const details = gate.regressions.length
    ? gate.regressions.map(item => `${label(item.key[1])} ${item.key[2]}: ${item.improvement.toFixed(2)} pts`).join(" · ")
    : "No comparable metric exceeded the regression tolerance.";
  target.innerHTML = `<div><p class="eyebrow">LATEST PARENT RETENTION GATE</p><h2><span class="gate-status ${gate.status}">${status}</span> ${escapeHtml(shortModel(gate.candidate_run.replace(/^published-/, "")))}</h2></div><p>${escapeHtml(gate.comparison_count)} matched measurements · tolerance ${escapeHtml(gate.max_regression_points)} points<br><strong>${escapeHtml(details)}</strong></p>`;
}

function renderCoverage() {
  const caps = state.data.capabilities;
  const head = `<div class="coverage-row coverage-head"><div class="coverage-cell">Model / capability</div>${caps.map(cap => `<div class="coverage-cell">${escapeHtml(cap.name)}</div>`).join("")}</div>`;
  const rows = state.data.models.map(model => {
    const cells = caps.map(cap => `<div class="coverage-cell" title="${model.capabilities.includes(cap.id) ? "Evidence present" : "No evidence"}"><span class="dot ${model.capabilities.includes(cap.id) ? "" : "missing"}"></span></div>`).join("");
    return `<div class="coverage-row"><div class="coverage-cell">${escapeHtml(shortModel(model.id))}<span class="model-revision">${model.metric_count} metrics</span></div>${cells}</div>`;
  }).join("");
  document.querySelector("#coverage").innerHTML = head + rows;
}

function renderCapabilities() {
  document.querySelector("#capabilities").innerHTML = state.data.capabilities.map((cap, index) => {
    const operational = cap.benchmarks.filter(item => item.status === "operational").length;
    return `<article class="capability-card"><span class="number">${String(index + 1).padStart(2, "0")}</span><h3>${escapeHtml(cap.name)}</h3><p>${escapeHtml(cap.question)}</p><div class="counts"><span>${cap.benchmarks.length} benchmarks</span><span>${operational} operational</span></div></article>`;
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
    const target = metric.metric === benchmark?.metric ? benchmark?.target : undefined;
    const targetText = target === undefined ? "—" : `${benchmark.direction === "lower" ? "≤" : "≥"} ${target}${metric.scale === "percentage" ? "%" : ""}`;
    const slice = [metric.language ? `lang: ${metric.language}` : "", metric.slice, metric.n ? `n=${metric.n}` : ""].filter(Boolean).join(" · ") || "—";
    return `<tr><td><span class="model-name">${escapeHtml(shortModel(run.model.id))}</span><span class="model-revision">${escapeHtml(shortSha(run.model.revision))}</span></td><td><span class="cap-label">${escapeHtml(label(metric.capability))}</span><span class="benchmark">${escapeHtml(benchmark?.name || label(metric.benchmark))}</span><span class="metric">${escapeHtml(metric.metric)}</span></td><td>${escapeHtml(slice)}</td><td><span class="score">${escapeHtml(scoreText(metric))}</span></td><td>${escapeHtml(targetText)}</td><td><span class="pill ${escapeHtml(run.provenance.kind)}">${escapeHtml(label(run.provenance.kind))}</span><a class="source" href="${escapeHtml(run.provenance.source)}">source ↗</a></td></tr>`;
  }).join("") : `<tr><td class="empty" colspan="6">No evidence matches these filters.</td></tr>`;
}

async function init() {
  const response = await fetch("data/index.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load data: ${response.status}`);
  state.data = await response.json();
  renderSummary(); renderGate(); renderCoverage(); renderCapabilities(); populateFilters(); renderResults();
  document.querySelector("#generated").textContent = `Last built ${new Date(state.data.generated_at).toLocaleString()}.`;
}

init().catch(error => { document.querySelector("main").innerHTML = `<section class="panel"><h2>Dashboard unavailable</h2><p>${escapeHtml(error.message)}</p></section>`; });
