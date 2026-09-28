(function () {
  'use strict';
  let savedPrimary = '';
  try { savedPrimary = window.localStorage.getItem('nanogym.primary') || ''; } catch (_) {}
  const state = { runs: [], primary: savedPrimary, playModel: '', compare: [], metrics: {}, smoothing: 0.2, socket: null, allowControl: false };
  const $ = id => document.getElementById(id);
  const fmt = (value, digits = 2) => value == null || Number.isNaN(Number(value)) ? '—' : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
  const fmtCompact = value => value == null ? '—' : Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(Number(value));
  const relativeTime = value => { if (!value) return 'no heartbeat yet'; const seconds = Math.max(0, Math.round(Date.now() / 1000 - Number(value))); return seconds < 2 ? 'heartbeat just now' : seconds < 60 ? `heartbeat ${seconds}s ago` : `heartbeat ${Math.floor(seconds / 60)}m ago`; };
  const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const seriesColors = () => ['--series-1', '--series-2', '--series-3', '--series-4'].map(cssVar);
  const escapeHtml = value => String(value).replace(/[&<>'"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[ch]));

  async function api(path, options) { const response = await fetch(path, options); if (!response.ok) { let message = response.statusText; try { message = (await response.json()).detail || message; } catch (_) {} throw new Error(message); } return response.json(); }

  async function loadControlOptions() {
    try {
      const [health, datasets] = await Promise.all([api('/api/health'), api('/api/datasets')]);
      state.allowControl = Boolean(health.allow_control);
      const dataset = $('experiment-dataset');
      dataset.innerHTML = datasets.map(item => `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)} · ${fmtCompact(item.size_bytes)}B</option>`).join('') || '<option value="">no .txt datasets found</option>';
      $('control-note').innerHTML = state.allowControl ? '' : 'Requires <code>serve.py --allow-control</code>';
      showControlStatus();
      renderCompute(health);
      [...$('experiment-device').options, ...$('resume-device').options].forEach(option => { if (option.value === 'cuda') option.disabled = !health.cuda_available; if (option.value === 'mps') option.disabled = !health.mps_available; });
      ['resume-run', 'stop-run', 'start-experiment', 'resume-device'].forEach(id => { $(id).disabled = !state.allowControl; });
    } catch (error) { $('control-note').textContent = 'control status unavailable'; $('compute-label').textContent = 'device unknown'; console.warn(error); }
  }

  // Header chip: which hardware PyTorch can train on here.
  function renderCompute(health) {
    const label = health.cuda_available ? `CUDA · ${health.cuda_device || 'GPU'}` : health.mps_available ? 'Apple MPS' : 'CPU only';
    const detail = health.cuda_available ? 'An NVIDIA GPU is available; runs on device "auto" or "cuda" train on it.' : health.mps_available ? 'Apple Silicon GPU (MPS) is available; runs on device "auto" or "mps" train on it.' : 'No GPU found: PyTorch is CPU-only here, so training runs on the CPU (fine for the tiny preset).';
    $('compute-label').textContent = label;
    $('compute-chip').title = detail;
    $('compute-chip').classList.toggle('cpu-only', !health.cuda_available && !health.mps_available);
  }

  // Only speak up when something needs attention: view-only mode, or a failed control request.
  function showControlStatus(error) {
    const status = $('control-status');
    const message = error || (state.allowControl ? '' : 'View-only mode: start the server with --allow-control (start-server.bat does this) to start, stop and resume runs from the dashboard. Pause still works.');
    status.textContent = message;
    status.classList.toggle('error', Boolean(error));
    status.classList.toggle('hidden', !message);
  }

  function setSelects() {
    const primary = $('primary-run'), compare = $('compare-runs'), play = $('play-model');
    const options = state.runs.map(run => `<option value="${escapeHtml(run.name)}">${escapeHtml(run.name)}</option>`).join('');
    primary.innerHTML = options || '<option value="">no runs</option>';
    compare.innerHTML = state.runs.map(run => `<option value="${escapeHtml(run.name)}">${escapeHtml(run.name)}</option>`).join('');
    const playable = state.runs.filter(run => run.has_best || run.has_latest);
    play.innerHTML = playable.map(run => `<option value="${escapeHtml(run.name)}">${escapeHtml(run.name)}</option>`).join('') || '<option value="">no checkpoints</option>';
    if (state.primary && state.runs.some(run => run.name === state.primary)) primary.value = state.primary; else state.primary = state.runs[0] ? state.runs[0].name : '';
    primary.value = state.primary;
    [...compare.options].forEach(option => { option.selected = state.compare.includes(option.value); });
    // The playground keeps its own selection; only default to the Training tab's primary run until the user picks one.
    if (!playable.some(run => run.name === state.playModel)) state.playModel = playable.some(run => run.name === state.primary) ? state.primary : (playable[0] ? playable[0].name : '');
    play.value = state.playModel;
    $('empty-state').classList.toggle('hidden', state.runs.length > 0);
  }

  function selectedNames() { return [state.primary].concat(state.compare.filter(name => name && name !== state.primary)).filter((name, index, values) => values.indexOf(name) === index); }

  function metricRows(name) { return state.metrics[name] || []; }
  function ema(values, alpha) { let result = [], previous = null; values.forEach(value => { if (value == null) { result.push(null); return; } previous = previous == null ? value : alpha * previous + (1 - alpha) * value; result.push(previous); }); return result; }
  function series(name, field, split) {
    const rows = metricRows(name).filter(row => !split || row.split === split);
    const byStep = new Map(rows.map(row => [Number(row.step), Number(row[field])]));
    return { steps: [...byStep.keys()].sort((a, b) => a - b), values: [...byStep.keys()].sort((a, b) => a - b).map(key => byStep.get(key)) };
  }
  function mergedChartData(field, split, smooth) {
    const names = selectedNames(), allSteps = [...new Set(names.flatMap(name => series(name, field, split).steps))].sort((a, b) => a - b);
    return { labels: allSteps, datasets: names.map((name, index) => { const values = metricRows(name).filter(row => !split || row.split === split); const map = new Map(values.map(row => [Number(row.step), row[field] == null ? null : Number(row[field])])); const points = allSteps.map(step => map.get(step) ?? null); return { label: name, data: smooth ? ema(points, smooth) : points, color: seriesColors()[index % 4] }; }) };
  }

  function renderCharts() {
    const names = selectedNames(); if (!names.length) return;
    const loss = mergedChartData('loss', 'train', state.smoothing); const val = mergedChartData('loss', 'val', state.smoothing);
    const labels = [...new Set(loss.labels.concat(val.labels))].sort((a, b) => a - b);
    const palette = seriesColors(); const lossSets = names.flatMap((name, index) => { const color = palette[index % 4]; const trainMap = new Map(metricRows(name).filter(row => row.split === 'train').map(row => [Number(row.step), Number(row.loss)])); const valMap = new Map(metricRows(name).filter(row => row.split === 'val').map(row => [Number(row.step), Number(row.loss)])); return [{ label: name + ' train', data: ema(labels.map(step => trainMap.get(step) ?? null), state.smoothing), color }, { label: name + ' val', data: ema(labels.map(step => valMap.get(step) ?? null), state.smoothing), color: color + '99' }]; });
    new NanoChart($('loss-chart'), { labels, datasets: lossSets, logScale: $('log-scale').checked });
    new NanoChart($('bpc-chart'), { labels: mergedChartData('bpc', 'val', state.smoothing).labels, datasets: mergedChartData('bpc', 'val', state.smoothing).datasets });
    const signalNames = names[0], rows = metricRows(signalNames).filter(row => row.split === 'train');
    const signalLabels = rows.map(row => Number(row.step)), signalData = field => rows.map(row => row[field] == null ? null : Number(row[field]));
    // Throughput (~1e5) and grad norm (~1) get separate charts; on one axis grad norm flattens to a line at zero.
    new NanoChart($('throughput-chart'), { labels: signalLabels, datasets: [{ label: 'tokens/sec', data: signalData('tok_per_sec'), color: cssVar('--series-2') }] });
    new NanoChart($('gradnorm-chart'), { labels: signalLabels, datasets: [{ label: 'grad norm', data: signalData('grad_norm'), color: cssVar('--series-1') }], logScale: true }); // log: early spikes would flatten the rest
    const gapRows = metricRows(signalNames).filter(row => row.split === 'val');
    new NanoChart($('gap-chart'), { labels: gapRows.map(row => Number(row.step)), datasets: [{ label: 'val − train', data: gapRows.map(row => row.train_val_gap == null ? null : Number(row.train_val_gap)), color: cssVar('--series-4') }], zeroLine: true });
  }

  function renderStatus() {
    const run = state.runs.find(item => item.name === state.primary); if (!run) { $('activity-banner').className = 'activity-banner'; $('activity-state').textContent = 'waiting for a run'; $('activity-detail').textContent = 'Select a run to see live training activity.'; $('activity-time').textContent = '—'; return; }
    $('status-state').textContent = run.state || 'unknown'; $('status-state').style.color = run.state === 'running' ? 'var(--accent)' : 'var(--text-strong)';
    $('status-device').textContent = `${run.device || 'device pending'} · ${run.parameters ? fmtCompact(run.parameters) + ' params' : 'no parameter count yet'}`;
    $('status-step').textContent = `${fmtCompact(run.step)} / ${fmtCompact(run.max_steps)}`; $('status-progress').textContent = run.max_steps ? `${Math.round((run.step / run.max_steps) * 100)}% complete` : 'step / max steps';
    $('status-loss').textContent = fmt(run.best_val_loss, 3); $('status-tokens').textContent = fmtCompact(run.tokens_seen);
    $('metric-count').textContent = `${fmt(state.metrics[state.primary] ? state.metrics[state.primary].length : 0, 0)} metric events`;
    renderActivity(run);
    // The explainer is for people browsing samples after the fact; hide it while a run is actively training.
    $('samples-intro').classList.toggle('hidden', ['running', 'starting'].includes(String(run.state || '').toLowerCase()));
  }

  function renderActivity(run) {
    const status = String(run.state || 'unknown').toLowerCase();
    const bannerClass = ['starting', 'running', 'finished', 'paused', 'stopped', 'failed'].includes(status) ? status : '';
    $('activity-banner').className = `activity-banner ${bannerClass} ${status === 'running' ? 'active' : ''}`;
    const labels = { starting: 'training is starting', running: 'training is live', paused: 'training is paused', stopped: 'training stopped', finished: 'training finished', failed: 'training failed', created: 'run created' };
    $('activity-state').textContent = labels[status] || status;
    $('activity-detail').textContent = status === 'failed' && run.error ? run.error : `${fmtCompact(run.step || 0)} / ${fmtCompact(run.max_steps)} steps · ${run.device || 'waiting for device'}${run.pid ? ` · pid ${run.pid}` : ''}`;
    $('activity-time').textContent = relativeTime(run.last_activity_at || run.updated_at);
  }

  async function loadMetrics(name) { try { const result = await api(`/api/runs/${encodeURIComponent(name)}/metrics?limit=20000`); state.metrics[name] = result.rows; } catch (error) { console.warn(error); } }
  async function loadActivity(name) { if (!name) return; try { const result = await api(`/api/runs/${encodeURIComponent(name)}/activity`); const run = state.runs.find(item => item.name === name); if (run && result.status && Object.keys(result.status).length) { Object.assign(run, result.status); renderStatus(); } const lines = (result.process_log || []).concat(result.events || []).slice(-14); $('activity-log-text').textContent = lines.length ? lines.join('\n') : 'No trainer output yet.'; } catch (error) { console.warn(error); } }
  async function refreshRuns() {
    try { state.runs = await api('/api/runs'); setSelects(); await Promise.all(selectedNames().map(loadMetrics)); renderStatus(); renderCharts(); await renderSamples(); await loadActivity(state.primary); $('last-sync').textContent = 'synced ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); } catch (error) { $('last-sync').textContent = 'server unavailable'; console.warn(error); }
  }

  async function renderSamples() {
    const list = $('samples-list'); if (!state.primary) { list.innerHTML = '<div class="empty-state">Select a run with sample events to browse its writing.</div>'; return; }
    try { const samples = await api(`/api/runs/${encodeURIComponent(state.primary)}/samples?limit=200`); if (!samples.length) { list.innerHTML = '<div class="empty-state">No samples yet. They appear at the configured sample interval.</div>'; return; } list.innerHTML = samples.slice().reverse().map(sample => `<article class="sample-card"><header><strong>step ${escapeHtml(fmtCompact(sample.step))}</strong><span>prompt: ${escapeHtml(sample.prompt || '∅')}</span></header><pre>${escapeHtml(sample.text || '')}</pre></article>`).join(''); } catch (error) { list.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`; }
  }

  async function control(command) { if (!state.primary) return; const body = { command }; if (command === 'resume') { const device = $('resume-device').value; if (device) body.device = device; } try { await api(`/api/runs/${encodeURIComponent(state.primary)}/control`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); await refreshRuns(); } catch (error) { showControlStatus(error.message); } }

  function openCloneForm() {
    const run = state.runs.find(item => item.name === state.primary);
    if (!run || !run.config || run.config.error) { alert(run ? 'This run has no readable config to clone.' : 'Select a run first.'); return; }
    const cfg = run.config, model = cfg.model || {}, train = cfg.train || {}, data = cfg.data || {};
    const setValue = (id, value) => { if (value != null) $(id).value = value; };
    setValue('experiment-name', `${run.name}-v2`);
    const path = String(data.path || '').replace(/\\/g, '/');
    const match = [...$('experiment-dataset').options].find(option => option.value && (path === option.value || path.endsWith('/' + option.value)));
    if (match) $('experiment-dataset').value = match.value;
    setValue('experiment-preset', model.preset); setValue('experiment-block', model.block_size); setValue('experiment-dropout', model.dropout);
    setValue('experiment-steps', train.max_steps); setValue('experiment-batch', train.batch_size); setValue('experiment-lr', train.lr); setValue('experiment-seed', train.seed);
    const device = $('experiment-device'), wanted = [...device.options].find(option => option.value === train.device && !option.disabled);
    device.value = wanted ? train.device : 'auto';
    setValue('experiment-val', data.val_fraction); $('experiment-extend-vocab').checked = Boolean((data.vocab || {}).extend_from_data);
    const form = $('experiment-form'); form.classList.remove('hidden'); $('experiment-feedback').className = 'builder-feedback'; $('experiment-feedback').textContent = `cloned settings from ${run.name} · rename and launch a fresh run`;
    $('experiment-name').focus(); $('experiment-name').select();
  }

  async function startExperiment(event) {
    event.preventDefault();
    const feedback = $('experiment-feedback'), button = $('start-experiment');
    const payload = { name: $('experiment-name').value.trim(), dataset: $('experiment-dataset').value, preset: $('experiment-preset').value, block_size: Number($('experiment-block').value), max_steps: Number($('experiment-steps').value), batch_size: Number($('experiment-batch').value), learning_rate: Number($('experiment-lr').value), dropout: Number($('experiment-dropout').value), val_fraction: Number($('experiment-val').value), seed: Number($('experiment-seed').value), device: $('experiment-device').value, extend_vocab: $('experiment-extend-vocab').checked };
    feedback.className = 'builder-feedback'; feedback.textContent = 'launching…'; button.disabled = true;
    try { const result = await api('/api/experiments', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }); state.primary = result.name; try { window.localStorage.setItem('nanogym.primary', state.primary); } catch (_) {} state.compare = []; feedback.className = 'builder-feedback success'; feedback.textContent = `started ${result.name}`; $('experiment-form').classList.add('hidden'); await refreshRuns(); connectSocket(); } catch (error) { feedback.className = 'builder-feedback error'; feedback.textContent = error.message; } finally { button.disabled = !state.allowControl; }
  }

  function connectSocket() {
    if (!state.primary || !window.WebSocket) return;
    if (state.socket) state.socket.close();
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const since = (state.metrics[state.primary] || []).length;
    state.socket = new WebSocket(`${protocol}//${location.host}/ws/runs/${encodeURIComponent(state.primary)}/metrics?since=${since}`);
    state.socket.onmessage = event => { const row = JSON.parse(event.data); state.metrics[state.primary] = (state.metrics[state.primary] || []).concat(row); renderStatus(); renderCharts(); };
    state.socket.onclose = () => { if (state.primary) setTimeout(connectSocket, 4000); };
  }

  function generateRequest() { return { model: state.playModel, prompt: $('prompt').value, checkpoint: 'best', temperature: Number($('temperature').value), top_k: Number($('top-k').value), top_p: Number($('top-p').value), max_new_chars: Number($('max-chars').value), seed: Number($('seed').value) || 1337, honor_tags: $('honor-tags').checked }; }
  function generate() {
    const request = generateRequest(), output = $('generation-output'), status = $('generation-status'), button = $('generate'); if (button.disabled) return; output.textContent = ''; status.textContent = 'generating'; status.classList.add('live'); button.disabled = true;
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'; const socket = new WebSocket(`${protocol}//${location.host}/ws/generate`);
    // Re-enable on every way the socket can end (done, error, dropped connection).
    socket.onclose = () => { button.disabled = false; status.classList.remove('live'); if (status.textContent === 'generating') status.textContent = 'stopped'; };
    socket.onopen = () => socket.send(JSON.stringify(request)); socket.onmessage = event => { const message = JSON.parse(event.data); if (message.type === 'token') output.textContent += message.text; if (message.type === 'done') { button.disabled = false; output.textContent = message.text; status.textContent = 'complete'; status.classList.remove('live'); loadConfidence(request); if ($('random-seed').checked) $('seed').value = Math.floor(Math.random() * 1e9); } if (message.type === 'error') { button.disabled = false; output.textContent = message.error; status.textContent = 'error'; status.classList.remove('live'); } output.scrollTop = output.scrollHeight; }; socket.onerror = () => { button.disabled = false; status.textContent = 'error'; status.classList.remove('live'); output.textContent = 'Could not connect to the generation socket.'; };
  }

  async function exportModel() {
    const name = state.playModel, button = $('export-model'), status = $('export-status');
    if (!name || button.disabled) return;
    button.disabled = true; status.className = 'export-status'; status.textContent = `exporting ${name}…`;
    try {
      const response = await fetch(`/api/runs/${encodeURIComponent(name)}/export`, { method: 'POST' });
      if (!response.ok) { let message = response.statusText; try { message = (await response.json()).detail || message; } catch (_) {} throw new Error(message); }
      const url = URL.createObjectURL(await response.blob()), link = document.createElement('a');
      link.href = url; link.download = `${name}-web.zip`; document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      status.className = 'export-status success';
      status.textContent = `Saved to ${response.headers.get('X-Export-Path') || 'exports/'} (${fmt(Number(response.headers.get('X-Export-Bytes')) / 1e6, 1)} MB) and downloaded as a zip. Serve the folder with any static web server; its README explains how.`;
    } catch (error) { status.className = 'export-status error'; status.textContent = `Export failed: ${error.message}`; } finally { button.disabled = false; }
  }

  async function loadConfidence(request) { try { const result = await api('/api/generate/details', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) }); const bar = $('confidence-bar'); bar.innerHTML = result.tokens.map(item => { const probability = Math.max(0, Math.min(1, Number(item.probability))); const alpha = (.1 + probability * .55).toFixed(2); return `<span class="confidence-char" title="${escapeHtml((probability * 100).toFixed(1) + '% likely')}" style="background:rgba(var(--accent-rgb),${alpha})">${escapeHtml(item.char === '\n' ? '↵' : item.char)}</span>`; }).join(''); $('confidence-label').textContent = `${result.tokens.length} generated characters · hover for confidence`; } catch (error) { $('confidence-bar').textContent = error.message; } }

  async function inspectAttention() {
    const heatmap = $('attention-heatmap'); heatmap.className = 'heatmap'; heatmap.textContent = 'loading attention…';
    try { const result = await api('/api/attention', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(generateRequest()) }); renderHeatmap(result); } catch (error) { heatmap.className = 'heatmap empty-heatmap'; heatmap.textContent = error.message; }
  }
  function renderHeatmap(result) {
    const heads = result.heads || [], tokens = result.tokens || []; if (!heads.length || !tokens.length) { $('attention-heatmap').textContent = 'No attention weights returned.'; return; }
    const head = Math.min(0, heads.length - 1), matrix = heads[head], max = Math.max(...matrix.flat()); let html = `<div class="heatmap-grid" style="grid-template-columns:repeat(${tokens.length}, 16px)">`;
    matrix.forEach(row => row.forEach(value => { const alpha = Math.max(.05, Number(value) / Math.max(.0001, max)); html += `<span class="heat-cell" title="${escapeHtml(Number(value).toFixed(3))}" style="background:rgba(var(--accent-rgb),${alpha})"></span>`; })); html += '</div><div class="heatmap-grid" style="grid-template-columns:repeat(' + tokens.length + ', 16px)">'; tokens.forEach(token => { const shown = token === '\n' ? '↵' : token.length > 1 ? '•' : token; html += `<span class="heat-label" title="${escapeHtml(token)}">${escapeHtml(shown)}</span>`; }); /* special tokens like <bos> don't fit a 16px cell */ html += '</div>'; $('attention-heatmap').innerHTML = html; $('attention-label').textContent = `head 1 / ${heads.length} · ${tokens.length} tokens`; }

  function savedTheme() { try { const theme = window.localStorage.getItem('nanogym.theme'); return theme === 'light' || theme === 'dark' ? theme : null; } catch (_) { return null; } }
  function applyTheme(theme) { document.documentElement.dataset.theme = theme; $('theme-label').textContent = theme === 'dark' ? 'light' : 'dark'; renderCharts(); }
  function toggleTheme() { const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'; try { window.localStorage.setItem('nanogym.theme', next); } catch (_) {} applyTheme(next); }

  // With tags honored, a bare <bos> prompt asks for a complete entry from the seed alone.
  function syncPromptWithTags() { const prompt = $('prompt'); if ($('honor-tags').checked) { if (!prompt.value.trim()) prompt.value = '<bos>'; } else if (prompt.value.trim() === '<bos>') prompt.value = ''; }

  function showTab(name) { document.querySelectorAll('.tab,.tab-panel').forEach(element => element.classList.remove('active')); document.querySelector(`.tab[data-tab="${name}"]`).classList.add('active'); $('tab-' + name).classList.add('active'); window.scrollTo(0, 0); if (name === 'training') renderCharts(); }

  function bind() {
    document.querySelectorAll('.tab').forEach(button => button.addEventListener('click', () => showTab(button.dataset.tab)));
    document.querySelectorAll('[data-goto-tab]').forEach(link => link.addEventListener('click', () => showTab(link.dataset.gotoTab)));
    $('primary-run').addEventListener('change', async event => { state.primary = event.target.value; try { window.localStorage.setItem('nanogym.primary', state.primary); } catch (_) {} showControlStatus(); state.compare = []; setSelects(); await loadMetrics(state.primary); renderStatus(); renderCharts(); renderSamples(); connectSocket(); });
    $('play-model').addEventListener('change', event => { state.playModel = event.target.value; });
    $('compare-runs').addEventListener('change', event => { state.compare = [...event.target.selectedOptions].map(option => option.value); renderCharts(); });
    $('pause-run').addEventListener('click', () => control('pause')); $('stop-run').addEventListener('click', () => control('stop')); $('resume-run').addEventListener('click', () => control('resume')); $('new-experiment-button').addEventListener('click', () => $('experiment-form').classList.toggle('hidden')); $('clone-experiment-button').addEventListener('click', openCloneForm); $('experiment-form').addEventListener('submit', startExperiment); $('generate').addEventListener('click', generate); $('inspect-attention').addEventListener('click', inspectAttention); $('export-model').addEventListener('click', exportModel); $('log-scale').addEventListener('change', renderCharts);
    $('smoothing').addEventListener('input', event => { state.smoothing = Number(event.target.value); $('smoothing-value').textContent = state.smoothing.toFixed(1); renderCharts(); });
    [['temperature', 'temperature-value'], ['max-chars', 'max-chars-value'], ['top-k', 'top-k-value'], ['top-p', 'top-p-value']].forEach(([input, output]) => $(input).addEventListener('input', event => $(output).textContent = event.target.value));
    window.addEventListener('resize', renderCharts);
    $('theme-toggle').addEventListener('click', toggleTheme);
    if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => { if (!savedTheme()) applyTheme(event.matches ? 'dark' : 'light'); });
    $('honor-tags').addEventListener('change', syncPromptWithTags);
  }
  bind(); applyTheme(document.documentElement.dataset.theme || 'light'); loadControlOptions(); refreshRuns().then(connectSocket); setInterval(refreshRuns, 3500); setInterval(() => loadActivity(state.primary), 1000);
})();
