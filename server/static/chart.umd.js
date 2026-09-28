/* A tiny dependency-free canvas chart renderer. It keeps the dashboard fully offline. */
(function () {
  'use strict';
  // Colors come from the page's theme tokens so charts follow light/dark mode.
  const token = (name, fallback) => (getComputedStyle(document.documentElement).getPropertyValue(name) || '').trim() || fallback;
  function NanoChart(canvas, options) { this.canvas = canvas; this.options = options || {}; this.draw(); }
  NanoChart.prototype.draw = function () {
    const canvas = this.canvas, box = canvas.parentElement, dpr = window.devicePixelRatio || 1;
    if (!box.clientWidth) return; // hidden tab; the next refresh or tab switch redraws at the real size
    // Measure the container, not the canvas: the canvas keeps its last drawn size, so it would never grow.
    const width = Math.max(260, box.clientWidth), height = Math.max(90, box.clientHeight || 230);
    canvas.width = width * dpr; canvas.height = height * dpr; canvas.style.width = width + 'px'; canvas.style.height = height + 'px';
    const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, height);
    const pad = { l: 43, r: 16, t: 15, b: 29 }, plotW = width - pad.l - pad.r, plotH = height - pad.t - pad.b;
    const labels = this.options.labels || [], series = this.options.datasets || [];
    const grid = token('--chart-grid', '#d8ece2'), axis = token('--chart-axis', '#a0c8b4'), label = token('--chart-label', '#609878');
    const COLORS = ['--series-1', '--series-2', '--series-3', '--series-4'].map(name => token(name, label));
    let values = []; series.forEach(s => (s.data || []).forEach(v => { if (v != null && isFinite(v)) values.push(Number(v)); }));
    if (!values.length) { ctx.fillStyle = label; ctx.font = '11px DM Mono, monospace'; ctx.fillText('waiting for metric events', pad.l, height / 2); return; }
    let min = this.options.min != null ? this.options.min : Math.min(...values), max = this.options.max != null ? this.options.max : Math.max(...values);
    if (this.options.logScale) { values = values.filter(v => v > 0); min = Math.max(0.0001, Math.min(...values)); max = Math.max(...values); }
    if (this.options.zeroLine && !this.options.logScale) { min = Math.min(min, 0); max = Math.max(max, 0); }
    if (this.options.logScale) { min /= 1.15; max *= 1.15; }
    else if (!this.options.min) { const span = Math.max(0.0001, max - min); const floor = min - span * .08; min = min >= 0 ? Math.max(0, floor) : floor; max += span * .08; }
    if (max === min) { max += 1; min = Math.max(0, min - 1); }
    const x = i => pad.l + (labels.length <= 1 ? 0 : i / (labels.length - 1)) * plotW;
    const y = v => { if (this.options.logScale) { const a = Math.log(Math.max(min, v)), b = Math.log(max); return pad.t + (1 - (a - Math.log(min)) / Math.max(.0001, b - Math.log(min))) * plotH; } return pad.t + (1 - (v - min) / (max - min)) * plotH; };
    ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.font = '10px DM Mono, monospace'; ctx.fillStyle = label;
    for (let i = 0; i < 4; i++) { const yy = pad.t + (plotH * i / 3); ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(width - pad.r, yy); ctx.stroke(); const val = this.options.logScale ? Math.exp(Math.log(max) - (Math.log(max) - Math.log(min)) * i / 3) : max - (max - min) * i / 3; ctx.fillText(this.format(val), 2, yy + 3); }
    if (this.options.zeroLine && min < 0 && max > 0) { ctx.save(); ctx.strokeStyle = axis; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(pad.l, y(0)); ctx.lineTo(width - pad.r, y(0)); ctx.stroke(); ctx.restore(); }
    ctx.strokeStyle = axis; ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, height - pad.b); ctx.lineTo(width - pad.r, height - pad.b); ctx.stroke();
    const step = Math.max(1, Math.ceil(labels.length / 5)); ctx.fillStyle = label; for (let i = 0; i < labels.length; i += step) ctx.fillText(String(labels[i]), x(i) - 10, height - 8);
    series.forEach((s, si) => { const data = s.data || [], color = s.color || COLORS[si % COLORS.length]; ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.beginPath(); let started = false; data.forEach((value, i) => { if (value == null || !isFinite(value)) { started = false; return; } const xx = x(i), yy = y(Number(value)); if (!started) { ctx.moveTo(xx, yy); started = true; } else ctx.lineTo(xx, yy); }); ctx.stroke(); if (s.label) { const last = data.reduce((found, value, i) => value != null && isFinite(value) ? i : found, -1); if (last >= 0) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x(last), y(Number(data[last])), 3, 0, Math.PI * 2); ctx.fill(); ctx.font = '10px DM Mono, monospace'; ctx.textAlign = 'right'; ctx.fillText(s.label, Math.max(pad.l + ctx.measureText(s.label).width, x(last) - 6), Math.max(12, y(Number(data[last])) - 8)); ctx.textAlign = 'left'; } } });
  };
  NanoChart.prototype.format = function (v) { if (Math.abs(v) >= 1000) return (v / 1000).toFixed(1) + 'k'; if (Math.abs(v) < .01) return v.toExponential(1); return v.toFixed(v < 10 ? 2 : 0); };
  window.NanoChart = NanoChart;
})();

