/* ─── main.js: FraudGuard AI Frontend Logic ─── */

'use strict';

// ─── Colour Palette ────────────────────────────────────────────────────────
const COLORS = {
  cyan:   '#00f2fe',
  violet: '#8b5cf6',
  red:    '#FF416C',
  green:  '#00d9a5',
  orange: '#f59e0b',
  white:  '#f0f4ff',
  muted:  '#4a5568',
};
const MODEL_COLORS = {
  'Logistic Regression': COLORS.cyan,
  'Decision Tree':       COLORS.violet,
  'KNN':                 COLORS.orange,
};

// ─── Global State ──────────────────────────────────────────────────────────
let perfData = null;
let sampleData = null;
let currentFeatures = {};
let gaugeAnimFrame = null;
let gaugeTargetAngle = -Math.PI * 0.8;
let gaugeCurrentAngle = -Math.PI * 0.8;
let charts = {};

// ─── Utility Helpers ───────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const fmt = n => n >= 1e6 ? (n/1e6).toFixed(2)+'M' : n >= 1e3 ? (n/1e3).toFixed(1)+'K' : String(n);
const pct = n => (n * 100).toFixed(1) + '%';
const fmtProb = n => (n * 100).toFixed(2) + '%';

function glow(color, alpha = 0.4) {
  const hex = color.replace('#', '');
  const r = parseInt(hex.slice(0,2),16);
  const g = parseInt(hex.slice(2,4),16);
  const b = parseInt(hex.slice(4,6),16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ─── Navigation ────────────────────────────────────────────────────────────
const SECTIONS = ['dashboard', 'eda', 'performance', 'predictor'];
const BREADCRUMBS = {
  dashboard:   'Dashboard Overview',
  eda:         'Data Analysis',
  performance: 'Model Performance',
  predictor:   'Fraud Predictor',
};

function activateSection(name) {
  SECTIONS.forEach(s => {
    $(`section-${s}`).classList.toggle('hidden', s !== name);
    $(`nav-${s}`).classList.toggle('active', s === name);
  });
  $('breadcrumb').textContent = BREADCRUMBS[name];

  // Lazy-render charts when their section becomes visible
  if (name === 'performance' && perfData && !charts.roc) renderPerformanceCharts();
  if (name === 'dashboard'   && perfData && !charts.imbalance) renderDashboardCharts();
}

document.querySelectorAll('.nav-item').forEach(el => {
  el.addEventListener('click', e => {
    e.preventDefault();
    activateSection(el.dataset.section);
    if (window.innerWidth <= 900) $('sidebar').classList.remove('open');
  });
});

$('hamburger').addEventListener('click', () => {
  $('sidebar').classList.toggle('open');
});

// ─── Chart.js Global Defaults ─────────────────────────────────────────────
Chart.defaults.color = COLORS.muted;
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;

// ─── Bootstrap ────────────────────────────────────────────────────────────
(async function init() {
  const [statsRes, perfRes, sampleRes] = await Promise.all([
    fetch('/api/dataset-stats'),
    fetch('/api/model-performance'),
    fetch('/api/sample-transactions'),
  ]);
  const stats  = await statsRes.json();
  perfData     = await perfRes.json();
  sampleData   = await sampleRes.json();

  populateDatasetStats(stats);
  populateModelCardsDashboard(perfData);
  populatePresetDropdown(sampleData);
  buildPCASliders();
  renderDashboardCharts();
  renderPerformanceSection(perfData);
})();

// ─── Dashboard: Dataset Stats ─────────────────────────────────────────────
function populateDatasetStats(s) {
  $('kpi-total-val').textContent   = fmt(s.total);
  $('kpi-normal-val').textContent  = fmt(s.normal);
  $('kpi-fraud-val').textContent   = fmt(s.fraud);
  $('kpi-ratio-val').textContent   = s.fraud_pct + '%';
  $('sidebar-total').textContent   = fmt(s.total);
  $('info-records').textContent    = s.total.toLocaleString();
  $('info-duration').textContent   = s.time_range_hrs + ' hours';
  $('info-avg-amount').textContent = '$' + s.amount_mean.toFixed(2);
  $('info-max-amount').textContent = '$' + s.amount_max.toLocaleString();
}

// ─── Dashboard: Model Overview Cards ─────────────────────────────────────
function populateModelCardsDashboard(data) {
  const grid = $('model-cards-grid');
  grid.innerHTML = '';
  Object.entries(data).forEach(([name, d]) => {
    const color = MODEL_COLORS[name] || COLORS.cyan;
    const aucPct = (d.roc_auc * 100).toFixed(2);
    const card = document.createElement('div');
    card.className = 'model-card';
    card.innerHTML = `
      <div class="model-card-name">${name}</div>
      <div class="model-card-auc" style="color:${color}">${aucPct}<span style="font-size:1rem;font-weight:400;color:var(--text-muted)">%</span></div>
      <div class="model-card-sub">ROC-AUC Score</div>
      <div class="model-card-metrics">
        <div class="mcm-item">
          <div class="mcm-val" style="color:${color}">${pct(d.precision)}</div>
          <div class="mcm-key">Precision</div>
        </div>
        <div class="mcm-item">
          <div class="mcm-val" style="color:${color}">${pct(d.recall)}</div>
          <div class="mcm-key">Recall</div>
        </div>
        <div class="mcm-item">
          <div class="mcm-val" style="color:${color}">${pct(d.f1_score)}</div>
          <div class="mcm-key">F1</div>
        </div>
      </div>
      <div class="model-card-bar" style="background:linear-gradient(90deg,${color},${glow(color,0.2)});width:${aucPct}%"></div>
    `;
    grid.appendChild(card);
  });
}

// ─── Dashboard Charts ────────────────────────────────────────────────────
function renderDashboardCharts() {
  if (!perfData) return;
  const first = Object.values(perfData)[0];
  const cm = first.confusion_matrix;
  const normal = (cm[0][0] + cm[0][1]);
  const fraud  = (cm[1][0] + cm[1][1]);

  if (charts.imbalance) charts.imbalance.destroy();
  charts.imbalance = new Chart($('chart-imbalance'), {
    type: 'doughnut',
    data: {
      labels: ['Normal', 'Fraud'],
      datasets: [{
        data: [normal, fraud],
        backgroundColor: [glow(COLORS.cyan, 0.6), glow(COLORS.red, 0.7)],
        borderColor:     [COLORS.cyan, COLORS.red],
        borderWidth: 2,
        hoverOffset: 6,
      }]
    },
    options: {
      responsive: true,
      cutout: '72%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: COLORS.muted, padding: 16, usePointStyle: true, pointStyleWidth: 8 }
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.raw.toLocaleString()} (${(ctx.raw/(normal+fraud)*100).toFixed(3)}%)`
          }
        }
      }
    }
  });
}

// ─── Performance Section ─────────────────────────────────────────────────
function renderPerformanceSection(data) {
  renderPerfCards(data);
  renderConfusionMatrices(data);
  renderPerformanceCharts();
}

function renderPerfCards(data) {
  const grid = $('perf-model-grid');
  grid.innerHTML = '';
  Object.entries(data).forEach(([name, d]) => {
    const color = MODEL_COLORS[name] || COLORS.cyan;
    const metrics = [
      { key: 'Accuracy',  val: d.accuracy  },
      { key: 'Precision', val: d.precision },
      { key: 'Recall',    val: d.recall    },
      { key: 'F1 Score',  val: d.f1_score  },
      { key: 'ROC-AUC',   val: d.roc_auc   },
    ];
    const card = document.createElement('div');
    card.className = 'perf-card';
    card.innerHTML = `
      <div class="perf-card-name" style="color:${color}">${name}</div>
      ${metrics.map(m => `
        <div class="perf-metric-row">
          <span class="perf-metric-label">${m.key}</span>
          <span class="perf-metric-val">${(m.val*100).toFixed(2)}%</span>
        </div>
        <div class="perf-bar-track">
          <div class="perf-bar-fill" style="width:${m.val*100}%;background:linear-gradient(90deg,${color},${glow(color,0.3)})"></div>
        </div>
      `).join('')}
    `;
    grid.appendChild(card);
  });
}

function renderConfusionMatrices(data) {
  const grid = $('cm-grid');
  grid.innerHTML = '';
  Object.entries(data).forEach(([name, d]) => {
    const cm = d.confusion_matrix;
    const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
    const card = document.createElement('div');
    card.className = 'cm-card';
    card.innerHTML = `
      <div class="cm-card-name">${name}</div>
      <div class="cm-axis-row"><span>Predicted Normal</span><span>Predicted Fraud</span></div>
      <div class="cm-matrix">
        <div class="cm-cell cm-tn">
          <span class="cm-cell-val">${tn.toLocaleString()}</span>
          <span class="cm-cell-label">True Negative</span>
        </div>
        <div class="cm-cell cm-fp">
          <span class="cm-cell-val">${fp.toLocaleString()}</span>
          <span class="cm-cell-label">False Positive</span>
        </div>
        <div class="cm-cell cm-fn">
          <span class="cm-cell-val">${fn.toLocaleString()}</span>
          <span class="cm-cell-label">False Negative</span>
        </div>
        <div class="cm-cell cm-tp">
          <span class="cm-cell-val">${tp.toLocaleString()}</span>
          <span class="cm-cell-label">True Positive</span>
        </div>
      </div>
      <div class="cm-axis-label">↑ Actual Normal (top) · Actual Fraud (bottom)</div>
    `;
    grid.appendChild(card);
  });
}

function renderPerformanceCharts() {
  if (!perfData) return;

  // ── ROC Curve ──
  if (charts.roc) charts.roc.destroy();
  charts.roc = new Chart($('chart-roc'), {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Random Classifier',
          data: [{x:0,y:0},{x:1,y:1}],
          borderColor: COLORS.muted,
          borderWidth: 1,
          borderDash: [5,5],
          pointRadius: 0,
          fill: false,
        },
        ...Object.entries(perfData).map(([name, d]) => ({
          label: `${name} (AUC=${(d.roc_auc*100).toFixed(1)}%)`,
          data: d.roc_curve.fpr.map((x, i) => ({ x, y: d.roc_curve.tpr[i] })),
          borderColor: MODEL_COLORS[name],
          backgroundColor: 'transparent',
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
          shadowColor: MODEL_COLORS[name],
        }))
      ]
    },
    options: {
      responsive: true,
      parsing: false,
      scales: {
        x: { type:'linear', min:0, max:1, title:{ display:true, text:'False Positive Rate', color:COLORS.muted }, grid:{ color:'rgba(255,255,255,0.04)' }, ticks:{ color:COLORS.muted } },
        y: { type:'linear', min:0, max:1, title:{ display:true, text:'True Positive Rate', color:COLORS.muted }, grid:{ color:'rgba(255,255,255,0.04)' }, ticks:{ color:COLORS.muted } },
      },
      plugins: { legend: { labels: { color: COLORS.muted, usePointStyle:true, pointStyleWidth:8 } } },
      interaction: { mode:'nearest', intersect:false }
    }
  });

  // ── PR Curve ──
  if (charts.pr) charts.pr.destroy();
  charts.pr = new Chart($('chart-pr'), {
    type: 'line',
    data: {
      datasets: Object.entries(perfData).map(([name, d]) => ({
        label: name,
        data: d.pr_curve.recall.map((x,i) => ({ x, y: d.pr_curve.precision[i] })),
        borderColor: MODEL_COLORS[name],
        backgroundColor: 'transparent',
        borderWidth: 2.5,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
      }))
    },
    options: {
      responsive: true,
      parsing: false,
      scales: {
        x: { type:'linear', min:0, max:1, title:{ display:true, text:'Recall', color:COLORS.muted }, grid:{ color:'rgba(255,255,255,0.04)' }, ticks:{ color:COLORS.muted } },
        y: { type:'linear', min:0, max:1, title:{ display:true, text:'Precision', color:COLORS.muted }, grid:{ color:'rgba(255,255,255,0.04)' }, ticks:{ color:COLORS.muted } },
      },
      plugins: { legend: { labels: { color: COLORS.muted, usePointStyle:true, pointStyleWidth:8 } } },
      interaction: { mode:'nearest', intersect:false }
    }
  });

  // ── Grouped Bar: Metrics Comparison ──
  const labels = ['Precision','Recall','F1 Score','ROC-AUC'];
  const metricKeys = ['precision','recall','f1_score','roc_auc'];

  // Build legend
  const legendRow = $('chart-metrics-legend');
  legendRow.innerHTML = Object.entries(MODEL_COLORS).map(([name, color]) =>
    `<div class="legend-item"><div class="legend-dot" style="background:${color}"></div>${name}</div>`
  ).join('');

  if (charts.metrics) charts.metrics.destroy();
  charts.metrics = new Chart($('chart-metrics'), {
    type: 'bar',
    data: {
      labels,
      datasets: Object.entries(perfData).map(([name, d]) => ({
        label: name,
        data: metricKeys.map(k => parseFloat((d[k]*100).toFixed(2))),
        backgroundColor: glow(MODEL_COLORS[name], 0.7),
        borderColor: MODEL_COLORS[name],
        borderWidth: 1.5,
        borderRadius: 4,
      }))
    },
    options: {
      responsive: true,
      scales: {
        x: { grid:{ color:'rgba(255,255,255,0.04)' }, ticks:{ color:COLORS.muted } },
        y: { min:0, max:100, grid:{ color:'rgba(255,255,255,0.04)' }, ticks:{ color:COLORS.muted, callback: v => v+'%' } },
      },
      plugins: { legend: { display:false } },
    }
  });
}

// ─── Predictor: Preset Dropdown ───────────────────────────────────────────
function populatePresetDropdown(data) {
  const sel = $('preset-select');
  const normalGroup = sel.querySelector('optgroup[label="Normal Transactions"]');
  const fraudGroup  = sel.querySelector('optgroup[label="Fraudulent Transactions"]');
  data.normal.forEach((t, i) => {
    const opt = document.createElement('option');
    opt.value = `normal_${i}`;
    opt.textContent = `${t.label} ($${t.amount.toFixed(2)})`;
    normalGroup.appendChild(opt);
  });
  data.fraud.forEach((t, i) => {
    const opt = document.createElement('option');
    opt.value = `fraud_${i}`;
    opt.textContent = `${t.label} ($${t.amount.toFixed(2)})`;
    fraudGroup.appendChild(opt);
  });

  sel.addEventListener('change', () => {
    const val = sel.value;
    if (!val) return;
    const [type, idx] = val.split('_');
    const tx = type === 'normal' ? data.normal[+idx] : data.fraud[+idx];
    applyPreset(tx.features);
  });
}

function applyPreset(features) {
  currentFeatures = { ...features };
  // Update sliders to reflect preset values
  const amountRaw = features.scaled_amount;
  $('sl-amount').value = Math.max(0, Math.min(2500, amountRaw * 90 + 100));
  $('lbl-amount').textContent = `${(features.scaled_amount).toFixed(3)} (scaled)`;
  $('sl-time').value = Math.max(-3, Math.min(3, features.scaled_time));
  $('lbl-time').textContent = features.scaled_time.toFixed(3);

  for (let i = 1; i <= 28; i++) {
    const key = `V${i}`;
    const sl = $(`sl-v${i}`);
    if (sl) {
      const val = features[key] ?? 0;
      sl.value = Math.max(-10, Math.min(10, val));
      $(`lbl-v${i}`).textContent = val.toFixed(3);
    }
  }
}

// ─── Predictor: PCA Sliders ───────────────────────────────────────────────
function buildPCASliders() {
  const grid = $('pca-sliders-grid');
  grid.innerHTML = '';
  for (let i = 1; i <= 28; i++) {
    const key = `V${i}`;
    currentFeatures[key] = 0;
    const wrap = document.createElement('div');
    wrap.className = 'slider-item';
    wrap.innerHTML = `
      <label class="slider-label">
        V${i} <span id="lbl-v${i}" class="slider-val">0.000</span>
      </label>
      <input type="range" id="sl-v${i}" class="cyber-slider"
             min="-10" max="10" step="0.01" value="0" />
    `;
    grid.appendChild(wrap);
    const sl = wrap.querySelector(`#sl-v${i}`);
    sl.addEventListener('input', () => {
      const v = parseFloat(sl.value);
      $(`lbl-v${i}`).textContent = v.toFixed(3);
      currentFeatures[key] = v;
    });
  }

  // Main sliders
  currentFeatures['scaled_amount'] = 0;
  currentFeatures['scaled_time']   = 0;

  $('sl-amount').addEventListener('input', () => {
    const raw = parseFloat($('sl-amount').value);
    const scaled = (raw - 100) / 90;
    $('lbl-amount').textContent = `$${raw}`;
    currentFeatures['scaled_amount'] = scaled;
  });
  $('sl-time').addEventListener('input', () => {
    const v = parseFloat($('sl-time').value);
    $('lbl-time').textContent = v.toFixed(2);
    currentFeatures['scaled_time'] = v;
  });
}

// ─── Predictor: Run Prediction ────────────────────────────────────────────
$('predict-btn').addEventListener('click', async () => {
  const btn = $('predict-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="predict-btn-icon">⏳</span> Analyzing…';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: currentFeatures })
    });
    const data = await res.json();
    renderPredictionResults(data);
  } catch (e) {
    alert('Prediction failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="predict-btn-icon">⚡</span> Run Fraud Analysis';
  }
});

function renderPredictionResults(data) {
  const ensemble = data.ensemble;
  const prob  = ensemble.probability;
  const isFraud = ensemble.prediction === 1;

  // Badge
  const badge = $('ensemble-badge');
  badge.textContent  = isFraud ? '🚨 FRAUD DETECTED' : '✅ NORMAL';
  badge.className    = 'card-badge ' + (isFraud ? 'card-badge--fraud' : 'card-badge--safe');

  // Gauge colour
  const gaugeColor = prob < 0.3 ? COLORS.green : prob < 0.65 ? COLORS.orange : COLORS.red;
  const gaugePct   = fmtProb(prob);
  $('gauge-probability').textContent = gaugePct;
  $('gauge-probability').style.color = gaugeColor;

  animateGauge(prob, gaugeColor);

  // Verdict list
  const list = $('verdict-list');
  list.innerHTML = Object.entries(data.models).map(([name, v]) => {
    const fraud = v.prediction === 1;
    return `
      <div class="verdict-item" style="border-color:${fraud ? 'rgba(255,65,108,0.25)' : 'rgba(0,217,165,0.15)'}">
        <span class="verdict-name">${name}</span>
        <div class="verdict-right">
          <span class="verdict-prob">${fmtProb(v.probability)}</span>
          <span class="verdict-tag ${fraud ? 'verdict-tag--fraud' : 'verdict-tag--safe'}">${fraud ? 'FRAUD' : 'SAFE'}</span>
        </div>
      </div>
    `;
  }).join('');

  // Bar chart
  if (charts.predBars) charts.predBars.destroy();
  const names = Object.keys(data.models);
  const probs = names.map(n => parseFloat((data.models[n].probability * 100).toFixed(2)));
  charts.predBars = new Chart($('chart-pred-bars'), {
    type: 'bar',
    data: {
      labels: names,
      datasets: [{
        data: probs,
        backgroundColor: names.map(n => glow(MODEL_COLORS[n] || COLORS.cyan, 0.65)),
        borderColor:     names.map(n => MODEL_COLORS[n] || COLORS.cyan),
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      scales: {
        x: { min:0, max:100, grid:{ color:'rgba(255,255,255,0.04)' }, ticks:{ color:COLORS.muted, callback: v => v+'%' } },
        y: { grid:{ display:false }, ticks:{ color:COLORS.muted } },
      },
      plugins: {
        legend: { display:false },
        annotation: {
          annotations: {
            threshold: {
              type: 'line', scaleID: 'x', value: 50,
              borderColor: COLORS.red, borderWidth: 1.5,
              borderDash: [4,4],
              label: { content: '50% Threshold', display: true, color: COLORS.red, font:{size:10} }
            }
          }
        }
      }
    }
  });
}

// ─── Gauge Canvas Animation ────────────────────────────────────────────────
function animateGauge(prob, color) {
  const canvas = $('gauge-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H - 10;
  const r  = Math.min(W, H*2) * 0.42;

  // Target angle: -π (far left = 0%) to 0 (far right = 100%)
  const startAngle = Math.PI;
  const endAngle   = 2 * Math.PI;
  gaugeTargetAngle = startAngle + prob * Math.PI;

  if (gaugeAnimFrame) cancelAnimationFrame(gaugeAnimFrame);

  function draw() {
    const diff = gaugeTargetAngle - gaugeCurrentAngle;
    gaugeCurrentAngle += diff * 0.08;

    ctx.clearRect(0, 0, W, H);

    // Track
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, 2*Math.PI);
    ctx.lineWidth = 18;
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineCap = 'round';
    ctx.stroke();

    // Gradient fill
    const grad = ctx.createLinearGradient(cx - r, 0, cx + r, 0);
    grad.addColorStop(0,   COLORS.green);
    grad.addColorStop(0.5, COLORS.orange);
    grad.addColorStop(1,   COLORS.red);
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, gaugeCurrentAngle);
    ctx.lineWidth = 18;
    ctx.strokeStyle = grad;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Needle
    const needleAngle = gaugeCurrentAngle;
    const nx = cx + (r - 4) * Math.cos(needleAngle);
    const ny = cy + (r - 4) * Math.sin(needleAngle);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(nx, ny);
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.lineCap = 'round';
    ctx.shadowBlur = 12;
    ctx.shadowColor = color;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Needle pivot
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, 2*Math.PI);
    ctx.fillStyle = color;
    ctx.shadowBlur = 10;
    ctx.shadowColor = color;
    ctx.fill();
    ctx.shadowBlur = 0;

    if (Math.abs(diff) > 0.003) {
      gaugeAnimFrame = requestAnimationFrame(draw);
    }
  }
  draw();
}

// Initialize gauge at 0 on page load
window.addEventListener('load', () => {
  animateGauge(0, COLORS.muted);
});
