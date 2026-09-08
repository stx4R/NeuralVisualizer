// main.js — 컨트롤 바 · 학습 루프 · 세 캔버스를 묶는 앱 진입점.
// 엔진(network/data/gradcheck)과 시각화(viz-*)는 그대로 쓰고 여기서 조립만 한다.

import { Network } from './network.js';
import { generate } from './data.js';
import { checkGradients } from './gradcheck.js';
import { drawBoundary } from './viz-boundary.js';
import { drawNetwork } from './viz-network.js';
import { drawLoss } from './viz-loss.js';
import { computeSurface, drawSurface } from './viz-surface.js';

const STEPS_PER_FRAME = 3;        // 한 프레임에 도는 학습 스텝 수
const HEAVY_REDRAW_EVERY = 5;     // 결정 경계·망 다이어그램 갱신 주기 (프레임)

const SAMPLE_COUNT = 200;
const DATA_SEED = 7;              // 설정을 바꿔도 같은 점 배치를 유지해 비교하기 쉽게
const LR_MIN = 0.01;
const LR_MAX = 3.0;
const LR_STEPS = 1000;            // 슬라이더 눈금 수 (로그 스케일로 매핑)
const MAX_HIDDEN_LAYERS = 4;
const MIN_NEURONS = 1;
const MAX_NEURONS = 8;
const GRAD_EPSILON = 1e-4;
const MAX_PATH_POINTS = 8000;     // 학습 궤적 상한 — 넘으면 오래된 점부터 버린다
const MIN_PITCH = 5;
const MAX_PITCH = 85;

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

const el = {
  dataset: document.getElementById('dataset'),
  noise: document.getElementById('noise'),
  noiseValue: document.getElementById('noise-value'),
  activation: document.getElementById('activation'),
  lr: document.getElementById('lr'),
  lrValue: document.getElementById('lr-value'),
  layers: document.getElementById('layers'),
  layerAdd: document.getElementById('layer-add'),
  layerRemove: document.getElementById('layer-remove'),
  layerSummary: document.getElementById('layer-summary'),
  toggle: document.getElementById('toggle'),
  step: document.getElementById('step'),
  reset: document.getElementById('reset'),
  gradcheck: document.getElementById('gradcheck'),
  gcResult: document.getElementById('gc-result'),
  statEpoch: document.getElementById('stat-epoch'),
  statLoss: document.getElementById('stat-loss'),
  statAcc: document.getElementById('stat-acc'),
  boundary: document.getElementById('canvas-boundary'),
  network: document.getElementById('canvas-network'),
  loss: document.getElementById('canvas-loss'),
  surface: document.getElementById('canvas-surface'),
  viewBoundary: document.getElementById('view-boundary'),
  viewSurface: document.getElementById('view-surface'),
  surfaceControls: document.getElementById('surface-controls'),
  paramA: document.getElementById('param-a'),
  paramB: document.getElementById('param-b'),
  steps: document.getElementById('steps'),
  span: document.getElementById('span'),
  spanValue: document.getElementById('span-value'),
  height: document.getElementById('height'),
  heightValue: document.getElementById('height-value'),
  compute: document.getElementById('compute'),
  computeStatus: document.getElementById('compute-status'),
  wireframe: document.getElementById('wireframe'),
};

const state = {
  dataset: 'xor',
  noise: 0.05,
  hidden: [4, 4],
  activation: 'tanh',
  learningRate: 0.5,
  running: false,
  frame: 0,
  epoch: 0,
  history: [],
  lastOutput: null,
  data: null,
  net: null,
  rafId: 0,

  // ── 손실 표면 뷰 ──
  view: 'boundary',       // 'boundary' | 'surface'
  surface: null,          // computeSurface 결과. '표면 계산'을 눌러야 채워진다
  path: [],               // 학습 궤적 [{a, b, loss}]
  camera: { yaw: 35, pitch: 35 },
  computing: false,
  surfaceOpts: { idxA: 0, idxB: 1, steps: 40, span: 2, heightScale: 0.8, wireframe: false },
};

// ── 학습률 로그 스케일 ↔ 슬라이더 위치 ──────────────────────
const lrFromSlider = (v) => LR_MIN * Math.pow(LR_MAX / LR_MIN, v / LR_STEPS);
const sliderFromLr = (lr) => Math.round((Math.log(lr / LR_MIN) / Math.log(LR_MAX / LR_MIN)) * LR_STEPS);

// ── 모델·데이터 ─────────────────────────────────────────────
function layerSizes() {
  return [2, ...state.hidden, 1];
}

function regenerateData() {
  state.data = generate(state.dataset, SAMPLE_COUNT, state.noise, DATA_SEED);
}

/** 망을 새 난수로 다시 만든다. 학습 기록도 함께 비운다. */
function resetNetwork() {
  state.net = new Network(layerSizes(), state.activation);
  state.epoch = 0;
  state.frame = 0;
  state.history = [];
  state.lastOutput = state.net.predict(state.data.X); // epoch 0의 손실·정확도도 보여준다
  state.path = [];
  state.surface = null; // 구조가 바뀌면 파라미터 인덱스의 뜻이 달라지므로 표면도 버린다
  hideGradCheck();
}

function accuracy() {
  const out = state.lastOutput ?? state.net.predict(state.data.X);
  const Y = state.data.Y[0];
  let ok = 0;
  for (let j = 0; j < Y.length; j++) if ((out[0][j] >= 0.5 ? 1 : 0) === Y[j]) ok++;
  return ok / Y.length;
}

/** 학습 한 스텝. history에는 갱신 직전의 손실을 남긴다. */
function trainStep() {
  const { X, Y } = state.data;
  const cache = state.net.forward(X);
  state.lastOutput = cache.output;
  const loss = state.net.loss(cache.output, Y);
  state.history.push(loss);

  // 갱신 '직전'의 파라미터와 손실을 궤적에 남긴다 — (θ_t, J(θ_t)) 짝이 맞아야 한다
  const flat = state.net.getFlatParams();
  state.path.push({ a: flat[state.surfaceOpts.idxA], b: flat[state.surfaceOpts.idxB], loss });
  if (state.path.length > MAX_PATH_POINTS) state.path.shift();

  state.net.step(state.net.backward(cache, Y), state.learningRate);
  state.epoch++;
}

// ── 그리기 ──────────────────────────────────────────────────
function drawHeavy() {
  if (state.view === 'surface') drawSurfaceView();
  else drawBoundary(el.boundary, state.net, state.data);
  drawNetwork(el.network, state.net);
}

function drawLight() {
  drawLoss(el.loss, state.history);
  updateStatus();
}

function drawAll() {
  drawHeavy();
  drawLight();
}

function updateStatus() {
  // 학습 전에는 기록이 없으므로 초기 예측에서 바로 손실을 구한다
  const last = state.history.length
    ? state.history[state.history.length - 1]
    : state.lastOutput && state.net.loss(state.lastOutput, state.data.Y);
  el.statEpoch.textContent = String(state.epoch);
  el.statLoss.textContent = Number.isFinite(last) ? last.toFixed(4) : '—';
  el.statAcc.textContent = state.lastOutput ? `${(accuracy() * 100).toFixed(1)}%` : '—';
}

// ── 학습 루프 ───────────────────────────────────────────────
function frame() {
  if (!state.running) return; // 일시정지하면 루프를 멈춘다 — 캔버스는 마지막 상태 그대로 남는다
  for (let i = 0; i < STEPS_PER_FRAME; i++) trainStep();
  state.frame++;
  drawLight();
  if (state.frame % HEAVY_REDRAW_EVERY === 0) drawHeavy(); // 무거운 둘만 5프레임마다
  state.rafId = requestAnimationFrame(frame);
}

function setRunning(running) {
  state.running = running;
  el.toggle.textContent = running ? '일시정지' : '재생';
  el.step.disabled = running;
  if (running) state.rafId = requestAnimationFrame(frame);
  else cancelAnimationFrame(state.rafId);
}

// ── 그래디언트 검증 ─────────────────────────────────────────
/** 행렬의 최대 절대행합 ‖W‖∞ — 한 층이 흔들림을 얼마나 증폭하는지의 상한. */
function maxRowSum(W) {
  let best = 0;
  for (const row of W) {
    let sum = 0;
    for (const v of row) sum += Math.abs(v);
    best = Math.max(best, sum);
  }
  return best;
}

/**
 * ReLU는 z=0에서 미분이 꺾인다. 어떤 표본의 z가 0에 너무 가까우면
 * (J(θ+ε) − J(θ−ε)) / 2ε 가 꺾임을 가로질러 수치 미분 자체가 무의미해진다.
 *
 * 그래서 층마다 두 값을 비교한다.
 *   margin : min|z| — 꺾임까지의 거리
 *   reach  : 파라미터 하나를 ε 흔들었을 때 그 층의 z가 움직일 수 있는 최대 폭
 * reach는 앞 층에서 생긴 흔들림이 뒤 층으로 가면서 ‖W‖∞ 만큼 증폭되므로
 * R[l] = ‖W[l]‖∞ · R[l−1] + ε · max(|a[l−1]|, 1) 로 누적한다 (|f′| ≤ 1 이용).
 */
function kinkInfo(net, X, epsilon) {
  const { A, Z } = net.forward(X);
  let margin = Infinity;
  let reach = 0;
  let slack = Infinity; // margin − reach 가 가장 빠듯한 층을 고른다
  let propagated = 0;

  for (let l = 0; l < Z.length; l++) {
    let maxPrev = 1; // 편향을 ε 흔드는 경우도 포함하려고 하한을 1로 둔다
    for (const row of A[l]) for (const v of row) maxPrev = Math.max(maxPrev, Math.abs(v));
    propagated = maxRowSum(net.W[l]) * propagated + epsilon * maxPrev;

    if (l === Z.length - 1) break; // 출력층은 시그모이드 — 꺾임 없음

    let layerMargin = Infinity;
    for (const row of Z[l]) for (const v of row) layerMargin = Math.min(layerMargin, Math.abs(v));
    if (layerMargin - propagated < slack) {
      slack = layerMargin - propagated;
      margin = layerMargin;
      reach = propagated;
    }
  }
  return { margin, reach, safe: margin > reach };
}

function hideGradCheck() {
  el.gcResult.hidden = true;
  el.gcResult.className = 'gc';
  el.gcResult.textContent = '';
}

function showGradCheck(relativeError, passed, note) {
  el.gcResult.hidden = false;
  el.gcResult.className = `gc ${passed ? 'ok' : 'fail'}`;
  el.gcResult.replaceChildren();
  el.gcResult.append(`그래디언트 검증 — 상대오차 ${relativeError.toExponential(3)} (기준 < 1e-7) `);

  const verdict = document.createElement('span');
  verdict.className = 'verdict';
  verdict.textContent = passed ? '통과' : '실패';
  el.gcResult.append(verdict);

  if (note) {
    const span = document.createElement('span');
    span.className = 'note';
    span.textContent = ` · ${note}`;
    el.gcResult.append(span);
  }
}

/** 앞에서부터 n개 표본만 잘라낸다. 데이터가 두 클래스를 번갈아 담고 있어 균형은 유지된다. */
function sliceColumns(X, Y, n) {
  if (n >= X[0].length) return { X, Y };
  return { X: X.map((row) => row.slice(0, n)), Y: Y.map((row) => row.slice(0, n)) };
}

/**
 * 꺾임에 걸리지 않는 검사 지점을 찾는다.
 *   ① 현재 망 그대로 (가능하면 사용자가 보고 있는 망을 그대로 검사한다)
 *   ② 같은 구조의 임시 망에 파라미터를 살짝 흔들어 넣은 것
 *      — 편향이 0이면 죽은 유닛의 z가 '정확히' 0이라 반드시 꺾임에 얹힌다. 흔들면 떨어진다.
 *   ③ 그래도 안 되면 표본 수를 줄인다. 표본이 많을수록 어느 하나가 z=0 근처에 놓일
 *      확률이 커지므로, 표본을 줄이는 것은 ReLU 그래디언트 체킹의 표준적인 대처다.
 */
function findKinkFreePoint(X, Y) {
  const total = X[0].length;
  for (const count of [total, 40, 10, 4]) {
    const sub = sliceColumns(X, Y, count);
    const sizeNote = count < total ? `표본 ${count}개` : '';

    if (kinkInfo(state.net, sub.X, GRAD_EPSILON).safe) {
      return { net: state.net, X: sub.X, Y: sub.Y, detail: sizeNote };
    }
    for (let attempt = 1; attempt <= 60; attempt++) {
      const candidate = new Network(layerSizes(), state.activation, { seed: attempt });
      const params = candidate.getFlatParams();
      candidate.setFlatParams(params.map((v) => v + (Math.random() - 0.5) * 0.6));
      if (kinkInfo(candidate, sub.X, GRAD_EPSILON).safe) {
        const detail = ['파라미터를 흔든 임시 망', sizeNote].filter(Boolean).join(', ');
        return { net: candidate, X: sub.X, Y: sub.Y, detail };
      }
    }
  }
  return null;
}

function runGradientCheck() {
  setRunning(false);
  const { X, Y } = state.data;

  // tanh·sigmoid는 매끄러워서 현재 지점 그대로 재면 된다
  if (state.activation !== 'relu') {
    const r = checkGradients(state.net, X, Y, GRAD_EPSILON);
    showGradCheck(r.relativeError, r.passed, '');
    return;
  }

  const here = kinkInfo(state.net, X, GRAD_EPSILON);
  const point = findKinkFreePoint(X, Y);
  if (!point) {
    el.gcResult.hidden = false;
    el.gcResult.className = 'gc fail';
    el.gcResult.textContent = 'ReLU 꺾임(z=0)에 걸리지 않는 검사 지점을 찾지 못했습니다.';
    return;
  }

  // 현재 망·전체 표본 그대로 잰 경우가 아니면 무엇을 바꿨는지 밝힌다
  const note = point.detail
    ? `ReLU 꺾임 회피 — 현재 지점은 min|z| ${here.margin.toExponential(2)} ≤ 섭동 반경 ` +
      `${here.reach.toExponential(2)} 이라 ${point.detail}에서 검사`
    : '';
  const r = checkGradients(point.net, point.X, point.Y, GRAD_EPSILON);
  showGradCheck(r.relativeError, r.passed, note);
}

// ── 손실 표면 ───────────────────────────────────────────────
/** 평탄화 파라미터에 사람이 읽을 이름을 붙인다. getFlatParams()와 순서가 같아야 한다. */
function parameterLabels() {
  const sizes = layerSizes();
  const labels = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const rows = sizes[l + 1];
    const cols = sizes[l];
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) labels.push(`W${l}[${i}][${j}]`);
    for (let i = 0; i < rows; i++) labels.push(`b${l}[${i}]`);
  }
  return labels;
}

/** 망 구조가 바뀌면 파라미터 목록을 다시 채운다. */
function refreshParamOptions() {
  const labels = parameterLabels();
  state.surfaceOpts.idxA = Math.min(state.surfaceOpts.idxA, labels.length - 1);
  state.surfaceOpts.idxB = Math.min(state.surfaceOpts.idxB, labels.length - 1);
  for (const [select, key] of [[el.paramA, 'idxA'], [el.paramB, 'idxB']]) {
    select.replaceChildren();
    labels.forEach((name, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = name;
      select.append(option);
    });
    select.value = String(state.surfaceOpts[key]);
  }
}

function drawSurfaceView() {
  const ctx = el.surface.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  if (!state.surface) {
    ctx.clearRect(0, 0, el.surface.width, el.surface.height);
    ctx.fillStyle = '#9a9aa2';
    ctx.font = '12px system-ui, -apple-system, "Malgun Gothic", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText("'표면 계산'을 누르면 지금 파라미터 주변을 훑습니다", el.surface.width / 2, el.surface.height / 2);
    return;
  }

  const labels = parameterLabels();
  // 표면을 계산한 뒤 파라미터 선택이 바뀌었다면 궤적은 다른 축의 값이므로 얹지 않는다
  const sameAxes =
    state.surface.idxA === state.surfaceOpts.idxA && state.surface.idxB === state.surfaceOpts.idxB;

  drawSurface(el.surface, state.surface, {
    yaw: state.camera.yaw,
    pitch: state.camera.pitch,
    heightScale: state.surfaceOpts.heightScale,
    wireframe: state.surfaceOpts.wireframe,
    path: sameAxes ? state.path : [],
    labelA: labels[state.surface.idxA],
    labelB: labels[state.surface.idxB],
  });
}

function setView(view) {
  state.view = view;
  const isSurface = view === 'surface';
  el.boundary.hidden = isSurface;
  el.surface.hidden = !isSurface;
  el.surfaceControls.hidden = !isSurface;
  el.viewBoundary.classList.toggle('is-active', !isSurface);
  el.viewSurface.classList.toggle('is-active', isSurface);
  drawHeavy();
}

/**
 * '표면 계산' 버튼. 학습 루프에서는 절대 부르지 않는다 —
 * 격자 한 장에 forward가 steps²번(40×40이면 1600번) 든다.
 */
function requestSurfaceCompute() {
  if (state.computing) return;
  state.computing = true;
  el.compute.disabled = true;
  el.computeStatus.textContent = '계산 중...';

  // 버튼 상태가 먼저 화면에 반영되도록 한 틱 양보한다.
  // (계산 자체는 동기라 도중에는 다시 그려지지 않는다 — onProgress는 실패 지점 보고용)
  setTimeout(() => {
    const { idxA, idxB, span, steps } = state.surfaceOpts;
    const started = performance.now();
    let rowsDone = 0;
    try {
      state.surface = computeSurface(
        state.net, state.data.X, state.data.Y, idxA, idxB, span, steps,
        (row) => { rowsDone = row; }
      );
      const ms = Math.round(performance.now() - started);
      el.computeStatus.textContent = `${steps}×${steps} 격자 ${steps * steps}점 · ${ms}ms · epoch ${state.epoch} 기준`;
    } catch (error) {
      state.surface = null;
      el.computeStatus.textContent = `계산 실패 (${rowsDone}/${steps}행): ${error.message}`;
    } finally {
      state.computing = false;
      el.compute.disabled = false;
      drawSurfaceView();
    }
  }, 16);
}

// ── 은닉층 구성 UI ──────────────────────────────────────────
function renderLayers() {
  el.layers.replaceChildren();
  state.hidden.forEach((count, index) => {
    const chip = document.createElement('div');
    chip.className = 'chip';

    const dec = document.createElement('button');
    dec.type = 'button';
    dec.textContent = '−';
    dec.title = `은닉 ${index + 1} 뉴런 줄이기`;
    dec.disabled = count <= MIN_NEURONS;
    dec.addEventListener('click', () => changeNeurons(index, -1));

    const value = document.createElement('span');
    value.className = 'count';
    value.textContent = String(count);

    const inc = document.createElement('button');
    inc.type = 'button';
    inc.textContent = '+';
    inc.title = `은닉 ${index + 1} 뉴런 늘리기`;
    inc.disabled = count >= MAX_NEURONS;
    inc.addEventListener('click', () => changeNeurons(index, +1));

    chip.append(dec, value, inc);
    el.layers.append(chip);
  });

  el.layerSummary.textContent = state.hidden.length ? `[${state.hidden.join(', ')}]` : '없음';
  el.layerAdd.disabled = state.hidden.length >= MAX_HIDDEN_LAYERS;
  el.layerRemove.disabled = state.hidden.length === 0;
}

function changeNeurons(index, delta) {
  const next = state.hidden[index] + delta;
  if (next < MIN_NEURONS || next > MAX_NEURONS) return;
  state.hidden[index] = next;
  renderLayers();
  rebuild();
}

/** 구조나 데이터가 바뀌면 학습을 멈추고 처음부터 다시 시작한다. */
function rebuild({ regenerate = false } = {}) {
  setRunning(false);
  if (regenerate) regenerateData();
  resetNetwork();
  refreshParamOptions();
  drawAll();
}

// ── 이벤트 연결 ─────────────────────────────────────────────
el.dataset.addEventListener('change', () => {
  state.dataset = el.dataset.value;
  rebuild({ regenerate: true });
});

el.noise.addEventListener('input', () => {
  state.noise = Number(el.noise.value);
  el.noiseValue.textContent = state.noise.toFixed(2);
  rebuild({ regenerate: true });
});

el.activation.addEventListener('change', () => {
  state.activation = el.activation.value;
  rebuild();
});

el.lr.addEventListener('input', () => {
  state.learningRate = lrFromSlider(Number(el.lr.value));
  el.lrValue.textContent = state.learningRate.toFixed(2);
});

el.layerAdd.addEventListener('click', () => {
  if (state.hidden.length >= MAX_HIDDEN_LAYERS) return;
  state.hidden.push(4);
  renderLayers();
  rebuild();
});

el.layerRemove.addEventListener('click', () => {
  if (!state.hidden.length) return;
  state.hidden.pop();
  renderLayers();
  rebuild();
});

el.toggle.addEventListener('click', () => setRunning(!state.running));

el.step.addEventListener('click', () => {
  trainStep();
  drawAll();
});

el.reset.addEventListener('click', () => rebuild());

el.gradcheck.addEventListener('click', runGradientCheck);

// ── 손실 표면 컨트롤 ────────────────────────────────────────
el.viewBoundary.addEventListener('click', () => setView('boundary'));
el.viewSurface.addEventListener('click', () => setView('surface'));

for (const [select, key] of [[el.paramA, 'idxA'], [el.paramB, 'idxB']]) {
  select.addEventListener('change', () => {
    state.surfaceOpts[key] = Number(select.value);
    state.path = []; // 축이 바뀌면 기존 궤적은 의미가 없다
    drawSurfaceView();
  });
}

el.steps.addEventListener('change', () => {
  state.surfaceOpts.steps = Number(el.steps.value);
});

el.span.addEventListener('input', () => {
  state.surfaceOpts.span = Number(el.span.value);
  el.spanValue.textContent = state.surfaceOpts.span.toFixed(1);
});

el.height.addEventListener('input', () => {
  state.surfaceOpts.heightScale = Number(el.height.value);
  el.heightValue.textContent = state.surfaceOpts.heightScale.toFixed(2);
  drawSurfaceView();
});

el.wireframe.addEventListener('change', () => {
  state.surfaceOpts.wireframe = el.wireframe.checked;
  drawSurfaceView();
});

el.compute.addEventListener('click', requestSurfaceCompute);

// 드래그로 회전 — 재계산 없이 drawSurface만 다시 부른다
let drag = null;
el.surface.addEventListener('pointerdown', (event) => {
  drag = { x: event.clientX, y: event.clientY };
  // 캔버스 밖으로 나가도 드래그가 이어지게 한다. 잡을 수 없는 포인터면 그냥 넘어간다.
  try {
    el.surface.setPointerCapture(event.pointerId);
  } catch {
    /* 포인터 캡처는 있으면 좋고 없어도 되는 기능이다 */
  }
});
el.surface.addEventListener('pointermove', (event) => {
  if (!drag) return;
  state.camera.yaw = (state.camera.yaw + (event.clientX - drag.x) * 0.5 + 360) % 360;
  state.camera.pitch = clamp(state.camera.pitch + (event.clientY - drag.y) * 0.5, MIN_PITCH, MAX_PITCH);
  drag = { x: event.clientX, y: event.clientY };
  drawSurfaceView();
});
for (const type of ['pointerup', 'pointercancel']) {
  el.surface.addEventListener(type, () => { drag = null; });
}

// ── 시작 ────────────────────────────────────────────────────
el.dataset.value = state.dataset;
el.activation.value = state.activation;
el.noise.value = String(state.noise);
el.noiseValue.textContent = state.noise.toFixed(2);
el.lr.max = String(LR_STEPS);
el.lr.value = String(sliderFromLr(state.learningRate));
el.lrValue.textContent = state.learningRate.toFixed(2);

el.span.value = String(state.surfaceOpts.span);
el.spanValue.textContent = state.surfaceOpts.span.toFixed(1);
el.height.value = String(state.surfaceOpts.heightScale);
el.heightValue.textContent = state.surfaceOpts.heightScale.toFixed(2);
el.steps.value = String(state.surfaceOpts.steps);
el.wireframe.checked = state.surfaceOpts.wireframe;

renderLayers();
regenerateData();
resetNetwork();
refreshParamOptions();
setView(state.view);
drawAll();
