// viz-surface.js — 손실 표면 3D 시각화. Canvas 2D에 직접 투영해서 그린다.
//
// ⚠️ 이 표면이 무엇인지 정확히 알고 봐야 한다.
// 선택한 두 파라미터만 격자로 훑고 나머지 파라미터는 '현재 값에 고정'한 2차원 단면이다.
// 실제 학습에서는 나머지 파라미터도 함께 움직이므로, 이 단면 위에 얹은 학습 궤적은 근사다.
// 궤적의 각 점이 실제로 가졌던 손실은 그때그때의 전체 파라미터로 계산된 값이라,
// 여기 그려진 표면의 높이와는 다를 수 있다. 학습이 끝난 뒤(파라미터가 거의 멈춘 뒤)
// 그 지점에서 계산한 표면일수록 이 오차가 작다.

import { rotateX, rotateY, multiply, applyMatrix, project, normal, normalize, lambert } from './proj3d.js';

const FOCAL = 900;
const DISTANCE = 3.2;
const MARGIN = 10;      // 캔버스 가장자리 여백
const LABEL_SPACE = 34; // 좌상단 축 라벨이 차지하는 높이

// 손실 팔레트 — 순차형 5스톱.
// 결정 경계의 파랑(#3a7ae8)·주황(#e8853a)과 뜻이 겹치면 안 되므로 다른 색계열을 쓴다.
const PALETTE = [
  [45, 27, 82],   // #2d1b52 낮음
  [33, 84, 138],  // #21548a
  [26, 140, 138], // #1a8c8a
  [124, 191, 90], // #7cbf5a
  [242, 230, 92], // #f2e65c 높음
];

const LIGHT = normalize({ x: -0.4, y: 0.75, z: -0.5 });

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/** 0~1 → 팔레트 색 [r, g, b]. 스톱 사이를 선형 보간한다. */
function paletteColor(t) {
  const x = clamp(t, 0, 1) * (PALETTE.length - 1);
  const i = Math.min(PALETTE.length - 2, Math.floor(x));
  const f = x - i;
  const a = PALETTE[i];
  const b = PALETTE[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/**
 * 손실 표면 계산. 선택한 두 파라미터만 격자로 훑는다.
 *
 * @param {object} network getFlatParams/setFlatParams/predict/loss 를 가진 망
 * @param {number[][]} X 입력 (특징 × m)
 * @param {number[][]} Y 라벨 (1 × m)
 * @param {number} idxA 평탄화 파라미터 인덱스 (가로축)
 * @param {number} idxB 평탄화 파라미터 인덱스 (세로축)
 * @param {number} span 중심에서 ± 얼마까지 훑을지
 * @param {number} steps 축당 격자점 수 (기본 40 → 1600점)
 * @param {(row: number, total: number) => void} [onProgress] 행 단위 진행 알림
 */
export function computeSurface(network, X, Y, idxA, idxB, span, steps = 40, onProgress = null) {
  const original = network.getFlatParams();
  const n = original.length;
  for (const [name, idx] of [['idxA', idxA], ['idxB', idxB]]) {
    if (!Number.isInteger(idx) || idx < 0 || idx >= n) {
      throw new RangeError(`computeSurface: ${name}는 0 이상 ${n - 1} 이하의 정수여야 합니다 — 받은 값 ${idx}.`);
    }
  }
  if (!Number.isInteger(steps) || steps < 2) {
    throw new RangeError(`computeSurface: steps는 2 이상의 정수여야 합니다 — 받은 값 ${steps}.`);
  }
  if (!(span > 0) || !Number.isFinite(span)) {
    throw new RangeError(`computeSurface: span은 0보다 큰 유한한 수여야 합니다 — 받은 값 ${span}.`);
  }

  const centerA = original[idxA];
  const centerB = original[idxB];
  const aVals = new Array(steps);
  const bVals = new Array(steps);
  for (let i = 0; i < steps; i++) {
    const t = (2 * span * i) / (steps - 1) - span;
    aVals[i] = centerA + t;
    bVals[i] = centerB + t;
  }

  const grid = new Array(steps);
  let minLoss = Infinity;
  let maxLoss = -Infinity;
  const probe = original.slice();

  try {
    for (let i = 0; i < steps; i++) {
      if (onProgress) onProgress(i, steps);
      probe[idxA] = aVals[i];
      const row = new Array(steps);
      for (let j = 0; j < steps; j++) {
        probe[idxB] = bVals[j];
        network.setFlatParams(probe);
        const value = network.loss(network.predict(X), Y);
        row[j] = value;
        if (value < minLoss) minLoss = value;
        if (value > maxLoss) maxLoss = value;
      }
      grid[i] = row;
    }
  } finally {
    network.setFlatParams(original); // gradcheck와 같은 규칙 — 망을 반드시 원상 복구한다
  }
  if (onProgress) onProgress(steps, steps);

  return { grid, aVals, bVals, minLoss, maxLoss, idxA, idxB, center: [centerA, centerB] };
}

/** 격자 좌표 → 정규화 3D 점. 경로도 반드시 같은 식을 쓴다. */
function toWorld(surface, a, b, heightScale, loss) {
  const { aVals, bVals, minLoss, maxLoss } = surface;
  const aMin = aVals[0];
  const aMax = aVals[aVals.length - 1];
  const bMin = bVals[0];
  const bMax = bVals[bVals.length - 1];
  const range = maxLoss - minLoss;
  return {
    x: aMax === aMin ? 0 : ((a - aMin) / (aMax - aMin)) * 2 - 1,
    z: bMax === bMin ? 0 : ((b - bMin) / (bMax - bMin)) * 2 - 1,
    y: range === 0 ? 0 : ((loss - minLoss) / range) * heightScale,
  };
}

/**
 * 격자를 3D로 돌려 투영하고, 셀을 화가 알고리즘 순서(먼 것 먼저)로 정렬해 돌려준다.
 * 캔버스를 쓰지 않으므로 그대로 검증에 쓸 수 있다. z-버퍼는 만들지 않는다.
 */
export function buildCells(surface, options = {}) {
  const { yaw = 35, pitch = 35, heightScale = 0.8, path = [] } = options;
  const { grid, aVals, bVals } = surface;
  const steps = grid.length;
  const rotation = multiply(rotateY(yaw), rotateX(pitch)); // Rx(pitch) → Ry(yaw)

  // 격자점을 한 번씩만 회전·투영해 둔다
  const rotated = new Array(steps);
  const screen = new Array(steps);
  for (let i = 0; i < steps; i++) {
    rotated[i] = new Array(steps);
    screen[i] = new Array(steps);
    for (let j = 0; j < steps; j++) {
      const world = toWorld(surface, aVals[i], bVals[j], heightScale, grid[i][j]);
      const r = applyMatrix(rotation, world);
      rotated[i][j] = r;
      screen[i][j] = project(r, FOCAL, DISTANCE);
    }
  }

  const range = surface.maxLoss - surface.minLoss;
  const cells = [];
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (let i = 0; i < steps - 1; i++) {
    for (let j = 0; j < steps - 1; j++) {
      const corners = [screen[i][j], screen[i + 1][j], screen[i + 1][j + 1], screen[i][j + 1]];
      if (corners.some((c) => c === null)) continue; // 카메라 뒤로 넘어간 셀은 건너뛴다

      let depth = 0;
      for (const c of corners) {
        depth += c.depth;
        if (c.x < minX) minX = c.x;
        if (c.x > maxX) maxX = c.x;
        if (c.y < minY) minY = c.y;
        if (c.y > maxY) maxY = c.y;
      }
      depth /= 4;

      const lossAvg = (grid[i][j] + grid[i + 1][j] + grid[i + 1][j + 1] + grid[i][j + 1]) / 4;
      // 법선은 회전 뒤 좌표에서 구한다 (조명이 카메라에 고정된 셈)
      let n = normal(rotated[i][j], rotated[i + 1][j], rotated[i + 1][j + 1]);
      if (n.z > 0) n = { x: -n.x, y: -n.y, z: -n.z }; // 카메라를 향하도록 뒤집는다
      cells.push({
        corners,
        depth,
        shade: lambert(n, LIGHT),
        t: range === 0 ? 0 : (lossAvg - surface.minLoss) / range,
      });
    }
  }

  // 화가 알고리즘 — 깊이 내림차순(먼 것 먼저)으로 칠한다
  cells.sort((p, q) => q.depth - p.depth);

  const pathPoints = path.map((point) => {
    const world = toWorld(surface, point.a, point.b, heightScale, point.loss);
    return project(applyMatrix(rotation, world), FOCAL, DISTANCE);
  });

  return { cells, pathPoints, bounds: { minX, maxX, minY, maxY } };
}

/**
 * 손실 표면을 그린다. 호출할 때마다 캔버스를 지우고 처음부터 다시 그린다.
 * @param {HTMLCanvasElement} canvas
 * @param {object} surface computeSurface의 반환값
 * @param {{yaw?: number, pitch?: number, heightScale?: number, path?: Array, wireframe?: boolean,
 *          labelA?: string, labelB?: string}} [options]
 */
export function drawSurface(canvas, surface, options = {}) {
  if (!canvas || typeof canvas.getContext !== 'function') {
    throw new TypeError('drawSurface: 첫 번째 인자는 canvas 요소여야 합니다.');
  }
  if (!surface || !Array.isArray(surface.grid)) {
    throw new TypeError('drawSurface: 두 번째 인자는 computeSurface()의 반환값이어야 합니다.');
  }
  const {
    yaw = 35,
    pitch = 35,
    heightScale = 0.8,
    path = [],
    wireframe = false,
    labelA = `θ[${surface.idxA}]`,
    labelB = `θ[${surface.idxB}]`,
  } = options;

  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  const { cells, pathPoints, bounds } = buildCells(surface, { yaw, pitch, heightScale, path });

  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, width, height);

  if (cells.length) {
    // 투영 결과를 캔버스 안에 담기게 맞춘다 (투영식은 그대로 두고 화면 배율만 조정)
    const spanX = Math.max(1e-6, bounds.maxX - bounds.minX);
    const spanY = Math.max(1e-6, bounds.maxY - bounds.minY);
    const scale = Math.min((width - 2 * MARGIN) / spanX, (height - 2 * MARGIN - LABEL_SPACE) / spanY);
    const ox = width / 2 - ((bounds.minX + bounds.maxX) / 2) * scale;
    const oy = LABEL_SPACE + (height - LABEL_SPACE) / 2 - ((bounds.minY + bounds.maxY) / 2) * scale;
    const sx = (p) => ox + p.x * scale;
    const sy = (p) => oy + p.y * scale;

    for (const cell of cells) {
      const [r, g, b] = paletteColor(cell.t);
      const s = cell.shade;
      ctx.beginPath();
      ctx.moveTo(sx(cell.corners[0]), sy(cell.corners[0]));
      for (let k = 1; k < 4; k++) ctx.lineTo(sx(cell.corners[k]), sy(cell.corners[k]));
      ctx.closePath();
      const fill = `rgb(${Math.round(r * s)}, ${Math.round(g * s)}, ${Math.round(b * s)})`;
      ctx.fillStyle = fill;
      ctx.fill();
      if (wireframe) {
        ctx.lineWidth = 0.5;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
      } else {
        // 이웃한 폴리곤 사이에 안티앨리어싱 틈이 생겨 배경이 비친다. 같은 색으로 덮어 메운다.
        ctx.lineWidth = 1;
        ctx.strokeStyle = fill;
      }
      ctx.stroke();
    }

    // ── 학습 궤적 ──
    const visible = pathPoints.filter((p) => p !== null);
    if (visible.length) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      visible.forEach((p, k) => (k === 0 ? ctx.moveTo(sx(p), sy(p)) : ctx.lineTo(sx(p), sy(p))));
      ctx.stroke();

      const last = visible[visible.length - 1];
      ctx.beginPath();
      ctx.arc(sx(last), sy(last), 5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = '#000000';
      ctx.stroke();
    }
  }

  // ── 축 라벨 ──
  ctx.fillStyle = '#33333b';
  ctx.font = '11px system-ui, -apple-system, "Malgun Gothic", sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(`가로 ${labelA}   세로 ${labelB}`, MARGIN, 15);
  ctx.fillText(`손실 ${surface.minLoss.toFixed(4)} ~ ${surface.maxLoss.toFixed(4)}`, MARGIN, 29);
  ctx.restore();
}
