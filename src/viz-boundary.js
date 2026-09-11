// viz-boundary.js — 결정 경계 히트맵 + 데이터 점.
// 캔버스 전체가 [-1.2, 1.2]² 를 담는다. 호출할 때마다 캔버스를 지우고 처음부터 다시 그린다.

const DOMAIN = 1.2;      // 좌표계 반경
const GRID = 60;         // 히트맵 해상도 (60×60 = 3600점을 순전파 한 번으로 계산)
const POINT_RADIUS = 4.5;

// 데이터 클래스 색 — UI 강조색(토스 블루)과 겹치지 않게 청록·주황을 쓴다. style.css의 --class-1/--class-0과 같은 값.
const ORANGE = [255, 138, 61]; // #ff8a3d — 출력 0
const WHITE = [255, 255, 255]; // 출력 0.5
const TEAL = [18, 165, 184];   // #12a5b8 — 출력 1

const CSS_ORANGE = '#ff8a3d';
const CSS_TEAL = '#12a5b8';

/** 출력값 0~1 → [r, g, b]. 0.5를 기준으로 주황↔흰색, 흰색↔청록 선형 보간. */
function colorFor(v) {
  const t = v <= 0.5 ? v * 2 : (v - 0.5) * 2;
  const from = v <= 0.5 ? ORANGE : WHITE;
  const to = v <= 0.5 ? WHITE : TEAL;
  return [
    Math.round(from[0] + (to[0] - from[0]) * t),
    Math.round(from[1] + (to[1] - from[1]) * t),
    Math.round(from[2] + (to[2] - from[2]) * t),
  ];
}

/** 데이터 좌표 → 캔버스 픽셀. y축은 위가 +. */
function toPixel(x, y, width, height) {
  return [((x + DOMAIN) / (2 * DOMAIN)) * width, ((DOMAIN - y) / (2 * DOMAIN)) * height];
}

/**
 * 결정 경계를 그린다.
 * @param {HTMLCanvasElement} canvas 정사각형 캔버스
 * @param {{predict: (X: number[][]) => number[][]}} network 출력 뉴런이 1개인 망
 * @param {{X: number[][], Y: number[][]}} [data] 함께 찍을 데이터 점 (없으면 히트맵만)
 */
export function drawBoundary(canvas, network, data) {
  if (!canvas || typeof canvas.getContext !== 'function') {
    throw new TypeError('drawBoundary: 첫 번째 인자는 canvas 요소여야 합니다.');
  }
  if (!network || typeof network.predict !== 'function') {
    throw new TypeError('drawBoundary: 두 번째 인자는 predict()를 가진 망이어야 합니다.');
  }
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;

  // ── 격자점을 (2 × 3600) 행렬 하나로 묶어 순전파를 딱 한 번만 돈다 ──
  const step = (2 * DOMAIN) / GRID;
  const total = GRID * GRID;
  const gx = new Array(total);
  const gy = new Array(total);
  for (let row = 0; row < GRID; row++) {
    const y = DOMAIN - (row + 0.5) * step;   // 위에서 아래로 (ImageData 순서와 맞춘다)
    for (let col = 0; col < GRID; col++) {
      const k = row * GRID + col;
      gx[k] = -DOMAIN + (col + 0.5) * step;
      gy[k] = y;
    }
  }
  const A = network.predict([gx, gy]);
  if (A.length !== 1) {
    throw new RangeError(`drawBoundary: 출력 뉴런이 1개인 망만 그릴 수 있습니다 — 받은 출력 ${A.length}개.`);
  }

  // ── 60×60 ImageData → 오프스크린 캔버스 → 부드럽게 확대 ──
  const off = document.createElement('canvas');
  off.width = GRID;
  off.height = GRID;
  const offCtx = off.getContext('2d');
  const img = offCtx.createImageData(GRID, GRID);
  const px = img.data;
  for (let k = 0; k < total; k++) {
    const [r, g, b] = colorFor(A[0][k]);
    const o = k * 4;
    px[o] = r;
    px[o + 1] = g;
    px[o + 2] = b;
    px[o + 3] = 255;
  }
  offCtx.putImageData(img, 0, 0);

  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(off, 0, 0, GRID, GRID, 0, 0, width, height);

  if (data && data.X && data.Y) {
    const { X, Y } = data;
    const m = X[0].length;
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#ffffff';
    for (let j = 0; j < m; j++) {
      const [cx, cy] = toPixel(X[0][j], X[1][j], width, height);
      ctx.beginPath();
      ctx.arc(cx, cy, POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = Y[0][j] === 1 ? CSS_TEAL : CSS_ORANGE;
      ctx.fill();
      ctx.stroke();
    }
  }
  ctx.restore();
}
