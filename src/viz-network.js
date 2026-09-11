// viz-network.js — 망 구조 다이어그램.
// 층은 왼쪽에서 오른쪽으로, 각 층의 뉴런은 세로로 균등 배치.
// 호출할 때마다 캔버스를 지우고 처음부터 다시 그린다.

const PAD_X = 30;        // 좌우 여백 (뉴런이 잘리지 않을 만큼)
const PAD_TOP = 14;
const PAD_BOTTOM = 6;
const LABEL_SPACE = 22;  // 층 이름을 적을 아래쪽 공간
const MAX_RADIUS = 12;
const MIN_RADIUS = 5;
const NEURON_GAP = 2;    // 가장 빽빽한 층에서 원 사이에 남길 최소 간격의 절반

const TEAL = [18, 165, 184];   // #12a5b8 — 양수 (결정 경계의 클래스 1과 같은 색)
const ORANGE = [255, 138, 61]; // #ff8a3d — 음수 (클래스 0)
const SCALE = 3;               // |값| 3 이상이면 색·투명도가 최대

const NEUTRAL_FILL = '#f2f4f6';            // 입력층 (grey-100)
const NEURON_STROKE = 'rgba(0, 0, 0, 0.14)';
const LABEL_COLOR = '#4e5968';             // grey-700
const LABEL_FONT = '500 12px "Pretendard Variable", Pretendard, system-ui, -apple-system, "Malgun Gothic", sans-serif';

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/** 부호에 따라 청록/주황, 크기에 따라 흰색에서 멀어지는 색. */
function signedFill(value) {
  const [r, g, b] = value >= 0 ? TEAL : ORANGE;
  const t = clamp(Math.abs(value) / SCALE, 0, 1);
  return `rgb(${Math.round(255 + (r - 255) * t)}, ${Math.round(255 + (g - 255) * t)}, ${Math.round(255 + (b - 255) * t)})`;
}

function layerLabel(index, layerCount) {
  if (index === 0) return '입력';
  if (index === layerCount - 1) return '출력';
  return `은닉 ${index}`;
}

function layout(layerSizes, width, height) {
  const usableH = height - PAD_TOP - PAD_BOTTOM - LABEL_SPACE;
  const maxNeurons = Math.max(...layerSizes);
  // 반지름은 가장 빽빽한 층 기준으로 자동 축소 (최대 12px, 최소 5px)
  const radius = clamp(usableH / maxNeurons / 2 - NEURON_GAP, MIN_RADIUS, MAX_RADIUS);
  const spanX = width - 2 * PAD_X;

  const positions = layerSizes.map((n, l) => {
    const x = layerSizes.length === 1 ? width / 2 : PAD_X + (spanX * l) / (layerSizes.length - 1);
    const spacing = usableH / n;
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = { x, y: PAD_TOP + (i + 0.5) * spacing };
    return out;
  });
  return { positions, radius };
}

/**
 * 망 다이어그램을 그린다.
 * @param {HTMLCanvasElement} canvas
 * @param {{layerSizes: number[], W: number[][][], b: number[][][]}} network
 */
export function drawNetwork(canvas, network) {
  if (!canvas || typeof canvas.getContext !== 'function') {
    throw new TypeError('drawNetwork: 첫 번째 인자는 canvas 요소여야 합니다.');
  }
  if (!network || !Array.isArray(network.layerSizes) || !Array.isArray(network.W)) {
    throw new TypeError('drawNetwork: 두 번째 인자는 { layerSizes, W, b } 를 가진 망이어야 합니다.');
  }
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  const { layerSizes, W, b } = network;
  const { positions, radius } = layout(layerSizes, width, height);

  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, width, height);

  // ── 가중치 선 (뉴런보다 먼저 그려 아래에 깔리게) ──
  for (let l = 0; l < W.length; l++) {
    const from = positions[l];
    const to = positions[l + 1];
    for (let i = 0; i < W[l].length; i++) {
      for (let j = 0; j < W[l][i].length; j++) {
        const w = W[l][i][j];
        const [r, g, bl] = w >= 0 ? TEAL : ORANGE;
        ctx.lineWidth = clamp(Math.abs(w) * 1.5, 0.5, 6);
        ctx.strokeStyle = `rgba(${r}, ${g}, ${bl}, ${clamp(Math.abs(w) / SCALE, 0.15, 1)})`;
        ctx.beginPath();
        ctx.moveTo(from[j].x, from[j].y);
        ctx.lineTo(to[i].x, to[i].y);
        ctx.stroke();
      }
    }
  }

  // ── 뉴런 (채움 = 편향) ──
  ctx.lineWidth = 1;
  ctx.strokeStyle = NEURON_STROKE;
  for (let l = 0; l < positions.length; l++) {
    for (let i = 0; i < positions[l].length; i++) {
      const p = positions[l][i];
      // 입력층에는 편향이 없다 — 중립색으로 둔다
      ctx.fillStyle = l === 0 ? NEUTRAL_FILL : signedFill(b[l - 1][i][0]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }

  ctx.fillStyle = LABEL_COLOR;
  ctx.font = LABEL_FONT;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  for (let l = 0; l < layerSizes.length; l++) {
    ctx.fillText(layerLabel(l, layerSizes.length), positions[l][0].x, height - PAD_BOTTOM);
  }
  ctx.restore();
}
