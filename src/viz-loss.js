// viz-loss.js — 손실 곡선.
// x축은 epoch, y축은 손실(0부터 자동 스케일).
// 호출할 때마다 캔버스를 지우고 처음부터 다시 그린다.

const PAD = { left: 44, right: 12, top: 16, bottom: 22 };
const LINE = '#3a7ae8';
const AXIS = '#c8ccd2';
const TEXT = '#5f6368';
const TICKS = 3; // 0, 중간, 최댓값

// 축 상단으로 쓸 '깔끔한' 수의 사다리. 촘촘해야 데이터에 딱 붙는 스케일이 나온다.
const NICE_STEPS = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];

function niceCeil(max) {
  if (!(max > 0)) return 1;
  const base = Math.pow(10, Math.floor(Math.log10(max)));
  const f = max / base;
  return (NICE_STEPS.find((s) => f <= s + 1e-12) ?? 10) * base;
}

/** 눈금 라벨: 유효숫자 3자리에서 불필요한 0을 떼어낸다. */
function fmtTick(v) {
  if (v === 0) return '0';
  return String(Number(v.toPrecision(3)));
}

/**
 * 점이 폭보다 많으면 균등 간격으로 솎아낸다. 처음과 마지막은 반드시 남긴다.
 * @returns {Array<[number, number]>} [epochIndex, loss] 쌍
 */
function downsample(history, maxPoints) {
  const n = history.length;
  if (n <= maxPoints) return history.map((v, i) => [i, v]);
  const out = new Array(maxPoints);
  for (let k = 0; k < maxPoints; k++) {
    const i = Math.round((k * (n - 1)) / (maxPoints - 1));
    out[k] = [i, history[i]];
  }
  return out;
}

/**
 * 손실 곡선을 그린다.
 * @param {HTMLCanvasElement} canvas
 * @param {number[]} history epoch별 손실값
 */
export function drawLoss(canvas, history) {
  if (!canvas || typeof canvas.getContext !== 'function') {
    throw new TypeError('drawLoss: 첫 번째 인자는 canvas 요소여야 합니다.');
  }
  if (!Array.isArray(history)) {
    throw new TypeError('drawLoss: history는 손실값 배열이어야 합니다.');
  }
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;
  const x0 = PAD.left;
  const y0 = PAD.top + plotH; // 바닥 = 손실 0

  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = '11px system-ui, -apple-system, "Malgun Gothic", sans-serif';
  ctx.textBaseline = 'middle';

  const finite = history.filter((v) => Number.isFinite(v));
  const yMax = niceCeil(finite.length ? Math.max(...finite) : 0);

  // ── 눈금 3개 (0, 중간, 최댓값) ──
  ctx.strokeStyle = AXIS;
  ctx.fillStyle = TEXT;
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  for (let t = 0; t < TICKS; t++) {
    const value = (yMax * t) / (TICKS - 1);
    const y = Math.round(y0 - (value / yMax) * plotH) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x0 - 4, y);
    ctx.lineTo(x0, y);
    ctx.stroke();
    ctx.fillText(fmtTick(value), x0 - 7, y);
  }

  ctx.beginPath();
  ctx.moveTo(x0 + 0.5, PAD.top);
  ctx.lineTo(x0 + 0.5, y0 + 0.5);
  ctx.lineTo(x0 + plotW, y0 + 0.5);
  ctx.stroke();

  if (finite.length >= 1) {
    const points = downsample(history, Math.max(2, Math.floor(plotW)));
    const lastEpoch = Math.max(1, history.length - 1);
    ctx.strokeStyle = LINE;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    let started = false;
    for (const [i, v] of points) {
      if (!Number.isFinite(v)) continue;
      const x = x0 + (i / lastEpoch) * plotW;
      const y = y0 - (Math.min(v, yMax) / yMax) * plotH;
      if (started) ctx.lineTo(x, y);
      else {
        ctx.moveTo(x, y);
        started = true;
      }
    }
    ctx.stroke();

    const last = history[history.length - 1];
    ctx.fillStyle = LINE;
    ctx.textAlign = 'right';
    ctx.font = '12px system-ui, -apple-system, "Malgun Gothic", sans-serif';
    ctx.fillText(`손실 ${Number.isFinite(last) ? last.toFixed(4) : '—'}`, width - PAD.right, PAD.top - 4);

    ctx.fillStyle = TEXT;
    ctx.font = '11px system-ui, -apple-system, "Malgun Gothic", sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`epoch ${history.length}`, width - PAD.right, y0 + 12);
  } else {
    ctx.fillStyle = TEXT;
    ctx.textAlign = 'center';
    ctx.fillText('아직 학습 기록이 없습니다', x0 + plotW / 2, PAD.top + plotH / 2);
  }
  ctx.restore();
}
