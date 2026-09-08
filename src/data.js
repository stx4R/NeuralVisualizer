// data.js — 2차원 이진 분류용 장난감 데이터셋 생성기.
// 반환 형태: { X: (2 × count) 행렬, Y: (1 × count) 행렬 }, 라벨은 0 또는 1.
// 좌표는 모두 대략 [-1, 1] 안에 들어온다.

import { mulberry32, makeGaussian } from './random.js';

const TYPES = ['xor', 'circle', 'spiral', 'gaussian'];

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * 데이터셋 생성.
 * @param {'xor'|'circle'|'spiral'|'gaussian'} type
 * @param {number} count 표본 개수
 * @param {number} noise 좌표에 더할 가우시안 잡음의 표준편차 (0 ~ 0.5, 범위 밖은 잘라냄)
 * @param {number} seed 재현용 시드
 * @returns {{X: number[][], Y: number[][]}}
 */
export function generate(type, count, noise = 0, seed = 1337) {
  if (!TYPES.includes(type)) {
    throw new RangeError(`generate: 알 수 없는 데이터 종류 '${type}'. 가능한 값: ${TYPES.join(', ')}.`);
  }
  if (!Number.isInteger(count) || count <= 0) {
    throw new RangeError(`generate: count는 1 이상의 정수여야 합니다 — 받은 값 ${count}.`);
  }
  if (typeof noise !== 'number' || !Number.isFinite(noise)) {
    throw new TypeError(`generate: noise는 유한한 수여야 합니다 — 받은 값 ${noise}.`);
  }
  const sd = clamp(noise, 0, 0.5);

  const random = mulberry32(seed);
  const gaussian = makeGaussian(random);

  // 각 생성기는 잡음 없는 좌표와 라벨을 만든다. 라벨은 '깨끗한' 좌표 기준으로 정하고,
  // 잡음은 그 뒤에 더한다 — 그래야 잡음이 실제로 경계를 흐리는 효과를 낸다.
  const points =
    type === 'xor' ? makeXor(count, random)
    : type === 'circle' ? makeCircle(count, random)
    : type === 'spiral' ? makeSpiral(count, random)
    : makeGaussianClusters(count, gaussian);

  const X = [new Array(count), new Array(count)];
  const Y = [new Array(count)];
  for (let i = 0; i < count; i++) {
    const p = points[i];
    X[0][i] = clamp(p.x + gaussian() * sd, -1, 1);
    X[1][i] = clamp(p.y + gaussian() * sd, -1, 1);
    Y[0][i] = p.label;
  }
  return { X, Y };
}

function makeXor(count, random) {
  const pad = 0.05; // 축에 딱 붙은 점은 살짝 밀어내 경계를 또렷하게 한다
  const out = new Array(count);
  for (let i = 0; i < count; i++) {
    let x = random() * 2 - 1;
    let y = random() * 2 - 1;
    x += x > 0 ? pad : -pad;
    y += y > 0 ? pad : -pad;
    out[i] = { x, y, label: x * y > 0 ? 1 : 0 };
  }
  return out;
}

function makeCircle(count, random) {
  const innerMax = 0.45;
  const ringMin = 0.65;
  const out = new Array(count);
  for (let i = 0; i < count; i++) {
    const inside = i % 2 === 0; // 두 클래스를 번갈아 만들어 개수를 맞춘다
    // sqrt를 씌워야 원판/고리 위에 고르게 퍼진다
    const r = inside
      ? innerMax * Math.sqrt(random())
      : Math.sqrt(ringMin * ringMin + random() * (1 - ringMin * ringMin));
    const theta = random() * 2 * Math.PI;
    out[i] = { x: r * Math.cos(theta), y: r * Math.sin(theta), label: inside ? 1 : 0 };
  }
  return out;
}

/** 나선: 두 갈래 아르키메데스 나선(r ∝ θ), 갈래마다 클래스가 다르다. */
function makeSpiral(count, random) {
  const turns = 1.75 * 2 * Math.PI; // 갈래당 회전량
  const out = new Array(count);
  for (let i = 0; i < count; i++) {
    const arm = i % 2; // 0번 갈래는 라벨 1, 1번 갈래는 라벨 0
    const t = (Math.floor(i / 2) + random()) / Math.ceil(count / 2); // 0~1
    const r = t; // 반지름은 최대 1
    const theta = t * turns + arm * Math.PI;
    out[i] = { x: r * Math.sin(theta), y: r * Math.cos(theta), label: arm === 0 ? 1 : 0 };
  }
  return out;
}

function makeGaussianClusters(count, gaussian) {
  const spread = 0.15; // 잡음과 별개로 군집 자체가 갖는 퍼짐
  const out = new Array(count);
  for (let i = 0; i < count; i++) {
    const positive = i % 2 === 0;
    const cx = positive ? 0.5 : -0.5;
    const cy = positive ? 0.5 : -0.5;
    out[i] = {
      x: cx + gaussian() * spread,
      y: cy + gaussian() * spread,
      label: positive ? 1 : 0,
    };
  }
  return out;
}
