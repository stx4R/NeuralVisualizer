// proj3d.js — 3차원 회전 · 원근 투영 · 램버트 음영.
// 순수 함수만 둔다. 캔버스도, DOM도 모른다.
//
// 벡터는 {x, y, z}, 행렬은 3×3 중첩 배열([행][열]).
// matrix.js를 재사용하지 않는 이유: 그쪽은 (특징 × 샘플) 학습용 행렬이라 목적이 다르다.
// 여기서는 크기가 3으로 고정이라 루프 없이 펼쳐 쓰는 편이 짧고 빠르다.

const DEG = Math.PI / 180;

export function rotateY(deg) {
  const t = deg * DEG;
  const c = Math.cos(t);
  const s = Math.sin(t);
  return [
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c],
  ];
}

export function rotateX(deg) {
  const t = deg * DEG;
  const c = Math.cos(t);
  const s = Math.sin(t);
  return [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ];
}

/** 행렬 곱 A·B. 회전을 미리 하나로 합쳐 두면 점마다 한 번만 적용하면 된다. */
export function multiply(A, B) {
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    }
  }
  return out;
}

export function applyMatrix(M, v) {
  return {
    x: M[0][0] * v.x + M[0][1] * v.y + M[0][2] * v.z,
    y: M[1][0] * v.x + M[1][1] * v.y + M[1][2] * v.z,
    z: M[2][0] * v.x + M[2][1] * v.y + M[2][2] * v.z,
  };
}

/**
 * 원근 투영. 화면 y는 위가 +가 되도록 부호를 뒤집는다.
 * @returns {{x: number, y: number, depth: number} | null} 카메라 뒤면 null
 */
export function project(v, focal, distance) {
  const z = v.z + distance;
  if (z <= 0) return null; // 카메라 뒤 — 그릴 수 없다
  const s = focal / z;
  return { x: v.x * s, y: -v.y * s, depth: z };
}

export function subtract(a, b) {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

export function cross(a, b) {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

export function dot(a, b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function length(v) {
  return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

/** 길이가 0이면 그대로 돌려준다 (NaN을 만들지 않는다). */
export function normalize(v) {
  const len = length(v);
  if (len === 0) return { x: 0, y: 0, z: 0 };
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}

/** 세 점이 이루는 면의 단위 법선. 세 점이 한 직선 위면 {0,1,0}을 돌려준다. */
export function normal(p1, p2, p3) {
  const n = cross(subtract(p2, p1), subtract(p3, p1));
  const len = length(n);
  if (len === 0) return { x: 0, y: 1, z: 0 };
  return { x: n.x / len, y: n.y / len, z: n.z / len };
}

/** 램버트 음영: ambient 0.35 + 0.65 · max(0, n·l). 0~1로 잘라 반환. */
export function lambert(n, lightDir) {
  const d = dot(n, lightDir);
  const shade = 0.35 + 0.65 * Math.max(0, d);
  return shade < 0 ? 0 : shade > 1 ? 1 : shade;
}
