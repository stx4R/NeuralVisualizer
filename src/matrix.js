// matrix.js — 2차원 배열(number[][])을 행렬로 다루는 순수 함수 모음.
// 규칙: 입력을 변형하지 않고 항상 새 배열을 반환한다. 행 인덱스가 먼저, 열 인덱스가 나중.

/**
 * 행렬의 형태를 검사하고 [행, 열]을 돌려준다. 직사각형이 아니면 throw.
 * @param {number[][]} A
 * @param {string} name 오류 메시지에 쓸 이름
 * @returns {[number, number]}
 */
export function shape(A, name = '행렬') {
  if (!Array.isArray(A) || A.length === 0 || !Array.isArray(A[0])) {
    throw new TypeError(`${name}: 비어 있지 않은 number[][] 여야 합니다.`);
  }
  const rows = A.length;
  const cols = A[0].length;
  if (cols === 0) throw new TypeError(`${name}: 열이 0개입니다.`);
  for (let i = 1; i < rows; i++) {
    if (!Array.isArray(A[i]) || A[i].length !== cols) {
      const got = Array.isArray(A[i]) ? `${A[i].length}열` : '배열 아님';
      throw new TypeError(`${name}: 행마다 길이가 다릅니다 — 행 0은 ${cols}열, 행 ${i}은 ${got}.`);
    }
  }
  return [rows, cols];
}

function fmt(A) {
  const rows = Array.isArray(A) ? A.length : 0;
  const cols = rows && Array.isArray(A[0]) ? A[0].length : 0;
  return `${rows}×${cols}`;
}

export function zeros(r, c) {
  if (!Number.isInteger(r) || !Number.isInteger(c) || r <= 0 || c <= 0) {
    throw new RangeError(`zeros: 행과 열은 1 이상의 정수여야 합니다 — 받은 값 (${r}, ${c}).`);
  }
  const out = new Array(r);
  for (let i = 0; i < r; i++) out[i] = new Array(c).fill(0);
  return out;
}

/** 원소별 변환. fn(value, i, j) → number */
export function map(A, fn) {
  const [r, c] = shape(A, 'map: A');
  if (typeof fn !== 'function') throw new TypeError('map: fn은 함수여야 합니다.');
  const out = new Array(r);
  for (let i = 0; i < r; i++) {
    const Ai = A[i];
    const Oi = new Array(c);
    for (let j = 0; j < c; j++) Oi[j] = fn(Ai[j], i, j);
    out[i] = Oi;
  }
  return out;
}

export function matmul(A, B) {
  const [ar, ac] = shape(A, 'matmul: A');
  const [br, bc] = shape(B, 'matmul: B');
  if (ac !== br) {
    throw new RangeError(
      `matmul: 차원 불일치 — A(${fmt(A)}) · B(${fmt(B)}). A의 열 수(${ac})와 B의 행 수(${br})가 같아야 합니다.`
    );
  }
  const C = zeros(ar, bc);
  for (let i = 0; i < ar; i++) {
    const Ai = A[i];
    const Ci = C[i];
    for (let k = 0; k < ac; k++) {
      const a = Ai[k];
      if (a === 0) continue;
      const Bk = B[k];
      for (let j = 0; j < bc; j++) Ci[j] += a * Bk[j];
    }
  }
  return C;
}

export function transpose(A) {
  const [r, c] = shape(A, 'transpose: A');
  const out = zeros(c, r);
  for (let i = 0; i < r; i++) {
    const Ai = A[i];
    for (let j = 0; j < c; j++) out[j][i] = Ai[j];
  }
  return out;
}

export function add(A, B) {
  const [ar, ac] = shape(A, 'add: A');
  const [br, bc] = shape(B, 'add: B');
  if (ar !== br || ac !== bc) {
    throw new RangeError(`add: 차원 불일치 — A(${fmt(A)}) + B(${fmt(B)}). 두 행렬의 형태가 같아야 합니다.`);
  }
  const out = new Array(ar);
  for (let i = 0; i < ar; i++) {
    const Ai = A[i];
    const Bi = B[i];
    const Oi = new Array(ac);
    for (let j = 0; j < ac; j++) Oi[j] = Ai[j] + Bi[j];
    out[i] = Oi;
  }
  return out;
}

export function scale(A, k) {
  if (typeof k !== 'number' || !Number.isFinite(k)) {
    throw new TypeError(`scale: k는 유한한 수여야 합니다 — 받은 값 ${k}.`);
  }
  return map(A, (v) => v * k);
}

export function hadamard(A, B) {
  const [ar, ac] = shape(A, 'hadamard: A');
  const [br, bc] = shape(B, 'hadamard: B');
  if (ar !== br || ac !== bc) {
    throw new RangeError(`hadamard: 차원 불일치 — A(${fmt(A)}) ⊙ B(${fmt(B)}). 두 행렬의 형태가 같아야 합니다.`);
  }
  const out = new Array(ar);
  for (let i = 0; i < ar; i++) {
    const Ai = A[i];
    const Bi = B[i];
    const Oi = new Array(ac);
    for (let j = 0; j < ac; j++) Oi[j] = Ai[j] * Bi[j];
    out[i] = Oi;
  }
  return out;
}

/** 행별 합. r×c → r×1 (편향 기울기 db 계산에 쓴다) */
export function sumRows(A) {
  const [r, c] = shape(A, 'sumRows: A');
  const out = zeros(r, 1);
  for (let i = 0; i < r; i++) {
    const Ai = A[i];
    let s = 0;
    for (let j = 0; j < c; j++) s += Ai[j];
    out[i][0] = s;
  }
  return out;
}

/**
 * 열벡터를 모든 열에 브로드캐스트해서 더한다. Z(n×m) + b(n×1).
 * add()는 형태가 정확히 같아야 하므로 편향 덧셈은 이 함수를 쓴다.
 */
export function addColumnVector(A, col) {
  const [ar, ac] = shape(A, 'addColumnVector: A');
  const [cr, cc] = shape(col, 'addColumnVector: col');
  if (cc !== 1) {
    throw new RangeError(`addColumnVector: col(${fmt(col)})은 열이 1개인 열벡터여야 합니다.`);
  }
  if (cr !== ar) {
    throw new RangeError(
      `addColumnVector: 차원 불일치 — A(${fmt(A)})의 행 수(${ar})와 col(${fmt(col)})의 행 수(${cr})가 같아야 합니다.`
    );
  }
  const out = new Array(ar);
  for (let i = 0; i < ar; i++) {
    const Ai = A[i];
    const b = col[i][0];
    const Oi = new Array(ac);
    for (let j = 0; j < ac; j++) Oi[j] = Ai[j] + b;
    out[i] = Oi;
  }
  return out;
}
