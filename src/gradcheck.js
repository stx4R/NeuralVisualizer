// gradcheck.js — 역전파로 구한 해석적 기울기를 수치 미분과 비교한다.
// 중앙 차분: (J(θ+ε) − J(θ−ε)) / (2ε)

/** 기울기를 getFlatParams()와 똑같은 순서(W[0], b[0], W[1], b[1], ...)로 평탄화. */
export function flattenGradients(grads) {
  if (!grads || !Array.isArray(grads.dW) || !Array.isArray(grads.db)) {
    throw new TypeError('flattenGradients: { dW, db } 객체여야 합니다.');
  }
  const out = [];
  for (let l = 0; l < grads.dW.length; l++) {
    const dW = grads.dW[l];
    for (let i = 0; i < dW.length; i++) for (let j = 0; j < dW[i].length; j++) out.push(dW[i][j]);
    const db = grads.db[l];
    for (let i = 0; i < db.length; i++) out.push(db[i][0]);
  }
  return out;
}

/**
 * 그래디언트 체킹.
 * @param {import('./network.js').Network} network
 * @param {number[][]} X (특징 × m)
 * @param {number[][]} Y (1 × m)
 * @param {number} epsilon 중앙 차분 폭
 * @returns {{relativeError: number, passed: boolean, worstIndex: number, analytic: number[], numeric: number[]}}
 */
export function checkGradients(network, X, Y, epsilon = 1e-4) {
  if (typeof epsilon !== 'number' || !Number.isFinite(epsilon) || epsilon <= 0) {
    throw new RangeError(`checkGradients: epsilon은 0보다 큰 유한한 수여야 합니다 — 받은 값 ${epsilon}.`);
  }

  const original = network.getFlatParams();
  const analytic = flattenGradients(network.backward(network.forward(X), Y));
  if (analytic.length !== original.length) {
    throw new RangeError(
      `checkGradients: 기울기 개수(${analytic.length})와 파라미터 개수(${original.length})가 다릅니다.`
    );
  }

  const n = original.length;
  const numeric = new Array(n);
  const probe = original.slice();
  try {
    for (let i = 0; i < n; i++) {
      probe[i] = original[i] + epsilon;
      network.setFlatParams(probe);
      const jPlus = network.loss(network.predict(X), Y);

      probe[i] = original[i] - epsilon;
      network.setFlatParams(probe);
      const jMinus = network.loss(network.predict(X), Y);

      probe[i] = original[i]; // 다음 파라미터를 위해 되돌린다
      numeric[i] = (jPlus - jMinus) / (2 * epsilon);
    }
  } finally {
    network.setFlatParams(original); // 도중에 실패해도 망은 원상 복구
  }

  // 상대오차 = ‖a − n‖ / (‖a‖ + ‖n‖)
  let diffSq = 0;
  let aSq = 0;
  let nSq = 0;
  let worstIndex = 0;
  let worstDiff = -1;
  for (let i = 0; i < n; i++) {
    const d = analytic[i] - numeric[i];
    diffSq += d * d;
    aSq += analytic[i] * analytic[i];
    nSq += numeric[i] * numeric[i];
    const ad = Math.abs(d);
    if (ad > worstDiff) {
      worstDiff = ad;
      worstIndex = i;
    }
  }
  const denom = Math.sqrt(aSq) + Math.sqrt(nSq);
  const relativeError = denom === 0 ? 0 : Math.sqrt(diffSq) / denom;

  return {
    relativeError,
    passed: relativeError < 1e-7,
    worstIndex,
    analytic,
    numeric,
  };
}
