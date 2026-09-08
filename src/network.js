// network.js — 다층 퍼셉트론(MLP) 학습 엔진.
// 표기: 행렬은 (특징 × 샘플). 즉 열 하나가 샘플 하나다.
//   z[l] = W[l] · a[l-1] + b[l],  a[l] = f(z[l]),  a[-1] = X
// 출력층은 항상 시그모이드 + 이진 교차 엔트로피(BCE)를 쓴다.

import { matmul, transpose, add, scale, hadamard, sumRows, zeros, map, addColumnVector, shape } from './matrix.js';
import { sigmoid, resolveActivation } from './activations.js';
import { mulberry32, makeGaussian } from './random.js';

const EPS = 1e-12; // log(0)을 막기 위한 클리핑 폭

export class Network {
  /**
   * @param {number[]} layerSizes 예: [2, 4, 4, 1] — 입력 2, 은닉 4·4, 출력 1
   * @param {string|object} hiddenActivation 은닉층 활성화 ('tanh' | 'relu' | 'sigmoid')
   * @param {{seed?: number, rng?: () => number}} [options] 초기화 난수 제어(재현용)
   */
  constructor(layerSizes, hiddenActivation = 'tanh', options = {}) {
    if (!Array.isArray(layerSizes) || layerSizes.length < 2) {
      throw new RangeError('Network: layerSizes는 길이 2 이상의 배열이어야 합니다. 예: [2, 4, 1].');
    }
    layerSizes.forEach((n, i) => {
      if (!Number.isInteger(n) || n <= 0) {
        throw new RangeError(`Network: layerSizes[${i}]는 1 이상의 정수여야 합니다 — 받은 값 ${n}.`);
      }
    });

    this.layerSizes = layerSizes.slice();
    this.hiddenActivation = resolveActivation(hiddenActivation);
    this.outputActivation = sigmoid;

    const depth = layerSizes.length - 1;
    this.acts = new Array(depth);
    for (let l = 0; l < depth; l++) {
      this.acts[l] = l === depth - 1 ? this.outputActivation : this.hiddenActivation;
    }

    const random = options.rng ?? mulberry32(options.seed ?? Math.floor(Math.random() * 4294967296));
    const gaussian = makeGaussian(random);

    this.W = new Array(depth);
    this.b = new Array(depth);
    for (let l = 0; l < depth; l++) {
      const fanIn = layerSizes[l];
      const fanOut = layerSizes[l + 1];
      const s = this.acts[l].initScale(fanIn);
      this.W[l] = map(zeros(fanOut, fanIn), () => gaussian() * s);
      this.b[l] = zeros(fanOut, 1);
    }
  }

  get depth() {
    return this.W.length;
  }

  get parameterCount() {
    let n = 0;
    for (let l = 0; l < this.depth; l++) n += this.W[l].length * this.W[l][0].length + this.b[l].length;
    return n;
  }

  /**
   * 순전파. X는 (입력특징 × m) 행렬.
   * @returns {{A: number[][][], Z: number[][][], output: number[][]}}
   *   A[0] = X, A[l+1] = f(Z[l]), Z[l]은 l번째 층의 가중합.
   */
  forward(X) {
    const [rows] = shape(X, 'forward: X');
    if (rows !== this.layerSizes[0]) {
      throw new RangeError(
        `forward: 입력 X의 행 수(${rows})가 입력층 크기(${this.layerSizes[0]})와 다릅니다. X는 (특징 × 샘플) 행렬이어야 합니다.`
      );
    }
    const A = [X];
    const Z = [];
    for (let l = 0; l < this.depth; l++) {
      const z = addColumnVector(matmul(this.W[l], A[l]), this.b[l]);
      const a = map(z, this.acts[l].fn);
      Z.push(z);
      A.push(a);
    }
    return { A, Z, output: A[A.length - 1] };
  }

  predict(X) {
    return this.forward(X).output;
  }

  /** 이진 교차 엔트로피 (샘플 평균). a는 [1e-12, 1-1e-12]로 클리핑. */
  loss(A_out, Y) {
    const [ar, ac] = shape(A_out, 'loss: A_out');
    const [yr, yc] = shape(Y, 'loss: Y');
    if (ar !== yr || ac !== yc) {
      throw new RangeError(`loss: 차원 불일치 — A_out(${ar}×${ac})과 Y(${yr}×${yc})의 형태가 같아야 합니다.`);
    }
    let sum = 0;
    for (let i = 0; i < ar; i++) {
      for (let j = 0; j < ac; j++) {
        const a = Math.min(1 - EPS, Math.max(EPS, A_out[i][j]));
        const y = Y[i][j];
        sum += -(y * Math.log(a) + (1 - y) * Math.log(1 - a));
      }
    }
    return sum / ac;
  }

  /**
   * 역전파.
   *   출력층 δ = a - y            (시그모이드 + BCE가 결합되어 도함수가 상쇄된다)
   *   은닉층 δ[l] = (W[l+1]ᵀ · δ[l+1]) ⊙ f'(z[l])
   *   dW[l] = (1/m)·δ[l]·a[l-1]ᵀ,  db[l] = (1/m)·Σ δ[l]
   * @returns {{dW: number[][][], db: number[][][]}}
   */
  backward(cache, Y) {
    if (!cache || !Array.isArray(cache.A) || !Array.isArray(cache.Z)) {
      throw new TypeError('backward: cache는 forward()가 돌려준 { A, Z } 객체여야 합니다.');
    }
    const L = this.depth;
    const A = cache.A;
    const Z = cache.Z;
    const [yr, m] = shape(Y, 'backward: Y');
    const [ar, ac] = shape(A[L], 'backward: A_out');
    if (ar !== yr || ac !== m) {
      throw new RangeError(`backward: 차원 불일치 — 출력(${ar}×${ac})과 Y(${yr}×${m})의 형태가 같아야 합니다.`);
    }

    const dW = new Array(L);
    const db = new Array(L);
    let delta = add(A[L], scale(Y, -1)); // a - y

    for (let l = L - 1; l >= 0; l--) {
      dW[l] = scale(matmul(delta, transpose(A[l])), 1 / m);
      db[l] = scale(sumRows(delta), 1 / m);
      if (l > 0) {
        const back = matmul(transpose(this.W[l]), delta);
        delta = hadamard(back, map(Z[l - 1], this.acts[l - 1].derivative));
      }
    }
    return { dW, db };
  }

  /** 경사하강 한 걸음: W ← W − lr·dW, b ← b − lr·db */
  step(grads, learningRate) {
    if (!grads || !Array.isArray(grads.dW) || !Array.isArray(grads.db)) {
      throw new TypeError('step: grads는 backward()가 돌려준 { dW, db } 객체여야 합니다.');
    }
    if (grads.dW.length !== this.depth || grads.db.length !== this.depth) {
      throw new RangeError(
        `step: 기울기 층 수(dW ${grads.dW.length}, db ${grads.db.length})가 망의 층 수(${this.depth})와 다릅니다.`
      );
    }
    if (typeof learningRate !== 'number' || !Number.isFinite(learningRate)) {
      throw new TypeError(`step: learningRate는 유한한 수여야 합니다 — 받은 값 ${learningRate}.`);
    }
    for (let l = 0; l < this.depth; l++) {
      this.W[l] = add(this.W[l], scale(grads.dW[l], -learningRate));
      this.b[l] = add(this.b[l], scale(grads.db[l], -learningRate));
    }
  }

  /**
   * 모든 파라미터를 1차원 배열로 직렬화. 순서는 W[0], b[0], W[1], b[1], ...
   * 각 행렬 안에서는 행 우선(row-major). 그래디언트 체킹 전용.
   */
  getFlatParams() {
    const out = [];
    for (let l = 0; l < this.depth; l++) {
      const W = this.W[l];
      for (let i = 0; i < W.length; i++) for (let j = 0; j < W[i].length; j++) out.push(W[i][j]);
      const b = this.b[l];
      for (let i = 0; i < b.length; i++) out.push(b[i][0]);
    }
    return out;
  }

  setFlatParams(flat) {
    if (!Array.isArray(flat) && !ArrayBuffer.isView(flat)) {
      throw new TypeError('setFlatParams: 1차원 배열이어야 합니다.');
    }
    const expected = this.parameterCount;
    if (flat.length !== expected) {
      throw new RangeError(`setFlatParams: 길이 불일치 — ${expected}개가 필요한데 ${flat.length}개를 받았습니다.`);
    }
    let k = 0;
    for (let l = 0; l < this.depth; l++) {
      const W = this.W[l];
      for (let i = 0; i < W.length; i++) for (let j = 0; j < W[i].length; j++) W[i][j] = flat[k++];
      const b = this.b[l];
      for (let i = 0; i < b.length; i++) b[i][0] = flat[k++];
    }
  }
}
