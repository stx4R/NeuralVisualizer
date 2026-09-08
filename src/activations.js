// activations.js — 활성화 함수 정의.
// 각 활성화는 { name, fn, derivative, initScale } 형태.
//  - fn(z)         : 활성값 a = f(z)
//  - derivative(z) : f'(z)  (활성값 a가 아니라 '가중합 z'를 받는다)
//  - initScale(fanIn) : 가중치 초기화 표준편차 계수

/** tanh — Xavier 초기화: sqrt(1 / fanIn) */
export const tanh = {
  name: 'tanh',
  fn: (z) => Math.tanh(z),
  derivative: (z) => {
    const t = Math.tanh(z);
    return 1 - t * t;
  },
  initScale: (fanIn) => Math.sqrt(1 / fanIn),
};

/** ReLU — He 초기화: sqrt(2 / fanIn) */
export const relu = {
  name: 'relu',
  fn: (z) => (z > 0 ? z : 0),
  // z = 0에서 미분 불가. 관례대로 0으로 둔다.
  derivative: (z) => (z > 0 ? 1 : 0),
  initScale: (fanIn) => Math.sqrt(2 / fanIn),
};

/** 로지스틱 시그모이드 — Xavier 초기화: sqrt(1 / fanIn) */
export const sigmoid = {
  name: 'sigmoid',
  // 큰 |z|에서 exp가 넘치지 않도록 부호별로 나눠 계산한다.
  fn: (z) => {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const e = Math.exp(z);
    return e / (1 + e);
  },
  derivative: (z) => {
    const s = sigmoid.fn(z);
    return s * (1 - s);
  },
  initScale: (fanIn) => Math.sqrt(1 / fanIn),
};

export const activations = { tanh, relu, sigmoid };

export function resolveActivation(spec) {
  if (typeof spec === 'string') {
    const found = activations[spec];
    if (!found) {
      throw new RangeError(
        `알 수 없는 활성화 함수 '${spec}'. 가능한 값: ${Object.keys(activations).join(', ')}.`
      );
    }
    return found;
  }
  if (spec && typeof spec.fn === 'function' && typeof spec.derivative === 'function' && typeof spec.initScale === 'function') {
    return spec;
  }
  throw new TypeError("활성화 함수는 이름 문자열이거나 { fn, derivative, initScale } 객체여야 합니다.");
}
