// random.js — 시드로 재현 가능한 난수.
// 데이터 생성(data.js)과 가중치 초기화(network.js)가 공유한다.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function random() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 균등 난수 생성기를 표준정규(평균 0, 표준편차 1) 생성기로 바꾼다.
 * Box–Muller 변환. 한 번에 두 개가 나오므로 하나는 캐시해 둔다.
 */
export function makeGaussian(random) {
  let spare = null;
  return function gaussian() {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    while (u === 0) u = random(); // log(0) 방지
    const r = Math.sqrt(-2 * Math.log(u));
    const theta = 2 * Math.PI * random();
    spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}
