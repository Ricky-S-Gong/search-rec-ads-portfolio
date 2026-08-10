export function weightedCosine(a: number[], b: number[], weights: number[]): number {
  if (a.length !== b.length || a.length !== weights.length) throw new Error('Feature and weight lengths must match.');
  let product = 0;
  let normA = 0;
  let normB = 0;
  for (let index = 0; index < weights.length; index += 1) {
    const weightedA = a[index] * weights[index];
    const weightedB = b[index] * weights[index];
    product += weightedA * weightedB;
    normA += weightedA * weightedA;
    normB += weightedB * weightedB;
  }
  return normA && normB ? product / Math.sqrt(normA * normB) : 0;
}
