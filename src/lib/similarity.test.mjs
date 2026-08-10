import assert from 'node:assert/strict';
import test from 'node:test';
import { formatCosinePercent, weightedCosine } from './similarity.ts';

test('weighted cosine handles identical, orthogonal, and emphasized vectors', () => {
  assert.equal(weightedCosine([1, 0], [1, 0], [1, 1]), 1);
  assert.equal(weightedCosine([1, 0], [0, 1], [1, 1]), 0);
  assert.ok(weightedCosine([1, 1], [1, 0], [2, 1]) > weightedCosine([1, 1], [1, 0], [1, 2]));
});
test('cosine formatter converts valid values to signed percentages', () => {
  assert.equal(formatCosinePercent(.9986), '99.86%');
  assert.equal(formatCosinePercent(-.125), '-12.50%');
  assert.equal(formatCosinePercent(1), '100.00%');
  assert.equal(formatCosinePercent(0), '0.00%');
  assert.equal(formatCosinePercent(.12345, 3), '12.345%');
});
test('cosine formatter clamps floating error and rejects invalid values', () => {
  assert.equal(formatCosinePercent(1 + 1e-13), '100.00%');
  assert.equal(formatCosinePercent(-1 - 1e-13), '-100.00%');
  assert.equal(formatCosinePercent(Number.NaN), '—');
  assert.equal(formatCosinePercent(Number.POSITIVE_INFINITY), '—');
  assert.equal(formatCosinePercent(1.01), '—');
  assert.equal(formatCosinePercent(-1.01), '—');
});
test('weighted cosine is deterministic and returns zero for a zero weighted vector', () => {
  const first = weightedCosine([.2, .8, .4], [.3, .7, .5], [1, 2, .5]);
  const second = weightedCosine([.2, .8, .4], [.3, .7, .5], [1, 2, .5]);
  assert.equal(first, second);
  assert.equal(weightedCosine([1, 0], [1, 0], [0, 0]), 0);
  assert.ok(first >= 0 && first <= 1);
});
