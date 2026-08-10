import assert from 'node:assert/strict';
import test from 'node:test';
import { cosineToAngularPercent, weightedCosine } from './similarity.ts';

test('weighted cosine handles identical, orthogonal, and emphasized vectors', () => {
  assert.equal(weightedCosine([1, 0], [1, 0], [1, 1]), 1);
  assert.equal(weightedCosine([1, 0], [0, 1], [1, 1]), 0);
  assert.ok(weightedCosine([1, 1], [1, 0], [2, 1]) > weightedCosine([1, 1], [1, 0], [1, 2]));
});

test('angular percentage is bounded and preserves cosine ranking', () => {
  assert.equal(cosineToAngularPercent(1), 100);
  assert.equal(cosineToAngularPercent(0), 0);
  assert.ok(cosineToAngularPercent(0.99) > cosineToAngularPercent(0.9));
  assert.equal(cosineToAngularPercent(2), 100);
});
