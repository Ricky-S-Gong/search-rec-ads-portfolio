import assert from 'node:assert/strict';
import test from 'node:test';
import { weightedCosine } from './similarity.ts';

test('weighted cosine handles identical, orthogonal, and emphasized vectors', () => {
  assert.equal(weightedCosine([1, 0], [1, 0], [1, 1]), 1);
  assert.equal(weightedCosine([1, 0], [0, 1], [1, 1]), 0);
  assert.ok(weightedCosine([1, 1], [1, 0], [2, 1]) > weightedCosine([1, 1], [1, 0], [1, 2]));
});
test('weighted cosine is deterministic and returns zero for a zero weighted vector', () => {
  const first = weightedCosine([.2, .8, .4], [.3, .7, .5], [1, 2, .5]);
  const second = weightedCosine([.2, .8, .4], [.3, .7, .5], [1, 2, .5]);
  assert.equal(first, second);
  assert.equal(weightedCosine([1, 0], [1, 0], [0, 0]), 0);
  assert.ok(first >= 0 && first <= 1);
});
