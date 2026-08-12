import assert from 'node:assert/strict';
import test from 'node:test';

import { chooseRecommendations, formatPercent } from './movielens-demo.mjs';

test('chooseRecommendations switches methods without mutating the sample', () => {
  const sample = { userCf: [{ title: 'A' }], itemCf: [{ title: 'B' }] };
  assert.equal(chooseRecommendations(sample, 'userCf')[0].title, 'A');
  assert.equal(chooseRecommendations(sample, 'itemCf')[0].title, 'B');
  assert.deepEqual(sample.userCf, [{ title: 'A' }]);
});

test('formatPercent converts ratios into readable percentages', () => {
  assert.equal(formatPercent(0.0235906, 2), '2.36%');
});
