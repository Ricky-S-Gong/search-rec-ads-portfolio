import assert from 'node:assert/strict';
import test from 'node:test';

import { chooseRecommendations, formatPercent, recommendationOverlap } from './movielens-demo.mjs';

test('chooseRecommendations switches methods without mutating the sample', () => {
  const sample = { userCf: [{ title: 'A' }], itemCf: [{ title: 'B' }] };
  assert.equal(chooseRecommendations(sample, 'userCf')[0].title, 'A');
  assert.equal(chooseRecommendations(sample, 'itemCf')[0].title, 'B');
  assert.deepEqual(sample.userCf, [{ title: 'A' }]);
});

test('formatPercent converts ratios into readable percentages', () => {
  assert.equal(formatPercent(0.0235906, 2), '2.36%');
});

test('chooseRecommendations supports popularity and artifact-v2 method maps', () => {
  const sample = { methods: { popularity: [{ movieId: 1 }], userCf: [], itemCf: [] } };
  assert.deepEqual(chooseRecommendations(sample, 'popularity'), [{ movieId: 1 }]);
});

test('recommendationOverlap reports shared movie ids without double counting', () => {
  const first = [{ movieId: 1 }, { movieId: 2 }, { movieId: 2 }];
  const second = [{ movieId: 2 }, { movieId: 3 }];
  assert.deepEqual(recommendationOverlap(first, second), { count: 1, movieIds: [2] });
});
