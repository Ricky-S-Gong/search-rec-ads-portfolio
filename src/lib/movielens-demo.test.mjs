import assert from 'node:assert/strict';
import test from 'node:test';

import { chooseRecommendations, formatPercent, recommendationOverlap, sampleCounts, visibleRecommendations } from './movielens-demo.mjs';

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

test('visibleRecommendations defaults to five rows and expands to ten', () => {
  const recommendations = Array.from({ length: 10 }, (_, index) => ({ movieId: index + 1 }));
  assert.equal(visibleRecommendations(recommendations, false).length, 5);
  assert.equal(visibleRecommendations(recommendations, true).length, 10);
  assert.deepEqual(visibleRecommendations(recommendations, false), recommendations.slice(0, 5));
});

test('sampleCounts distinguishes model history totals from displayed examples', () => {
  const sample = {
    activity: 42,
    historyTotal: 42,
    history: Array.from({ length: 5 }),
    relevantTestTotal: 8,
    relevantTest: Array.from({ length: 8 }),
  };

  assert.deepEqual(sampleCounts(sample), {
    historyTotal: 42,
    historyShown: 5,
    relevantTotal: 8,
    relevantShown: 5,
  });
});
