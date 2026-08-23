export function chooseRecommendations(sample, method) {
  if (sample.methods) return sample.methods[method] ?? [];
  return method === 'itemCf' ? sample.itemCf : sample.userCf;
}

export function formatPercent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

export function recommendationOverlap(first, second) {
  const secondIds = new Set(second.map((item) => item.movieId));
  const movieIds = [...new Set(first.map((item) => item.movieId))]
    .filter((movieId) => secondIds.has(movieId))
    .sort((a, b) => a - b);
  return { count: movieIds.length, movieIds };
}

export function visibleRecommendations(recommendations, expanded, defaultCount = 5) {
  return recommendations.slice(0, expanded ? 10 : defaultCount);
}

export function sampleCounts(sample, defaultCount = 5) {
  return {
    historyTotal: sample.historyTotal ?? sample.activity ?? sample.history.length,
    historyShown: Math.min(defaultCount, sample.history.length),
    relevantTotal: sample.relevantTestTotal ?? sample.relevantTest.length,
    relevantShown: Math.min(defaultCount, sample.relevantTest.length),
  };
}
