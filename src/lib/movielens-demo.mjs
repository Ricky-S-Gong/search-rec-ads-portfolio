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
