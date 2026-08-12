export function chooseRecommendations(sample, method) {
  return method === 'itemCf' ? sample.itemCf : sample.userCf;
}

export function formatPercent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}
