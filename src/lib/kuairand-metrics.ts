export function formatKuaiRandMetric(value: number | null | undefined, metric: string, lang: 'en' | 'zh') {
  if (value == null) return lang === 'zh' ? '未提供' : 'Not provided';
  return metric === 'recall_at_100' ? `${(value * 100).toFixed(6)}%` : value.toFixed(9);
}

export function coverageBreakdown(queries: number, covered: number, hits: number) {
  if (![queries, covered, hits].every(Number.isInteger) || queries <= 0 || hits < 0 || hits > covered || covered > queries) {
    throw new Error('Invalid coverage counts');
  }
  return [queries - covered, covered - hits, hits].map((count) => ({ count, share: count / queries }));
}
