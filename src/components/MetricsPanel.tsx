import { useEffect, useState } from 'react';

type Metrics = {
  dataset: { rawRows: number; deduplicatedRows: number; demoRows: number };
  evaluation: {
    exactLatencyMs: number; clusterLatencyMs: number; euclideanOverlapWithCosineAt10: number;
    clusterRecallAt10: number; diversityAt10: number; coverageAt10: number;
    neighborhoodDistanceAt10: number; recommendedPopularityMean: number; catalogPopularityMean: number;
    clusterSilhouette: number;
  };
};

const assetBase = `${import.meta.env.BASE_URL.replace(/\/$/, '')}/`;

export default function MetricsPanel({ lang }: { lang: 'en' | 'zh' }) {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    fetch(`${assetBase}artifacts/spotify/metrics.json`)
      .then((response) => response.json() as Promise<Metrics>)
      .then(setMetrics)
      .catch((requestError) => { console.error(requestError); setError(true); });
  }, []);
  if (error) return <p role="alert">{lang === 'en' ? 'Metrics did not load. Regenerate the experiment artifacts and refresh.' : '指标加载失败，请重新生成实验产物并刷新页面。'}</p>;
  if (!metrics) return <p className="muted">{lang === 'en' ? 'Loading reproduced results…' : '正在加载复现实验结果…'}</p>;
  const items = lang === 'en'
    ? [
        ['Exact query', `${metrics.evaluation.exactLatencyMs.toFixed(2)} ms`],
        ['Cluster query', `${metrics.evaluation.clusterLatencyMs.toFixed(2)} ms`],
        ['Cluster recall@10', `${(metrics.evaluation.clusterRecallAt10 * 100).toFixed(1)}%`],
        ['KNN overlap with cosine', `${(metrics.evaluation.euclideanOverlapWithCosineAt10 * 100).toFixed(1)}%`],
        ['Diversity@10', metrics.evaluation.diversityAt10.toFixed(3)],
        ['Popularity delta', (metrics.evaluation.recommendedPopularityMean - metrics.evaluation.catalogPopularityMean).toFixed(2)],
      ]
    : [
        ['精确查询耗时', `${metrics.evaluation.exactLatencyMs.toFixed(2)} ms`],
        ['聚类查询耗时', `${metrics.evaluation.clusterLatencyMs.toFixed(2)} ms`],
        ['聚类召回 Recall@10', `${(metrics.evaluation.clusterRecallAt10 * 100).toFixed(1)}%`],
        ['KNN 与余弦重合率', `${(metrics.evaluation.euclideanOverlapWithCosineAt10 * 100).toFixed(1)}%`],
        ['多样性@10', metrics.evaluation.diversityAt10.toFixed(3)],
        ['热门度差值', (metrics.evaluation.recommendedPopularityMean - metrics.evaluation.catalogPopularityMean).toFixed(2)],
      ];
  return <div className="metric-grid">{items.map(([label, value]) => <div className="metric" key={label}><span className="eyebrow">{label}</span><strong>{value}</strong></div>)}</div>;
}
