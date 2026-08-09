import { useEffect, useState } from 'react';

type Metrics = {
  dataset: { rawRows: number; deduplicatedRows: number; demoRows: number };
  evaluation: { exactLatencyMs: number; clusterLatencyMs: number; clusterRecallAt10: number; diversityAt10: number; coverageAt10: number };
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
        ['Deduplicated catalog', metrics.dataset.deduplicatedRows.toLocaleString()],
        ['Exact query', `${metrics.evaluation.exactLatencyMs.toFixed(2)} ms`],
        ['Cluster recall@10', `${(metrics.evaluation.clusterRecallAt10 * 100).toFixed(1)}%`],
        ['Diversity@10', metrics.evaluation.diversityAt10.toFixed(3)],
      ]
    : [
        ['去重后的歌曲目录', metrics.dataset.deduplicatedRows.toLocaleString()],
        ['精确查询耗时', `${metrics.evaluation.exactLatencyMs.toFixed(2)} ms`],
        ['聚类召回 Recall@10', `${(metrics.evaluation.clusterRecallAt10 * 100).toFixed(1)}%`],
        ['多样性@10', metrics.evaluation.diversityAt10.toFixed(3)],
      ];
  return <div className="metric-grid">{items.map(([label, value]) => <div className="metric" key={label}><span className="eyebrow">{label}</span><strong>{value}</strong></div>)}</div>;
}
