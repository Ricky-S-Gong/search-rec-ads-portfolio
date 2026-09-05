import { useState } from 'react';
import './GoodBooksMetricsExplorer.css';

type MetricKey = 'ndcg_at_10' | 'recall_at_10' | 'precision_at_10' | 'rmse' | 'mae' | 'training_seconds';

export interface GoodBooksModelMetric {
  model: string;
  model_label: string;
  model_role: string;
  ndcg_at_10: number;
  recall_at_10: number;
  precision_at_10: number;
  rmse: number;
  mae: number;
  training_seconds: number;
}

const metrics: Array<{ key: MetricKey; higherIsBetter: boolean; label: [string, string] }> = [
  { key: 'ndcg_at_10', higherIsBetter: true, label: ['NDCG@10', 'NDCG@10'] },
  { key: 'recall_at_10', higherIsBetter: true, label: ['Recall@10', 'Recall@10'] },
  { key: 'precision_at_10', higherIsBetter: true, label: ['Precision@10', 'Precision@10'] },
  { key: 'rmse', higherIsBetter: false, label: ['RMSE', 'RMSE'] },
  { key: 'mae', higherIsBetter: false, label: ['MAE', 'MAE'] },
  { key: 'training_seconds', higherIsBetter: false, label: ['Training time', '训练时间'] },
];

const format = (key: MetricKey, value: number, zh: boolean) => {
  if (key.endsWith('_at_10')) return `${(value * 100).toFixed(2)}%`;
  if (key === 'training_seconds') return `${value.toFixed(2)} ${zh ? '秒' : 's'}`;
  return value.toFixed(3);
};

export default function GoodBooksMetricsExplorer({
  lang,
  models,
  champion,
}: {
  lang: 'en' | 'zh';
  models: GoodBooksModelMetric[];
  champion: string;
}) {
  const zh = lang === 'zh';
  const [metricKey, setMetricKey] = useState<MetricKey>('ndcg_at_10');
  const metric = metrics.find((entry) => entry.key === metricKey)!;
  const values = models.map((model) => model[metricKey]);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = high - low || 1;
  const best = models
    .filter((model) => model.model_role !== 'additional diagnostic')
    .reduce((current, model) => (
      metric.higherIsBetter
        ? model[metricKey] > current[metricKey] ? model : current
        : model[metricKey] < current[metricKey] ? model : current
    ));

  return (
    <section className="goodbooks-explorer" aria-label={zh ? '模型指标比较' : 'Model metric comparison'}>
      <div className="goodbooks-explorer__topline">
        <div>
          <p>{zh ? '切换一个指标；每次只比较同一列。' : 'Choose one metric; compare values only within that column.'}</p>
          <strong>{metric.higherIsBetter ? (zh ? '越高越好' : 'Higher is better') : (zh ? '越低越好' : 'Lower is better')}</strong>
        </div>
        <div className="goodbooks-explorer__controls" aria-label={zh ? '选择指标' : 'Select a metric'}>
          {metrics.map((entry) => (
            <button
              key={entry.key}
              type="button"
              aria-pressed={entry.key === metricKey}
              onClick={() => setMetricKey(entry.key)}
            >
              {entry.label[zh ? 1 : 0]}
            </button>
          ))}
        </div>
      </div>
      <div className="goodbooks-explorer__rows">
        {models.map((model) => {
          const normalized = metric.higherIsBetter
            ? (model[metricKey] - low) / span
            : (high - model[metricKey]) / span;
          const isBest = model.model === best.model;
          return (
            <article className={model.model_role === 'additional diagnostic' ? 'is-diagnostic' : ''} key={model.model}>
              <div className="goodbooks-explorer__label">
                <span>{model.model_label}</span>
                {model.model === champion && <small>{zh ? '离线候选' : 'Offline candidate'}</small>}
                {model.model_role === 'additional diagnostic' && <small>{zh ? '诊断' : 'Diagnostic'}</small>}
              </div>
              <div className="goodbooks-explorer__bar" aria-hidden="true"><i style={{ width: `${18 + normalized * 82}%` }} /></div>
              <strong className={isBest ? 'is-best' : ''}>{format(metricKey, model[metricKey], zh)}</strong>
            </article>
          );
        })}
      </div>
      <p className="goodbooks-explorer__note">
        {zh
          ? '“离线候选”固定使用五个正式模型中 NDCG@10 最高者。Bias-aware ALS 是额外诊断，不参与冠军选择。'
          : 'The offline candidate is the planned model with the highest NDCG@10. Bias-aware ALS is an additional diagnostic and never selects the champion.'}
      </p>
    </section>
  );
}
