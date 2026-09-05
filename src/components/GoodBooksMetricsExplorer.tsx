import { useState } from 'react';
import './GoodBooksMetricsExplorer.css';
import {
  barWidth,
  formatGoodbooksMetric,
  goodbooksMetricOptions,
  selectBestPlannedModel,
  type GoodBooksMetricKey,
  type GoodBooksMetricRow,
} from '../lib/goodbooks-metrics';

export interface GoodBooksModelMetric extends GoodBooksMetricRow {
  model_label: string;
}

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
  const [metricKey, setMetricKey] = useState<GoodBooksMetricKey>('ndcg_at_10');
  const metric = goodbooksMetricOptions.find((entry) => entry.key === metricKey)!;
  const values = models.map((model) => model[metricKey]);
  const best = selectBestPlannedModel(models, metricKey, metric.higherIsBetter);

  return (
    <section className="goodbooks-explorer" aria-label={zh ? '模型指标比较' : 'Model metric comparison'}>
      <div className="goodbooks-explorer__topline">
        <div>
          <p>{zh ? '切换一个指标；每次只比较同一列。' : 'Choose one metric; compare values only within that column.'}</p>
          <strong>{metric.higherIsBetter ? (zh ? '越高越好' : 'Higher is better') : (zh ? '越低越好' : 'Lower is better')}</strong>
        </div>
        <div className="goodbooks-explorer__controls" aria-label={zh ? '选择指标' : 'Select a metric'}>
          {goodbooksMetricOptions.map((entry) => (
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
          const isBest = model.model === best.model;
          return (
            <article className={model.model_role === 'additional diagnostic' ? 'is-diagnostic' : ''} key={model.model}>
              <div className="goodbooks-explorer__label">
                <span>{model.model_label}</span>
                {model.model === champion && <small>{zh ? '离线候选' : 'Offline candidate'}</small>}
                {model.model_role === 'additional diagnostic' && <small>{zh ? '诊断' : 'Diagnostic'}</small>}
              </div>
              <div className="goodbooks-explorer__bar" aria-hidden="true"><i style={{ width: `${barWidth(model[metricKey], values)}%` }} /></div>
              <strong className={isBest ? 'is-best' : ''}>{formatGoodbooksMetric(metricKey, model[metricKey], zh)}</strong>
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
