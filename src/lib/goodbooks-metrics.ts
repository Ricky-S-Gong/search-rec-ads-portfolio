export type GoodBooksMetricKey =
  | 'rmse'
  | 'mae'
  | 'precision_at_5'
  | 'precision_at_10'
  | 'precision_at_20'
  | 'recall_at_5'
  | 'recall_at_10'
  | 'recall_at_20'
  | 'ndcg_at_5'
  | 'ndcg_at_10'
  | 'ndcg_at_20'
  | 'training_seconds'
  | 'inference_seconds';

export interface GoodBooksMetricRow {
  model: string;
  model_role: string;
  rmse: number;
  mae: number;
  precision_at_5: number;
  precision_at_10: number;
  precision_at_20: number;
  recall_at_5: number;
  recall_at_10: number;
  recall_at_20: number;
  ndcg_at_5: number;
  ndcg_at_10: number;
  ndcg_at_20: number;
  training_seconds: number;
  inference_seconds: number;
}

export const goodbooksTableMetrics: Array<{
  key: GoodBooksMetricKey;
  higherIsBetter: boolean;
  group: 'rating' | 'precision' | 'recall' | 'ndcg' | 'cost';
  label: [string, string];
}> = [
  { key: 'rmse', higherIsBetter: false, group: 'rating', label: ['RMSE', 'RMSE'] },
  { key: 'mae', higherIsBetter: false, group: 'rating', label: ['MAE', 'MAE'] },
  { key: 'precision_at_5', higherIsBetter: true, group: 'precision', label: ['P@5', 'P@5'] },
  { key: 'precision_at_10', higherIsBetter: true, group: 'precision', label: ['P@10', 'P@10'] },
  { key: 'precision_at_20', higherIsBetter: true, group: 'precision', label: ['P@20', 'P@20'] },
  { key: 'recall_at_5', higherIsBetter: true, group: 'recall', label: ['R@5', 'R@5'] },
  { key: 'recall_at_10', higherIsBetter: true, group: 'recall', label: ['R@10', 'R@10'] },
  { key: 'recall_at_20', higherIsBetter: true, group: 'recall', label: ['R@20', 'R@20'] },
  { key: 'ndcg_at_5', higherIsBetter: true, group: 'ndcg', label: ['NDCG@5', 'NDCG@5'] },
  { key: 'ndcg_at_10', higherIsBetter: true, group: 'ndcg', label: ['NDCG@10', 'NDCG@10'] },
  { key: 'ndcg_at_20', higherIsBetter: true, group: 'ndcg', label: ['NDCG@20', 'NDCG@20'] },
  { key: 'training_seconds', higherIsBetter: false, group: 'cost', label: ['Training time', '训练时间'] },
  { key: 'inference_seconds', higherIsBetter: false, group: 'cost', label: ['Inference time', '推理时间'] },
];

export const goodbooksMetricOptions = goodbooksTableMetrics.filter((metric) => (
  metric.key === 'ndcg_at_10' || metric.key === 'recall_at_10' || metric.key === 'precision_at_10'
  || metric.key === 'rmse' || metric.key === 'mae' || metric.key === 'training_seconds' || metric.key === 'inference_seconds'
));

export function metricDefinition(key: GoodBooksMetricKey) {
  const metric = goodbooksTableMetrics.find((entry) => entry.key === key);
  if (!metric) throw new Error(`unknown GoodBooks metric: ${key}`);
  return metric;
}

export function formatGoodbooksMetric(key: GoodBooksMetricKey, value: number, zh: boolean) {
  if (key.includes('_at_')) return `${(value * 100).toFixed(2)}%`;
  if (key.endsWith('_seconds')) return `${value.toFixed(2)} ${zh ? '秒' : 's'}`;
  return value.toFixed(3);
}

export function selectBestPlannedModel<T extends GoodBooksMetricRow>(
  models: T[],
  key: GoodBooksMetricKey,
  higherIsBetter: boolean,
) {
  const planned = models.filter((model) => model.model_role === 'planned');
  if (!planned.length) throw new Error('at least one planned model is required');
  return planned.reduce((current, model) => (
    higherIsBetter
      ? model[key] > current[key] ? model : current
      : model[key] < current[key] ? model : current
  ));
}

export function bestModelsForMetric<T extends GoodBooksMetricRow>(models: T[], key: GoodBooksMetricKey) {
  if (!models.length) return [];
  const { higherIsBetter } = metricDefinition(key);
  const best = models.reduce((value, model) => (
    higherIsBetter ? Math.max(value, model[key]) : Math.min(value, model[key])
  ), models[0][key]);
  return models.filter((model) => model[key] === best).map((model) => model.model);
}

export function barWidth(value: number, values: number[]) {
  const maximum = Math.max(0, ...values);
  return maximum ? Math.min(100, Math.max(0, (value / maximum) * 100)) : 0;
}
