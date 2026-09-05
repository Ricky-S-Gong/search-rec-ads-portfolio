import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import {
  barWidth,
  bestModelsForMetric,
  formatGoodbooksMetric,
  goodbooksTableMetrics,
  selectBestPlannedModel,
} from './goodbooks-metrics.ts';

const models = [
  {
    model: 'basic_mf', model_role: 'planned', rmse: 1.149, mae: 0.9,
    precision_at_5: 0.02, precision_at_10: 0.02, precision_at_20: 0.02,
    recall_at_5: 0.02, recall_at_10: 0.02, recall_at_20: 0.02,
    ndcg_at_5: 0.02, ndcg_at_10: 0.024, ndcg_at_20: 0.02,
    training_seconds: 10, inference_seconds: 2,
  },
  {
    model: 'svdpp', model_role: 'planned', rmse: 0.861, mae: 0.8,
    precision_at_5: 0.01, precision_at_10: 0.01, precision_at_20: 0.01,
    recall_at_5: 0.01, recall_at_10: 0.01, recall_at_20: 0.01,
    ndcg_at_5: 0.01, ndcg_at_10: 0.005, ndcg_at_20: 0.01,
    training_seconds: 20, inference_seconds: 3,
  },
  {
    model: 'bias_aware_als', model_role: 'additional diagnostic', rmse: 0.1, mae: 0.1,
    precision_at_5: 0.5, precision_at_10: 0.5, precision_at_20: 0.5,
    recall_at_5: 0.5, recall_at_10: 0.5, recall_at_20: 0.5,
    ndcg_at_5: 0.6, ndcg_at_10: 0.6, ndcg_at_20: 0.6,
    training_seconds: 1, inference_seconds: 1,
  },
];

test('formats ranking, rating, and duration metrics for the explorer', () => {
  assert.equal(formatGoodbooksMetric('ndcg_at_10', 0.023946, false), '2.39%');
  assert.equal(formatGoodbooksMetric('rmse', 0.860567, false), '0.861');
  assert.equal(formatGoodbooksMetric('inference_seconds', 5.599, true), '5.60 秒');
});

test('uses a zero baseline and excludes the diagnostic row from best planned model', () => {
  assert.equal(barWidth(0, [0, 2, 4]), 0);
  assert.equal(barWidth(2, [0, 2, 4]), 50);
  assert.equal(barWidth(4, [0, 2, 4]), 100);
  assert.equal(selectBestPlannedModel(models, 'ndcg_at_10', true).model, 'basic_mf');
  assert.equal(selectBestPlannedModel(models, 'rmse', false).model, 'svdpp');
});

test('defines every result-table column and marks all raw-value ties as best', () => {
  assert.equal(goodbooksTableMetrics.length, 13);
  assert.deepEqual(bestModelsForMetric(models, 'rmse'), ['bias_aware_als']);
  assert.deepEqual(bestModelsForMetric([
    { ...models[0], rmse: 0.5 },
    { ...models[1], rmse: 0.5 },
  ], 'rmse'), ['basic_mf', 'svdpp']);
  assert.deepEqual(bestModelsForMetric(models, 'precision_at_5'), ['bias_aware_als']);
});

test('published GoodBooks artifact preserves the canonical six model roles and Basic MF champion', async () => {
  const artifact = JSON.parse(await readFile(new URL('../../public/artifacts/goodbooks/metrics.json', import.meta.url), 'utf8'));
  assert.equal(artifact.schema_version, 'goodbooks-frontend-artifact-v1');
  assert.equal(artifact.champion.model, 'basic_mf');
  assert.deepEqual(artifact.models.map((model) => model.model_role), [
    'planned', 'planned', 'planned', 'planned', 'planned', 'additional diagnostic',
  ]);
  assert.equal(artifact.models.every((model) => Number.isFinite(model.inference_seconds)), true);
});
