import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { buildArtifact } from '../../scripts/build-kuairand-artifacts.mjs';
import { coverageBreakdown, formatKuaiRandMetric } from './kuairand-metrics.ts';

const artifact = JSON.parse(await readFile(new URL('../../public/artifacts/kuairand/metrics.json', import.meta.url), 'utf8'));

test('KuaiRand published artifact exactly reproduces its source report and Pure aggregates', async () => {
  assert.deepEqual(artifact, await buildArtifact());
  assert.match(artifact.source.report_sha256, /^[a-f0-9]{64}$/);
  assert.equal(artifact.source.test_evaluated, false);
  assert.equal(artifact.comparison.test_evaluated, false);
  assert.equal(artifact.comparison.model_rows.length, 4);
  for (const row of artifact.comparison.model_rows) {
    assert.ok(row.evidence_status);
    assert.ok(row.recall_at_100 >= 0 && row.recall_at_100 <= 1);
    assert.ok(row.ndcg_at_100 >= 0 && row.ndcg_at_100 <= row.recall_at_100);
  }
  assert.equal(artifact.comparison.model_rows[1].selected_epoch, null);
  assert.match(artifact.comparison.model_rows[1].evidence_status, /historical/);
});

test('KuaiRand best checkpoint, early stop and matched-epoch comparison agree with history', () => {
  assert.deepEqual(artifact.epochs.map(e => e.epoch), Array.from({ length: 17 }, (_, i) => i + 1));
  const best = [...artifact.epochs].sort((a, b) => b['validation/recall_at_100'] - a['validation/recall_at_100'] || b['validation/ndcg_at_100'] - a['validation/ndcg_at_100'])[0];
  assert.equal(best.epoch, artifact.early_stopping.best_epoch);
  assert.equal(best['validation/recall_at_100'], artifact.comparison.model_rows[3].recall_at_100);
  assert.equal(best['validation/ndcg_at_100'], artifact.comparison.model_rows[3].ndcg_at_100);
  assert.equal(artifact.epochs.at(-1).epoch, artifact.early_stopping.epoch);
  assert.equal(artifact.epochs.at(-1)['validation/stopped_early'], true);
  assert.equal(artifact.epochs.at(-1)['validation/epochs_without_improvement'], artifact.early_stopping.patience);
  const matchedEpoch = artifact.epochs.find(e => e.epoch === artifact.matched_epoch7.epoch);
  assert.equal(matchedEpoch['validation/recall_at_100'], artifact.matched_epoch7.logq_recall);
  assert.equal(matchedEpoch['validation/ndcg_at_100'], artifact.matched_epoch7.logq_ndcg);
  assert.equal(artifact.comparison.model_rows[2].recall_at_100, artifact.matched_epoch7.no_logq_recall);
  for (const e of artifact.epochs) {
    assert.equal(e['validation/queries'], artifact.comparison.queries);
    assert.equal(e['validation/catalog_size'], artifact.comparison.catalog_size);
    assert.equal(e['validation/target_coverage'], artifact.comparison.target_coverage);
    assert.ok(Math.abs(e['validation/hits'] / artifact.comparison.queries - e['validation/recall_at_100']) < 1e-12);
  }
});

test('coverage decomposes all events without dropping uncovered targets', () => {
  const n = artifact.comparison.queries;
  const best = artifact.epochs.find(e => e.epoch === artifact.early_stopping.best_epoch);
  const segments = coverageBreakdown(n, artifact.covered_targets, best['validation/hits']);
  assert.equal(segments.reduce((sum, row) => sum + row.count, 0), n);
  assert.ok(Math.abs(segments.reduce((sum, row) => sum + row.share, 0) - 1) < 1e-12);
  assert.equal(artifact.covered_targets / n, artifact.comparison.target_coverage);
  assert.equal(segments[2].share, artifact.comparison.model_rows[3].recall_at_100);
  assert.deepEqual(coverageBreakdown(10, 0, 0).map(r => r.count), [10, 0, 0]);
  assert.deepEqual(coverageBreakdown(10, 10, 10).map(r => r.count), [0, 0, 10]);
  assert.throws(() => coverageBreakdown(10, 5, 6));
  assert.throws(() => coverageBreakdown(0, 0, 0));
});

test('formatting distinguishes percentages, NDCG and missing values in both locales', () => {
  assert.equal(formatKuaiRandMetric(0.005139676466724374, 'recall_at_100', 'en'), '0.513968%');
  assert.equal(formatKuaiRandMetric(0.0012306225651929504, 'ndcg_at_100', 'zh'), '0.001230623');
  assert.equal(formatKuaiRandMetric(null, 'recall_at_100', 'en'), 'Not provided');
  assert.equal(formatKuaiRandMetric(undefined, 'ndcg_at_100', 'zh'), '未提供');
  assert.equal(formatKuaiRandMetric(0, 'recall_at_100', 'en'), '0.000000%');
});
