import { createHash } from 'node:crypto';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const root = new URL('../', import.meta.url);
const reportPath = 'research/kuairand/reports/1k/2026-10-17-logq-vs-popularity-vs-two-tower.md';
const purePath = 'research/kuairand/reports/week1/evidence/content-v1/stage_report.json';

export async function buildArtifact() {
  const report = await readFile(new URL(reportPath, root), 'utf8');
  const blocks = [...report.matchAll(/```json\s*\n([\s\S]*?)\n```/g)];
  if (blocks.length !== 1) throw new Error('Expected one authoritative JSON evidence block');
  const data = JSON.parse(blocks[0][1]);
  const pure = JSON.parse(await readFile(new URL(purePath, root), 'utf8'));
  const evidence = data.evidence;
  const history = evidence['logq-validation-history.json'];
  const popularity = evidence['popularity-validation-metrics.json'];
  return {
    schema_version: 1,
    source: {
      report_path: reportPath,
      report_sha256: createHash('sha256').update(report).digest('hex'),
      report_url: `https://github.com/Ricky-S-Gong/search-rec-ads-portfolio/blob/5e33772e177c409c8f4ed445264cd0c33ebd40ae/${reportPath}`,
      ...evidence['provenance.json'],
      run_url: history.run_url,
      colab_url: history.colab_url,
    },
    comparison: {
      ...data.comparison,
      model_rows: data.comparison.model_rows.map(({ recall_percent, ...row }) => row),
    },
    train_queries: popularity.train_queries,
    covered_targets: popularity.covered_targets,
    epochs: history.epochs,
    early_stopping: history.early_stopping,
    matched_epoch7: data.matched_epoch7_logq_comparison,
    configs: data.configs,
    pure_history: {
      source_path: purePath,
      protocol: pure.protocol,
      models: Object.fromEntries(Object.entries(pure.models).map(([key, value]) => [key, value.metrics])),
      resume_passed: pure.resume_verification.passed,
    },
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const output = new URL('public/artifacts/kuairand/', root);
  const json = `${JSON.stringify(await buildArtifact(), null, 2)}\n`;
  if (process.argv.includes('--check')) {
    if (json !== await readFile(new URL('metrics.json', output), 'utf8')) {
      throw new Error('KuaiRand artifact is stale; run npm run artifacts:kuairand');
    }
    console.log('KuaiRand artifact matches source evidence');
  } else {
    await mkdir(output, { recursive: true });
    await writeFile(new URL('metrics.json', output), json);
    console.log('Generated public/artifacts/kuairand/metrics.json');
  }
}
