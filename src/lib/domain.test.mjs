import assert from 'node:assert/strict';
import test from 'node:test';
import { parseAlgorithmDomain } from './domain.ts';

test('algorithm domain parser accepts every supported domain', () => {
  assert.equal(parseAlgorithmDomain('search'), 'search');
  assert.equal(parseAlgorithmDomain('ads'), 'ads');
  assert.equal(parseAlgorithmDomain('recommendation'), 'recommendation');
});

test('algorithm domain parser falls back for missing and unknown values', () => {
  assert.equal(parseAlgorithmDomain(null), 'recommendation');
  assert.equal(parseAlgorithmDomain('unknown'), 'recommendation');
});

test('algorithm domain parser uses the first value from repeated query parameters', () => {
  const params = new URLSearchParams('domain=search&domain=ads');
  assert.equal(parseAlgorithmDomain(params.get('domain')), 'search');
});
