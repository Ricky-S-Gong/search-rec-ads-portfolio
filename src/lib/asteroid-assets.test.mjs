import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const asteroidAssets = [
  'public/images/cosmos/asteroids/bennu-cutout.webp',
  'public/images/cosmos/asteroids/itokawa-cutout.webp',
];

function webpHasAlpha(buffer) {
  if (buffer.includes(Buffer.from('ALPH'))) return true;

  const extendedHeader = buffer.indexOf(Buffer.from('VP8X'));
  if (extendedHeader >= 0 && (buffer[extendedHeader + 8] & 0x10) !== 0) return true;

  const losslessHeader = buffer.indexOf(Buffer.from('VP8L'));
  if (losslessHeader < 0 || buffer[losslessHeader + 8] !== 0x2f) return false;
  const losslessBits = buffer.readUInt32LE(losslessHeader + 9);
  return (losslessBits & 0x10000000) !== 0;
}

test('asteroid images have real transparency so their source rectangles cannot show', async () => {
  for (const asset of asteroidAssets) {
    const image = await readFile(asset);
    assert.equal(webpHasAlpha(image), true, `${asset} must include a WebP alpha channel`);
  }
});

test('asteroids render normally instead of relying on screen blending to hide black backgrounds', async () => {
  const component = await readFile('src/components/cosmos/AsteroidLayer.astro', 'utf8');
  assert.doesNotMatch(component, /mix-blend-mode\s*:\s*screen/);
});
