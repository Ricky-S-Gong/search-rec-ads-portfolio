import { useDeferredValue, useEffect, useMemo, useState } from 'react';
import './SpotifyLab.css';

type Lang = 'en' | 'zh';
type Track = { id: string; name: string; artist: string; year: number; popularity: number; features: number[] };
type Artifact = { version: string; featureNames: string[]; tracks: Track[] };
type ScoredTrack = Track & { score: number };

const DEFAULT_WEIGHTS = [1, 1, 1, 1, 1, 1, 1, 1, 1];
const assetBase = `${import.meta.env.BASE_URL.replace(/\/$/, '')}/`;

const ui = {
  en: {
    title: 'Tune the definition of “similar”', search: 'Seed track', placeholder: 'Search title or artist…',
    weights: 'Feature weights', results: 'Top recommendations', reset: 'Reset weights', loading: 'Loading the static experiment…',
    empty: 'No track matches that search.', zero: 'Raise at least one feature weight to calculate similarity.', score: 'similarity',
    note: 'Computed locally from a stratified 5,000-track demonstration set. No Spotify API or user data.', error: 'The experiment artifact did not load. Refresh the page or check the generated JSON.',
  },
  zh: {
    title: '调整“相似”的定义', search: '种子歌曲', placeholder: '搜索歌曲或艺术家…',
    weights: '特征权重', results: '推荐结果', reset: '重置权重', loading: '正在加载静态实验数据…',
    empty: '没有匹配的歌曲。', zero: '至少提高一个特征权重才能计算相似度。', score: '相似度',
    note: '基于分层抽取的 5,000 首演示歌曲在浏览器本地计算，不调用 Spotify API，也不使用用户数据。', error: '实验数据未能加载，请刷新页面或检查生成的 JSON。',
  },
};

const dot = (a: number[], b: number[], weights: number[]) => {
  let product = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < weights.length; i += 1) {
    const wa = a[i] * weights[i];
    const wb = b[i] * weights[i];
    product += wa * wb;
    normA += wa * wa;
    normB += wb * wb;
  }
  return normA && normB ? product / Math.sqrt(normA * normB) : 0;
};

function topMatches(seed: Track, tracks: Track[], weights: number[], count = 10): ScoredTrack[] {
  const unique = new Map<string, ScoredTrack>();
  const seedKey = `${seed.name}\u0000${seed.artist}`.toLocaleLowerCase();
  for (const track of tracks) {
    const key = `${track.name}\u0000${track.artist}`.toLocaleLowerCase();
    if (track.id === seed.id || key === seedKey) continue;
    const candidate = { ...track, score: dot(seed.features, track.features, weights) };
    const previous = unique.get(key);
    if (!previous || candidate.score > previous.score) unique.set(key, candidate);
  }
  const best: ScoredTrack[] = [];
  for (const candidate of unique.values()) {
    let index = 0;
    while (index < best.length && best[index].score >= candidate.score) index += 1;
    if (index < count) best.splice(index, 0, candidate);
    if (best.length > count) best.pop();
  }
  return best;
}

function radarPath(values: number[]) {
  return values.map((value, index) => {
    const angle = (Math.PI * 2 * index) / values.length - Math.PI / 2;
    const radius = 18 + value * 62;
    return `${index ? 'L' : 'M'} ${(90 + Math.cos(angle) * radius).toFixed(1)} ${(90 + Math.sin(angle) * radius).toFixed(1)}`;
  }).join(' ') + ' Z';
}

export default function SpotifyLab({ lang }: { lang: Lang }) {
  const t = ui[lang];
  const [artifact, setArtifact] = useState<Artifact | null>(null);
  const [error, setError] = useState(false);
  const [seed, setSeed] = useState<Track | null>(null);
  const [query, setQuery] = useState('');
  const [weights, setWeights] = useState<number[]>(() => [...DEFAULT_WEIGHTS]);
  const deferredQuery = useDeferredValue(query);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${assetBase}artifacts/spotify/demo-tracks.json`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Artifact request failed: ${response.status}`);
        return response.json() as Promise<Artifact>;
      })
      .then((data) => { setArtifact(data); setSeed(data.tracks[0] ?? null); })
      .catch((requestError: Error) => { if (requestError.name !== 'AbortError') { console.error(requestError); setError(true); } });
    return () => controller.abort();
  }, []);

  const suggestions = useMemo(() => {
    if (!artifact || !deferredQuery.trim()) return [];
    const needle = deferredQuery.toLocaleLowerCase();
    const matches: Track[] = [];
    for (const track of artifact.tracks) {
      if (`${track.name} ${track.artist}`.toLocaleLowerCase().includes(needle)) matches.push(track);
      if (matches.length === 8) break;
    }
    return matches;
  }, [artifact, deferredQuery]);

  const hasWeight = weights.some((weight) => weight > 0);
  const results = useMemo(
    () => (artifact && seed && hasWeight ? topMatches(seed, artifact.tracks, weights) : []),
    [artifact, seed, weights, hasWeight],
  );

  if (error) return <p className="lab-status" role="alert">{t.error}</p>;
  if (!artifact || !seed) return <p className="lab-status" aria-live="polite">{t.loading}</p>;

  return (
    <section className="lab" aria-labelledby="lab-title">
      <header><p className="eyebrow">Interactive / CPU</p><h2 id="lab-title">{t.title}</h2><p>{t.note}</p></header>
      <div className="lab-grid">
        <div className="controls">
          <label className="search-label" htmlFor="track-search">{t.search}</label>
          <input id="track-search" name="track-search" type="search" value={query} placeholder={t.placeholder} autoComplete="off" onChange={(event) => setQuery(event.target.value)} />
          {query ? (
            <div className="suggestions" aria-live="polite">
              {suggestions.length ? suggestions.map((track) => (
                <button key={track.id} type="button" onClick={() => { setSeed(track); setQuery(''); }}>
                  <strong>{track.name}</strong><span>{track.artist} · {track.year}</span>
                </button>
              )) : <p>{t.empty}</p>}
            </div>
          ) : null}
          <div className="seed"><span>{t.search}</span><strong>{seed.name}</strong><small>{seed.artist} · {seed.year}</small></div>
          <div className="weight-heading"><strong>{t.weights}</strong><button type="button" onClick={() => setWeights([...DEFAULT_WEIGHTS])}>{t.reset}</button></div>
          <div className="sliders">
            {artifact.featureNames.map((name, index) => (
              <label key={name}><span>{name}</span><output>{weights[index].toFixed(1)}×</output>
                <input name={`weight-${name}`} type="range" min="0" max="2" step="0.1" value={weights[index]} onChange={(event) => {
                  const next = [...weights]; next[index] = Number(event.target.value); setWeights(next);
                }} />
              </label>
            ))}
          </div>
        </div>
        <div className="results">
          <div className="radar-wrap">
            <svg className="radar" viewBox="0 0 180 180" role="img" aria-label={`${seed.name} feature profile`}>
              <circle cx="90" cy="90" r="80" /><circle cx="90" cy="90" r="49" /><path d={radarPath(seed.features)} />
            </svg>
            <div><span>{t.results}</span><strong>Weighted cosine</strong><small>n = {artifact.tracks.length.toLocaleString()}</small></div>
          </div>
          {hasWeight ? <ol>{results.map((track) => (
            <li key={track.id}><span className="rank"></span><div><strong>{track.name}</strong><small>{track.artist} · {track.year}</small></div><output>{(track.score * 100).toFixed(1)}% <small>{t.score}</small></output></li>
          ))}</ol> : <p className="warning">{t.zero}</p>}
        </div>
      </div>
    </section>
  );
}
