import { useDeferredValue, useEffect, useMemo, useState } from 'react';
import { cosineToAngularPercent, weightedCosine } from '../lib/similarity';
import './SpotifyLab.css';

type Lang = 'en' | 'zh';
type Track = { id: string; name: string; artist: string; year: number; popularity: number; features: number[] };
type Artifact = { version: string; featureNames: string[]; tracks: Track[] };
type ScoredTrack = Track & { score: number };

const DEFAULT_WEIGHTS = [1, 1, 1, 1, 1, 1, 1, 1, 1];
const FEATURE_COLORS = ['#35d0e2', '#a78bfa', '#f7b955', '#55d98b', '#ff7d8b', '#79a8d8', '#d9a7ff', '#68e0cf', '#ffca7a'];
const assetBase = `${import.meta.env.BASE_URL.replace(/\/$/, '')}/`;

const featureNamesZh: Record<string, string> = {
  acousticness: '原声度', danceability: '舞蹈性', energy: '能量', instrumentalness: '器乐性',
  liveness: '现场感', loudness: '响度', speechiness: '语音性', tempo: '速度', valence: '情绪正向度',
};

const ui = {
  en: {
    title: 'Tune the definition of “similar”', search: 'Seed track', placeholder: 'Search title or artist…',
    weights: 'Feature weights', results: 'Top recommendations', reset: 'Reset weights', loading: 'Loading the static experiment…',
    empty: 'No track matches that search.', zero: 'Raise at least one feature weight to calculate similarity.', score: 'angular match', raw: 'cosine',
    note: 'Computed locally from a stratified 5,000-track demonstration set. No Spotify API or user data.', error: 'The experiment artifact did not load. Refresh the page or check the generated JSON.',
    scoreTitle: 'How is the percentage calculated?',
    scoreMethod: 'Nine audio features are MinMax-scaled to [0, 1] and stored to four decimal places. The sliders rescale dimensions, then raw weighted cosine ranks candidates. For display only, the cosine is converted to its angle over the 0–90° range of non-negative vectors. The percentage keeps two decimals and raw cosine keeps four. Identical stored feature vectors can still correctly show 100.00%.',
    featureGuide: 'Radar feature key', seedValue: 'Seed', candidateValue: 'Selected result', difference: 'Difference', compare: 'Compare on radar',
  },
  zh: {
    title: '调整“相似”的定义', search: '种子歌曲', placeholder: '搜索歌曲或艺术家…',
    weights: '特征权重', results: '推荐结果', reset: '重置权重', loading: '正在加载静态实验数据…',
    empty: '没有匹配的歌曲。', zero: '至少提高一个特征权重才能计算相似度。', score: '角度匹配度', raw: '原始余弦',
    note: '基于分层抽取的 5,000 首演示歌曲在浏览器本地计算，不调用 Spotify API，也不使用用户数据。', error: '实验数据未能加载，请刷新页面或检查生成的 JSON。',
    scoreTitle: '百分比是如何计算的？',
    scoreMethod: '九个音频特征先按列进行 MinMax 缩放到 [0, 1]，并以四位小数存入演示数据。滑块用于缩放特征维度，候选排序仍使用原始加权余弦。页面只在展示时把余弦转换为非负向量 0–90° 范围内的角度匹配度；百分比保留两位小数，原始余弦保留四位。如果两首歌存储的九项特征完全相同，100.00% 是真实结果，不会被人为压低。',
    featureGuide: '雷达图特征对照', seedValue: '种子歌曲', candidateValue: '当前对比结果', difference: '差值', compare: '在雷达图中对比',
  },
};

function topMatches(seed: Track, tracks: Track[], weights: number[], count = 10): ScoredTrack[] {
  const unique = new Map<string, ScoredTrack>();
  const seedKey = `${seed.name}\u0000${seed.artist}`.toLocaleLowerCase();
  for (const track of tracks) {
    const key = `${track.name}\u0000${track.artist}`.toLocaleLowerCase();
    if (track.id === seed.id || key === seedKey) continue;
    const candidate = { ...track, score: weightedCosine(seed.features, track.features, weights) };
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

function radarPoint(value: number, index: number, count: number, maxRadius = 70) {
  const angle = (Math.PI * 2 * index) / count - Math.PI / 2;
  const radius = Math.max(0, Math.min(1, value)) * maxRadius;
  return { x: 90 + Math.cos(angle) * radius, y: 90 + Math.sin(angle) * radius };
}

function radarPath(values: number[]) {
  return values.map((value, index) => {
    const point = radarPoint(value, index, values.length);
    return `${index ? 'L' : 'M'} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`;
  }).join(' ') + ' Z';
}

export default function SpotifyLab({ lang }: { lang: Lang }) {
  const t = ui[lang];
  const [artifact, setArtifact] = useState<Artifact | null>(null);
  const [error, setError] = useState(false);
  const [seed, setSeed] = useState<Track | null>(null);
  const [query, setQuery] = useState('');
  const [weights, setWeights] = useState<number[]>(() => [...DEFAULT_WEIGHTS]);
  const [comparisonId, setComparisonId] = useState('');
  const [activeFeature, setActiveFeature] = useState<number | null>(null);
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
  const comparison = results.find((track) => track.id === comparisonId) ?? results[0] ?? null;

  if (error) return <p className="lab-status" role="alert">{t.error}</p>;
  if (!artifact || !seed) return <p className="lab-status" aria-live="polite">{t.loading}</p>;

  const featureLabel = (name: string) => lang === 'zh' ? `${featureNamesZh[name] ?? name}（${name}）` : name;
  const activeName = activeFeature === null ? null : artifact.featureNames[activeFeature];

  return (
    <section className="lab" aria-labelledby="lab-title">
      <header>
        <p className="eyebrow">Interactive / CPU</p><h2 id="lab-title">{t.title}</h2><p>{t.note}</p>
        <details className="score-method"><summary>{t.scoreTitle}</summary><p>{t.scoreMethod}</p><code>match% = [1 − arccos(cosine) ÷ (π / 2)] × 100</code></details>
      </header>
      <div className="lab-grid">
        <div className="controls">
          <label className="search-label" htmlFor="track-search">{t.search}</label>
          <input id="track-search" name="track-search" type="search" value={query} placeholder={t.placeholder} autoComplete="off" onChange={(event) => setQuery(event.target.value)} />
          {query ? (
            <div className="suggestions" aria-live="polite">
              {suggestions.length ? suggestions.map((track) => (
                <button key={track.id} type="button" onClick={() => { setSeed(track); setComparisonId(''); setQuery(''); }}>
                  <strong>{track.name}</strong><span>{track.artist} · {track.year}</span>
                </button>
              )) : <p>{t.empty}</p>}
            </div>
          ) : null}
          <div className="seed"><span>{t.search}</span><strong>{seed.name}</strong><small>{seed.artist} · {seed.year}</small></div>
          <div className="weight-heading"><strong>{t.weights}</strong><button type="button" onClick={() => setWeights([...DEFAULT_WEIGHTS])}>{t.reset}</button></div>
          <div className="sliders">
            {artifact.featureNames.map((name, index) => (
              <label key={name}><span>{featureLabel(name)}</span><output>{weights[index].toFixed(1)}×</output>
                <input name={`weight-${name}`} type="range" min="0" max="2" step="0.1" value={weights[index]} onChange={(event) => {
                  const next = [...weights]; next[index] = Number(event.target.value); setWeights(next);
                }} />
              </label>
            ))}
          </div>
        </div>
        <div className="results">
          <div className="radar-panel">
            <div className="radar-visual">
              <svg className="radar" viewBox="0 0 180 180" role="img" aria-label={`${seed.name} and ${comparison?.name ?? ''} audio-feature comparison`}>
                {[17.5, 35, 52.5, 70].map((radius) => <circle key={radius} cx="90" cy="90" r={radius} />)}
                {artifact.featureNames.map((name, index) => {
                  const end = radarPoint(1, index, artifact.featureNames.length);
                  const label = radarPoint(1, index, artifact.featureNames.length, 82);
                  const seedPoint = radarPoint(seed.features[index], index, artifact.featureNames.length);
                  const comparisonPoint = comparison ? radarPoint(comparison.features[index], index, artifact.featureNames.length) : null;
                  const active = activeFeature === index;
                  return <g key={name} className={active ? 'feature-active' : undefined}>
                    <line x1="90" y1="90" x2={end.x} y2={end.y} />
                    <text x={label.x} y={label.y} fill={FEATURE_COLORS[index]}>{index + 1}</text>
                    <circle className="seed-point" cx={seedPoint.x} cy={seedPoint.y} r={active ? 4.5 : 3} style={{ fill: FEATURE_COLORS[index] }} tabIndex={0} role="button" aria-label={`${featureLabel(name)}: ${t.seedValue} ${seed.features[index].toFixed(4)}`} onMouseEnter={() => setActiveFeature(index)} onMouseLeave={() => setActiveFeature(null)} onFocus={() => setActiveFeature(index)} onBlur={() => setActiveFeature(null)} onClick={() => setActiveFeature(index)} />
                    {comparisonPoint && <circle className="comparison-point" cx={comparisonPoint.x} cy={comparisonPoint.y} r={active ? 4.5 : 3} style={{ fill: FEATURE_COLORS[index] }} tabIndex={0} role="button" aria-label={`${featureLabel(name)}: ${t.candidateValue} ${comparison.features[index].toFixed(4)}`} onMouseEnter={() => setActiveFeature(index)} onMouseLeave={() => setActiveFeature(null)} onFocus={() => setActiveFeature(index)} onBlur={() => setActiveFeature(null)} onClick={() => setActiveFeature(index)} />}
                  </g>;
                })}
                <path className="seed-shape" d={radarPath(seed.features)} />
                {comparison && <path className="comparison-shape" d={radarPath(comparison.features)} />}
              </svg>
              <div className="radar-series"><span><i className="seed-swatch"></i>{t.seedValue}</span><span><i className="comparison-swatch"></i>{t.candidateValue}</span></div>
              {activeFeature !== null && comparison && activeName && <div className="feature-tooltip" aria-live="polite"><strong>{activeFeature + 1}. {featureLabel(activeName)}</strong><span>{t.seedValue}: {seed.features[activeFeature].toFixed(4)}</span><span>{t.candidateValue}: {comparison.features[activeFeature].toFixed(4)}</span><span>{t.difference}: {Math.abs(seed.features[activeFeature] - comparison.features[activeFeature]).toFixed(4)}</span></div>}
            </div>
            <div className="feature-legend" aria-label={t.featureGuide}>
              <strong>{t.featureGuide}</strong>
              <div>{artifact.featureNames.map((name, index) => (
                <button key={name} type="button" className={activeFeature === index ? 'active' : ''} onMouseEnter={() => setActiveFeature(index)} onMouseLeave={() => setActiveFeature(null)} onFocus={() => setActiveFeature(index)} onBlur={() => setActiveFeature(null)} onClick={() => setActiveFeature(index)}>
                  <b style={{ color: FEATURE_COLORS[index] }}>{index + 1}</b><span>{lang === 'zh' ? featureNamesZh[name] ?? name : name}</span><small>{lang === 'zh' ? `${name} · ` : ''}{seed.features[index].toFixed(4)} / {comparison?.features[index].toFixed(4) ?? '—'}</small>
                </button>
              ))}</div>
            </div>
          </div>
          <div className="results-heading"><div><span>{t.results}</span><strong>{comparison ? `${seed.name} → ${comparison.name}` : 'Weighted cosine'}</strong><small>n = {artifact.tracks.length.toLocaleString()}</small></div></div>
          {hasWeight ? <ol>{results.map((track) => (
            <li key={track.id} className={comparison?.id === track.id ? 'selected' : ''}><button type="button" onClick={() => setComparisonId(track.id)} aria-pressed={comparison?.id === track.id} aria-label={`${t.compare}: ${track.name}`}><span className="rank"></span><div><strong>{track.name}</strong><small>{track.artist} · {track.year}</small></div><output><strong>{cosineToAngularPercent(track.score).toFixed(2)}%</strong><small>{t.score}</small><small>{t.raw} {track.score.toFixed(4)}</small></output></button></li>
          ))}</ol> : <p className="warning">{t.zero}</p>}
        </div>
      </div>
    </section>
  );
}
