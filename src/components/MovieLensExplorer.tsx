import { useState } from 'react';
import { recommendationOverlap, sampleCounts, visibleRecommendations } from '../lib/movielens-demo.mjs';
import './MovieLensExplorer.css';

type Movie = { movieId: number; title: string; genres: string[]; popularityBand: string };
type Evidence = { source: string; similarity: number; residual: number; contribution: number };
type Rec = Movie & { rankScore: number; secondaryScore?: number; secondaryScoreName?: string; ratingEstimate: number | null; scoreWasClipped: boolean; similarityWeight?: number; neighbors: number; fallback: boolean; hit: boolean; evidence: Evidence[] };
type UserSample = { userId: number; user: string; activity: number; historyTotal?: number; relevantTestTotal?: number; history: Array<Movie & { rating: number }>; relevantTest: Array<Movie & { rating: number }>; methods: { popularity: Rec[]; userCf: Rec[]; itemCf: Rec[] } };
type Samples = { users: UserSample[] };

const METHOD_KEYS = ['popularity', 'userCf', 'itemCf'] as const;

export default function MovieLensExplorer({ lang, samples }: { lang: 'en' | 'zh'; samples: Samples }) {
  const [userIndex, setUserIndex] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const [truthExpanded, setTruthExpanded] = useState(false);
  const zh = lang === 'zh';
  const sample = samples.users[userIndex];
  const overlap = recommendationOverlap(sample.methods.userCf, sample.methods.itemCf);
  const counts = sampleCounts(sample);
  const visibleTruth = sample.relevantTest.slice(0, truthExpanded ? sample.relevantTest.length : 5);
  const methodLabels = { popularity: zh ? '热门基线' : 'Popularity baseline', userCf: 'User-CF', itemCf: 'Item-CF' };

  return <div className="ml-explorer">
    <div className="sample-picker">
      <label>{zh ? '选择固定测试用户' : 'Choose a deterministic test user'}
        <select value={userIndex} onChange={(event) => { setUserIndex(Number(event.target.value)); setTruthExpanded(false); setExpanded(false); }}>
          {samples.users.map((user, index) => <option value={index} key={user.userId}>{user.user} · {user.activity} {zh ? '条训练评分' : 'training ratings'}</option>)}
        </select>
      </label>
      <p>{zh ? `User-CF 与 Item-CF 的 Top-10 重合 ${overlap.count} 部电影。` : `User-CF and Item-CF share ${overlap.count} movie${overlap.count === 1 ? '' : 's'} in their Top-10.`}</p>
    </div>
    <div className="evidence-strip">
      <section><p className="mini-label">{zh ? '训练历史示例' : 'Training-history examples'}</p><p className="evidence-count">{zh ? `模型实际使用 ${counts.historyTotal} 条训练评分；这里只展示最高评分的 ${counts.historyShown} 条。` : `The model used all ${counts.historyTotal} training ratings; only the top-rated ${counts.historyShown} are shown.`}</p><ul>{sample.history.map((movie) => <li key={movie.movieId}><span>{movie.title}</span><strong>{movie.rating.toFixed(0)}★</strong></li>)}</ul></section>
      <section><p className="mini-label">{zh ? '隐藏的未来相关电影' : 'Hidden future-relevant movies'}</p><p className="evidence-count">{zh ? `测试期共有 ${counts.relevantTotal} 部评分 ≥4 的相关电影；当前展示 ${visibleTruth.length} 部。` : `The test period contains ${counts.relevantTotal} relevant movies rated ≥4; ${visibleTruth.length} are shown.`}</p><ul>{visibleTruth.map((movie) => <li key={movie.movieId}><span>{movie.title}</span><strong>{movie.rating.toFixed(0)}★</strong></li>)}</ul>{sample.relevantTest.length > 5 && <button className="evidence-expand" type="button" aria-expanded={truthExpanded} onClick={() => setTruthExpanded((value) => !value)}>{truthExpanded ? (zh ? '收起未来电影' : 'Show fewer future movies') : (zh ? '展开全部未来电影' : 'Show all future movies')}</button>}</section>
    </div>
    <div className="method-comparison">
      {METHOD_KEYS.map((key) => {
        const recommendations = sample.methods[key];
        const hits = recommendations.filter((movie) => movie.hit).length;
        const tail = recommendations.filter((movie) => movie.popularityBand === 'long-tail').length;
        const shown = visibleRecommendations(recommendations, expanded) as Rec[];
        const fullyTied = recommendations.length > 1 && recommendations.every((movie) => movie.rankScore === recommendations[0].rankScore);
        return <section className={`method-column method-${key}`} key={key}>
          <header><div><p className="mini-label">{methodLabels[key]}</p><strong>{hits} {zh ? '次命中' : `hit${hits === 1 ? '' : 's'}`} · {tail}/10 {zh ? '长尾' : 'long-tail'}</strong>{fullyTied && <small>{zh ? '主分并列；依次比较 Bayesian 分、证据强度，ID 仅作最终确定键' : 'primary-score tie; Bayesian score and evidence break it before movie ID'}</small>}</div><span>{key === 'popularity' ? (zh ? '所有用户同一排序' : 'same ranking for everyone') : (zh ? '个性化' : 'personalized')}</span></header>
          <ol>{shown.map((movie, index) => <li className={movie.hit ? 'is-hit' : ''} key={movie.movieId}>
            <span className="rank">{String(index + 1).padStart(2, '0')}</span>
            <div><strong>{movie.title}</strong><small>{movie.genres.slice(0, 2).join(' · ')} · {movie.popularityBand === 'head' ? (zh ? '头部' : 'head') : (zh ? '长尾' : 'long-tail')}</small>{movie.evidence.length > 0 && <details><summary>{zh ? '为什么推荐？' : 'Why this movie?'}</summary>{movie.evidence.map((item) => <p key={item.source}>{item.source}<br/><small>sim {item.similarity.toFixed(3)} · contribution {item.contribution >= 0 ? '+' : ''}{item.contribution.toFixed(3)}</small></p>)}</details>}</div>
            <div className="score"><strong>{movie.rankScore.toFixed(3)}</strong><small>{zh ? '主排序分' : 'primary score'}</small>{key !== 'popularity' && movie.secondaryScore !== undefined && <small>Bayes {movie.secondaryScore.toFixed(3)}</small>}{movie.ratingEstimate !== null && <small>{zh ? `展示 ${movie.ratingEstimate.toFixed(2)}${movie.scoreWasClipped ? '（截断）' : ''}` : `display ${movie.ratingEstimate.toFixed(2)}${movie.scoreWasClipped ? ' (clipped)' : ''}`}</small>}{key !== 'popularity' && <small>{movie.neighbors} {zh ? '个邻居' : 'neighbors'} · Σ|sim| {(movie.similarityWeight ?? 0).toFixed(2)}</small>}</div>
            {movie.hit && <b>{zh ? '命中' : 'HIT'}</b>}
          </li>)}</ol>
        </section>;
      })}
    </div>
    <button className="expand-lists" type="button" aria-expanded={expanded} onClick={() => setExpanded((value) => !value)}>{expanded ? (zh ? '收起为 Top-5' : 'Show Top-5') : (zh ? '展开 Top-10' : 'Expand to Top-10')}</button>
    <p className="demo-note">{zh ? '训练历史与未来真值的数量不同，是因为展示上限和每位用户测试期相关电影数不同，不是训练/测试泄漏。“命中”也不代表用户一定会观看。' : 'Training-history and future-truth counts differ because display limits and each user’s test activity differ—not because of leakage. A “hit” still does not prove the user would watch.'}</p>
  </div>;
}
