import { useState } from 'react';
import { recommendationOverlap } from '../lib/movielens-demo.mjs';
import './MovieLensExplorer.css';

type Movie = { movieId: number; title: string; genres: string[]; popularityBand: string };
type Evidence = { source: string; similarity: number; residual: number; contribution: number };
type Rec = Movie & {
  rankScore: number;
  ratingEstimate: number | null;
  scoreWasClipped: boolean;
  neighbors: number;
  fallback: boolean;
  hit: boolean;
  evidence: Evidence[];
};
type UserSample = {
  userId: number;
  user: string;
  activity: number;
  history: Array<Movie & { rating: number }>;
  relevantTest: Array<Movie & { rating: number }>;
  methods: { popularity: Rec[]; userCf: Rec[]; itemCf: Rec[] };
};
type Related = { seed: Movie; neighbors: Array<Movie & { similarity: number; support: number }> };
type Samples = { users: UserSample[]; relatedItems: Related[] };

const METHOD_KEYS = ['popularity', 'userCf', 'itemCf'] as const;

export default function MovieLensExplorer({ lang, samples }: { lang: 'en' | 'zh'; samples: Samples }) {
  const [userIndex, setUserIndex] = useState(0);
  const [view, setView] = useState<'users' | 'movies'>('users');
  const [seedIndex, setSeedIndex] = useState(0);
  const zh = lang === 'zh';
  const sample = samples.users[userIndex];
  const related = samples.relatedItems[seedIndex];
  const methodLabels = {
    popularity: zh ? '热门基线' : 'Popularity baseline',
    userCf: 'User-CF',
    itemCf: 'Item-CF',
  };
  const overlap = recommendationOverlap(sample.methods.userCf, sample.methods.itemCf);

  return <div className="ml-explorer">
    <div className="view-tabs" role="tablist" aria-label={zh ? '演示模式' : 'Explorer mode'}>
      <button type="button" role="tab" aria-selected={view === 'users'} onClick={() => setView('users')}>{zh ? '用户推荐对比' : 'User recommendation comparison'}</button>
      <button type="button" role="tab" aria-selected={view === 'movies'} onClick={() => setView('movies')}>{zh ? '相似电影' : 'Related movies'}</button>
    </div>

    {view === 'users' ? <>
      <div className="sample-picker">
        <label>{zh ? '选择固定测试用户' : 'Choose a deterministic test user'}
          <select value={userIndex} onChange={(event) => setUserIndex(Number(event.target.value))}>
            {samples.users.map((user, index) => <option value={index} key={user.userId}>{user.user} · {user.activity} {zh ? '条训练评分' : 'training ratings'}</option>)}
          </select>
        </label>
        <p>{zh ? `User-CF 与 Item-CF 的 Top-10 重合 ${overlap.count} 部电影。` : `User-CF and Item-CF share ${overlap.count} movie${overlap.count === 1 ? '' : 's'} in their Top-10.`}</p>
      </div>
      <div className="evidence-strip">
        <section><p className="mini-label">{zh ? '模型看到的高评分历史' : 'High-rated history available to models'}</p><ul>{sample.history.map((movie) => <li key={movie.movieId}><span>{movie.title}</span><strong>{movie.rating.toFixed(0)}★</strong></li>)}</ul></section>
        <section><p className="mini-label">{zh ? '模型没有看到的未来相关电影' : 'Future relevant movies hidden from models'}</p><ul>{sample.relevantTest.map((movie) => <li key={movie.movieId}><span>{movie.title}</span><strong>{movie.rating.toFixed(0)}★</strong></li>)}</ul></section>
      </div>
      <div className="method-comparison">
        {METHOD_KEYS.map((key) => {
          const recommendations = sample.methods[key];
          const hits = recommendations.filter((movie) => movie.hit).length;
          const tail = recommendations.filter((movie) => movie.popularityBand === 'long-tail').length;
          return <section className={`method-column method-${key}`} key={key}>
            <header><div><p className="mini-label">{methodLabels[key]}</p><strong>{hits} {zh ? '次命中' : `hit${hits === 1 ? '' : 's'}`} · {tail}/10 {zh ? '长尾' : 'long-tail'}</strong></div><span>{key === 'popularity' ? (zh ? '所有用户同一排序' : 'same ranking for everyone') : (zh ? '个性化' : 'personalized')}</span></header>
            <ol>{recommendations.map((movie, index) => <li className={movie.hit ? 'is-hit' : ''} key={movie.movieId}>
              <span className="rank">{String(index + 1).padStart(2, '0')}</span>
              <div><strong>{movie.title}</strong><small>{movie.genres.slice(0, 2).join(' · ')} · {movie.popularityBand === 'head' ? (zh ? '头部' : 'head') : (zh ? '长尾' : 'long-tail')}</small>
                {movie.evidence.length > 0 && <details><summary>{zh ? '为什么推荐？' : 'Why this movie?'}</summary>{movie.evidence.map((item) => <p key={item.source}>{item.source}<br/><small>sim {item.similarity.toFixed(3)} · contribution {item.contribution >= 0 ? '+' : ''}{item.contribution.toFixed(3)}</small></p>)}</details>}
              </div>
              <div className="score"><strong>{movie.rankScore.toFixed(3)}</strong><small>{zh ? '排序分' : 'rank score'}</small>{movie.ratingEstimate !== null && <small>{zh ? `展示 ${movie.ratingEstimate.toFixed(2)}${movie.scoreWasClipped ? '（截断）' : ''}` : `display ${movie.ratingEstimate.toFixed(2)}${movie.scoreWasClipped ? ' (clipped)' : ''}`}</small>}</div>
              {movie.hit && <b>{zh ? '命中' : 'HIT'}</b>}
            </li>)}</ol>
          </section>;
        })}
      </div>
    </> : <>
      <div className="sample-picker"><label>{zh ? '种子电影' : 'Seed movie'}<select value={seedIndex} onChange={(event) => setSeedIndex(Number(event.target.value))}>{samples.relatedItems.map((group, index) => <option value={index} key={group.seed.movieId}>{group.seed.title}</option>)}</select></label><p>{zh ? 'Item-CF 使用共同评分行为，而不是电影类型标签来建立关系。' : 'Item-CF builds these relationships from shared rating behavior—not genre labels.'}</p></div>
      <div className="related-view"><article className="seed-card"><span>{zh ? '因为你喜欢' : 'Because you liked'}</span><h3>{related.seed.title}</h3><p>{related.seed.genres.join(' · ')}</p></article><ol>{related.neighbors.map((movie, index) => <li key={movie.movieId}><span className="rank">{String(index + 1).padStart(2, '0')}</span><div><strong>{movie.title}</strong><small>{movie.genres.join(' · ')}</small></div><div className="score"><strong>{movie.similarity.toFixed(3)}</strong><small>{zh ? '收缩后相似度' : 'shrunk similarity'}</small><small>{movie.support.toLocaleString()} {zh ? '位共同评分用户' : 'co-raters'}</small></div></li>)}</ol></div>
    </>}
    <p className="demo-note">{zh ? '演示只读取完整实验生成的固定 JSON。未来测试电影从训练中隐藏；“命中”不代表用户一定会观看。' : 'This demo reads fixed JSON from the full experiment. Future test movies were hidden from training; a “hit” still does not prove the user would watch.'}</p>
  </div>;
}
