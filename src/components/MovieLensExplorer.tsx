import { useMemo, useState } from 'react';
import { chooseRecommendations } from '../lib/movielens-demo.mjs';
import './MovieLensExplorer.css';

type Rec = { title: string; score: number; neighbors: number; fallback: boolean };
type UserSample = {
  user: string;
  activity: number;
  history: Array<{ title: string; rating: number }>;
  userCf: Rec[];
  itemCf: Rec[];
};
type Samples = {
  users: UserSample[];
  relatedItems: Array<{ seed: string; neighbors: Array<{ title: string; similarity: number }> }>;
};

export default function MovieLensExplorer({ lang, samples }: { lang: 'en' | 'zh'; samples: Samples }) {
  const [userIndex, setUserIndex] = useState(0);
  const [method, setMethod] = useState<'userCf' | 'itemCf'>('userCf');
  const sample = samples.users[userIndex];
  const recommendations: Rec[] = useMemo(() => chooseRecommendations(sample, method), [sample, method]);
  const zh = lang === 'zh';

  return <div className="ml-explorer">
    <div className="ml-controls">
      <label>{zh ? '离线示例用户' : 'Offline sample user'}
        <select value={userIndex} onChange={(event) => setUserIndex(Number(event.target.value))}>
          {samples.users.map((user, index) => <option value={index} key={user.user}>{user.user} · {user.activity} {zh ? '条训练评分' : 'training ratings'}</option>)}
        </select>
      </label>
      <fieldset><legend>{zh ? '候选方法' : 'Candidate method'}</legend>
        <button type="button" aria-pressed={method === 'userCf'} onClick={() => setMethod('userCf')}>User-CF</button>
        <button type="button" aria-pressed={method === 'itemCf'} onClick={() => setMethod('itemCf')}>Item-CF</button>
      </fieldset>
    </div>
    <div className="ml-demo-grid">
      <section><p className="mini-label">{zh ? '高评分历史' : 'High-rated history'}</p>
        <ol className="history-list">{sample.history.map((item) => <li key={item.title}><span>{item.title}</span><strong>{item.rating.toFixed(0)}★</strong></li>)}</ol>
      </section>
      <section><p className="mini-label">Top-10 · {method === 'userCf' ? 'User-CF' : 'Item-CF'}</p>
        <ol className="recommendation-list">{recommendations.map((item, index) => <li key={`${item.title}-${index}`}>
          <span className="rank">{String(index + 1).padStart(2, '0')}</span><span>{item.title}<small>{item.fallback ? (zh ? '热门兜底' : 'popularity fallback') : `${item.neighbors} ${zh ? '个邻居证据' : 'neighbor signals'}`}</small></span><strong>{item.score.toFixed(2)}</strong>
        </li>)}</ol>
      </section>
    </div>
    <section className="related-panel"><p className="mini-label">{zh ? 'Item-CF 相似电影证据' : 'Item-CF related-movie evidence'}</p>
      <div className="related-grid">{samples.relatedItems.map((group) => <article key={group.seed}><strong>{group.seed}</strong><ol>{group.neighbors.map((movie) => <li key={movie.title}><span>{movie.title}</span><small>{movie.similarity.toFixed(3)}</small></li>)}</ol></article>)}</div>
    </section>
    <p className="demo-note">{zh ? '这些是固定实验产物，不会把浏览器选择写回模型。分数用于排序，不是喜欢概率。' : 'These are fixed experiment artifacts; browser choices never update the model. Scores rank candidates—they are not preference probabilities.'}</p>
  </div>;
}
