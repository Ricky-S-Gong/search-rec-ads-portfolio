export type Lang = 'en' | 'zh';
export type Domain = 'search' | 'recommendation' | 'ads';

export interface ProjectMeta {
  slug: string;
  domain: Domain;
  status: 'complete' | 'planned';
  compute: 'CPU' | 'GPU recommended' | 'GPU required';
  algorithms: string[];
  dataset: string;
  authors: string[];
  title: Record<Lang, string>;
  summary: Record<Lang, string>;
}

export interface AlgorithmMeta {
  slug: string;
  domain: Domain;
  family: string;
  requiredSignals: string;
  coldStart: string;
  trainCost: string;
  serveCost: string;
  explainability: 'High' | 'Medium' | 'Low';
  name: Record<Lang, string>;
  intuition: Record<Lang, string>;
  useWhen: Record<Lang, string>;
  avoidWhen: Record<Lang, string>;
}

export const languages: Lang[] = ['en', 'zh'];

export const copy = {
  en: {
    siteName: 'Search · Rec · Ads Cosmos',
    byline: 'A joint systems portfolio by Ricky Gong & Ziqi Xu',
    nav: { home: 'Mission control', projects: 'Projects', compare: 'Compare', roadmap: 'Roadmap', about: 'Crew' },
    language: '中文',
    heroEyebrow: 'Evidence before architecture',
    heroTitle: 'We map signals to decisions.',
    heroBody: 'Reproducible studies of retrieval, ranking, and optimization—built to explain not only what worked, but when it should be used and where it breaks.',
    explore: 'Open the first mission',
    compare: 'Compare algorithms',
    projectsTitle: 'Mission log',
    projectsIntro: 'Each project connects a dataset, a business constraint, an algorithm choice, and an honest evaluation.',
    decisionTitle: 'The decision surface',
    decisionIntro: 'An algorithm is not “best” in isolation. Its value depends on the signals, latency budget, and failure mode a product can tolerate.',
    complete: 'Complete', planned: 'Planned', compute: 'Compute', dataset: 'Dataset',
  },
  zh: {
    siteName: '搜广推算法宇宙',
    byline: 'Ricky Gong 与 Ziqi Xu 的联合技术作品集',
    nav: { home: '任务中心', projects: '项目', compare: '算法对比', roadmap: '学习路线', about: '成员' },
    language: 'English',
    heroEyebrow: '先有证据，再谈架构',
    heroTitle: '从信号出发，抵达决策。',
    heroBody: '通过可复现实验研究召回、排序与优化：不仅说明什么有效，也明确它何时适用、何处失效。',
    explore: '进入首个任务',
    compare: '比较算法',
    projectsTitle: '任务日志',
    projectsIntro: '每个项目都连接数据、业务约束、算法选择与诚实的评估。',
    decisionTitle: '算法决策面',
    decisionIntro: '算法没有脱离场景的“最佳”。它的价值取决于可用信号、延迟预算和产品能够承担的失效方式。',
    complete: '已完成', planned: '计划中', compute: '计算环境', dataset: '数据集',
  },
} as const;

export const projects: ProjectMeta[] = [
  {
    slug: 'spotify-content-recommender',
    domain: 'recommendation',
    status: 'complete',
    compute: 'CPU',
    algorithms: ['Popularity', 'Euclidean KNN', 'Cosine', 'Weighted cosine', 'K-Means retrieval'],
    dataset: 'Kaggle Spotify dataset · 170,653 tracks',
    authors: ['Ricky Gong', 'Ziqi Xu'],
    title: { en: 'Spotify content recommender', zh: 'Spotify 内容推荐系统' },
    summary: {
      en: 'An honest item-to-item recommender built from audio attributes, with exact and approximate retrieval compared under cold-start and latency constraints.',
      zh: '基于音频属性构建相似歌曲推荐，在冷启动与延迟约束下比较精确检索和近似候选召回。',
    },
  },
  {
    slug: 'movielens-collaborative-filtering',
    domain: 'recommendation',
    status: 'planned',
    compute: 'CPU',
    algorithms: ['UserCF', 'ItemCF', 'Matrix factorization'],
    dataset: 'MovieLens · user-item interactions',
    authors: ['Ricky Gong', 'Ziqi Xu'],
    title: { en: 'Collaborative signals at work', zh: '协同行为信号实战' },
    summary: {
      en: 'The next mission adds real user-item interactions so ranking metrics and personalization become valid.',
      zh: '下一项任务引入真实用户—物品交互，使排序指标和个性化评测真正成立。',
    },
  },
];

export const algorithms: AlgorithmMeta[] = [
  {
    slug: 'popularity', domain: 'recommendation', family: 'Non-personalized baseline',
    requiredSignals: 'Item popularity', coldStart: 'New users: strong · New items: weak', trainCost: 'O(n)', serveCost: 'O(k)', explainability: 'High',
    name: { en: 'Popularity baseline', zh: '流行度基线' },
    intuition: { en: 'Recommend what works for the largest audience when no preference signal exists.', zh: '没有用户偏好信号时，推荐对最多人有效的内容。' },
    useWhen: { en: 'Anonymous traffic, fallback paths, and a sanity-check baseline.', zh: '匿名流量、兜底策略，以及所有复杂模型必须超过的基线。' },
    avoidWhen: { en: 'Long-tail discovery or meaningful personalization is the objective.', zh: '目标是长尾发现或真正个性化时。' },
  },
  {
    slug: 'content-cosine', domain: 'recommendation', family: 'Content-based retrieval',
    requiredSignals: 'Item feature vectors', coldStart: 'New users: weak · New items: strong', trainCost: 'O(nd)', serveCost: 'O(nd)', explainability: 'High',
    name: { en: 'Content cosine similarity', zh: '内容余弦相似度' },
    intuition: { en: 'Items pointing in similar directions in feature space should feel similar.', zh: '在特征空间中方向接近的物品，应具有相似内容属性。' },
    useWhen: { en: 'Item-to-item discovery, transparent controls, and rich item metadata.', zh: '相似物品发现、可控推荐，以及物品特征丰富的场景。' },
    avoidWhen: { en: 'Taste depends on collaborative behavior that metadata cannot express.', zh: '用户兴趣主要由内容特征无法表达的协同行为决定时。' },
  },
  {
    slug: 'euclidean-knn', domain: 'recommendation', family: 'Content-based retrieval',
    requiredSignals: 'Scaled item feature vectors', coldStart: 'New users: weak · New items: strong', trainCost: 'O(nd)', serveCost: 'O(nd)', explainability: 'High',
    name: { en: 'Euclidean KNN', zh: '欧氏距离 KNN' },
    intuition: { en: 'Retrieve the closest items after putting every feature on a comparable scale.', zh: '统一特征尺度后，直接检索距离最近的物品。' },
    useWhen: { en: 'Magnitude differences are meaningful and the catalog is moderate.', zh: '特征差值有明确意义且目录规模适中时。' },
    avoidWhen: { en: 'Unscaled dimensions dominate distance or exact scans miss the latency budget.', zh: '特征未缩放，或精确扫描无法满足延迟预算时。' },
  },
  {
    slug: 'kmeans-retrieval', domain: 'recommendation', family: 'Approximate candidate retrieval',
    requiredSignals: 'Dense item vectors', coldStart: 'New users: weak · New items: medium', trainCost: 'O(nkdi)', serveCost: 'O(nd/k)', explainability: 'Medium',
    name: { en: 'K-Means candidate retrieval', zh: 'K-Means 候选召回' },
    intuition: { en: 'Search a relevant region before applying a more precise similarity function.', zh: '先定位相似区域，再在小候选集内执行精确相似度计算。' },
    useWhen: { en: 'A simple latency-quality trade-off is more valuable than perfect recall.', zh: '需要简单、可解释的延迟—效果权衡时。' },
    avoidWhen: { en: 'Cluster boundaries are unstable or approximate nearest-neighbor indexes are available.', zh: '聚类边界不稳定，或已有成熟 ANN 索引时。' },
  },
  {
    slug: 'matrix-factorization', domain: 'recommendation', family: 'Collaborative filtering',
    requiredSignals: 'User-item interactions', coldStart: 'New users: weak · New items: weak', trainCost: 'O(|R|df)', serveCost: 'O(nd)', explainability: 'Low',
    name: { en: 'Matrix factorization', zh: '矩阵分解' },
    intuition: { en: 'Learn latent user and item factors whose dot product predicts preference.', zh: '学习用户与物品的隐向量，用点积预测偏好。' },
    useWhen: { en: 'Interaction history is dense enough and collaborative taste matters.', zh: '交互数据足够、协同兴趣模式重要时。' },
    avoidWhen: { en: 'Only item metadata exists—the Spotify project intentionally does not fake this signal.', zh: '只有物品属性时；Spotify 项目不会伪造这类信号。' },
  },
];

export const oppositeLang = (lang: Lang): Lang => (lang === 'en' ? 'zh' : 'en');
export const basePath = import.meta.env.BASE_URL.replace(/\/$/, '');
export const href = (path: string) => `${basePath}${path}`;
