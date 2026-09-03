export type Lang = 'en' | 'zh';
export type Domain = 'search' | 'ads' | 'recommendation';
export type ProjectStatus = 'complete' | 'in-progress' | 'planned';

type Localized = Record<Lang, string>;

export interface ProjectMeta {
  slug: string;
  domain: Domain;
  status: ProjectStatus;
  compute: 'CPU' | 'GPU recommended' | 'GPU required';
  algorithms: string[];
  dataset: string;
  sourceUrl?: string;
  authors: string[];
  title: Localized;
  summary: Localized;
}

export interface AlgorithmMeta {
  slug: string;
  domain: Domain;
  name: Localized;
  family: Localized;
  requiredSignals: Localized;
  coldStart: Localized;
  trainCost: Localized;
  serveCost: Localized;
  explainability: Localized;
  bestFor: Localized;
  limitation: Localized;
  intuition: Localized;
  useWhen: Localized;
  avoidWhen: Localized;
}

export interface ProfileMeta {
  name: Localized;
  photo: string;
  photoAlt: Localized;
  school: Localized;
  major: Localized;
  industryRole: Localized;
  linkedin?: string;
  github: string;
  email: string;
}

export interface RoadmapTrack {
  domain: Domain;
  introduction: Localized;
  stages: Array<{
    title: Localized;
    topics: Localized;
    project?: Localized;
    status: ProjectStatus;
    compute: string;
    href?: string;
  }>;
}

export interface MetricDefinition {
  key: string;
  label: Localized;
  latex?: string;
  variables: Localized;
  intuition: Localized;
  businessMeaning: Localized;
}

export const languages: Lang[] = ['en', 'zh'];
export const domains: Domain[] = ['search', 'ads', 'recommendation'];

export const domainNames: Record<Domain, Localized> = {
  search: { en: 'Search', zh: '搜索' },
  ads: { en: 'Ads', zh: '广告' },
  recommendation: { en: 'Recommendation', zh: '推荐' },
};

export const copy = {
  en: {
    siteName: 'SAR Cosmos Lab',
    byline: 'A joint technical portfolio by Ricky Gong, Ziqi Xu & Yutao Rao',
    nav: { home: 'Home', projects: 'Projects', compare: 'Algorithms', roadmap: 'Roadmap', about: 'About' },
    language: '中文',
    heroTitle: 'SAR Cosmos Lab',
    heroBody: 'SAR stands for Search, Advertising, and Recommendation. This technical portfolio shows how the algorithms work, how they are implemented and evaluated, and where they fit in real systems.',
    domainCtas: { search: 'View Search projects', ads: 'View Ads projects', recommendation: 'View Recommendation projects' },
    algorithmCtas: { search: 'View Search algorithms', ads: 'View Ads algorithms', recommendation: 'View Recommendation algorithms' },
    explore: 'Explore recommendation case studies',
    compare: 'Compare algorithm families',
    projectsTitle: 'Projects by domain',
    projectsIntro: 'Projects focus on Search, Advertising, and Recommendation. Every completed project includes reproducible code and real experiment results.',
    decisionTitle: 'Algorithm families by domain',
    decisionIntro: 'Compare broad approaches within the same problem area. Project-specific models and implementation details stay in each case study.',
    complete: 'Completed', inProgress: 'In progress', planned: 'Planned', compute: 'Compute', dataset: 'Dataset',
  },
  zh: {
    siteName: '搜广推宇宙实验室',
    byline: 'Ricky Gong、Ziqi Xu 与 Yutao Rao 的联合技术作品集',
    nav: { home: '首页', projects: '项目', compare: '算法对比', roadmap: '路线图', about: '关于我们' },
    language: 'English',
    heroTitle: '搜广推宇宙实验室',
    heroBody: '这是一个围绕搜索、广告与推荐算法构建的技术作品集，展示不同算法的原理、代码实现、实验过程与应用场景。',
    domainCtas: { search: '查看搜索项目', ads: '查看广告项目', recommendation: '查看推荐项目' },
    algorithmCtas: { search: '查看搜索算法', ads: '查看广告算法', recommendation: '查看推荐算法' },
    explore: '探索推荐系统项目',
    compare: '比较算法类别',
    projectsTitle: '按方向查看项目',
    projectsIntro: '项目主要聚焦搜索、广告与推荐。每个已完成项目都提供可复现代码和真实实验结果。',
    decisionTitle: '按方向比较算法类别',
    decisionIntro: '在同一业务方向下比较不同方法；具体模型、距离函数和工程实现放在对应项目中。',
    complete: '已完成', inProgress: '进行中', planned: '计划中', compute: '计算环境', dataset: '数据集',
  },
} as const;

export const projects: ProjectMeta[] = [
  {
    slug: 'spotify-content-recommender', domain: 'recommendation', status: 'complete', compute: 'CPU',
    algorithms: ['Popularity', 'Euclidean KNN', 'Cosine', 'Weighted cosine', 'K-Means retrieval'],
    dataset: 'Kaggle Spotify dataset · 170,653 tracks', authors: ['Ricky Gong', 'Ziqi Xu'],
    sourceUrl: 'https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset',
    title: { en: 'Spotify content recommender', zh: 'Spotify 内容推荐' },
    summary: {
      en: 'An item-to-item recommender built from audio attributes, with exact and approximate retrieval compared under cold-start and latency constraints.',
      zh: '基于音频属性推荐相似歌曲，并在冷启动与延迟约束下比较精确和近似召回。',
    },
  },
  {
    slug: 'movielens-collaborative-filtering', domain: 'recommendation', status: 'complete', compute: 'CPU',
    algorithms: ['Bayesian popularity', 'User-CF', 'Item-CF'], dataset: 'MovieLens 1M · 1,000,209 ratings',
    authors: ['Ricky Gong', 'Ziqi Xu'],
    sourceUrl: 'https://grouplens.org/datasets/movielens/1m/',
    title: { en: 'MovieLens collaborative filtering', zh: 'MovieLens 协同过滤' },
    summary: {
      en: 'A reproducible User-CF and Item-CF study with temporal evaluation, full-catalog Top-10 ranking, coverage analysis, and CPU serving evidence.',
      zh: '可复现的 User-CF 与 Item-CF 实验，包含时间切分、全目录 Top-10、覆盖率分析与 CPU 服务证据。',
    },
  },
];

export const profiles: ProfileMeta[] = [
  {
    name: { en: 'Ricky Gong', zh: 'Ricky Gong' },
    photo: '/images/people/ricky-gong.jpg',
    photoAlt: { en: 'Ricky Gong under cherry blossoms', zh: '樱花树下的 Ricky Gong' },
    school: { en: 'University of Pennsylvania', zh: '宾夕法尼亚大学' },
    major: { en: 'MSE in Data Science', zh: '数据科学硕士' },
    industryRole: { en: 'Data Science Intern at Corsair', zh: 'Corsair 数据科学实习生' },
    linkedin: 'https://www.linkedin.com/in/shangyu-ricky-gong',
    github: 'https://github.com/Ricky-S-Gong',
    email: 'sgong.recruiting@gmail.com',
  },
  {
    name: { en: 'Ziqi Xu', zh: 'Ziqi Xu' },
    photo: '/images/people/ziqi-xu.jpg',
    photoAlt: { en: 'Ziqi Xu on an oak-lined path', zh: '林荫小路上的 Ziqi Xu' },
    school: { en: 'University of Illinois Urbana-Champaign', zh: '伊利诺伊大学厄巴纳-香槟分校' },
    major: { en: 'Statistics and Actuarial Science', zh: '统计与精算科学' },
    industryRole: { en: 'Former Intern at Chubb', zh: '曾任 Chubb 实习生' },
    linkedin: 'https://www.linkedin.com/in/ziqi12/',
    github: 'https://github.com/ziqixu22',
    email: 'xuziqi2003@gmail.com',
  },
  {
    name: { en: 'Yutao Rao', zh: 'Yutao Rao' },
    photo: '/images/people/yutao-rao.jpg',
    photoAlt: { en: 'Portrait of Yutao Rao by a lakeside', zh: '湖畔的 Yutao Rao' },
    school: { en: 'Stanford University', zh: '斯坦福大学' },
    major: { en: 'MS in Management Science and Engineering', zh: '管理科学与工程硕士' },
    industryRole: { en: 'Quantitative Development Intern at 平方和投资', zh: '平方和投资量化开发实习生' },
    linkedin: 'https://www.linkedin.com/in/yutao-rao-88527628b/',
    github: 'https://github.com/yutaor2',
    email: 'yutaorao004@gmail.com',
  },
];

export const roadmapTracks: RoadmapTrack[] = [
  {
    domain: 'search',
    introduction: {
      en: 'Build from transparent lexical retrieval to learned and semantic ranking.',
      zh: '从透明的词法检索出发，逐步进入学习排序与语义搜索。',
    },
    stages: [
      {
        title: { en: 'Information retrieval foundations', zh: '信息检索基础' },
        topics: { en: 'Inverted indexes · text processing · query analysis', zh: '倒排索引 · 文本预处理 · 查询分析' },
        status: 'planned', compute: 'CPU',
      },
      {
        title: { en: 'Lexical retrieval and evaluation', zh: '词法检索与离线评估' },
        topics: { en: 'TF-IDF · BM25 · relevance metrics', zh: 'TF-IDF · BM25 · 相关性指标' },
        status: 'planned', compute: 'CPU',
      },
      {
        title: { en: 'Semantic and hybrid retrieval', zh: '语义与混合检索' },
        topics: { en: 'Dense retrieval · vector search · hybrid fusion', zh: 'Dense Retrieval · 向量检索 · 混合融合' },
        status: 'planned', compute: 'GPU recommended',
      },
      {
        title: { en: 'Learned ranking and query understanding', zh: '学习排序与查询理解' },
        topics: { en: 'Learning to Rank · intent · LLM search', zh: 'Learning to Rank · 意图识别 · LLM Search' },
        status: 'planned', compute: 'GPU recommended',
      },
    ],
  },
  {
    domain: 'ads',
    introduction: {
      en: 'Connect response prediction to calibrated value and auction decisions.',
      zh: '把响应预估逐步连接到价值校准、竞价与预算决策。',
    },
    stages: [
      {
        title: { en: 'Ads systems and objectives', zh: '广告系统与目标' },
        topics: { en: 'Ecosystem · auctions · CTR/CVR metrics', zh: '广告生态 · 拍卖 · CTR/CVR 指标' },
        status: 'planned', compute: 'CPU',
      },
      {
        title: { en: 'Response prediction baselines', zh: '响应预估基线' },
        topics: { en: 'Feature engineering · negative sampling · logistic CTR', zh: '特征工程 · 负采样 · 逻辑回归 CTR' },
        status: 'planned', compute: 'CPU',
      },
      {
        title: { en: 'Deep prediction and calibration', zh: '深度预估与概率校准' },
        topics: { en: 'DeepFM · multi-task learning · calibration', zh: 'DeepFM · 多任务学习 · 概率校准' },
        status: 'planned', compute: 'GPU recommended',
      },
      {
        title: { en: 'Bidding and online optimization', zh: '竞价与在线优化' },
        topics: { en: 'Bidding · budgets · online learning · experiments', zh: '出价 · 预算控制 · 在线学习 · 实验' },
        status: 'planned', compute: 'CPU → GPU',
      },
    ],
  },
  {
    domain: 'recommendation',
    introduction: {
      en: 'Add signals in stages: item attributes, interactions, ranking, then real-time objectives.',
      zh: '按阶段增加信号：物品属性、用户交互、排序，再到实时多目标。',
    },
    stages: [
      {
        title: { en: 'Content signals and item cold start', zh: '内容信号与新物品冷启动' },
        topics: { en: 'Audio features · cosine · KNN · candidate retrieval', zh: '音频特征 · 余弦 · KNN · 候选召回' },
        project: { en: 'Spotify content recommender', zh: 'Spotify 内容推荐' },
        status: 'complete', compute: 'CPU', href: '/projects/spotify-content-recommender/',
      },
      {
        title: { en: 'Collaborative signals', zh: '协同信号' },
        topics: { en: 'Popularity · User-CF · Item-CF · temporal evaluation', zh: '热门基线 · User-CF · Item-CF · 时间切分评估' },
        project: { en: 'MovieLens collaborative filtering', zh: 'MovieLens 协同过滤' },
        status: 'complete', compute: 'CPU', href: '/projects/movielens-collaborative-filtering/',
      },
      {
        title: { en: 'Model-based recommendation', zh: '模型化推荐' },
        topics: { en: 'Matrix factorization · implicit feedback · negative sampling', zh: '矩阵分解 · 隐式反馈 · 负采样' },
        project: { en: 'Latent-factor ranking benchmark', zh: '隐因子排序基准' },
        status: 'planned', compute: 'CPU → GPU',
      },
      {
        title: { en: 'Retrieval, ranking, and online systems', zh: '召回、排序与在线系统' },
        topics: { en: 'Two-Tower · reranking · hybrid signals · online experiments', zh: '双塔 · 重排 · 混合信号 · 在线实验' },
        project: { en: 'Real-time multi-objective study', zh: '实时多目标推荐实验' },
        status: 'planned', compute: 'GPU recommended',
      },
    ],
  },
];

const method = (
  slug: string, domain: Domain, name: Localized, family: Localized,
  requiredSignals: Localized, coldStart: Localized, trainCost: Localized, serveCost: Localized,
  explainability: Localized, bestFor: Localized, limitation: Localized, intuition: Localized,
  useWhen: Localized, avoidWhen: Localized,
): AlgorithmMeta => ({ slug, domain, name, family, requiredSignals, coldStart, trainCost, serveCost, explainability, bestFor, limitation, intuition, useWhen, avoidWhen });

export const algorithms: AlgorithmMeta[] = [
  method('tf-idf', 'search', { en: 'TF-IDF retrieval', zh: 'TF-IDF 检索' }, { en: 'Lexical retrieval', zh: '词法检索' },
    { en: 'Query and document text', zh: '查询词与文档文本' }, { en: 'Works immediately for new text', zh: '新文本可立即检索' },
    { en: 'Low', zh: '低' }, { en: 'Low', zh: '低' }, { en: 'High', zh: '高' },
    { en: 'Small, explainable text search', zh: '小规模、可解释的文本搜索' }, { en: 'Ignores word order and meaning beyond exact terms', zh: '忽略词序，也难以理解同义表达' },
    { en: 'Terms that are frequent in one document but rare across the collection receive more weight.', zh: '某篇文档中常见、但全库少见的词获得更高权重。' },
    { en: 'You need a transparent baseline with no labels.', zh: '没有标注数据，需要透明基线时。' }, { en: 'Users express the same intent with different vocabulary.', zh: '用户经常用不同词表达同一意图时。' }),
  method('bm25', 'search', { en: 'BM25', zh: 'BM25' }, { en: 'Lexical retrieval', zh: '词法检索' },
    { en: 'Query and document text', zh: '查询词与文档文本' }, { en: 'Works immediately for new text', zh: '新文本可立即检索' },
    { en: 'Low', zh: '低' }, { en: 'Low', zh: '低' }, { en: 'High', zh: '高' },
    { en: 'Strong general-purpose keyword retrieval', zh: '通用关键词检索' }, { en: 'Cannot match concepts that share no terms', zh: '无法匹配没有共同词的相似概念' },
    { en: 'BM25 improves lexical scoring by limiting repeated-term gains and normalizing document length.', zh: 'BM25 限制重复词带来的收益，并校正文档长度，让词法匹配更稳健。' },
    { en: 'Exact terms, product attributes, and rare entities matter.', zh: '关键词、商品属性和稀有实体很重要时。' }, { en: 'Semantic matching dominates exact wording.', zh: '语义匹配远比精确用词重要时。' }),
  method('dense-retrieval', 'search', { en: 'Dense retrieval', zh: '向量语义检索' }, { en: 'Semantic retrieval', zh: '语义检索' },
    { en: 'Text plus a pretrained or trained encoder', zh: '文本与预训练或训练后的编码器' }, { en: 'New text can be encoded; new intents may fail', zh: '新文本可编码，但新意图可能失效' },
    { en: 'Medium–high', zh: '中到高' }, { en: 'Medium', zh: '中' }, { en: 'Medium', zh: '中' },
    { en: 'Matching meaning across different wording', zh: '匹配用词不同但含义相近的内容' }, { en: 'Harder debugging and possible exact-term misses', zh: '更难调试，也可能漏掉必须精确匹配的词' },
    { en: 'An encoder turns text into vectors, so nearby vectors represent similar meaning.', zh: '编码器把文本变成向量；向量接近表示语义接近。' },
    { en: 'Synonyms and natural-language questions are common.', zh: '查询包含同义词和自然语言问题时。' }, { en: 'The domain changes faster than the encoder can be updated.', zh: '领域变化快于编码器更新速度时。' }),
  method('learning-to-rank', 'search', { en: 'Learning to Rank', zh: 'Learning to Rank' }, { en: 'Supervised ranking', zh: '监督排序' },
    { en: 'Relevance labels or interaction logs', zh: '相关性标注或交互日志' }, { en: 'Weak for unseen query patterns', zh: '未见查询模式较弱' },
    { en: 'Medium', zh: '中' }, { en: 'Low–medium', zh: '低到中' }, { en: 'Medium', zh: '中' },
    { en: 'Combining many relevance and business signals', zh: '融合多种相关性与业务信号' }, { en: 'Biased labels can teach biased rankings', zh: '有偏标注会训练出有偏排序' },
    { en: 'A ranking model learns how much each signal should affect document order.', zh: '排序模型从数据中学习每种信号应如何影响结果顺序。' },
    { en: 'You have reliable judgments and multiple candidate features.', zh: '有可靠标注和多种候选特征时。' }, { en: 'Only a small or strongly biased click log exists.', zh: '只有少量或严重偏置的点击日志时。' }),
  method('hybrid-search', 'search', { en: 'Hybrid search', zh: '混合搜索' }, { en: 'Multi-channel retrieval', zh: '多路召回' },
    { en: 'Text, lexical index, and vector encoder', zh: '文本、词法索引与向量编码器' }, { en: 'Stronger than either channel alone', zh: '通常比单路冷启动更稳健' },
    { en: 'Medium–high', zh: '中到高' }, { en: 'Medium–high', zh: '中到高' }, { en: 'Medium', zh: '中' },
    { en: 'Balancing exact terms and semantic intent', zh: '兼顾精确词与语义意图' }, { en: 'More infrastructure and score calibration', zh: '基础设施更多，分数也需要校准' },
    { en: 'Lexical and dense candidates are merged, then reranked with a shared scoring rule.', zh: '合并词法与向量候选，再用统一规则重排。' },
    { en: 'Neither lexical nor semantic retrieval is reliable alone.', zh: '词法或语义单独使用都不够稳定时。' }, { en: 'Latency and operational simplicity are strict constraints.', zh: '延迟与运维简单性是硬约束时。' }),

  method('content-based-filtering', 'recommendation', { en: 'Content-Based Filtering', zh: '基于内容的推荐' }, { en: 'Item-feature recommendation', zh: '物品特征推荐' },
    { en: 'Item attributes; optional user profile', zh: '物品属性；可选用户画像' }, { en: 'New items: strong · New users: limited', zh: '新物品强 · 新用户有限' },
    { en: 'Low–medium', zh: '低到中' }, { en: 'Low–medium', zh: '低到中' }, { en: 'High', zh: '高' },
    { en: 'Explainable similarity and new-item discovery', zh: '可解释的相似推荐与新物品发现' }, { en: 'Similarity does not prove user preference', zh: '内容相似不能证明用户喜欢' },
    { en: 'Recommend items whose measurable attributes resemble an item or profile the user already chose.', zh: '推荐在可测属性上接近用户已选择物品或用户画像的内容。' },
    { en: 'Item metadata is rich and interactions are sparse.', zh: '物品属性丰富、交互稀疏时。' }, { en: 'Taste depends on social or collaborative behavior.', zh: '偏好主要来自社交或协同行为时。' }),
  method('user-based-cf', 'recommendation', { en: 'User-Based Collaborative Filtering', zh: '基于用户的协同过滤' }, { en: 'Neighborhood recommendation', zh: '邻域协同推荐' },
    { en: 'User–item interactions', zh: '用户—物品交互' }, { en: 'Weak for new users and new items', zh: '新用户与新物品都较弱' },
    { en: 'Low', zh: '低' }, { en: 'Medium–high', zh: '中到高' }, { en: 'Medium', zh: '中' },
    { en: 'Small communities with stable user overlap', zh: '用户重叠稳定的小型社区' }, { en: 'User neighborhoods become unstable at scale', zh: '规模扩大后用户邻域不稳定' },
    { en: 'Find people with similar histories, then recommend what those neighbors liked.', zh: '先找到历史行为相似的用户，再推荐邻居喜欢的物品。' },
    { en: 'The user base is moderate and tastes overlap.', zh: '用户规模适中且兴趣有明显重叠时。' }, { en: 'Interactions are extremely sparse or users change quickly.', zh: '交互极稀疏或用户兴趣变化很快时。' }),
  method('item-based-cf', 'recommendation', { en: 'Item-Based Collaborative Filtering', zh: '基于物品的协同过滤' }, { en: 'Neighborhood recommendation', zh: '邻域协同推荐' },
    { en: 'User–item interactions', zh: '用户—物品交互' }, { en: 'New items: weak · New users need initial actions', zh: '新物品弱 · 新用户需要初始行为' },
    { en: 'Medium', zh: '中' }, { en: 'Low', zh: '低' }, { en: 'Medium–high', zh: '中到高' },
    { en: 'Stable catalogs and fast item-to-item serving', zh: '稳定目录与快速物品到物品推荐' }, { en: 'Popular items dominate co-occurrence', zh: '热门物品容易主导共现关系' },
    { en: 'Items are similar when many of the same users interact with both.', zh: '如果大量相同用户与两个物品互动，它们就被视为相似。' },
    { en: 'Item relationships change more slowly than user tastes.', zh: '物品关系比用户兴趣变化更慢时。' }, { en: 'New inventory arrives constantly without interactions.', zh: '新物品持续进入且缺少交互时。' }),
  method('matrix-factorization', 'recommendation', { en: 'Matrix Factorization', zh: '矩阵分解' }, { en: 'Latent-factor recommendation', zh: '隐因子推荐' },
    { en: 'User–item interactions', zh: '用户—物品交互' }, { en: 'Weak for new users and new items', zh: '新用户与新物品都较弱' },
    { en: 'Medium', zh: '中' }, { en: 'Low–medium', zh: '低到中' }, { en: 'Low', zh: '低' },
    { en: 'Learning compact preference patterns from sparse data', zh: '从稀疏数据中学习紧凑偏好模式' }, { en: 'Latent factors are difficult to explain', zh: '隐因子难以直接解释' },
    { en: 'Represent each user and item with learned vectors whose dot product predicts preference.', zh: '把用户和物品表示为学习得到的向量，用点积预测偏好。' },
    { en: 'Interaction data is large enough to reveal shared taste.', zh: '交互量足以体现共同兴趣时。' }, { en: 'Cold start and attribute-level explanations are central.', zh: '冷启动和属性级解释是核心需求时。' }),
  method('two-tower', 'recommendation', { en: 'Two-Tower Model', zh: '双塔模型' }, { en: 'Neural candidate retrieval', zh: '神经网络候选召回' },
    { en: 'Interactions plus user and item features', zh: '交互数据及用户、物品特征' }, { en: 'Medium with usable side features', zh: '有侧信息时冷启动能力中等' },
    { en: 'High', zh: '高' }, { en: 'Low with vector index', zh: '配合向量索引后较低' }, { en: 'Low', zh: '低' },
    { en: 'Large-scale personalized candidate retrieval', zh: '大规模个性化候选召回' }, { en: 'Needs substantial data, tuning, and retrieval infrastructure', zh: '需要大量数据、调参与向量检索设施' },
    { en: 'Separate networks encode users and items into a shared vector space for fast matching.', zh: '两个网络分别把用户和物品编码到同一向量空间，再快速匹配。' },
    { en: 'The catalog and traffic justify neural training and ANN serving.', zh: '目录和流量足以支撑神经网络训练与 ANN 服务时。' }, { en: 'A simpler collaborative or content method already meets the goal.', zh: '更简单的协同或内容方法已满足目标时。' }),

  method('logistic-ctr', 'ads', { en: 'Logistic Regression CTR', zh: '逻辑回归 CTR' }, { en: 'Linear response prediction', zh: '线性响应预估' },
    { en: 'Impressions, clicks, and encoded features', zh: '曝光、点击与编码特征' }, { en: 'New entities depend on shared features', zh: '新实体依赖共享特征' },
    { en: 'Low', zh: '低' }, { en: 'Low', zh: '低' }, { en: 'High', zh: '高' },
    { en: 'Fast, calibrated CTR baseline', zh: '快速且便于校准的 CTR 基线' }, { en: 'Cannot learn complex feature interactions directly', zh: '无法直接学习复杂特征交互' },
    { en: 'A weighted sum of features is converted into a click probability.', zh: '对特征加权求和，再转换为点击概率。' },
    { en: 'Latency, stability, and probability calibration matter.', zh: '延迟、稳定性与概率校准重要时。' }, { en: 'Most signal lies in nonlinear feature interactions.', zh: '主要信号来自非线性特征组合时。' }),
  method('gbdt-lr', 'ads', { en: 'GBDT + LR', zh: 'GBDT + LR' }, { en: 'Tree-assisted CTR prediction', zh: '树模型辅助 CTR 预估' },
    { en: 'Impressions, clicks, dense and categorical features', zh: '曝光、点击、数值与类别特征' }, { en: 'Medium through shared tree paths', zh: '通过共享树路径获得中等能力' },
    { en: 'Medium', zh: '中' }, { en: 'Low–medium', zh: '低到中' }, { en: 'Medium', zh: '中' },
    { en: 'Tabular CTR data with nonlinear interactions', zh: '包含非线性交互的表格 CTR 数据' }, { en: 'Two-stage training and feature staleness', zh: '两阶段训练且树特征可能过时' },
    { en: 'Trees discover useful feature combinations; logistic regression turns them into probabilities.', zh: '树模型发现有效特征组合，逻辑回归再把它们转换为概率。' },
    { en: 'A linear model underfits but deep training is unnecessary.', zh: '线性模型欠拟合，但暂不需要深度模型时。' }, { en: 'Features and traffic patterns change extremely quickly.', zh: '特征与流量模式变化极快时。' }),
  method('deepfm', 'ads', { en: 'DeepFM', zh: 'DeepFM' }, { en: 'Deep CTR prediction', zh: '深度 CTR 预估' },
    { en: 'Large impression logs and categorical features', zh: '大规模曝光日志与类别特征' }, { en: 'Weak for unseen IDs; better with side features', zh: '未知 ID 较弱；侧信息可改善' },
    { en: 'High', zh: '高' }, { en: 'Medium', zh: '中' }, { en: 'Low', zh: '低' },
    { en: 'Automatic low- and high-order feature interactions', zh: '自动学习低阶与高阶特征交互' }, { en: 'More tuning and harder probability diagnosis', zh: '调参更多，概率问题更难诊断' },
    { en: 'A factorization component learns pairwise interactions while a neural network learns higher-order patterns.', zh: '因子分解部分学习两两交互，神经网络学习更高阶模式。' },
    { en: 'Traffic and feature cardinality justify a deep model.', zh: '流量和特征规模足以支撑深度模型时。' }, { en: 'A linear or tree model already meets business goals.', zh: '线性或树模型已满足业务目标时。' }),
  method('multi-task-ads', 'ads', { en: 'Multi-Task Prediction', zh: '多任务预估' }, { en: 'Multi-objective response prediction', zh: '多目标响应预估' },
    { en: 'Clicks plus conversion or value labels', zh: '点击及转化或价值标签' }, { en: 'Depends on shared features and task transfer', zh: '依赖共享特征与任务迁移' },
    { en: 'High', zh: '高' }, { en: 'Medium', zh: '中' }, { en: 'Low', zh: '低' },
    { en: 'Joint click, conversion, and value prediction', zh: '联合预估点击、转化与价值' }, { en: 'Task conflict can harm one objective', zh: '任务冲突可能损害某个目标' },
    { en: 'Related outcomes share representations while retaining separate prediction heads.', zh: '相关目标共享底层表示，同时保留各自预测头。' },
    { en: 'Clicks are abundant but downstream conversions are sparse.', zh: '点击丰富、下游转化稀疏时。' }, { en: 'Tasks are weakly related or labels have different delays.', zh: '任务关联弱或标签延迟差异很大时。' }),
  method('bid-optimization', 'ads', { en: 'Bid Optimization', zh: '出价优化' }, { en: 'Auction decision policy', zh: '竞价决策策略' },
    { en: 'Response predictions, value, cost, and budget', zh: '响应预估、价值、成本与预算' }, { en: 'Requires fallback rules for new campaigns', zh: '新广告活动需要规则兜底' },
    { en: 'Medium–high', zh: '中到高' }, { en: 'Low under strict latency', zh: '严格延迟下较低' }, { en: 'Medium', zh: '中' },
    { en: 'Converting predictions into budget-aware auction actions', zh: '把预估转化为受预算约束的竞价动作' }, { en: 'Sensitive to calibration and market feedback loops', zh: '对概率校准和市场反馈循环敏感' },
    { en: 'Choose bids that maximize expected value while respecting cost and delivery constraints.', zh: '在成本与投放约束下选择能最大化期望价值的出价。' },
    { en: 'Prediction quality is reliable and the business objective is explicit.', zh: '预估可靠且业务目标明确时。' }, { en: 'CTR is uncalibrated or budget constraints are not modeled.', zh: 'CTR 未校准或预算约束没有建模时。' }),
];

export const oppositeLang = (lang: Lang): Lang => (lang === 'en' ? 'zh' : 'en');
export const basePath = import.meta.env.BASE_URL.replace(/\/$/, '');
export const href = (path: string) => `${basePath}${path}`;
