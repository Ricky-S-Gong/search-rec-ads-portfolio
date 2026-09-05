# Repository Context

Last checked: 2026-08-23 (America/New_York)

本文件记录本次 `git pull` 后的仓库事实，供后续开发会话使用。实验数字以生成脚本和 `public/artifacts/` 中的版本化产物为准，不要为了叙事效果手工修改指标。

## Git 与 GitHub 状态

- Repository: `https://github.com/Ricky-S-Gong/search-rec-ads-portfolio`
- Local path: `/Users/ricky/Desktop/Rocommendation`
- Production: `https://ricky-s-gong.github.io/search-rec-ads-portfolio/`
- Default branch: `main`
- `origin/main`: `846bbbf` — `feat: make MovieLens CF bias-aware (#22)`
- 当前工作树分支：`feature/20-movielens-evidence-flow`
- 当前分支提交：`4c36a51` — `feat: refine MovieLens evidence flow`
- 当前分支已与远端同名分支同步，但相对 `origin/main` 为 `ahead 1 / behind 2`。PR #21 已通过提交 `6f2b1d1` 合并进 `main`，所以开始新开发前应切回并更新 `main`；不要把旧功能分支误当成最新代码。
- 本次拉取发现新远端分支 `origin/ziqi/movielens-bias-aware-v4`；其工作已通过 PR #22 合并到 `main`。
- GitHub 当前没有 open PR。
- Open issue 仅见 #20 `Refine MovieLens evidence flow and evaluation explanation`，但相关 PR #21 已合并，issue 状态可能需要人工关闭或确认。
- 最新 `main` 工作流 `Check and deploy`（run `32616629348`）已于 2026-08-23 成功完成。这里只确认 Actions 结果，没有在本次会话中人工 smoke-test 线上页面。

`HANDOFF.md` 的仓库快照仍停留在 `7af273a` / MovieLens v3，已经过期；涉及当前状态时优先参考本文件、`origin/main` 和生成产物。

## 产品现状

这是一个面向面试展示的中英双语 Search / Ads / Recommendation 算法作品集：

- 英文路由：`/en/`
- 中文路由：`/zh/`
- Spotify 内容推荐：完成，CPU 实验，有可复现实验产物和交互式 weighted-cosine demo。
- MovieLens 协同过滤：仍标记为进行中，CPU 实验；最新 `main` 已升级到 bias-aware User-CF / Item-CF（v4）。
- Search：尚未选定项目和数据集。
- Ads：尚未选定项目和数据集。
- About 页公开成员为 Ricky Gong、Ziqi Xu、Yutao Rao；项目作者字段按实际贡献者单独维护。

不要为尚未完成或尚未选题的项目虚构数据、指标或线上业务效果。中英文内容必须同步更新。

## 最新变更

### PR #21：MovieLens evidence flow

- 改进 MovieLens 案例页的信息顺序、公式说明、数据证据和 explorer 呈现。
- 增加 `FormulaLegend.astro`、`MovieLensFlow.astro` 等展示组件。
- 补充 split、artifact、evaluation 的可追溯信息和测试。

### PR #22：MovieLens bias-aware v4

- 将 User-CF / Item-CF 从旧的 mean-centered 版本升级为 baseline-residual cosine 的 bias-aware 版本。
- 引入全局均值、用户偏置和物品偏置基线，并使用验证集选择超参数。
- 加入 Bayesian popularity 的稳定次级排序，消除 v3 中严重的 Top-10 精确并列。
- 增加用户活跃度分段、`rating >= 5` 只读敏感性分析、paired bootstrap 对比和运行时诊断。
- 产物 schema/version 更新为 `movielens-cf-v4`。
- v4 是在观察到 v3 test tie failure 后进行的 post-hoc 迭代；虽然超参数仍只用 validation 选择，但不能把它描述成完全未接触 test 的 confirmatory experiment。

## 关键实验结论

### Spotify

Source of truth: `public/artifacts/spotify/metrics.json` and `research/spotify-music/`.

- 数据集：170,653 tracks，1921–2020，9 个 audio features。
- Exact cosine median latency: `1.6629 ms`。
- K-Means candidate retrieval median latency: `0.3168 ms`。
- Cluster Recall@10 against exact cosine: `0.7775`。
- Euclidean KNN vs cosine Top-10 overlap: `0.4075`。
- Diversity@10: `0.003558`，结果高度集中，不应描述为广泛探索。
- 数据没有用户相关性标签，因此这些是 proxy/system metrics；不要报告 Precision@K、NDCG 或用户偏好概率。
- 页面显示的 similarity 是 weighted cosine × 100 后的百分比格式，不是概率或准确率。

### MovieLens v4

Source of truth: `public/artifacts/movielens/metrics.json` and `research/movielens-cf/` on `origin/main`.

- 数据集：MovieLens 1M，1,000,209 条 explicit ratings。
- Protocol：per-user temporal 80/10/10；最终用 train + validation refit；候选集是完整 fitted catalog 去除 seen movies；主 relevance 为 `rating >= 4`。
- Split：797,758 train；99,692 validation；897,450 fitted；102,759 test；ranking cohort 5,820 users；fitted catalog 3,683 movies。
- Bayesian popularity：Hit Rate@10 `0.17921`，NDCG@10 `0.03355`，Recall@10 `0.02689`，coverage `0.02769`。
- Bias-aware User-CF：Hit Rate@10 `0.21942`，NDCG@10 `0.03924`，Recall@10 `0.03456`，coverage `0.25496`，long-tail share `0.05811`，RMSE `0.97324`。
- Bias-aware Item-CF：Hit Rate@10 `0.22165`，NDCG@10 `0.04162`，Recall@10 `0.03421`，coverage `0.43334`，long-tail share `0.13820`，RMSE `0.94851`。
- 当前主要结论：v4 的个性化 CF 在主 ranking metrics 上超过 popularity；Item-CF 的 NDCG 和 Hit Rate 略高，User-CF 的 Recall 略高；Item-CF 同时覆盖更多 catalog 和 long-tail。不存在所有维度上的单一赢家。
- v4 personalized lists 的 exact tie 已降为 0；不要继续把 v3 的 `77.96% fully tied` 当作当前模型结果，它只属于 legacy diagnostic。
- Rating baseline 本身的 RMSE 为 `0.92445`，优于两种邻域模型；因此不能宣称 CF 在 rating prediction 上全面胜出。
- `rating >= 5` 是只读敏感性分析，不参与模型选择，也不替代主协议。
- 所有结果都是 offline explicit-rating 指标，不代表 CTR、watch time、retention、revenue 或 causal lift。

## 技术结构

- Astro 5 static site + TypeScript。
- React 19 islands 用于交互组件，主要是 `SpotifyLab.tsx` 和 `MovieLensExplorer.tsx`。
- KaTeX 数学公式；本地 IBM Plex / Noto 字体。
- Python 3.11–3.13；pandas、NumPy、SciPy、scikit-learn、Matplotlib 用于实验。
- GitHub Pages 由 `.github/workflows/ci-pages.yml` 构建和部署。
- 无 backend、database、online inference service 或外部运行时 CDN。

关键目录：

```text
src/pages/[lang]/                 # 中英文静态路由
src/components/                   # Astro / React UI
src/components/cosmos/            # 首页宇宙场景
src/data/site.ts                  # 项目、成员、roadmap、共享文案
src/styles/global.css             # 全局设计系统
public/artifacts/spotify/         # Spotify 版本化前端产物
public/artifacts/movielens/       # MovieLens 版本化前端产物
research/spotify-music/           # Spotify 实验与测试
research/movielens-cf/            # MovieLens 实验与测试
.github/workflows/ci-pages.yml    # CI 和 Pages 部署
```

原始数据集不提交到 Git；网站只读取生成后的轻量产物。修改实验逻辑时应重新运行 pipeline 生成产物，而不是单独修改 JSON 或 SVG 数字。

## 本地开发与验证

```bash
npm install
npm run dev

uv sync --locked

npm test
npm run check
npm run build
uv run pytest
git diff --check
```

如果本地 `pytest` launcher 在 collection 前挂住，可以使用仓库 fallback：

```bash
.venv/bin/python research/run_python_tests.py
```

不要声称测试通过，除非在当前会话实际运行过。此次只更新上下文文档，未运行完整 JS/Python/build 测试。

## 下一步建议

1. 开始新任务前切换到 `main` 并拉取 `846bbbf` 之后的最新提交；当前旧功能分支不应继续作为基线。
2. 确认并关闭已完成的 GitHub issue #20（如无剩余验收项）。
3. 人工 smoke-test 线上 `/en/`、`/zh/` 与 MovieLens explorer，补足 Actions 成功之外的部署验证。
4. MovieLens 仍是 in-progress；完成前继续核对双语叙事、v4 指标、post-hoc caveat 和交互展示一致性。
5. Search / Ads 开工前先确定 dataset license、labels、evaluation protocol 和 compute budget，不要恢复未经确认的旧候选项目。
6. `README.md` 仍将作品集描述为 Ricky Gong & Ziqi Xu，但 About 页已有 Yutao Rao；修改前先确认作者署名范围。
