# Search · Ads · Recommendation Portfolio

[Live site](https://ricky-s-gong.github.io/search-rec-ads-portfolio/) · [Contribution guide](CONTRIBUTING.md) · [Workspace guide](docs/workspace.md)

A bilingual recommendation portfolio with reproducible experiments, measured results and interactive case studies. 当前仓库同时包含 Astro/React 展示网站、CPU 推荐实验和 KuaiRand GPU 双塔研究；搜索与广告是作品集方向，尚无独立的广告训练项目。

## Projects / 当前项目

| 项目 | 当前实现与状态 | 文档 |
| --- | --- | --- |
| Spotify 内容推荐 | 音频属性相似度、Euclidean KNN、精确／聚类 cosine；已发布内容一致性与检索成本指标 | [Spotify](research/spotify-music/README.md) |
| MovieLens 协同过滤 | Bayesian popularity、bias-aware User-CF / Item-CF；已发布 v4 评估和置信区间 | [MovieLens](research/movielens-cf/README.md) |
| Goodreads Poetry 矩阵分解 | Basic MF、FunkSVD、ALS、NMF、SVD++，另含 Bias-aware ALS；统一比较与前端产物已完成 | [GoodBooks](research/goodbooks-mf/README.md) |
| KuaiRand 双塔召回 | Pure 内容属性基线已完成训练、向量导出、精确评估和受控 GPU 恢复检查；1K 扩展与基线协议设计待完成 | [KuaiRand](research/kuairand/README.md) |

这些项目的指标定义和评估人群不同，不能直接横向比较数值。已提交实验是离线结果，不代表线上收益或推荐服务已经部署。

## Run the website / 网站启动

Node.js 22；使用已提交的 npm lockfile：

```bash
npm ci
npm run dev
# 静态构建与本地预览
npm run build
npm run preview
```

页面内容和交互组件位于 `src/`，读取 `public/artifacts/` 中的版本化产物。GitHub Pages 在 `main` 的构建检查通过后部署；PR 只运行检查，不部署。

## Python environments / 研究环境

两个环境独立管理，不复制 `.venv`，不跨环境更新 lockfile：

| 环境 | Python | 用途 |
| --- | --- | --- |
| 仓库根目录 | `.python-version`：3.12 | Spotify、MovieLens、GoodBooks、KuaiRand 初始数据流水线 |
| `research/kuairand/training/` | 3.11.15 | KuaiRand 模型、训练、导出及统一评估；真实训练在 Colab CUDA GPU 执行 |

两处配置均要求 uv **0.12.1**。首次安装 uv 后，在仓库根目录执行：

```bash
python3 -m pip install uv==0.12.1
uv sync --locked
uv sync --locked --directory research/kuairand/training --python 3.11.15
```

各项目 README 提供具体数据与实验入口；阅读现有结果无需重新下载数据或训练。

## Validation / 验证

```bash
npm run build
npm test
uv run --locked pytest
uv run --locked --directory research/kuairand/training pytest tests -q
uv run --locked --directory research/kuairand/training ruff check . ../evaluation --exclude vendor --config pyproject.toml
```

根目录 pytest 不包含独立训练环境的测试，必须分别运行。`scripts/run_python_tests.py` 仅作为受限环境中 pytest 无法启动时的兼容入口；正常流程使用 pytest。CI 定义见 [网站与研究检查](.github/workflows/ci-pages.yml) 和 [KuaiRand CPU 检查](.github/workflows/kuairand-training.yml)。CI 不执行真实 GPU 训练。

## Repository layout / 目录

| 路径 | 职责 |
| --- | --- |
| `src/pages/`, `src/layouts/` | 网站路由与布局 |
| `src/components/shared/` | 导航、卡片、数学展示等公共组件 |
| `src/components/{cosmos,spotify,movielens,goodbooks}/` | 视觉主题与项目交互组件 |
| `src/lib/`, `src/data/`, `src/styles/` | 公共逻辑、中英文内容、样式 |
| `public/artifacts/` | 网站读取的轻量版本化产物 |
| `research/<project>/` | 实验代码、配置、测试与聚合结果 |
| `research/kuairand/{data_pipeline,training,evaluation,reports}/` | 数据准备、GPU 训练、精确召回评估和 LaTeX 报告 |
| `scripts/`, `docs/`, `.github/` | 通用命令、工作区说明、CI 与协作模板 |

当前 KuaiRand 中文报告：[PDF](research/kuairand/reports/week1/week1.pdf) · [LaTeX 源文件](research/kuairand/reports/week1/week1.tex)。详细特征、监督信号和评估定义以报告及对应 JSON 为依据。

## Data, artifacts and collaboration / 数据与协作

- Git：代码、测试、配置、文档和允许发布的聚合证据。原始／处理后交互、凭证和大型模型文件不提交。
- KuaiRand Drive：共享数据、完整 checkpoint、向量、逐请求预测和 release；W&B：训练日志；Colab：执行环境。
- CPU 项目数据：各项目忽略的 `data/` 目录；下载权限及数据许可见项目指南。Spotify 浏览器示例使用脚本生成的精简曲目产物，不是用户行为数据。
- 使用 Issue 和 PR 协作，先核对冻结版本，再恢复训练或发布产物。历史说明位于 `docs/archive/` 与项目交接文档，不作为当前路径说明。

Code: MIT. Original prose and charts: CC BY 4.0. Third-party datasets and material retain their own terms; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
