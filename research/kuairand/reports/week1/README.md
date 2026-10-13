# KuaiRand 第一周技术报告

[中文 PDF](week1.pdf) · [可编辑 LaTeX](week1.tex) · [版本索引](report_evidence.json) · [最新实验 JSON](evidence/content-v1/stage_report.json)

正文只呈现 `week1-pure-content-v1-001` 内容属性实验，实际执行代码为 `562682341f1b7a9cc49e518fc43b948052ecf871`。Basic / Enhanced 两组物品输入均为七个类别与四个数值，共 60 维；Enhanced 用户历史复用本模型的物品塔。旧 ID-only 聚合证据保留追溯，不进入当前正文。

## Contents / 内容

- Pure 与 27K / 1K 的区别、实际规模、时间切分和历史预热。
- 用户／物品塔实际输入、原始值到编码示例、完整逐层架构图。
- 随机／常规曝光定义、`long_view` 监督、对角线分数矩阵、重复目标屏蔽和批内 loss 的解释边界。
- 真实训练曲线、训练成本、checkpoint/向量与受控恢复验证。
- 验证请求定义、热门与双塔排序、Recall 分子分母、单目标 NDCG 的 ground truth、覆盖和评估成本。
- 下一周：1K 预处理入 Drive、基线及随机曝光评估协议设计、架构调整、超参数实验、逐 epoch 评估和最佳验证 checkpoint 部署规划。

## Evidence / 依据

当前聚合依据位于 `evidence/content-v1/`，包括 audit、两组训练摘要、三组评估、release manifest 与 stage report。报告曲线来自训练摘要，表格与评估 JSON 核对；逐请求命中/NDCG 已独立复算。真实私有数据、checkpoint、向量和榜单不放入 Git。

[完整 Drive release](https://drive.google.com/drive/folders/18qFm162pVnjsusK1s0mux6dTWIbtdKZw) · [已执行 Colab notebook](https://colab.research.google.com/drive/1rQwo_Fe_HufE3q7VvA1RTNmPiDBHQ_Yv) · [仓库训练 notebook](../../training/notebooks/week1_content_retraining.ipynb)

常规曝光验证受旧推荐策略影响；随机曝光基线仍待设计。发布属性采用固定内容假设，恢复验证只覆盖同一 runtime 内的受控中断。正式测试集未参与报告，当前模型没有超过热门基线。

## Edit and compile / 编辑与编译

直接修改 `week1.tex`，优先使用 Codex 内置 LaTeX 编译器和 PDF 预览。源码使用 CJKutf8、标准表格和内嵌 TikZ/PGFPlots；没有外部图片或 BibTeX 依赖。本地导出使用含中文字体的 LaTeX 环境，在独立临时目录编译，仅保留源文件与最终 `week1.pdf`，清除 `.aux/.log/.out/.toc` 等中间文件。

编译后逐页检查特征名是否单行、图箭头与标题留白、公式、引用、分页和表格溢出。`build_report.py` 是历史英文报告生成器，不能用其输出覆盖当前人工整理的中文报告。历史交接与旧 evidence 不代表最新结果。
