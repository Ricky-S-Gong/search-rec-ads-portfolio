# 第一周技术报告

`week1.tex` 是当前可编辑的中文报告，`week1.pdf` 为对应 PDF。
正文保留 Pure 数据范围、时间审计、实际输入、模型与损失、训练与评估结果、主要限制和下一周安排。长校验值、历史实验细节和完整原始特征列表不再重复放入正文。

`report_evidence.json` 与 `evidence/` 保留冻结版本、三组评估和训练曲线的聚合依据。私有 checkpoint、向量和逐请求预测留在共享 Drive。

当前正文仅包含 `week1-pure-content-v1-001` 的最新实验，依据位于 `evidence/content-v1/`。执行代码为 `562682341f1b7a9cc49e518fc43b948052ecf871`；两组物品输入均为 7 个类别与 4 个数值，共 60 维，Enhanced 历史复用同一物品塔。此前 ID-only 聚合证据仍保留供追溯，不进入当前正文。

完整新产物：[Drive release](https://drive.google.com/drive/folders/18qFm162pVnjsusK1s0mux6dTWIbtdKZw)。训练与评估入口为 `../../training/notebooks/week1_content_retraining.ipynb`；执行输出保存在原 Drive notebook 的同一链接。发布属性采用固定内容假设，测试集保持封存。

优先在 Codex 内置 LaTeX 编辑器中编译 `week1.tex`。源码使用 CJKutf8、标准 LaTeX 表格和内嵌 PGFPlots，无需额外图片或 BibTeX 文件。导出 PDF 时只更新 `week1.pdf`；本地编译的中间文件应放入临时目录并清除。

`build_report.py` 是之前英文版本的产物生成器，保留用于历史复现；它不包含本次中文精简和排版编辑，不能用其输出直接覆盖当前报告。当前章节数据应与相邻 evidence JSON 核对。测试集不参与报告。
