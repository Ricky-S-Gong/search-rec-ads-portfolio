# 第一次运行：请从这里开始

这个包已完成本地代码准备；你要在 Colab 完成 GPU 验收与正式训练。不要把“运行包准备好”理解为 B0/R0 已训练完成。

## 你现在先做什么

1. 下载 `KuaiRand_Stage12_Ready_Pack.zip`，在电脑上双击解压。
2. 打开 https://colab.research.google.com/ ，选择“文件 → 上传笔记本”，上传解压后的 `START_HERE_Ziqi_Stage12.ipynb`。
3. 在“运行时 → 更改运行时类型”选择 GPU。如果可用，优先选 RAM 足够的 runtime；A100可优先使用。不要先改模型 batch。
4. 从第1步开始，每次只点一格左侧的 ▶。第一格要上传的是**完整 ZIP**，不是 notebook，也不是旧训练包。
5. 每格完成后才运行下一格。看到红色报错就停下，保留完整报错文字。
6. 先运行到第8步 `READY FOR B0`，再运行第11步下载反馈包，发回当前聊天。我可以检查后帮助你判断接下来如何跑。

## 你需要准备的账号/权限

- Google账号：能读取团队共享的SAR文件夹，自己的Drive有足够剩余空间。
- Colab GPU：需要在界面中连接。网页关掉或断线时，后台不一定马上停。
- W&B：能向 `SAR-Lab/kuairand-two-tower` 写入。把 API key 放在 Colab 左侧钥匙图标的 Secrets，名称 `WANDB_API_KEY`，开启 Notebook access。不要把密钥发到聊天。
- 运行包沿用原实验的 W&B API 地址；如果你使用其它托管服务器，先核对其API地址。登录/权限错误请发文字，不要发密钥。

## 准备、第一阶段、第二阶段分别做什么

| Notebook 步骤 | 你做的动作 | 成功标志 | 是否正式实验 |
| --- | --- | --- | --- |
| 1 | 上传ZIP、解压与校验 | SOURCE CHECK PASS | 否 |
| 2 | Google授权与挂载Drive | DRIVE READY | 否 |
| 3 | 等待环境安装，确认GPU | ENVIRONMENT PASS | 否 |
| 4 | 设置Secret并运行 | SECRET READY，密钥不会显示 | 否 |
| 5 | 小样本测试、B0/R0中断恢复、W&B上传 | STAGE 1 PASS | 否；小样本分数不算成绩 |
| 6 | 下载并校验train/validation与enriched特征 | DATA READY | 否；不下载test目标 |
| 7 | 六个架构的真实数据零更新检查 | REAL PREFLIGHT PASS | 否 |
| 8 | 每配置200次更新，检测真实数据运行情况 | READY FOR B0 | 否；权重不带入正式训练 |
| 9 | 把START_B0改成True再运行 | B0 FINISHED | 是，B0 seed42 |
| 10 | B0完成后把START_R0改成True | R0 FINISHED | 是，R0 seed42 |
| 11 | 下载反馈ZIP | SEND_TO_CHATGPT.zip | 用于反馈 |

当前 notebook 的所有正式训练开关默认关闭。不要点击“全部运行”。先完成第1–8步，第9/10步可以等你有足够算力和时间时再开启。

## 什么叫epoch、checkpoint和validation

- 一个epoch：训练数据完整经过模型一次。
- step/update：用一小批训练样本做一次权重更新。这里batch=256，每epoch约5218次更新。
- checkpoint：训练进度存档。`latest.pt`是最近进度，断线后继续训练用它；`best.pt`是截至目前validation最好的一轮。
- validation：用验证数据衡量效果，不更新权重。这里要对437万余视频做完整排序，可能很慢。
- early stopping：如果连续3轮没有更好的validation表现，就停止，不一定跑满20轮。这是正常结束。
- W&B Artifact：在线保存的模型存档，可追溯到具体run和epoch。

## 需要发给我什么

最省事的方法：运行第11步，把下载的 `SEND_TO_CHATGPT.zip` 上传到聊天。它不包含原始数据、模型权重或Secret。

建议三个时间点各发一次：

1. 第8步完成后：我核对GPU型号、特征维度、六组preflight与200-step探测、恢复和上传证据。
2. B0完成后：我核对每epoch曲线、best_epoch、stop_reason、coverage、checkpoint与Artifact。
3. R0完成后：我核对同一训练配置，准备控制组交接与下一阶段架构实验。

如果失败也发反馈包，再加一句“卡在第几步”。如果第1–3步就失败、反馈包无法生成，先粘贴该格完整红色报错即可。不要只发最后一句“报错了”。

## 不需要一直盯着屏幕，但要检查这几次

- 刚开始：看config/seed、131D用户输入/64D物品输入和GPU名字。
- 首个epoch结束：看loss是正常数字、Recall/NDCG、coverage，以及epoch文件和Artifact同步。
- 之后每隔一段时间看一次会话是否活着，尤其Colab提示即将断开或空间不足时。
- 结束：看 `complete=true`、`stop_reason=max_epochs/early_stopping`、`best_epoch` 与Artifact同步。

`phase=validation`、`catalog encoding`、`evaluation .../555677` 表示验证还在进行；这时训练step不增长是正常的。loss不需要每轮严格下降。真实Recall很低也不代表程序一定错，先按一致协议看结果。

## 断线后怎么继续

1. 重新连接GPU，运行第1–4步，使用同一个ZIP、同一个Google账号和同一路径。
2. 检查旧会话是否已经停止，避免两个会话同时写一个任务。出现Task occupied时，只在确认旧runtime已结束后使用notebook最后的解锁格。
3. 同GPU/源码/环境下可复用已通过的检查；第5步会核对已有smoke的GPU、源码与torch版本。若不匹配，保留旧目录，换新的SMOKE_NAME重新验收。
4. 第6步已下载的数据会校验并跳过重复下载；第7步可重新检查。200-step probe完成的任务会跳过。
5. 再运行第9或第10步，会自动从 `latest.pt` 恢复，保留原run ID。
6. 出现binding differs时停下来发反馈；不要删除checkpoint、修改hash或更换单组超参数。

## 本轮保持不变的规则

B0=原shallow MLP+mean历史；R0=Deep MLP+Attention。两者都无BN/Residual，dropout=0，No Correction；Adam lr=.001，batch=256，τ=.1，最多20epochs/patience3。

本次不启动A1/A2/A3/A5的完整训练、不跑H5–H8、不读取test选模型。它们的配置与短程检查已准备，正式架构实验等控制组通过后继续。

## 给Annabel的交接

R0正式完成后，将同一个运行包、`training/configs/r0.yaml`、反馈结果、W&B run链接和Drive中的checkpoint索引交接。不要把短程probe当R0成绩；也不要只分享权重而漏掉源码/数据/配置版本。本包未替Annabel完成balanced sampler等采样实验。
