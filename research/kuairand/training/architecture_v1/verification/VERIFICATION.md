# 验证记录与边界

日期：2026-10-31。

## 已实际执行

- 六个唯一架构配置一致性检查；18个config×seed任务，A4复用B0。
- 24项本地测试：22项通过，2项跳过，0失败、0错误。
- 测试覆盖：原有训练/采样逻辑、所有架构的enriched输入、0/1/50条历史与padding、LSTM顺序、重复item mask、固定口径validation loss、按样本加权epoch train loss、每epoch checkpoint、精确中断恢复、上传失败不误报成功、checkpoint已提交但manifest未写完时的恢复。
- 独立CPU合成smoke：六个配置preflight通过；B0/R0均完成不中断训练与中断恢复比较，模型/optimizer/scaler/RNG/step等一致。
- Notebook JSON与所有Python代码格语法检查；ZIP解压结构与源码校验清单检查。

## 本地环境

Python3.12.2、PyTorch2.2.2、NumPy1.26.4、PyArrow15.0.2，CPU。使用已有依赖，未修改用户的conda环境。

原锁定环境要求较新PyTorch unified GradScaler API，本机版本缺失。因此仅在本地测试进程内使用旧CUDA scaler API（CPU下disabled）的兼容shim；交付的engine和Colab notebook不包含该shim。CPU GradScaler溢出重试测试、CUDA/cuDNN专用测试在本地跳过；Colab使用锁定环境执行原生测试。

## 尚未执行，必须在用户Colab完成

- uv.lock对应的Python3.11/CUDA环境安装与运行；GPU/cuDNN/AMP原生检查。
- 真实1K数据读取、特征校验、六组零更新preflight和200-step探测。
- 实际W&B登录、写权限、Artifact上传/下载服务器连通性；本地仅测试上传失败状态处理。
- B0/R0正式seed42训练及其validation结果。

本地合成smoke结果不是1K实验成绩，不能填入正式结果表。完整日志见local_tests.log/local_smoke.log；机器可读结论见local_test_result.json与local_synthetic_smoke.json。
