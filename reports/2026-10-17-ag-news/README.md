# AG News：统一入口的完整 GPU 实验

2026-10-17，使用之前运行 Tetris 实验的 EC2 `i-0daca64f56056aa1c`
（g6.xlarge / NVIDIA L4），通过根目录 `finetune.py` 完成一次完整 SFT。
数据是独立的 AG News 新闻分类子集；任务是根据新闻标题和摘要，从四个栏目中选择一个。
标签使用公开数据集原始标签，没有教师调用。

**完整流程和预先配置的质量门槛均通过，退出码为 0。**
测试准确率只多答对 1 题，候选换序的一致性有所下降，因此不能将本轮结果解读为显著或全面提升。

## 独立数据与测试结果

训练 1,600 条；验证、校准、测试各 200 条。每份数据的四个类别均衡。
测试数据来自官方 test，其余三份来自官方 train。
排除了规范化后相同标题或摘要，并逐条核对了 2,200 个标签与原始 CSV 的对应关系。

四份实际输入已归档到 [data/](data)，可直接作为统一入口的 `--data` 参数。
来源、原始文件 SHA-256、抽样种子及分区散列见 [provenance.json](data/provenance.json)；
标签核对记录见 [source-verification.json](data/source-verification.json)。

| 指标 | 原版 v19 | 微调版 |
| --- | ---: | ---: |
| 测试准确率 | 90.5%（181/200） | 91.0%（182/200） |
| 候选反序并改名后的准确率 | 90.0%（180/200） | 92.5%（185/200） |
| 两种顺序下实际选择一致率 | 98.5%（197/200） | 96.5%（193/200） |
| 测试平均 NLL | 0.237844 | 0.226055 |

门槛在训练前设定：准确率至少 80%、不低于原版、选择一致率至少 85%。
三项均通过；一致率门槛并不要求相对原版改善。
原版与微调版使用同一份独立测试集，未根据测试结果调整训练参数或门槛。

这是一轮训练、200 条测试新闻的结果，不能证明统计显著性或其他业务任务的提升。
标题/摘要去重不能排除同一事件的近似报道；公开数据也可能出现在基座预训练语料中。

## 训练与校准

| 项目 | 实际结果 |
| --- | --- |
| 基座 / 起点 | Qwen3.5-2B-Base / Strands Decider v19 |
| 更新范围 | 原有 LoRA 和 pointer head；基座冻结 |
| 可训练参数 | LoRA 16,819,200；head 1,053,184 |
| 训练配置 | 1 epoch；batch 2；梯度累积 8；100 次更新 |
| 学习率 | LoRA `2e-5`；head `1e-4` |
| 最长训练输入 | 443 token，无截断 |
| 原版验证准确率 | 90.5% |
| 第 50 / 100 步验证准确率 | 92.5% / 91.5% |
| 最佳检查点 | 第 50 步，仅依据 validation 选择 |
| 概率校准 | 独立 200 条 calibration 数据，温度 1.4125376 |
| 训练循环耗时 | 321.78 秒 |
| 完整入口任务耗时 | 470.64 秒，约 7 分 51 秒 |
| PyTorch 峰值 allocated 显存 | 4.10 GiB，不含所有驱动及 CUDA 占用 |
| 运行环境 | torch 2.7.1+cu126；transformers 5.17.0；peft 0.21.0；tokenizers 0.23.2 |

实际执行的入口命令：

```bash
/usr/local/bin/python finetune.py \
  --data dataset/choices \
  --config config.json \
  --output /var/cache/decider/ag-news-unified-20261017-000530/result/pipeline
```

使用已有容器环境，`runtime.auto_setup=false`；完整配置见 [config.json](config.json)。
这次没有重新验证自动安装 CUDA 依赖的路径。
阶段依次为数据导入与审计、预检、训练、原版评估、微调版校准与评估、独立进程重新加载。
本例为新闻分类，因此关闭俄罗斯方块对局评估。

## 验证及服务恢复

- LoRA 与 head 在训练中的梯度有限且非零，训练前后权重散列不同。
- 最佳检查点重新加载后的 LoRA 与 head 散列也不同于初始模型。
- 独立进程重新加载最终校准模型，16 条测试探针的预测与 NLL 核对通过。
- 下载后的 57 个文件全部通过 SHA-256 核对；入口自身的 47 个产物文件再次校验通过。
- 远程实际使用的四份规范化数据，与本地原始导入结果逐字节相同。
- 已归档 gzip 数据经统一入口重新导入，也与原始导入结果逐字节相同。
- 现有 85 项自动化测试及 Ruff 检查通过。

复用了指定实例上的 GPU，临时暂停演示服务及其自动控制器。
任务结束后恢复原 desired count、任务定义、Lambda 并发配置和调度规则；
确认服务为 1 个运行任务、0 个待启动任务，模型应用启动完成。
没有启动或停止 EC2，没有替换线上模型，也没有使用 vLLM 那台实例。
恢复证据见 [service-restoration.json](service-restoration.json)。

## 产物与复现

- [指标与质量门槛](summary.json)
- [训练记录](training.json)与[逐步日志](steps.jsonl)
- [原版评估](baseline/report.json)与[微调版评估](adapted/report.json)
- [重新加载验证](reload-verification.json)与[验证汇总](verification-summary.json)
- [阶段状态](pipeline.json)与[任务退出记录](job-status.json)
- [仓库报告校验清单](FILES.sha256.json)

本机完整运行目录：

```text
artifacts/ag-news-gpu-job-reuse/download/pipeline/
```

最终模型位于该目录的 `eval-v1/adapted/calibrated/`。
产物包含 LoRA、决策头、tokenizer 和配置；使用时仍需原 Qwen3.5-2B-Base 基座。
检查点中的基座路径为 `/opt/models/base`；换主机时按
[检查点加载说明](../../finetuning/README.md#使用产出的检查点)在副本中调整路径。

私有完整备份：

```text
s3://strands-decider-tetris-training-632930644527/ag-news-unified-20261017-000530/results/
```

使用归档数据重新训练：

```bash
uv sync --locked
cp examples/ag-news-finetune.json my-ag-news.json
# 填写 model.checkpoint 和 model.base_model 的实际路径。
uv run python finetune.py \
  --data reports/2026-10-17-ag-news/data \
  --config my-ag-news.json \
  --output artifacts/runs/ag-news-repeat
```

完整示例说明见 [AG News GPU 微调](../../docs/AG_NEWS_EXAMPLE.zh.md)。
