# 用独立 AG News 数据运行完整 GPU 微调

这个例子将真实新闻分到 World、Sports、Business、Science and Technology 四个栏目。
输入是新闻标题和摘要，输出是四个候选栏目中的一个。标签来自公开 AG News 数据集，
没有生成教师标签。它使用根目录 `finetune.py` 完成训练、校准、原版对照和保存后重新加载验证。

2026-10-17 已在原 Tetris 实验使用的 NVIDIA L4 上完成全流程，退出码为 0。
独立测试准确率从 90.5% 到 91.0%，换序选择一致率从 98.5% 到 96.5%；
属于小规模示例，不能据此认定全面提升。
实际数据、训练记录、验证及服务恢复证据见 [完整实验报告](../reports/2026-10-17-ag-news/README.md)。

## 数据

在项目根目录运行：

```bash
uv sync --locked
uv run python -m scripts.prepare_ag_news --output artifacts/ag-news-example
```

下载来源固定到公开镜像的 commit，生成器保存原始 CSV、来源 URL、SHA-256、
原始行号和标签，以及独立的四分区 JSONL。数据与 Tetris 和 20 条演示数据完全分开。

| 分区 | 数量 | 每类数量 | 来源 |
| --- | ---: | ---: | --- |
| train | 1,600 | 400 | 官方 train |
| validation | 200 | 50 | 官方 train |
| calibration | 200 | 50 | 官方 train |
| test | 200 | 50 | 官方 test |

本次实际使用的四份数据也保存在 [`reports/2026-10-17-ag-news/data/`](../reports/2026-10-17-ag-news/data)，
采用 JSONL gzip 格式，附带来源及逐条标签核对记录。
下面训练命令的 `--data` 可以直接改成 `reports/2026-10-17-ag-news/data`，无需重新下载数据。

先对完整官方 test 去重，再从官方 train 中排除相同标题或摘要。
比较时规范化 Unicode、大小写、标点和空白。训练、验证、校准三份数据互不重叠。
每题的栏目顺序和候选 ID 随机化；训练时继续随机排列候选。

这能排除相同标题/摘要，不能保证不存在同一事件的近似报道，也不能排除基座预训练见过公开数据。
结果只能代表这个小规模、类别平衡的基准子集。

## 运行

需要 Python 3.11+、uv、Decider v19 和 Qwen3.5-2B-Base 权重，以及支持 bf16 的 NVIDIA GPU。
预检要求至少 18 GiB 空闲显存。

```bash
cp examples/ag-news-finetune.json my-ag-news.json
# 修改 model.checkpoint 和 model.base_model，填写真实模型路径。

uv run python finetune.py \
  --data artifacts/ag-news-example/choices \
  --config my-ag-news.json \
  --output artifacts/runs/ag-news-v1
```

默认自动创建独立 CUDA 环境。已有兼容训练环境时，可以在配置中填写
`runtime.python`、`runtime.upstream`，并设置 `runtime.auto_setup=false`。
这条命令在当前主机运行；使用远程 GPU 时需先把代码、数据和配置放到该主机。

固定训练设置：1 epoch、小批次 2、梯度累积 8、有效批次 16，共 100 次优化器更新；
LoRA 学习率 `2e-5`，决策头学习率 `1e-4`；每 50 步及训练结束计算验证集指标。
冻结基座，继续更新已有 LoRA 和 pointer head。

验收门槛在训练前固定：测试准确率至少 80%，不低于原版；
候选反序并改名后实际选择的一致率至少 85%。测试结果不用于选择检查点或调整温度。
本例是新闻分类任务，因此关闭 Tetris 对局评估。

## 输出与验证

输出目录内：

- `REPORT.md` 和 `report.json`：原版与微调版指标及质量判定。
- `run-v1/run.json`：实际 GPU、依赖版本、梯度、权重变化和训练记录。
- `eval-v1/adapted/calibrated/`：最终校准模型，加载时仍需原基座。
- `verification/report.json`：独立进程重新加载后的预测与损失核对。
- `input/`、`data/`：输入快照和审计后的四份数据。
- `logs/`：各阶段日志。

流程正常且达到门槛返回 `0`；完成但未达门槛返回 `2`，仍保留模型和报告。
可用以下命令事后核对产物完整性：

```bash
uv run python finetune.py --verify-artifacts artifacts/runs/ag-news-v1
```

数据出处：[AG News 原始语料页面](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)、
[分类基准论文](https://arxiv.org/abs/1509.01626)、
[固定版本 CSV 镜像](https://github.com/mhjabreel/CharCnn_Keras/tree/03836ce08fce38daac53e9b4255ef16e0b9f6303/data/ag_news_csv)。
