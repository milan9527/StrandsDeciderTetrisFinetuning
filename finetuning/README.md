# Decider 的俄罗斯方块 SFT

新任务推荐使用根目录的 [`finetune.py`](../docs/FINETUNE.md)，统一完成数据导入、
训练与验证；本页保留首轮 Tetris 实验和底层命令说明。
本轮实验的原理、真实样本、执行顺序及流程图片见
[微调全过程](../docs/TETRIS_FINETUNING_WALKTHROUGH.zh.md)。

第一阶段训练单步候选比较：冻结 Qwen3.5-2B 基座，继续训练 v19 已有的 LoRA
和 pointer 决策头。数据标签来自游戏引擎的严格指标支配关系，没有教师模型调用。
若后续加入教师标注，按本次要求仅使用 GPT，并单独记录模型配置、请求和标签来源。

已在指定 EC2 上完成首轮训练，详见 [实测结果](RESULTS.md)。

此训练目标是降低指标劣选、改善顺序稳定性。它不保证长期最优，也不能仅凭选择题
准确率宣布游戏能力提高。完整候选中各有取舍的落点需要后续搜索或 GPT 教师数据。
当前采样出的多选题有一个严格支配其他选项的答案，难度分布与完整实战候选不同。

## 数据与审计

```bash
uv run python -m finetuning.data --output artifacts/sft/data-v1
uv run python -m finetuning.data --output artifacts/sft/data-v1 --audit-only
```

默认生成 144 个种子的对局，每局最多 40 块。访问棋盘的策略混合随机落点、
随机非劣落点和带少量探索的指标排序；这些策略只控制访问的状态，不提供训练标签。
每个棋盘生成二选一与 3～12 选一，分别采用项目现有的数字和自然语言输入。
候选 ID 随机命名，存储时随机排列，训练 collator 每次再次打乱并同步调整标签。

2026-10-16 本地 `data-v1`：

| 分区 | 种子范围 | 问题数 |
| --- | --- | ---: |
| train | 10000～10095 | 12590 |
| validation | 20000～20015 | 2120 |
| calibration | 30000～30015 | 2090 |
| test | 40000～40015 | 2068 |

每行 JSONL 遵循上游 `Example` 格式，`_meta` 额外保存种子、步数、原始落点 ID、
状态散列与标签依据。比较的六项指标与 `backend.policy.quality_vector` 完全相同。
指标相同的候选不强制产生唯一正确答案。所有变体留在同一分区；状态去重忽略
种子、方块颜色和累计消行数，防止同一棋盘因为元数据不同而跨分区泄漏。

`manifest.json` 保存种子与文件 SHA-256。`audit.json` 验证每条样本的可达性、
完整候选后果、严格支配标签和分区隔离。审计重新从棋盘计算，不能只通过修改标签
和重算文件 SHA-256 绕过。输出目录必须不存在或为空，不覆盖已有数据。

## 独立训练环境

模型训练依赖不加入 Web 应用环境。上游代码固定为：
`6039cf8b64c6074e75804000c07ac3d4cce795f7`。

```bash
git clone https://github.com/strands-labs/strands-decider.git artifacts/sft/upstream
git -C artifacts/sft/upstream checkout 6039cf8b64c6074e75804000c07ac3d4cce795f7
uv venv artifacts/sft/venv --python 3.11
uv pip install --python artifacts/sft/venv/bin/python -r finetuning/requirements.txt

# Linux CUDA 主机另外安装 Gated DeltaNet 的 GPU 内核。
uv pip install --python artifacts/sft/venv/bin/python flash-linear-attention
```

CUDA 主机应使用兼容 CUDA 的 PyTorch wheel，并在全量训练前通过真实权重
`--smoke` 检查。单卡 24GB 是计划的起点，最终显存及耗时以实测为准。

不要使用上游 `training.train` 的 `init_from` 当作原模型续训：该版本会换掉
pointer head 并冻结主干。这里使用独立入口加载原权重，只开启原 LoRA A/B
与 pointer head 的梯度，检查启用前后权重散列不变、训练梯度有限且非零，
并在训练结束确认两部分权重都更新。旧温度在训练时重置，训练后另行校准。

## 运行

从项目根目录运行，路径指向本机真实文件：

```bash
export SFT_PYTHON="$PWD/artifacts/sft/venv/bin/python"
export SFT_UPSTREAM="$PWD/artifacts/sft/upstream"
export SFT_CHECKPOINT=/path/to/strands-decider-v19
export SFT_BASE_MODEL=/path/to/Qwen3.5-2B-Base
export SFT_DATA="$PWD/artifacts/sft/data-v1"
export SFT_RUN="$PWD/artifacts/sft/run-v1"
export SFT_EVAL="$PWD/artifacts/sft/eval-v1"
bash finetuning/run_gpu.sh
```

首轮：1 epoch、micro-batch 2、累积 8、有效 batch 16、LoRA 学习率 2e-5、
head 学习率 1e-4、梯度裁剪 1.0。共 787 次优化器更新。
每 200 步及结束时计算验证集准确率，仅依据验证集保存 `best/`；`last/` 保存
最终权重。不保证新检查点优于原版，也不自动替换在线模型。

原始模型只被复制；在副本配置中调整本地基座路径。模型权重、原始应用和线上部署
无需修改。输出保存 `source/`、`best/`、`last/`、每步损失、梯度范数、
输入数据清单、运行配置及权重散列。

CPU 只允许有限的流程检查：

```bash
artifacts/sft/venv/bin/python -m finetuning.train \
  --upstream artifacts/sft/upstream --checkpoint /path/to/strands-decider-v19 \
  --base-model /path/to/Qwen3.5-2B-Base --data artifacts/sft/data-v1 \
  --output artifacts/sft/smoke --device cpu --smoke --batch-size 1 --accumulation 1
```

`--smoke` 使用真实模型和极少量真实引擎标签，但只有一次更新，记录
`run_kind=smoke_only`，不能用它报告微调收益。

## 独立评估

`finetuning.evaluate` 在固定检查点上运行：

- test 分区的选择准确率，按两种输入分别统计。
- 同一题候选反序并重命名后的准确率及实际落点一致率。
- 原始与微调模型在五个新种子 50000～50004、两种输入上的完整对局，
  每局最多 50 块，统计落块、消行、得分、劣选和决策延迟。
- 对局经过原应用的候选分组和执行流程，保存原始响应并重放校验。

这里的延迟为训练主机上的本地模型调用耗时，不能直接与此前含网络和排队的
在线服务耗时比较。回执中的模型名称是本地评估配置名，检查点路径与数据散列
保存在报告中。

`--calibrate` 只使用 calibration 分区，通过 NLL 选择 choice 温度，保存为独立
`calibrated/` 检查点，不修改输入模型。此校准仅针对指标比较数据，
不代表实战胜率或完整候选下的置信度已得到校准。

若指标判断改善但整局成绩没有改善，应如实保留结论；下一步再加入有取舍的搜索
或 GPT 标签，不能用测试集反复调参后继续把它称作独立测试。

## 验证

```bash
uv run pytest -q
uv run ruff check backend tests scripts finetuning
```

新增测试覆盖错误标签、候选事实被篡改、跨分区棋盘泄漏、排列后的标签映射、
等指标候选不误标，以及数据可复现。

## 在指定的现有 EC2 上执行

本轮指定复用 `us-east-1` 的 `i-0daca64f56056aa1c`（g6.xlarge / L4）。
`finetuning.aws_existing` 使用该实例已缓存的 CUDA 镜像及 `/opt/models` 权重，
不启动或终止 EC2。训练期间临时暂停原 ECS 模型服务和它的自动控制器，
任务结束、失败或超时后恢复原 desired count、Lambda 并发数和调度规则。

```bash
uv run python -m finetuning.aws_existing prepare
uv run python -m finetuning.aws_existing run
uv run python -m finetuning.aws_existing download
uv run python -m finetuning.summarize artifacts/sft/aws-existing/job-v1/results
# 若本地监控进程被外部中断，可依据保存的原配置恢复：
uv run python -m finetuning.aws_existing restore
```

准备阶段在隔离目录提交本项目源代码快照，上游用固定 commit 的 `git archive` 导出，
并记录每个文件的 SHA-256。容器无需安装 git；导出内容在导入前逐文件校验。
代码、数据和检查点保存在启用公开访问阻止与服务端加密的独立 S3 bucket。
临时任务角色只允许访问本次任务前缀。

GPU 任务先做单步真实训练检查，再做完整训练与独立评估；每 45 秒备份日志和
已保存的检查点。远程工作进程限制为 110 分钟，本地监控限制为 120 分钟。
备份和最终 SHA-256 清单用于下载校验；主机原权重不被覆盖。

复用镜像中的 tokenizers 为 0.23.2，本地预检为 0.23.3；torch、transformers、
peft 主版本配置一致。实际环境写入训练报告，原始与微调模型在同一 GPU 镜像内评估。

## 使用产出的检查点

下载后的 `results/eval-v1/adapted/calibrated/` 包含校准后的 LoRA、决策头和
tokenizer；`results/run-v1/best/` 是校准前的最佳验证集检查点。它们需要原有
Qwen3.5-2B-Base 基座及上述固定版本的上游运行代码，不包含完整基座权重。
原始配置中的基座路径为 GPU 镜像内的 `/opt/models/base`。

在其他主机加载时，先将基座路径写入一个新的检查点副本：

```bash
artifacts/sft/venv/bin/python - <<'PY'
from finetuning.runtime import stage_source, use_upstream
use_upstream("artifacts/sft/upstream")
stage_source(
    "artifacts/sft/aws-existing/job-v1/results/eval-v1/adapted/calibrated",
    "artifacts/sft/decider-tetris-local",
    base_model="/path/to/Qwen3.5-2B-Base",
)
PY
```

模型服务应使用返回名称 `strands-decider-tetris-sft`，以满足本项目的模型身份校验。
切换线上服务属于后续部署操作；本轮执行仅保存检查点、评估结果并恢复原服务。
