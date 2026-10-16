# 一条命令完成 Decider 微调与验证

想先理解本次 Tetris 微调如何完成，可阅读带流程图片的
[微调全过程](TETRIS_FINETUNING_WALKTHROUGH.zh.md)；本页说明如何运行统一入口。

入口是仓库根目录的 `finetune.py`。提供带标签的数据位置和一个 JSON 配置文件，即可依次完成：
数据快照与审计 → 环境和模型预检 → SFT → 原版/微调版独立测试 → 温度校准 → 保存后重新加载验证 → 质量判定与报告。

当前支持 **Decider 的 choice 决策任务**，沿用本项目已验证的方法：冻结基座，继续训练已有的 LoRA 和 pointer head。
不支持直接导入聊天 `messages`、无标签文本、`noul` 或 `score` 任务；不是 PPO/GRPO。
标签由数据提供者提供，可以来自业务规则、人工或经审核的 GPT 标注。本入口不调用教师模型。

## 1. 准备配置并运行

需要 Python 3.11+、Git、[uv](https://docs.astral.sh/uv/) 和模型权重。
完整训练使用支持 bf16 的 NVIDIA GPU；建议 24GB 级显卡，预检要求至少 18 GiB 空闲显存。
长输入或更大的 batch 仍可能超出显存，需要降低 batch 或缩短输入。

```bash
uv sync --locked
cp examples/finetune.json my-finetune.json
# 编辑 my-finetune.json，填写原始 Decider 检查点及对应基座的本地目录。

uv run python finetune.py \
  --data /path/to/choices.jsonl \
  --config my-finetune.json \
  --output artifacts/runs/my-decider-v1
```

数据也可以放在 S3：

```bash
uv run python finetune.py \
  --data s3://your-bucket/datasets/choices.jsonl \
  --config my-finetune.json \
  --output artifacts/runs/my-decider-v1
```

S3 使用正常的 AWS 凭证链，例如 EC2 IAM role 或 `AWS_PROFILE`。单文件需要 `s3:GetObject`；
目录前缀还需要 `s3:ListBucket`。下载到本次运行目录后固定使用该快照。

首次运行自动获取固定 commit 的上游代码，并在 `artifacts/training-env-cuda/` 建立独立训练环境。
自动 CUDA 安装面向 Linux x86_64，使用 PyTorch 2.7.1 的 CUDA 12.6 wheel；
主机需要兼容的 NVIDIA 驱动。CPU 检查使用独立的 `training-env-cpu/`。
核心包固定版本，实际版本写入预检和训练报告。

已有环境可以这样配置，脚本不会修改指定的 Python 环境：

```json
{
  "model": {
    "checkpoint": "/models/decider",
    "base_model": "/models/base"
  },
  "runtime": {
    "python": "/opt/training-env/bin/python",
    "upstream": "/opt/strands-decider",
    "device": "cuda",
    "auto_setup": false
  }
}
```

`checkpoint` 必须包含已有 LoRA、pointer head、配置及 tokenizer；可以使用本地目录或 Hugging Face 模型仓库 ID。
`base_model` 是对应基座的本地目录。若省略，使用原检查点配置中的基座位置，可能需要网络下载。
推荐使用本地基座，便于校验和复现。不是所有 Decider 版本都兼容，当前固定的上游 commit 是
`6039cf8b64c6074e75804000c07ac3d4cce795f7`，本项目实测为 v19 / Qwen3.5-2B-Base。

## 2. 数据格式

最简单的是每行一个样本的 JSONL：

```json
{"state":"两个承运商都能按时到达。","instructions":"选择费用更低的承运商。","options":[["A","费用 12 元"],["B","费用 20 元"]],"answer":"A","group_id":"order-001"}
```

| 字段 | 说明 |
| --- | --- |
| `state` | 非空文本、JSON 对象或数组；模型做决策时可见的事实 |
| `instructions` | 非空决策要求 |
| `options` | 2～24 个 `[ID, 描述]`；也可写成 `{"A":"描述A","B":"描述B"}` |
| `answer` 或 `label` | 正确选项 ID，或从 **0** 开始的选项下标；二者同时提供时必须一致 |
| `group_id` | 可选；同一个用户、会话、订单、游戏等相关样本的共同分组标识 |
| `task` | 可选任务名称，用于分任务统计，默认 `custom` |
| `instruction_variants` | 可选的同义指令列表，训练时随机采样；正确答案必须保持不变 |
| `kind`、`weight` | 可省略；若提供，必须分别是 `"choice"`、`1` |

选项 ID 和渲染后的描述各自不能重复。不要在 `state` 中泄露正确答案。
附加字段不会自动进入模型输入；需要让模型看到的信息应放入上述输入字段。

也支持 `.jsonl.gz`、`.csv`、`.csv.gz`。CSV 列名与上表一致，
`options`、`instruction_variants` 单元格填写 JSON，`label` 填整数，`answer` 填选项 ID。
复杂单元格需符合标准 CSV 引号转义规则。

仓库的 [choices.jsonl](../examples/choices.jsonl) 有 20 条演示数据，仅用于检查流程，不适合评估业务效果。
真实标签的语义正确性由数据提供者负责：脚本能发现格式错误、重复题、同题冲突标签、分区泄漏，
不能自动判断任意业务答案是否正确。

## 3. 分区与泄漏检查

单文件默认按约 **70% / 10% / 10% / 10%** 划分 train、validation、calibration、test。
相同状态或相同 `group_id` 的样本通过连通分组留在同一分区，因此实际比例可能有偏差。
至少需要四个独立分组，且每个分区必须非空。少量样本的测试结果波动很大，应使用足够多的独立业务案例。

相同状态检查会规范化 JSON 键顺序和外围空白；它不能识别语义近似样本。
例如同一订单只改了流水号、同一棋局的不同描述，必须由数据提供者赋予共同的 `group_id`。
这也意味着无法仅通过格式审计证明数据完全没有业务层面的泄漏。

已有分区可以提供本地目录或 S3 前缀：

```text
dataset/
  train.jsonl
  validation.jsonl
  calibration.jsonl
  test.jsonl
```

各分区也可分别用 CSV/gzip；每个分区只提供一个文件。已有分区保留原分配，但仍检查跨分区的状态和分组重叠。
S3 示例：`--data s3://your-bucket/dataset/`。前缀下只读取上述四类分区文件和可选 `manifest.json`。

优化器只使用 train；validation 用于挑选 best；calibration 只用于温度校准；test 用于最终报告和门槛判定。
根据测试结果反复调参后，应另外留出新的测试集。

本项目原有的引擎数据目录（包含原始 `manifest.json`）会自动进入 Tetris 审计模式，
继续逐条重新计算标签和候选结果，而非降级为仅检查格式。此模式默认还运行原版、新版各 10 局游戏，
核验完整回放，并确保游戏种子与所有数据分区隔离。

```bash
uv run python -m finetuning.data --output artifacts/sft/data-v1
uv run python finetune.py --data artifacts/sft/data-v1 \
  --config my-finetune.json --output artifacts/runs/tetris-v2
```

## 4. 常用设置

未填写的项采用默认值；未知字段会报错，以免拼写错误悄悄失效。
配置文件内的相对路径按该文件所在目录解析，命令行路径按当前目录解析。
命令行的 `--data`、`--python`、`--device` 覆盖配置文件中的对应项。

| 设置 | 默认值 | 用途 |
| --- | --- | --- |
| `data.location` | 无 | 可代替命令行 `--data` |
| `data.mode` | `"auto"` | 自动识别；也可指定 `"choice"` 或 `"tetris"` |
| `data.fractions` | `[0.7,0.1,0.1,0.1]` | 单文件分区比例 |
| `data.seed` | `17` | 分区随机种子 |
| `runtime.auto_setup` | `true` | 获取上游代码并安装独立训练环境 |
| `runtime.device` | `"cuda"` | CPU 仅支持 `--smoke` 检查 |
| `runtime.timeout_minutes` | `180` | 每个子命令的最长运行时间，不是整个任务的总时限 |
| `training.epochs` | `1` | 训练轮数 |
| `training.batch_size` | `2` | 每次前向计算的样本数 |
| `training.accumulation` | `8` | 梯度累积次数；默认有效 batch 为 16 |
| `training.lr` / `training.head_lr` | `2e-5` / `1e-4` | LoRA / pointer head 学习率 |
| `training.eval_every` | `200` | 每多少更新步做 validation；最后一步也评估 |
| `training.seed` / `training.cpu_threads` | `17` / `4` | 随机种子 / CPU 线程数 |
| `evaluation.batch_size` | `2` | 评估 batch |
| `evaluation.games` | `"auto"` | Tetris 默认运行游戏；通用数据默认不运行，可显式 `false` |
| `evaluation.seeds` / `evaluation.pieces` | `50000`～`50004` / `50` | Tetris 对局种子 / 每局方块上限 |
| `quality.min_accuracy` | `0` | 微调后测试准确率的最低值 |
| `quality.max_accuracy_drop` | `0` | 相对原版最多允许下降多少；`0.02` 表示 2 个百分点 |
| `quality.min_order_consistency` | `0` | 候选反序并改名后，仍选同一个实际选项的最低比例 |

以上比例都在 0～1 之间。[示例配置](../examples/finetune.json) 主动设置了 80% 的准确率和顺序一致性门槛，
实际业务应在训练前设定门槛。默认只要求准确率不低于原版，并不保证业务可用性。
游戏胜负和消行数会报告，但目前不参与数值门槛；回放或执行一致性失败则整个流程失败。

## 5. 只检查数据、检查流程和恢复运行

只检查数据，不加载模型，也不要求填写模型位置：

```bash
uv run python finetune.py --data examples/choices.jsonl \
  --output artifacts/runs/data-check --validate-only
```

用真实权重做一次 CPU 更新并跑完小规模验证：

```bash
uv run python finetune.py --data examples/choices.jsonl \
  --config my-finetune.json --device cpu --smoke \
  --output artifacts/runs/cpu-smoke
```

Smoke 会用 1 条训练数据、最多 2 条验证/校准/测试数据，仍实际更新 LoRA 和决策头、保存模型、重新加载并推理。
结果标为 `smoke_only`、`quality_passed: null`；即使退出码为 0，也只代表流程跑通，不能证明效果提升。
CPU 模式需要能容纳基座、梯度及优化器的内存；本项目在 32GB 内存主机上实测。

失败后用完全相同的参数追加 `--resume`：

```bash
uv run python finetune.py --data /path/to/choices.jsonl \
  --config my-finetune.json --output artifacts/runs/my-decider-v1 --resume
```

恢复粒度是**阶段**。已完成阶段先核验 SHA-256，再跳过；失败阶段的产物移到 `failed/`，然后重新执行该阶段。
如果训练中断，训练阶段从原模型重新开始，**不会续接优化器步数**。
配置、代码、已保存数据或检查点改变时会拒绝恢复；已完成数据快照后，原始数据位置发生变化不会被重新导入。
本地原模型和基座文件也会核验；远程下载的基座依赖原配置中的模型修订信息。
调整设置或替换数据时应选择新的输出目录。

同一输出目录不能同时运行两个进程。日志位于 `logs/`，运行中会定期输出当前阶段和耗时。
超时、子进程报错、非有限梯度或显存不足都会停止流程并保存失败状态。

## 6. 输出与验收

主要产物：

```text
run/
  config.resolved.json        # 实际配置
  pipeline.json              # 阶段状态、源代码散列
  input/                     # 输入快照
  data/                      # 四个分区、manifest 和 audit
  preflight/report.json      # 环境、长度、原模型/基座散列
  run-v1/source/             # 原版 LoRA/head 副本
  run-v1/best/                # validation 最佳检查点
  run-v1/last/                # 最后一步检查点
  run-v1/run.json             # 参数、梯度、权重变化和验证曲线记录
  eval-v1/baseline/           # 原版测试报告
  eval-v1/adapted/calibrated/ # 最终校准后的检查点
  verification/report.json   # 另一个进程重新加载后的推理核对
  REPORT.md                  # 可阅读的对比结果
  report.json                # 质量门槛、对比统计及适用范围
  SHA256.json                # 产物校验清单
  logs/
```

完整流程检查 LoRA/head 确实被更新，拒绝输入截断；原版与新版使用同一 test；
评估原始顺序、反序并改名后的准确率和实际选项一致率；单独使用 calibration 拟合温度。
校准后的 checkpoint 会在另一个进程重新加载，对最多 16 条测试样本核对预测及 NLL。
最后核验产物文件散列。准确率针对所提供的业务数据分布，不能推断所有任务的泛化能力。

| 退出码/状态 | 含义 |
| --- | --- |
| `0` / `complete` | 完整训练、验证及所有配置的质量门槛通过 |
| `0` / `smoke_verified` | 仅流程检查通过 |
| `0` / `data_validated` | 仅数据审计通过 |
| `2` / `quality_failed` | 模型与报告已生成，但效果未达到门槛 |
| 其他非零 / `failed` | 执行失败，查看 `pipeline.json` 和日志；配置错误可能在创建目录前退出 |

事后只核对文件是否改变：

```bash
uv run python finetune.py --verify-artifacts artifacts/runs/my-decider-v1
```

此命令不重新运行模型；模型重新加载验证已包含在完整训练流程中。
`--validate-only` 不生成这份最终 `SHA256.json`。
模型产物包含 LoRA、head、tokenizer 和配置，**不包含基座**；迁移到其他主机时需要相同基座并调整副本的基座路径，
见 [检查点加载方法](../finetuning/README.md#使用产出的检查点)。

该入口在执行命令的主机上运行，不创建 EC2、不停止现有服务、不自动部署线上模型。
如在已有 EC2 上运行，先准备好数据、权重和足够空闲的 GPU 资源。
