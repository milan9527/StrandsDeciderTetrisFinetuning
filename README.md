# Strands Decider Tetris Finetuning

独立的俄罗斯方块微调项目：从游戏引擎生成可审计的比较标签，继续训练
Strands Decider v19 的 LoRA 和 pointer 决策头，并用独立测试题及完整对局评估。
本轮使用 SFT，没有教师模型调用，也没有运行 PPO 或 GRPO。

## 已完成的首轮实验

2026-10-16，在现有 NVIDIA L4 实例训练 12,590 条样本、1 epoch、787 步。

| 指标 | 原始 v19 | SFT |
| --- | ---: | ---: |
| 独立测试题准确率 | 60.35% | 99.95% |
| 候选换序后实际落点一致率 | 35.44% | 99.85% |
| 数字输入：平均消行 | 1.8 | 12.2 |
| 自然语言输入：平均消行 | 3.0 | 14.2 |

游戏评估每种输入各五个新种子，每局上限 50 块；这些结果不能推断无限存活能力。
见 [完整报告](finetuning/RESULTS.md) 和 [原始评估记录](reports/2026-10-16)。

## 快速开始

需要 Python 3.11+、Git 和 [uv](https://docs.astral.sh/uv/)。
数据生成、审计和回放检查不需要 GPU，也不需要模型 API 密钥。

```bash
uv sync --extra dev
uv run python -m finetuning.data --output artifacts/sft/data-v1
uv run python -m finetuning.data --output artifacts/sft/data-v1 --audit-only
uv run python -m scripts.verify_reports
uv run pytest -q
uv run ruff check backend finetuning scripts tests
```

训练使用独立环境，不安装到上面的轻量工具环境中：

```bash
git clone https://github.com/strands-labs/strands-decider.git artifacts/sft/upstream
git -C artifacts/sft/upstream checkout 6039cf8b64c6074e75804000c07ac3d4cce795f7
uv venv artifacts/sft/venv --python 3.11
uv pip install --python artifacts/sft/venv/bin/python -r finetuning/requirements.txt
# CUDA 主机需要兼容的 PyTorch wheel，以及训练内核。
uv pip install --python artifacts/sft/venv/bin/python flash-linear-attention

export SFT_PYTHON="$PWD/artifacts/sft/venv/bin/python"
export SFT_UPSTREAM="$PWD/artifacts/sft/upstream"
export SFT_CHECKPOINT=/path/to/strands-decider-v19
export SFT_BASE_MODEL=/path/to/Qwen3.5-2B-Base
bash finetuning/run_gpu.sh
```

详细的数据格式、标签依据、GPU 要求、参数、AWS 执行及检查点加载方法见
[运行说明](finetuning/README.md)。AWS 入口保留了首轮实验的资源默认值，
只适用于已有对应 ECS 服务的环境；普通训练使用上面的 `run_gpu.sh`。

## 项目内容

| 路径 | 内容 |
| --- | --- |
| `finetuning/` | 数据、训练、校准、评估、汇总和可选 ECS 执行入口 |
| `backend/` | 固定版本的棋盘引擎、输入格式、候选选择及对局逻辑 |
| `scripts/` | 原始回放校验逻辑、报告验证及已有服务的有限对局工具 |
| `tests/` | 数据审计、棋盘规则、模型响应及候选选择测试 |
| `reports/2026-10-16/` | 首轮统计、训练记录、数据清单和 20 局压缩回执 |
| `SOURCE.json` | 拆分来源、训练时源代码快照和原始复制文件散列 |

此仓库可独立生成数据和运行训练，不需要相邻的 Tetris 应用目录。
基座、原始 v19 权重和微调权重保存在仓库外；检查点备份位置见实测报告。
若后续需要教师标注，使用 GPT，并记录标签来源。

引擎及评估逻辑来自
[StrandsDeciderTetris](https://github.com/milan9527/StrandsDeciderTetris)。
上游模型运行代码按固定 commit 单独获取，其许可证和模型权重使用条件以原项目为准。
