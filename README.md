# Strands Decider Tetris Finetuning

独立的俄罗斯方块微调项目：从游戏引擎生成可审计的比较标签，继续训练
Strands Decider v19 的 LoRA 和 pointer 决策头，并用独立测试题及完整对局评估。
本轮使用 SFT，没有教师模型调用，也没有运行 PPO 或 GRPO。

完整图解见 [俄罗斯方块微调全过程](docs/TETRIS_FINETUNING_WALKTHROUGH.zh.md)，
涵盖真实训练样本、四份数据的用途、LoRA 更新、GPU 执行、评估和后续上线。

图片：[整体流程](docs/diagrams/tetris-sft-workflow.png) ·
[LoRA 更新示意](docs/diagrams/lora-training-step.png)。

## 一键微调自己的决策数据

现在也支持通用的带标签 choice 数据。编辑示例中的模型路径，即可用一条命令完成
数据审计、SFT、原版对照、校准、检查点重新加载验证和质量门槛判定：

```bash
uv sync --locked
cp examples/finetune.json my-finetune.json
# 在 my-finetune.json 中填写原始 Decider 和基座目录。
uv run python finetune.py --data /path/to/choices.jsonl \
  --config my-finetune.json --output artifacts/runs/my-decider-v1
```

支持本地/S3、JSONL/CSV/gzip、自动分区或已有四分区。
完整训练需要 CUDA；`--device cpu --smoke` 可做真实权重流程检查。
结果保存在 `REPORT.md` 和 `report.json`；未达到质量门槛返回退出码 2。

见 [完整使用说明](docs/FINETUNE.md)、[配置示例](examples/finetune.json)
和 [数据示例](examples/choices.jsonl)。本入口不自动部署或调整线上服务。
真实 CPU 流程检查及测试范围见 [验证记录](docs/VERIFICATION.md)。

独立数据示例：[AG News 新闻分类 GPU 微调](docs/AG_NEWS_EXAMPLE.zh.md)，包含可复现的数据准备脚本、
四份独立分区和专用配置。已在 L4 上通过统一入口完成正式训练、校准、独立评估和重新加载验证；
实际数据及结果见 [2026-10-17 实验报告](reports/2026-10-17-ag-news/README.md)。

寻找更需要微调的任务：参考 Jev 场景，已在原版 v19 上实测六类任务、910 道独立筛选题。
完成核验为 38.75%，检索可回答性为 56.88%，中文证据核对为 60.00%；
六套独立训练数据、实际错例及复现入口见
[场景筛选报告](reports/2026-10-17-scenario-screening/README.md)。本轮筛选未更新模型权重。

随后已分别微调完成核验和检索可回答性：独立测试准确率为 43% → 92.5%、54% → 78%。
前者的概率重新加载复核未通过，后者未达到预设 80% 准确率门槛；模型作为实验产物保留。
数据、逐题对照、检查点位置及失败记录见
[两个场景的微调报告](reports/2026-10-17-scenario-finetuning/README.md)。

浏览器动作选择也已完成微调：独立合成测试准确率 53% → 100%，换序一致率 73.5% → 100%，
完整流程、重新加载验证和质量门槛通过。四类动作成绩、实际修正案例和模型位置见
[浏览器微调报告](reports/2026-10-17-browser-finetuning/README.md)；本轮没有执行真实浏览器操作。

## HTTP 微调服务

可通过 HTTP 提交同一个微调流程，后台 GPU worker 按队列串行执行。
支持 bearer token、幂等提交、任务查询、增量日志、取消和产物下载；
任务记录持久保存在 SQLite，API 重启不停止独立 worker 中的训练。

提供单机 Docker Compose 部署，模型和数据位置由使用者在自己的 AWS 环境配置。
当前适用于一个受信任团队共享一台 GPU。见 [HTTP 接口与部署说明](docs/HTTP_API.zh.md)
和 [服务配置示例](examples/http-service.json)。

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
| `finetune.py`、`docs/`、`examples/` | 统一入口、中文说明及配置/数据示例 |
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
