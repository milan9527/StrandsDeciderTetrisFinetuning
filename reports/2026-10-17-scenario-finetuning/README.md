# 两个弱项场景的首轮 GPU 微调

2026-10-17，在指定的原实验 EC2 `i-0daca64f56056aa1c`（g6.xlarge / NVIDIA L4）上，
通过统一入口 `finetune.py`，分别从原始 Strands Decider v19 训练两个独立模型。
两项测试准确率均改善，**两项都未达到完整验收条件**：完成核验的概率复核失败；检索可回答性未达到预设 80% 准确率门槛。
模型作为实验产物保存，没有替换 demo 服务的模型。

场景与数据来自此前的 [Jev 场景筛选](../2026-10-17-scenario-screening/README.md)。
这里是本项目独立构造的 choice 任务，不是对 Jev 原仓库成绩的复现或排名。

## 测试结果

每个场景各有 200 条独立 test。以下数字来自这次 test，不能与此前 discovery 的百分比直接相减。
每题原始输入、标签、两版预测及修正/退化分类均已归档，统计从逐题结果重新计算。

| 场景 | 原版准确率 | 微调准确率 | 提升 | 换序一致率：原版 → 微调 | 完整验收 |
| --- | ---: | ---: | ---: | ---: | --- |
| 完成凭据判断 | 43%（86/200） | **92.5%（185/200）** | +49.5 个百分点 | 70.5% → 98% | 概率重新加载复核失败 |
| 检索可回答性 | 54%（108/200） | **78%（156/200）** | +24 个百分点 | 96% → 94% | 未达到准确率 80% 门槛 |

| 其他指标 | 完成核验：原版 → 微调 | 检索可回答性：原版 → 微调 |
| --- | ---: | ---: |
| 候选反序并改名后的准确率 | 47.5% → 93.5% | 55% → 79% |
| 测试平均 NLL | 1.040316 → 0.140852 | 0.922799 → 0.473455 |

### 完成凭据判断

规则要求使用目标版本的最新检查回执，并确认对应版本的产物存在。
在 133 道不应确认完成的题中，**错误确认从 77/133（57.89%）降到 0/133**。
67 道确实完成的题，两版均正确确认全部 67 道。
微调版剩余错误：11 道“证据不足”被判为“明确否定完成”，4 道“明确否定完成”被判为“证据不足”。
逐题比较修正了原版 99 道错误，没有把原本正确的题改错。

实际测试题 `row=1`：当前版本的 smoke 回执在序号 100 时通过、137 时失败，
Agent 仍声称完成；原版确认完成，微调版正确判为当前证据否定完成。
仍错误的 `row=17`：当前版本的 unit/replay 最新检查均通过，但 smoke 只有旧版本回执，
应判证据不足；微调版却判为明确失败。

这证明本轮模型对这套合成规则的拟合明显改善，不证明真实 Agent 日志的泛化能力，也不证明实际错误放行率为零。
版本匹配、最新回执选择、明确状态聚合等结构化规则仍可直接用代码实现。

- [指标](completion_receipts/summary.json)
- [分类成绩、混淆矩阵、实际案例](completion_receipts/diagnostics.json)
- [全部 200 题前后对照](completion_receipts/test-comparison.jsonl.gz)

### 检索可回答性

使用 SQuAD 2.0 原始标签，将任务改为判断给定段落是否含有所问答案。
100 道无答案题中，**错误判断已有答案从 92/100 降到 34/100**。
代价是 100 道有答案题的正确识别从 100/100 降到 90/100，即新增 10 道误拒绝。
换序一致率也从 96% 降到 94%；不能将本轮结果称为全面改善。
逐题比较修正 58 道原版错误，新增 10 道错误。

退化案例 `row=21` 问维持结构刚性的力是什么，段落明确说明 normal force
维持桌子和地板的结构完整性。原版正确认为有答案，微调版误判需要更多证据。

这只是给定段落的可回答性判断，没有运行检索系统，也没有测试端到端回答正确率。
原始标签可能存在争议，公开数据也可能与基座预训练语料重叠。

- [入口最终报告](retrieval_answerability/summary.json)
- [分类成绩、混淆矩阵、实际案例](retrieval_answerability/diagnostics.json)
- [全部 200 题前后对照](retrieval_answerability/test-comparison.jsonl.gz)

## 训练、分区与模型选择

两项各使用 train 1,600 / validation 200 / calibration 200 / test 200。
此前 discovery 的 160 条筛选题没有进入训练、验证、校准或本次测试。
四份实际 GPU 输入与归档源数据规范化后的内容逐条相同。

完成核验使用不同种子生成、完整状态去重的合成回执，分区共享生成模板。
检索任务按文章分组，test 来自官方 dev，其余分区来自官方 train，并排除跨分区文章重叠。
输入最大长度全部检查通过，没有截断。

| 配置或记录 | 完成核验 | 检索可回答性 |
| --- | ---: | ---: |
| epoch / 优化步数 | 1 / 100 | 1 / 100 |
| microbatch / 梯度累积 | 2 / 8 | 2 / 8 |
| LoRA / head 学习率 | `2e-5` / `1e-4` | `2e-5` / `1e-4` |
| 随机种子 | 20261017 | 20261017 |
| 原版验证准确率 | 43% | 54.5% |
| 第 50 / 100 步验证准确率 | 93% / 93.5% | 78% / 79.5% |
| 最佳检查点 | 第 100 步 | 第 100 步 |
| 最长训练/验证输入 | 1,562 token | 660 token |
| 训练循环时间 | 1,166.50 秒 | 338.27 秒 |
| PyTorch 峰值 allocated 显存 | 4.88 GiB | 4.25 GiB |
| 独立校准温度 | 1.4125376 | 1.1220185 |

两项都继续训练原有 LoRA（16,819,200 参数）和 pointer head（1,053,184 参数），冻结基座。
训练前权重与筛选实验中的原始 v19 散列一致，最后和最佳检查点的 LoRA/head 都确实更新。
检查点只依据 validation 选择；本轮没有看 test 后调整训练参数或验收门槛。

## 未通过的验收项

训练前固定的数值门槛为准确率至少 80%、不低于原版、换序一致率至少 90%。
概率重新加载校验另外要求 16 条探针的选择相同，平均 NLL 在相对 1% 或绝对 `1e-4` 容差内一致。
这些门槛保持不变。

**完成核验：**训练和两版完整评估都已完成，但首个任务在 25 分钟边界于 reload 阶段超时。
随后两次通过同一入口 `--resume`，核验并复用已完成的五个阶段，只重跑 reload。
两次均因概率/NLL 不一致失败，第二次复用了原任务的 CUDA/Triton 缓存，仍未解决。

独立诊断中的 16 条预测全部与评估探针一致；连续两次重新推理的结果也完全一致。
但 NLL 从参考值 `0.0506887280` 变为 `0.0489905284`，绝对差 `0.0016981996`，相对差约 **3.35%**。
重新加载的 LoRA/head 散列与选中的第 100 步模型完全相同；校准前后权重和 tokenizer 文件也一致。
差异的具体原因尚未完全定位，不能将其宣称为已验证通过的概率输出。

完成核验没有入口生成的最终成功报告或最终 SHA256 清单。
归档中的 `summary.json` 明确标为根据已完成评估整理，`pipeline_verified=false`、`quality_passed=null`。
三项任务指标门槛虽然通过，整体流程仍失败。保存了所有已完成阶段的散列清单及失败记录。
见 [NLL 诊断](completion_receipts/reload-diagnostic.json)、[恢复日志](completion_receipts/resume-console.log)
和 [原始错误日志](completion_receipts/reload.log)。

**检索可回答性：**六个流程阶段全部完成，16 条探针的重新加载预测和 NLL 核对通过。
但测试准确率 78% 低于预设 80%，入口正确返回退出码 **2**、`quality_passed=false`。
一致率 94% 达到预设 90% 门槛，但低于原版 96%。
见 [入口状态](retrieval_answerability/pipeline.json)和[重新加载记录](retrieval_answerability/reload-verification.json)。

## 产物与复现

数据继续使用筛选报告中的独立分区，模型路径填入实际原始 v19 和基座位置：

```bash
cp examples/scenario-finetune.json my-scenario.json
# 设置 model.checkpoint 和 model.base_model。
uv run python finetune.py \
  --data reports/2026-10-17-scenario-screening/data/completion_receipts \
  --config my-scenario.json --output artifacts/runs/completion-repeat
# 换成 retrieval_answerability 目录和新的输出目录，即可运行第二个场景。
```

本次使用已有容器环境，因此远端实际配置 `runtime.auto_setup=false`。
没有重新测试 CUDA 自动安装路径。两份实际配置和逐步记录分别保存在任务子目录。

本机模型目录（都仍需原始 `/opt/models/base` 基座）：

```text
artifacts/scenario-finetuning/retrieval_answerability/download/completion-resumed/eval-v1/adapted/calibrated/
artifacts/scenario-finetuning/retrieval_answerability/download/pipeline/eval-v1/adapted/calibrated/
```

私有备份前缀：

```text
s3://strands-decider-tetris-training-632930644527/answerability-sft-20261017-014016/results/
```

其中 `completion-resumed/` 保存完成核验的模型、完整评估及未通过的恢复状态；`pipeline/` 保存检索实验。
完整产物的下载散列、训练源文件散列、逐题核对结果和模型位置见各任务的 `verification.json`。
新增的逐题预测存储使统一入口直接保存 `eval-v1/*/predictions.jsonl.gz`，可按测试行号回查错误。
91 项测试和 Ruff 检查通过。后续本地代码补充了 NLL 失败信息中的期望值、实际值和容差，
没有改变判断逻辑；远端运行的原始代码散列仍保存在验证记录中。

训练任务结束后，已恢复原 demo 服务的 desired count、任务定义、控制器并发和调度规则，
核验模型应用在 8099 端口完成启动。没有启动新 EC2，没有使用 vLLM 实例。
恢复证据见各任务的 `service-restoration.json`。

下一步应先定位完成核验的概率复现差异，再用真实任务回执验证其分类收益。
检索场景适合补充范围、时间、对象错配以及有答案但表达不同的案例，并复核目标领域标签；
后续按 validation 调整训练，并另留未见过的测试数据，避免反复针对这 200 条已知测试题调参。
