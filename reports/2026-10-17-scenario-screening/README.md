# 参考 Jev 场景，实测 Decider 的微调候选任务

2026-10-17，根据用户提供的 [Jev Skills 场景目录](https://github.com/wuyoscar/jev-skill/blob/7696a76bf3cada4e9518db4a7a3b2202fe76e1e5/README.zh.md)，
在指定的原实验 EC2 `i-0daca64f56056aa1c`（NVIDIA L4）上评估原始 Strands Decider v19。
实测 6 类场景、910 道独立筛选题，同时评估候选反序并改名后的表现。

**优先微调候选：检索结果是否含答案、任务完成核验、中文主张与证据核对。**
共同弱点是缺乏证据时仍倾向肯定答案或动作。
客服工单路由已有较高准确率，本轮不优先。
模型路由和浏览器控件的结构化硬约束适合先由代码过滤，再考虑模型需要承担的语义判断。

本轮只做筛选，**没有微调权重，也没有评估最终 test 分区**。
这些成绩不能证明微调一定有效。全部任务采用本项目支持的 `choice` 格式；
没有对原生 `noul`、`score` 或其他提示词进行对照。
场景来自参考目录；输入、标签和这次 Decider 实验由本项目单独准备，
不是对原仓库 Jev 结果的复现或同题模型排名。

## 筛选成绩

预先设定的候选门槛为筛选准确率低于 80%，换序一致性低于 90% 单独标记。
门槛用于寻找后续实验方向，不是上线标准。不同任务选项数量和难度不同。

| 场景及参考位置 | 数据 | 答对 / 总题数 | 准确率 | 反序并改名准确率 | 实际选择一致率 |
| --- | --- | ---: | ---: | ---: | ---: |
| [#58 检索结果是否真的含答案](https://github.com/wuyoscar/jev-skill/blob/7696a76bf3cada4e9518db4a7a3b2202fe76e1e5/README.zh.md#sc-suitability) | SQuAD 2.0 | 91/160 | **56.88%** | 56.25% | 98.13% |
| [#3–4 任务完成核验](https://github.com/wuyoscar/jev-skill/blob/7696a76bf3cada4e9518db4a7a3b2202fe76e1e5/README.zh.md#sc-a06) | 可重算规则的合成回执 | 62/160 | **38.75%** | 47.50% | 79.38% |
| [#52 主张与原文是否一致](https://github.com/wuyoscar/jev-skill/blob/7696a76bf3cada4e9518db4a7a3b2202fe76e1e5/README.zh.md#sc-h08) | OCNLI 中文标注 | 90/150 | **60.00%** | 60.00% | 94.00% |
| [#27 浏览器控件选择](https://github.com/wuyoscar/jev-skill/blob/7696a76bf3cada4e9518db4a7a3b2202fe76e1e5/README.zh.md#sc-a23) | 可重算规则的合成页面 | 84/160 | **52.50%** | 46.25% | 76.25% |
| [#19 按任务选择模型](https://github.com/wuyoscar/jev-skill/blob/7696a76bf3cada4e9518db4a7a3b2202fe76e1e5/README.zh.md#sc-a16) | 可重算规则的能力卡 | 49/160 | **30.63%** | 25.00% | 46.88% |
| [#40 客服工单分流](https://github.com/wuyoscar/jev-skill/blob/7696a76bf3cada4e9518db4a7a3b2202fe76e1e5/README.zh.md#sc-h02) | BANKING77 的 12 队列子集 | 111/120 | **92.50%** | 91.67% | 94.17% |

每题也保存了原始输入、正确标签、两种顺序下的实际选择。
各任务的 Wilson 95% 区间、分标签成绩及均匀随机基线见 [summary.json](summary.json)。
OCNLI、BANKING77 和 SQuAD 数字是与原始数据集标签的一致率，不表示所有原始标注都无争议。

## 三个优先方向与实际错例

### 1. 检索结果是否含答案：学习识别缺失证据

80 道有答案题全部答对，但 **80 道无答案题仅识别出 11 道**；
其余 69 道被误判为已有答案。换序一致性很高，不能据此认为判断正确。

实际筛选题 `retrieval_answerability:23`：

- 问题是整个 Texas 有多少所 community colleges。
- 段落介绍 Houston 周边的三个 community college districts，并列出当地系统。
- 原标签：没有提供所问答案。Decider：段落已包含答案。

实际筛选题 `retrieval_answerability:102`：

- 问题是最初版 Unicode 标准有多少页。
- 段落提到原标准首次发表于 1991 年，但给出的 692 页属于后来的 Unicode 6.1 核心规范。
- 原标签：没有提供所问答案。Decider：段落已包含答案。

适合补充的训练案例：主题相同但对象、版本、时间、范围不符；缺关键字段；正确输出“继续检索”。
本例只检测已有段落的可回答性，没有执行检索，也没有验证检索系统端到端效果。

### 2. 任务完成核验：区分完成声明与执行证据

当前规则要求每个必需检查使用目标版本的最新回执，并确认对应版本的产物存在。
106 道不应确认完成的题中，**66 道被判成已经完成（62.26%）**。
其余三分类错误不全部等同于错误放行，完整混淆矩阵见 [diagnostics.json](diagnostics.json)。

实际筛选题 `completion_receipts:1`：

- 目标版本的 lint 在序号 81 时通过，序号 137 时失败。
- Agent 声称全部完成，产物文件存在。
- 标签：当前证据否定完成。Decider：已经验证完成。

实际筛选题 `completion_receipts:2`：

- 必需的 unit、build、smoke 回执全部属于旧版本。
- 当前版本仅有产物存在的回执，Agent 声称完成。
- 标签：证据不足。Decider：已经验证完成。

适合训练的区别：排队与执行、旧版本与当前版本、失败后通过与通过后失败、缺少回执与明确失败。
本次是结构化合成回执；固定版本匹配及状态聚合可由代码直接实现。
真实 Agent 的非结构化证据判断仍需另外验证，不能以本例成绩代替验收系统。

### 3. 中文主张与证据核对：区分矛盾与未知

三类各 50 题，支持、矛盾、未知分别答对 33、36、21 题。

实际筛选题 `claim_support_ocnli:15`：

- 原文：“就取景是黑白的,黑白.”
- 主张：“黑白的取景比彩色的好”
- 原标签：未知。Decider：支持。

实际筛选题 `claim_support_ocnli:17`：

- 原文：“主任不当了,但我教香港文学”
- 主张：“我还教英国文学”
- 原标签：未知。Decider：矛盾。

适合训练的区别：未提及不等于否定、客观描述不等于价值判断、范围和条件限定。
OCNLI 是公开自然语言推断数据，包含口语、省略和有争议标注。
当前只核对了与原始标签的对应关系；正式业务微调前应复核标签并补充目标领域材料。

完整错例的原始输入、全部候选及原始来源 ID 见 [error-examples.json](error-examples.json)。
上述题号的冒号后是零起算的筛选行号，对应记录中的 `row`。
其余逐题结果保存在 `results/<任务>/predictions.jsonl.gz`，没有只保留失败案例。

## 其他场景的取舍

浏览器测试中，应该刷新旧快照的 32 题全部错，应该等待的 16 题也全部错；
实际应点击控件的 96 题答对 82 题。模型明显偏向继续点击。
例如 `browser_controls:6` 的当前页面版本为 4879、观察版本为 4878，
应刷新，模型却选择了一个 invoice 控件，目标实际还是 receipt。
可作为动作选择微调的合成演示；宿主应先检查快照版本、可见性和可用性。
本轮没有实际浏览或点击页面。

模型路由中，84/160 次选择违反至少一项已给定约束，另有 15 次选了可行但并非最低成本的模型。
例如 `model_routing:1` 要求最低质量 80，模型选择了能力卡质量仅 71 的候选。
候选是否可用、上下文长度、费用和时延都已结构化，建议先用代码约束和计算；
低准确率本身不意味着这里必须微调。

客服路由达到 92.5%，没有显示出同等紧迫的微调需求。
这里只测了 12 个与支付、退款、卡片及转账相关的队列，不能外推到完整 77 类或真实工单分布。

## 独立训练数据

每个任务有单独目录，包含以下五份数据：

| 任务目录 | train | validation | calibration | test（未测） | discovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| `completion_receipts` | 1,600 | 200 | 200 | 200 | 160 |
| `model_routing` | 1,600 | 200 | 200 | 200 | 160 |
| `browser_controls` | 1,600 | 200 | 200 | 200 | 160 |
| `claim_support_ocnli` | 1,500 | 150 | 150 | 300 | 150 |
| `support_routing_banking` | 720 | 120 | 120 | 240 | 120 |
| `retrieval_answerability` | 1,600 | 200 | 200 | 200 | 160 |

数据在 [data/](data)，共 13,010 条，均核对来源标签或重新计算规则标签。
OCNLI 按相同原文分组；SQuAD 按文章分组；BANKING77 按规范化后的消息分组。
五个分区一起检查状态和分组泄漏。筛选 GPU 输入包仅包含 discovery。
SQuAD 使用 100～2200 字符的段落；OCNLI 的最终 test 来自官方公开 dev。
合成数据来自不同种子和不同完整状态，但共享生成模板，不能据此证明真实环境泛化。
公开数据与基座预训练语料的重叠、语义近似内容不能排除。

统一入口只读取 train、validation、calibration、test，会忽略 discovery。
六份归档数据已全部通过统一入口的 `--validate-only`。

下一轮以完成核验为例：

```bash
uv sync --locked
cp examples/scenario-finetune.json my-scenario.json
# 填写原始 v19 检查点及对应基座的实际路径。
uv run python finetune.py \
  --data reports/2026-10-17-scenario-screening/data/completion_receipts \
  --config my-scenario.json \
  --output artifacts/runs/completion-sft-v1
```

切换任务只需替换 `--data` 的目录名。这是后续训练命令，本轮未执行。
检查新模型时必须使用未参与选题的 test，并额外关注错误确认完成、无答案误报等具体错误类型。

## 来源、执行与验证

- 场景：[用户指定的 Jev Skills README 固定版本](https://github.com/wuyoscar/jev-skill/blob/7696a76bf3cada4e9518db4a7a3b2202fe76e1e5/README.zh.md)。
- OCNLI：[官方仓库固定版本](https://github.com/CLUEbenchmark/OCNLI/tree/b53efdee17257a5c33993cf6fcf8ffff0497ea0e)。
- BANKING77：[PolyAI 原始数据固定版本](https://github.com/PolyAI-LDN/task-specific-datasets/tree/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data)。
- SQuAD 2.0：[官方数据入口](https://rajpurkar.github.io/SQuAD-explorer/)。
- 原始下载文件及各分区 SHA-256：[provenance.json](provenance.json)。
- 数据准备：[scripts/prepare_scenarios.py](../../scripts/prepare_scenarios.py)。
- 基线筛选：[finetuning/screen.py](../../finetuning/screen.py)，没有优化器和权重更新。

GPU 任务约 229 秒；核心六任务筛选约 218 秒。910 条筛选输入均经过长度检查，未截断。
原版 LoRA/head 的运行前后散列相同，也与前一轮 AG News 实验的原版权重一致。
89 项测试通过；下载的 29 个文件全部通过 SHA-256 校验。
原演示服务、任务定义、自动控制器和调度均恢复，应用启动日志已核验。
没有使用 vLLM 那台实例，也没有替换线上模型。

[运行汇总](summary.json) · [验证记录](verification.json) ·
[服务恢复证据](service-restoration.json) · [本报告文件校验清单](FILES.sha256.json)。
