# 微调流程图片

| 图片 | 内容 |
| --- | --- |
| [整体流程 PNG](tetris-sft-workflow.png) / [SVG](tetris-sft-workflow.svg) | 本轮数据、GPU 训练、四分区用途、评估与独立部署 |
| [LoRA 更新 PNG](lora-training-step.png) / [SVG](lora-training-step.svg) | 冻结权重、LoRA、决策头、标签损失与参数更新 |

正文使用 PNG，确保 GitHub 和导出工具直接显示中文；SVG 可以无损放大。
图是依据实验记录绘制的说明图，不是运行日志截图。
流程中的部署为单独操作，不表示 `finetune.py` 会自动发布。

用标准库生成 SVG：

```bash
python docs/diagrams/generate.py
```

同时生成 PNG 需要 Playwright、Chromium 和中文字体（本次使用 Noto Sans CJK SC）：

```bash
uv run --no-project --with playwright python -m playwright install chromium
uv run --no-project --with playwright python docs/diagrams/generate.py --png
```

PNG 以 1.5 倍像素密度渲染。生成器会检查卡片内文本是否超出边界。
颜色与布局在 `generate.py` 中维护；修改后同时更新 SVG 和 PNG，并检查箭头与文字是否清楚。
