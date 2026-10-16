"""Generate documentation SVGs; --png also renders them with Playwright/Chromium."""

import argparse
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
INK, MUTED = "#172B4D", "#53657B"
BLUE, GREEN, AMBER = "#2563EB", "#087F72", "#B45309"
PALE_BLUE, PALE_GREEN, PALE_AMBER = "#EFF6FF", "#ECFDF5", "#FFFBEB"


class Drawing:
    def __init__(self, width, height, title, description):
        self.width, self.height = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{escape(title)}</title><desc id=\"desc\">{escape(description)}</desc>",
            '<defs><style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif}'
            '</style>',
        ]
        for name, color in (("blue", BLUE), ("green", GREEN), ("amber", AMBER)):
            self.parts.append(
                f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
            )
        self.parts += ["</defs>", f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>']
        self.text(64, 65, title, size=34, weight=700)
        self.text(64, 105, description, size=20, color=MUTED)

    def text(self, x, y, value, size=20, color=INK, weight=400, anchor="start"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>'
        )

    def box(self, x, y, w, h, title, lines=(), color=BLUE, fill=PALE_BLUE, dashed=False):
        self.parts.append('<g class="card">')
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" '
            f'fill="{fill}" stroke="{color}" stroke-width="2"{dash}/>'
        )
        self.text(x + 22, y + 34, title, size=23, color=color, weight=700)
        for i, line in enumerate(lines):
            self.text(x + 22, y + 65 + 27 * i, line, size=19)
        self.parts.append("</g>")

    def arrow(self, points, color="blue", dashed=False):
        stroke = {"blue": BLUE, "green": GREEN, "amber": AMBER}[color]
        data = "M " + " L ".join(f"{x} {y}" for x, y in points)
        dash = ' stroke-dasharray="9 7"' if dashed else ""
        self.parts.append(
            f'<path d="{data}" fill="none" stroke="{stroke}" stroke-width="2.5" '
            f'stroke-linejoin="round" marker-end="url(#{color})"{dash}/>'
        )

    def line(self, points, color=BLUE):
        data = "M " + " L ".join(f"{x} {y}" for x, y in points)
        self.parts.append(f'<path d="{data}" fill="none" stroke="{color}" stroke-width="2"/>')

    def save(self, name):
        path = HERE / f"{name}.svg"
        path.write_text("\n".join([*self.parts, "</svg>"]) + "\n", encoding="utf-8")
        return path


def workflow():
    d = Drawing(1400, 1590, "俄罗斯方块微调：从引擎标签到线上对比",
                "首轮 GPU 实验 · 2026-10-16 · Decider v19 · NVIDIA L4 · SFT")
    d.box(360, 150, 600, 104, "01  采样局面与模拟候选",
          ["144 个种子，每局最多 40 块；枚举合法落点",
           "引擎计算消行、空洞、高度、起伏与生成安全性"])
    d.arrow([(660, 254), (660, 290)])
    d.box(360, 290, 600, 104, "02  严格支配标签与独立审计",
          ["六项指标都不差，至少一项更好；共 18,868 题",
           "重算候选与答案；按种子、棋盘隔离四个分区"])
    d.line([(660, 394), (660, 422)])
    d.line([(197, 422), (1202, 422)])
    for x, title, count, purpose in (
        (55, "validation / 验证", "2,120 条", "只用于选择 best"),
        (390, "train / 训练", "12,590 条", "只用于参数更新"),
        (725, "calibration / 校准", "2,090 条", "只用于温度拟合"),
        (1060, "test / 测试", "2,068 条", "用于最终效果报告"),
    ):
        d.arrow([(x + 142, 422), (x + 142, 450)])
        d.box(x, 450, 285, 112, title, [count, purpose])

    d.arrow([(532, 562), (532, 600)])
    d.arrow([(197, 562), (197, 794), (360, 794)])
    d.arrow([(867, 562), (867, 584), (1040, 584), (1040, 938), (960, 938)])
    d.arrow([(1202, 562), (1202, 1098), (960, 1098)])
    d.box(360, 600, 600, 104, "03  GPU 监督微调",
          ["冻结 Qwen 基座；续训已有 LoRA 和 pointer head",
           "1 epoch · 787 步 · 训练子进程约 64 分钟"],
          color=GREEN, fill=PALE_GREEN)
    d.arrow([(660, 704), (660, 748)])
    d.box(360, 748, 600, 92, "04  验证集选择最佳检查点",
          ["每 200 步及结束时验证；best 为第 787 步"])
    d.arrow([(660, 840), (660, 888)])
    d.box(360, 888, 600, 100, "05  校准 choice 概率",
          ["独立校准集选温度；本轮 T = 1.0"])
    d.arrow([(660, 988), (660, 1040)])
    d.box(360, 1040, 600, 116, "06  原版与微调版独立测试",
          ["准确率：60.35% → 99.95%",
           "换序并改名后实际落点一致率：35.44% → 99.85%"])
    d.arrow([(660, 1156), (660, 1208)])
    d.box(360, 1208, 600, 116, "07  完整对局、回放与归档",
          ["每模型：5 个新种子 × 2 种输入；每局上限 50 块",
           "共 20 局、895 次落块；保存模型、报告与校验清单"])
    d.arrow([(660, 1324), (660, 1410)], color="amber", dashed=True)
    d.text(682, 1377, "训练完成后的独立部署", size=18, color=AMBER)
    d.box(360, 1410, 600, 88, "08  AWS 保留原版，新增 SFT 版",
          ["页面可选新旧模型；相同输入格式与方块种子对比"],
          color=AMBER, fill=PALE_AMBER, dashed=True)
    d.text(64, 1552, "图示为已完成实验。统一脚本负责训练与验证，上线部署单独执行。",
           size=20, color=MUTED)
    return d.save("tetris-sft-workflow")


def lora():
    d = Drawing(1280, 1200, "一次 SFT 更新：哪些参数真正发生变化",
                "蓝色实线：前向计算　绿色模块：可训练参数　橙色虚线：参数更新方向")
    d.box(390, 145, 500, 92, "棋盘 + 指令 + 候选描述",
          ["分词与向量化，进入 24 层 Qwen 基座"])
    d.parts.append('<rect x="220" y="287" width="840" height="365" rx="20" '
                   'fill="#F8FAFC" stroke="#CBD5E1" stroke-width="2"/>')
    d.text(245, 324, "基座内的一个目标线性层", size=23, weight=700)
    d.arrow([(640, 237), (640, 350)])
    d.arrow([(640, 350), (420, 350), (420, 388)])
    d.arrow([(640, 350), (820, 350), (820, 388)])
    d.box(260, 388, 320, 114, "原始权重 W₀ · 冻结",
          ["计算 W₀x", "原有参数保持不变"], color=MUTED, fill="#F1F5F9")
    d.box(640, 388, 360, 114, "已有 LoRA A / B · 续训",
          ["计算 (α / r) × B(Ax)", "r = 16，α = 32，dropout = 0.05"],
          color=GREEN, fill=PALE_GREEN)
    d.arrow([(420, 502), (420, 559), (617, 559)])
    d.arrow([(820, 502), (820, 559), (663, 559)])
    d.parts.append('<circle cx="640" cy="559" r="22" fill="#FFFFFF" '
                   f'stroke="{BLUE}" stroke-width="2"/>')
    d.text(640, 568, "+", size=29, color=BLUE, weight=700, anchor="middle")
    d.text(640, 619, "后续层继续计算，得到最终向量表示", size=21, anchor="middle")
    d.arrow([(640, 581), (640, 591)])
    d.arrow([(640, 652), (640, 710)])
    d.box(390, 710, 500, 96, "Pointer 决策头 · 续训",
          ["结合整道题和各候选的表示，计算候选分数"],
          color=GREEN, fill=PALE_GREEN)
    d.arrow([(640, 806), (640, 848)])
    d.box(390, 848, 500, 90, "候选概率",
          ["softmax(scores) → P(选项 1)、P(选项 2)…"])
    d.arrow([(640, 938), (640, 1000)])
    d.box(60, 1000, 260, 90, "正确标签 y",
          ["由游戏引擎生成"])
    d.arrow([(320, 1045), (390, 1045)])
    d.box(390, 1000, 500, 90, "损失 L = −log P(y)",
          ["反向传播；梯度累积后由 AdamW 更新"])
    d.arrow([(890, 1045), (1150, 1045), (1150, 445), (1000, 445)],
            color="amber", dashed=True)
    d.arrow([(1150, 758), (890, 758)], color="amber", dashed=True)
    d.text(1180, 685, "更新", size=22, color=AMBER, weight=700)
    d.text(64, 1154, "本轮更新：LoRA 16,819,200 参数 + 决策头 1,053,184 参数；原基座权重冻结。",
           size=20, color=MUTED)
    return d.save("lora-training-step")


def render(paths):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=1.5)
        for path in paths:
            page.goto(path.as_uri())
            page.evaluate("document.fonts.ready")
            errors = page.evaluate("""() => [...document.querySelectorAll('.card')].flatMap(g => {
                const box = g.querySelector('rect').getBBox();
                return [...g.querySelectorAll('text')].filter(t => {
                    const b = t.getBBox();
                    return b.x < box.x || b.y < box.y ||
                           b.x + b.width > box.x + box.width ||
                           b.y + b.height > box.y + box.height;
                }).map(t => t.textContent);
            })""")
            if errors:
                raise ValueError(f"Text exceeds card boundaries in {path.name}: {errors}")
            page.locator("svg").screenshot(path=str(path.with_suffix(".png")))
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    paths = [workflow(), lora()]
    if args.png:
        render(paths)
    print("\n".join(str(path) for path in paths))
