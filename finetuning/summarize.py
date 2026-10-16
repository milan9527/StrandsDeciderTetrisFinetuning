"""Summarize a finished, downloaded baseline/adapted evaluation without new inference."""

import argparse
import json
from pathlib import Path


def summarize(folder):
    folder = Path(folder)
    reports = {
        name: json.loads((folder / "eval-v1" / name / "report.json").read_text())
        for name in ("baseline", "adapted")
    }
    a, b = reports.values()
    if a["data_manifest"] != b["data_manifest"]:
        raise ValueError("Models were evaluated on different data")
    def key(game):
        return game["seed"], game["policy"], game["max_pieces"]
    if sorted(map(key, a["games"])) != sorted(map(key, b["games"])):
        raise ValueError("Full-game comparison settings differ")
    training = json.loads((folder / "run-v1/run.json").read_text())
    if training["run_kind"] != "sft" or training.get("completed_steps", 0) < 1:
        raise ValueError("Not a completed SFT run")
    summary = {"training": {
        k: training[k] for k in ("completed_steps", "train_rows", "validation_rows",
                                "best_step", "wall_seconds", "max_gpu_memory_bytes")
    }, "models": {}}
    for name, report in reports.items():
        groups = {}
        for policy in ("original", "model"):
            games = [game for game in report["games"] if game["policy"] == policy]
            if not games or not all(game["board_replay_matches"] and
                                    game["raw_choice_matches_applied_placement"] for game in games):
                raise ValueError("Missing or unverified games")
            total_moves = sum(game["pieces"] for game in games)
            groups[policy] = {
                "games": len(games),
                "reached_piece_limit": sum(game["finish_reason"] == "piece_limit" for game in games),
                "mean_pieces": sum(game["pieces"] for game in games) / len(games),
                "mean_lines": sum(game["lines"] for game in games) / len(games),
                "mean_score": sum(game["score"] for game in games) / len(games),
                "dominated_moves": sum(game["stats"]["dominated_moves"] for game in games),
                "total_moves": total_moves,
            }
        summary["models"][name] = {
            "choices": report["heldout_choices"], "games": groups,
            "per_game": [{k: game[k] for k in (
                "seed", "policy", "pieces", "lines", "score", "finish_reason"
            )} for game in report["games"]],
        }
    return summary


def markdown(summary):
    lines = [
        "# Tetris SFT 首轮结果", "",
        "训练标签为引擎单步指标支配关系。本轮没有教师模型调用；未运行 RL。",
        "以下为同一 GPU 环境的描述性对比，不代表多训练种子的统计结论。", "",
        "| 指标 | 原始 v19 | SFT |", "| --- | ---: | ---: |",
    ]
    models = summary["models"]
    for title, get in (
        ("独立测试题准确率", lambda c: c["canonical"]["accuracy"]),
        ("反序并重命名后的准确率", lambda c: c["reversed_and_renamed"]["accuracy"]),
        ("两次实际落点一致率", lambda c: c["same_actual_choice_fraction"]),
    ):
        values = [get(models[name]["choices"]) for name in ("baseline", "adapted")]
        lines.append(f"| {title} | {values[0]:.2%} | {values[1]:.2%} |")
    lines += ["", "| 模型 | 输入 | 达到上限 | 平均落块 | 平均消行 | 指标劣选 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, label in (("baseline", "原始 v19"), ("adapted", "SFT")):
        for policy, title in (("original", "数字"), ("model", "自然语言")):
            g = models[name]["games"][policy]
            lines.append(
                f"| {label} | {title} | {g['reached_piece_limit']}/{g['games']} | "
                f"{g['mean_pieces']:.1f} | {g['mean_lines']:.1f} | "
                f"{g['dominated_moves']}/{g['total_moves']} |"
            )
    lines += ["", "每个模型、每种输入使用相同的五个新种子，每局最多 50 块。",
              "所有对局的原始选择与棋盘回放一致；指标劣选不是长期最优性的证明。",
              "校准仅针对引擎比较题，不代表胜率校准。新检查点未自动替换线上模型。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    report = summarize(args.results)
    (args.results / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.results / "RESULTS.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
