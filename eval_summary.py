"""把各任务 summary.json 收成一张表，打印到标准输出。"""

import json
import sys
from pathlib import Path

# 目录名、表上的任务名、计划一致的分母。成功率相同时按这个顺序排。
TASKS = [
    ("lift", "Lift", "lift"),
    ("door", "Door", "rounds"),
    ("stack", "Stack", "episodes"),
    ("wipe", "Wipe", "rounds"),
    ("pickplace_PickPlaceCan", "PickPlaceCan", "rounds"),
    ("pickplace_PickPlaceBread", "PickPlaceBread", "rounds"),
    ("pickplace_PickPlaceCereal", "PickPlaceCereal", "rounds"),
    ("pickplace_PickPlaceMilk", "PickPlaceMilk", "rounds"),
    ("nut_NutAssemblySquare", "NutAssemblySquare", "rounds"),
    ("nut_NutAssemblyRound", "NutAssemblyRound", "rounds"),
    ("nut_NutAssembly", "NutAssembly（双螺母）", "rounds"),
    ("pickplace_PickPlace", "PickPlace（四物体）", "rounds"),
]


def percent(num, den):
    rate = 100.0 * num / den
    if abs(rate - round(rate)) < 1e-6:
        return "{:.0f}%".format(rate)
    return "{:.1f}%".format(rate)


def load_summary(out, dirname):
    path = out / dirname / "summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def plan_cell(summary, kind):
    if kind == "lift" or "plan_matches" not in summary:
        return "不适用"
    matched = int(summary["plan_matches"])
    if kind == "rounds":
        den = int(summary.get("rounds", summary["episodes"]))
    else:
        den = int(summary["episodes"])
    return "{}/{}".format(matched, den)


def render(out):
    rows = []
    for index, (dirname, title, kind) in enumerate(TASKS):
        summary = load_summary(out, dirname)
        if summary is None:
            rows.append({
                "index": index,
                "title": title,
                "ok": False,
                "rate": 0.0,
                "success": "无结果",
                "plan": "—",
                "successes": 0,
                "episodes": 0,
            })
            continue
        successes = int(summary["successes"])
        episodes = int(summary["episodes"])
        rate = 0.0 if episodes == 0 else successes / episodes
        success = "无结果" if episodes == 0 else "**{}/{} = {}**".format(
            successes, episodes, percent(successes, episodes)
        )
        rows.append({
            "index": index,
            "title": title,
            "ok": episodes > 0,
            "rate": rate,
            "success": success,
            "plan": plan_cell(summary, kind),
            "successes": successes,
            "episodes": episodes,
        })

    rows.sort(key=lambda row: (not row["ok"], -row["rate"], row["index"]))
    lines = [
        "| 任务 | 成功率 | 计划一致 |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| {} | {} | {} |".format(row["title"], row["success"], row["plan"]))
    done = [row for row in rows if row["ok"]]
    total_success = sum(row["successes"] for row in done)
    total_episodes = sum(row["episodes"] for row in done)
    lines.append("")
    if not done:
        lines.append("合计（0 个设置）：无结果")
    else:
        if len(done) == len(rows):
            head = "合计（{} 个设置）".format(len(done))
        else:
            head = "合计（{}/{} 个设置有结果）".format(len(done), len(rows))
        lines.append("{}：{}/{} ≈ {:.1f}%".format(
            head, total_success, total_episodes, 100.0 * total_success / total_episodes
        ))
    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        print("用法: eval_summary.py <评测输出目录>", file=sys.stderr)
        return 2
    print(render(Path(sys.argv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
