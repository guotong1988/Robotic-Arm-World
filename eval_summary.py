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
        return None, "未完成"
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except (OSError, ValueError):
        return None, "summary.json 无法读取"


def plan_cell(summary, kind):
    if kind == "lift" or "plan_matches" not in summary:
        return "不适用"
    matched = int(summary["plan_matches"])
    if kind == "rounds":
        den = int(summary.get("rounds", summary["episodes"]))
    else:
        den = int(summary["episodes"])
    return "{}/{}".format(matched, den)


def notes_for(summary, kind, missing_note):
    if summary is None:
        return missing_note
    notes = []
    if kind == "lift":
        notes.append("Lift 不比整条计划，成功率就是抓住并抬起的比例")
        episodes = int(summary["episodes"])
        successes = int(summary["successes"])
        grasps = summary.get("grasps")
        failed = episodes - successes
        if grasps is not None and failed > 0:
            grasps = int(grasps)
            ungrasped = episodes - grasps
            not_lifted = grasps - successes
            if not_lifted <= 0:
                notes.append("失败 {} 条未抓住".format(ungrasped))
            else:
                parts = []
                if ungrasped:
                    parts.append("未抓住 {} 条".format(ungrasped))
                if not_lifted:
                    parts.append("抓住但未抬起 {} 条".format(not_lifted))
                notes.append("，".join(parts))
    failures = summary.get("parse_failures") or 0
    if failures:
        notes.append("{} 次解析失败".format(int(failures)))
    return "；".join(notes)


def render(out):
    rows = []
    for index, (dirname, title, kind) in enumerate(TASKS):
        summary, missing_note = load_summary(out, dirname)
        if summary is None:
            rows.append({
                "index": index,
                "title": title,
                "ok": False,
                "rate": 0.0,
                "success": "无结果",
                "plan": "—",
                "note": notes_for(None, kind, missing_note),
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
            "note": notes_for(summary, kind, ""),
            "successes": successes,
            "episodes": episodes,
        })

    rows.sort(key=lambda row: (not row["ok"], -row["rate"], row["index"]))
    lines = [
        "| 任务 | 成功率 | 计划一致 | 备注 |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append("| {} | {} | {} | {} |".format(row["title"], row["success"], row["plan"], row["note"]))
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
