"""PickPlace、NutAssembly、Door、Wipe 共用的模型测评流程。

每条测评做这些事：

1. 按 collect_common 的方式摆好场景，手臂移到随机出发点（和采集同一个 prepare_scene）
2. 只把一张 frontview RGB 和一句指令放进对话，模型写出从这一帧起还要做的全部步骤
3. 计划里的像素只用深度相机换成世界坐标（depth_locate）：按像素在计划里的角色
   （要抓的物体、格子底面、柱顶、门板上的点……）从附近的深度点量出它所在的平面，
   取视线和平面的交点。标注点常被挡住或在物体内部，所以不直接读该像素的深度
4. 交给 task_plan.execute_plan，末端用世界坐标 OSC 跟踪，执行中不再问模型；
   多物体时每抓下一个之前用当前帧的深度重量它（开局挡在它前面的物体可能已经搬走）
5. 没做成且任务允许多轮（Wipe 会漏几个点），按当前画面重新问一次，只做剩下的
6. 以 env._check_success 判定成功

仿真真值只用于事后统计像素和世界误差，不进模型输入，也不参与定位和控制。
--oracle 用真值步骤代替模型回答，其余流程不变，用来确认深度定位和执行器本身够用。
"""

import time
from pathlib import Path

import gl_backend

gl_backend.configure()

import numpy as np

from collect_common import eef_pose, make_env, prepare_scene, render_plan
from collect_lift_mllm import format_pixel, project_world, round_pixel, world_to_pixel_matrix
from depth_locate import locate_plan, pick_starts, real_depth, relocate
from eval_lift_core import (
    ServerError,
    camera_image,
    camera_size,
    format_cm,
    format_err_axes,
    format_xyz,
    grab_frame,
    pos_error,
    write_mp4,
)
from eval_lift_mllm_vllm import VLLMClient
from eval_stack_mllm_vllm import VIDEO_FPS, mean_or_none, plan_matches, write_results
from task_plan import ParseError, check_parser, execute_plan, format_plan, parse_plan


def plan_pixels(subtasks):
    """计划里用到的所有像素，去重保序。"""
    seen = []

    def add(pixel):
        pixel = np.asarray(pixel, dtype=float).reshape(2)
        if not any(np.allclose(pixel, other) for other in seen):
            seen.append(pixel)

    for step in subtasks:
        kind = step["type"]
        if kind in ("move_above", "move_down", "lift"):
            add(step["pixel"])
            if "held" in step:
                add(step["held"])
        elif kind == "reach":
            add(step["pixel"])
        elif kind == "turn":
            add(step["pivot"])
        elif kind == "wipe":
            for pixel in step["path"]:
                add(pixel)
    return seen


def gt_points(task, env, matrix):
    """任务的真值点和它们的像素、相机深度。只给事后统计用，不进模型，不参与定位。"""
    out = []
    for name, world in task.points(env).items():
        pixel, depth = project_world(world, matrix)
        out.append({"name": name, "world": np.asarray(world, dtype=float), "pixel": np.asarray(pixel, dtype=float), "depth": depth})
    return out


def gt_pool(gt, role):
    """统计误差时，按角色只和同类的真值点配对。螺母中心和柱顶在画面里可能只差一两个像素。"""
    if role == "bin":
        pool = [item for item in gt if item["name"].endswith("_bin")]
    elif role == "peg":
        pool = [item for item in gt if item["name"].endswith("_peg")]
    elif role in ("object", "nut"):
        pool = [item for item in gt if not item["name"].endswith(("_bin", "_peg"))]
    else:
        pool = []
    return pool or gt


def point_group(name):
    """dirt12 归到 dirt，其余名字（Milk_bin、SquareNut_grasp、handle、path_first）不动。"""
    return name.rstrip("0123456789") or name


def match_targets(points, gt):
    """每个定位点配像素距离最近的同类真值点，记录像素和世界误差。"""
    targets = []
    for point in points:
        pool = gt_pool(gt, point["role"])
        nearest = min(pool, key=lambda item: float(np.linalg.norm(point["pixel"] - item["pixel"])))
        dist = float(np.linalg.norm(point["pixel"] - nearest["pixel"]))
        err = pos_error(point["world"], nearest["world"])
        targets.append(
            {
                "point": nearest["name"],
                "group": point_group(nearest["name"]),
                "pixel": round_pixel(point["pixel"]),
                "gt_pixel": round_pixel(nearest["pixel"]),
                "pixel_err": round(dist, 2),
                "world": [round(float(v), 4) for v in point["world"]],
                "gt_world": [round(float(v), 4) for v in nearest["world"]],
                "err_m": {k: round(v, 4) for k, v in err.items()},
                "depth": point["info"],
            }
        )
    return targets


def run_episode(env, task, client, rng, episode, image_size, matrix, oracle, style, video_path=None):
    frames = []
    latest = {}

    def on_step(obs):
        if obs is not None:
            latest["obs"] = obs
        if video_path is not None and obs is not None:
            frames.append(grab_frame(obs, image_size))

    record = None
    try:
        obs, hold_ori = prepare_scene(env, task, rng, on_step)
        record = {"episode": episode, "task": task.task, "success": False, "rounds": []}
        for round_index in range(int(task.max_rounds)):
            plan = task.build_plan(env, rng)
            if bool(task.success(env)):
                break
            instruction = task.instruction(env, plan, 0, style, False)
            gt_subtasks, _ = render_plan(task, env, plan, 0, matrix, False)
            if not gt_subtasks:
                break
            gt_answer = format_plan(gt_subtasks)
            image = camera_image(obs, image_size)
            depth = real_depth(env, obs)
            # 真值只用来统计误差，不进模型，不进定位和控制。
            gt = gt_points(task, env, matrix)
            started = time.perf_counter()
            answer = gt_answer if oracle else client.complete(image, instruction)
            round_rec = {
                "round": round_index,
                "instruction": instruction,
                "seconds": round(time.perf_counter() - started, 3),
                "answer": answer,
            }
            record["rounds"].append(round_rec)
            print("episode {:02d} round {} instruction: {}".format(episode, round_index, instruction), flush=True)
            print(
                "episode {:02d} round {} answer ({}s):\n{}".format(episode, round_index, round_rec["seconds"], answer),
                flush=True,
            )
            try:
                subtasks = parse_plan(answer, image_size=image_size)["subtasks"]
                round_rec["steps"] = [step["type"] for step in subtasks]
                round_rec["plan_match"] = plan_matches(subtasks, gt_answer, image_size)
                points = locate_plan(task.task, subtasks, depth, matrix, image_size)
                round_rec["targets"] = match_targets(points, gt)
                for target in round_rec["targets"]:
                    print(
                        "episode {:02d} {} pixel gt={} pred={} pos gt={} pred={} {}".format(
                            episode,
                            target["point"],
                            format_pixel(target["gt_pixel"]),
                            format_pixel(target["pixel"]),
                            format_xyz(target["gt_world"]),
                            format_xyz(target["world"]),
                            format_err_axes(target["err_m"]),
                        ),
                        flush=True,
                    )
                starts = set(pick_starts(task.task, subtasks))
                relocated = []

                def on_progress(index, _step):
                    if index in starts and "obs" in latest:
                        point = relocate(
                            task.task, subtasks, index, real_depth(env, latest["obs"]), matrix, image_size
                        )
                        if point is not None:
                            relocated.extend(match_targets([point], gt_points(task, env, matrix)))

                final, trace = execute_plan(
                    env,
                    subtasks,
                    hold_ori,
                    [point["world"] for point in points],
                    matrix,
                    grip=-1.0,
                    on_step=on_step,
                    on_progress=on_progress,
                )
                round_rec["trace"] = trace
                if relocated:
                    round_rec["relocated"] = relocated
                if final is not None:
                    obs = final
                hold_ori = eef_pose(env)[1]
            except ValueError as exc:
                # ParseError 和反投影的 ValueError：这一轮没法执行，记下了事
                round_rec["error"] = str(exc)
                break
        record["success"] = bool(task.success(env))
        record["describe"] = task.describe(env)
        record.update(task.record_extra(env))
        return record
    finally:
        if video_path is not None and frames:
            try:
                saved = write_mp4(video_path, frames, fps=VIDEO_FPS)
            except Exception as exc:
                print("video failed: {}".format(exc), flush=True)
                if record is not None:
                    record["video_error"] = str(exc)
            else:
                if record is not None:
                    record["video"] = "videos/" + saved.name


def summarize(records):
    total = len(records)
    rounds = [rr for record in records for rr in record.get("rounds", [])]
    groups = {}
    for record in records:
        first = record.get("rounds") or []
        # 点的统计只用第一轮，各条测评之间可比
        for target in (first[0].get("targets") if first else []) or []:
            groups.setdefault(target["group"], []).append(target)
    out = {
        "episodes": total,
        "successes": sum(int(bool(r["success"])) for r in records),
        "rounds": len(rounds),
        "parse_failures": sum(int("steps" not in rr) for rr in rounds),
        "plan_matches": sum(int(bool(rr.get("plan_match"))) for rr in rounds),
    }
    out["success_rate"] = None if total == 0 else out["successes"] / total
    out["points"] = {
        name: {
            "n": len(items),
            "pixel_err": mean_or_none([t["pixel_err"] for t in items]),
            "xy_err_m": mean_or_none([t["err_m"]["xy"] for t in items]),
            "z_err_m": mean_or_none([abs(t["err_m"]["z"]) for t in items]),
        }
        for name, items in sorted(groups.items())
    }
    return out


def print_episode(record):
    rounds = record.get("rounds") or []
    steps = ">".join(step for rr in rounds for step in rr.get("steps", []))
    print(
        "episode {:02d} success={} {} rounds={} steps={}".format(
            record["episode"],
            record["success"],
            record.get("describe") or "",
            len(rounds),
            steps or "n/a",
        ),
        flush=True,
    )
    for rr in rounds:
        failed = [step for step in rr.get("trace") or [] if not step["ok"]]
        if failed:
            print(
                "  round {} 未到位：{}".format(
                    rr["round"], ", ".join("{}({})".format(s["type"], format_cm(s["err_m"])) for s in failed)
                ),
                flush=True,
            )
        if rr.get("error"):
            print("  round {} error: {}".format(rr["round"], rr["error"]), flush=True)
    if record.get("video"):
        print("  video {}".format(record["video"]), flush=True)


def print_summary(summary):
    total = summary["episodes"]
    if total == 0:
        print("没有完成任何一条测评", flush=True)
        return
    print(
        "成功率 {}/{} = {:.1%}    解析失败 {}    步骤与标注一致 {}/{}".format(
            summary["successes"],
            total,
            summary["success_rate"],
            summary["parse_failures"],
            summary["plan_matches"],
            summary["rounds"],
        ),
        flush=True,
    )
    for name, stats in summary["points"].items():
        print(
            "{} n={} 像素误差 {}  水平误差 {}  高度误差 {}".format(
                name,
                stats["n"],
                "n/a" if stats["pixel_err"] is None else "{:.1f}px".format(stats["pixel_err"]),
                format_cm(stats["xy_err_m"]),
                format_cm(stats["z_err_m"]),
            ),
            flush=True,
        )


def add_common_args(parser, default_out):
    parser.add_argument("--model", default="", help="模型名。留空则用服务 /v1/models 的第一个")
    parser.add_argument("--episodes", type=int, default=20, help="测评多少条")
    parser.add_argument("--seed", type=int, default=1000, help="和采集用的 seed 0 错开；传 0 则和采集同批场景")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=2048, help="多物体任务的计划有三十几步，比 Stack 长")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system", default="", help="可选的 system prompt，默认不发")
    parser.add_argument("--text-template", default="{instruction}", help="拼在图片后面的文本，默认就是指令原文")
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--no-video", action="store_true", help="不把每条测评的 frontview 录成 mp4")
    parser.add_argument("--oracle", action="store_true", help="用真值步骤代替模型，检查执行器")
    parser.add_argument("--self-test", action="store_true", help="只检查像素收集和解析")
    return parser


def make_client(args):
    if args.oracle:
        print("oracle mode, skip model server", flush=True)
        return None
    client = VLLMClient(
        args.api_base,
        args.model,
        args.timeout,
        args.max_tokens,
        args.temperature,
        args.system,
        args.text_template,
    )
    print("api {} model {}".format(client.api_base, client.resolve_model()), flush=True)
    return client


def run_eval(task, args, client):
    check_parser()
    rng = np.random.default_rng(args.seed)
    env = make_env(task, args.image_size, args.seed, use_camera=True, camera_depths=True)
    image_size = camera_size(env)
    matrix = world_to_pixel_matrix(env, image_size)
    video_dir = None if args.no_video else Path(args.out) / "videos"
    records = []
    try:
        for episode in range(args.episodes):
            style = task.styles[int(rng.integers(len(task.styles)))]
            video_path = None if video_dir is None else video_dir / "ep{:04d}.mp4".format(episode)
            try:
                record = run_episode(
                    env, task, client, rng, episode, image_size, matrix, args.oracle, style, video_path
                )
            except ServerError as exc:
                print(exc, flush=True)
                break
            records.append(record)
            print_episode(record)
    finally:
        summary = summarize(records)
        summary["task"] = task.task
        summary["env"] = task.env_name
        summary["oracle"] = bool(args.oracle)
        summary["seed"] = args.seed
        summary["backend"] = "oracle" if args.oracle else "vllm"
        if client is not None:
            summary["model"] = client.model
            summary["api_base"] = client.api_base
        jsonl_path, summary_path = write_results(args.out, records, summary)
        env.close()
        print_summary(summary)
        print("wrote {} and {}".format(jsonl_path, summary_path), flush=True)
        if video_dir is not None:
            print("videos in {}".format(video_dir), flush=True)


def self_test():
    check_parser()
    # Door：reach 的 target、turn 的 pivot 都要收进来
    door = parse_plan(
        format_plan(
            [
                {"type": "reach", "target": "[150 240]", "approach": [-0.14, -0.99, 0.0], "standoff": 0.1},
                {"type": "reach", "target": "[150 240]", "approach": [-0.14, -0.99, 0.0], "standoff": 0.0},
                {"type": "close_gripper"},
                {"type": "turn", "pivot": "[160 250]", "axis": [-0.14, -0.99, 0.0], "angle": -83},
                {"type": "turn", "pivot": "[120 200]", "axis": [0.0, 0.0, 1.0], "angle": 23},
                {"type": "stabilize", "duration": 0.5},
            ]
        ),
        image_size=512,
    )["subtasks"]
    pixels = [tuple(np.round(p, 1)) for p in plan_pixels(door)]
    assert pixels == [(150.0, 240.0), (160.0, 250.0), (120.0, 200.0)], pixels
    # Nut：held 也收；Wipe：path 每一点都收
    nut = parse_plan(
        format_plan(
            [
                {"type": "move_above", "target": "[200 310]", "clearance": 0.1, "yaw": -35},
                {"type": "move_down", "target": "[200 310]", "offset_z": 0.0},
                {"type": "close_gripper"},
                {"type": "lift", "target": "[200 310]", "delta_z": 0.17},
                {"type": "move_above", "target": "[320 280]", "clearance": 0.05, "held": "[205 312]"},
                {"type": "move_down", "target": "[320 280]", "offset_z": -0.165, "held": "[205 312]"},
                {"type": "open_gripper"},
                {"type": "move_above", "target": "[320 280]", "clearance": 0.06, "held": "[205 312]"},
                {"type": "stabilize", "duration": 0.5},
            ]
        ),
        image_size=512,
    )["subtasks"]
    pixels = [tuple(np.round(p, 1)) for p in plan_pixels(nut)]
    assert pixels == [(200.0, 310.0), (320.0, 280.0), (205.0, 312.0)], pixels
    wipe = parse_plan(
        format_plan(
            [
                {"type": "move_above", "target": "[200 300]", "clearance": 0.1, "yaw": 0},
                {"type": "move_down", "target": "[200 300]", "offset_z": 0.0},
                {"type": "wipe", "path": ["[200 300]", "[210 305]", "[220 310]"], "offset_z": -0.01},
                {"type": "lift", "target": "[220 310]", "delta_z": 0.1},
                {"type": "stabilize", "duration": 0.5},
            ]
        ),
        image_size=512,
    )["subtasks"]
    pixels = [tuple(np.round(p, 1)) for p in plan_pixels(wipe)]
    assert pixels == [(200.0, 300.0), (210.0, 305.0), (220.0, 310.0)], pixels
    print("self-test ok", flush=True)
