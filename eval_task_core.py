"""PickPlace、NutAssembly、Door、Wipe 共用的模型测评流程。

每条测评做这些事：

1. 按 collect_common 的方式摆好场景，手臂移到随机出发点（和采集同一个 prepare_scene）
2. 指令以一个 <image> 占位开头（图走 image_url），话术后面带字段说明；
   模型写出从这一帧起还要做的全部步骤：空间字段给 frontview 像素，
   抓取步骤给两指落点对（target_1、target_2，target_kind=grip），
   每步带一句 desc
3. 计划里的像素只用深度相机换成世界坐标（depth_locate）：按像素的角色（模型在计划里
   用 *_kind 标注的类别优先，没标注按任务和计划结构推断。要抓的物体、格子底面、柱顶、
   门板上的点、两指落点……）从附近的深度点量出它所在的平面，
   取视线和平面的交点。标注点常被挡住或在物体内部，所以不直接读该像素的深度；
   两指落点成对还原（grip），执行取两落点的中点，统计误差也用中点配真值
4. 交给 task_plan.execute_plan，末端用世界坐标 OSC 跟踪，执行中不再问模型；
   多物体时每抓下一个之前用当前帧的深度重量它（开局挡在它前面的物体可能已经搬走）
5. 没做成且任务允许多轮（Wipe 会漏几个点），按当前画面重新问一次，只做剩下的
6. 以 env._check_success 判定成功

仿真真值只用于事后统计像素和世界误差，不进模型输入，也不参与定位和控制。
--oracle 用真值步骤代替模型回答，其余流程不变，用来确认深度定位和执行器本身够用。
录像使用 frontview 单画面，标出模型回答的原始像素，不叠加深度点云。
"""

import time
from pathlib import Path

import gl_backend

gl_backend.configure()

import numpy as np

from collect_common import eef_pose, make_env, prepare_scene, render_plan, task_instruction
from collect_lift_mllm import (
    CAMERAS,
    DEFAULT_CAMERA,
    format_pixel,
    project_world,
    round_pixel,
    world_to_pixel_matrices,
)
from depth_locate import locate_plan, make_views, pick_starts, relocate
from eval_lift_mllm_vllm import (
    ServerError,
    VLLMClient,
    camera_images,
    camera_size,
    format_cm,
    format_err_axes,
    format_xyz,
    pos_error,
    reset_episode_rng,
    write_mp4,
)
from eval_video import EvalVideo
from eval_stack_mllm_vllm import VIDEO_FPS, mean_or_none, plan_matches, write_results
from task_plan import ParseError, check_parser, execute_plan, format_plan, parse_plan


def plan_pixels(subtasks):
    """计划里用到的所有像素，去重保序。"""
    seen = []

    def add(pixel):
        if pixel is None:
            # 兼容旧数据里的空像素，跳过统计。
            return
        pixel = np.asarray(pixel, dtype=float).reshape(2)
        if not any(np.allclose(pixel, other) for other in seen):
            seen.append(pixel)

    for step in subtasks:
        kind = step["type"]
        if kind in ("move_above", "move_down", "lift"):
            add(step["pixel"])
            if "pixel_2" in step:
                # 两指落点对的第二个落点也收进统计。
                add(step["pixel_2"])
            if "align_point" in step:
                add(step["align_point"])
        elif kind == "reach":
            add(step["pixel"])
        elif kind == "turn":
            add(step["pivot"])
        elif kind == "wipe":
            for pixel in step["path"]:
                add(pixel)
    return seen


def gt_points(task, env, matrices):
    """任务的真值点和它们在 frontview 里的像素、相机深度。只给事后统计用，不进模型，不参与定位。"""
    out = []
    for name, world in task.points(env).items():
        pixel, depth = project_world(world, matrices[DEFAULT_CAMERA])
        item = {
            "name": name,
            "world": np.asarray(world, dtype=float),
            "pixel": np.asarray(pixel, dtype=float),
            "depth": depth,
        }
        out.append(item)
    return out


def gt_pool(gt, role):
    """统计误差时，按角色只和同类的真值点配对。螺母中心和柱顶在画面里可能只差一两个像素。"""
    if role == "bin":
        pool = [item for item in gt if item["name"].endswith("_bin")]
    elif role == "peg":
        pool = [item for item in gt if item["name"].endswith("_peg")]
    elif role == "grip":
        # 两落点的中点是抓取高度，不和物体几何中心或放置目标混配。
        pool = [item for item in gt if item["name"].endswith("_grasp")]
    elif role in ("object", "cube_center", "nut"):
        pool = [item for item in gt if not item["name"].endswith(("_bin", "_peg", "_grasp"))]
    else:
        pool = []
    return pool or gt


def point_group(name):
    """dirt12 归到 dirt，其余名字（Milk_bin、SquareNut_grasp、handle、path_first）不动。"""
    return name.rstrip("0123456789") or name


def closing_yaw(first, second):
    """双落点连线对应的夹爪 yaw；正反两个方向相差 180 度，物理上等价。"""
    axis = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    return float(np.degrees(np.arctan2(-axis[0], axis[1])))


def yaw_error_180(pred, gt):
    """有符号 yaw 误差，折到 [-90, 90)，避免等价朝向被记成 180 度误差。"""
    return (float(pred) - float(gt) + 90.0) % 180.0 - 90.0


def match_targets(points, gt, grip_gt=None):
    """每个定位点配像素距离最近的同类真值点，记录像素和世界误差。

    grip_gt 是可选的落点对真值函数，只用于补充宽度、朝向诊断。
    """
    targets = []
    for point in points:
        pool = gt_pool(gt, point["role"])
        # 两指落点用中点配真值：中点是物体中心；单点步骤没有 center 字段，原样。
        front = point.get("center_pixel", point.get("pixel"))
        world = point.get("center_world", point["world"])

        def metric(item):
            return float(np.linalg.norm(front - item["pixel"]))

        nearest = min(pool, key=metric)
        err = pos_error(world, nearest["world"])
        target = {
            "point": nearest["name"],
            "group": point_group(nearest["name"]),
            "world": [round(float(v), 4) for v in world],
            "gt_world": [round(float(v), 4) for v in nearest["world"]],
            "err_m": {k: round(v, 4) for k, v in err.items()},
            "depth": point["info"],
        }
        if front is not None:
            target["pixel"] = round_pixel(front)
            target["gt_pixel"] = round_pixel(nearest["pixel"])
            target["pixel_err"] = round(float(np.linalg.norm(front - nearest["pixel"])), 2)
        if point["role"] == "grip" and point.get("land_2") is not None:
            first = np.asarray(point["land_1"], dtype=float)
            second = np.asarray(point["land_2"], dtype=float)
            axis = second - first
            width = float(np.linalg.norm(axis[:2]))
            yaw = closing_yaw(first, second)
            target["land_1"] = np.round(first, 4).tolist()
            target["land_2"] = np.round(second, 4).tolist()
            target["grip_width_m"] = round(width, 4)
            target["closing_yaw_deg"] = round(yaw, 2)
            pair = grip_gt(nearest["name"]) if grip_gt is not None else None
            if pair is not None:
                gt_first, gt_second = (np.asarray(item, dtype=float) for item in pair)
                gt_width = float(np.linalg.norm((gt_second - gt_first)[:2]))
                gt_yaw = closing_yaw(gt_first, gt_second)
                yaw_err = yaw_error_180(yaw, gt_yaw)
                target["gt_land_1"] = np.round(gt_first, 4).tolist()
                target["gt_land_2"] = np.round(gt_second, 4).tolist()
                target["gt_grip_width_m"] = round(gt_width, 4)
                target["gt_closing_yaw_deg"] = round(gt_yaw, 2)
                target["grip_width_err_m"] = round(width - gt_width, 4)
                target["grip_width_abs_err_m"] = round(abs(width - gt_width), 4)
                target["closing_yaw_err_deg"] = round(yaw_err, 2)
                target["closing_yaw_abs_err_deg"] = round(abs(yaw_err), 2)
        targets.append(target)
    return targets


def run_episode(env, task, client, rng, episode, image_size, matrices, oracle, style, video_path=None):
    latest = {}
    video = None if video_path is None else EvalVideo(env, image_size)

    def on_step(obs):
        if obs is not None:
            latest["obs"] = obs
        if video is not None:
            video.add(obs)

    record = None
    try:
        obs, hold_ori = prepare_scene(env, task, rng, on_step)
        record = {"episode": episode, "task": task.task, "success": False, "rounds": []}
        for round_index in range(int(task.max_rounds)):
            plan = task.build_plan(env, rng)
            if bool(task.success(env)):
                break
            instruction = task_instruction(task, env, plan, 0, style, False)
            gt_subtasks, _ = render_plan(task, env, plan, 0, matrices, False)
            if not gt_subtasks:
                break
            gt_answer = format_plan(gt_subtasks)
            images = camera_images(obs, image_size)
            views = make_views(env, obs, matrices)
            # 真值只用来统计误差，不进模型，不进定位和控制。
            gt = gt_points(task, env, matrices)
            grip_gt = None
            if hasattr(task, "grip_points"):
                grip_gt = lambda name: task.grip_points(env, name)
            started = time.perf_counter()
            answer = gt_answer if oracle else client.complete(images, instruction)
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
                if video is not None:
                    video.set_plan(subtasks, obs)
                round_rec["steps"] = [step["type"] for step in subtasks]
                round_rec["plan_match"] = plan_matches(subtasks, gt_answer, image_size)
                points = locate_plan(task.task, subtasks, views, image_size)
                round_rec["targets"] = match_targets(points, gt, grip_gt)
                for target in round_rec["targets"]:
                    print(
                        "episode {:02d} {} pixel gt={} pred={} pos gt={} pred={} {}".format(
                            episode,
                            target["point"],
                            format_pixel(target["gt_pixel"]) if target.get("gt_pixel") is not None else "mask",
                            format_pixel(target["pixel"]) if target.get("pixel") is not None else "mask",
                            format_xyz(target["gt_world"]),
                            format_xyz(target["world"]),
                            format_err_axes(target["err_m"]),
                        ),
                        flush=True,
                    )
                    if "gt_grip_width_m" in target:
                        print(
                            "episode {:02d} {} pair width gt={} pred={} err={} "
                            "yaw gt={:.1f}deg pred={:.1f}deg err={:+.1f}deg".format(
                                episode,
                                target["point"],
                                format_cm(target["gt_grip_width_m"]),
                                format_cm(target["grip_width_m"]),
                                format_cm(target["grip_width_err_m"]),
                                target["gt_closing_yaw_deg"],
                                target["closing_yaw_deg"],
                                target["closing_yaw_err_deg"],
                            ),
                            flush=True,
                        )
                starts = set(pick_starts(task.task, subtasks))
                relocated = []
                grasp_diagnostics = []
                grasp_context = {"name": None, "offset": None}

                def grip_name(index):
                    for previous in reversed(subtasks[:index]):
                        world = previous.get("world", {})
                        if "pixel_2" not in world:
                            continue
                        center = 0.5 * (np.asarray(world["pixel"]) + np.asarray(world["pixel_2"]))
                        candidates = [item for item in gt if item["name"].endswith("_grasp")]
                        if not candidates:
                            return None
                        nearest = min(candidates, key=lambda item: np.linalg.norm(center - item["world"]))
                        return nearest["name"][: -len("_grasp")]
                    return None

                def grasp_snapshot(phase, index):
                    name = grasp_context["name"]
                    if name is None or not hasattr(task, "grasp_state"):
                        return
                    state = task.grasp_state(env).get(name)
                    if state is None:
                        return
                    # 直接读物理末端位置；诊断不能调用 arm.update，避免影响后续控制。
                    site = env.robots[0].eef_site_id["right"]
                    eef = np.array(env.sim.data.site_xpos[site], dtype=float)
                    center = np.asarray(state["center"], dtype=float)
                    offset = center - eef
                    entry = {
                        "phase": phase,
                        "step_index": int(index),
                        "object": name,
                        "grasped": bool(state["grasped"]),
                        "eef_pos": np.round(eef, 4).tolist(),
                        "object_pos": np.round(center, 4).tolist(),
                        "align_point_offset_actual": np.round(offset, 4).tolist(),
                    }
                    if phase == "close":
                        grasp_context["offset"] = offset.copy()
                        for later in subtasks[index + 1 :]:
                            if later["type"] == "open_gripper":
                                break
                            predicted = later.get("world", {}).get("align_point")
                            if predicted is not None:
                                predicted_offset = np.asarray(predicted, dtype=float) - eef
                                entry["align_point_offset_pred"] = np.round(predicted_offset, 4).tolist()
                                entry["align_point_offset_err_m"] = round(float(np.linalg.norm(predicted_offset - offset)), 4)
                                break
                    elif phase == "lift" and grasp_context["offset"] is not None:
                        entry["align_point_offset_drift_m"] = round(
                            float(np.linalg.norm(offset - grasp_context["offset"])), 4
                        )
                    elif phase == "pre_place":
                        peg = np.asarray(state["peg"], dtype=float)
                        entry["peg_pos"] = np.round(peg, 4).tolist()
                        entry["object_peg_xy_m"] = round(float(np.linalg.norm(center[:2] - peg[:2])), 4)
                    grasp_diagnostics.append(entry)

                def on_progress(index, step):
                    if step["type"] == "close_gripper":
                        grasp_context["name"] = grip_name(index)
                        grasp_context["offset"] = None
                    elif step["type"] == "move_down" and "align_point" in step:
                        grasp_snapshot("pre_place", index)
                    if index in starts and "obs" in latest:
                        point = relocate(
                            task.task, subtasks, index, make_views(env, latest["obs"], matrices), image_size
                        )
                        if point is not None:
                            relocated.extend(match_targets([point], gt_points(task, env, matrices), grip_gt))

                def on_complete(index, step):
                    if step["type"] == "close_gripper":
                        grasp_snapshot("close", index)
                    elif step["type"] == "lift":
                        grasp_snapshot("lift", index)
                    elif step["type"] == "open_gripper":
                        grasp_context["name"] = None
                        grasp_context["offset"] = None

                if hasattr(task, "grasp_state"):
                    round_rec["grasp_diagnostics"] = grasp_diagnostics
                final, trace = execute_plan(
                    env,
                    subtasks,
                    hold_ori,
                    [point["world"] for point in points],
                    matrices,
                    grip=-1.0,
                    on_step=on_step,
                    on_progress=on_progress,
                    on_complete=on_complete,
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
        if video is not None and video.frames:
            try:
                saved = write_mp4(video_path, video.frames, fps=VIDEO_FPS)
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
            "pixel_err": mean_or_none([t.get("pixel_err") for t in items]),
            "xy_err_m": mean_or_none([t["err_m"]["xy"] for t in items]),
            "z_err_m": mean_or_none([abs(t["err_m"]["z"]) for t in items]),
            "grip_width_abs_err_m": mean_or_none([t.get("grip_width_abs_err_m") for t in items]),
            "closing_yaw_abs_err_deg": mean_or_none([t.get("closing_yaw_abs_err_deg") for t in items]),
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
        for item in rr.get("grasp_diagnostics") or []:
            if item["phase"] == "pre_place":
                print(
                    "  round {} pre_place {} actual_obj_peg_xy={}".format(
                        rr["round"], item["object"], format_cm(item["object_peg_xy_m"])
                    ),
                    flush=True,
                )
            else:
                print(
                    "  round {} {} {} grasped={} align_point_offset_err={} offset_drift={}".format(
                        rr["round"],
                        item["phase"],
                        item["object"],
                        item["grasped"],
                        format_cm(item.get("align_point_offset_err_m")),
                        format_cm(item.get("align_point_offset_drift_m")),
                    ),
                    flush=True,
                )
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
        extra = ""
        if stats.get("grip_width_abs_err_m") is not None:
            extra = "  宽度误差 {}  yaw误差 {:.1f}deg".format(
                format_cm(stats["grip_width_abs_err_m"]), stats["closing_yaw_abs_err_deg"]
            )
        print(
            "{} n={} 像素误差 {}  水平误差 {}  高度误差 {}{}".format(
                name,
                stats["n"],
                "n/a" if stats["pixel_err"] is None else "{:.1f}px".format(stats["pixel_err"]),
                format_cm(stats["xy_err_m"]),
                format_cm(stats["z_err_m"]),
                extra,
            ),
            flush=True,
        )


def add_common_args(parser, default_out):
    parser.add_argument("--model", default="", help="模型名。留空则用服务 /v1/models 的第一个")
    parser.add_argument("--episodes", type=int, default=20, help="测评多少条")
    parser.add_argument("--seed", type=int, default=1000, help="和采集用的 seed 0 错开；传 0 则和采集同批场景")
    parser.add_argument("--fixed-initial", action="store_true", help="每条测评都使用同一个 seed 对应的初始场景")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=2048, help="多物体任务的计划有三十几步，比 Stack 长")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system", default="", help="可选的 system prompt，默认不发")
    parser.add_argument("--text-template", default="{instruction}", help="拼在图片后面的文本，默认就是指令原文")
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--no-video", action="store_true", help="不把每条测评录成 frontview mp4")
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
    matrices = world_to_pixel_matrices(env, image_size)
    print("native image size {}x{} cameras {}".format(image_size, image_size, ",".join(CAMERAS)), flush=True)
    video_dir = None if args.no_video else Path(args.out) / "videos"
    records = []
    try:
        for episode in range(args.episodes):
            if args.fixed_initial:
                rng = reset_episode_rng(env, args.seed)
            style = task.styles[int(rng.integers(len(task.styles)))]
            video_path = None if video_dir is None else video_dir / "ep{:04d}.mp4".format(episode)
            try:
                record = run_episode(
                    env, task, client, rng, episode, image_size, matrices, args.oracle, style, video_path
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
        summary["fixed_initial"] = bool(args.fixed_initial)
        summary["image_size"] = [int(image_size), int(image_size)]
        summary["cameras"] = list(CAMERAS)
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
    assert closing_yaw([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == 0.0
    assert abs(yaw_error_180(-170.0, 170.0) - 20.0) < 1e-9
    assert abs(yaw_error_180(170.0, -170.0) + 20.0) < 1e-9
    # Door：reach 的 target_1、turn 的 pivot 都要收进来
    door = parse_plan(
        format_plan(
            [
                {"type": "reach", "target_1": "[150 240]", "approach": [-0.14, -0.99, 0.0], "standoff": 0.1},
                {"type": "reach", "target_1": "[150 240]", "approach": [-0.14, -0.99, 0.0], "standoff": 0.0},
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
    # Nut：抓螺母的步骤带两指落点对（target_1、target_2、target_kind=grip），
    # 两落点都收；套柱子用 align_point
    nut = parse_plan(
        format_plan(
            [
                {
                    "type": "move_above",
                    "target_1": "[195 305]",
                    "target_2": "[205 315]",
                    "target_kind": "grip",
                    "clearance": 0.1,
                },
                {
                    "type": "move_down",
                    "target_1": "[195 305]",
                    "target_2": "[205 315]",
                    "target_kind": "grip",
                    "offset_z": 0.0,
                },
                {"type": "close_gripper"},
                {
                    "type": "lift",
                    "target_1": "[195 305]",
                    "target_2": "[205 315]",
                    "target_kind": "grip",
                    "delta_z": 0.17,
                },
                {"type": "move_above", "target_1": "[320 280]", "clearance": 0.05, "yaw": 12, "align_point": "[205 312]"},
                {"type": "move_down", "target_1": "[320 280]", "offset_z": -0.165, "align_point": "[205 312]"},
                {"type": "open_gripper"},
                {"type": "move_above", "target_1": "[320 280]", "clearance": 0.06, "align_point": "[205 312]"},
                {"type": "stabilize", "duration": 0.5},
            ]
        ),
        image_size=512,
    )["subtasks"]
    pixels = [tuple(np.round(p, 1)) for p in plan_pixels(nut)]
    assert pixels == [(195.0, 305.0), (205.0, 315.0), (320.0, 280.0), (205.0, 312.0)], pixels
    # Wipe：path 每一点都收
    wipe = parse_plan(
        format_plan(
            [
                {"type": "move_above", "target_1": "[200 300]", "clearance": 0.1, "yaw": 0},
                {"type": "move_down", "target_1": "[200 300]", "offset_z": 0.0},
                {"type": "wipe", "path": ["[200 300]", "[210 305]", "[220 310]"], "offset_z": -0.01},
                {"type": "lift", "target_1": "[220 310]", "delta_z": 0.1},
                {"type": "stabilize", "duration": 0.5},
            ]
        ),
        image_size=512,
    )["subtasks"]
    pixels = [tuple(np.round(p, 1)) for p in plan_pixels(wipe)]
    assert pixels == [(200.0, 300.0), (210.0, 305.0), (220.0, 310.0)], pixels
    print("self-test ok", flush=True)
