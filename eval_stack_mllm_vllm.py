"""用微调后的 Qwen 做 Stack 测评，直连 vLLM 的 OpenAI chat completions。

    python3 eval_stack_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20
    python3 eval_stack_mllm_vllm.py --oracle --episodes 5

每条测评做这些事：

1. 按 collect_stack_mllm.py 的方式摆好两个方块，手臂移到随机出发点
2. 只把相机 RGB 和「两个 <image> 分别来自哪台相机 + 把红色方块叠到绿色方块上」放进对话，
   只问一次；指令和采集一样以每台相机一个 <image> 占位开头
3. 模型写出全部步骤：抓哪个像素、抬多高、放到哪个像素、放置高度、松爪、抬开。这些数都来自模型，
   抓红块的步骤给两指落点对（target_1、target_2），放绿块的步骤给绿块中心（target_1）
4. 每个像素用深度相机换成世界坐标：放绿块的中心像素走平面量法（视线和「桌面到顶面中点」
   那个水平面求交，两个视图各自定位再融合，被 mask 的视图不参与）；抓红块的落点对走
   recover_landings 成对还原，取两落点的中点执行。不读仿真里的物体位置，不用真值挑物体
5. 交给 task_plan.execute_plan，末端用世界坐标 OSC 跟踪，执行过程中不再看图、不再问模型
6. 以 robosuite 的 Stack 成功判定为准：红块离开桌面、压在绿块上、夹爪已经松开

仿真真值只用于事后统计误差，不进入模型输入，也不参与控制。
--oracle 用真值像素和真值步骤代替模型，其余流程不变，用来确认深度定位和执行器本身够用。
"""

import argparse
import json
import time
from pathlib import Path

import gl_backend

gl_backend.configure()

import numpy as np

from collect_lift_mllm import (
    BACK_CAMERA,
    DEFAULT_CAMERA,
    add_camera_arg,
    apply_cameras,
    format_pixel,
    project_world,
    round_pixel,
    with_views,
    world_to_pixel_matrices,
)
from collect_stack_mllm import (
    INSTRUCTION_STYLES,
    half_z,
    is_grasped,
    make_env,
    place_offset,
    plan_desc,
    plan_for,
    positions_from_obs,
    prepare_scene,
)
from eval_lift_mllm_vllm import (
    ServerError,
    VLLMClient,
    camera_image,
    camera_images,
    camera_size,
    format_cm,
    format_err_axes,
    format_xyz,
    pos_error,
    write_mp4,
)
from eval_video import EvalVideo
from depth_locate import (
    backproject_disk,
    make_views,
    measure_views,
    ray_at_height,
    recover_landings,
)
from task_plan import SPATIAL, ParseError, check_parser, execute_plan, parse_plan, plausible, resolve_pixel

# 物体顶面取像素附近这个半径里的点，桌面取更大的半径。单位像素，按 512 边长定，其他边长按比例缩放。
OBJECT_RADIUS = 12
TABLE_RADIUS = 40
# 大半径里多数点落在桌面上，取低分位数当桌面高度；小半径里取高分位数当物体顶面。
TABLE_PERCENTILE = 30
TOP_PERCENTILE = 95
# 估出来的物体高度限制在这个范围，米。夹爪或手臂挡在像素附近时不会把目标抬得太离谱。
MIN_HEIGHT = 0.005
MAX_HEIGHT = 0.12
VIDEO_FPS = 20


def locate(pixel, depth, matrix, image_size):
    """模型给的像素换成世界坐标。水平位置完全由像素决定，深度图只给出物体中心的高度。

    放在桌上的物体，中心高度是桌面和顶面的中点。标注里的像素是物体中心的投影，
    所以像素准的时候，视线和这个水平面的交点就是物体中心。
    """
    scale = float(image_size) / 512.0
    obj = backproject_disk(depth, matrix, pixel, max(2, int(round(OBJECT_RADIUS * scale))))
    table = backproject_disk(depth, matrix, pixel, max(4, int(round(TABLE_RADIUS * scale))))
    table_z = float(np.percentile(table[:, 2], TABLE_PERCENTILE))
    top_z = float(np.percentile(obj[:, 2], TOP_PERCENTILE))
    height = float(np.clip(top_z - table_z, MIN_HEIGHT, MAX_HEIGHT))
    world = ray_at_height(pixel, table_z + 0.5 * height, matrix)
    if not plausible(world):
        raise ParseError("反投影超出工作空间：{}".format(np.round(world, 4).tolist()))
    return world, {"table_z": round(table_z, 4), "top_z": round(top_z, 4), "height": round(height, 4)}


def spatial_pixels(subtasks):
    """计划里所有空间步骤的像素组 (front, back, front2, back2)，按第一个可见视图的像素去重后保序。

    front/back 是 target_1 在 frontview/backview 的像素，被 mask 的是 None；
    抓红块的步骤 front2/back2 是 target_2 在这两台相机上的像素，放绿块的步骤两个都是 None。
    """
    seen = []
    keys = set()

    def opt(value):
        return None if value is None else np.asarray(value, dtype=float)

    for step in subtasks:
        if step["type"] not in SPATIAL:
            continue
        front, back = opt(step.get("pixel")), opt(step.get("pixel_back"))
        first = front if front is not None else back
        if first is None:
            continue
        key = tuple(np.round(first, 3))
        if key in keys:
            continue
        keys.add(key)
        seen.append((front, back, opt(step.get("pixel_2")), opt(step.get("pixel_2_back"))))
    return seen


def locate_merged(front, back, views, image_size):
    """测量级融合：两个视图的邻域点云合成一片，量桌面和顶面，视线用 frontview（被 mask 时用另一路）。"""
    usable = [(cam, pix) for cam, pix in ((DEFAULT_CAMERA, front), (BACK_CAMERA, back)) if pix is not None and cam in views]
    # usable 里 frontview 排在最前；front 被 mask 时只剩 backview 一路。
    ray_cam, ray_pixel = usable[0]
    ray_matrix = views[ray_cam][1]

    def disk(radius):
        clouds = [backproject_disk(views[cam][0], views[cam][1], pix, radius) for cam, pix in usable]
        return np.concatenate(clouds) if len(clouds) > 1 else clouds[0]

    scale = float(image_size) / 512.0
    obj = disk(max(2, int(round(OBJECT_RADIUS * scale))))
    table = disk(max(4, int(round(TABLE_RADIUS * scale))))
    table_z = float(np.percentile(table[:, 2], TABLE_PERCENTILE))
    top_z = float(np.percentile(obj[:, 2], TOP_PERCENTILE))
    height = float(np.clip(top_z - table_z, MIN_HEIGHT, MAX_HEIGHT))
    world = ray_at_height(ray_pixel, table_z + 0.5 * height, ray_matrix)
    if not plausible(world):
        raise ParseError("反投影超出工作空间：{}".format(np.round(world, 4).tolist()))
    tag = "+".join(sorted(cam for cam, _ in usable)) if len(usable) > 1 else ray_cam
    return world, {"table_z": round(table_z, 4), "top_z": round(top_z, 4), "height": round(height, 4), "view": tag}


def perceive(subtasks, views, image_size, axis=None):
    """每个像素各自定位：中心像素走平面量法，两指落点对走 recover_landings 成对还原。

    中心像素两条视线一致时合并两路点云做测量级融合，否则逐视图融合。
    融合结果同时写回 step["world"]["pixel"]（落点对还写 world["pixel_2"]，执行取两点中点），
    execute_plan 直接用；返回的点给统计用，也当 execute_plan 兜底用的 landmarks。
    axis 是抓取步骤里两落点的连线方向（世界系），stack 用当前保持的 hold_ori 第一列。
    """
    points = []
    by_key = {}
    for front, back, front2, back2 in spatial_pixels(subtasks):
        if front2 is not None or back2 is not None:
            # 两指落点对：成对还原，两点的中点是抓取位置。
            land_front, land_back, info = recover_landings(
                {"front": front, "front_back": back, "same": front2, "back": back2}, views, image_size, axis
            )
            world = 0.5 * (land_front + land_back)
            center_pixel, _ = project_world(world, views[DEFAULT_CAMERA][1])
            point = {
                "pixel": front,
                "pixel_back": back,
                "center_pixel": center_pixel,
                "center_pixel_back": project_world(world, views[BACK_CAMERA][1])[0] if BACK_CAMERA in views else None,
                "world": world,
                "land_1": land_front,
                "land_2": land_back,
                "info": dict(info, kind="grip"),
            }
            located = {"pixel": land_front, "pixel_2": land_back}
        else:
            # 中心点使用共享的自洽顶面估计：只在中心竖直小柱里找顶面，
            # 不把旁边手臂的高差点直接当方块顶面。object 两视图独立估计后融合，
            # 被手臂污染的一路偏高时由共享规则剔除。
            pixels = {DEFAULT_CAMERA: front, BACK_CAMERA: back}
            world, info = measure_views("object", pixels, views, image_size)
            point = {"pixel": front, "pixel_back": back, "world": world, "info": dict(info, kind="object")}
            located = {"pixel": world}
        points.append(point)
        key_source = front if front is not None else back
        by_key[tuple(np.round(key_source, 3))] = (point, located)
    for step in subtasks:
        if step["type"] not in SPATIAL:
            continue
        value = step.get("pixel")
        if value is None:
            value = step.get("pixel_back")
        if value is None:
            continue
        entry = by_key.get(tuple(np.round(np.asarray(value, dtype=float), 3)))
        if entry is not None:
            step.setdefault("world", {}).update(entry[1])
    return points


def nearest_object(pixel, gt_pixels):
    dists = [float(np.linalg.norm(np.asarray(pixel, dtype=float) - np.asarray(gt, dtype=float))) for gt in gt_pixels]
    return int(np.argmin(dists)), dists


def point_target(point, gt_pix, gt_pix_back, gt_a, gt_b):
    """把一个定位点对照真值方块整理成统计记录。像素误差按「点代表的那个位置」算：
    落点对用两落点中点的像素，中心点用自己的像素；frontview 被 mask 时用 backview 的。"""
    stat_pixel = point.get("center_pixel")
    if stat_pixel is None:
        stat_pixel = point.get("pixel")
    stat_back = point.get("center_pixel_back")
    if stat_back is None:
        stat_back = point.get("pixel_back")
    if stat_pixel is not None:
        index, dists = nearest_object(stat_pixel, gt_pix)
        stat_gt = gt_pix[index]
    else:
        stat_pixel = stat_back
        index, dists = nearest_object(stat_pixel, gt_pix_back)
        stat_gt = gt_pix_back[index]
    gt = gt_a if index == 0 else gt_b
    err = pos_error(point["world"], gt)
    target = {
        "object": "red" if index == 0 else "green",
        "kind": point["info"].get("kind", "object"),
        "pixel": round_pixel(stat_pixel),
        "gt_pixel": round_pixel(stat_gt),
        "pixel_err": round(dists[index], 2),
        "world": [round(float(v), 4) for v in point["world"]],
        "gt_world": [round(float(v), 4) for v in gt],
        "err_m": {k: round(v, 4) for k, v in err.items()},
        "depth": point["info"],
    }
    if stat_back is not None and gt_pix_back[index] is not None:
        delta = np.asarray(stat_back, dtype=float) - np.asarray(gt_pix_back[index], dtype=float)
        target["pixel_back"] = round_pixel(stat_back)
        target["gt_pixel_back"] = round_pixel(gt_pix_back[index])
        target["pixel_back_err"] = round(float(np.linalg.norm(delta)), 2)
    if target["kind"] == "grip" and point.get("land_2") is not None:
        first = np.asarray(point["land_1"], dtype=float)
        second = np.asarray(point["land_2"], dtype=float)
        axis = second - first
        target["land_1"] = np.round(first, 4).tolist()
        target["land_2"] = np.round(second, 4).tolist()
        target["grip_width_m"] = round(float(np.linalg.norm(axis[:2])), 4)
        target["closing_yaw_deg"] = round(float(np.degrees(np.arctan2(-axis[0], axis[1]))), 2)
    line = "{} pixel gt={} pred={} pos gt={} pred={} {}".format(
        target["object"],
        format_pixel(stat_gt),
        format_pixel(stat_pixel),
        format_xyz(gt),
        format_xyz(point["world"]),
        format_err_axes(err),
    )
    return target, line


def plan_matches(pred_steps, gt_answer, image_size):
    """步骤类型一致，放置高度、抬升量也接近标注。"""
    gt_steps = parse_plan(gt_answer, image_size=image_size)["subtasks"]
    if [step["type"] for step in pred_steps] != [step["type"] for step in gt_steps]:
        return False
    for pred, gt in zip(pred_steps, gt_steps):
        if pred["type"] == "move_down" and abs(pred["offset_z"] - gt["offset_z"]) > 0.01:
            return False
        if pred["type"] == "lift" and abs(pred["delta_z"] - gt["delta_z"]) > 0.02:
            return False
    return True


def run_episode(env, client, rng, episode, image_size, oracle, video_path=None):
    video = None if video_path is None else EvalVideo(env, image_size)

    def on_step(obs):
        if video is not None:
            video.add(obs)

    record = None
    try:
        obs, hold_ori, rest_z = prepare_scene(env, rng, on_step)
        style = INSTRUCTION_STYLES[int(rng.integers(len(INSTRUCTION_STYLES)))]
        # 和采集一致：指令以每台相机一个 <image> 占位开头、话术后面带字段说明，图走 image_url 后占位被剥掉。
        instruction = with_views(style[0], plan_desc())
        matrices = world_to_pixel_matrices(env, image_size)
        images = camera_images(obs, image_size)
        views = make_views(env, obs, matrices)
        matrix = matrices[DEFAULT_CAMERA]
        # 评测执行保持 hold_ori，两指合拢轴就是它的第一列。
        axis = np.asarray(hold_ori, dtype=float)[:, 0].copy()

        # 以下真值只用来统计，不进模型，不进控制。
        gt_a, gt_b, _ = positions_from_obs(obs)
        gt_pix = [project_world(gt_a, matrix)[0], project_world(gt_b, matrix)[0]]
        gt_pix_back = [None, None]
        if BACK_CAMERA in matrices:
            gt_pix_back = [
                project_world(gt_a, matrices[BACK_CAMERA])[0],
                project_world(gt_b, matrices[BACK_CAMERA])[0],
            ]
        gt_plan = plan_for(
            env,
            obs,
            rest_z,
            place_offset(env),
            matrices,
            style,
            False,
            False,
            half_z(env.cubeA, 0.02),
            axis,
            env.sim.data.cam_xpos[env.sim.model.camera_name2id(DEFAULT_CAMERA)],
        )
        # format_plan 返回的就是模型输出格式的 JSON 文本，oracle 时直接喂给 parse_plan。
        gt_answer = gt_plan["answer"]

        record = {"episode": episode, "instruction": instruction, "success": False}
        started = time.perf_counter()
        answer = gt_answer if oracle else client.complete(images, instruction)
        record["seconds"] = round(time.perf_counter() - started, 3)
        record["answer"] = answer
        print("episode {:02d} instruction: {}".format(episode, instruction), flush=True)
        print("episode {:02d} answer ({}s):\n{}".format(episode, record["seconds"], answer), flush=True)
        try:
            subtasks = parse_plan(answer, image_size=image_size)["subtasks"]
            record["steps"] = [step["type"] for step in subtasks]
            record["plan_match"] = plan_matches(subtasks, gt_answer, image_size)
            points = perceive(subtasks, views, image_size, axis=axis)
            if video is not None:
                video.set_points(points, obs)
            targets = []
            for point in points:
                target, line = point_target(point, gt_pix, gt_pix_back, gt_a, gt_b)
                targets.append(target)
                print("episode {:02d} {}".format(episode, line), flush=True)
            record["targets"] = targets
            final, trace = execute_plan(
                env,
                subtasks,
                hold_ori,
                [point["world"] for point in points],
                matrices,
                grip=-1.0,
                on_step=on_step,
            )
            record["trace"] = trace
            if final is not None:
                obs = final
        except ParseError as exc:
            record["error"] = str(exc)
        cube_a, cube_b, _ = positions_from_obs(obs)
        record["success"] = bool(env._check_success())
        record["grasped_at_end"] = is_grasped(env)
        record["final_horiz_m"] = round(float(np.linalg.norm(cube_a[:2] - cube_b[:2])), 4)
        record["final_cubeA_z"] = round(float(cube_a[2]), 4)
        record["final_cubeB_z"] = round(float(cube_b[2]), 4)
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


def first_target(record, name):
    for target in record.get("targets") or []:
        if target["object"] == name:
            return target
    return None


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize(records):
    total = len(records)
    out = {
        "episodes": total,
        "successes": sum(int(r["success"]) for r in records),
        "parse_failures": sum(int("steps" not in r) for r in records),
        "plan_matches": sum(int(bool(r.get("plan_match"))) for r in records),
    }
    out["success_rate"] = None if total == 0 else out["successes"] / total
    for name in ("red", "green"):
        hits = [first_target(r, name) for r in records]
        hits = [t for t in hits if t is not None]
        out[name + "_found"] = len(hits)
        out[name + "_pixel_err"] = mean_or_none([t["pixel_err"] for t in hits])
        out[name + "_xy_err_m"] = mean_or_none([t["err_m"]["xy"] for t in hits])
        out[name + "_z_err_m"] = mean_or_none([abs(t["err_m"]["z"]) for t in hits])
    return out


def print_episode(record):
    print(
        "episode {:02d} success={} steps={} plan_match={} horiz={} zA={:.3f} zB={:.3f}".format(
            record["episode"],
            record["success"],
            ">".join(record.get("steps") or []) or "n/a",
            record.get("plan_match"),
            format_cm(record["final_horiz_m"]),
            record["final_cubeA_z"],
            record["final_cubeB_z"],
        ),
        flush=True,
    )
    failed = [step for step in record.get("trace") or [] if not step["ok"]]
    if failed:
        print("  未到位：{}".format(", ".join("{}({})".format(s["type"], format_cm(s["err_m"])) for s in failed)), flush=True)
    if record.get("error"):
        print("  error: {}".format(record["error"]), flush=True)
    if record.get("video"):
        print("  video {}".format(record["video"]), flush=True)


def print_summary(summary):
    total = summary["episodes"]
    if total == 0:
        print("没有完成任何一条测评", flush=True)
        return
    print(
        "成功率 {}/{} = {:.1%}    解析失败 {}    步骤与标注一致 {}/{}".format(
            summary["successes"], total, summary["success_rate"], summary["parse_failures"], summary["plan_matches"], total
        ),
        flush=True,
    )
    for name, label in (("red", "红块"), ("green", "绿块")):
        if not summary[name + "_found"]:
            print("{} 没有指到".format(label), flush=True)
            continue
        print(
            "{} n={} 像素误差 {:.1f}px  水平误差 {}  高度误差 {}".format(
                label,
                summary[name + "_found"],
                summary[name + "_pixel_err"],
                format_cm(summary[name + "_xy_err_m"]),
                format_cm(summary[name + "_z_err_m"]),
            ),
            flush=True,
        )


def write_results(out_dir, records, summary):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonl_path, summary_path


def self_test():
    check_parser()
    matrix = np.array(
        [
            [-300.0, 600.0, 0.0, 256.0],
            [-200.0, 0.0, -560.0, 1000.0],
            [-0.93, 0.0, -0.37, 1.8],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    center = np.array([0.03, -0.05, 0.82])
    pixel, _ = project_world(center, matrix)
    assert np.allclose(ray_at_height(pixel, center[2], matrix), center, atol=1e-9)
    other = np.array([-0.04, 0.08, 0.825])
    landmarks = [center, other]
    for point in landmarks:
        assert np.allclose(resolve_pixel(project_world(point, matrix)[0], landmarks, matrix), point, atol=1e-9)
    print("self-test ok", flush=True)


def main():
    parser = argparse.ArgumentParser(description="直连 vLLM，在 robosuite Stack 上测叠放成功率")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI 接口地址")
    parser.add_argument("--model", default="", help="模型名。留空则用服务 /v1/models 的第一个")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000, help="和采集用的 seed 0 错开，避免同一批初始场景")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system", default="", help="可选的 system prompt，默认不发")
    parser.add_argument("--text-template", default="{instruction}", help="拼在图片后面的文本，默认就是指令原文")
    parser.add_argument("--out", default="data/stack_eval")
    parser.add_argument("--no-video", action="store_true", help="不把每条测评录成并排 mp4")
    parser.add_argument("--oracle", action="store_true", help="用真值步骤代替模型，检查深度定位和执行器")
    parser.add_argument("--self-test", action="store_true", help="只检查解析和坐标换算")
    add_camera_arg(parser)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    client = None
    if args.oracle:
        print("oracle mode, skip model server", flush=True)
    else:
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

    apply_cameras(args)
    rng = np.random.default_rng(args.seed)
    env = make_env(args.image_size, args.seed, use_camera=True, camera_depths=True)
    image_size = camera_size(env)
    video_dir = None if args.no_video else Path(args.out) / "videos"
    records = []
    try:
        for episode in range(args.episodes):
            video_path = None if video_dir is None else video_dir / "ep{:04d}.mp4".format(episode)
            try:
                record = run_episode(env, client, rng, episode, image_size, args.oracle, video_path)
            except ServerError as exc:
                print(exc, flush=True)
                break
            records.append(record)
            print_episode(record)
    finally:
        summary = summarize(records)
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


if __name__ == "__main__":
    main()
