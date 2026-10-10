"""用微调后的 Qwen 做 Stack 测评，直连 vLLM 的 OpenAI chat completions。

    python3 eval_stack_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20
    python3 eval_stack_mllm_vllm.py --oracle --episodes 5

每条测评做这些事：

1. 按 collect_stack_mllm.py 的方式摆好两个方块，手臂移到随机出发点
2. 只把一张 frontview RGB 和一句「把红色方块叠到绿色方块上」放进对话，只问一次
3. 模型写出全部步骤：抓哪个像素、抬多高、放到哪个像素、放置高度、松爪、抬开。这些数都来自模型
4. 每个像素用深度相机换成世界坐标：像素决定视线，深度图只用来估物体中心离桌面多高，
    再取视线和这个高度的水平面的交点。不读仿真里的物体位置，不用真值挑物体
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
from robosuite.utils.camera_utils import get_real_depth_map

from collect_lift_mllm import (
    DEFAULT_CAMERA,
    format_pixel,
    project_world,
    round_pixel,
    unproject_pixel,
    world_to_pixel_matrix,
)
from collect_stack_mllm import (
    INSTRUCTION_STYLES,
    is_grasped,
    make_env,
    place_offset,
    plan_for,
    positions_from_obs,
    prepare_scene,
)
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


def real_depth(env, obs):
    raw = np.asarray(obs[DEFAULT_CAMERA + "_depth"], dtype=float)
    depth = np.asarray(get_real_depth_map(env.sim, raw), dtype=float)
    return depth.reshape(depth.shape[0], depth.shape[1])


def backproject_disk(depth, matrix, pixel, radius):
    """深度图上以 pixel 为圆心、radius 为半径的像素，反投影成世界坐标点。"""
    height, width = depth.shape
    u0, v0 = int(np.rint(pixel[0])), int(np.rint(pixel[1]))
    if not (0 <= u0 < width and 0 <= v0 < height):
        raise ParseError("像素不在画面里：{}".format(format_pixel(pixel)))
    us = np.arange(max(0, u0 - radius), min(width, u0 + radius + 1))
    vs = np.arange(max(0, v0 - radius), min(height, v0 + radius + 1))
    uu, vv = np.meshgrid(us, vs)
    keep = (uu - u0) ** 2 + (vv - v0) ** 2 <= radius * radius
    uu, vv = uu[keep], vv[keep]
    z = depth[vv, uu]
    # 深度图第 (v, u) 格对应的是像素中心 (u + 0.5, v + 0.5)。
    cu, cv = uu + 0.5, vv + 0.5
    cam = np.stack([cu * z, cv * z, z, np.ones_like(z)])
    world = np.linalg.inv(np.asarray(matrix, dtype=float)) @ cam
    return world[:3].T


def ray_at_height(pixel, height_z, matrix):
    """像素对应的视线和水平面 z=height_z 的交点。"""
    near = unproject_pixel(pixel, 0.5, matrix)
    far = unproject_pixel(pixel, 3.0, matrix)
    direction = far - near
    if abs(float(direction[2])) < 1e-6:
        raise ParseError("视线接近水平，无法求交")
    t = (float(height_z) - float(near[2])) / float(direction[2])
    return near + t * direction


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
    """计划里所有空间步骤的像素，去重后保持顺序。"""
    seen = []
    for step in subtasks:
        if step["type"] not in SPATIAL:
            continue
        pixel = np.asarray(step["pixel"], dtype=float)
        if not any(np.allclose(pixel, other) for other in seen):
            seen.append(pixel)
    return seen


def perceive(subtasks, depth, matrix, image_size):
    """每个像素各自定位。返回的点就是 execute_plan 的 landmarks。

    resolve_pixel 取离像素最近的 landmark 的相机深度来反投影。这里每个 landmark 正好落在
    自己像素的视线上，所以每一步反投影回到的就是为该像素定出的点，不会借用别的物体。
    """
    points = []
    for pixel in spatial_pixels(subtasks):
        world, info = locate(pixel, depth, matrix, image_size)
        points.append({"pixel": pixel, "world": world, "info": info})
    return points


def nearest_object(pixel, gt_pixels):
    dists = [float(np.linalg.norm(np.asarray(pixel, dtype=float) - np.asarray(gt, dtype=float))) for gt in gt_pixels]
    return int(np.argmin(dists)), dists


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
    frames = []

    def on_step(obs):
        if video_path is not None and obs is not None:
            frames.append(grab_frame(obs, image_size))

    record = None
    try:
        obs, hold_ori, rest_z = prepare_scene(env, rng, on_step)
        style = INSTRUCTION_STYLES[int(rng.integers(len(INSTRUCTION_STYLES)))]
        instruction = style[0]
        matrix = world_to_pixel_matrix(env, image_size)
        image = camera_image(obs, image_size)
        depth = real_depth(env, obs)

        # 以下真值只用来统计，不进模型，不进控制。
        gt_a, gt_b, _ = positions_from_obs(obs)
        gt_pix = [project_world(gt_a, matrix)[0], project_world(gt_b, matrix)[0]]
        gt_answer = plan_for(obs, rest_z, place_offset(env), matrix, style, False, False)["answer"]

        record = {"episode": episode, "instruction": instruction, "success": False}
        started = time.perf_counter()
        answer = gt_answer if oracle else client.complete(image, instruction)
        record["seconds"] = round(time.perf_counter() - started, 3)
        record["answer"] = answer
        print("episode {:02d} instruction: {}".format(episode, instruction), flush=True)
        print("episode {:02d} answer ({}s):\n{}".format(episode, record["seconds"], answer), flush=True)
        try:
            subtasks = parse_plan(answer, image_size=image_size)["subtasks"]
            record["steps"] = [step["type"] for step in subtasks]
            record["plan_match"] = plan_matches(subtasks, gt_answer, image_size)
            points = perceive(subtasks, depth, matrix, image_size)
            targets = []
            for point in points:
                index, dists = nearest_object(point["pixel"], gt_pix)
                gt = gt_a if index == 0 else gt_b
                err = pos_error(point["world"], gt)
                targets.append(
                    {
                        "object": "red" if index == 0 else "green",
                        "pixel": round_pixel(point["pixel"]),
                        "gt_pixel": round_pixel(gt_pix[index]),
                        "pixel_err": round(dists[index], 2),
                        "world": [round(float(v), 4) for v in point["world"]],
                        "gt_world": [round(float(v), 4) for v in gt],
                        "err_m": {k: round(v, 4) for k, v in err.items()},
                        "depth": point["info"],
                    }
                )
                print(
                    "episode {:02d} {} pixel gt={} pred={} pos gt={} pred={} {}".format(
                        episode,
                        targets[-1]["object"],
                        format_pixel(gt_pix[index]),
                        format_pixel(point["pixel"]),
                        format_xyz(gt),
                        format_xyz(point["world"]),
                        format_err_axes(err),
                    ),
                    flush=True,
                )
            record["targets"] = targets
            final, trace = execute_plan(
                env,
                subtasks,
                hold_ori,
                [point["world"] for point in points],
                matrix,
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
    parser.add_argument("--no-video", action="store_true", help="不把每条测评的 frontview 录成 mp4")
    parser.add_argument("--oracle", action="store_true", help="用真值步骤代替模型，检查深度定位和执行器")
    parser.add_argument("--self-test", action="store_true", help="只检查解析和坐标换算")
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
