"""Stack 反向指令实验：训练时都是「把红色方块叠到绿色方块上」，这里改问「把绿色方块叠到红色方块上」。

    python3 eval_stack_reverse_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20
    python3 eval_stack_reverse_mllm_vllm.py --oracle --episodes 5

场景、感知、执行都和 eval_stack_mllm_vllm.py 相同，同一个 --seed 摆出的场景也相同，可以逐条对照。
不同的地方：

1. 指令换成反向说法，也可以用 --instruction 指定一句；指令和采集一样以每台相机一个 <image> 占位开头
2. 成功按反向判定：绿块中心离桌面超过 4cm、和红块接触、夹爪没有抓着绿块。阈值和 robosuite 的 Stack 一样
3. 同时记下原任务是否成功（红块叠在绿块上），看模型是不是无视指令照旧做
4. 按模型合爪前、松爪前指的像素各自离哪个方块近，判断它实际抓了哪个、放到哪个上面
5. --oracle 用反向真值像素和真值步骤，先确认抓绿块、放红块这套动作本身能做成
"""

import argparse
import time
from pathlib import Path

import gl_backend

gl_backend.configure()

import numpy as np

from collect_lift_mllm import (
    BACK_CAMERA,
    DEFAULT_CAMERA,
    MASK,
    add_camera_arg,
    apply_cameras,
    format_pixel,
    grip_view_pixels,
    project_world,
    view_pixel,
    with_views,
    world_to_pixel_matrices,
)
from collect_common import order_landings_at
from collect_stack_mllm import (
    half_z,
    make_env,
    place_offset,
    plan_desc,
    positions_from_obs,
    prepare_scene,
    remaining_subtasks,
)
from eval_lift_mllm_vllm import (
    ServerError,
    VLLMClient,
    camera_images,
    camera_size,
    format_cm,
    grab_frame,
    write_mp4,
)
from depth_locate import make_views
from eval_stack_mllm_vllm import (
    VIDEO_FPS,
    nearest_object,
    perceive,
    plan_matches,
    point_target,
    print_summary,
    summarize,
    write_results,
)
from task_plan import ParseError, check_parser, execute_plan, format_plan, parse_plan

# 和 collect_stack_mllm.INSTRUCTION_STYLES 一样是三种说法，rng 消耗一致，同一 seed 场景不变。
REVERSE_INSTRUCTIONS = (
    "把绿色方块叠到红色方块上。",
    "将绿块抓起，放到红块上面。",
    "抓起绿色方块，叠到红色方块上。",
)
LIFTED_ABOVE_TABLE = 0.04
OBJECT_NAMES = ("red", "green")


def is_holding_green(env):
    return bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeB))


def is_holding_red(env):
    return bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeA))


def reverse_success(env, obs):
    """绿块叠在红块上：绿块离开桌面、和红块接触、已经松开。"""
    _, cube_b, _ = positions_from_obs(obs)
    lifted = float(cube_b[2]) > float(env.table_offset[2]) + LIFTED_ABOVE_TABLE
    touching = bool(env.check_contact(env.cubeB, env.cubeA))
    return lifted and touching and not is_holding_green(env)


def reverse_plan(env, obs, offset, matrices, half, axis, camera_pos):
    """抓绿块、放到红块上的真值步骤。复用正向的步骤生成，只把两块对调。

    抓绿块的步骤写两指落点对（贴绿块两侧），放红块的步骤写红块中心，和正向任务一致。
    被相机挡住、看不到的落点写 "mask"；两台相机都看不到时 frontview 保留像素兜底。
    """
    cube_a, cube_b, eef = positions_from_obs(obs)
    side = np.asarray(axis, dtype=float) * float(half)
    # 两指落点贴被叠的绿块两侧，朝 frontview 的一侧写 target_1。
    land_1, land_2 = order_landings_at(cube_b, camera_pos, (cube_b + side, cube_b - side))
    grip_pixels = grip_view_pixels(env, land_1, land_2, matrices)
    pix_place = view_pixel(env, cube_a, DEFAULT_CAMERA, matrices[DEFAULT_CAMERA], own_body=env.cubeA_body_id)
    pix_place_back = None
    if BACK_CAMERA in matrices:
        pix_place_back = view_pixel(env, cube_a, BACK_CAMERA, matrices[BACK_CAMERA], own_body=env.cubeA_body_id)
        if pix_place == MASK and pix_place_back == MASK:
            pix_place = format_pixel(project_world(cube_a, matrices[DEFAULT_CAMERA])[0])
    steps = remaining_subtasks(
        cube_b, cube_a, eef, grip_pixels, pix_place, float(cube_b[2]), offset, False, False, pix_place_back
    )
    # format_plan 直接返回模型输出格式的 JSON 文本，oracle 时喂给 parse_plan。
    return format_plan(steps)


def step_before(subtasks, kind):
    """kind 这一步之前最近一个带像素的步骤的像素和所在视图。front 被 mask 时回退 backview。"""
    pixel = None
    view = None
    for step in subtasks:
        if step["type"] == kind:
            return pixel, view
        if "pixel" in step:
            if step["pixel"] is not None:
                pixel, view = step["pixel"], DEFAULT_CAMERA
            elif step.get("pixel_back") is not None:
                pixel, view = step["pixel_back"], BACK_CAMERA
    return None, None


def classify_behavior(subtasks, gt_pix, gt_pix_back):
    """按合爪、松爪前的像素判断抓了哪块、放到哪块上。像素在哪个视图就用哪个视图的真值。"""
    pick, pick_view = step_before(subtasks, "close_gripper")
    place, place_view = step_before(subtasks, "open_gripper")

    def nearest(pixel, view):
        if pixel is None:
            return None
        gt = gt_pix if view == DEFAULT_CAMERA else gt_pix_back
        return OBJECT_NAMES[nearest_object(pixel, gt)[0]]

    picked = nearest(pick, pick_view)
    placed_on = nearest(place, place_view)
    if picked == "green" and placed_on == "red":
        behavior = "reverse"
    elif picked == "red" and placed_on == "green":
        behavior = "original"
    else:
        behavior = "other"
    return picked, placed_on, behavior


def run_episode(env, client, rng, episode, image_size, oracle, instruction=None, video_path=None):
    frames = []

    def on_step(obs):
        if video_path is not None and obs is not None:
            frames.append(grab_frame(obs, image_size))

    record = None
    try:
        obs, hold_ori, _rest_z = prepare_scene(env, rng, on_step)
        styled = instruction or REVERSE_INSTRUCTIONS[int(rng.integers(len(REVERSE_INSTRUCTIONS)))]
        # 和采集一致：指令以每台相机一个 <image> 占位开头、话术后面带字段说明，图走 image_url 后占位被剥掉。
        instruction = with_views(styled, plan_desc())
        matrices = world_to_pixel_matrices(env, image_size)
        images = camera_images(obs, image_size)
        views = make_views(env, obs, matrices)
        matrix = matrices[DEFAULT_CAMERA]
        # 评测执行保持 hold_ori，两指合拢轴就是它的第一列。
        axis = np.asarray(hold_ori, dtype=float)[:, 0].copy()
        cam_pos = env.sim.data.cam_xpos[env.sim.model.camera_name2id(DEFAULT_CAMERA)]

        # 以下真值只用来统计，不进模型，不进控制。
        gt_a, gt_b, _ = positions_from_obs(obs)
        gt_pix = [project_world(gt_a, matrix)[0], project_world(gt_b, matrix)[0]]
        gt_pix_back = [None, None]
        if BACK_CAMERA in matrices:
            gt_pix_back = [
                project_world(gt_a, matrices[BACK_CAMERA])[0],
                project_world(gt_b, matrices[BACK_CAMERA])[0],
            ]
        gt_answer = reverse_plan(
            env, obs, place_offset(env), matrices, half_z(env.cubeA, 0.02), axis, cam_pos
        )

        record = {"episode": episode, "instruction": instruction, "success": False, "behavior": None}
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
            record["picked"], record["placed_on"], record["behavior"] = classify_behavior(subtasks, gt_pix, gt_pix_back)
            points = perceive(subtasks, views, image_size, axis=axis)
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
        record["success"] = reverse_success(env, obs)
        record["original_success"] = bool(env._check_success())
        record["holding_green_at_end"] = is_holding_green(env)
        record["holding_red_at_end"] = is_holding_red(env)
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


def summarize_reverse(records):
    out = summarize(records)
    out["original_successes"] = sum(int(bool(r.get("original_success"))) for r in records)
    for behavior in ("reverse", "original", "other"):
        out["behavior_" + behavior] = sum(int(r.get("behavior") == behavior) for r in records)
    return out


def print_episode(record):
    print(
        "episode {:02d} success={} original_success={} behavior={} picked={} placed_on={} steps={} plan_match={} horiz={} zA={:.3f} zB={:.3f}".format(
            record["episode"],
            record["success"],
            record["original_success"],
            record.get("behavior"),
            record.get("picked"),
            record.get("placed_on"),
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


def print_reverse_summary(summary):
    print_summary(summary)
    total = summary["episodes"]
    if total == 0:
        return
    print(
        "原任务（红叠绿）成功 {}/{}    实际动作：绿叠红 {}  红叠绿 {}  其他 {}".format(
            summary["original_successes"],
            total,
            summary["behavior_reverse"],
            summary["behavior_original"],
            summary["behavior_other"],
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Stack 反向指令实验：让模型把绿块叠到红块上")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI 接口地址")
    parser.add_argument("--model", default="", help="模型名。留空则用服务 /v1/models 的第一个")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000, help="默认和 eval_stack_mllm_vllm.py 相同，场景可逐条对照")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system", default="", help="可选的 system prompt，默认不发")
    parser.add_argument("--text-template", default="{instruction}", help="拼在图片后面的文本，默认就是指令原文")
    parser.add_argument("--instruction", default="", help="固定用这句指令。留空则在三种反向说法里随机选")
    parser.add_argument("--out", default="data/stack_reverse_eval")
    parser.add_argument("--no-video", action="store_true", help="不把每条测评的 frontview 录成 mp4")
    parser.add_argument("--oracle", action="store_true", help="用反向真值步骤代替模型，检查抓绿放红能否做成")
    add_camera_arg(parser)
    args = parser.parse_args()

    check_parser()
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
                record = run_episode(
                    env, client, rng, episode, image_size, args.oracle, args.instruction or None, video_path
                )
            except ServerError as exc:
                print(exc, flush=True)
                break
            records.append(record)
            print_episode(record)
    finally:
        summary = summarize_reverse(records)
        summary["task"] = "stack_reverse"
        summary["oracle"] = bool(args.oracle)
        summary["seed"] = args.seed
        summary["backend"] = "oracle" if args.oracle else "vllm"
        summary["instruction"] = args.instruction or list(REVERSE_INSTRUCTIONS)
        if client is not None:
            summary["model"] = client.model
            summary["api_base"] = client.api_base
        jsonl_path, summary_path = write_results(args.out, records, summary)
        env.close()
        print_reverse_summary(summary)
        print("wrote {} and {}".format(jsonl_path, summary_path), flush=True)
        if video_dir is not None:
            print("videos in {}".format(video_dir), flush=True)


if __name__ == "__main__":
    main()
