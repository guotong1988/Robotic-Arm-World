"""Stack 反向指令实验：训练时都是「把红色方块叠到绿色方块上」，这里改问「把绿色方块叠到红色方块上」。

    python3 eval_stack_reverse_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20
    python3 eval_stack_reverse_mllm_vllm.py --oracle --episodes 5

场景、感知、执行都和 eval_stack_mllm_vllm.py 相同，同一个 --seed 摆出的场景也相同，可以逐条对照。
不同的地方：

1. 指令换成反向说法，也可以用 --instruction 指定一句
2. 成功按反向判定：绿块中心离桌面超过 4cm、和红块接触、夹爪没有抓着绿块。阈值和 robosuite 的 Stack 一样
3. 同时记下原任务是否成功（红块叠在绿块上），看模型是不是无视指令照旧做
4. 按模型合爪前、松爪前指的像素各自离哪个方块近，判断它实际抓了哪个、放到哪个上面
5. --oracle 用反向真值步骤，先确认抓绿块、放红块这套动作本身能做成
"""

import argparse
import time
from pathlib import Path

import gl_backend

gl_backend.configure()

import numpy as np

from collect_lift_mllm import format_pixel, project_world, round_pixel, world_to_pixel_matrix
from collect_stack_mllm import (
    make_env,
    place_offset,
    positions_from_obs,
    prepare_scene,
    remaining_subtasks,
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
from eval_stack_mllm_vllm import (
    VIDEO_FPS,
    nearest_object,
    perceive,
    plan_matches,
    print_summary,
    real_depth,
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


def reverse_plan(obs, offset, matrix):
    """抓绿块、放到红块上的真值步骤。复用正向的步骤生成，只把两块对调。"""
    cube_a, cube_b, eef = positions_from_obs(obs)
    pix_a, _ = project_world(cube_a, matrix)
    pix_b, _ = project_world(cube_b, matrix)
    steps = remaining_subtasks(cube_b, cube_a, eef, pix_b, pix_a, float(cube_b[2]), offset, False, False)
    return format_plan(steps)


def step_before(subtasks, kind):
    """kind 这一步之前最近一个带像素的步骤的像素。"""
    pixel = None
    for step in subtasks:
        if step["type"] == kind:
            return pixel
        if "pixel" in step:
            pixel = step["pixel"]
    return None


def classify_behavior(subtasks, gt_pix):
    """按合爪、松爪前的像素判断抓了哪块、放到哪块上。"""
    pick = step_before(subtasks, "close_gripper")
    place = step_before(subtasks, "open_gripper")
    picked = None if pick is None else OBJECT_NAMES[nearest_object(pick, gt_pix)[0]]
    placed_on = None if place is None else OBJECT_NAMES[nearest_object(place, gt_pix)[0]]
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
        styled = REVERSE_INSTRUCTIONS[int(rng.integers(len(REVERSE_INSTRUCTIONS)))]
        instruction = instruction or styled
        matrix = world_to_pixel_matrix(env, image_size)
        image = camera_image(obs, image_size)
        depth = real_depth(env, obs)

        # 以下真值只用来统计，不进模型，不进控制。
        gt_a, gt_b, _ = positions_from_obs(obs)
        gt_pix = [project_world(gt_a, matrix)[0], project_world(gt_b, matrix)[0]]
        gt_answer = reverse_plan(obs, place_offset(env), matrix)

        record = {"episode": episode, "instruction": instruction, "success": False, "behavior": None}
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
            record["picked"], record["placed_on"], record["behavior"] = classify_behavior(subtasks, gt_pix)
            points = perceive(subtasks, depth, matrix, image_size)
            targets = []
            for point in points:
                index, dists = nearest_object(point["pixel"], gt_pix)
                gt = gt_a if index == 0 else gt_b
                err = pos_error(point["world"], gt)
                targets.append(
                    {
                        "object": OBJECT_NAMES[index],
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
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system", default="", help="可选的 system prompt，默认不发")
    parser.add_argument("--text-template", default="{instruction}", help="拼在图片后面的文本，默认就是指令原文")
    parser.add_argument("--instruction", default="", help="固定用这句指令。留空则在三种反向说法里随机选")
    parser.add_argument("--out", default="data/stack_reverse_eval")
    parser.add_argument("--no-video", action="store_true", help="不把每条测评的 frontview 录成 mp4")
    parser.add_argument("--oracle", action="store_true", help="用反向真值步骤代替模型，检查抓绿放红能否做成")
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
