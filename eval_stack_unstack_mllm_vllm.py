"""Stack 复合指令实验：「把 X 方块叠到 Y 方块上，再拿下来。」

    python3 eval_stack_unstack_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20
    python3 eval_stack_unstack_mllm_vllm.py --oracle --episodes 5

场景、感知、执行都和 eval_stack_mllm_vllm.py 相同，同一个 --seed 摆出的场景也相同。
不同的地方：

1. 指令是叠上去再拿下来。两个方向随机选：红叠绿再拿下、绿叠红再拿下，
   也可以用 --direction 固定一个方向、用 --instruction 固定一句话
2. 真值计划分两段：先按 eval_stack_mllm_vllm.py 的方式叠上去（抓的步骤写两指落点对），
   再降回垫底块两侧重新抓住被叠的块、抬起来放回它原来在桌上的位置、松爪、抬开；
   指令和采集一样以每台相机一个 <image> 占位开头
3. 成功分两步看：执行过程中曾经叠起来过（被叠的块离开桌面、和另一块接触、已经松开），
   并且结束时被叠的块回到桌面高度、没被抓着、两块不再接触。放回位置误差单独记录
4. 按每次合爪、松爪前指的像素各自离哪个方块近，判断它两次分别抓了哪块、放到哪块上面
5. --oracle 用真值像素和真值步骤代替模型，先确认叠上去再拿下来这套动作本身能做成
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
    TRANSPORT_DZ,
    grip_step,
    half_z,
    make_env,
    place_offset,
    place_step,
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
    mean_or_none,
    nearest_object,
    perceive,
    plan_matches,
    point_target,
    print_summary,
    summarize,
    write_results,
)
from task_plan import (
    DEFAULT_CLEARANCE,
    STABILIZE_DURATION,
    ParseError,
    check_parser,
    execute_plan,
    format_plan,
    parse_plan,
)

# rng 消耗方式和 eval_stack_mllm_vllm.py 类似：prepare_scene 之后才抽方向和说法，同一 seed 场景不变。
DIRECTIONS = ("red_on_green", "green_on_red")
INSTRUCTION_STYLES = {
    "red_on_green": (
        "把红色方块叠到绿色方块上，再拿下来。",
        "将红块放到绿块上面，然后再取下来。",
        "抓起红色方块叠到绿色方块上，随后再拿下来。",
    ),
    "green_on_red": (
        "把绿色方块叠到红色方块上，再拿下来。",
        "将绿块放到红块上面，然后再取下来。",
        "抓起绿色方块叠到红色方块上，随后再拿下来。",
    ),
}
LIFTED_ABOVE_TABLE = 0.04
ON_TABLE_Z_TOL = 0.02
OBJECT_NAMES = ("red", "green")


def moved_cube(env, direction):
    return env.cubeA if direction == "red_on_green" else env.cubeB


def is_holding_moved(env, direction):
    return bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=moved_cube(env, direction)))


def stack_state(env, obs, direction):
    """被叠的块压在另一块上：离开桌面、两块接触、已经松开。"""
    cube_a, cube_b, _ = positions_from_obs(obs)
    moved = cube_a if direction == "red_on_green" else cube_b
    lifted = float(moved[2]) > float(env.table_offset[2]) + LIFTED_ABOVE_TABLE
    touching = bool(env.check_contact(env.cubeA, env.cubeB))
    return lifted and touching and not is_holding_moved(env, direction)


def unstack_success(env, obs, direction, orig_moved, stacked_seen):
    """拿下来：被叠的块回到桌面高度、没被抓着、两块分开。前提是之前真的叠起来过。"""
    cube_a, cube_b, _ = positions_from_obs(obs)
    moved = cube_a if direction == "red_on_green" else cube_b
    on_table = abs(float(moved[2]) - float(orig_moved[2])) <= ON_TABLE_Z_TOL
    released = not is_holding_moved(env, direction)
    separated = not bool(env.check_contact(env.cubeA, env.cubeB))
    return bool(stacked_seen and on_table and released and separated)


def unstack_plan(env, obs, offset, matrices, direction, half, axis, camera_pos):
    """叠上去再拿下来的真值步骤。

    前半段复用正向的步骤生成，只把要抓的块和垫底的块按方向对调，去掉结尾的 stabilize；
    后半段降到垫底块两侧（被叠的块此刻正压在垫底块上，贴垫底块两侧就是贴它两侧）
    重新抓住、抬高、放回它原来在桌上的位置。
    像素都来自提问这一帧：叠好的块在垫底块的像素上，原位的像素还空着。
    被相机挡住、看不到的点写 "mask"；两台相机都看不到时 frontview 保留像素兜底。
    """
    cube_a, cube_b, eef = positions_from_obs(obs)
    moved, base = (cube_a, cube_b) if direction == "red_on_green" else (cube_b, cube_a)
    moved_body = env.cubeA_body_id if direction == "red_on_green" else env.cubeB_body_id
    base_body = env.cubeB_body_id if direction == "red_on_green" else env.cubeA_body_id
    side = np.asarray(axis, dtype=float) * float(half)

    def center_pixel(point, own_body):
        pixel = view_pixel(env, point, DEFAULT_CAMERA, matrices[DEFAULT_CAMERA], own_body=own_body)
        pixel_back = None
        if BACK_CAMERA in matrices:
            pixel_back = view_pixel(env, point, BACK_CAMERA, matrices[BACK_CAMERA], own_body=own_body)
            if pixel == MASK and pixel_back == MASK:
                pixel = format_pixel(project_world(point, matrices[DEFAULT_CAMERA])[0])
        return pixel, pixel_back

    # 第一次抓取：两指落点贴被叠块两侧，朝 frontview 的一侧写 target_1。
    land_1, land_2 = order_landings_at(moved, camera_pos, (moved + side, moved - side))
    grip_pixels = grip_view_pixels(env, land_1, land_2, matrices)
    pix_base, pix_base_back = center_pixel(base, base_body)
    # 第二次抓取：落点贴垫底块两侧，两指落到垫底块上那个被叠块的两侧。
    land_3, land_4 = order_landings_at(base, camera_pos, (base + side, base - side))
    grip_re = grip_view_pixels(env, land_3, land_4, matrices)
    # 放回桌上原来的位置：写空位的中心像素，目标中心加距离。
    pix_moved, pix_moved_back = center_pixel(moved, moved_body)
    steps = remaining_subtasks(
        moved, base, eef, grip_pixels, pix_base, float(moved[2]), offset, False, False, pix_base_back
    )
    steps = steps[:-1]  # 去掉叠好后的 stabilize，接着做拿下来
    # 降到被叠块的中心高度，重新抓住；第二次抓取的两指落点贴垫底块两侧。
    steps.append(grip_step("move_down", "降到叠好的块上", grip_re, {"offset_z": round(float(offset), 3)}))
    steps.append({"type": "close_gripper", "desc": "合爪夹住叠好的块"})
    steps.append(grip_step("lift", "把叠好的块抬起来", grip_re, {"delta_z": TRANSPORT_DZ}))
    steps.append(place_step(
        "move_above", "移回它原来的位置上方", pix_moved,
        {"clearance": round(float(DEFAULT_CLEARANCE), 2)}, pix_moved_back,
    ))
    steps.append(place_step("move_down", "把块放回桌面", pix_moved, {"offset_z": 0.0}, pix_moved_back))
    steps.append({"type": "open_gripper", "desc": "松爪放下"})
    steps.append(place_step(
        "move_above", "向上抬开夹爪", pix_moved,
        {"clearance": round(float(DEFAULT_CLEARANCE), 2)}, pix_moved_back,
    ))
    steps.append({"type": "stabilize", "desc": "停在原地保持稳定", "duration": STABILIZE_DURATION})
    # format_plan 直接返回模型输出格式的 JSON 文本，oracle 时喂给 parse_plan。
    return format_plan(steps)


def pixels_before(subtasks, kind):
    """每次 kind 这一步之前，最近一个带像素的步骤的 (像素, 视图)。返回和出现次数一样长的列表。

    front 被 mask 时回退 backview 的像素，视图用来选对应的真值列表。
    """
    out = []
    pixel = None
    view = None
    for step in subtasks:
        if step["type"] == kind:
            out.append((pixel, view))
        if "pixel" in step:
            if step["pixel"] is not None:
                pixel, view = step["pixel"], DEFAULT_CAMERA
            elif step.get("pixel_back") is not None:
                pixel, view = step["pixel_back"], BACK_CAMERA
    return out


def classify(pixels, gt_pix, gt_pix_back):
    out = []
    for pixel, view in pixels:
        if pixel is None:
            out.append(None)
            continue
        gt = gt_pix if view == DEFAULT_CAMERA else gt_pix_back
        out.append(OBJECT_NAMES[nearest_object(pixel, gt)[0]])
    return out


def run_episode(env, client, rng, episode, image_size, oracle, direction, instruction=None, video_path=None):
    frames = []
    chosen = {"direction": None}
    stacked = {"seen": False}
    track = {"enabled": False}

    def on_step(obs):
        if video_path is not None and obs is not None:
            frames.append(grab_frame(obs, image_size))
        if obs is None or not track["enabled"] or stacked["seen"]:
            return
        try:
            if stack_state(env, obs, chosen["direction"]):
                stacked["seen"] = True
        except Exception:
            pass

    record = None
    try:
        obs, hold_ori, _rest_z = prepare_scene(env, rng, on_step)
        if direction == "random":
            direction = DIRECTIONS[int(rng.integers(len(DIRECTIONS)))]
        chosen["direction"] = direction
        styles = INSTRUCTION_STYLES[direction]
        styled = instruction or styles[int(rng.integers(len(styles)))]
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
        orig_moved = np.array(gt_a if direction == "red_on_green" else gt_b, dtype=float)
        orig_base = np.array(gt_b if direction == "red_on_green" else gt_a, dtype=float)
        gt_pix = [project_world(gt_a, matrix)[0], project_world(gt_b, matrix)[0]]
        gt_pix_back = [None, None]
        if BACK_CAMERA in matrices:
            gt_pix_back = [
                project_world(gt_a, matrices[BACK_CAMERA])[0],
                project_world(gt_b, matrices[BACK_CAMERA])[0],
            ]
        gt_answer = unstack_plan(
            env, obs, place_offset(env), matrices, direction, half_z(env.cubeA, 0.02), axis, cam_pos
        )

        record = {
            "episode": episode,
            "direction": direction,
            "instruction": instruction,
            "move_object": "red" if direction == "red_on_green" else "green",
            "base_object": "green" if direction == "red_on_green" else "red",
            "success": False,
            "stacked": False,
        }
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
            record["picked_list"] = classify(pixels_before(subtasks, "close_gripper"), gt_pix, gt_pix_back)
            record["placed_list"] = classify(pixels_before(subtasks, "open_gripper"), gt_pix, gt_pix_back)
            record["picked"] = record["picked_list"][0] if record["picked_list"] else None
            record["placed_on"] = record["placed_list"][0] if record["placed_list"] else None
            points = perceive(subtasks, views, image_size, axis=axis)
            targets = []
            for point in points:
                target, line = point_target(point, gt_pix, gt_pix_back, gt_a, gt_b)
                targets.append(target)
                print("episode {:02d} {}".format(episode, line), flush=True)
            record["targets"] = targets
            track["enabled"] = True
            try:
                final, trace = execute_plan(
                    env,
                    subtasks,
                    hold_ori,
                    [point["world"] for point in points],
                    matrices,
                    grip=-1.0,
                    on_step=on_step,
                )
            finally:
                track["enabled"] = False
            record["trace"] = trace
            if final is not None:
                obs = final
        except ParseError as exc:
            record["error"] = str(exc)
        track["enabled"] = False
        cube_a, cube_b, _ = positions_from_obs(obs)
        moved = cube_a if direction == "red_on_green" else cube_b
        base = cube_b if direction == "red_on_green" else cube_a
        record["stacked"] = stacked["seen"]
        record["success"] = unstack_success(env, obs, direction, orig_moved, stacked["seen"])
        record["holding_moved_at_end"] = is_holding_moved(env, direction)
        record["contact_at_end"] = bool(env.check_contact(env.cubeA, env.cubeB))
        record["return_err_m"] = round(float(np.linalg.norm(moved[:2] - orig_moved[:2])), 4)
        record["base_shift_m"] = round(float(np.linalg.norm(base[:2] - orig_base[:2])), 4)
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


def summarize_unstack(records):
    out = summarize(records)
    out["stacked"] = sum(int(bool(r.get("stacked"))) for r in records)
    out["return_err_m"] = mean_or_none([r.get("return_err_m") for r in records])
    out["base_shift_m"] = mean_or_none([r.get("base_shift_m") for r in records])
    for field, wanted in (("picked_move", "move_object"), ("picked_base", "base_object")):
        out[field] = sum(int(r.get("picked") == r.get(wanted)) for r in records)
    out["picked_other"] = sum(
        int(r.get("picked") not in (None, r.get("move_object"), r.get("base_object"))) for r in records
    )
    out["second_pick_base"] = sum(
        int(len(r.get("picked_list") or []) >= 2 and r["picked_list"][1] == r.get("base_object")) for r in records
    )
    out["two_grasps"] = sum(int(len(r.get("picked_list") or []) >= 2) for r in records)
    return out


def print_episode(record):
    print(
        "episode {:02d} success={} stacked={} direction={} picked={} placed_on={} picks={} places={} plan_match={} return={} base_shift={} horiz={} zA={:.3f} zB={:.3f}".format(
            record["episode"],
            record["success"],
            record["stacked"],
            record.get("direction"),
            record.get("picked"),
            record.get("placed_on"),
            record.get("picked_list"),
            record.get("placed_list"),
            record.get("plan_match"),
            format_cm(record.get("return_err_m")),
            format_cm(record.get("base_shift_m")),
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


def print_unstack_summary(summary):
    print_summary(summary)
    total = summary["episodes"]
    if total == 0:
        return
    print(
        "过程中真的叠起来过 {}/{}    放回误差 {}    垫底块被碰动 {}".format(
            summary["stacked"],
            total,
            format_cm(summary["return_err_m"]),
            format_cm(summary["base_shift_m"]),
        ),
        flush=True,
    )
    print(
        "第一把抓对块 {}  抓了垫底块 {}  其他 {}    写出两次抓取 {}  第二把抓垫底块上的块 {}".format(
            summary["picked_move"],
            summary["picked_base"],
            summary["picked_other"],
            summary["two_grasps"],
            summary["second_pick_base"],
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Stack 复合指令实验：叠上去再拿下来")
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
    parser.add_argument("--instruction", default="", help="固定用这句指令。留空则按方向在三种说法里随机选")
    parser.add_argument(
        "--direction",
        choices=("random",) + DIRECTIONS,
        default="random",
        help="叠放方向。random 则每条随机选红叠绿或绿叠红",
    )
    parser.add_argument("--out", default="data/stack_unstack_eval")
    parser.add_argument("--no-video", action="store_true", help="不把每条测评的 frontview 录成 mp4")
    parser.add_argument("--oracle", action="store_true", help="用真值步骤代替模型，检查叠上去再拿下来能否做成")
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
                    env, client, rng, episode, image_size, args.oracle, args.direction, args.instruction or None, video_path
                )
            except ServerError as exc:
                print(exc, flush=True)
                break
            records.append(record)
            print_episode(record)
    finally:
        summary = summarize_unstack(records)
        summary["task"] = "stack_unstack"
        summary["oracle"] = bool(args.oracle)
        summary["seed"] = args.seed
        summary["backend"] = "oracle" if args.oracle else "vllm"
        summary["direction"] = args.direction
        summary["instruction"] = args.instruction or {k: list(v) for k, v in INSTRUCTION_STYLES.items()}
        if client is not None:
            summary["model"] = client.model
            summary["api_base"] = client.api_base
        jsonl_path, summary_path = write_results(args.out, records, summary)
        env.close()
        print_unstack_summary(summary)
        print("wrote {} and {}".format(jsonl_path, summary_path), flush=True)
        if video_dir is not None:
            print("videos in {}".format(video_dir), flush=True)


if __name__ == "__main__":
    main()
