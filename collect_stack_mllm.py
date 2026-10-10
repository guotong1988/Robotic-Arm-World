"""采集 robosuite Stack 数据，用来训练能按任务改步骤的 MLLM。

每条样本的输入仍是 frontview RGB 和一句指令。输出是从这一帧接着做的步骤。
红块还在桌上时，步骤是移到红块上方、下降、合爪、抬高、移到绿块上方、下降、松爪、抬开、停住。
已经抓住并且对准绿块时，前面的步骤不再写。叠好之后 subtasks 为空。

空间步骤各自带像素。评测用提问那一帧离该像素最近的物体中心深度做反投影，
再交给 task_plan.execute_plan。执行器不认识 Stack，只按步骤走。

    python collect_stack_mllm.py --episodes 50 --out data/stack_mllm

叠放失败的轨迹整条丢掉，图片和样本都不写盘。

多个任务可以写进同一个 --out。样本 id 和图片名带 stack_ 前缀，
samples.jsonl 按任务合并，后跑的采集不会盖掉已经写下的其他任务。
"""

import argparse
import json
from pathlib import Path

import gl_backend

gl_backend.configure()

import numpy as np
import robosuite as suite
import robosuite.macros as macros
from robosuite.utils.placement_samplers import UniformRandomSampler

from mllm_dataset import make_sample_id, publish_task
from collect_lift_mllm import (
    ABOVE_XY_TOL,
    DEFAULT_CAMERA,
    GRASP_Z_TOL,
    START_CLEARANCE,
    START_X_RANGE,
    START_Y_RANGE,
    YAW_RANGE,
    arm_controller,
    format_pixel,
    hold,
    matrix_to_list,
    move_until,
    panda_world_osc,
    project_world,
    rotated_about_z,
    round_pixel,
    save_rgb,
    world_to_pixel_matrix,
)
from task_plan import (
    CLOSE_STEPS,
    DEFAULT_CLEARANCE,
    MOVE_ABOVE_STEPS,
    MOVE_ABOVE_TOL,
    MOVE_DOWN_STEPS,
    MOVE_DOWN_TOL,
    OPEN_STEPS,
    PLACE_DOWN_TOL,
    STABILIZE_DURATION,
    format_plan,
)

macros.IMAGE_CONVENTION = "opencv"

# 抓起后先抬到这个高度，再横移到绿块上方，避免带着红块扫过绿块。
TRANSPORT_DZ = 0.10
# 红块中心放到绿块顶面之后再下压一点，让两块接触，指尖仍留在绿块上表面之上。
PLACE_PRESS = 0.004
CUBE_X_RANGE = (-0.08, 0.06)
CUBE_Y_RANGE = (-0.10, 0.10)
INSTRUCTION_STYLES = (
    ("把红色方块叠到绿色方块上。", "把拿着的红色方块放到绿色方块上。", "红色方块已经叠好。"),
    ("将红块抓起，放到绿块上面。", "将红块放到绿块上面。", "红块已经放到绿块上面。"),
    ("抓起红色方块，叠到绿色方块上。", "把红色方块叠放到绿色方块上。", "两个方块已经叠好。"),
)
RECORD_PHASES = ("hover", "grasp", "lift", "carry", "place", "retreat")


def cube_placement(seed):
    return UniformRandomSampler(
        name="ObjectSampler",
        x_range=CUBE_X_RANGE,
        y_range=CUBE_Y_RANGE,
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0.0, 0.0, 0.8),
        z_offset=0.01,
        rng=np.random.default_rng(seed),
    )


def make_env(image_size, seed, use_camera=True, camera_depths=False):
    try:
        env = suite.make(
            env_name="Stack",
            robots="Panda",
            controller_configs=panda_world_osc(),
            initialization_noise=None,
            placement_initializer=cube_placement(seed),
            has_renderer=False,
            has_offscreen_renderer=bool(use_camera),
            ignore_done=True,
            use_camera_obs=bool(use_camera),
            use_object_obs=True,
            camera_names=DEFAULT_CAMERA,
            camera_heights=image_size,
            camera_widths=image_size,
            camera_depths=bool(use_camera and camera_depths),
            reward_shaping=False,
            control_freq=20,
            horizon=4000,
            hard_reset=False,
            seed=seed,
        )
    except ImportError as exc:
        text = str(exc)
        if "EGL" in text or "OSMesa" in text or "glGetError" in text:
            raise ImportError(text + "\n\n" + gl_backend.install_hint("离屏渲染初始化失败。")) from exc
        raise
    return env


def half_z(obj, default):
    size = getattr(obj, "size", None)
    if size is None:
        return float(default)
    return float(np.asarray(size, dtype=float).reshape(-1)[2])


def place_offset(env):
    """绿块中心到红块中心的放置高度。含一点下压，保证接触。"""
    offset = half_z(env.cubeA, 0.02) + half_z(env.cubeB, 0.025) - PLACE_PRESS
    return round(float(offset), 3)


def positions_from_obs(obs):
    cube_a = np.array(obs["cubeA_pos"], dtype=float)
    cube_b = np.array(obs["cubeB_pos"], dtype=float)
    eef = np.array(obs["robot0_eef_pos"], dtype=float)
    return cube_a, cube_b, eef


def xy_close(left, right):
    return float(np.linalg.norm(np.asarray(left)[:2] - np.asarray(right)[:2])) <= ABOVE_XY_TOL


def is_grasped(env):
    return bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeA))


def sample_start_pos(rng, cube_a, cube_b, rest_z):
    """多数时候从两个方块侧面出发，少数时候已经在红块上方。"""
    if rng.random() < 0.25:
        xy = cube_a[:2] + rng.uniform(-0.012, 0.012, size=2)
    else:
        xy = np.array(cube_a[:2], dtype=float)
        for _ in range(30):
            xy = np.array([rng.uniform(*START_X_RANGE), rng.uniform(*START_Y_RANGE)])
            clear_a = np.linalg.norm(xy - cube_a[:2]) >= 0.06
            clear_b = np.linalg.norm(xy - cube_b[:2]) >= 0.06
            if clear_a and clear_b:
                break
    return np.array([xy[0], xy[1], rest_z + float(rng.uniform(*START_CLEARANCE))])


def move_above(pixel, clearance):
    return {"type": "move_above", "target": format_pixel(pixel), "clearance": round(float(clearance), 2)}


def move_down(pixel, offset_z):
    return {"type": "move_down", "target": format_pixel(pixel), "offset_z": round(float(offset_z), 3)}


def lift_step(pixel, delta_z):
    return {"type": "lift", "target": format_pixel(pixel), "delta_z": round(float(delta_z), 2)}


def remaining_subtasks(cube_a, cube_b, eef, pix_a, pix_b, rest_z, offset, grasped, done):
    """从当前画面接着做的步骤。已经完成的不写。"""
    if done:
        return []
    above_a = xy_close(eef, cube_a)
    above_b = xy_close(eef, cube_b)
    at_pick = above_a and float(eef[2]) - float(cube_a[2]) <= GRASP_Z_TOL
    at_place = above_b and float(eef[2]) <= float(cube_b[2]) + float(offset) + GRASP_Z_TOL
    steps = []
    if not grasped:
        if not above_a:
            steps.append(move_above(pix_a, DEFAULT_CLEARANCE))
        if not at_pick:
            steps.append(move_down(pix_a, 0.0))
        steps.append({"type": "close_gripper"})
    # 已经对准绿块并开始下降时，红块高度会掉回运输高度以下，这时不要再插入 lift。
    lift_left = round(float(rest_z) + TRANSPORT_DZ - float(cube_a[2]), 2)
    if lift_left >= 0.01 and not (grasped and above_b):
        steps.append(lift_step(pix_a, lift_left))
    # 还没抓住时，即使手臂碰巧在绿块上方，抓起之后也得到绿块上方再放下。
    if not (grasped and above_b):
        steps.append(move_above(pix_b, DEFAULT_CLEARANCE))
    if not (grasped and at_place):
        steps.append(move_down(pix_b, offset))
    steps.append({"type": "open_gripper"})
    steps.append(move_above(pix_b, DEFAULT_CLEARANCE))
    steps.append({"type": "stabilize", "duration": STABILIZE_DURATION})
    return steps


def plan_for(obs, rest_z, offset, matrix, style, grasped, done):
    cube_a, cube_b, eef = positions_from_obs(obs)
    pix_a, _ = project_world(cube_a, matrix)
    pix_b, _ = project_world(cube_b, matrix)
    subtasks = remaining_subtasks(cube_a, cube_b, eef, pix_a, pix_b, rest_z, offset, grasped, done)
    if done:
        instruction = style[2]
    elif grasped:
        instruction = style[1]
    else:
        instruction = style[0]
    return {
        "instruction": instruction,
        "answer": format_plan(subtasks),
        "subtasks": subtasks,
        "above_pick": xy_close(eef, cube_a),
        "above_place": xy_close(eef, cube_b),
        "cubeA_pixel": round_pixel(pix_a),
        "cubeB_pixel": round_pixel(pix_b),
    }


def prepare_scene(env, rng, on_step=None):
    """方块落地后，手臂移到随机出发点。这段不交给模型。"""
    if on_step is None:
        on_step = lambda _obs: None
    obs = env.reset()
    on_step(obs)
    arm = arm_controller(env)
    arm.update(force=True)
    hold_ori = np.array(arm.ref_ori_mat, dtype=float).copy()
    neutral = np.zeros(env.action_dim)
    for _ in range(20):
        obs, _, _, _ = env.step(neutral)
        on_step(obs)
    cube_a, cube_b, _ = positions_from_obs(obs)
    rest_z = float(cube_a[2])
    home = np.array(arm_controller(env).ref_pos, dtype=float)
    safe_z = max(float(home[2]), rest_z + 0.18, float(cube_b[2]) + 0.18)
    yaw = float(rng.uniform(*YAW_RANGE))
    hold_ori = rotated_about_z(hold_ori, yaw)
    start = sample_start_pos(rng, cube_a, cube_b, rest_z)
    move_until(env, np.array([home[0], home[1], safe_z]), hold_ori, -1.0, 0.02, 80, on_step)
    move_until(env, np.array([start[0], start[1], safe_z]), hold_ori, -1.0, 0.02, 160, on_step)
    obs, _, _ = move_until(env, start, hold_ori, -1.0, 0.02, 80, on_step)
    return obs, hold_ori, rest_z


class DatasetWriter:
    def __init__(self, out_dir, image_size, env):
        self.out_dir = Path(out_dir)
        self.image_dir = self.out_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "samples.jsonl"
        self.meta_path = self.out_dir / "meta.json"
        self.image_size = int(image_size)
        self.world_to_pixel = world_to_pixel_matrix(env, image_size)
        self.place_offset_m = place_offset(env)
        self.task = "stack"
        self.samples = []
        self.episodes = []
        self.pending = []
        self.discarded = 0
        self.merged_count = 0

    def add(self, episode, step, phase, obs, plan):
        """先缓存在内存里，等这条轨迹成功了才写盘。"""
        cube_a, cube_b, eef = positions_from_obs(obs)
        sample_id = make_sample_id(self.task, episode, phase, step)
        image_name = "{}.png".format(sample_id)
        record = {
            "id": sample_id,
            "task": self.task,
            "episode": episode,
            "step": int(step),
            "phase": phase,
            "image": "images/{}".format(image_name),
            "camera": DEFAULT_CAMERA,
            "instruction": plan["instruction"],
            "above_pick": bool(plan["above_pick"]),
            "above_place": bool(plan["above_place"]),
            "cubeA_pos": [round(float(value), 4) for value in cube_a],
            "cubeB_pos": [round(float(value), 4) for value in cube_b],
            "cubeA_pixel": plan["cubeA_pixel"],
            "cubeB_pixel": plan["cubeB_pixel"],
            "eef_pos": [round(float(value), 4) for value in eef],
            "answer": plan["answer"],
            "success": False,
        }
        self.pending.append((record, np.array(obs[DEFAULT_CAMERA + "_image"], copy=True)))
        return record

    def finish_episode(self, episode, success):
        pending, self.pending = self.pending, []
        if not success:
            self.discarded += 1
            return
        for record, rgb in pending:
            save_rgb(self.image_dir / Path(record["image"]).name, rgb)
            record["success"] = True
            self.samples.append(record)
        self.episodes.append({"episode": episode, "success": True})

    def close(self):
        meta = {
            "env": "Stack",
            "robot": "Panda",
            "camera": DEFAULT_CAMERA,
            "image_size": [self.image_size, self.image_size],
            "inputs": ["image", "instruction"],
            "output": {
                "answer": "JSON 字符串，只有 subtasks，是从这一帧还要做的步骤。move_above、move_down、lift 各自带 target 像素，格式是 \"[x y]\"，x 向右、y 向下。move_above 有 clearance（米），move_down 有 offset_z（米），lift 有 delta_z（米）。另外有 close_gripper、open_gripper、stabilize。叠好后 subtasks 为空。",
            },
            "primitives": [
                "move_above",
                "move_down",
                "close_gripper",
                "open_gripper",
                "lift",
                "stabilize",
            ],
            "place_offset_m": self.place_offset_m,
            "transport_dz_m": TRANSPORT_DZ,
            "coordinate_frame": "target 是 frontview 像素。cubeA_pos、cubeB_pos 是世界坐标，米",
            "camera_world_to_pixel": matrix_to_list(self.world_to_pixel),
            "instruction": "红块还没抓住时，指令是把红色方块叠到绿色方块上。抓住之后改成把拿着的红块放到绿块上。叠好之后说明已经完成。同一条轨迹里用同一套说法。",
            "num_episodes": len(self.episodes),
            "num_discarded_episodes": self.discarded,
            "num_samples": len(self.samples),
            "episodes": self.episodes,
        }
        self.merged_count = publish_task(self.out_dir, self.task, self.samples, meta)


def _bucket(phase, eef, cube_a, cube_b):
    if phase == "hover":
        return int(np.rint(np.linalg.norm(eef[:2] - cube_a[:2]) * 100.0))
    if phase == "carry":
        return int(np.rint(np.linalg.norm(eef[:2] - cube_b[:2]) * 100.0))
    if phase in ("grasp", "lift", "place", "retreat"):
        return int(np.rint(float(eef[2]) * 100.0))
    return 0


def collect_episode(env, writer, episode, style, rng):
    state = {"step": 0, "phase": "setup", "last_key": None, "last_step": None}
    offset = writer.place_offset_m
    matrix = writer.world_to_pixel

    def record(current_obs, phase, force):
        grasped = is_grasped(env)
        done = bool(env._check_success())
        plan = plan_for(current_obs, rest_z, offset, matrix, style, grasped, done)
        cube_a, cube_b, eef = positions_from_obs(current_obs)
        types = tuple(step["type"] for step in plan["subtasks"])
        lifts = tuple(step["delta_z"] for step in plan["subtasks"] if step["type"] == "lift")
        key = (phase, types, lifts, _bucket(phase, eef, cube_a, cube_b))
        if state["last_step"] == state["step"] and state["last_key"] == key:
            return None
        if not force and key == state["last_key"]:
            return None
        sample = writer.add(episode, state["step"], phase, current_obs, plan)
        state["last_key"] = key
        state["last_step"] = state["step"]
        return sample

    def on_step(new_obs):
        state["step"] += 1
        if state["phase"] in RECORD_PHASES:
            record(new_obs, state["phase"], force=False)
        return new_obs

    def tick(_obs):
        state["step"] += 1

    obs, hold_ori, rest_z = prepare_scene(env, rng, on_step=tick)
    record(obs, "initial", force=True)

    cube_a, _, _ = positions_from_obs(obs)
    hover = np.array([cube_a[0], cube_a[1], cube_a[2] + DEFAULT_CLEARANCE])
    state["phase"] = "hover"
    obs, hover_ok, hover_err = move_until(
        env, hover, hold_ori, -1.0, MOVE_ABOVE_TOL, MOVE_ABOVE_STEPS, on_step
    )
    record(obs, "hover", force=True)

    cube_a, _, _ = positions_from_obs(obs)
    grasp = np.array(cube_a, dtype=float)
    state["phase"] = "grasp"
    obs, grasp_ok, grasp_err = move_until(
        env, grasp, hold_ori, -1.0, MOVE_DOWN_TOL, MOVE_DOWN_STEPS, on_step
    )
    record(obs, "grasp", force=True)

    state["phase"] = "close"
    obs = hold(env, grasp, hold_ori, 1.0, CLOSE_STEPS, on_step)
    grasped = is_grasped(env)
    record(obs, "close", force=True)

    cube_a, cube_b, _ = positions_from_obs(obs)
    lift_pos = np.array([cube_a[0], cube_a[1], rest_z + TRANSPORT_DZ])
    state["phase"] = "lift"
    obs, lift_ok, lift_err = move_until(env, lift_pos, hold_ori, 1.0, 0.02, 180, on_step)
    record(obs, "lift", force=True)

    _, cube_b, _ = positions_from_obs(obs)
    carry = np.array([cube_b[0], cube_b[1], cube_b[2] + DEFAULT_CLEARANCE])
    state["phase"] = "carry"
    obs, carry_ok, carry_err = move_until(
        env, carry, hold_ori, 1.0, MOVE_ABOVE_TOL, MOVE_ABOVE_STEPS, on_step
    )
    record(obs, "carry", force=True)

    _, cube_b, _ = positions_from_obs(obs)
    place = np.array([cube_b[0], cube_b[1], cube_b[2] + offset])
    state["phase"] = "place"
    obs, place_ok, place_err = move_until(
        env, place, hold_ori, 1.0, PLACE_DOWN_TOL, MOVE_DOWN_STEPS, on_step
    )
    record(obs, "place", force=True)

    state["phase"] = "open"
    obs = hold(env, place, hold_ori, -1.0, OPEN_STEPS, on_step)
    record(obs, "open", force=True)

    _, cube_b, _ = positions_from_obs(obs)
    retreat = np.array([cube_b[0], cube_b[1], cube_b[2] + DEFAULT_CLEARANCE])
    state["phase"] = "retreat"
    obs, retreat_ok, retreat_err = move_until(
        env, retreat, hold_ori, -1.0, 0.02, 120, on_step
    )
    obs = hold(env, retreat, hold_ori, -1.0, int(round(STABILIZE_DURATION * 20)), on_step)
    success = bool(env._check_success())
    record(obs, "retreat", force=True)

    cube_a, cube_b, _ = positions_from_obs(obs)
    horiz = float(np.linalg.norm(cube_a[:2] - cube_b[:2]))
    print(
        "episode {:d} hover={}({:.3f}) grasp={}({:.3f}) grasped={} lift={}({:.3f}) carry={}({:.3f}) place={}({:.3f}) retreat={}({:.3f}) success={} horiz={:.3f} zA={:.3f} zB={:.3f}".format(
            episode,
            hover_ok,
            hover_err,
            grasp_ok,
            grasp_err,
            grasped,
            lift_ok,
            lift_err,
            carry_ok,
            carry_err,
            place_ok,
            place_err,
            retreat_ok,
            retreat_err,
            success,
            horiz,
            cube_a[2],
            cube_b[2],
        )
    )
    writer.finish_episode(episode, success)
    return success


def main():
    parser = argparse.ArgumentParser(description="采集 Stack / Panda 的 MLLM 图文步骤")
    parser.add_argument("--episodes", type=int, default=50, help="采集多少条叠放")
    parser.add_argument("--out", type=str, default="data/stack_mllm", help="输出目录。可以和 Lift 共用，按 task 合并")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = make_env(args.image_size, args.seed, use_camera=True)
    writer = DatasetWriter(args.out, args.image_size, env)
    successes = 0
    try:
        for episode in range(args.episodes):
            style = INSTRUCTION_STYLES[int(rng.integers(len(INSTRUCTION_STYLES)))]
            success = collect_episode(env, writer, episode, style, rng)
            successes += int(success)
    finally:
        writer.close()
        env.close()
    print(
        "saved {} samples from {} succeeded episodes ({} failed, discarded); {} samples in {}".format(
            len(writer.samples), successes, writer.discarded, writer.merged_count, writer.out_dir
        )
    )


if __name__ == "__main__":
    main()
