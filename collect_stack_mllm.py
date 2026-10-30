"""采集 robosuite Stack 的单目 MLLM 数据。

每条成功叠放只保存开局的一帧 frontview RGB。红块和绿块按左右、前后、对角、
一近一远等相对位置摆放，各自再绕竖直轴旋转。失败轨迹不发布。

- 输入：image 和 instruction，指令以一个 <image> 占位开头。
- 抓红块：move_above、move_down、lift 写 target_1、target_2 两指落点，
  target_kind=grip；两个像素都来自 frontview，背侧落点被挡住也按形状写出。
- 放绿块：写绿块中心 target_1，target_kind=cube_center；move_down 的 offset_z
  是相对该中心的高度，move_above 的 clearance 是悬停高度。
- 每个 subtask 带 desc。叠好后 subtasks 为空。

评测使用同一 frontview 的深度把像素还原成世界坐标，再交给 task_plan.execute_plan。
例如：python collect_stack_mllm.py --episodes 50 --out data/stack_mllm
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

from mllm_dataset import finish_image_staging, make_image_staging_dir, make_sample_id, publish_task
from collect_common import order_landings_at
from collect_lift_mllm import (
    ABOVE_XY_TOL,
    CAMERAS,
    DEFAULT_CAMERA,
    GRASP_Z_TOL,
    START_CLEARANCE,
    START_X_RANGE,
    START_Y_RANGE,
    YAW_RANGE,
    arm_controller,
    format_pixel,
    grip_view_pixels,
    hold,
    input_fields,
    matrix_to_list,
    move_until,
    panda_world_osc,
    project_world,
    rotated_about_z,
    round_pixel,
    set_free_xyaw,
    save_rgb,
    with_views,
    world_to_pixel_matrices,
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

# 先验知识：先抬高再横移，放下时再压一点。只决定末端高度，不解关节角。
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
# 两个方块的水平范围，相对桌面中心。x 正方向远离机械臂。
STACK_X = (-0.14, 0.10)
STACK_Y = (-0.18, 0.18)


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
            camera_names=list(CAMERAS),
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


def _in_stack_box(xy):
    return STACK_X[0] <= float(xy[0]) <= STACK_X[1] and STACK_Y[0] <= float(xy[1]) <= STACK_Y[1]


def _stack_delta(rng, mode):
    """绿块相对红块的水平偏移。0 左右，1 前后，2 对角，3 绿块更远，4 斜向分开。"""
    if mode == 0:
        return np.array([rng.uniform(-0.03, 0.03), float(rng.choice([-1.0, 1.0])) * rng.uniform(0.10, 0.22)])
    if mode == 1:
        return np.array([float(rng.choice([-1.0, 1.0])) * rng.uniform(0.10, 0.18), rng.uniform(-0.03, 0.03)])
    if mode == 3:
        return np.array([rng.uniform(0.10, 0.18), rng.uniform(-0.08, 0.08)])
    if mode == 4:
        return np.array(
            [
                float(rng.choice([-1.0, 1.0])) * rng.uniform(0.10, 0.16),
                float(rng.choice([-1.0, 1.0])) * rng.uniform(0.10, 0.16),
            ]
        )
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    dist = float(rng.uniform(0.10, 0.20))
    return dist * np.array([np.cos(angle), np.sin(angle)])


def place_stack_cubes(env, rng):
    """红块和绿块分开摆。绿块是叠放目标，相对红块有左右、前后、对角、一近一远几种摆法。"""
    mode = int(rng.integers(5))
    red, green = np.array([-0.08, -0.10]), np.array([0.02, 0.10])
    for _ in range(60):
        cand_red = np.array([rng.uniform(*STACK_X), rng.uniform(*STACK_Y)])
        delta = _stack_delta(rng, mode)
        cand_green = cand_red + delta
        if _in_stack_box(cand_green) and np.linalg.norm(delta) >= 0.09:
            red, green = cand_red, cand_green
            break
    set_free_xyaw(env, env.cubeA, red, yaw=float(rng.uniform(0.0, 2.0 * np.pi)))
    set_free_xyaw(env, env.cubeB, green, yaw=float(rng.uniform(0.0, 2.0 * np.pi)))
    env.sim.forward()


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


def plan_desc():
    """跟在指令话术后面的字段说明：两指落点和方块中心各写在哪些步骤上。"""
    return (
        "抓红块的步骤里 target_1、target_2 是贴住红块两侧的两指落点的像素，都在正面相机图上；"
        "另一个落点被红块挡住时也根据方块形状和朝向补出它的像素，不写 mask。两落点的中点是抓取点；"
        "放绿块的步骤只写 target_1，target_kind 是 cube_center（方块中心），offset_z 是相对这个中心的高度"
    )


def grip_step(kind, desc, pixels, params):
    """抓红块的步骤：pixels 是 grip_view_pixels 的返回（{"target_1", ...}），target_kind 是 grip。"""
    out = {"type": kind, "desc": desc}
    out.update(pixels)
    out["target_kind"] = "grip"
    out.update(params)
    return out


def place_step(kind, desc, pixel, params):
    """放绿块的步骤：target_1 写方块中心，target_kind 是 cube_center，叠的时候用中心加距离。"""
    out = {"type": kind, "desc": desc, "target_1": pixel, "target_kind": "cube_center"}
    out.update(params)
    return out


def remaining_subtasks(cube_a, cube_b, eef, grip_pixels, pix_b, rest_z, offset, grasped, done):
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
            steps.append(grip_step(
                "move_above", "移到红块正上方", grip_pixels,
                {"clearance": round(float(DEFAULT_CLEARANCE), 2)},
            ))
        if not at_pick:
            steps.append(grip_step(
                "move_down", "竖直降到红块处", grip_pixels, {"offset_z": 0.0},
            ))
        steps.append({"type": "close_gripper", "desc": "合爪夹住红块"})
    # 已经对准绿块并开始下降时，红块高度会掉回运输高度以下，这时不要再插入 lift。
    lift_left = round(float(rest_z) + TRANSPORT_DZ - float(cube_a[2]), 2)
    if lift_left >= 0.01 and not (grasped and above_b):
        steps.append(grip_step(
            "lift", "竖直抬起红块", grip_pixels, {"delta_z": lift_left},
        ))
    # 还没抓住时，即使手臂碰巧在绿块上方，抓起之后也得到绿块上方再放下。
    if not (grasped and above_b):
        steps.append(place_step(
            "move_above", "移到绿块正上方", pix_b,
            {"clearance": round(float(DEFAULT_CLEARANCE), 2)},
        ))
    if not (grasped and at_place):
        steps.append(place_step(
            "move_down", "把红块放到绿块上", pix_b,
            {"offset_z": round(float(offset), 3)},
        ))
    steps.append({"type": "open_gripper", "desc": "松爪放下红块"})
    steps.append(place_step(
        "move_above", "向上抬开夹爪", pix_b,
        {"clearance": round(float(DEFAULT_CLEARANCE), 2)},
    ))
    steps.append({"type": "stabilize", "desc": "停在原地保持稳定", "duration": STABILIZE_DURATION})
    return steps


def plan_for(env, obs, rest_z, offset, matrices, style, grasped, done, half, axis, camera_pos):
    """从当前画面生成标注。

    half 是红块半边长，axis 是合拢轴（末端朝向矩阵的第一列）。两指落点贴住红块两侧，
    target_1 写朝 frontview 那侧，camera_pos 用来排两个落点的先后。
    """
    cube_a, cube_b, eef = positions_from_obs(obs)
    side = np.asarray(axis, dtype=float) * float(half)
    land_1, land_2 = order_landings_at(cube_a, camera_pos, (cube_a + side, cube_a - side))
    grip_pixels = grip_view_pixels(land_1, land_2, matrices)
    pix_b = format_pixel(project_world(cube_b, matrices[DEFAULT_CAMERA])[0])
    subtasks = remaining_subtasks(cube_a, cube_b, eef, grip_pixels, pix_b, rest_z, offset, grasped, done)
    if done:
        instruction = style[2]
    elif grasped:
        instruction = style[1]
    else:
        instruction = style[0]
    plan = {
        "instruction": with_views(instruction, plan_desc()),
        "answer": format_plan(subtasks),
        "subtasks": subtasks,
        "above_pick": xy_close(eef, cube_a),
        "above_place": xy_close(eef, cube_b),
        "cubeA_pixel": round_pixel(project_world(cube_a, matrices[DEFAULT_CAMERA])[0]),
        "cubeB_pixel": round_pixel(project_world(cube_b, matrices[DEFAULT_CAMERA])[0]),
    }
    return plan


def prepare_scene(env, rng, on_step=None):
    """方块落地后，手臂移到随机出发点。这段不交给模型。"""
    if on_step is None:
        on_step = lambda _obs: None
    obs = env.reset()
    place_stack_cubes(env, rng)
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
        self.image_dir = make_image_staging_dir(self.out_dir, "stack")
        self.jsonl_path = self.out_dir / "samples.jsonl"
        self.meta_path = self.out_dir / "meta.json"
        self.image_size = int(image_size)
        self.world_to_pixel = world_to_pixel_matrices(env, image_size)
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
        rgbs = {name: np.array(obs[name + "_image"], copy=True) for name in CAMERAS}
        self.pending.append((record, rgbs))
        return record

    def finish_episode(self, episode, success):
        pending, self.pending = self.pending, []
        if not success:
            self.discarded += 1
            return
        for record, rgbs in pending:
            save_rgb(self.image_dir / Path(record["image"]).name, rgbs[DEFAULT_CAMERA])
            record["success"] = True
            self.samples.append(record)
        self.episodes.append({"episode": episode, "success": True})

    def close(self, publish=True):
        pixels = (
            "move_above、move_down、lift 各自带 target_1 像素，格式是 \"[x y]\"，x 向右、y 向下，对应 image（frontview）。"
            "抓红块的步骤还有 target_2，是另一指落点在同一张图上的像素，被红块挡住也照写。"
        )
        frame = "target_1、target_2 都是 frontview 像素。cubeA_pos、cubeB_pos 是世界坐标，米"
        meta = {
            "env": "Stack",
            "robot": "Panda",
            "camera": DEFAULT_CAMERA,
            "cameras": list(CAMERAS),
            "image_size": [self.image_size, self.image_size],
            "inputs": input_fields(),
            "output": {
                "answer": (
                    "JSON 字符串。subtasks 是从这一帧还要做的步骤，"
                    "每个 subtask 的 desc 用 20 字以内说明这一步做什么；各字段的含义跟在指令话术后面。"
                    "抓红块的 move_above、move_down、lift 带 target_1、target_2，"
                    "是两指贴住红块两侧的落点像素，两落点的中点是抓取位置，target_kind 是 grip；"
                    "放绿块的步骤只带 target_1，target_kind 是 cube_center（方块中心）。"
                    "move_above 有 clearance（米），move_down 的 offset_z 是相对方块中心的高度（米），lift 有 delta_z（米）。"
                    "另外有 close_gripper、open_gripper、stabilize。叠好后 subtasks 为空。" + pixels
                ),
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
            "coordinate_frame": frame,
            "camera_world_to_pixel": {name: matrix_to_list(matrix) for name, matrix in self.world_to_pixel.items()},
            "instruction": "指令以一个 <image> 占位开头，对应 inputs 里的 frontview 图片。话术后面带 answer 各字段的说明。红块还没抓住时，指令是把红色方块叠到绿色方块上。抓住之后改成把拿着的红块放到绿块上。叠好之后说明已经完成。同一条轨迹里用同一套说法。",
            "num_episodes": len(self.episodes),
            "num_discarded_episodes": self.discarded,
            "num_samples": len(self.samples),
            "episodes": self.episodes,
        }
        finish_image_staging(self.out_dir, self.image_dir, publish)
        if publish:
            self.merged_count = publish_task(self.out_dir, self.task, self.samples, meta)


def collect_episode(env, writer, episode, style, rng):
    state = {"step": 0, "phase": "setup", "last_key": None, "last_step": None}
    offset = writer.place_offset_m
    matrix = writer.world_to_pixel
    half = half_z(env.cubeA, 0.02)
    camid = env.sim.model.camera_name2id(DEFAULT_CAMERA)
    camera_pos = np.asarray(env.sim.data.cam_xpos[camid], dtype=float)
    axis = None

    def record(current_obs, phase, force):
        grasped = is_grasped(env)
        done = bool(env._check_success())
        plan = plan_for(env, current_obs, rest_z, offset, matrix, style, grasped, done, half, axis, camera_pos)
        types = tuple(step["type"] for step in plan["subtasks"])
        lifts = tuple(step["delta_z"] for step in plan["subtasks"] if step["type"] == "lift")
        key = (phase, types, lifts)
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
        return new_obs

    def tick(_obs):
        state["step"] += 1

    obs, hold_ori, rest_z = prepare_scene(env, rng, on_step=tick)
    # 合拢轴是末端朝向矩阵的第一列：topdown_ori(yaw) 的合拢方向 = rotation_z(yaw) @ [0 1 0]。
    axis = np.asarray(hold_ori, dtype=float)[:, 0].copy()
    record(obs, "initial", force=True)

    cube_a, _, _ = positions_from_obs(obs)
    hover = np.array([cube_a[0], cube_a[1], cube_a[2] + DEFAULT_CLEARANCE])
    state["phase"] = "hover"
    obs, hover_ok, hover_err = move_until(
        env, hover, hold_ori, -1.0, MOVE_ABOVE_TOL, MOVE_ABOVE_STEPS, on_step
    )
    cube_a, _, _ = positions_from_obs(obs)
    grasp = np.array(cube_a, dtype=float)
    state["phase"] = "grasp"
    obs, grasp_ok, grasp_err = move_until(
        env, grasp, hold_ori, -1.0, MOVE_DOWN_TOL, MOVE_DOWN_STEPS, on_step
    )
    state["phase"] = "close"
    obs = hold(env, grasp, hold_ori, 1.0, CLOSE_STEPS, on_step)
    grasped = is_grasped(env)

    cube_a, cube_b, _ = positions_from_obs(obs)
    lift_pos = np.array([cube_a[0], cube_a[1], rest_z + TRANSPORT_DZ])
    state["phase"] = "lift"
    obs, lift_ok, lift_err = move_until(env, lift_pos, hold_ori, 1.0, 0.02, 180, on_step)
    _, cube_b, _ = positions_from_obs(obs)
    carry = np.array([cube_b[0], cube_b[1], cube_b[2] + DEFAULT_CLEARANCE])
    state["phase"] = "carry"
    obs, carry_ok, carry_err = move_until(
        env, carry, hold_ori, 1.0, MOVE_ABOVE_TOL, MOVE_ABOVE_STEPS, on_step
    )
    _, cube_b, _ = positions_from_obs(obs)
    place = np.array([cube_b[0], cube_b[1], cube_b[2] + offset])
    state["phase"] = "place"
    obs, place_ok, place_err = move_until(
        env, place, hold_ori, 1.0, PLACE_DOWN_TOL, MOVE_DOWN_STEPS, on_step
    )
    state["phase"] = "open"
    obs = hold(env, place, hold_ori, -1.0, OPEN_STEPS, on_step)
    _, cube_b, _ = positions_from_obs(obs)
    retreat = np.array([cube_b[0], cube_b[1], cube_b[2] + DEFAULT_CLEARANCE])
    state["phase"] = "retreat"
    obs, retreat_ok, retreat_err = move_until(
        env, retreat, hold_ori, -1.0, 0.02, 120, on_step
    )
    obs = hold(env, retreat, hold_ori, -1.0, int(round(STABILIZE_DURATION * 20)), on_step)
    success = bool(env._check_success())

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
    parser.add_argument("--episodes", type=int, default=50, help="采够多少条成功叠放才停，失败的不占名额")
    parser.add_argument("--max-attempts", type=int, default=0, help="最多尝试多少条，防止一直失败死循环。0 表示 episodes 的 10 倍")
    parser.add_argument("--out", type=str, default="data/stack_mllm", help="输出目录。可以和 Lift 共用，按 task 合并")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-offset", type=int, default=0, help="首个 episode 编号；并行分片时用不同偏移避免样本 ID 冲突")
    parser.add_argument("--episode-stride", type=int, default=1, help="每次尝试后的 episode 编号步长；并行分片使用 worker 总数")
    parser.add_argument("--image-size", type=int, default=1024)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = make_env(args.image_size, args.seed, use_camera=True)
    writer = DatasetWriter(args.out, args.image_size, env)
    max_attempts = args.max_attempts or args.episodes * 10
    successes = 0
    episode = args.episode_offset
    attempts = 0
    complete = False
    try:
        while successes < args.episodes and attempts < max_attempts:
            style = INSTRUCTION_STYLES[int(rng.integers(len(INSTRUCTION_STYLES)))]
            success = collect_episode(env, writer, episode, style, rng)
            successes += int(success)
            episode += args.episode_stride
            attempts += 1
        complete = successes >= args.episodes
    finally:
        writer.close(publish=complete)
        env.close()
    if not complete:
        raise RuntimeError(
            "达到最大尝试次数 {}，只采到 {} / {} 条成功叠放；未发布，保留原有数据".format(
                max_attempts, successes, args.episodes
            )
        )
    print(
        "saved {} samples from {} succeeded episodes ({} failed, discarded); {} samples in {}".format(
            len(writer.samples), successes, writer.discarded, writer.merged_count, writer.out_dir
        )
    )


if __name__ == "__main__":
    main()
