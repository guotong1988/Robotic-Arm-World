"""采集 robosuite Lift 数据，用来训练 MLLM。

每条样本对应某一时刻的场景：

- 输入：桌子场景默认机位 frontview 的 RGB 图，再加上一条指令
- 指令：夹爪不在方块正上方时，先移动到目标上方，再抬剩余高度；已经对准后只写还要抬多少
- 输出：一条抓取计划。target 是这一帧方块中心的整数像素，subtasks 是从这一帧接着做的步骤

方块中心来自观测 cube_pos。投到 frontview 上得到像素，x 向右、y 向下，原点在图像左上角。
评测时再用现场的方块深度把预测像素反投影回世界坐标，采集不存深度图。
每条抓取先定一个目标抬升高度。
某一帧如果方块已经抬起了一段，指令和 lift 的 delta_z 写的都是剩下的高度。
例如目标 14 厘米、已经抬起 5 厘米，而且夹爪已在正上方，这一帧就是再抬起 9 厘米，
计划里不再包含 move_above、move_down、close_gripper。夹爪还在侧面时，指令是
「先移动到目标上方，再将方块向上抬起 14 厘米」，计划从 move_above 开始。

输出例子：

    {"task": "grasp", "target": "[282 311]", "subtasks": [{"type": "move_above"}, {"type": "move_down"}, {"type": "close_gripper"}, {"type": "lift", "delta_z": 0.14}, {"type": "stabilize", "duration": 0.5}]}

为了拉开画面差异，方块会撒在桌面一块更大的区域里，每条抓取开始前
手臂还会先移到随机的水平位置、高度和绕竖直轴的朝向。相机仍是 frontview。

    python collect_lift_mllm.py --episodes 50 --out data/lift_mllm
"""

import argparse
import json
import os
from pathlib import Path

# mjpython 会在子线程建窗口，采集时不要用它。渲染后端要在 import robosuite 之前定好。
import gl_backend

gl_backend.configure()

import numpy as np
import robosuite as suite
import robosuite.macros as macros
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import get_camera_transform_matrix
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import mat2quat, quat2axisangle

# 存成正着的图。必须在创建环境之前设置，观测相机会按这个约定渲染。
macros.IMAGE_CONVENTION = "opencv"

# Lift 的默认渲染机位。table arena 里这台相机对着桌面正前方。
DEFAULT_CAMERA = "frontview"
# Panda 默认 OSC 每个控制步的位置、姿态缩放，和 default_panda.json 一致。
POS_SCALE = 0.05
ORI_SCALE = 0.5


def panda_world_osc():
    """默认 Panda 控制器，但把增量改到世界坐标系，方便对准方块。"""
    config = load_composite_controller_config(robot="Panda")
    arm = config["body_parts"]["right"]
    arm["input_ref_frame"] = "world"
    return config


def cube_placement(seed):
    """比 Lift 默认的 ±3 厘米更大，方块会出现在桌面的不同位置。"""
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


def make_env(image_size, seed, camera_depths=False):
    try:
        env = suite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=panda_world_osc(),
            initialization_noise=None,
            placement_initializer=cube_placement(seed),
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=DEFAULT_CAMERA,
            camera_heights=image_size,
            camera_widths=image_size,
            camera_depths=camera_depths,
            reward_shaping=False,
            control_freq=20,
            horizon=2000,
            seed=seed,
        )
    except ImportError as exc:
        text = str(exc)
        if "EGL" in text or "OSMesa" in text or "glGetError" in text:
            raise ImportError(text + "\n\n" + gl_backend.install_hint("离屏渲染初始化失败。")) from exc
        raise
    return env


# 同一条抓取里说法保持一致，变的是还要再抬的高度。
INSTRUCTION_STYLES = (
    ("cm", "将方块抓起 {:.0f} 厘米。", "将方块再抓起 {:.0f} 厘米。"),
    ("m", "将方块抓起 {:.2f} 米。", "将方块再抓起 {:.2f} 米。"),
    ("cm", "将方块向上抬起 {:.0f} 厘米。", "将方块再向上抬起 {:.0f} 厘米。"),
    ("m", "将方块抬高 {:.2f} 米。", "将方块再抬高 {:.2f} 米。"),
)
DONE_INSTRUCTION = "方块已经抬到目标高度。"
# 夹爪末端和方块中心的水平距离小于这个值，算已经在正上方。
ABOVE_XY_TOL = 0.03
# 末端已经降到方块中心附近，就不必再写 move_down。
GRASP_Z_TOL = 0.02
# 抬完后停住的时间，写进 stabilize。
STABILIZE_DURATION = 0.5
# 相对桌面中心。x 正方向远离机械臂，范围故意不对称，避免伸到够不着的地方。
CUBE_X_RANGE = (-0.10, 0.06)
CUBE_Y_RANGE = (-0.12, 0.12)
START_X_RANGE = (-0.16, 0.08)
START_Y_RANGE = (-0.16, 0.16)
START_CLEARANCE = (0.08, 0.20)
HOVER_CLEARANCE = (0.08, 0.15)
YAW_RANGE = (-np.pi / 2.0, np.pi / 2.0)


def remaining_lift(cube_z, rest_z, target_lift):
    """已经抬起的高度，以及离目标还剩多少。都相对放在桌面上的初始高度。"""
    lifted = max(0.0, float(cube_z) - float(rest_z))
    remaining = max(0.0, float(target_lift) - lifted)
    return lifted, remaining


def spoken_height(remaining_m, unit):
    """指令里的数字。厘米取整厘米，米保留两位。"""
    if unit == "cm":
        return int(np.rint(remaining_m * 100.0)) / 100.0
    return round(float(remaining_m), 2)


def rotated_about_z(ori_mat, yaw):
    """绕世界竖直轴转动夹爪，手指朝向变了，仍然朝下抓。"""
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    return rotation @ ori_mat


def sample_start_pos(rng, cube_pos, rest_z):
    """多数时候从方块侧面的不同位置出发，少数时候已经在正上方。"""
    if rng.random() < 0.25:
        xy = cube_pos[:2] + rng.uniform(-0.015, 0.015, size=2)
    else:
        xy = cube_pos[:2]
        for _ in range(20):
            xy = np.array([rng.uniform(*START_X_RANGE), rng.uniform(*START_Y_RANGE)])
            if np.linalg.norm(xy - cube_pos[:2]) >= 0.05:
                break
    clearance = float(rng.uniform(*START_CLEARANCE))
    return np.array([xy[0], xy[1], rest_z + clearance])


def above_cube(eef_pos, cube_pos):
    """末端和方块的水平距离足够近，才算在正上方。"""
    return float(np.linalg.norm(np.asarray(eef_pos)[:2] - np.asarray(cube_pos)[:2])) <= ABOVE_XY_TOL


def instruction_for(style, remaining_m, lifted_m, above):
    unit, fresh, again = style
    spoken = spoken_height(remaining_m, unit)
    if spoken <= 0.0:
        return DONE_INSTRUCTION, 0.0
    # 还没对准时后半句用完整抬升说法，避免「再……再……」。
    template = again if above and lifted_m >= 0.005 else fresh
    amount = spoken * 100.0 if unit == "cm" else spoken
    lift_clause = template.format(amount)
    if above:
        return lift_clause, spoken
    body = lift_clause[:-1] if lift_clause.endswith("。") else lift_clause
    return "先移动到目标上方，再{}。".format(body), spoken


def world_to_pixel_matrix(env, image_size):
    """frontview 的世界坐标到像素的 4x4 矩阵。图像是正方形。"""
    size = int(image_size)
    return get_camera_transform_matrix(
        sim=env.sim,
        camera_name=DEFAULT_CAMERA,
        camera_height=size,
        camera_width=size,
    )


def project_world(world_pos, world_to_pixel):
    """世界点投到像素。返回 (x, y) 和该点在相机坐标系下的深度 z，单位米。

    x 向右，y 向下，原点在左上角，和 opencv 图一致。
    深度是视线方向的 z，不是深度图上的表面距离。同一个点用这个深度反投影，回到的是它自己。
    """
    point = np.ones(4, dtype=float)
    point[:3] = np.asarray(world_pos, dtype=float).reshape(3)
    cam = np.asarray(world_to_pixel, dtype=float) @ point
    depth = float(cam[2])
    if not np.isfinite(depth) or depth <= 1e-6:
        raise ValueError("方块深度无效：{}".format(depth))
    return cam[:2] / depth, depth


def unproject_pixel(pixel_xy, depth, world_to_pixel):
    """像素 (x, y) 配上方块中心的相机深度，还原世界坐标。"""
    u = float(pixel_xy[0])
    v = float(pixel_xy[1])
    z = float(depth)
    if not np.isfinite(u) or not np.isfinite(v) or not np.isfinite(z) or z <= 1e-6:
        raise ValueError("像素或深度无效：pixel={} depth={}".format(pixel_xy, depth))
    cam = np.array([u * z, v * z, z, 1.0], dtype=float)
    world = np.linalg.inv(np.asarray(world_to_pixel, dtype=float)) @ cam
    return np.asarray(world[:3], dtype=float)


def round_pixel(pixel_xy):
    return [int(np.rint(value)) for value in np.asarray(pixel_xy, dtype=float).reshape(-1)[:2]]


def format_pixel(pixel_xy):
    """整数像素，空格分隔。x 向右，y 向下。"""
    coords = " ".join("{:d}".format(value) for value in round_pixel(pixel_xy))
    return "[{}]".format(coords)


def matrix_to_list(matrix):
    return [[float(value) for value in row] for row in np.asarray(matrix, dtype=float)]


def build_subtasks(phase, above, eef_z, cube_z, delta_z):
    """从当前画面接着做的步骤。已经做完的不再写入。"""
    subtasks = []
    if phase != "lift":
        if not above:
            subtasks.append({"type": "move_above"})
        if (not above) or (float(eef_z) - float(cube_z) > GRASP_Z_TOL):
            subtasks.append({"type": "move_down"})
        subtasks.append({"type": "close_gripper"})
    if float(delta_z) >= 0.005:
        subtasks.append({"type": "lift", "delta_z": round(float(delta_z), 2)})
    if subtasks:
        subtasks.append({"type": "stabilize", "duration": STABILIZE_DURATION})
    return subtasks


def format_answer(pixel_xy, cube_pos, eef_pos, phase, above, delta_z):
    """MLLM 要生成的抓取计划。target 是方块中心的像素，delta_z 是还要抬的高度。"""
    plan = {
        "task": "grasp",
        "target": format_pixel(pixel_xy),
        "subtasks": build_subtasks(phase, above, eef_pos[2], np.asarray(cube_pos, dtype=float)[2], delta_z),
    }
    return json.dumps(plan, ensure_ascii=False, separators=(", ", ": "))


def positions_from_obs(obs):
    """方块和夹爪末端都直接读环境观测，不自己推算。"""
    cube_pos = np.array(obs["cube_pos"], dtype=float)
    eef_pos = np.array(obs["robot0_eef_pos"], dtype=float)
    return cube_pos, eef_pos


def save_rgb(path, image):
    image = np.asarray(image, dtype=np.uint8)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, image)
    except ImportError:
        from PIL import Image

        Image.fromarray(image).save(path)


class DatasetWriter:
    def __init__(self, out_dir, image_size, env):
        self.out_dir = Path(out_dir)
        self.image_dir = self.out_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "samples.jsonl"
        self.meta_path = self.out_dir / "meta.json"
        self.image_size = image_size
        self.env = env
        self.world_to_pixel = world_to_pixel_matrix(env, image_size)
        self.samples = []
        self.episodes = []

    def add(self, episode, step, phase, obs, instruction, lift_height, target_lift, lifted, above, success):
        cube_pos, eef_pos = positions_from_obs(obs)
        pixel, _ = project_world(cube_pos, self.world_to_pixel)
        pixel_xy = round_pixel(pixel)
        cube_pos = [round(float(value), 4) for value in cube_pos]
        eef_pos = [round(float(value), 4) for value in eef_pos]
        sample_id = "ep{:04d}_{}_{:04d}".format(episode, phase, step)
        image_name = "{}.png".format(sample_id)
        save_rgb(self.image_dir / image_name, obs[DEFAULT_CAMERA + "_image"])
        record = {
            "id": sample_id,
            "episode": episode,
            "step": int(step),
            "phase": phase,
            "image": "images/{}".format(image_name),
            "camera": DEFAULT_CAMERA,
            "instruction": instruction,
            "above_cube": bool(above),
            "lift_height_m": round(float(lift_height), 4),
            "target_lift_m": round(float(target_lift), 4),
            "lifted_m": round(float(lifted), 4),
            "cube_pos": cube_pos,
            "cube_pixel": pixel_xy,
            "eef_pos": eef_pos,
            "answer": format_answer(pixel_xy, cube_pos, eef_pos, phase, above, lift_height),
            "success": bool(success),
        }
        self.samples.append(record)
        return record

    def finish_episode(self, episode, target_lift, success):
        self.episodes.append(
            {
                "episode": episode,
                "target_lift_m": round(float(target_lift), 4),
                "success": bool(success),
            }
        )

    def close(self):
        with self.jsonl_path.open("w", encoding="utf-8") as handle:
            for record in self.samples:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta = {
            "env": "Lift",
            "robot": "Panda",
            "camera": DEFAULT_CAMERA,
            "image_size": [self.image_size, self.image_size],
            "inputs": ["image", "instruction"],
            "output": {
                "answer": "JSON 字符串。task 固定为 grasp；target 是方块中心的整数像素，形如 \"[x y]\"，x 向右、y 向下，原点在图像左上角；subtasks 是从这一帧还要做的步骤，按顺序可能包含 move_above、move_down、close_gripper、lift（delta_z 为剩余抬升，米）、stabilize（duration 0.5 秒）。已经完成的步骤不写。抬到目标高度后 subtasks 为空。",
            },
            "coordinate_frame": "target 是 frontview 像素；cube_pos 仍是世界坐标，单位米",
            "units": {"target": "pixel", "cube_pos": "meter"},
            "camera_world_to_pixel": matrix_to_list(self.world_to_pixel),
            "position_source": {
                "cube_pos": "obs['cube_pos']，世界坐标，米",
                "cube_pixel": "方块中心经 camera_world_to_pixel 投到图像的整数像素 [x, y]，和 answer.target 一致",
                "eef_pos": "obs['robot0_eef_pos']",
            },
            "instruction": "夹爪不在方块正上方时，指令是先移动到目标上方，再抬剩余高度。对准之后只写剩余高度。lift_height_m 是还要抬的高度，lifted_m 是已经抬起的高度，target_lift_m 是总目标。above_cube 表示末端是否已在方块正上方。方块位置、手臂出发位置、悬停高度和夹爪绕竖直轴的朝向每条抓取都会变。",
            "num_episodes": len(self.episodes),
            "num_samples": len(self.samples),
            "episodes": self.episodes,
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def arm_controller(env):
    return env.robots[0].part_controllers["right"]


def step_toward(env, target_pos, hold_ori, grip):
    arm = arm_controller(env)
    arm.update(force=True)
    delta_pos = target_pos - arm.ref_pos
    pos_action = np.clip(delta_pos / POS_SCALE, -1.0, 1.0)
    delta_rot = hold_ori @ arm.ref_ori_mat.T
    ori_action = np.clip(quat2axisangle(mat2quat(delta_rot)) / ORI_SCALE, -1.0, 1.0)
    action = env.robots[0].create_action_vector(
        {
            "right": np.concatenate([pos_action, ori_action]),
            "right_gripper": np.array([grip]),
        }
    )
    return env.step(action)


def move_until(env, target_pos, hold_ori, grip, pos_tol, max_steps, on_step):
    obs = None
    pos_err = None
    for _ in range(max_steps):
        obs, _, _, _ = step_toward(env, target_pos, hold_ori, grip)
        on_step(obs)
        pos_err = np.linalg.norm(arm_controller(env).ref_pos - target_pos)
        if pos_err < pos_tol:
            return obs, True, pos_err
    return obs, False, pos_err


def hold(env, target_pos, hold_ori, grip, steps, on_step):
    obs = None
    for _ in range(steps):
        obs, _, _, _ = step_toward(env, target_pos, hold_ori, grip)
        on_step(obs)
    return obs


def collect_episode(env, writer, episode, target_lift, style, rng):
    obs = env.reset()
    arm = arm_controller(env)
    arm.update(force=True)
    hold_ori = np.array(arm.ref_ori_mat, dtype=float).copy()
    # 方块刚放下时会悬空一点点，先停住让它落到桌面，再把这个高度当作 0。
    neutral = np.zeros(env.action_dim)
    for _ in range(20):
        obs, _, _, _ = env.step(neutral)
    rest_z = positions_from_obs(obs)[0][2]

    state = {"step": 20, "phase": "initial", "last_key": None, "last_step": None}

    def record(current_obs, phase, force):
        cube_pos, eef_pos = positions_from_obs(current_obs)
        lifted, remaining = remaining_lift(cube_pos[2], rest_z, target_lift)
        above = above_cube(eef_pos, cube_pos)
        instruction, spoken = instruction_for(style, remaining, lifted, above)
        xy_cm = int(np.rint(np.linalg.norm(eef_pos[:2] - cube_pos[:2]) * 100.0))
        z_cm = int(np.rint(eef_pos[2] * 100.0))
        # 靠近看水平距离，下降和上抬看末端高度，避免同一种画面存太多张。
        if phase == "hover":
            bucket = xy_cm
        elif phase in ("grasp", "lift"):
            bucket = z_cm
        else:
            bucket = 0
        key = (phase, spoken, above, bucket)
        if state["last_step"] == state["step"] and state["last_key"] == key:
            return None
        if not force and key == state["last_key"]:
            return None
        sample = writer.add(
            episode,
            state["step"],
            phase,
            current_obs,
            instruction,
            spoken,
            target_lift,
            lifted,
            above,
            success=False,
        )
        state["last_key"] = key
        state["last_step"] = state["step"]
        return sample

    def on_step(new_obs):
        state["step"] += 1
        if state["phase"] in ("hover", "grasp", "lift"):
            record(new_obs, state["phase"], force=False)
        return new_obs

    def tick(_obs):
        state["step"] += 1

    # 先抬高，再平移到随机出发点，避免横着扫过方块。这段不写入数据集。
    state["phase"] = "setup"
    arm_controller(env).update(force=True)
    home = np.array(arm_controller(env).ref_pos)
    safe_z = max(home[2], rest_z + 0.18)
    yaw = float(rng.uniform(*YAW_RANGE))
    hold_ori = rotated_about_z(hold_ori, yaw)
    start = sample_start_pos(rng, positions_from_obs(obs)[0], rest_z)
    move_until(env, np.array([home[0], home[1], safe_z]), hold_ori, grip=-1.0, pos_tol=0.02, max_steps=80, on_step=tick)
    move_until(env, np.array([start[0], start[1], safe_z]), hold_ori, grip=-1.0, pos_tol=0.02, max_steps=160, on_step=tick)
    obs, _, _ = move_until(env, start, hold_ori, grip=-1.0, pos_tol=0.02, max_steps=80, on_step=tick)

    record(obs, "initial", force=True)
    cube_pos, _ = positions_from_obs(obs)

    hover_clearance = float(rng.uniform(*HOVER_CLEARANCE))
    hover = cube_pos + np.array([0.0, 0.0, hover_clearance])
    state["phase"] = "hover"
    obs, hover_ok, hover_err = move_until(
        env, hover, hold_ori, grip=-1.0, pos_tol=0.015, max_steps=160, on_step=on_step
    )
    record(obs, "hover", force=True)

    cube_pos, _ = positions_from_obs(obs)
    grasp = np.array([cube_pos[0], cube_pos[1], cube_pos[2]])
    state["phase"] = "grasp"
    obs, grasp_ok, grasp_err = move_until(
        env, grasp, hold_ori, grip=-1.0, pos_tol=0.012, max_steps=120, on_step=on_step
    )
    record(obs, "grasp", force=True)

    state["phase"] = "close"
    obs = hold(env, grasp, hold_ori, grip=1.0, steps=25, on_step=on_step)
    grasped = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cube)
    record(obs, "close", force=True)

    lift = grasp + np.array([0.0, 0.0, target_lift])
    state["phase"] = "lift"
    obs, lift_ok, lift_err = move_until(
        env, lift, hold_ori, grip=1.0, pos_tol=0.02, max_steps=160, on_step=on_step
    )
    success = bool(env._check_success()) and grasped
    record(obs, "lift", force=True)
    for sample in writer.samples:
        if sample["episode"] == episode:
            sample["success"] = success

    cube_pos, _ = positions_from_obs(obs)
    _, remaining = remaining_lift(cube_pos[2], rest_z, target_lift)
    print(
        "episode {:d} target={:.2f}m remaining={:.3f}m hover={}({:.3f}) grasp={}({:.3f}) grasped={} lift={}({:.3f}) success={} cube_z={:.3f}".format(
            episode,
            target_lift,
            remaining,
            hover_ok,
            hover_err,
            grasp_ok,
            grasp_err,
            grasped,
            lift_ok,
            lift_err,
            success,
            cube_pos[2],
        )
    )
    writer.finish_episode(episode, target_lift, success)
    return success


def main():
    parser = argparse.ArgumentParser(description="采集 Lift / Panda 的 MLLM 图文抓取计划")
    parser.add_argument("--episodes", type=int, default=50, help="采集多少条抓取")
    parser.add_argument("--out", type=str, default="data/lift_mllm", help="输出目录")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--height-min", type=float, default=0.06, help="抬升高度下限，米")
    parser.add_argument("--height-max", type=float, default=0.18, help="抬升高度上限，米")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = make_env(args.image_size, args.seed)
    writer = DatasetWriter(args.out, args.image_size, env)
    successes = 0
    try:
        for episode in range(args.episodes):
            target_lift = float(rng.uniform(args.height_min, args.height_max))
            style = INSTRUCTION_STYLES[int(rng.integers(len(INSTRUCTION_STYLES)))]
            success = collect_episode(env, writer, episode, target_lift, style, rng)
            successes += int(success)
    finally:
        writer.close()
        env.close()

    print(
        "saved {} samples from {} episodes ({} succeeded) to {}".format(
            len(writer.samples), args.episodes, successes, writer.out_dir
        )
    )


if __name__ == "__main__":
    main()
