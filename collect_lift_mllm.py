"""采集 robosuite Lift 数据，用来训练 MLLM。

每条样本对应某一时刻的场景：

- 输入：一张 frontview RGB，指令以一个 <image> 占位开头
- 指令：夹爪不在方块正上方时，先移动到目标上方，再抬剩余高度；已经对准后只写还要抬多少。
  话术后面带字段说明（target_1、target_2 这些字段各是什么），模型按说明写 answer
- 输出：一条抓取计划。每个 subtask 带一个 desc，用 20 字以内说明这一步做什么。
  要动位置的步骤带两个夹爪落点的像素，不带深度

落点是两指合拢时贴住方块表面的那两点。两指沿世界 x 轴合拢，一个朝 frontview，一个朝对面。
target_1 是夹爪1（朝 frontview 那侧）的落点，target_2 是夹爪2（朝后）的落点。
两个落点都写在正面相机图上。像素 x 向右、y 向下，原点在图像左上角。
被挡住的落点也根据方块形状和朝向写出像素，不使用 mask。
评测时模型按同样的格式给出像素，用 frontview 深度还原后做逆运动学。

每条抓取先定一个目标抬升高度。
某一帧如果方块已经抬起了一段，指令和 lift 的 delta_z 写的都是剩下的高度。
例如目标 14 厘米、已经抬起 5 厘米，而且夹爪已在正上方，这一帧就是再抬起 9 厘米，
计划里不再包含 move_above、move_down、close_gripper。夹爪还在侧面时，指令是
「<image> 先移动到目标上方，再将方块向上抬起 14 厘米」，计划从 move_above 开始。

输入例子。样本只有 image 字段，一个占位后面接话术和字段说明：

    <image>
    先移动到目标上方，再将方块向上抬起 14 厘米。回答的 JSON 中，target_1 是夹爪1落点的像素，target_2 是夹爪2落点的像素，都在正面相机图上，两落点的中点是抓取位置。

输出例子（还没抓起、夹爪还在侧面时）：

    {"subtasks": [{"type": "move_above", "desc": "移到方块正上方", "target_1": "[282 311]", "target_2": "[302 311]", "target_kind": "grip"}, {"type": "move_down", "desc": "竖直降到方块处", "target_1": "[282 311]", "target_2": "[302 311]", "target_kind": "grip"}, {"type": "close_gripper", "desc": "合爪夹住方块"}, {"type": "lift", "desc": "竖直向上抬起方块", "target_1": "[282 311]", "target_2": "[302 311]", "target_kind": "grip", "delta_z": 0.14}, {"type": "stabilize", "desc": "停在原地保持稳定", "duration": 0.5}]}

每条成功抓取只存开局那一帧（phase 为 initial），不存悬停、下降、合爪、上抬这些中途画面。
泛化放在开局：方块抽到桌面的不同区域，朝向绕竖直轴随机转；抬升高度在低、中、高三档里抽。
手臂在写入这一帧之前会先移到随机的水平位置和高度。夹爪竖直朝下，两指沿世界 x 轴合拢。

例如凑满 50 条成功抓取再停，图片和样本写到 data/lift_mllm：

python collect_lift_mllm.py --episodes 50 --out data/lift_mllm

抓取失败（没抓住或没抬到目标高度）的轨迹整条丢掉，图片和样本都不写盘。

多个任务可以写进同一个 --out。样本 id 和图片名带 lift_ 前缀，
samples.jsonl 按任务合并，后跑的采集不会盖掉已经写下的其他任务。
"""

import argparse
import json
import os
from pathlib import Path

from mllm_dataset import finish_image_staging, make_image_staging_dir, make_sample_id, publish_task

# mjpython 会在子线程建窗口，采集时不要用它。渲染后端要在 import robosuite 之前定好。
import gl_backend

gl_backend.configure()

import mujoco
import numpy as np
import robosuite as suite
import robosuite.macros as macros
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import get_camera_transform_matrix
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import mat2quat, quat2axisangle

from grip_ik import GRIP_YAW, box_contacts, grasp_eef, joint_position_config, pad_offset, topdown_ori

# 存成正着的图。必须在创建环境之前设置，观测相机会按这个约定渲染。
macros.IMAGE_CONVENTION = "opencv"

DEFAULT_CAMERA = "frontview"
CAMERAS = [DEFAULT_CAMERA]
# 原机位沿朝向桌面的方向平移：抬高 15 cm，同时向目标靠近 15 cm，保持俯角。
FRONT_CAMERA_POS = np.array([0.75917886, 0.0, 1.91035886])
FRONT_CAMERA_LOOKAT = np.array([0.0, 0.0, 0.9125])
FRONT_CAMERA_FOVY = 45.0


def image_prefix():
    return "<image>\n"


def with_views(clause, note=None):
    """把任务话术包上图片占位，作为完整指令。note 是 answer 各字段的说明，接在话术后面。"""
    text = image_prefix() + clause
    if note:
        text += "回答的 JSON 中，" + note + "。"
    return text


def input_fields():
    return ["image", "instruction"]


def _lookat_quat(pos, lookat):
    """MuJoCo 相机四元数（w x y z）：相机 -z 指向 lookat，y 轴尽量朝上。

    robosuite 的 mat2quat 返回 (x, y, z, w)，MuJoCo XML 的 quat 是 (w, x, y, z)。
    """
    forward = np.asarray(lookat, dtype=float) - np.asarray(pos, dtype=float)
    forward = forward / np.linalg.norm(forward)
    z_axis = -forward
    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    quat = mat2quat(np.column_stack([x_axis, y_axis, z_axis]))
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def _install_camera(name, pos, lookat, fovy):
    """给 TableArena / BinsArena / PegsArena 加一台固定相机（幂等）。

    在创建环境前调用；arena 的 XML 编译前把 camera 元素挂到 worldbody 上。
    """
    from xml.etree import ElementTree as ET

    from robosuite.models.arenas import BinsArena, PegsArena, TableArena

    quat = _lookat_quat(pos, lookat)
    flag = "_camera_patched_{}".format(name)
    for cls in (TableArena, BinsArena, PegsArena):
        if getattr(cls, flag, False):
            continue
        orig_init = cls.__init__

        def patched_init(self, *args, _orig=orig_init, _name=name, _pos=np.array(pos), _quat=quat, _fovy=fovy, **kwargs):
            _orig(self, *args, **kwargs)
            camera = self.worldbody.find(".//camera[@name='{}']".format(_name))
            if camera is None:
                camera = ET.Element("camera", attrib={"name": _name})
                self.worldbody.append(camera)
            camera.attrib.update({
                "mode": "fixed",
                "pos": "{:.6f} {:.6f} {:.6f}".format(*_pos),
                "quat": "{:.8f} {:.8f} {:.8f} {:.8f}".format(*_quat),
                "fovy": "{:.1f}".format(_fovy),
            })

        cls.__init__ = patched_init
        setattr(cls, flag, True)


_install_camera(DEFAULT_CAMERA, FRONT_CAMERA_POS, FRONT_CAMERA_LOOKAT, FRONT_CAMERA_FOVY)
# Panda 默认 OSC 每个控制步的位置、姿态缩放，和 default_panda.json 一致。
POS_SCALE = 0.05
ORI_SCALE = 0.5


def panda_world_osc():
    """默认 Panda 的操作空间控制器，把增量改到世界坐标系，方便对准方块。
    这里发的是末端位置和朝向，不解关节角。"""
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


def make_env(image_size, seed, camera_depths=False, use_ik=False):
    try:
        env = suite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=joint_position_config() if use_ik else panda_world_osc(),
            initialization_noise=None,
            placement_initializer=cube_placement(seed),
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=list(CAMERAS),
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
# 每个 subtask 的 desc：这一步做什么，20 字以内。
STEP_DESC = {
    "move_above": "移到方块正上方",
    "move_down": "竖直降到方块处",
    "close_gripper": "合爪夹住方块",
    "lift": "竖直向上抬起方块",
    "stabilize": "停在原地保持稳定",
}
# 相对桌面中心。x 正方向远离机械臂，范围故意不对称，避免伸到够不着的地方。
CUBE_X_RANGE = (-0.10, 0.06)
CUBE_Y_RANGE = (-0.12, 0.12)
# 开局把方块放进其中一块，避免总挤在桌子正中。坐标相对桌面中心，米。
CUBE_REGIONS = (
    ((-0.16, -0.06), (-0.18, -0.06)),
    ((-0.16, -0.06), (0.06, 0.18)),
    ((-0.02, 0.10), (-0.18, -0.06)),
    ((-0.02, 0.10), (0.06, 0.18)),
    ((-0.10, 0.04), (-0.06, 0.06)),
)
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


def yaw_quat(yaw):
    """绕竖直轴的四元数，w x y z。自由关节的 qpos 用这个顺序。"""
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def set_free_xyaw(env, obj, xy, yaw=None):
    """改自由关节的水平位置。yaw 为 None 时保持原来的朝向，高度不动。"""
    qpos = np.array(env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float)
    qpos[0] = float(xy[0])
    qpos[1] = float(xy[1])
    if yaw is not None:
        qpos[3:7] = yaw_quat(yaw)
    env.sim.data.set_joint_qpos(obj.joints[0], qpos)


def place_lift_cube(env, rng):
    """方块放到抽中的那块区域里，再绕竖直轴转一个随机角度。"""
    region = CUBE_REGIONS[int(rng.integers(len(CUBE_REGIONS)))]
    xy = np.array([rng.uniform(*region[0]), rng.uniform(*region[1])])
    set_free_xyaw(env, env.cube, xy, yaw=float(rng.uniform(0.0, 2.0 * np.pi)))
    env.sim.forward()


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
        return with_views(DONE_INSTRUCTION, plan_desc()), 0.0
    # 还没对准时后半句用完整抬升说法，避免「再……再……」。
    template = again if above and lifted_m >= 0.005 else fresh
    amount = spoken * 100.0 if unit == "cm" else spoken
    lift_clause = template.format(amount)
    if above:
        return with_views(lift_clause, plan_desc()), spoken
    body = lift_clause[:-1] if lift_clause.endswith("。") else lift_clause
    return with_views("先移动到目标上方，再{}。".format(body), plan_desc()), spoken


def world_to_pixel_matrices(env, image_size):
    """每台相机的世界坐标到像素的 4x4 矩阵，{相机名: 矩阵}。图像是正方形。"""
    size = int(image_size)
    return {
        name: get_camera_transform_matrix(
            sim=env.sim,
            camera_name=name,
            camera_height=size,
            camera_width=size,
        )
        for name in CAMERAS
    }


def world_to_pixel_matrix(env, image_size):
    """旧接口：只返回 frontview 的矩阵。新代码用 world_to_pixel_matrices。"""
    return world_to_pixel_matrices(env, image_size)[DEFAULT_CAMERA]


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


def grip_view_pixels(front, back, matrices):
    """两个夹爪落点的 frontview 像素。"""
    return {
        "target_1": format_pixel(project_world(front, matrices[DEFAULT_CAMERA])[0]),
        "target_2": format_pixel(project_world(back, matrices[DEFAULT_CAMERA])[0]),
    }


def matrix_to_list(matrix):
    return [[float(value) for value in row] for row in np.asarray(matrix, dtype=float)]


def grip_landings(env):
    """方块前侧和背侧的两个夹爪落点，世界坐标，米。两点在方块表面上，沿世界 x 轴。"""
    center = np.array(env.sim.data.body_xpos[env.cube_body_id], dtype=float)
    half = np.asarray(env.cube.size, dtype=float)
    rot = np.array(env.sim.data.body_xmat[env.cube_body_id], dtype=float).reshape(3, 3)
    return box_contacts(center, half, rot, np.array([1.0, 0.0, 0.0]))


def plan_desc():
    return (
        "target_1 是夹爪1落点的像素，target_2 是夹爪2落点的像素，都在正面相机图上；"
        "target_2 被方块挡住时也根据方块形状和朝向补出它的像素，不写 mask。两落点的中点是抓取位置"
    )


def build_subtasks(phase, above, eef_z, cube_z, delta_z, pixels):
    """从当前画面接着做的步骤。已经做完的不再写入。落点像素写在要动位置的步骤上。

    pixels 是 grip_view_pixels 的返回：{"target_1", ...}，值是 frontview 的 "[x y]"，键序就是写出顺序。
    每步带 desc，说明这一步做什么。
    """

    def spatial(step):
        step["desc"] = STEP_DESC[step["type"]]
        step.update(pixels)
        step["target_kind"] = "grip"
        return step

    subtasks = []
    if phase != "lift":
        if not above:
            subtasks.append(spatial({"type": "move_above"}))
        if (not above) or (float(eef_z) - float(cube_z) > GRASP_Z_TOL):
            subtasks.append(spatial({"type": "move_down"}))
        subtasks.append({"type": "close_gripper", "desc": STEP_DESC["close_gripper"]})
    if float(delta_z) >= 0.005:
        lift = spatial({"type": "lift"})
        lift["delta_z"] = round(float(delta_z), 2)
        subtasks.append(lift)
    if subtasks:
        subtasks.append({"type": "stabilize", "desc": STEP_DESC["stabilize"], "duration": STABILIZE_DURATION})
    return subtasks


def format_answer(pixels, cube_pos, eef_pos, phase, above, delta_z):
    """MLLM 要生成的抓取计划。要动位置的步骤带两个夹爪落点的像素，delta_z 是还要抬的高度；字段含义在指令里。"""
    plan = {
        "subtasks": build_subtasks(
            phase,
            above,
            eef_pos[2],
            np.asarray(cube_pos, dtype=float)[2],
            delta_z,
            pixels,
        ),
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
        self.image_dir = make_image_staging_dir(self.out_dir, "lift")
        self.jsonl_path = self.out_dir / "samples.jsonl"
        self.meta_path = self.out_dir / "meta.json"
        self.image_size = image_size
        self.env = env
        self.world_to_pixel = world_to_pixel_matrices(env, image_size)
        self.task = "lift"
        self.samples = []
        self.episodes = []
        self.pending = []
        self.discarded = 0
        self.merged_count = 0

    def add(self, episode, step, phase, obs, instruction, lift_height, target_lift, lifted, above):
        """先缓存在内存里，等这条轨迹成功了才写盘。"""
        cube_pos, eef_pos = positions_from_obs(obs)
        front, back = grip_landings(self.env)
        pixels = grip_view_pixels(front, back, self.world_to_pixel)
        cube_pos = [round(float(value), 4) for value in cube_pos]
        eef_pos = [round(float(value), 4) for value in eef_pos]
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
            "instruction": instruction,
            "above_cube": bool(above),
            "lift_height_m": round(float(lift_height), 4),
            "target_lift_m": round(float(target_lift), 4),
            "lifted_m": round(float(lifted), 4),
            "cube_pos": cube_pos,
            "grip_front": [round(float(value), 4) for value in front],
            "grip_back": [round(float(value), 4) for value in back],
            "grip_pixel": pixels["target_1"],
            "eef_pos": eef_pos,
            "answer": format_answer(pixels, cube_pos, eef_pos, phase, above, lift_height),
            "success": False,
        }
        record["grip_pixel_2"] = pixels["target_2"]
        rgbs = {name: np.array(obs[name + "_image"], copy=True) for name in CAMERAS}
        self.pending.append((record, rgbs))
        return record

    def finish_episode(self, episode, target_lift, success):
        pending, self.pending = self.pending, []
        if not success:
            self.discarded += 1
            return
        for record, rgbs in pending:
            save_rgb(self.image_dir / Path(record["image"]).name, rgbs[DEFAULT_CAMERA])
            record["success"] = True
            self.samples.append(record)
        self.episodes.append(
            {
                "episode": episode,
                "target_lift_m": round(float(target_lift), 4),
                "success": True,
            }
        )

    def close(self, publish=True):
        desc_note = "每个 subtask 都带 desc，用 20 字以内说明这一步做什么；各字段的含义跟在指令话术后面，不在 answer 里重复。"
        pixels = (
            "要动位置的步骤带 target_1 和 target_2，都是 frontview 的整数像素，形如 \"[x y]\"，x 向右、y 向下，原点在图像左上角。"
            "target_1 是朝向相机的落点，target_2 是方块背面的落点，挡住了也照写。" + desc_note
        )
        frame = "target_1、target_2 都是 frontview 像素。grip_front、grip_back 是两个落点的世界坐标，米"
        units = {"target_1": "pixel", "target_2": "pixel", "grip_front": "meter", "grip_back": "meter"}
        sources = {
            "cube_pos": "obs['cube_pos']，世界坐标，米",
            "grip_front": "方块表面朝 +x（frontview）的夹爪落点，世界坐标，米",
            "grip_back": "方块表面朝 -x（背侧）的夹爪落点，世界坐标，米",
            "grip_pixel": "grip_front 投到 frontview 的像素字符串，和 subtasks 里的 target_1 一致",
            "eef_pos": "obs['robot0_eef_pos']",
        }
        sources["grip_pixel_2"] = "grip_back 投到 frontview 的像素字符串，和 subtasks 里的 target_2 一致。这个点常被方块挡住"
        meta = {
            "env": "Lift",
            "robot": "Panda",
            "camera": DEFAULT_CAMERA,
            "cameras": list(CAMERAS),
            "image_size": [self.image_size, self.image_size],
            "inputs": input_fields(),
            "output": {
                "answer": "JSON 字符串。subtasks 按顺序可能包含 move_above、move_down、close_gripper、lift（delta_z 为剩余抬升，米）、stabilize（duration 0.5 秒），每个 subtask 的 desc 用 20 字以内说明这一步做什么。" + pixels + "各空间步骤还带 target_kind，固定为 grip（夹爪落点，不是方块中心）。已经完成的步骤不写。抬到目标高度后 subtasks 为空。",
            },
            "coordinate_frame": frame,
            "units": units,
            "camera_world_to_pixel": {name: matrix_to_list(matrix) for name, matrix in self.world_to_pixel.items()},
            "position_source": sources,
            "instruction": "指令以一个 <image> 占位开头，和 inputs 里的 frontview 图片对应。话术后面带 answer 各字段的说明（target_1、target_2 是两个夹爪落点）。夹爪不在方块正上方时，指令是先移动到目标上方，再抬剩余高度。对准之后只写剩余高度。lift_height_m 是还要抬的高度，lifted_m 是已经抬起的高度，target_lift_m 是总目标。above_cube 表示末端是否已在方块正上方。方块位置、手臂出发位置和悬停高度每条抓取都会变。夹爪竖直朝下，两指沿世界 x 轴合拢。",
            "num_episodes": len(self.episodes),
            "num_discarded_episodes": self.discarded,
            "num_samples": len(self.samples),
            "episodes": self.episodes,
        }
        finish_image_staging(self.out_dir, self.image_dir, publish)
        if publish:
            self.merged_count = publish_task(self.out_dir, self.task, self.samples, meta)


def arm_controller(env):
    return env.robots[0].part_controllers["right"]


def step_toward(env, target_pos, hold_ori, grip, max_speed=None):
    """世界坐标 OSC：把末端位置、朝向误差收成控制器增量。不解关节角，逆运动学不在这条路径上。
    max_speed 是每个控制步最多走多少米，不给就按控制器缩放走。"""
    arm = arm_controller(env)
    arm.update(force=True)
    delta_pos = target_pos - arm.ref_pos
    if max_speed is not None:
        dist = float(np.linalg.norm(delta_pos))
        if dist > float(max_speed):
            delta_pos = delta_pos * (float(max_speed) / dist)
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


def move_until(env, target_pos, hold_ori, grip, pos_tol, max_steps, on_step, max_speed=None):
    obs = None
    pos_err = None
    for _ in range(max_steps):
        obs, _, _, _ = step_toward(env, target_pos, hold_ori, grip, max_speed)
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
    place_lift_cube(env, rng)
    arm = arm_controller(env)
    arm.update(force=True)
    # 方块刚放下时会悬空一点点，先停住让它落到桌面，再把这个高度当作 0。
    neutral = np.zeros(env.action_dim)
    for _ in range(20):
        obs, _, _, _ = env.step(neutral)
    rest_z = positions_from_obs(obs)[0][2]
    # 指腹中点相对末端的偏移，和开度无关。两指沿世界 x 合拢，一个落点朝前、一个朝后。
    finger_offset = pad_offset(env)
    hold_ori = topdown_ori(GRIP_YAW)

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
        )
        state["last_key"] = key
        state["last_step"] = state["step"]
        return sample

    def on_step(new_obs):
        state["step"] += 1
        return new_obs

    def tick(_obs):
        state["step"] += 1

    # 先抬高，再平移到随机出发点，避免横着扫过方块。这段不写入数据集。
    state["phase"] = "setup"
    arm_controller(env).update(force=True)
    home = np.array(arm_controller(env).ref_pos)
    safe_z = max(home[2], rest_z + 0.18)
    start = sample_start_pos(rng, positions_from_obs(obs)[0], rest_z)
    move_until(env, np.array([home[0], home[1], safe_z]), hold_ori, grip=-1.0, pos_tol=0.02, max_steps=80, on_step=tick)
    move_until(env, np.array([start[0], start[1], safe_z]), hold_ori, grip=-1.0, pos_tol=0.02, max_steps=160, on_step=tick)
    obs, _, _ = move_until(env, start, hold_ori, grip=-1.0, pos_tol=0.02, max_steps=80, on_step=tick)

    record(obs, "initial", force=True)

    hover_clearance = float(rng.uniform(*HOVER_CLEARANCE))
    grasp_pos, _ = grasp_eef(*grip_landings(env), finger_offset)
    hover = grasp_pos + np.array([0.0, 0.0, hover_clearance])
    state["phase"] = "hover"
    obs, hover_ok, hover_err = move_until(
        env, hover, hold_ori, grip=-1.0, pos_tol=0.015, max_steps=160, on_step=on_step
    )

    grasp, _ = grasp_eef(*grip_landings(env), finger_offset)
    state["phase"] = "grasp"
    obs, grasp_ok, grasp_err = move_until(
        env, grasp, hold_ori, grip=-1.0, pos_tol=0.012, max_steps=120, on_step=on_step
    )

    state["phase"] = "close"
    obs = hold(env, grasp, hold_ori, grip=1.0, steps=25, on_step=on_step)
    grasped = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cube)

    lift = grasp + np.array([0.0, 0.0, target_lift])
    state["phase"] = "lift"
    obs, lift_ok, lift_err = move_until(
        env, lift, hold_ori, grip=1.0, pos_tol=0.02, max_steps=160, on_step=on_step
    )
    success = bool(env._check_success()) and grasped

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
    parser.add_argument("--episodes", type=int, default=50, help="采够多少条成功抓取才停，失败的不占名额")
    parser.add_argument("--max-attempts", type=int, default=0, help="最多尝试多少条，防止一直失败死循环。0 表示 episodes 的 10 倍")
    parser.add_argument("--out", type=str, default="data/lift_mllm", help="输出目录。可以和 Stack 共用，按 task 合并")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-offset", type=int, default=0, help="首个 episode 编号；并行分片时用不同偏移避免样本 ID 冲突")
    parser.add_argument("--episode-stride", type=int, default=1, help="每次尝试后的 episode 编号步长；并行分片使用 worker 总数")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--height-min", type=float, default=0.06, help="抬升高度下限，米")
    parser.add_argument("--height-max", type=float, default=0.18, help="抬升高度上限，米")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = make_env(args.image_size, args.seed)
    writer = DatasetWriter(args.out, args.image_size, env)
    max_attempts = args.max_attempts or args.episodes * 10
    successes = 0
    episode = args.episode_offset
    attempts = 0
    complete = False
    try:
        while successes < args.episodes and attempts < max_attempts:
            # 低、中、高三档各抽，避免高度都挤在区间中间。
            if args.height_max <= args.height_min:
                target_lift = float(args.height_min)
            else:
                edges = np.linspace(args.height_min, args.height_max, 4)
                band = int(rng.integers(3))
                target_lift = float(rng.uniform(edges[band], edges[band + 1]))
            style = INSTRUCTION_STYLES[int(rng.integers(len(INSTRUCTION_STYLES)))]
            success = collect_episode(env, writer, episode, target_lift, style, rng)
            successes += int(success)
            episode += args.episode_stride
            attempts += 1
        complete = successes >= args.episodes
    finally:
        writer.close(publish=complete)
        env.close()
    if not complete:
        raise RuntimeError(
            "达到最大尝试次数 {}，只采到 {} / {} 条成功抓取；未发布，保留原有数据".format(
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
