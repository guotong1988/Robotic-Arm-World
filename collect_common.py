"""PickPlace、NutAssembly、Door、Wipe 共用的采集流程。

每个任务给一份抽象计划。步骤里不写像素，只写引用的点名，比如 "Milk"、"Milk_bin"。
像素每一帧按当时的位置现算，所以同一份计划在任何一帧都能写成那一帧的 subtasks。

采集先按初始画面写出 subtasks，解析后交给 task_plan.execute_plan，和评测是同一个执行器。
每条成功轨迹只保存开局那一帧（phase 为 initial），标注是从开局起的整条步骤。
中途画面不存。泛化放在开局布局上：任务可以在 reset 前改采样范围（prepare_reset），
reset 之后再挪物体和目标（vary_initial）。没完成的轨迹整条丢掉，图片和样本都不写盘。
失败的轨迹不占 --episodes 名额，会一直补采到凑满为止，--max-attempts 封顶。

多个任务可以写进同一个 --out。样本 id 和图片名带任务名前缀，samples.jsonl 按任务合并。
任务名是小写的环境名，比如 pickplacecan，彼此不是前缀关系，清理旧图片时不会误删。
"""

import itertools
from pathlib import Path

import gl_backend

gl_backend.configure()

import numpy as np
import robosuite as suite
import robosuite.macros as macros
from robosuite.utils.transform_utils import mat2quat

from collect_lift_mllm import (
    CAMERAS,
    DEFAULT_CAMERA,
    arm_controller,
    format_pixel,
    grip_view_pixels,
    input_fields,
    matrix_to_list,
    move_until,
    panda_world_osc,
    project_world,
    rotated_about_z,
    round_pixel,
    save_rgb,
    with_views,
    world_to_pixel_matrices,
)
from mllm_dataset import finish_image_staging, make_image_staging_dir, make_sample_id, publish_task
from task_plan import (
    LIFT_TOL,
    MOVE_ABOVE_TOL,
    MOVE_DOWN_TOL,
    PLACE_DOWN_TOL,
    ParseError,
    REACH_TOL,
    STABILIZE_DURATION,
    approach_ori,
    execute_plan,
    format_plan,
    parse_plan,
    topdown_ori,
)

macros.IMAGE_CONVENTION = "opencv"

# 朝向和目标相差不到这个角度，才算这一步的朝向已经到位，弧度。
ORI_DONE = 0.1
# 剩余转角小于这个值就不再写 turn，度。
TURN_DONE = 2.0
# 夹爪竖直朝下时手腕可转范围的中心 yaw，度。见 wrist_yaw。
WRIST_CENTER = 45.0
# 手腕 yaw 离中心不超过这个角度就算安全，度。q7 限位是 ±166。
WRIST_SAFE = 140.0
_ANSWER_KIND = (
    "每个像素字段注明指的是哪类东西：target_kind、align_point_kind、pivot_kind、path_kind，取值 object（放在桌面或箱底上的物体中心）、"
    "cube_center（方块的中心）、bin（箱子里格子的底面）、nut（平放的螺母中心）、peg（柱子顶端中心）、table（贴在桌面上的点）、"
    "door_handle（门把手上的抓取点）、door_axis（门锁转轴、铰链）、grip（夹爪合拢时贴住物体表面的落点）。"
    "双落点 grip 步骤不写 yaw，合拢方向由两点连线确定。单点 move_above（clearance，米）、move_down（offset_z，米）"
    "可带 yaw（度，夹爪竖直朝下，两指合拢方向从世界 y 轴绕竖直轴转过的角度）和 align_point（手里物体上要对准目标的那一点的像素）。"
    "lift 的 delta_z 是还要抬的高度，米。"
    "reach 沿 approach（世界坐标方向）伸向 target，停在 standoff 米处。turn 绕过 pivot 像素、方向为 axis 的转轴转 angle 度。"
    "wipe 贴着桌面依次走过 path 里的像素，offset_z 是往下压的量。另外有 close_gripper、open_gripper、stabilize。完成后 subtasks 为空。"
)
# 每步 desc 的通用说法。任务实现 step_desc 时用自己的，可以带上物体名。
COMMON_STEP_DESC = {
    "move_above": "移到目标点上方",
    "move_down": "降到目标点处",
    "lift": "竖直抬起物体",
    "reach": "伸向目标点",
    "turn": "绕转轴转动",
    "wipe": "沿路径擦拭桌面",
    "close_gripper": "合爪夹住",
    "open_gripper": "松爪放下",
    "stabilize": "停在原地保持稳定",
}


def step_desc(task, env, step):
    """每步的 desc：这一步做什么，20 字以内。任务实现 step_desc(env, step) 时用自己的。"""
    fn = getattr(task, "step_desc", None)
    if fn is not None:
        text = fn(env, step)
        if text:
            return text
    return COMMON_STEP_DESC.get(step["type"], step["type"])


def plan_desc(task):
    """跟在指令话术后面的字段说明：解释 target_1 等字段。任务实现 plan_desc() 时用自己的（比如两夹爪落点的任务）。"""
    fn = getattr(task, "plan_desc", None)
    if fn is not None:
        text = fn()
        if text:
            return text
    return "target_1 是要对准的点在正面相机图上的像素"


def order_landings_at(center, camera, pair):
    """两个落点排序成 (朝相机的, 背对的)。"""
    first, second = (np.asarray(point, dtype=float).reshape(3) for point in pair)
    camera = np.asarray(camera, dtype=float).reshape(3)
    center = np.asarray(center, dtype=float).reshape(3)
    if float(np.dot(first - center, camera - center)) >= float(np.dot(second - center, camera - center)):
        return first, second
    return second, first


def order_landings(env, center, pair):
    """两个落点排序成 (朝 frontview 的, 背对的)。评测里 target_1 写正面相机看得到的那个。"""
    camid = env.sim.model.camera_name2id(DEFAULT_CAMERA)
    camera = np.asarray(env.sim.data.cam_xpos[camid], dtype=float)
    return order_landings_at(center, camera, pair)


def answer_spec():
    lead = (
        "JSON 字符串。subtasks 是从这一帧还要做的步骤，每个 subtask 的 desc 用 20 字以内说明这一步做什么；"
        "各字段的含义跟在指令话术后面，不在 answer 里重复。像素都写成 \"[x y]\"，x 向右、y 向下，原点在左上角。"
    )
    views = "每个空间字段给 frontview 的像素：target_1 对应 image；align_point、pivot、path 同理。"
    return lead + views + _ANSWER_KIND


def coordinate_frame():
    return "subtasks 里的 target_1、pivot、align_point、path 是 frontview 像素；points 里的 pos 是世界坐标，米"


def wrap_angle(deg, period):
    """把角度折到 [-period/2, period/2)。"""
    period = float(period)
    return (float(deg) + period / 2.0) % period - period / 2.0


def body_yaw_deg(env, body_id):
    """物体自身 x 轴在水平面里的朝向，度。"""
    mat = np.asarray(env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
    return float(np.degrees(np.arctan2(mat[1, 0], mat[0, 0])))


def wrist_yaw(env):
    """夹爪竖直朝下时现在的 yaw，不折回 ±180，度。不是逆运动学，只是朝下时 q7 和 yaw 的对应：
    Panda 第 7 关节 q7 约等于 45 度减 yaw，q7 限位 ±166 度，所以竖直朝下能转到的 yaw 是 [-121, 211]，中心在 45。"""
    robot = env.robots[0]
    q7 = float(env.sim.data.qpos[robot._ref_joint_pos_indexes[-1]])
    return WRIST_CENTER - float(np.degrees(q7))


def choose_yaws(current, options):
    """先验知识，不是逆运动学：两指对称，抓取 yaw 加减 180 度是同一种抓法，并躲开手腕限位。
    options 是每次抓取的 (抓取 yaw, 抓住后再转多少度)。
    控制器按最短方向转到下一个 yaw，这里顺着计划把手腕的实际转角一路累加。
    先保证手腕不超出安全范围，再挑总转动最少的一组，接近半圈的大转动容易在半路把手臂带偏。
    返回每次抓取的 (抓取 yaw, 转完后的 yaw)，都折到 ±180。"""
    best = None
    for flips in itertools.product((0.0, 180.0), repeat=len(options)):
        yaw, worst, travel, out = float(current), abs(float(current) - WRIST_CENTER), 0.0, []
        for (base, turn), flip in zip(options, flips):
            grasp = float(base) + flip
            step = wrap_angle(grasp - yaw, 360.0)
            yaw += step
            worst = max(worst, abs(yaw - WRIST_CENTER))
            yaw += float(turn)
            worst = max(worst, abs(yaw - WRIST_CENTER))
            travel += abs(step)
            out.append((wrap_angle(grasp, 360.0), wrap_angle(grasp + float(turn), 360.0)))
        key = (max(0.0, worst - WRIST_SAFE), travel)
        if best is None or key < best[0]:
            best = (key, out)
    return best[1]


def eef_pose(env):
    robot = env.robots[0]
    site = robot.eef_site_id["right"]
    pos = np.array(env.sim.data.site_xpos[site], dtype=float)
    arm = arm_controller(env)
    arm.update(force=True)
    return pos, np.array(arm.ref_ori_mat, dtype=float)


def is_grasping(env, obj):
    return bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=obj))


def _ori_close(current, target):
    dot = abs(float(np.dot(mat2quat(np.asarray(current)), mat2quat(np.asarray(target)))))
    return 2.0 * float(np.arccos(np.clip(dot, 0.0, 1.0))) <= ORI_DONE


def _vector(values):
    return [round(float(value), 2) for value in values]


def point_kind(task, name):
    """引用的点指的是哪类东西（target_kind 等的取值，见 depth_locate.LOCATORS）。任务没实现 point_kind 就不写。"""
    fn = getattr(task, "point_kind", None)
    return fn(name) if fn is not None else None


def render_step(task, env, step, points, matrices, eef, eef_ori, current):
    """抽象步骤写成这一帧的 subtask。current 表示正在做这一步，已经到位就返回 None。

    每个空间字段只写 frontview 像素，被挡住时也根据物体形状写出像素。
    每个像素字段带 *_kind 类别标注（任务实现 point_kind 时），评测按类别选深度量法。
    任务实现 grip_points(env, name) 且 name 是夹爪落点时，写 target_1、target_2，
    target_kind 标 grip，执行器取两点的中点，
    并由两点连线推导 yaw，所以抓取标签不再重复写 yaw。每步带 desc，说明这一步做什么。
    """

    def put_views(out, key, name):
        out[key] = format_pixel(project_world(points[name], matrices[DEFAULT_CAMERA])[0])

    def put_grip(out, name):
        """落点型的引用点：写两个夹爪落点在 frontview 上的像素，返回这对世界坐标。"""
        fn = getattr(task, "grip_points", None)
        if fn is None:
            return None
        pair = fn(env, name)
        if pair is None:
            return None
        out.update(grip_view_pixels(pair[0], pair[1], matrices))
        out["target_kind"] = "grip"
        return pair

    kind = step["type"]
    if kind in ("move_above", "move_down"):
        out = {"type": kind, "desc": step_desc(task, env, step)}
        grip_pair = put_grip(out, step["ref"])
        if grip_pair is None:
            put_views(out, "target_1", step["ref"])
            ref_kind = point_kind(task, step["ref"])
            if ref_kind is not None:
                out["target_kind"] = ref_kind
        if kind == "move_above":
            extra = float(step["clearance"])
            out["clearance"] = round(extra, 2)
            tol = MOVE_ABOVE_TOL
        else:
            extra = float(step["offset_z"])
            out["offset_z"] = round(extra, 3)
            tol = PLACE_DOWN_TOL if extra >= 0.02 else MOVE_DOWN_TOL
        if "yaw" in step and grip_pair is None:
            out["yaw"] = int(round(float(step["yaw"])))
        if "align_point" in step:
            put_views(out, "align_point", step["align_point"])
            align_point_kind = point_kind(task, step["align_point"])
            if align_point_kind is not None:
                out["align_point_kind"] = align_point_kind
        if current:
            base = points[step["ref"]] if grip_pair is None else 0.5 * (grip_pair[0] + grip_pair[1])
            goal = np.asarray(base, dtype=float) + np.array([0.0, 0.0, extra])
            moving = eef if "align_point" not in step else np.asarray(points[step["align_point"]], dtype=float)
            ori_ok = "yaw" not in step or _ori_close(eef_ori, topdown_ori(step["yaw"]))
            if float(np.linalg.norm(moving - goal)) <= tol and ori_ok:
                return None
        return out
    if kind == "lift":
        delta = float(step["goal_z"]) - float(points[step["ref"]][2])
        if current and float(eef[2]) >= float(step["goal_z"]) - LIFT_TOL:
            return None
        if delta < 0.005:
            return None
        out = {
            "type": "lift",
            "desc": step_desc(task, env, step),
            "delta_z": round(delta, 2),
        }
        if put_grip(out, step["ref"]) is None:
            put_views(out, "target_1", step["ref"])
            ref_kind = point_kind(task, step["ref"])
            if ref_kind is not None:
                out["target_kind"] = ref_kind
        return out
    if kind == "reach":
        approach = np.asarray(step["approach"], dtype=float)
        out = {"type": "reach", "desc": step_desc(task, env, step)}
        put_views(out, "target_1", step["ref"])
        out["approach"] = _vector(approach)
        out["standoff"] = round(float(step["standoff"]), 2)
        ref_kind = point_kind(task, step["ref"])
        if ref_kind is not None:
            out["target_kind"] = ref_kind
        if current:
            goal = np.asarray(points[step["ref"]], dtype=float) - approach * float(step["standoff"])
            if float(np.linalg.norm(eef - goal)) <= REACH_TOL and _ori_close(eef_ori, approach_ori(approach)):
                return None
        return out
    if kind == "turn":
        angle = float(task.turn_remaining(env, step))
        if abs(angle) < TURN_DONE:
            return None
        out = {"type": "turn", "desc": step_desc(task, env, step)}
        put_views(out, "pivot", step["pivot"])
        out["axis"] = _vector(step["axis"])
        out["angle"] = int(round(angle))
        pivot_kind = point_kind(task, step["pivot"])
        if pivot_kind is not None:
            out["pivot_kind"] = pivot_kind
        return out
    if kind == "wipe":
        names = task.wipe_path(env)
        if not names:
            return None
        out = {
            "type": "wipe",
            "desc": step_desc(task, env, step),
            "path": [format_pixel(project_world(points[name], matrices[DEFAULT_CAMERA])[0]) for name in names],
        }
        out["offset_z"] = round(float(step["offset_z"]), 3)
        path_kind = point_kind(task, names[0])
        if path_kind is not None:
            out["path_kind"] = path_kind
        return out
    if kind == "stabilize":
        return {"type": "stabilize", "desc": step_desc(task, env, step), "duration": STABILIZE_DURATION}
    return {"type": kind, "desc": step_desc(task, env, step)}


def render_plan(task, env, plan, index, matrices, done):
    """从第 index 步起还要做的 subtasks，以及每条对应抽象计划里的第几步。"""
    if done:
        return [], []
    points = task.points(env)
    eef, eef_ori = eef_pose(env)
    subtasks, mapping = [], []
    for k in range(int(index), len(plan)):
        step = render_step(task, env, plan[k], points, matrices, eef, eef_ori, current=(k == index))
        if step is not None:
            subtasks.append(step)
            mapping.append(k)
    return subtasks, mapping


def attach_execution_world(task, env, abstract, concrete):
    """采集执行直接用已有真值点；像素反投影只用于没有真值的模型评测。"""
    points = task.points(env)
    world = {}
    ref = abstract.get("ref")
    if ref is not None:
        grip_fn = getattr(task, "grip_points", None)
        pair = grip_fn(env, ref) if grip_fn is not None else None
        if pair is not None and "pixel_2" in concrete:
            world["pixel"], world["pixel_2"] = pair
        elif "pixel" in concrete:
            world["pixel"] = points[ref]
    for abstract_key, concrete_key in (("align_point", "align_point"), ("pivot", "pivot")):
        if abstract_key in abstract and concrete_key in concrete:
            world[concrete_key] = points[abstract[abstract_key]]
    if abstract["type"] == "wipe" and "path" in concrete:
        world["path"] = [points[name] for name in task.wipe_path(env)]
    if world:
        concrete["world"] = world


def plan_signature(subtasks):
    """去重用。步骤种类和会随动作连续变化的量，粗略分档。"""
    items = []
    for step in subtasks:
        item = [step["type"]]
        if "delta_z" in step:
            item.append(int(np.rint(float(step["delta_z"]) * 50.0)))
        if "angle" in step:
            item.append(int(np.rint(float(step["angle"]) / 5.0)))
        if "path" in step:
            item.append(len(step["path"]))
        items.append(tuple(item))
    return tuple(items)


def make_env(task, image_size, seed, use_camera=True, camera_depths=False):
    kwargs = dict(
        env_name=task.env_name,
        robots="Panda",
        controller_configs=panda_world_osc(),
        initialization_noise=None,
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
    kwargs.update(task.env_kwargs())
    try:
        return suite.make(**kwargs)
    except ImportError as exc:
        text = str(exc)
        if "EGL" in text or "OSMesa" in text or "glGetError" in text:
            raise ImportError(text + "\n\n" + gl_backend.install_hint("离屏渲染初始化失败。")) from exc
        raise


def prepare_scene(env, task, rng, on_step):
    """物体落稳之后，手臂先抬高，再平移到随机出发点。这段不交给模型。"""
    prepare = getattr(task, "prepare_reset", None)
    if prepare is not None:
        prepare(env, rng)
    obs = env.reset()
    vary = getattr(task, "vary_initial", None)
    if vary is not None:
        vary(env, rng)
        env.sim.forward()
    on_step(obs)
    arm = arm_controller(env)
    arm.update(force=True)
    base_ori = np.array(arm.ref_ori_mat, dtype=float).copy()
    neutral = np.zeros(env.action_dim)
    for _ in range(int(task.settle_steps)):
        obs, _, _, _ = env.step(neutral)
        on_step(obs)
    start, yaw, safe_z = task.start_pose(env, rng)
    hold_ori = rotated_about_z(base_ori, yaw)
    home = np.array(arm_controller(env).ref_pos, dtype=float)
    safe_z = max(float(safe_z), float(start[2]))
    move_until(env, np.array([home[0], home[1], safe_z]), hold_ori, -1.0, 0.02, 80, on_step)
    move_until(env, np.array([start[0], start[1], safe_z]), hold_ori, -1.0, 0.02, 160, on_step)
    obs, _, _ = move_until(env, np.asarray(start, dtype=float), hold_ori, -1.0, 0.02, 80, on_step)
    return obs, hold_ori


class TaskWriter:
    """样本先缓存在内存里，整条轨迹成功了才写盘。"""

    def __init__(self, out_dir, task, image_size):
        self.out_dir = Path(out_dir)
        self.image_dir = make_image_staging_dir(self.out_dir, task)
        self.task = task
        self.image_size = int(image_size)
        self.samples = []
        self.episodes = []
        self.pending = []
        self.discarded = 0
        self.merged_count = 0

    def add(self, record, rgbs):
        """rgbs 是 {相机名: 图像}。"""
        self.pending.append((record, {name: np.array(rgb, copy=True) for name, rgb in rgbs.items()}))

    def finish_episode(self, episode, success, info=None):
        pending, self.pending = self.pending, []
        if not success:
            self.discarded += 1
            return
        for record, rgbs in pending:
            save_rgb(self.image_dir / Path(record["image"]).name, rgbs[DEFAULT_CAMERA])
            record["success"] = True
            self.samples.append(record)
        entry = {"episode": int(episode), "success": True, "num_samples": len(pending)}
        if info:
            entry.update(info)
        self.episodes.append(entry)

    def close(self, meta, publish=True):
        meta = dict(meta)
        meta["num_episodes"] = len(self.episodes)
        meta["num_discarded_episodes"] = self.discarded
        meta["num_samples"] = len(self.samples)
        meta["episodes"] = self.episodes
        finish_image_staging(self.out_dir, self.image_dir, publish)
        if publish:
            self.merged_count = publish_task(self.out_dir, self.task, self.samples, meta)


def task_instruction(task, env, plan, index, style, done):
    """采集与评测共用同一条指令，避免字段说明只在一边更新。"""
    return with_views(task.instruction(env, plan, index, style, done), plan_desc(task))


def run_episode(env, task, episode, rng, style, matrices, image_size, writer=None, bucket_m=0.02):
    """采一条轨迹。writer 为 None 时只执行、不渲染标注，用来检查计划本身能不能做成。"""
    state = {"step": 0, "index": 0, "plan": [], "obs": None, "last_key": None, "last_step": None}

    def label():
        done = bool(task.success(env))
        subtasks, mapping = render_plan(task, env, state["plan"], state["index"], matrices, done)
        # 指令带 <image> 占位、话术后面带字段说明，和训练样本的格式一致。
        instruction = task_instruction(task, env, state["plan"], state["index"], style, done)
        return subtasks, mapping, instruction

    def record(phase, force):
        if writer is None or state["obs"] is None:
            return
        subtasks, _, instruction = label()
        eef, _ = eef_pose(env)
        bucket = tuple(int(value) for value in np.floor(eef / float(bucket_m)))
        key = (phase, plan_signature(subtasks), bucket)
        if state["last_step"] == state["step"] and state["last_key"] == key:
            return
        if not force and key == state["last_key"]:
            return
        sample_id = make_sample_id(task.task, episode, phase, state["step"])
        points = task.points(env)
        shown = {}
        for name in task.record_points(env):
            pixel, _ = project_world(points[name], matrices[DEFAULT_CAMERA])
            shown[name] = {
                "pos": [round(float(value), 4) for value in points[name]],
                "pixel": round_pixel(pixel),
            }
        sample = {
            "id": sample_id,
            "task": task.task,
            "episode": int(episode),
            "step": int(state["step"]),
            "phase": phase,
            "image": "images/{}.png".format(sample_id),
            "camera": DEFAULT_CAMERA,
            "instruction": instruction,
            "eef_pos": [round(float(value), 4) for value in eef],
            "points": shown,
            "answer": format_plan(subtasks),
            "success": False,
        }
        sample.update(task.record_extra(env))
        writer.add(sample, {name: state["obs"][name + "_image"] for name in CAMERAS})
        state["last_key"] = key
        state["last_step"] = state["step"]

    def tick(new_obs):
        state["step"] += 1
        state["obs"] = new_obs

    def on_step(new_obs):
        state["step"] += 1
        state["obs"] = new_obs

    obs, hold_ori = prepare_scene(env, task, rng, tick)
    state["obs"] = obs
    traces = []
    for round_index in range(int(task.max_rounds)):
        state["plan"] = task.build_plan(env, rng)
        state["index"] = 0
        if round_index == 0:
            record("initial", force=True)
        subtasks, mapping, _ = label()
        if not subtasks:
            break
        parsed = parse_plan(format_plan(subtasks), image_size=image_size)
        for concrete, abstract_index in zip(parsed["subtasks"], mapping):
            attach_execution_world(task, env, state["plan"][abstract_index], concrete)

        def on_progress(concrete, _step, mapping=mapping):
            state["index"] = mapping[concrete]

        final, trace = execute_plan(
            env,
            parsed["subtasks"],
            hold_ori,
            task.landmarks(env),
            matrices,
            grip=-1.0,
            on_step=on_step,
            on_progress=on_progress,
        )
        traces.append(trace)
        if final is not None:
            state["obs"] = final
        hold_ori = eef_pose(env)[1]
        if task.success(env):
            break
    state["index"] = len(state["plan"])
    success = bool(task.success(env))
    failed = [
        "{}#{}({})".format(step["type"], k, step["err_m"])
        for trace in traces
        for k, step in enumerate(trace)
        if not step["ok"]
    ]
    print(
        "{} episode {:d} success={} {} rounds={} unreached={}".format(
            task.task, episode, success, task.describe(env), len(traces), ",".join(failed) or "-"
        ),
        flush=True,
    )
    if writer is not None:
        writer.finish_episode(episode, success, task.episode_info(env))
    return success


def task_meta(task, matrices, image_size):
    meta = {
        "env": task.env_name,
        "robot": "Panda",
        "camera": DEFAULT_CAMERA,
        "cameras": list(CAMERAS),
        "image_size": [int(image_size), int(image_size)],
        "inputs": input_fields(),
        "output": {"answer": answer_spec()},
        "primitives": list(task.primitives),
        "coordinate_frame": coordinate_frame(),
        "camera_world_to_pixel": {name: matrix_to_list(matrix) for name, matrix in matrices.items()},
        "instruction": "指令以一个 <image> 占位开头，和 inputs 里的 frontview 图片对应。话术后面带 answer 各字段的说明。" + task.instruction_note,
        "sampling": "每条成功轨迹只保存 phase=initial 的一帧。泛化来自开局布局：物体位置、朝向和目标位置，不存轨迹中途的画面。",
    }
    meta.update(task.meta_extra())
    return meta


def add_common_args(parser, default_out):
    parser.add_argument("--episodes", type=int, default=50, help="采够多少条成功轨迹才停，失败的不占名额")
    parser.add_argument("--max-attempts", type=int, default=0, help="最多尝试多少条，防止一直失败死循环。0 表示 episodes 的 10 倍")
    parser.add_argument("--out", type=str, default=default_out, help="输出目录。可以和其他任务共用，按 task 合并")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-offset", type=int, default=0, help="首个 episode 编号；并行分片时用不同偏移避免样本 ID 冲突")
    parser.add_argument("--episode-stride", type=int, default=1, help="每次尝试后的 episode 编号步长；并行分片使用 worker 总数")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--bucket-cm", type=float, default=2.0, help="保留参数。现在每条成功轨迹只存开局一帧，不再按位移抽帧")


def run_collection(task, args):
    rng = np.random.default_rng(args.seed)
    env = make_env(task, args.image_size, args.seed, use_camera=True)
    matrices = world_to_pixel_matrices(env, args.image_size)
    writer = TaskWriter(args.out, task.task, args.image_size)
    max_attempts = args.max_attempts or args.episodes * 10
    successes = 0
    episode = args.episode_offset
    attempts = 0
    complete = False
    try:
        while successes < args.episodes and attempts < max_attempts:
            style = task.styles[int(rng.integers(len(task.styles)))]
            try:
                success = run_episode(
                    env, task, episode, rng, style, matrices, args.image_size, writer, args.bucket_cm / 100.0
                )
            except ParseError as exc:
                writer.finish_episode(episode, False)
                print("{} episode {:d} 计划无效，丢弃并补采：{}".format(task.task, episode, exc), flush=True)
                success = False
            successes += int(success)
            episode += args.episode_stride
            attempts += 1
        complete = successes >= args.episodes
    finally:
        writer.close(task_meta(task, matrices, args.image_size), publish=complete)
        env.close()
    if not complete:
        raise RuntimeError(
            "达到最大尝试次数 {}，只采到 {} / {} 条成功轨迹；未发布，保留原有数据".format(
                max_attempts, successes, args.episodes
            )
        )
    print(
        "saved {} samples from {} succeeded episodes ({} failed, discarded); {} samples in {}".format(
            len(writer.samples), successes, writer.discarded, writer.merged_count, writer.out_dir
        )
    )
