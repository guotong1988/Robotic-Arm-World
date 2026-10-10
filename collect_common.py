"""PickPlace、NutAssembly、Door、Wipe 共用的采集流程。

每个任务给一份抽象计划。步骤里不写像素，只写引用的点名，比如 "Milk"、"Milk_bin"。
像素每一帧按当时的位置现算，所以同一份计划在任何一帧都能写成那一帧的 subtasks。

采集先按初始画面写出 subtasks，解析后交给 task_plan.execute_plan，和评测是同一个执行器。
执行过程中，每一帧按「正在做第几步」和当前状态，重写从这一步起还要做的步骤，作为这一帧的标注：
正在做的移动已经到位就不再写；lift、turn 写剩下的量；wipe 只写还没擦掉的污渍。
任务完成后 subtasks 为空。没完成的轨迹整条丢掉，图片和样本都不写盘。

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
    DEFAULT_CAMERA,
    arm_controller,
    format_pixel,
    matrix_to_list,
    move_until,
    panda_world_osc,
    project_world,
    rotated_about_z,
    round_pixel,
    save_rgb,
    world_to_pixel_matrix,
)
from mllm_dataset import make_sample_id, publish_task
from task_plan import (
    LIFT_TOL,
    MOVE_ABOVE_TOL,
    MOVE_DOWN_TOL,
    PLACE_DOWN_TOL,
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
ANSWER_SPEC = (
    "JSON 字符串，只有 subtasks，是从这一帧还要做的步骤。像素都写成 \"[x y]\"，x 向右、y 向下，原点在左上角。"
    "move_above（clearance，米）、move_down（offset_z，米）可带 yaw（度，夹爪竖直朝下，两指合拢方向从世界 y 轴绕竖直轴转过的角度）"
    "和 held（手里物体上要对准目标的那一点的像素）。lift 的 delta_z 是还要抬的高度，米。"
    "reach 沿 approach（世界坐标方向）伸向 target，停在 standoff 米处。turn 绕过 pivot 像素、方向为 axis 的转轴转 angle 度。"
    "wipe 贴着桌面依次走过 path 里的像素，offset_z 是往下压的量。另外有 close_gripper、open_gripper、stabilize。完成后 subtasks 为空。"
)


def wrap_angle(deg, period):
    """把角度折到 [-period/2, period/2)。"""
    period = float(period)
    return (float(deg) + period / 2.0) % period - period / 2.0


def body_yaw_deg(env, body_id):
    """物体自身 x 轴在水平面里的朝向，度。"""
    mat = np.asarray(env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
    return float(np.degrees(np.arctan2(mat[1, 0], mat[0, 0])))


def wrist_yaw(env):
    """夹爪竖直朝下时现在的 yaw，不折回 ±180，度。Panda 第 7 关节 q7 约等于 45 度减 yaw，
    q7 限位 ±166 度，所以竖直朝下能转到的 yaw 是 [-121, 211]，中心在 45。"""
    robot = env.robots[0]
    q7 = float(env.sim.data.qpos[robot._ref_joint_pos_indexes[-1]])
    return WRIST_CENTER - float(np.degrees(q7))


def choose_yaws(current, options):
    """两指对称，抓取 yaw 加减 180 度是同一种抓法。options 是每次抓取的 (抓取 yaw, 抓住后再转多少度)。
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


def render_step(task, env, step, points, matrix, eef, eef_ori, current):
    """抽象步骤写成这一帧的 subtask。current 表示正在做这一步，已经到位就返回 None。"""

    def pixel(name):
        return format_pixel(project_world(points[name], matrix)[0])

    kind = step["type"]
    if kind in ("move_above", "move_down"):
        out = {"type": kind, "target": pixel(step["ref"])}
        if kind == "move_above":
            extra = float(step["clearance"])
            out["clearance"] = round(extra, 2)
            tol = MOVE_ABOVE_TOL
        else:
            extra = float(step["offset_z"])
            out["offset_z"] = round(extra, 3)
            tol = PLACE_DOWN_TOL if extra >= 0.02 else MOVE_DOWN_TOL
        if "yaw" in step:
            out["yaw"] = int(round(float(step["yaw"])))
        if "held" in step:
            out["held"] = pixel(step["held"])
        if current:
            goal = np.asarray(points[step["ref"]], dtype=float) + np.array([0.0, 0.0, extra])
            moving = eef if "held" not in step else np.asarray(points[step["held"]], dtype=float)
            ori_ok = "yaw" not in step or _ori_close(eef_ori, topdown_ori(out["yaw"]))
            if float(np.linalg.norm(moving - goal)) <= tol and ori_ok:
                return None
        return out
    if kind == "lift":
        delta = float(step["goal_z"]) - float(points[step["ref"]][2])
        if current and float(eef[2]) >= float(step["goal_z"]) - LIFT_TOL:
            return None
        if delta < 0.005:
            return None
        return {"type": "lift", "target": pixel(step["ref"]), "delta_z": round(delta, 2)}
    if kind == "reach":
        approach = np.asarray(step["approach"], dtype=float)
        out = {
            "type": "reach",
            "target": pixel(step["ref"]),
            "approach": _vector(approach),
            "standoff": round(float(step["standoff"]), 2),
        }
        if current:
            goal = np.asarray(points[step["ref"]], dtype=float) - approach * float(step["standoff"])
            if float(np.linalg.norm(eef - goal)) <= REACH_TOL and _ori_close(eef_ori, approach_ori(approach)):
                return None
        return out
    if kind == "turn":
        angle = float(task.turn_remaining(env, step))
        if abs(angle) < TURN_DONE:
            return None
        return {
            "type": "turn",
            "pivot": pixel(step["pivot"]),
            "axis": _vector(step["axis"]),
            "angle": int(round(angle)),
        }
    if kind == "wipe":
        names = task.wipe_path(env)
        if not names:
            return None
        return {"type": "wipe", "path": [pixel(name) for name in names], "offset_z": round(float(step["offset_z"]), 3)}
    if kind == "stabilize":
        return {"type": "stabilize", "duration": STABILIZE_DURATION}
    return {"type": kind}


def render_plan(task, env, plan, index, matrix, done):
    """从第 index 步起还要做的 subtasks，以及每条对应抽象计划里的第几步。"""
    if done:
        return [], []
    points = task.points(env)
    eef, eef_ori = eef_pose(env)
    subtasks, mapping = [], []
    for k in range(int(index), len(plan)):
        step = render_step(task, env, plan[k], points, matrix, eef, eef_ori, current=(k == index))
        if step is not None:
            subtasks.append(step)
            mapping.append(k)
    return subtasks, mapping


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


def make_env(task, image_size, seed, use_camera=True):
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
        camera_names=DEFAULT_CAMERA,
        camera_heights=image_size,
        camera_widths=image_size,
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
    obs = env.reset()
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
        self.image_dir = self.out_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.image_size = int(image_size)
        self.samples = []
        self.episodes = []
        self.pending = []
        self.discarded = 0
        self.merged_count = 0

    def add(self, record, rgb):
        self.pending.append((record, np.array(rgb, copy=True)))

    def finish_episode(self, episode, success, info=None):
        pending, self.pending = self.pending, []
        if not success:
            self.discarded += 1
            return
        for record, rgb in pending:
            save_rgb(self.image_dir / Path(record["image"]).name, rgb)
            record["success"] = True
            self.samples.append(record)
        entry = {"episode": int(episode), "success": True, "num_samples": len(pending)}
        if info:
            entry.update(info)
        self.episodes.append(entry)

    def close(self, meta):
        meta = dict(meta)
        meta["num_episodes"] = len(self.episodes)
        meta["num_discarded_episodes"] = self.discarded
        meta["num_samples"] = len(self.samples)
        meta["episodes"] = self.episodes
        self.merged_count = publish_task(self.out_dir, self.task, self.samples, meta)


def run_episode(env, task, episode, rng, style, matrix, image_size, writer=None, bucket_m=0.02):
    """采一条轨迹。writer 为 None 时只执行、不渲染标注，用来检查计划本身能不能做成。"""
    state = {"step": 0, "index": 0, "plan": [], "obs": None, "last_key": None, "last_step": None}

    def label():
        done = bool(task.success(env))
        subtasks, mapping = render_plan(task, env, state["plan"], state["index"], matrix, done)
        instruction = task.instruction(env, state["plan"], state["index"], style, done)
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
            pixel, _ = project_world(points[name], matrix)
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
        writer.add(sample, state["obs"][DEFAULT_CAMERA + "_image"])
        state["last_key"] = key
        state["last_step"] = state["step"]

    def phase_name():
        plan, index = state["plan"], state["index"]
        return plan[index]["type"] if 0 <= index < len(plan) else "final"

    def tick(new_obs):
        state["step"] += 1
        state["obs"] = new_obs

    def on_step(new_obs):
        state["step"] += 1
        state["obs"] = new_obs
        if writer is not None:
            record(phase_name(), force=False)

    obs, hold_ori = prepare_scene(env, task, rng, tick)
    state["obs"] = obs
    traces = []
    for round_index in range(int(task.max_rounds)):
        state["plan"] = task.build_plan(env, rng)
        state["index"] = 0
        record("initial" if round_index == 0 else "replan", force=True)
        subtasks, mapping, _ = label()
        if not subtasks:
            break
        parsed = parse_plan(format_plan(subtasks), image_size=image_size)

        def on_progress(concrete, _step, mapping=mapping):
            state["index"] = mapping[concrete]
            record(phase_name(), force=True)

        final, trace = execute_plan(
            env,
            parsed["subtasks"],
            hold_ori,
            task.landmarks(env),
            matrix,
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
    record("final", force=True)
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


def task_meta(task, matrix, image_size):
    meta = {
        "env": task.env_name,
        "robot": "Panda",
        "camera": DEFAULT_CAMERA,
        "image_size": [int(image_size), int(image_size)],
        "inputs": ["image", "instruction"],
        "output": {"answer": ANSWER_SPEC},
        "primitives": list(task.primitives),
        "coordinate_frame": "subtasks 里的 target、pivot、held、path 是 frontview 像素；points 里的 pos 是世界坐标，米",
        "camera_world_to_pixel": matrix_to_list(matrix),
        "instruction": task.instruction_note,
    }
    meta.update(task.meta_extra())
    return meta


def add_common_args(parser, default_out):
    parser.add_argument("--episodes", type=int, default=50, help="采集多少条轨迹，失败的不算")
    parser.add_argument("--out", type=str, default=default_out, help="输出目录。可以和其他任务共用，按 task 合并")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--bucket-cm", type=float, default=2.0, help="夹爪大约每移动这么多厘米存一帧")


def run_collection(task, args):
    rng = np.random.default_rng(args.seed)
    env = make_env(task, args.image_size, args.seed, use_camera=True)
    matrix = world_to_pixel_matrix(env, args.image_size)
    writer = TaskWriter(args.out, task.task, args.image_size)
    successes = 0
    try:
        for episode in range(args.episodes):
            style = task.styles[int(rng.integers(len(task.styles)))]
            successes += int(
                run_episode(env, task, episode, rng, style, matrix, args.image_size, writer, args.bucket_cm / 100.0)
            )
    finally:
        writer.close(task_meta(task, matrix, args.image_size))
        env.close()
    print(
        "saved {} samples from {} succeeded episodes ({} failed, discarded); {} samples in {}".format(
            len(writer.samples), successes, writer.discarded, writer.merged_count, writer.out_dir
        )
    )
