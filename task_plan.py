"""任务无关的执行步骤。

模型看到的是一张 RGB 和一句指令，写出的计划是有顺序的步骤。
带位置的步骤各自有一个像素，不共用同一个目标。所有任务都只写 subtasks。
Lift 的每一步指向同一个方块，Stack 先指向红块再指向绿块，执行器不用知道是哪个任务。

评测时把计划里的像素配上「提问这一帧、离该像素最近的物体中心」的相机深度，
反投影成世界坐标，之后不再改。物体随后被抓起来，也不能改用新的位置，
否则后半段会指到已经搬走的物体上。

步骤：

- move_above：到该像素上方，clearance 是相对物体中心再抬高多少，米
- move_down：降到该像素，offset_z 是相对物体中心的高度，米。抓取是 0，放置要加上被抓物体的高度
- close_gripper / open_gripper
- lift：把该像素对应的点再抬高 delta_z，米，夹爪保持闭合
- stabilize：停住 duration 秒
- reach：夹爪沿 approach（世界坐标方向）伸向该像素，停在离它 standoff 米处。两指上下合拢
- turn：夹爪连同手里的东西绕转轴转 angle 度。转轴过 pivot 像素，方向是 axis，右手定则
- wipe：贴着桌面依次走过 path 里的像素，offset_z 是相对这些点再往下压多少，米

move_above、move_down 可以带 yaw：夹爪竖直朝下，两指合拢方向从世界 y 轴绕竖直轴转 yaw 度。
不带 yaw 时保持当前朝向。还可以带 held，是手里物体上某一点在提问那一帧的像素。
这时对准目标的是这个点，不是夹爪：合爪时该点相对夹爪的位置，按夹爪之后转过的角度一起转，
再从目标里减掉。计划开头已经抓着东西时，用开头的夹爪位姿。
"""

import json
import re

import numpy as np
from robosuite.utils.transform_utils import axisangle2quat, mat2quat, quat2mat, quat_slerp

from collect_lift_mllm import (
    STABILIZE_DURATION,
    arm_controller,
    hold,
    move_until,
    project_world,
    unproject_pixel,
)

DEFAULT_CLEARANCE = 0.12
MOVE_ABOVE_TOL = 0.015
MOVE_ABOVE_STEPS = 200
MOVE_DOWN_TOL = 0.012
PLACE_DOWN_TOL = 0.02
MOVE_DOWN_STEPS = 140
CLOSE_STEPS = 25
OPEN_STEPS = 30
LIFT_TOL = 0.02
LIFT_STEPS = 180
REACH_TOL = 0.01
REACH_STEPS = 200
TURN_STEP = np.radians(6.0)
TURN_TOL = 0.01
TURN_STEPS = 40
WIPE_TOL = 0.01
WIPE_STEPS = 40
WIPE_SPEED = 0.01
WIPE_PRESS = 0.01
# 朝向一次变化超过这个角度时分段插值。目标离当前朝向接近半圈时，
# 控制器自己选的转向会随一点点误差翻到另一边，把手腕带到关节极限。
ORI_STEP = 0.15
# 竖直朝下、两指沿世界 y 合拢的夹爪朝向。各列是夹爪 x、y、z 轴，x 是两指合拢方向。
TOPDOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
SPATIAL = ("move_above", "move_down", "lift")
GRIP = ("close_gripper", "open_gripper")
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
PIXEL_TEXT = re.compile(r"^\[\s*({n})(?:\s*,\s*|\s+)({n})\s*\]$".format(n=NUM))
VECTOR_TEXT = re.compile(r"^\[\s*({n})(?:\s*,\s*|\s+)({n})(?:\s*,\s*|\s+)({n})\s*\]$".format(n=NUM))
THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
FENCE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


class ParseError(ValueError):
    pass


def clip_workspace(pos):
    pos = np.array(pos, dtype=float)
    pos[0] = np.clip(pos[0], -0.35, 0.35)
    pos[1] = np.clip(pos[1], -0.45, 0.45)
    pos[2] = np.clip(pos[2], 0.75, 1.40)
    return pos


def plausible(pos):
    pos = np.asarray(pos, dtype=float)
    if pos.shape != (3,) or not np.all(np.isfinite(pos)):
        return False
    if abs(pos[0]) > 1.5 or abs(pos[1]) > 1.5:
        return False
    return 0.4 <= pos[2] <= 2.0


def _bounded(value, low, high, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ParseError("{} 不是数字".format(name)) from exc
    if not np.isfinite(value) or value < low or value > high:
        raise ParseError("{} 超出范围：{}".format(name, value))
    return value


def parse_pixel(value, image_size):
    """采集写的是 \"[x y]\"。也接受逗号和长度为 2 的数组。"""
    if isinstance(value, str):
        match = PIXEL_TEXT.match(value.strip())
        if not match:
            raise ParseError("target 应为 [x y] 像素，收到 {}".format(value[:80]))
        pixel = np.array([float(item) for item in match.groups()], dtype=float)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        pixel = np.array([float(item) for item in value], dtype=float)
    else:
        raise ParseError("target 应为 [x y] 像素")
    if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
        raise ParseError("target 像素无效")
    limit = 4096.0 if image_size is None else float(image_size)
    if np.any(pixel < -limit) or np.any(pixel > 2.0 * limit):
        raise ParseError("target 像素超出画面：{}".format(pixel.tolist()))
    return pixel


def parse_vector(value, name):
    """世界坐标里的方向，长度为 3 的数组，或 \"[x y z]\"。返回单位向量。"""
    try:
        if isinstance(value, str):
            match = VECTOR_TEXT.match(value.strip())
            if not match:
                raise ParseError("{} 应为 [x y z]".format(name))
            vec = np.array([float(item) for item in match.groups()], dtype=float)
        elif isinstance(value, (list, tuple)) and len(value) == 3:
            vec = np.array([float(item) for item in value], dtype=float)
        else:
            raise ParseError("{} 应为 [x y z]".format(name))
    except (TypeError, ValueError) as exc:
        raise ParseError("{} 不是数字".format(name)) from exc
    norm = float(np.linalg.norm(vec))
    if not np.all(np.isfinite(vec)) or norm < 1e-6:
        raise ParseError("{} 无效".format(name))
    return vec / norm


def _json_blobs(text):
    blobs = []
    for match in FENCE.findall(text):
        blobs.append(match.strip())
    blobs.append(text.strip())
    return blobs


def _load_object(text):
    start = text.find("{")
    if start < 0:
        raise ParseError("没有 JSON 对象")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ParseError("JSON 无法解析") from exc
    if not isinstance(obj, dict):
        raise ParseError("JSON 不是对象")
    return obj


def _required(item, key, kind):
    if key not in item:
        raise ParseError("{} 缺少 {}".format(kind, key))
    return item[key]


def _parse_subtasks(items, image_size):
    if not isinstance(items, list):
        raise ParseError("subtasks 应为数组")
    steps = []
    for item in items:
        if not isinstance(item, dict) or "type" not in item:
            raise ParseError("子任务缺少 type")
        kind = item["type"]
        step = {"type": kind}
        if kind in SPATIAL or kind == "reach":
            step["pixel"] = parse_pixel(_required(item, "target", kind), image_size)
        if kind in ("move_above", "move_down"):
            if "yaw" in item:
                step["yaw"] = _bounded(item["yaw"], -180.0, 180.0, "yaw")
            if "held" in item:
                step["held"] = parse_pixel(item["held"], image_size)
        if kind == "move_above":
            step["clearance"] = _bounded(item.get("clearance", DEFAULT_CLEARANCE), 0.0, 0.4, "clearance")
        elif kind == "move_down":
            step["offset_z"] = _bounded(item.get("offset_z", 0.0), -0.3, 0.4, "offset_z")
        elif kind == "lift":
            step["delta_z"] = _bounded(_required(item, "delta_z", kind), 0.0, 0.5, "delta_z")
        elif kind == "reach":
            step["approach"] = parse_vector(_required(item, "approach", kind), "approach")
            step["standoff"] = _bounded(item.get("standoff", 0.0), 0.0, 0.3, "standoff")
        elif kind == "turn":
            step["pivot"] = parse_pixel(_required(item, "pivot", kind), image_size)
            step["axis"] = parse_vector(_required(item, "axis", kind), "axis")
            step["angle"] = _bounded(_required(item, "angle", kind), -180.0, 180.0, "angle")
        elif kind == "wipe":
            path = _required(item, "path", kind)
            if not isinstance(path, list) or not path:
                raise ParseError("wipe 的 path 应为非空数组")
            step["path"] = [parse_pixel(pixel, image_size) for pixel in path]
            step["offset_z"] = _bounded(item.get("offset_z", -WIPE_PRESS), -0.05, 0.05, "offset_z")
        elif kind == "stabilize":
            step["duration"] = _bounded(item.get("duration", STABILIZE_DURATION), 0.0, 5.0, "duration")
        elif kind in GRIP:
            pass
        else:
            raise ParseError("未知子任务 {}".format(kind))
        steps.append(step)
    return steps


def parse_plan(text, image_size=None):
    """从模型回答里取出 subtasks。空间步骤解析成像素。"""
    if not text or not str(text).strip():
        raise ParseError("模型没有返回文本")
    raw = str(text).strip()
    stripped = THINK.sub("", raw).strip()
    candidates = []
    for candidate in (stripped, raw):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    last_error = None
    for candidate in candidates:
        for blob in _json_blobs(candidate):
            try:
                obj = _load_object(blob)
            except ParseError as exc:
                last_error = exc
                continue
            if "subtasks" not in obj:
                last_error = ParseError("缺少 subtasks")
                continue
            try:
                steps = _parse_subtasks(obj["subtasks"], image_size)
            except ParseError as exc:
                last_error = exc
                continue
            return {"subtasks": steps}
    raise ParseError("无法解析执行步骤：{}".format(raw[:300])) from last_error


def format_plan(subtasks):
    plan = {"subtasks": subtasks}
    return json.dumps(plan, ensure_ascii=False, separators=(", ", ": "))


def resolve_pixel(pixel, landmarks, matrix):
    """用离该像素最近的物体中心的相机深度，把像素反投影回世界坐标。"""
    pixel = np.asarray(pixel, dtype=float).reshape(2)
    best_depth = None
    best_dist = None
    for pos in landmarks:
        proj, depth = project_world(pos, matrix)
        dist = float(np.linalg.norm(np.asarray(proj, dtype=float) - pixel))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_depth = depth
    if best_depth is None:
        raise ParseError("没有可用的物体深度")
    try:
        world = unproject_pixel(pixel, best_depth, matrix)
    except ValueError as exc:
        raise ParseError(str(exc)) from exc
    if not plausible(world):
        raise ParseError("反投影超出工作空间：{}".format(np.round(world, 4).tolist()))
    return world


def rotation_z(angle_rad):
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def axis_rotation(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    return quat2mat(axisangle2quat(axis / np.linalg.norm(axis) * float(angle_rad)))


def topdown_ori(yaw_deg):
    return rotation_z(np.radians(float(yaw_deg))) @ TOPDOWN


def approach_ori(approach):
    """夹爪 z 轴沿 approach。两指合拢方向取竖直向下在垂直面里的投影，也就是上下合拢。"""
    z_axis = np.asarray(approach, dtype=float)
    z_axis = z_axis / np.linalg.norm(z_axis)
    down = np.array([0.0, 0.0, -1.0])
    x_axis = down - float(np.dot(down, z_axis)) * z_axis
    if np.linalg.norm(x_axis) < 1e-3:
        x_axis = TOPDOWN[:, 0] - float(np.dot(TOPDOWN[:, 0], z_axis)) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    return np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])


def _ori_angle(left, right):
    dot = abs(float(np.dot(mat2quat(left), mat2quat(right))))
    return 2.0 * float(np.arccos(np.clip(dot, 0.0, 1.0)))


def goto(env, dest, ori, grip, tol, limit, on_step, max_speed=None):
    """位置和朝向一起走。朝向要转得多时，先按插值的中间朝向分段走。"""
    arm = arm_controller(env)
    arm.update(force=True)
    start = np.array(arm.ref_pos, dtype=float)
    start_ori = np.array(arm.ref_ori_mat, dtype=float)
    angle = _ori_angle(start_ori, ori)
    if angle > 2.0 * ORI_STEP:
        q0, q1 = mat2quat(start_ori), mat2quat(ori)
        pieces = int(np.ceil(angle / ORI_STEP))
        for k in range(1, pieces):
            frac = k / float(pieces)
            middle = quat2mat(quat_slerp(q0, q1, frac))
            move_until(env, start + frac * (dest - start), middle, grip, 0.03, 15, on_step)
    return move_until(env, dest, ori, grip, tol, limit, on_step, max_speed=max_speed)


def held_offset(held, grasp_point, grasp_ori, ori):
    """手里物体上的点相对夹爪的位置。物体跟着夹爪刚性转动，所以按夹爪从合爪起转过的角度旋转。"""
    rel = np.asarray(held, dtype=float) - np.asarray(grasp_point, dtype=float)
    return (np.asarray(ori, dtype=float) @ np.asarray(grasp_ori, dtype=float).T) @ rel


def _dest_for(step, landmarks, matrix, grasp_point, grasp_ori, ori):
    world = clip_workspace(resolve_pixel(step["pixel"], landmarks, matrix))
    kind = step["type"]
    if kind == "move_above":
        extra = float(step["clearance"])
    elif kind == "move_down":
        extra = float(step["offset_z"])
    else:
        extra = float(step["delta_z"])
    dest = world + np.array([0.0, 0.0, extra])
    if "held" in step:
        held = resolve_pixel(step["held"], landmarks, matrix)
        dest = dest - held_offset(held, grasp_point, grasp_ori, ori)
    return clip_workspace(dest)


def _turn(env, pivot, axis, angle_deg, ori, grip, on_step):
    arm = arm_controller(env)
    arm.update(force=True)
    start = np.array(arm.ref_pos, dtype=float)
    angle = np.radians(float(angle_deg))
    pieces = max(1, int(np.ceil(abs(angle) / TURN_STEP)))
    obs, ok, err = None, True, 0.0
    rot = np.eye(3)
    dest = start
    for k in range(1, pieces + 1):
        rot = axis_rotation(axis, angle * k / float(pieces))
        dest = clip_workspace(pivot + rot @ (start - pivot))
        obs, ok, err = move_until(env, dest, rot @ ori, grip, TURN_TOL, TURN_STEPS, on_step)
    return obs, ok, err, rot @ ori, dest


def execute_plan(env, steps, hold_ori, landmarks, matrix, grip=-1.0, on_step=None, on_progress=None):
    """按步骤顺序执行。landmarks 是提问那一帧的物体中心，执行过程中不更新。

    on_progress(index, step) 在每一步开始前调用，采集用它知道正在做第几步。
    """
    if on_step is None:
        on_step = lambda _obs: None
    landmarks = [np.array(pos, dtype=float).reshape(3).copy() for pos in landmarks]
    ori = np.array(hold_ori, dtype=float)
    arm_controller(env).update(force=True)
    anchor = np.array(arm_controller(env).ref_pos, dtype=float)
    grasp_point = anchor.copy()
    grasp_ori = np.array(arm_controller(env).ref_ori_mat, dtype=float)
    freq = int(round(float(getattr(env, "control_freq", 20))))
    obs = None
    trace = []
    for index, step in enumerate(steps):
        if on_progress is not None:
            on_progress(index, step)
        kind = step["type"]
        if kind in SPATIAL:
            if "yaw" in step:
                ori = topdown_ori(step["yaw"])
            dest = _dest_for(step, landmarks, matrix, grasp_point, grasp_ori, ori)
            if kind == "move_above":
                tol, limit, command = MOVE_ABOVE_TOL, MOVE_ABOVE_STEPS, grip
            elif kind == "move_down":
                tol = PLACE_DOWN_TOL if float(step["offset_z"]) >= 0.02 else MOVE_DOWN_TOL
                limit, command = MOVE_DOWN_STEPS, grip
            else:
                tol, limit, command = LIFT_TOL, LIFT_STEPS, 1.0
            obs, ok, err = goto(env, dest, ori, command, tol, limit, on_step)
            anchor = dest
            grip = command
        elif kind == "reach":
            ori = approach_ori(step["approach"])
            target = resolve_pixel(step["pixel"], landmarks, matrix)
            dest = clip_workspace(target - np.asarray(step["approach"]) * float(step["standoff"]))
            obs, ok, err = goto(env, dest, ori, grip, REACH_TOL, REACH_STEPS, on_step)
            anchor = dest
        elif kind == "turn":
            pivot = resolve_pixel(step["pivot"], landmarks, matrix)
            obs, ok, err, ori, anchor = _turn(env, pivot, step["axis"], step["angle"], ori, grip, on_step)
        elif kind == "wipe":
            ok, err = True, 0.0
            for pixel in step["path"]:
                point = resolve_pixel(pixel, landmarks, matrix)
                anchor = clip_workspace(point + np.array([0.0, 0.0, float(step["offset_z"])]))
                obs, reached, err = move_until(
                    env, anchor, ori, grip, WIPE_TOL, WIPE_STEPS, on_step, max_speed=WIPE_SPEED
                )
                ok = ok and reached
        elif kind == "close_gripper":
            obs = hold(env, anchor, ori, 1.0, CLOSE_STEPS, on_step)
            ok, err, grip = True, 0.0, 1.0
            arm_controller(env).update(force=True)
            grasp_point = np.array(arm_controller(env).ref_pos, dtype=float)
            grasp_ori = np.array(arm_controller(env).ref_ori_mat, dtype=float)
        elif kind == "open_gripper":
            obs = hold(env, anchor, ori, -1.0, OPEN_STEPS, on_step)
            ok, err, grip = True, 0.0, -1.0
        elif kind == "stabilize":
            ticks = int(round(float(step["duration"]) * freq))
            arm_controller(env).update(force=True)
            parked = clip_workspace(arm_controller(env).ref_pos)
            if ticks <= 0:
                ok, err = True, 0.0
            else:
                obs = hold(env, parked, ori, grip, ticks, on_step)
                ok, err = True, 0.0
        else:
            raise ParseError("未知子任务 {}".format(kind))
        trace.append(
            {
                "type": kind,
                "ok": bool(ok),
                "err_m": None if err is None else round(float(err), 4),
            }
        )
    return obs, trace


def check_parser():
    """不启动仿真，确认各任务都是只有 subtasks，而且每步可以有自己的像素。"""
    stack = format_plan(
        [
            {"type": "move_above", "target": "[210 300]", "clearance": 0.12},
            {"type": "move_down", "target": "[210 300]", "offset_z": 0.0},
            {"type": "close_gripper"},
            {"type": "lift", "target": "[210 300]", "delta_z": 0.1},
            {"type": "move_above", "target": "[320 280]", "clearance": 0.12},
            {"type": "move_down", "target": "[320 280]", "offset_z": 0.041},
            {"type": "open_gripper"},
            {"type": "move_above", "target": "[320 280]", "clearance": 0.12},
            {"type": "stabilize", "duration": 0.5},
        ],
    )
    plan = parse_plan(stack, image_size=512)
    assert "task" not in json.loads(stack)
    assert [step["type"] for step in plan["subtasks"]] == [
        "move_above",
        "move_down",
        "close_gripper",
        "lift",
        "move_above",
        "move_down",
        "open_gripper",
        "move_above",
        "stabilize",
    ]
    assert np.allclose(plan["subtasks"][0]["pixel"], [210, 300])
    assert np.allclose(plan["subtasks"][4]["pixel"], [320, 280])
    assert abs(plan["subtasks"][5]["offset_z"] - 0.041) < 1e-9
    lift = '{"subtasks": [{"type": "move_down", "target": "[282 314]"}, {"type": "close_gripper"}, {"type": "lift", "target": "[282 314]", "delta_z": 0.14}, {"type": "stabilize", "duration": 0.5}]}'
    plan = parse_plan(lift, image_size=512)
    assert np.allclose(plan["subtasks"][0]["pixel"], [282, 314])
    assert np.allclose(plan["subtasks"][2]["pixel"], [282, 314])
    assert abs(plan["subtasks"][2]["delta_z"] - 0.14) < 1e-9
    assert parse_plan('{"subtasks": []}', image_size=512)["subtasks"] == []
    others = format_plan(
        [
            {"type": "move_above", "target": "[200 310]", "clearance": 0.1, "yaw": -35},
            {"type": "move_down", "target": "[250 300]", "offset_z": -0.09, "held": "[240 305]"},
            {"type": "reach", "target": "[150 240]", "approach": [-0.14, -0.99, 0.0], "standoff": 0.1},
            {"type": "turn", "pivot": "[160 250]", "axis": [0.0, 0.0, 1.0], "angle": 23},
            {"type": "wipe", "path": ["[200 300]", "[210 305]"], "offset_z": -0.01},
        ]
    )
    plan = parse_plan(others, image_size=512)["subtasks"]
    assert abs(plan[0]["yaw"] + 35.0) < 1e-9
    assert np.allclose(plan[1]["held"], [240, 305])
    assert abs(np.linalg.norm(plan[2]["approach"]) - 1.0) < 1e-9
    assert np.allclose(plan[3]["axis"], [0.0, 0.0, 1.0]) and abs(plan[3]["angle"] - 23.0) < 1e-9
    assert len(plan[4]["path"]) == 2
    assert np.allclose(topdown_ori(0.0), TOPDOWN)
    for bad in ('{"subtasks": [{"type": "fly"}]}', '{"subtasks": [{"type": "wipe"}]}'):
        try:
            parse_plan(bad, image_size=512)
        except ParseError:
            pass
        else:
            raise AssertionError("无效步骤应被拒绝：{}".format(bad))
