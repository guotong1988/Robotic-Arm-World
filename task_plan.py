"""任务无关的执行步骤。

模型看到的是每台相机一张 RGB 和一句指令，写出的计划是有顺序的步骤。
带位置的步骤各自有每个视图一个像素：target_1 是该点在 frontview 的像素，
target_1_back 是同一点在 backview（frontview 正对面）的像素，held、pivot、path 同理。
某台相机看不到那个点时对应字段写 "mask"，不脑补这张图的像素，解析成 None，
执行时改用另一台相机的像素和矩阵还原世界坐标；两台都看不到时采集会在 frontview 兜底留像素。
单目只有 frontview，没有 *_back 字段，也不写 "mask"。
夹爪落点是一对：target_1、target_2（各自还有 *_back），两落点的中点才是要到的位置。
answer 顶层和每个 subtask 都可以带 desc，是给人看的说明，解析器直接忽略。
所有任务都只写 subtasks。
Lift 的每一步指向同一个方块，Stack 先指向红块再指向绿块，执行器不用知道是哪个任务。

评测时用提问这一帧的深度图把计划里的像素换成世界坐标（depth_locate），之后不再改。
物体随后被抓起来，也不能改用新的位置，否则后半段会指到已经搬走的物体上。
采集时没有深度图，像素配「离它最近的物体中心」的相机深度反投影。

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

每个像素字段还可以带类别：target_kind、held_kind、pivot_kind、path_kind，
注明这个像素指的是哪类东西（object、cube_center、bin、nut、peg、table、door_handle、door_axis）。
评测按类别选对应的深度量法（depth_locate）；不写时按任务和计划结构推断。
"""

import json
import re

import numpy as np
from robosuite.utils.transform_utils import axisangle2quat, mat2quat, quat2mat, quat_slerp

from collect_lift_mllm import (
    BACK_CAMERA,
    DEFAULT_CAMERA,
    MASK,
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
# 两指落点的水平间距必须小于夹爪最大开度；异常点对不能继续取中点执行。
MIN_GRIP_WIDTH = 0.005
MAX_GRIP_WIDTH = 0.085
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
    """采集写的是 \"[x y]\"。也接受逗号和长度为 2 的数组。
    被遮挡的视图写 \"mask\"（或 null），返回 None：这个视图不脑补像素，执行时换另一视图。"""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == MASK:
            return None
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


def _required_target(item, kind):
    """夹爪1落点的像素字段：新名 target_1，旧名 target，都接受。"""
    if "target_1" in item:
        return item["target_1"]
    return _required(item, "target", kind)


def _optional_kind(item, key):
    """target_kind 等：这个像素指的是哪类东西，短字符串，取值见 depth_locate.LOCATORS。没给返回 None。"""
    value = item.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 40:
        raise ParseError("{} 应为短字符串".format(key))
    return value.strip()


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
            step["pixel"] = parse_pixel(_required_target(item, kind), image_size)
            if "target_1_back" in item:
                step["pixel_back"] = parse_pixel(item["target_1_back"], image_size)
            if step["pixel"] is None and step.get("pixel_back") is None:
                raise ParseError("{} 的 target_1 在所有视图都被 mask".format(kind))
            if "target_2" in item or "target_2_back" in item:
                # 夹爪2落点。两落点的中点是执行目标，见 _dest_for。
                pixel_2 = parse_pixel(item.get("target_2"), image_size)
                pixel_2_back = parse_pixel(item.get("target_2_back"), image_size)
                if pixel_2 is not None or pixel_2_back is not None:
                    step["pixel_2"] = pixel_2
                    if "target_2_back" in item:
                        step["pixel_2_back"] = pixel_2_back
            target_kind = _optional_kind(item, "target_kind")
            if target_kind is not None:
                step["pixel_kind"] = target_kind
        if kind in ("move_above", "move_down"):
            if "yaw" in item:
                step["yaw"] = _bounded(item["yaw"], -180.0, 180.0, "yaw")
            if "held" in item or "held_back" in item:
                held = parse_pixel(item.get("held"), image_size)
                held_back = parse_pixel(item.get("held_back"), image_size)
                if held is not None or held_back is not None:
                    step["held"] = held
                    if "held_back" in item:
                        step["held_back"] = held_back
                    held_kind = _optional_kind(item, "held_kind")
                    if held_kind is not None:
                        step["held_kind"] = held_kind
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
            if "pivot_back" in item:
                step["pivot_back"] = parse_pixel(item["pivot_back"], image_size)
            if step["pivot"] is None and step.get("pivot_back") is None:
                raise ParseError("turn 的 pivot 在所有视图都被 mask")
            pivot_kind = _optional_kind(item, "pivot_kind")
            if pivot_kind is not None:
                step["pivot_kind"] = pivot_kind
            step["axis"] = parse_vector(_required(item, "axis", kind), "axis")
            step["angle"] = _bounded(_required(item, "angle", kind), -180.0, 180.0, "angle")
        elif kind == "wipe":
            path = _required(item, "path", kind)
            if not isinstance(path, list) or not path:
                raise ParseError("wipe 的 path 应为非空数组")
            step["path"] = [parse_pixel(pixel, image_size) for pixel in path]
            if "path_back" in item:
                path_back = item["path_back"]
                if not isinstance(path_back, list) or len(path_back) != len(step["path"]):
                    raise ParseError("wipe 的 path_back 应和 path 等长")
                step["path_back"] = [parse_pixel(pixel, image_size) for pixel in path_back]
            path_kind = _optional_kind(item, "path_kind")
            if path_kind is not None:
                step["path_kind"] = path_kind
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
    """序列化成模型输出的格式。字段说明跟在指令话术后面，不在回答里重复。"""
    return json.dumps({"subtasks": subtasks}, ensure_ascii=False, separators=(", ", ": "))


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
    """位置和朝向一起走。目标位姿来自计划，这一步只让 OSC 跟踪末端，不做逆运动学。
    朝向要转得多时，先按插值的中间朝向分段走。"""
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


def _resolve_view(pixel, pixel_back, landmarks, matrices):
    """优先 frontview 的像素；被 mask（None）时回退 backview。两个视图都没有就报错。"""
    if pixel is not None:
        return resolve_pixel(pixel, landmarks, matrices[DEFAULT_CAMERA])
    if pixel_back is not None and BACK_CAMERA in matrices:
        return resolve_pixel(pixel_back, landmarks, matrices[BACK_CAMERA])
    raise ParseError("该像素在所有视图都被 mask，无法还原世界坐标")


def step_point(step, key, landmarks, matrices):
    """step 里 key 像素的世界坐标。评测已经按角色定好的（step["world"]）优先，
    否则用离像素最近的 landmark 的深度反投影：frontview 的像素被 mask 时改用 backview 的。
    matrices 是每台相机的 world_to_pixel 矩阵。wipe 的 path 返回一串点。"""
    placed = step.get("world", {}).get(key)
    if placed is not None:
        return placed
    if key == "path":
        backs = step.get("path_back") or [None] * len(step["path"])
        return [
            _resolve_view(pixel, back, landmarks, matrices)
            for pixel, back in zip(step["path"], backs)
        ]
    return _resolve_view(step.get(key), step.get(key + "_back"), landmarks, matrices)


def _grip_world(step, landmarks, matrices):
    """返回两个落点并校验水平开度，避免错误点对被中点掩盖后继续执行。"""
    first = np.asarray(step_point(step, "pixel", landmarks, matrices), dtype=float)
    second = np.asarray(step_point(step, "pixel_2", landmarks, matrices), dtype=float)
    axis = second - first
    axis[2] = 0.0
    width = float(np.linalg.norm(axis))
    if not np.isfinite(width) or not MIN_GRIP_WIDTH <= width <= MAX_GRIP_WIDTH:
        raise ParseError("两个落点的水平间距 {:.3f} 米，夹爪夹不了".format(width))
    return first, second


def _grip_ori(first, second, current):
    """由落点连线确定竖直抓取的 yaw；两个对称朝向里选离当前更近的，避免无谓翻转。"""
    axis = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    axis[2] = 0.0
    axis /= np.linalg.norm(axis)
    yaw = float(np.degrees(np.arctan2(-axis[0], axis[1])))
    ori = topdown_ori(yaw)
    flipped = topdown_ori(yaw + 180.0)
    if float(np.dot(flipped[:, 0], current[:, 0])) > float(np.dot(ori[:, 0], current[:, 0])):
        return flipped
    return ori


def _dest_for(step, landmarks, matrices, grasp_point, grasp_ori, ori):
    world = clip_workspace(step_point(step, "pixel", landmarks, matrices))
    if "pixel_2" in step:
        # target_1、target_2 是两指合拢时的两个落点，两点的中点才是要到的位置。
        first, second = _grip_world(step, landmarks, matrices)
        world = 0.5 * (clip_workspace(first) + clip_workspace(second))
    kind = step["type"]
    if kind == "move_above":
        extra = float(step["clearance"])
    elif kind == "move_down":
        extra = float(step["offset_z"])
    else:
        extra = float(step["delta_z"])
    dest = world + np.array([0.0, 0.0, extra])
    if "held" in step:
        held = step_point(step, "held", landmarks, matrices)
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


def execute_plan(env, steps, hold_ori, landmarks, matrices, grip=-1.0, on_step=None, on_progress=None):
    """按步骤顺序执行。步骤带 world（评测按深度图定好的坐标）时直接用；
    否则用 landmarks（采集时提问那一帧的物体中心）反投影。执行过程中都不更新。
    matrices 是每台相机的 world_to_pixel 矩阵，被 mask 的像素换另一台相机的还原。

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
        entry = {"type": kind}
        if kind in SPATIAL:
            if "yaw" in step:
                # 任务给了明确 yaw 时优先使用，它通常还考虑了手腕限位和后续搬运。
                ori = topdown_ori(step["yaw"])
            elif "pixel_2" in step:
                first, second = _grip_world(step, landmarks, matrices)
                ori = _grip_ori(first, second, ori)
            dest = _dest_for(step, landmarks, matrices, grasp_point, grasp_ori, ori)
            target_axis = np.asarray(ori, dtype=float)[:, 0]
            entry["target_pos"] = np.round(dest, 4).tolist()
            entry["target_yaw_deg"] = round(float(np.degrees(np.arctan2(-target_axis[0], target_axis[1]))), 2)
            if "pixel_2" in step:
                first, second = _grip_world(step, landmarks, matrices)
                entry["grip_width_m"] = round(float(np.linalg.norm((second - first)[:2])), 4)
            if kind == "move_above":
                tol, limit, command = MOVE_ABOVE_TOL, MOVE_ABOVE_STEPS, grip
            elif kind == "move_down":
                tol = PLACE_DOWN_TOL if float(step["offset_z"]) >= 0.02 else MOVE_DOWN_TOL
                limit, command = MOVE_DOWN_STEPS, grip
            else:
                tol, limit, command = LIFT_TOL, LIFT_STEPS, 1.0
            obs, ok, err = goto(env, dest, ori, command, tol, limit, on_step)
            arm = arm_controller(env)
            actual_axis = np.asarray(arm.ref_ori_mat, dtype=float)[:, 0]
            entry["actual_pos"] = np.round(arm.ref_pos, 4).tolist()
            entry["actual_yaw_deg"] = round(float(np.degrees(np.arctan2(-actual_axis[0], actual_axis[1]))), 2)
            entry["ori_err_deg"] = round(float(np.degrees(_ori_angle(arm.ref_ori_mat, ori))), 2)
            anchor = dest
            grip = command
        elif kind == "reach":
            ori = approach_ori(step["approach"])
            target = step_point(step, "pixel", landmarks, matrices)
            dest = clip_workspace(target - np.asarray(step["approach"]) * float(step["standoff"]))
            obs, ok, err = goto(env, dest, ori, grip, REACH_TOL, REACH_STEPS, on_step)
            anchor = dest
        elif kind == "turn":
            pivot = step_point(step, "pivot", landmarks, matrices)
            obs, ok, err, ori, anchor = _turn(env, pivot, step["axis"], step["angle"], ori, grip, on_step)
        elif kind == "wipe":
            ok, err = True, 0.0
            for point in step_point(step, "path", landmarks, matrices):
                anchor = clip_workspace(np.asarray(point) + np.array([0.0, 0.0, float(step["offset_z"])]))
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
        entry.update({"ok": bool(ok), "err_m": None if err is None else round(float(err), 4)})
        trace.append(entry)
    return obs, trace


def check_parser():
    """不启动仿真，确认各任务都是只有 subtasks，而且每步可以有自己的像素。
    还确认 target_1/target_2（两夹爪落点）能解析、被 mask 的视图解析成 None、
    desc（说明文字）被忽略、旧名 target 仍兼容。"""
    stack = format_plan(
        [
            {"type": "move_above", "desc": "移到红块上方", "target_1": "[210 300]", "clearance": 0.12},
            {"type": "move_down", "desc": "降到红块处", "target_1": "[210 300]", "offset_z": 0.0},
            {"type": "close_gripper", "desc": "合爪夹住红块"},
            {"type": "lift", "desc": "抬起红块", "target_1": "[210 300]", "delta_z": 0.1},
            {"type": "move_above", "desc": "移到绿块上方", "target_1": "[320 280]", "clearance": 0.12},
            {"type": "move_down", "desc": "放到绿块上", "target_1": "[320 280]", "offset_z": 0.041},
            {"type": "open_gripper", "desc": "松爪放下"},
            {"type": "move_above", "desc": "向上抬开", "target_1": "[320 280]", "clearance": 0.12},
            {"type": "stabilize", "desc": "停在原地", "duration": 0.5},
        ],
    )
    plan = parse_plan(stack, image_size=512)
    raw = json.loads(stack)
    assert "task" not in raw and "desc" not in raw
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
    assert all("desc" not in step for step in plan["subtasks"])
    assert np.allclose(plan["subtasks"][0]["pixel"], [210, 300])
    assert np.allclose(plan["subtasks"][4]["pixel"], [320, 280])
    assert abs(plan["subtasks"][5]["offset_z"] - 0.041) < 1e-9
    lift = (
        '{"desc": "两落点的中点是抓取位置", "subtasks": ['
        '{"type": "move_down", "desc": "降", "target_1": "[282 314]", "target_1_back": "mask", "target_2": "mask", "target_2_back": "[301 318]"}, '
        '{"type": "close_gripper", "desc": "合爪"}, '
        '{"type": "lift", "desc": "抬", "target_1": "[282 314]", "target_2": "[301 318]", "target_2_back": "[265 250]", "delta_z": 0.14}, '
        '{"type": "stabilize", "duration": 0.5}]}'
    )
    plan = parse_plan(lift, image_size=512)
    assert np.allclose(plan["subtasks"][0]["pixel"], [282, 314])
    assert plan["subtasks"][0]["pixel_back"] is None
    assert plan["subtasks"][0]["pixel_2"] is None
    assert np.allclose(plan["subtasks"][0]["pixel_2_back"], [301, 318])
    assert np.allclose(plan["subtasks"][2]["pixel_2"], [301, 318])
    assert np.allclose(plan["subtasks"][2]["pixel_2_back"], [265, 250])
    assert abs(plan["subtasks"][2]["delta_z"] - 0.14) < 1e-9
    legacy = parse_plan(
        '{"subtasks": [{"type": "move_down", "target": "[282 314]"}]}', image_size=512
    )["subtasks"][0]
    assert np.allclose(legacy["pixel"], [282, 314]) and "pixel_back" not in legacy
    assert parse_plan('{"subtasks": []}', image_size=512)["subtasks"] == []
    others = format_plan(
        [
            {"type": "move_above", "target_1": "[200 310]", "target_1_back": "[150 262]", "target_kind": "object", "clearance": 0.1, "yaw": -35},
            {"type": "move_down", "target_1": "[250 300]", "target_kind": "bin", "offset_z": -0.09, "held": "[240 305]", "held_back": "[190 268]", "held_kind": "object"},
            {"type": "reach", "target_1": "[150 240]", "target_kind": "door_handle", "approach": [-0.14, -0.99, 0.0], "standoff": 0.1},
            {"type": "turn", "pivot": "[160 250]", "pivot_back": "[120 210]", "pivot_kind": "door_axis", "axis": [0.0, 0.0, 1.0], "angle": 23},
            {"type": "wipe", "path": ["[200 300]", "mask"], "path_back": ["[150 250]", "[160 255]"], "path_kind": "table", "offset_z": -0.01},
        ]
    )
    plan = parse_plan(others, image_size=512)["subtasks"]
    assert abs(plan[0]["yaw"] + 35.0) < 1e-9
    assert np.allclose(plan[0]["pixel_back"], [150, 262])
    assert plan[0]["pixel_kind"] == "object"
    assert np.allclose(plan[1]["held"], [240, 305])
    assert np.allclose(plan[1]["held_back"], [190, 268])
    assert plan[1]["pixel_kind"] == "bin" and plan[1]["held_kind"] == "object"
    assert abs(np.linalg.norm(plan[2]["approach"]) - 1.0) < 1e-9
    assert "pixel_back" not in plan[2]
    assert plan[2]["pixel_kind"] == "door_handle"
    assert np.allclose(plan[3]["axis"], [0.0, 0.0, 1.0]) and abs(plan[3]["angle"] - 23.0) < 1e-9
    assert np.allclose(plan[3]["pivot_back"], [120, 210])
    assert plan[3]["pivot_kind"] == "door_axis"
    assert len(plan[4]["path"]) == 2
    assert plan[4]["path"][1] is None
    assert np.allclose(plan[4]["path_back"][1], [160, 255])
    assert plan[4]["path_kind"] == "table"
    assert np.allclose(topdown_ori(0.0), TOPDOWN)
    pair_step = {
        "world": {
            "pixel": np.array([-0.02, 0.0, 0.83]),
            "pixel_2": np.array([0.02, 0.0, 0.83]),
        }
    }
    first, second = _grip_world(pair_step, [], {})
    assert np.allclose(_grip_ori(first, second, topdown_ori(-90.0))[:, 0], [1.0, 0.0, 0.0])
    assert np.allclose(_grip_ori(first, second, topdown_ori(90.0))[:, 0], [-1.0, 0.0, 0.0])
    pair_step["world"]["pixel_2"] = np.array([0.2, 0.0, 0.83])
    try:
        _grip_world(pair_step, [], {})
    except ParseError:
        pass
    else:
        raise AssertionError("超过夹爪开度的落点对应被拒绝")
    for bad in (
        '{"subtasks": [{"type": "fly"}]}',
        '{"subtasks": [{"type": "wipe"}]}',
        '{"subtasks": [{"type": "wipe", "path": ["[1 2]"], "path_back": ["[1 2]", "[3 4]"]}]}',
        '{"subtasks": [{"type": "lift", "target": "[1 2]", "delta_z": 0.1, "target_kind": 3}]}',
        '{"subtasks": [{"type": "lift", "target_2": "[1 2]", "delta_z": 0.1}]}',
        '{"subtasks": [{"type": "move_down", "target_1": "mask"}]}',
        '{"subtasks": [{"type": "move_down", "target_1": "mask", "target_1_back": "mask"}]}',
    ):
        try:
            parse_plan(bad, image_size=512)
        except ParseError:
            pass
        else:
            raise AssertionError("无效步骤应被拒绝：{}".format(bad))
