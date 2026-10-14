"""评测时把模型给的像素换成世界坐标，只用深度相机，不读仿真里的物体位置。

像素只决定一条视线。直接读这个像素的深度不行：深度图给的是视线碰到的第一个表面，
标注点常在物体内部（方块中心）、被挡住（箱壁后面的格子底面、门框立柱后面的门把手），
或是一个洞（螺母中心）。所以不量距离，而是在像素附近取一片深度点，换成世界坐标，
量出标注点所在的那个平面（多数是某个高度的水平面），再加上物体形状的常识
（放在桌上的方块，中心在桌面和顶面中间），取视线和这个平面的交点。

每个像素按它在计划里的角色定位，角色只看计划的结构（比如 lift 指向的是要抓的物体），
不看真值：

- object：放在桌面或箱底的物体中心。支撑面是附近最低的一层水平面，顶面取物体竖直上方
  那一小柱里的最高处，中心在两者中间
- bin：PickPlace 右箱格子底面的中心。底面被箱壁挡住看不到，量箱壁顶面再减去壁高
- nut：平放在桌上的螺母，中心和把手上的抓取点一样高。桌面高度加螺母半厚
- peg：柱顶中心。柱顶是像素附近最高的水平面
- table：贴着桌面的点（Wipe 的污渍）
- door_axis、door_handle：Door 的门锁转轴、铰链在门板中面上，把手抓取点在中面前方。
  门几乎侧对相机，水平面求交很不稳（高度差 1 毫米，水平位置差 5 毫米），改用竖直的门板平面。
  门板法向就是模型给的 approach（门锁转轴方向），沿法向投影后最密的一层点是门板朝相机的那面
"""

import numpy as np
from robosuite.utils.camera_utils import get_real_depth_map

from collect_lift_mllm import DEFAULT_CAMERA, unproject_pixel
from task_plan import ParseError, plausible

UP = np.array([0.0, 0.0, 1.0])
# 圈的半径按 512 边长定，单位像素，其他边长按比例缩放。
SUPPORT_RADIUS = 40
LOCAL_RADIUS = 12
DOOR_RADIUS = 60
# 一层平面的厚度，米。支撑面是大圈里最低的一层，这一层里至少要有这个比例的点。
SURFACE_THICKNESS = 0.01
SUPPORT_FRACTION = 0.15
TOP_PERCENTILE = 95
# 物体顶面只在中心竖直上方这么粗的一柱里找，米。避开旁边的箱壁、别的物体。
COLUMN_RADIUS = 0.02
# 物体高度的合理范围，米。
MIN_HEIGHT = 0.005
MAX_HEIGHT = 0.20
# 从高往低扫中心高度的步长，米；柱子里至少要有这么多点才算量到了顶面。
SCAN_STEP = 0.0025
MIN_COLUMN_POINTS = 5
# 物体中心不会比像素处看到的表面更靠近相机，即不高于那个表面；容差，米。
VISIBLE_TOLERANCE = 0.005
# PickPlace 箱壁顶面相对箱子原点的高度，米。和 collect_pickplace_mllm.WALL_HEIGHT 一致。
BIN_WALL_HEIGHT = 0.10
# 螺母中心比桌面高这么多，米，约为螺母厚度的一半。
NUT_HALF_THICKNESS = 0.009
# 门板厚度的一半；把手抓取点在门板中面前方多远，米。
PANEL_HALF_THICKNESS = 0.02
HANDLE_OFFSET = 0.085


def real_depth(env, obs):
    raw = np.asarray(obs[DEFAULT_CAMERA + "_depth"], dtype=float)
    depth = np.asarray(get_real_depth_map(env.sim, raw), dtype=float)
    return depth.reshape(depth.shape[0], depth.shape[1])


def backproject_disk(depth, matrix, pixel, radius):
    """深度图上以 pixel 为圆心、radius 为半径的像素，反投影成世界坐标点。"""
    height, width = depth.shape
    u0, v0 = int(np.rint(pixel[0])), int(np.rint(pixel[1]))
    if not (0 <= u0 < width and 0 <= v0 < height):
        raise ParseError("像素不在画面里：[{:.0f} {:.0f}]".format(pixel[0], pixel[1]))
    us = np.arange(max(0, u0 - radius), min(width, u0 + radius + 1))
    vs = np.arange(max(0, v0 - radius), min(height, v0 + radius + 1))
    uu, vv = np.meshgrid(us, vs)
    keep = (uu - u0) ** 2 + (vv - v0) ** 2 <= radius * radius
    uu, vv = uu[keep], vv[keep]
    z = depth[vv, uu]
    # 深度图第 (v, u) 格对应的是像素中心 (u + 0.5, v + 0.5)。
    cu, cv = uu + 0.5, vv + 0.5
    cam = np.stack([cu * z, cv * z, z, np.ones_like(z)])
    world = np.linalg.inv(np.asarray(matrix, dtype=float)) @ cam
    return world[:3].T


def ray_at_plane(pixel, normal, offset, matrix):
    """像素对应的视线和平面 normal·p = offset 的交点。"""
    near = unproject_pixel(pixel, 0.5, matrix)
    far = unproject_pixel(pixel, 3.0, matrix)
    direction = far - near
    normal = np.asarray(normal, dtype=float)
    denom = float(np.dot(normal, direction))
    if abs(denom) < 1e-6:
        raise ParseError("视线和平面接近平行，无法求交")
    t = (float(offset) - float(np.dot(normal, near))) / denom
    return near + t * direction


def ray_at_height(pixel, height_z, matrix):
    """像素对应的视线和水平面 z=height_z 的交点。"""
    return ray_at_plane(pixel, UP, height_z, matrix)


def camera_center(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return np.linalg.solve(matrix[:3, :3], -matrix[:3, 3])


def _radius(base, image_size):
    return max(2, int(round(base * float(image_size) / 512.0)))


def _layers(values):
    """把一维的值排好序，返回每个起点往上 SURFACE_THICKNESS 以内的点数和终点。"""
    values = np.sort(values)
    ends = np.searchsorted(values, values + SURFACE_THICKNESS, side="right")
    return values, ends, ends - np.arange(len(values))


def lowest_surface(points):
    """点云里最低的、占够比例的一层水平面的高度。旁边的物体、箱壁都比它高；
    圈子伸出桌边看到的地面虽然更低，但占的比例不够。"""
    z, ends, counts = _layers(points[:, 2])
    hits = np.nonzero(counts >= SUPPORT_FRACTION * len(z))[0]
    if len(hits) == 0:
        return float(np.median(z))
    start = int(hits[0])
    return float(np.median(z[start : ends[start]]))


def densest_layer(values):
    values, ends, counts = _layers(values)
    start = int(np.argmax(counts))
    return float(np.median(values[start : ends[start]]))


def locate_object(pixel, depth, matrix, image_size, _params):
    """中心高度 h 要自洽：视线在高度 h 处的那一柱里，顶面 top 满足 h = (support + top) / 2。
    从像素处可见表面的高度往低扫，取第一个自洽的 h。更低的 h 在视线上更远，那里的柱子常常
    落在物体后面的别的东西上；物体本身是视线先碰到的那个。"""
    points = backproject_disk(depth, matrix, pixel, _radius(SUPPORT_RADIUS, image_size))
    support = lowest_surface(points)
    above = points[(points[:, 2] > support + MIN_HEIGHT) & (points[:, 2] < support + MAX_HEIGHT)]
    visible_z = float(np.median(backproject_disk(depth, matrix, pixel, 1)[:, 2]))
    highest = min(support + 0.5 * MAX_HEIGHT, visible_z + VISIBLE_TOLERANCE)
    best = None
    for center_z in np.arange(highest, support + 0.5 * MIN_HEIGHT, -SCAN_STEP):
        xy = ray_at_height(pixel, center_z, matrix)[:2]
        column = above[np.linalg.norm(above[:, :2] - xy, axis=1) <= COLUMN_RADIUS]
        if len(column) < MIN_COLUMN_POINTS:
            continue
        top = float(np.percentile(column[:, 2], TOP_PERCENTILE))
        residual = abs(0.5 * (support + top) - center_z)
        if best is None or residual < best[0]:
            best = (residual, top)
        if residual <= SCAN_STEP:
            break
    if best is None:
        # 被前面的东西整个挡住时量不到顶面，只能取像素处的可见表面
        center_z = float(np.clip(visible_z, support + MIN_HEIGHT, support + 0.5 * MAX_HEIGHT))
        return UP, center_z, {"support_z": round(support, 4), "top_z": None}
    _, top = best
    return UP, 0.5 * (support + top), {"support_z": round(support, 4), "top_z": round(top, 4)}


def locate_bin(pixel, depth, matrix, image_size, _params):
    points = backproject_disk(depth, matrix, pixel, _radius(SUPPORT_RADIUS, image_size))
    support = lowest_surface(points)
    rims = points[(points[:, 2] > support) & (points[:, 2] < support + 2.0 * BIN_WALL_HEIGHT)]
    rim = float(np.percentile(rims[:, 2], TOP_PERCENTILE)) if len(rims) else support
    return UP, rim - BIN_WALL_HEIGHT, {"rim_z": round(rim, 4)}


def locate_nut(pixel, depth, matrix, image_size, _params):
    points = backproject_disk(depth, matrix, pixel, _radius(SUPPORT_RADIUS, image_size))
    support = lowest_surface(points)
    return UP, support + NUT_HALF_THICKNESS, {"support_z": round(support, 4)}


def locate_peg(pixel, depth, matrix, image_size, _params):
    points = backproject_disk(depth, matrix, pixel, _radius(LOCAL_RADIUS, image_size) // 2)
    top = float(np.percentile(points[:, 2], TOP_PERCENTILE))
    return UP, top, {"top_z": round(top, 4)}


def locate_table(pixel, depth, matrix, image_size, _params):
    points = backproject_disk(depth, matrix, pixel, _radius(LOCAL_RADIUS, image_size))
    table = float(np.median(points[:, 2]))
    return UP, table, {"table_z": round(table, 4)}


def _door_plane(pixel, depth, matrix, image_size, params, offset):
    """门板朝相机那面是附近沿法向投影后最密的一层；中面再往里半个板厚，offset 是相对中面朝相机一侧的距离。"""
    normal = params.get("normal")
    if normal is None:
        raise ParseError("计划里没有门的法向（reach 的 approach）")
    normal = np.asarray(normal, dtype=float)
    normal = normal - float(np.dot(normal, UP)) * UP
    normal = normal / np.linalg.norm(normal)
    points = backproject_disk(depth, matrix, pixel, _radius(DOOR_RADIUS, image_size))
    face = densest_layer(points @ normal)
    toward = 1.0 if float(np.dot(normal, camera_center(matrix))) > face else -1.0
    middle = face - toward * PANEL_HALF_THICKNESS
    return normal, middle + toward * offset, {"face": round(face, 4)}


def locate_door_axis(pixel, depth, matrix, image_size, params):
    return _door_plane(pixel, depth, matrix, image_size, params, 0.0)


def locate_door_handle(pixel, depth, matrix, image_size, params):
    return _door_plane(pixel, depth, matrix, image_size, params, HANDLE_OFFSET)


LOCATORS = {
    "object": locate_object,
    "bin": locate_bin,
    "nut": locate_nut,
    "peg": locate_peg,
    "table": locate_table,
    "door_axis": locate_door_axis,
    "door_handle": locate_door_handle,
}


def locate(role, pixel, depth, matrix, image_size, params=None):
    """按角色量出标注点所在的平面，返回视线和它的交点，以及量测记录。"""
    pixel = np.asarray(pixel, dtype=float)
    normal, offset, info = LOCATORS[role](pixel, depth, matrix, image_size, params or {})
    world = ray_at_plane(pixel, normal, offset, matrix)
    if not plausible(world):
        raise ParseError("反投影超出工作空间：{}".format(np.round(world, 4).tolist()))
    return world, dict(info, role=role)


def _role(task_name, pick, kind):
    """pick 表示这个像素指的是要抓的东西（或手里的东西在提问那一帧的位置），否则是放置目标。"""
    if task_name.startswith("pickplace"):
        return "object" if pick else "bin"
    if task_name.startswith("nutassembly"):
        return "nut" if pick else "peg"
    if task_name == "door":
        return "door_handle" if kind == "reach" else "door_axis"
    if task_name == "wipe":
        return "table"
    return "object"


def step_roles(task_name, subtasks):
    """每一步里各个像素的角色，和 subtasks 对齐，形如 {"pixel": "object", "held": "nut"}。

    合爪之前指向的是要抓的东西；合爪之后到松爪之前指向放置目标，held 仍是手里的东西；
    lift 指的是正抓着的那个东西。只看步骤顺序，不看真值，也不靠像素远近判断：
    螺母中心和柱顶在画面里可能只差一两个像素。
    """
    out = []
    carrying = False
    placed = []
    for step in subtasks:
        kind = step["type"]
        roles = {}
        if kind == "close_gripper":
            carrying = True
        elif kind == "open_gripper":
            carrying = False
        elif kind in ("move_above", "move_down", "reach"):
            # 松爪后退开的那一步还指着刚放下的位置
            place = carrying or any(np.allclose(step["pixel"], other) for other in placed)
            if place:
                placed.append(np.asarray(step["pixel"], dtype=float))
            roles["pixel"] = _role(task_name, not place, kind)
            if "held" in step:
                roles["held"] = _role(task_name, True, kind)
        elif kind == "lift":
            roles["pixel"] = _role(task_name, True, kind)
        elif kind == "turn":
            roles["pivot"] = _role(task_name, True, kind)
        elif kind == "wipe":
            roles["path"] = _role(task_name, True, kind)
        out.append(roles)
    return out


def locate_plan(task_name, subtasks, depth, matrix, image_size):
    """给计划里每一步填上 step["world"]（各像素量出来的世界坐标），execute_plan 直接用。

    返回定位过的点 [{"pixel", "role", "world", "info"}]，同一像素同一角色只量一次，给统计用。
    Door 的门板法向取计划里第一个 reach 的 approach。
    """
    params = {}
    for step in subtasks:
        if step["type"] == "reach":
            params["normal"] = np.asarray(step["approach"], dtype=float)
            break
    points = []

    def measure(pixel, role):
        pixel = np.asarray(pixel, dtype=float)
        for point in points:
            if point["role"] == role and np.allclose(point["pixel"], pixel):
                return point["world"]
        world, info = locate(role, pixel, depth, matrix, image_size, params)
        points.append({"pixel": pixel, "role": role, "world": world, "info": info})
        return world

    for step, roles in zip(subtasks, step_roles(task_name, subtasks)):
        world = {}
        for key, role in roles.items():
            if key == "path":
                world[key] = [measure(pixel, role) for pixel in step["path"]]
            else:
                world[key] = measure(step[key], role)
        step["world"] = world
    return points


MOVABLE = ("object", "nut")


def pick_starts(task_name, subtasks):
    """计划里开始去抓下一个物体的那些步（第一个物体除外）。"""
    roles = step_roles(task_name, subtasks)
    starts = []
    released = False
    for index, (step, step_roles_) in enumerate(zip(subtasks, roles)):
        if step["type"] == "open_gripper":
            released = True
        elif step["type"] == "close_gripper":
            released = False
        elif released and step["type"] in ("move_above", "move_down", "reach") and step_roles_["pixel"] in MOVABLE:
            starts.append(index)
            released = False
    return starts


def relocate(task_name, subtasks, start, depth, matrix, image_size):
    """抓下一个物体前用当前这一帧的深度重新量它：开局时挡在它前面的东西可能已经搬走了。

    只更新从 start 起指向这个物体的 pixel / held；箱子、柱子还用开局量的，
    这时夹爪可能正悬在它们上方。量不出来就保留原值。返回新的量测记录或 None。
    """
    roles = step_roles(task_name, subtasks)
    pixel = np.asarray(subtasks[start]["pixel"], dtype=float)
    role = roles[start]["pixel"]
    try:
        world, info = locate(role, pixel, depth, matrix, image_size)
    except ValueError:
        return None
    for step, step_role in zip(subtasks[start:], roles[start:]):
        for key in ("pixel", "held"):
            if step_role.get(key) == role and np.allclose(step[key], pixel):
                step["world"][key] = world
    return {"pixel": pixel, "role": role, "world": world, "info": info}
