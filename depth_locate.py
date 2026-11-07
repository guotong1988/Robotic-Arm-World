"""评测时把模型给的像素换成世界坐标，只用当前深度图，不读仿真物体真值。

像素决定一条视线，当前点云决定要相交的表面。标注点可能在物体内部、被遮挡，
或落在螺母中心的洞里，因此不能直接把中心像素的深度当作目标点。

每个像素必须由模型在当前画面上用 target_kind、align_point_kind、pivot_kind、path_kind
注明角色（取值见 LOCATORS）。缺少类别或类别未知时明确拒绝，不按任务名猜测。

- object、cube_center、nut：从当前邻域点云估计支撑层和可见顶面，中心在两者之间
- bin：从当前邻域水平层估计箱底，不使用预设箱壁高度
- peg：当前邻域最高的水平层
- table：当前邻域的桌面
- door_axis、door_handle：分别使用当前门板、把手可见竖直表面，不使用预设板厚或把手偏移
- grip：竖直抓取的两条像素射线与当前第一个可见接触面的水平面求交；
  reach 上下侧抓的两条射线与当前把手可见竖直平面求交。恢复失败时不使用固定宽度兜底

所有像素都来自 frontview，定位直接使用这台相机的当前深度图。
"""

import numpy as np
from robosuite.utils.camera_utils import get_real_depth_map

from collect_lift_mllm import (
    CAMERAS,
    DEFAULT_CAMERA,
    project_world,
    unproject_pixel,
)
from task_plan import MAX_GRIP_WIDTH, MIN_GRIP_WIDTH, ParseError, plausible

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
# 从高往低扫中心高度的步长，米；柱子里至少要有这么多点才算量到了顶面。
SCAN_STEP = 0.0025
MIN_COLUMN_POINTS = 5
# 自洽顶面搜索允许的高度容差，米。
VISIBLE_TOLERANCE = 0.005


def real_depth(env, obs, camera=DEFAULT_CAMERA):
    """指定相机的深度图，米。默认 frontview。"""
    raw = np.asarray(obs[camera + "_depth"], dtype=float)
    depth = np.asarray(get_real_depth_map(env.sim, raw), dtype=float)
    return depth.reshape(depth.shape[0], depth.shape[1])


def make_views(env, obs, matrices, cameras=CAMERAS):
    """{相机名: (深度图, 世界到像素矩阵)}，locate_plan / relocate 的输入。"""
    return {name: (real_depth(env, obs, name), matrices[name]) for name in cameras if name in matrices}


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


def locate_object(pixel, disk, matrix, image_size, _params):
    """从当前点云找满足 center=(support+top)/2 的水平面。

    中心像素落在洞里而直接看到支撑面时，从局部邻域取顶面提示；其余情况从
    中心可见表面往下扫。整个估计只依赖当前深度，不使用物体预设高度。
    """
    points = disk(_radius(SUPPORT_RADIUS, image_size))
    support = lowest_surface(points)
    above = points[points[:, 2] > support + SURFACE_THICKNESS]
    visible_z = float(np.median(disk(1, merged=False)[:, 2]))
    highest = visible_z + VISIBLE_TOLERANCE
    if visible_z <= support + SURFACE_THICKNESS:
        local = disk(_radius(LOCAL_RADIUS, image_size))
        highest = max(highest, float(np.percentile(local[:, 2], TOP_PERCENTILE)) + VISIBLE_TOLERANCE)
    best = None
    for center_z in np.arange(highest, support, -SCAN_STEP):
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
        # 中心孔可能让整个中心柱都看到支撑面，而可见顶面位于孔外。只在中心柱
        # 失败时读取当前局部邻域的水平层，不假定孔径、物体高度或顶面厚度。
        local = disk(_radius(LOCAL_RADIUS, image_size))
        above = local[local[:, 2] > support + SURFACE_THICKNESS]
        if len(above) < MIN_COLUMN_POINTS:
            raise ParseError("当前点云无法同时估计支撑面和物体顶面")
        top = densest_layer(above[:, 2])
        method = "local_visible_top"
    else:
        _, top = best
        method = "center_column"
    return UP, 0.5 * (support + top), {
        "support_z": round(support, 4),
        "top_z": round(top, 4),
        "method": method,
    }


def locate_bin(pixel, disk, matrix, image_size, _params):
    points = disk(_radius(SUPPORT_RADIUS, image_size))
    bottom = lowest_surface(points)
    return UP, bottom, {"bottom_z": round(bottom, 4)}


def locate_peg(pixel, disk, matrix, image_size, _params):
    points = disk(_radius(LOCAL_RADIUS, image_size) // 2)
    top = float(np.percentile(points[:, 2], TOP_PERCENTILE))
    return UP, top, {"top_z": round(top, 4)}


def locate_table(pixel, disk, matrix, image_size, _params):
    # 小窗口可能完全被机械臂等前景占满；在更大的当前局部邻域里，连续桌面
    # 通常形成最密的水平层。只估计当帧平面，不使用预设桌高或遮挡物尺寸。
    points = disk(_radius(SUPPORT_RADIUS, image_size))
    table = densest_layer(points[:, 2])
    return UP, table, {"table_z": round(table, 4), "method": "densest_local_layer"}


def _door_plane(pixel, disk, matrix, image_size, params, nearest=False):
    """从当前点云估计竖直表面；把手取目标处可见面，转轴取门板最密层。"""
    normal = params.get("normal")
    if normal is None:
        raise ParseError("计划里没有竖直表面的法向（reach 的 approach）")
    normal = np.asarray(normal, dtype=float)
    normal = normal - float(np.dot(normal, UP)) * UP
    if float(np.linalg.norm(normal)) < 1e-6:
        raise ParseError("竖直表面的法向无效")
    normal = normal / np.linalg.norm(normal)
    if nearest:
        points = disk(1, merged=False)
        face = float(np.median(points @ normal))
        method = "visible_handle_plane"
    else:
        points = disk(_radius(DOOR_RADIUS, image_size))
        face = densest_layer(points @ normal)
        method = "visible_panel_plane"
    if not np.isfinite(face):
        raise ParseError("当前点云无法估计竖直表面")
    return normal, face, {"face": round(face, 4), "method": method}


def locate_door_axis(pixel, disk, matrix, image_size, params):
    return _door_plane(pixel, disk, matrix, image_size, params)


def locate_door_handle(pixel, disk, matrix, image_size, params):
    return _door_plane(pixel, disk, matrix, image_size, params, nearest=True)


# 每个角色一种量法。模型在计划里写的 *_kind 类别，取值就是这里的名字。
LOCATORS = {
    "object": locate_object,
    "cube_center": locate_object,
    "bin": locate_bin,
    "nut": locate_object,
    "peg": locate_peg,
    "table": locate_table,
    "door_axis": locate_door_axis,
    "door_handle": locate_door_handle,
}




def depth_median(depth, pixel, radius=4):
    """像素附近一圈深度的中位数，米。躲开单个像素打到缝上或背景上。"""
    depth = np.asarray(depth, dtype=float)
    height, width = depth.shape
    u, v = int(np.rint(pixel[0])), int(np.rint(pixel[1]))
    if not (0 <= u < width and 0 <= v < height):
        raise ParseError("像素不在画面里：[{:.0f} {:.0f}]".format(pixel[0], pixel[1]))
    patch = depth[max(0, v - radius) : v + radius + 1, max(0, u - radius) : u + radius + 1]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 4.0)]
    if len(valid) == 0:
        raise ParseError("像素附近没有深度")
    return float(np.median(valid))


def surface_point(pixel, depth, matrix, radius=4):
    """像素配上附近的深度，得到视线碰到的那个表面点。"""
    world = unproject_pixel(pixel, depth_median(depth, pixel, radius), matrix)
    if not plausible(world):
        raise ParseError("反投影超出工作空间：{}".format(np.round(world, 4).tolist()))
    return world


def landing_pair_ok(front, back):
    """两个竖直抓取落点的水平间距必须在夹爪物理开度内。"""
    delta = np.asarray(back, dtype=float) - np.asarray(front, dtype=float)
    delta[2] = 0.0
    width = float(np.linalg.norm(delta))
    return MIN_GRIP_WIDTH <= width <= MAX_GRIP_WIDTH


def side_landing_pair_ok(first, second):
    """侧抓使用三维开度，并要求 target_1 在上、target_2 在下。"""
    delta = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    width = float(np.linalg.norm(delta))
    return MIN_GRIP_WIDTH <= width <= MAX_GRIP_WIDTH and float(delta[2]) < -0.5 * width


def recover_landings(pixels, views, image_size, axis=None):
    """两条当前像素射线与第一个可见接触面的共同水平面求交。

    axis 只用于记录当前合拢轴与恢复轴的符号关系，不参与位置计算。
    任一点无效或开度超过机器人能力时明确失败，不借用预设物体尺寸。
    """
    front_px = pixels.get("front")
    same_px = pixels.get("same")
    if front_px is None or same_px is None or DEFAULT_CAMERA not in views:
        raise ParseError("竖直抓取缺少 frontview 双落点")
    depth, matrix = views[DEFAULT_CAMERA]
    visible = surface_point(front_px, depth, matrix, radius=1)
    plane_z = float(visible[2])
    first = ray_at_height(front_px, plane_z, matrix)
    second = ray_at_height(same_px, plane_z, matrix)
    if not plausible(first) or not plausible(second) or not landing_pair_ok(first, second):
        raise ParseError("当前双落点恢复结果超出工作空间或夹爪开度")
    info = {"mode": "mono", "method": "visible_plane", "plane_z": round(plane_z, 4)}
    if axis is not None:
        axis = np.asarray(axis, dtype=float).reshape(3)
        axis[2] = 0.0
        delta = second - first
        delta[2] = 0.0
        if float(np.linalg.norm(axis)) > 1e-6 and float(np.linalg.norm(delta)) > 1e-6:
            info["axis_dot"] = round(float(axis @ delta / (np.linalg.norm(axis) * np.linalg.norm(delta))), 4)
    return first, second, info


def recover_side_landings(pixels, views, image_size, approach):
    """恢复 Door 上下侧抓落点：两条像素射线与同一个把手竖直平面求交。

    不能直接读两个像素的可见表面深度：落点在把手上下表面，背面和门板遮挡会让
    可见深度不代表接触点。平面量法与单点 door_handle 相同，但两个点共用一次平面估计。
    """
    front_px = pixels.get("front")
    same_px = pixels.get("same")
    if front_px is None or same_px is None or DEFAULT_CAMERA not in views:
        raise ParseError("Door 侧抓缺少 frontview 双落点")
    depth, matrix = views[DEFAULT_CAMERA]
    center_px = 0.5 * (np.asarray(front_px, dtype=float) + np.asarray(same_px, dtype=float))

    def disk(radius, merged=True):
        return backproject_disk(depth, matrix, center_px, radius)

    normal, offset, info = _door_plane(
        center_px, disk, matrix, image_size, {"normal": np.asarray(approach, dtype=float)}, nearest=True
    )
    first = ray_at_plane(front_px, normal, offset, matrix)
    second = ray_at_plane(same_px, normal, offset, matrix)
    if not plausible(first) or not plausible(second) or not side_landing_pair_ok(first, second):
        raise ParseError("Door 侧抓双落点恢复后的间距或上下顺序无效")
    return first, second, dict(info, mode="mono", method="door_handle_plane", view=DEFAULT_CAMERA)


def locate(role, pixel, depth, matrix, image_size, params=None):
    """单视图定位：按角色量出标注点所在的平面，返回视线和它的交点，以及量测记录。"""
    pixel = np.asarray(pixel, dtype=float)

    def disk(radius, merged=True):
        return backproject_disk(depth, matrix, pixel, radius)

    normal, offset, info = LOCATORS[role](pixel, disk, matrix, image_size, params or {})
    world = ray_at_plane(pixel, normal, offset, matrix)
    if not plausible(world):
        raise ParseError("反投影超出工作空间：{}".format(np.round(world, 4).tolist()))
    return world, dict(info, role=role)


def measure_views(role, pixels, views, image_size, params=None):
    """用 frontview 像素和深度定位一个标注点。"""
    pixel = pixels.get(DEFAULT_CAMERA)
    if pixel is None or DEFAULT_CAMERA not in views:
        raise ParseError("模型没给出 frontview 像素")
    depth, matrix = views[DEFAULT_CAMERA]
    world, info = locate(role, pixel, depth, matrix, image_size, params)
    return world, dict(info, view=DEFAULT_CAMERA)


def _step_roles_full(subtasks):
    """读取模型为当前画面标注的角色；缺失、未知或双落点类别错误都拒绝。"""
    out = []

    def apply(step, key):
        role = step.get(key + "_kind")
        if role is None:
            raise ParseError("{} 缺少 {}_kind 视觉类别".format(step["type"], key))
        if role not in LOCATORS and role != "grip":
            raise ParseError("未知视觉类别：{}".format(role))
        extra = {"kind": role}
        if "pixel_2" in step and key == "pixel":
            if role != "grip":
                raise ParseError("双落点步骤的 target_kind 必须是 grip")
            extra["pair"] = True
        elif role == "grip":
            raise ParseError("grip 类别必须同时提供 target_1 和 target_2")
        return role, extra

    for step in subtasks:
        kind = step["type"]
        roles = {}
        if kind in ("move_above", "move_down", "lift", "reach"):
            roles["pixel"] = apply(step, "pixel")
            if "align_point" in step:
                roles["align_point"] = apply(step, "align_point")
        elif kind == "turn":
            roles["pivot"] = apply(step, "pivot")
        elif kind == "wipe":
            roles["path"] = apply(step, "path")
        out.append(roles)
    return out


def step_roles(subtasks):
    """每一步里各个像素的角色，和 subtasks 对齐。"""
    return [{key: role for key, (role, _extra) in roles.items()} for roles in _step_roles_full(subtasks)]


def _pixel_match(left, right):
    return left is not None and right is not None and bool(
        np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    )


def _view_pixels(step, key):
    if step.get(key) is None:
        return {}
    return {DEFAULT_CAMERA: np.asarray(step[key], dtype=float)}


def _pixel_key(pixels):
    return tuple(sorted((camera, tuple(np.round(pixel, 3))) for camera, pixel in pixels.items()))


def _dedupe_key(role, pixels):
    return (role, _pixel_key(pixels))


def grip_pixels(step, views):
    return {
        "front": np.asarray(step["pixel"], dtype=float),
        "same": np.asarray(step["pixel_2"], dtype=float),
    }


def _recover_step_pair(step, pixels, views, image_size, axis):
    """reach 双落点表示通用侧抓，其余双落点表示竖直抓取。"""
    if step["type"] == "reach":
        return recover_side_landings(pixels, views, image_size, step["approach"])
    return recover_landings(pixels, views, image_size, axis)


def _plan_params(subtasks):
    for step in subtasks:
        if step["type"] == "reach":
            return {"normal": np.asarray(step["approach"], dtype=float)}
    return {}


def locate_plan(subtasks, views, image_size, axis=None):
    """给计划里每一步填上 step["world"]（各像素量出来的世界坐标），execute_plan 直接用。

    views 只有 frontview 的 (深度图, 世界到像素矩阵)。所有角色都来自模型回答。
    带 pixel_2 的步骤按 grip 成对还原；reach 用 approach 定位上下侧抓点。
    返回定位过的点，同一角色和像素组合只量一次。
    """
    params = _plan_params(subtasks)
    points = []
    seen = {}

    def measure(pixels, role, extra=None):
        key = _dedupe_key(role, pixels)
        if key in seen:
            return seen[key]
        world, info = measure_views(role, pixels, views, image_size, params)
        if extra:
            info.update(extra)
        seen[key] = world
        points.append(
            {
                "pixel": pixels.get(DEFAULT_CAMERA),
                "role": role,
                "world": world,
                "info": info,
            }
        )
        return world

    for step, roles in zip(subtasks, _step_roles_full(subtasks)):
        world = {}
        for key, (role, extra) in roles.items():
            if key == "path":
                world[key] = [
                    measure({DEFAULT_CAMERA: pixel}, role, extra)
                    for pixel in step["path"]
                ]
            elif role == "grip" and key == "pixel":
                pixels = grip_pixels(step, views)
                key_pair = _dedupe_key(role, pixels)
                if key_pair in seen:
                    first, second = seen[key_pair]
                else:
                    first, second, info = _recover_step_pair(step, pixels, views, image_size, axis)
                    if extra:
                        info.update(extra)
                    seen[key_pair] = (first, second)
                    # 两落点的中点是物体中心，统计误差用它配真值；执行照样用两落点。
                    center = 0.5 * (first + second)
                    points.append(
                        {
                            "pixel": pixels["front"],
                            "pixel_2": pixels["same"],
                            "role": role,
                            "world": first,
                            "world_2": second,
                            "land_1": first,
                            "land_2": second,
                            "center_world": center,
                            "center_pixel": project_world(center, views[DEFAULT_CAMERA][1])[0],
                            "info": info,
                        }
                    )
                world["pixel"] = first
                world["pixel_2"] = second
            else:
                world[key] = measure(_view_pixels(step, key), role, extra)
        step["world"] = world
    return points


MOVABLE = ("object", "cube_center", "nut", "grip")


def pick_starts(subtasks):
    """计划里开始去抓下一个物体的那些步（第一个物体除外）。"""
    roles = step_roles(subtasks)
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


def relocate(subtasks, start, views, image_size, axis=None):
    """抓下一个物体前用当前这一帧的深度重新量它：开局时挡在它前面的东西可能已经搬走了。

    只更新从 start 起指向这个物体的 pixel / align_point；箱子、柱子还用开局量的，
    这时夹爪可能正悬在它们上方。两指落点（grip）成对重测。
    量不出来就保留原值。返回新的量测记录或 None。
    """
    roles = step_roles(subtasks)
    role = roles[start]["pixel"]
    start_step = subtasks[start]
    if role == "grip":
        pixels = grip_pixels(start_step, views)
        try:
            first, second, info = _recover_step_pair(start_step, pixels, views, image_size, axis)
        except ValueError:
            return None
        for step, step_role in zip(subtasks[start:], roles[start:]):
            if step_role.get("pixel") != "grip":
                continue
            same_first = _pixel_match(step.get("pixel"), start_step.get("pixel"))
            same_second = _pixel_match(step.get("pixel_2"), start_step.get("pixel_2"))
            if same_first and same_second:
                step["world"]["pixel"] = first
                step["world"]["pixel_2"] = second
        center = 0.5 * (first + second)
        return {
            "pixel": pixels["front"],
            "role": role,
            "world": first,
            "world_2": second,
            "land_1": first,
            "land_2": second,
            "center_world": center,
            "center_pixel": project_world(center, views[DEFAULT_CAMERA][1])[0],
            "info": info,
        }
    pixels = _view_pixels(start_step, "pixel")
    try:
        world, info = measure_views(role, pixels, views, image_size, _plan_params(subtasks))
    except ValueError:
        return None
    for step, step_role in zip(subtasks[start:], roles[start:]):
        for key in ("pixel", "align_point"):
            if step_role.get(key) != role:
                continue
            if _pixel_match(step.get(key), start_step.get("pixel")):
                step["world"][key] = world
    return {
        "pixel": pixels.get(DEFAULT_CAMERA),
        "role": role,
        "world": world,
        "info": info,
    }
