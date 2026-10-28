"""评测时把模型给的像素换成世界坐标，只用深度相机，不读仿真里的物体位置。

像素只决定一条视线。直接读这个像素的深度不行：深度图给的是视线碰到的第一个表面，
标注点常在物体内部（方块中心）、被挡住（箱壁后面的格子底面、门框立柱后面的门把手），
或是一个洞（螺母中心）。所以不量距离，而是在像素附近取一片深度点，换成世界坐标，
量出标注点所在的那个平面（多数是某个高度的水平面），再加上物体形状的常识
（放在桌上的方块，中心在桌面和顶面中间），取视线和这个平面的交点。

每个像素按角色定位。模型认得指的是哪类东西时，在计划里用 target_kind、held_kind、
pivot_kind、path_kind 写明角色（取值就是 LOCATORS 的名字），优先按它选量法，
depth_locate 不用按任务名硬编码；没写或写了不认识的类别时，退回按任务名和计划结构推断
（比如 lift 指向的是要抓的物体）。不看真值：

- object：放在桌面或箱底的物体中心。支撑面是附近最低的一层水平面，顶面取物体竖直上方
  那一小柱里的最高处，中心在两者中间
- cube_center：方块的中心（Stack 里被叠的那块）。量法和 object 相同
- bin：PickPlace 右箱格子底面的中心。底面被箱壁挡住看不到，量箱壁顶面再减去壁高
- nut：平放在桌上的螺母，中心和把手上的抓取点一样高。桌面高度加螺母半厚
- peg：柱顶中心。柱顶是像素附近最高的水平面
- table：贴着桌面的点（Wipe 的污渍）
- door_axis、door_handle：Door 的门锁转轴、铰链在门板中面上，把手抓取点在中面前方。
  门几乎侧对相机，水平面求交很不稳（高度差 1 毫米，水平位置差 5 毫米），改用竖直的门板平面。
  门板法向就是模型给的 approach（门锁转轴方向），沿法向投影后最密的一层点是门板朝相机的那面
- grip：两指落点是一对，用 recover_landings 成对还原：双目各看到一侧时分别读表面深度；
  单目（或落点在背面相机被 mask）时，Nut 用支撑面加半厚、PickPlace 结合物品顶面先验
  和可见指腹表面确定共同水平面，补出另一条视线上的落点；其他任务退回落点附近点云。执行取两点中点

每个标注点有两个视图：frontview（平视）和 backview（在 frontview 正对面）。
模型在两个视图里各给一个像素，被遮挡的视图写 "mask"（解析成 None），定位只用看得到的视图。
两个视图都有像素时，定位分两层：

- 测量级融合（bin、nut、peg、table）：两个视图的邻域点云都反投影到世界坐标系合成一片，
  在这片合并点云上量支撑面、顶面、中位数。单个视图被手臂等遮挡时，污染点是少数，
  被多数派统计自动投出去。模型在两个视图里指的不是同一个点（视线最近距离超过 1 厘米）
  时不合并，退回逐视图融合。
- 决策级融合（object、door_axis、door_handle）：各视图独立定位再融合。
  object 的顶面沿主视线窄柱搜索，柱内遮挡不会被多数派投出；门板是竖直平面。
  两台相机平视对称，主视图统一是 frontview，分歧超过 3 厘米时
  取 z 低的那一路（遮挡物都在桌面上方，被挡的视图量出来的点偏高）。

视线求交用主视图的像素；主视图量不出来或被 mask 时用另一视图兜底。
disagreement_m（两视图独立定位的距离）和 ray_gap_m（两条视线的距离）记进 info 作一致性校验。
"""

import numpy as np
from robosuite.utils.camera_utils import get_real_depth_map

from collect_lift_mllm import (
    BACK_CAMERA,
    CAMERAS,
    DEFAULT_CAMERA,
    project_world,
    unproject_pixel,
)
from task_plan import ParseError, plausible, topdown_ori

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
# PickPlace 指尖落在物品顶面以下的深度，和采集的 GRASP_DEPTH 一致。
PICKPLACE_GRASP_DEPTH = 0.03
# 螺母中心比桌面高这么多，米，约为螺母厚度的一半。
NUT_HALF_THICKNESS = 0.009
# 门板厚度的一半；把手抓取点在门板中面前方多远，米。
PANEL_HALF_THICKNESS = 0.02
HANDLE_OFFSET = 0.085


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
    """中心高度 h 要自洽：视线在高度 h 处的那一柱里，顶面 top 满足 h = (support + top) / 2。
    从像素处可见表面的高度往低扫，取第一个自洽的 h。更低的 h 在视线上更远，那里的柱子常常
    落在物体后面的别的东西上；物体本身是视线先碰到的那个。

    disk(radius) 取像素周围的点云（多视图时已合并）；disk(1, merged=False) 只取主视线那一路。"""
    points = disk(_radius(SUPPORT_RADIUS, image_size))
    support = lowest_surface(points)
    above = points[(points[:, 2] > support + MIN_HEIGHT) & (points[:, 2] < support + MAX_HEIGHT)]
    visible_z = float(np.median(disk(1, merged=False)[:, 2]))
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


def locate_bin(pixel, disk, matrix, image_size, _params):
    points = disk(_radius(SUPPORT_RADIUS, image_size))
    support = lowest_surface(points)
    rims = points[(points[:, 2] > support) & (points[:, 2] < support + 2.0 * BIN_WALL_HEIGHT)]
    rim = float(np.percentile(rims[:, 2], TOP_PERCENTILE)) if len(rims) else support
    return UP, rim - BIN_WALL_HEIGHT, {"rim_z": round(rim, 4)}


def locate_nut(pixel, disk, matrix, image_size, _params):
    points = disk(_radius(SUPPORT_RADIUS, image_size))
    support = lowest_surface(points)
    return UP, support + NUT_HALF_THICKNESS, {"support_z": round(support, 4)}


def locate_peg(pixel, disk, matrix, image_size, _params):
    points = disk(_radius(LOCAL_RADIUS, image_size) // 2)
    top = float(np.percentile(points[:, 2], TOP_PERCENTILE))
    return UP, top, {"top_z": round(top, 4)}


def locate_table(pixel, disk, matrix, image_size, _params):
    points = disk(_radius(LOCAL_RADIUS, image_size))
    table = float(np.median(points[:, 2]))
    return UP, table, {"table_z": round(table, 4)}


def _door_plane(pixel, disk, matrix, image_size, params, offset):
    """门板朝相机那面是附近沿法向投影后最密的一层；中面再往里半个板厚，offset 是相对中面朝相机一侧的距离。"""
    normal = params.get("normal")
    if normal is None:
        raise ParseError("计划里没有门的法向（reach 的 approach）")
    normal = np.asarray(normal, dtype=float)
    normal = normal - float(np.dot(normal, UP)) * UP
    normal = normal / np.linalg.norm(normal)
    points = disk(_radius(DOOR_RADIUS, image_size))
    face = densest_layer(points @ normal)
    toward = 1.0 if float(np.dot(normal, camera_center(matrix))) > face else -1.0
    middle = face - toward * PANEL_HALF_THICKNESS
    return normal, middle + toward * offset, {"face": round(face, 4)}


def locate_door_axis(pixel, disk, matrix, image_size, params):
    return _door_plane(pixel, disk, matrix, image_size, params, 0.0)


def locate_door_handle(pixel, disk, matrix, image_size, params):
    return _door_plane(pixel, disk, matrix, image_size, params, HANDLE_OFFSET)


# 每个角色一种量法。模型在计划里写的 *_kind 类别，取值就是这里的名字。
LOCATORS = {
    "object": locate_object,
    "cube_center": locate_object,
    "bin": locate_bin,
    "nut": locate_nut,
    "peg": locate_peg,
    "table": locate_table,
    "door_axis": locate_door_axis,
    "door_handle": locate_door_handle,
}

# 两指落点对的合理宽度，米。太近是两个像素指到了同一个点，太远是有一个指飞了。
PAIR_MIN_WIDTH = 0.015
PAIR_MAX_WIDTH = 0.08
# 单目恢复背面落点时，落点附近点云的取样半径，按 512 边长定，单位像素。
PAIR_RADIUS = 36
# 朝后落点深度估计失败时，退回朝前落点沿 axis 再往回这么多，米。
PAIR_FALLBACK = 0.04


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
    """两个落点的水平间距是否在一个物体的宽度范围内。"""
    delta = np.asarray(back, dtype=float) - np.asarray(front, dtype=float)
    delta[2] = 0.0
    width = float(np.linalg.norm(delta))
    return PAIR_MIN_WIDTH <= width <= PAIR_MAX_WIDTH


def grip_axis(step):
    """两落点的连线方向，世界系。带 yaw 的步骤按执行时的朝向算
    （夹爪竖直朝下时两指合拢方向就是朝向矩阵第一列），不带 yaw 的默认世界 x 轴。"""
    yaw = step.get("yaw")
    if yaw is None:
        return np.array([1.0, 0.0, 0.0])
    ori = np.asarray(topdown_ori(float(yaw)), dtype=float)
    return ori[:, 0] / float(np.linalg.norm(ori[:, 0]))


def recover_landings(pixels, views, image_size, axis=None, plane_offset=None, plane_z=None):
    """两个落点像素换成世界坐标，返回 (朝前的点, 朝后的点, 记录)。

    pixels 里 front 是朝前落点的 frontview 像素（被 mask 时 front_back 是它在 backview 的像素）；
    back 是朝后落点的 backview 像素（双目且这台相机看得到）；
    same 是朝后落点在 frontview 的像素（backview 被 mask 或单目，挡住也照写）。
    双目：每台相机用自己的深度还原它看得到的那个落点。
    same 分支：朝前的落点直接读深度；朝后的那个被物体挡住，视线先打到正面，
    改用落点附近点云里物体顶面沿 axis 的远端当作背面的深度。
    axis 是两落点的连线方向（世界系，默认世界 x 轴，lift 的合拢方向），
    方向不限：same 分支会把它规范成指向相机（back 比 front 远离相机，
    在 axis 的负侧），调用方给执行朝向的第一列也不怕符号反。
    plane_offset 给出落点相对支撑面的高度（如 Nut 的半厚），plane_z 也可直接
    给出绝对高度（如 PickPlace 由顶面先验和可见指腹表面确定的抓取高度）。same 分支优先用
    这个水平面和两条像素射线求交，避免遮挡落点套用固定宽度 fallback。
    """
    if axis is None:
        axis = np.array([1.0, 0.0, 0.0])
    axis = np.asarray(axis, dtype=float).reshape(3)
    front_px = pixels.get("front")
    front_cam = DEFAULT_CAMERA
    if front_px is None:
        front_px = pixels.get("front_back")
        front_cam = BACK_CAMERA
    if front_px is None or front_cam not in views:
        raise ParseError("朝前落点在所有视图都被 mask")
    front_depth, front_matrix = views[front_cam]
    front = surface_point(front_px, front_depth, front_matrix)
    back_px = pixels.get("back")
    if back_px is not None and BACK_CAMERA in views:
        back = surface_point(back_px, *views[BACK_CAMERA])
        return front, back, {"mode": "stereo"}
    same = pixels.get("same")
    if same is None:
        raise ParseError("缺少第二个落点")
    # same 像素指向背落点：两个像素各自的视线和同一水平面（front 的高度）的交点差
    # 就是 front 指向 back 的水平方向。axis 规范成指向相机（back 比 front 远离相机，
    # 在 axis 的负侧），调用方给执行朝向的第一列也不怕符号反。
    same_depth, same_matrix = views[DEFAULT_CAMERA]
    radius = max(8, int(round(PAIR_RADIUS * float(image_size) / 512.0)))
    pair_z = float(front[2]) if plane_z is None else float(plane_z)
    support = None
    if plane_z is None and plane_offset is not None:
        disk = backproject_disk(front_depth, front_matrix, front_px, radius)
        support = lowest_surface(disk)
        pair_z = support + float(plane_offset)
    p_front = ray_at_height(front_px, pair_z, front_matrix)
    p_same = ray_at_height(np.asarray(same, dtype=float), pair_z, same_matrix)
    delta = p_same - p_front
    if float(np.linalg.norm(delta)) > 1e-6 and float(axis @ delta) > 0.0:
        axis = -axis
    # 竖直抓取的两个指腹落点同高。已知抓取水平面时，被挡点的像素射线与
    # 这个平面的交点就是它的三维位置，不用固定宽度 fallback。
    if (plane_z is not None or support is not None) and plausible(p_front) and plausible(p_same) and landing_pair_ok(p_front, p_same):
        if plane_z is not None:
            info = {"mode": "mono", "method": "known_plane", "plane_z": round(pair_z, 4)}
        else:
            info = {"mode": "mono", "method": "support_plane", "support_z": round(support, 4)}
        return p_front, p_same, info
    disk = backproject_disk(front_depth, front_matrix, front_px, radius)
    top = disk[(disk[:, 2] > front[2] + 0.008) & (disk[:, 2] < front[2] + 0.045)]
    if len(top) >= 8:
        s_back = float(np.percentile(top @ axis, 10))
    else:
        s_back = float(front @ axis - PAIR_FALLBACK)
    try:
        back = ray_at_plane(same, axis, s_back, same_matrix)
    except ValueError:
        back = None
    if back is None or not plausible(back) or not landing_pair_ok(front, back):
        back = front - axis * PAIR_FALLBACK
    if not landing_pair_ok(front, back):
        back = np.array([front[0] - PAIR_FALLBACK, front[1], front[2]], dtype=float)
    return front, back, {"mode": "mono", "method": "surface_or_fallback", "s_back": round(float(s_back), 4)}


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


def locate_merged(role, pixels, views, image_size, params=None):
    """测量级融合：各视图像素周围的点云都反投影到世界坐标系、合成一片，在这上面量平面。

    单个视图被手臂等遮挡时，它的污染点在合并点云里是少数，被 lowest_surface / 中位数
    这类多数派统计自动投出去，不用事后的分歧规则兜底。视线求交用主视图的像素
    （没有主视图像素就用仅有的那路）。返回 (world, info)。
    """
    usable = {camera: np.asarray(pixels[camera], dtype=float) for camera in views if pixels.get(camera) is not None}
    if not usable:
        raise ParseError("模型没给出任何视图的像素")
    primary = primary_camera(role)
    ray_cam = primary if primary in usable else sorted(usable)[0]
    ray_pixel = usable[ray_cam]
    ray_depth, ray_matrix = views[ray_cam]

    def disk(radius, merged=True):
        if not merged or len(usable) == 1:
            return backproject_disk(ray_depth, ray_matrix, ray_pixel, radius)
        clouds = [backproject_disk(views[cam][0], views[cam][1], pixel, radius) for cam, pixel in usable.items()]
        return np.concatenate(clouds)

    normal, offset, info = LOCATORS[role](ray_pixel, disk, ray_matrix, image_size, params or {})
    world = ray_at_plane(ray_pixel, normal, offset, ray_matrix)
    if not plausible(world):
        raise ParseError("反投影超出工作空间：{}".format(np.round(world, 4).tolist()))
    tag = "+".join(sorted(usable)) if len(usable) > 1 else ray_cam
    return world, dict(info, role=role, view=tag)


def _ray(pixel, matrix):
    near = unproject_pixel(pixel, 0.5, matrix)
    far = unproject_pixel(pixel, 3.0, matrix)
    return near, far - near


def ray_gap(pixels, views):
    """两个视图像素对应的视线的最近距离，米。大说明模型在两个视图里指的不是同一个点。"""
    usable = [(pixels[cam], views[cam][1]) for cam in sorted(views) if pixels.get(cam) is not None]
    if len(usable) < 2:
        return None
    (p1, d1), (p2, d2) = [_ray(pixel, matrix) for pixel, matrix in usable[:2]]
    w0 = p1 - p2
    a, b, c = float(np.dot(d1, d1)), float(np.dot(d1, d2)), float(np.dot(d2, d2))
    d, e = float(np.dot(d1, w0)), float(np.dot(d2, w0))
    denom = a * c - b * b
    if denom < 1e-12:
        return None
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


# 门板是竖直平面，侧向的视线接近和门板平行，求交很不稳；这两个角色不做测量级融合，
# 各视图独立定位再融合（见 measure_views）。
FRONT_PRIMARY_ROLES = ("door_axis", "door_handle")
# 两个视图的分歧超过这个距离，说明至少一路被污染（手臂、箱壁挡在标注点上方），
# 污染都把估计往高拉，取 z 低的那个。
DISAGREE_TOL = 0.03
# 这些角色的平面量测是对整片邻域点云的多数派统计（最低层、中位数、分位数），
# 两个视图的点云合并后，单视图里的遮挡点被投出去，可以做测量级融合。
# object 的顶面沿主视线的窄柱搜索，柱内遮挡不会被投出，仍走逐视图融合；门板见上。
MERGED_ROLES = ("bin", "nut", "peg", "table")
# 两条视线的最近距离超过这个值，说明模型在两个视图里指的不是同一个点，不合并点云。
RAY_CONSISTENCY_TOL = 0.01


def primary_camera(role):
    """主视图统一是 frontview：backview 在它正对面，两台相机平视对称，没有谁量水平面更稳。
    主视图被 mask 或量不出来时，调用方（fuse_estimates / locate_merged）自动退到另一路。"""
    return DEFAULT_CAMERA


def fuse_estimates(role, estimates):
    """estimates 是 {相机名: (world, info)}。取主视图的结果，另一视图做校验和兜底。

    返回 (world, info)；info 里 view 是实际用的相机，有两个视图时带另一视图的
    world 和两者的距离 disagreement_m（跨视图一致性，大说明至少一路定位不可靠）。
    分歧超过 DISAGREE_TOL 时改用 z 低的一路：遮挡物都在桌面上方，被挡的视图量偏高。
    """
    if not estimates:
        raise ParseError("没有可用的定位结果")
    primary = primary_camera(role)
    used = primary if primary in estimates else sorted(estimates)[0]
    if len(estimates) >= 2:
        worlds = {camera: np.asarray(item[0], dtype=float) for camera, item in estimates.items()}
        dist = float(np.linalg.norm(worlds[used] - next(w for cam, w in worlds.items() if cam != used)))
        if dist > DISAGREE_TOL:
            used = min(worlds, key=lambda camera: worlds[camera][2])
    world, info = estimates[used]
    info = dict(info, view=used)
    for camera in sorted(estimates):
        if camera == used:
            continue
        other_world = np.asarray(estimates[camera][0], dtype=float)
        info["other_view"] = camera
        info["other_world"] = [round(float(value), 4) for value in other_world]
        info["disagreement_m"] = round(float(np.linalg.norm(np.asarray(world) - other_world)), 4)
    return world, info


def measure_views(role, pixels, views, image_size, params=None):
    """同一个标注点的多视图定位。pixels 是 {相机名: 像素，可以 None}。

    只有一个视图：直接单视图定位。门板角色：各视图独立定位再融合（主视图 frontview）。
    MERGED_ROLES：两条视线一致（ray_gap 小）时合并两路点云做测量级融合，
    模型指点不一致或合并失败时退回逐视图估计再融合。
    """
    usable = {camera: pixels.get(camera) for camera in views if pixels.get(camera) is not None}
    if not usable:
        raise ParseError("模型没给出任何视图的像素")
    if len(usable) == 1:
        camera = next(iter(usable))
        depth, matrix = views[camera]
        world, info = locate(role, usable[camera], depth, matrix, image_size, params)
        return world, dict(info, view=camera)
    estimates = {}
    errors = {}
    for camera in sorted(usable):
        depth, matrix = views[camera]
        try:
            estimates[camera] = locate(role, usable[camera], depth, matrix, image_size, params)
        except ValueError as exc:
            errors[camera] = str(exc)
    if role in FRONT_PRIMARY_ROLES:
        if not estimates:
            raise ParseError("各视图都定位失败：{}".format(errors))
        return fuse_estimates(role, estimates)
    gap = ray_gap(pixels, views)
    merged_error = None
    if role in MERGED_ROLES and (gap is None or gap <= RAY_CONSISTENCY_TOL):
        try:
            world, info = locate_merged(role, pixels, views, image_size, params)
        except ValueError as exc:
            merged_error = str(exc)
        else:
            if gap is not None:
                info["ray_gap_m"] = round(gap, 4)
            if len(estimates) >= 2:
                cams = sorted(estimates)
                info["disagreement_m"] = round(
                    float(np.linalg.norm(np.asarray(estimates[cams[0]][0]) - np.asarray(estimates[cams[1]][0]))), 4
                )
            return world, info
    if estimates:
        world, info = fuse_estimates(role, estimates)
        if gap is not None:
            info["ray_gap_m"] = round(gap, 4)
        return world, info
    raise ParseError("各视图都定位失败：{}；合并：{}".format(errors, merged_error or "视线不一致，未合并"))


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


def _step_roles_full(task_name, subtasks):
    """每一步里各个像素的 (角色, 记录)，和 subtasks 对齐，形如 {"pixel": ("object", {"kind": ...})}。

    模型在步骤里标注了指的是哪类东西（解析成 pixel_kind、held_kind、pivot_kind、path_kind，
    取值就是 LOCATORS 的角色名）时优先按标注选量法，depth_locate 不用按任务名硬编码；
    没标注或标注不认识时退回按任务名和计划结构的推断（_role）。结构推断只看步骤顺序：
    合爪之前指向的是要抓的东西；合爪之后到松爪之前指向放置目标，held 仍是手里的东西；
    lift 指的是正抓着的那个东西。不看真值，也不靠像素远近判断：
    螺母中心和柱顶在画面里可能只差一两个像素。

    同一个点在多步里重复出现时沿用第一次定的角色，免得模型前后标注不一致时
    同一点量出两个世界坐标；此时记录里 role_reused 是实际沿用的角色。
    记录 kind 是模型标注的原文，plan_role 是标注和结构推断不一致时推断的角色，
    unknown_kind 表示标注的不是 LOCATORS 里的角色（这一步仍按结构推断）。
    带 pixel_2（target_2）的步骤是两指落点，成对定位，角色记 grip，去重键带上两个像素。
    """
    out = []
    carrying = False
    placed = []
    seen = {}

    def apply(step, key, structural):
        kind = step.get(key + "_kind")
        role, extra = structural, {}
        if kind is not None:
            extra["kind"] = kind
            if kind in LOCATORS:
                role = kind
                if kind != structural:
                    extra["plan_role"] = structural
            else:
                extra["unknown_kind"] = True
        if "pixel_2" in step and key == "pixel":
            # target_1、target_2 是一对两指落点，不是单点平面量法；
            # 模型标了别的类别也按落点对处理。落点在某台相机被 mask 时那个字段是 None。
            role = "grip"
            extra["pair"] = True
        if key in ("pixel", "held"):
            pixels = _view_pixels(step, key)
            if role == "grip" and "pixel_2" in step:
                for name in ("pixel_2", "pixel_2_back"):
                    if step.get(name) is not None:
                        pixels[name] = np.asarray(step[name], dtype=float)
            pk = _pixel_key(pixels)
            if pk in seen:
                if seen[pk] != role:
                    extra["role_reused"] = seen[pk]
                    role = seen[pk]
            else:
                seen[pk] = role
        return role, extra

    for step in subtasks:
        kind = step["type"]
        roles = {}
        if kind == "close_gripper":
            carrying = True
        elif kind == "open_gripper":
            carrying = False
        elif kind in ("move_above", "move_down", "reach"):
            # 松爪后退开的那一步还指着刚放下的位置
            place = carrying or any(
                _pixel_match(step.get("pixel"), step.get("pixel_back"), other, other_back)
                for other, other_back in placed
            )
            if place:
                placed.append((step.get("pixel"), step.get("pixel_back")))
            roles["pixel"] = apply(step, "pixel", _role(task_name, not place, kind))
            if "held" in step:
                roles["held"] = apply(step, "held", _role(task_name, True, kind))
        elif kind == "lift":
            roles["pixel"] = apply(step, "pixel", _role(task_name, True, kind))
        elif kind == "turn":
            roles["pivot"] = apply(step, "pivot", _role(task_name, True, kind))
        elif kind == "wipe":
            roles["path"] = apply(step, "path", _role(task_name, True, kind))
        out.append(roles)
    return out


def step_roles(task_name, subtasks):
    """每一步里各个像素的角色，和 subtasks 对齐，形如 {"pixel": "object", "held": "nut"}。"""
    return [{key: role for key, (role, _extra) in roles.items()} for roles in _step_roles_full(task_name, subtasks)]


def _pixel_match(front_a, back_a, front_b, back_b):
    """两个标注是否指同一个点：任一共同可见的视图里像素一致。被 mask（None）的视图不参与；
    没有共同可见的视图时按不是同一个点。"""
    for left, right in ((front_a, front_b), (back_a, back_b)):
        if left is not None and right is not None:
            return bool(np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float)))
    return False


def _view_pixels(step, key):
    """这一步 key 字段在各视图里的像素。被 mask 的视图是 None，不放进返回字典；
    单目没有 *_back 字段。解析保证了至少有一个视图可见。"""
    pixels = {}
    if step.get(key) is not None:
        pixels[DEFAULT_CAMERA] = np.asarray(step[key], dtype=float)
    pixel_back = step.get(key + "_back")
    if pixel_back is not None:
        pixels[BACK_CAMERA] = np.asarray(pixel_back, dtype=float)
    return pixels


def _pixel_key(pixels):
    return tuple(sorted((camera, tuple(np.round(pixel, 3))) for camera, pixel in pixels.items()))


def _dedupe_key(role, pixels):
    return (role, _pixel_key(pixels))


def grip_pixels(step, views):
    """落点对在各相机的像素。front 是朝前落点（target_1）的 frontview 像素，
    被 mask 时 front_back 是它在 backview 的像素（target_1_back）。双目时朝后落点（target_2）
    优先用 backview 像素（target_2_back），被 mask 时用它在 frontview 的像素（same）；
    单目 same 就是 target_2（挡住也照写）。"""
    pixels = {}
    if step.get("pixel") is not None:
        pixels["front"] = np.asarray(step["pixel"], dtype=float)
    elif step.get("pixel_back") is not None:
        pixels["front_back"] = np.asarray(step["pixel_back"], dtype=float)
    if BACK_CAMERA in views and step.get("pixel_2_back") is not None:
        pixels["back"] = np.asarray(step["pixel_2_back"], dtype=float)
    elif step.get("pixel_2") is not None:
        pixels["same"] = np.asarray(step["pixel_2"], dtype=float)
    return pixels


def _pair_axis(axis, step):
    """落点对的连线方向：调用方给了就用调用方的（stack 的合拢轴随开局朝向变），
    否则带 yaw 的步骤按 yaw 算（nut 抓把手），都不带用世界 x 轴。"""
    return axis if axis is not None else grip_axis(step)


def _pickplace_object_plane(pixel, depth, matrix, image_size):
    """按物体顶面和固定抓取深度估计水平面。"""
    center, info = locate("object", pixel, depth, matrix, image_size)
    if info["top_z"] is None:
        raise ParseError("PickPlace 物品顶面估计失败")
    return float(info["top_z"]) - PICKPLACE_GRASP_DEPTH, info


def _pickplace_plane(pixels, views, image_size):
    """顶面先验为主；高物品的可见接触面明显更高时，用它纠正箱壁造成的低估。"""
    camera = DEFAULT_CAMERA if pixels.get("front") is not None else BACK_CAMERA
    key = "front" if camera == DEFAULT_CAMERA else "front_back"
    if pixels.get(key) is None or camera not in views:
        raise ParseError("PickPlace 朝前落点没有可用视图")
    if pixels.get("same") is None:
        raise ParseError("缺少第二个落点")
    depth, matrix = views[camera]
    front = surface_point(pixels[key], depth, matrix, radius=1)
    visible_z = float(front[2])
    info = {"view": camera, "surface_z": round(visible_z, 4)}
    pixel = 0.5 * (np.asarray(pixels[key], dtype=float) + np.asarray(pixels["same"], dtype=float))
    try:
        object_z, object_info = _pickplace_object_plane(pixel, depth, matrix, image_size)
    except ValueError:
        return visible_z, dict(info, source="visible")
    info["object_plane"] = object_info
    # 圆柱剪影的深度会偏低；Milk 的尖顶又会让通用中心定位锁到箱壁。
    # 只有可见面高出顶面估计足够多时才覆盖，普通箱壁遮挡仍用顶面减抓取深度。
    if visible_z > object_z + 0.045:
        return visible_z, dict(info, source="visible", object_z=round(object_z, 4))
    return object_z, dict(info, source="object_top", object_z=round(object_z, 4))


def _recover_for_task(task_name, pixels, views, image_size, axis):
    """按物体已有的高度先验恢复落点对，遮挡时不用统一的方块宽度。"""
    offset = NUT_HALF_THICKNESS if task_name.startswith("nutassembly") else None
    # 双目各看一侧时直接读两个可见表面；只有 same 分支才需要 PickPlace 的抓取平面。
    needs_plane = task_name.startswith("pickplace") and not (
        pixels.get("back") is not None and BACK_CAMERA in views
    )
    plane_info = None
    if needs_plane:
        plane_z, plane_info = _pickplace_plane(pixels, views, image_size)
    else:
        plane_z = None
    first, second, info = recover_landings(pixels, views, image_size, axis, plane_offset=offset, plane_z=plane_z)
    if plane_info is not None:
        info["visible_plane"] = plane_info
    return first, second, info


def locate_plan(task_name, subtasks, views, image_size, axis=None):
    """给计划里每一步填上 step["world"]（各像素量出来的世界坐标），execute_plan 直接用。

    views 是 {相机名: (深度图, 世界到像素矩阵)}，每个标注点在有的视图里各自定位再融合。
    带 pixel_2 的步骤是两指落点，按 grip 成对还原，同时填 world 的 pixel 和 pixel_2；
    axis 是落点对连线方向，None 时按步骤的 yaw 算，都没有用世界 x 轴。
    返回定位过的点 [{"pixel", "pixel_back", "role", "world", "info"}]，同一点只量一次，给统计用；
    落点对多 world_2 和 pixel_2、pixel_2_back。被 mask 的视图字段是 None。
    Door 的门板法向取计划里第一个 reach 的 approach。
    """
    params = {}
    for step in subtasks:
        if step["type"] == "reach":
            params["normal"] = np.asarray(step["approach"], dtype=float)
            break
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
                "pixel_back": pixels.get(BACK_CAMERA),
                "role": role,
                "world": world,
                "info": info,
            }
        )
        return world

    for step, roles in zip(subtasks, _step_roles_full(task_name, subtasks)):
        world = {}
        for key, (role, extra) in roles.items():
            if key == "path":
                front_path = step["path"]
                back_path = step.get("path_back") or [None] * len(front_path)
                world[key] = [
                    measure(
                        {
                            camera: pixel
                            for camera, pixel in (
                                (DEFAULT_CAMERA, front),
                                (BACK_CAMERA, back),
                            )
                            if pixel is not None
                        },
                        role,
                        extra,
                    )
                    for front, back in zip(front_path, back_path)
                ]
            elif role == "grip" and key == "pixel":
                pixels = grip_pixels(step, views)
                key_pair = _dedupe_key(role, pixels)
                if key_pair in seen:
                    first, second = seen[key_pair]
                else:
                    first, second, info = _recover_for_task(task_name, pixels, views, image_size, _pair_axis(axis, step))
                    if extra:
                        info.update(extra)
                    seen[key_pair] = (first, second)
                    # 两落点的中点是物体中心，统计误差用它配真值；执行照样用两落点。
                    center = 0.5 * (first + second)
                    points.append(
                        {
                            "pixel": pixels.get("front", pixels.get("front_back")),
                            "pixel_back": step.get("pixel_back"),
                            "pixel_2": pixels.get("back", pixels.get("same")),
                            "pixel_2_back": step.get("pixel_2_back"),
                            "role": role,
                            "world": first,
                            "world_2": second,
                            "land_1": first,
                            "land_2": second,
                            "center_world": center,
                            "center_pixel": project_world(center, views[DEFAULT_CAMERA][1])[0],
                            "center_pixel_back": project_world(center, views[BACK_CAMERA][1])[0]
                            if BACK_CAMERA in views
                            else None,
                            "info": info,
                        }
                    )
                world["pixel"] = first
                world["pixel_2"] = second
            else:
                world[key] = measure(_view_pixels(step, key), role, extra)
        step["world"] = world
    return points


MOVABLE = ("object", "nut", "grip")


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


def relocate(task_name, subtasks, start, views, image_size, axis=None):
    """抓下一个物体前用当前这一帧的深度重新量它：开局时挡在它前面的东西可能已经搬走了。

    只更新从 start 起指向这个物体的 pixel / held；箱子、柱子还用开局量的，
    这时夹爪可能正悬在它们上方。两指落点（grip）成对重测。
    量不出来就保留原值。返回新的量测记录或 None。
    """
    roles = step_roles(task_name, subtasks)
    role = roles[start]["pixel"]
    start_step = subtasks[start]
    if role == "grip":
        pixels = grip_pixels(start_step, views)
        try:
            first, second, info = _recover_for_task(task_name, pixels, views, image_size, _pair_axis(axis, start_step))
        except ValueError:
            return None
        for step, step_role in zip(subtasks[start:], roles[start:]):
            if step_role.get("pixel") != "grip":
                continue
            same_first = _pixel_match(
                step.get("pixel"), step.get("pixel_back"), start_step.get("pixel"), start_step.get("pixel_back")
            )
            same_second = _pixel_match(
                step.get("pixel_2"), step.get("pixel_2_back"),
                start_step.get("pixel_2"), start_step.get("pixel_2_back"),
            )
            if same_first and same_second:
                step["world"]["pixel"] = first
                step["world"]["pixel_2"] = second
        center = 0.5 * (first + second)
        return {
            "pixel": pixels.get("front", pixels.get("front_back")),
            "pixel_back": start_step.get("pixel_back"),
            "role": role,
            "world": first,
            "world_2": second,
            "land_1": first,
            "land_2": second,
            "center_world": center,
            "center_pixel": project_world(center, views[DEFAULT_CAMERA][1])[0],
            "center_pixel_back": project_world(center, views[BACK_CAMERA][1])[0]
            if BACK_CAMERA in views
            else None,
            "info": info,
        }
    pixels = _view_pixels(start_step, "pixel")
    try:
        world, info = measure_views(role, pixels, views, image_size)
    except ValueError:
        return None
    for step, step_role in zip(subtasks[start:], roles[start:]):
        for key in ("pixel", "held"):
            if step_role.get(key) != role:
                continue
            if _pixel_match(
                step.get(key), step.get(key + "_back"), start_step.get("pixel"), start_step.get("pixel_back")
            ):
                step["world"][key] = world
    return {
        "pixel": pixels.get(DEFAULT_CAMERA),
        "pixel_back": pixels.get(BACK_CAMERA),
        "role": role,
        "world": world,
        "info": info,
    }
