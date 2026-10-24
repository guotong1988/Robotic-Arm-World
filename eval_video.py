"""测评视频左右并排。

左边是现在的 frontview，标出模型给出的像素，以及定位时取深度的那一圈。
右边是把这些画面解析成世界坐标时用的三维：frontview（有 backview 时再并上）
的深度反投影点云，加上每个像素周围拿来量平面的点、视线和量出来的世界坐标。
点云每帧跟着场景更新；像素、测量点和定位结果停在提问那一帧，重定位时再更新。
"""

import os

import numpy as np

# 固定的斜上方机位，对着桌面中心。边长 512 时桌面附近大约 10 厘米占 30 像素，
# 工作空间的四个角落在画面里。
VIEW_EYE = np.array([1.10, -1.15, 1.78])
VIEW_TARGET = np.array([0.0, 0.0, 0.86])
VIEW_FOCAL = 1.18
TABLE_Z = 0.80
GAP = 4
PALETTE = (
    (255, 72, 72),
    (60, 210, 95),
    (70, 150, 255),
    (255, 196, 48),
    (206, 90, 255),
    (40, 214, 206),
    (255, 132, 48),
    (255, 120, 170),
)
BACKGROUND = (22, 24, 30)


def _axes():
    forward = VIEW_TARGET - VIEW_EYE
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward


_RIGHT, _UP, _FORWARD = _axes()


def as_uint8_rgb(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(image)


def project_view(points, size):
    """世界点投到右边这幅图。返回 u、v 和机位前方的深度，点在机位背后时深度为负。"""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    rel = points - VIEW_EYE
    depth = rel @ _FORWARD
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = VIEW_FOCAL * float(size) / depth
        u = size * 0.5 + (rel @ _RIGHT) * scale
        v = size * 0.5 - (rel @ _UP) * scale
    return u, v, depth


def _font(size):
    """返回 (字体, 能否写中文)。没有中文字体时标题退回英文。"""
    candidates = (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    )
    try:
        from PIL import ImageFont
    except ImportError:
        return None, False
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size), True
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size), False
    except TypeError:
        return ImageFont.load_default(), False


_FONT, _CJK = _font(15)
CAPTION_LEFT = "2D 像素" if _CJK else "2D pixels"
CAPTION_RIGHT = "3D 深度" if _CJK else "3D depth"


def _grid():
    xs = np.arange(-0.45, 0.451, 0.05)
    ys = np.arange(-0.50, 0.501, 0.008)
    major = np.array([96, 104, 120], dtype=np.uint8)
    minor = np.array([58, 64, 76], dtype=np.uint8)
    points = []
    colors = []
    for x in xs:
        color = major if abs(x / 0.1 - round(x / 0.1)) < 1e-3 else minor
        line = np.column_stack([np.full(len(ys), x), ys, np.full(len(ys), TABLE_Z)])
        points.append(line)
        colors.append(np.broadcast_to(color, (len(ys), 3)))
    ys = np.arange(-0.50, 0.501, 0.05)
    xs = np.arange(-0.45, 0.451, 0.008)
    for y in ys:
        color = major if abs(y / 0.1 - round(y / 0.1)) < 1e-3 else minor
        line = np.column_stack([xs, np.full(len(xs), y), np.full(len(xs), TABLE_Z)])
        points.append(line)
        colors.append(np.broadcast_to(color, (len(xs), 3)))
    return np.concatenate(points), np.concatenate(colors).copy()


_GRID_POINTS, _GRID_COLORS = _grid()


def _keep_nearest(ui, vi, depth, color, width, height):
    ok = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height) & np.isfinite(depth) & (depth > 0.02)
    empty_c = np.zeros((0, 3), dtype=np.uint8)
    if not np.any(ok):
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0), empty_c
    ui, vi, depth, color = ui[ok], vi[ok], depth[ok], color[ok]
    order = np.argsort(depth, kind="mergesort")
    ui, vi, depth, color = ui[order], vi[order], depth[order], color[order]
    linear = vi.astype(np.int64) * width + ui.astype(np.int64)
    _, first = np.unique(linear, return_index=True)
    return ui[first], vi[first], depth[first], color[first]


def _paint(canvas, zbuf, points, colors, expand):
    """把世界点画进画面。同一像素留离机位更近的；和已有深度打平的话，后画的盖住先画的。"""
    if len(points) == 0:
        return
    height, width = canvas.shape[:2]
    u, v, depth = project_view(points, width)
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    if expand:
        ui = np.concatenate([ui, ui + 1, ui, ui + 1])
        vi = np.concatenate([vi, vi, vi + 1, vi + 1])
        depth = np.concatenate([depth, depth, depth, depth])
        colors = np.concatenate([colors, colors, colors, colors])
    ui, vi, depth, colors = _keep_nearest(ui, vi, depth, colors, width, height)
    if len(ui) == 0:
        return
    nearer = depth <= zbuf[vi, ui]
    ui, vi, depth, colors = ui[nearer], vi[nearer], depth[nearer], colors[nearer]
    zbuf[vi, ui] = depth
    canvas[vi, ui] = colors


def _in_workspace(points):
    return (
        np.all(np.isfinite(points), axis=1)
        & (np.abs(points[:, 0]) < 0.85)
        & (np.abs(points[:, 1]) < 0.85)
        & (points[:, 2] > 0.55)
        & (points[:, 2] < 1.60)
    )


def cloud_from_depth(depth, matrix_inv, rgb, stride):
    """深度图按 stride 抽样，反投影成世界坐标，颜色取同一像素的 RGB。"""
    depth = np.asarray(depth, dtype=float)
    height, width = depth.shape[:2]
    vs, us = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[vs, us]
    ok = np.isfinite(z) & (z > 0.05) & (z < 4.0)
    if not np.any(ok):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    us, vs, z = us[ok], vs[ok], z[ok]
    cam = np.stack([(us + 0.5) * z, (vs + 0.5) * z, z, np.ones_like(z)])
    world = np.asarray(matrix_inv, dtype=float) @ cam
    points = world[:3].T
    keep = _in_workspace(points)
    points = points[keep]
    if rgb is None:
        colors = _height_color(points[:, 2])
    else:
        colors = np.asarray(rgb)[vs[keep], us[keep]]
    return points, np.ascontiguousarray(colors)


def _height_color(z):
    t = np.clip((z - 0.75) / 0.45, 0.0, 1.0)
    return np.column_stack(
        [
            40 + 200 * t,
            90 + 120 * t,
            210 - 150 * t,
        ]
    ).astype(np.uint8)


def _stem(world, info):
    """定位时量到的高度范围：支撑面到顶面，或箱底到箱沿。没有就在点下方留一小段。"""
    info = info or {}
    world_z = float(world[2])
    if info.get("rim_z") is not None:
        return world_z, float(info["rim_z"])
    low = None
    for key in ("support_z", "table_z"):
        if info.get(key) is not None:
            low = float(info[key])
            break
    high = float(info["top_z"]) if info.get("top_z") is not None else world_z
    if low is None:
        low = world_z - 0.04
    return low, max(high, world_z)


def _draw():
    from PIL import Image, ImageDraw

    return Image, ImageDraw


def _text(draw, xy, text, color):
    if _FONT is None or not text:
        return
    x, y = xy
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=_FONT)
    draw.text((x, y), text, fill=tuple(int(c) for c in color), font=_FONT)


def _marker_label(index, role, world, count, with_xyz):
    label = str(index)
    if role and count <= 8:
        label = "{} {}".format(index, role)
    if with_xyz and count <= 6:
        label = "{}  {:.2f} {:.2f} {:.2f}".format(label, world[0], world[1], world[2])
    return label


def draw_left(rgb, markers):
    """在 frontview 上画出模型给的像素和取深度的那一圈。"""
    canvas = np.array(rgb, copy=True)
    if not markers:
        return _caption(canvas, CAPTION_LEFT)
    try:
        Image, ImageDraw = _draw()
    except ImportError:
        return _dots(canvas, [(item["pixel"], item["color"]) for item in markers if item["pixel"] is not None])
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    count = len(markers)
    placed = []
    for index, item in enumerate(markers, start=1):
        pixel = item["pixel"]
        if pixel is None:
            continue
        u, v = float(pixel[0]), float(pixel[1])
        color = tuple(int(c) for c in item["color"])
        radius = int(item["radius"])
        dot = 8 if count <= 8 else 5
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline=color, width=2)
        draw.line((u - dot, v, u + dot, v), fill=color, width=2)
        draw.line((u, v - dot, u, v + dot), fill=color, width=2)
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=color)
        if _far_enough(placed, u, v + radius + 2) or count <= 8:
            _text(draw, (u + 6, v + 6), _marker_label(index, item["role"], item["world"], count, False), color)
            placed.append((u, v))
    return _caption(np.array(image), CAPTION_LEFT)


def _far_enough(placed, u, v, gap=22):
    return all(abs(u - x) > gap or abs(v - y) > gap for x, y in placed)


def _caption(canvas, text):
    try:
        Image, ImageDraw = _draw()
    except ImportError:
        return canvas
    image = Image.fromarray(np.ascontiguousarray(canvas))
    draw = ImageDraw.Draw(image)
    _text(draw, (8, 6), text, (255, 255, 255))
    return np.array(image)


def _dots(canvas, spots):
    height, width = canvas.shape[:2]
    for pixel, color in spots:
        if pixel is None:
            continue
        u = int(round(float(pixel[0])))
        v = int(round(float(pixel[1])))
        if 1 <= u < width - 1 and 1 <= v < height - 1:
            canvas[v - 1 : v + 2, u - 1 : u + 2] = color
    return canvas


def _polyline(draw, points, size, color, width):
    if points is None or len(points) < 2:
        return
    u, v, depth = project_view(points, size)
    if np.any(depth <= 0.02) or not np.all(np.isfinite(u)):
        keep = (depth > 0.02) & np.isfinite(u) & np.isfinite(v)
        if np.count_nonzero(keep) < 2:
            return
        u, v = u[keep], v[keep]
    draw.line(list(zip(u.tolist(), v.tolist())), fill=tuple(int(c) for c in color), width=width)


def render_right(size, cloud, cloud_color, markers, eef):
    """斜上方看的深度点云。测量点、视线和定位结果画在点云上面。"""
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[...] = BACKGROUND
    zbuf = np.full((size, size), np.inf)
    _paint(canvas, zbuf, _GRID_POINTS, _GRID_COLORS, expand=False)
    if len(cloud):
        _paint(canvas, zbuf, cloud, cloud_color, expand=True)
    try:
        Image, ImageDraw = _draw()
    except ImportError:
        return _caption(canvas, CAPTION_RIGHT)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    count = len(markers)
    placed = []
    for index, item in enumerate(markers, start=1):
        color = item["color"]
        samples = item.get("samples")
        if samples is not None and len(samples):
            u, v, depth = project_view(samples, size)
            ok = (depth > 0.02) & np.isfinite(u) & np.isfinite(v)
            for x, y in zip(u[ok], v[ok]):
                draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=tuple(int(c) for c in color))
        world = item["world"]
        low, high = _stem(world, item.get("info"))
        _polyline(
            draw,
            np.array([[world[0], world[1], low], [world[0], world[1], high]]),
            size,
            color,
            2,
        )
        if item.get("ray_from") is not None:
            _polyline(draw, np.stack([item["ray_from"], item["ray_to"]]), size, color, 2)
            join = item.get("ray_join")
            if join is not None:
                _polyline(draw, np.stack([item["ray_to"], join]), size, (240, 240, 240), 1)
        u, v, depth = project_view(world, size)
        if depth[0] > 0.02 and np.isfinite(u[0]):
            x, y = float(u[0]), float(v[0])
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(0, 0, 0), width=3)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=tuple(int(c) for c in color), width=2)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=tuple(int(c) for c in color))
            if _far_enough(placed, x, y) or count <= 8:
                _text(
                    draw,
                    (x + 8, y - 8),
                    _marker_label(index, item["role"], world, count, True),
                    color,
                )
                placed.append((x, y))
    if eef is not None:
        u, v, depth = project_view(eef, size)
        if depth[0] > 0.02 and np.isfinite(u[0]):
            x, y = float(u[0]), float(v[0])
            draw.line((x - 6, y, x + 6, y), fill=(245, 245, 245), width=2)
            draw.line((x, y - 6, x, y + 6), fill=(245, 245, 245), width=2)
            _text(draw, (x + 7, y + 4), "eef", (230, 230, 230))
    return _caption(np.array(image), CAPTION_RIGHT)


def side_by_side(left, right):
    gap = np.full((left.shape[0], GAP, 3), 12, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate([left, gap, right], axis=1))


def _camera_center(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return np.linalg.solve(matrix[:3, :3], -matrix[:3, 3])


def _unproject(pixel, depth, matrix_inv):
    u, v = float(pixel[0]), float(pixel[1])
    z = float(depth)
    cam = np.array([u * z, v * z, z, 1.0])
    return np.asarray(matrix_inv, dtype=float) @ cam


class EvalVideo:
    """一条测评的并排画面。matrices 在第一次收到观测时再算，那时仿真已经 reset。"""

    def __init__(self, env, image_size):
        self.env = env
        self.image_size = int(image_size)
        self.frames = []
        self.markers = []
        self._ready = False
        self._warned = False

    def _ensure(self):
        if self._ready:
            return
        from collect_lift_mllm import DEFAULT_CAMERA, CAMERAS, world_to_pixel_matrices

        self.camera = DEFAULT_CAMERA
        self.cameras = [name for name in CAMERAS]
        self.matrices = world_to_pixel_matrices(self.env, self.image_size)
        self.inv = {name: np.linalg.inv(np.asarray(matrix, dtype=float)) for name, matrix in self.matrices.items()}
        self.center = _camera_center(self.matrices[self.camera])
        self._ready = True

    def add(self, obs):
        if obs is None:
            return
        rgb = self._rgb(obs)
        try:
            self.frames.append(self._compose(obs, rgb))
        except Exception as exc:
            if not self._warned:
                print("video overlay failed: {}".format(exc), flush=True)
                self._warned = True
            blank = np.empty_like(rgb)
            blank[...] = BACKGROUND
            self.frames.append(side_by_side(rgb, blank))

    def set_points(self, points, obs=None):
        """换上这一轮定位结果。obs 是提问那一帧，用来抓测量点，并重画刚刚录下的那一帧。"""
        try:
            self._ensure()
            raw_points = list(points or [])
            self.markers = [self._marker(raw, index, obs, len(raw_points)) for index, raw in enumerate(raw_points)]
            if obs is not None and self.frames:
                self.frames[-1] = self._compose(obs, self._rgb(obs))
        except Exception as exc:
            print("video markers failed: {}".format(exc), flush=True)

    def update_point(self, point, obs=None):
        """重定位之后改对应像素的世界坐标和测量点。"""
        try:
            pixel = np.asarray(point["pixel"], dtype=float).reshape(2)
        except (KeyError, TypeError, ValueError):
            return
        for index, item in enumerate(self.markers):
            if item["pixel"] is not None and np.linalg.norm(item["pixel"] - pixel) <= 3.0:
                self.markers[index] = self._marker(point, index, obs, len(self.markers))
                return

    def _marker(self, raw, index, obs, count):
        info = dict(raw.get("info") or {})
        role = raw.get("role") or info.get("role") or "object"
        world = np.asarray(raw["world"], dtype=float).reshape(3)
        pixel = raw.get("pixel")
        pixel = None if pixel is None else np.asarray(pixel, dtype=float).reshape(2)
        radius = _disk_radius(count, self.image_size)
        item = {
            "pixel": pixel,
            "world": world,
            "role": str(role),
            "info": info,
            "color": PALETTE[index % len(PALETTE)],
            "radius": radius,
            "samples": self._samples(obs, pixel, raw.get("pixel_back"), radius, count),
            "ray_from": None,
            "ray_to": None,
            "ray_join": None,
        }
        if pixel is not None and self._ready:
            item.update(self._ray(pixel, world))
        return item

    def _ray(self, pixel, world):
        hit = _unproject(pixel, 1.0, self.inv[self.camera])[:3]
        direction = hit - self.center
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return {}
        direction = direction / norm
        closest = self.center + direction * float(np.dot(world - self.center, direction))
        out = {"ray_from": closest - direction * 0.22, "ray_to": closest}
        if float(np.linalg.norm(closest - world)) > 0.012:
            out["ray_join"] = world
        return out

    def _samples(self, obs, pixel, pixel_back, radius, count):
        """提问那一帧上，这个像素周围拿来量平面的深度点。两个视图都有就拼在一起。"""
        if obs is None:
            return None
        limit = max(24, min(100, 1600 // max(1, count)))
        chunks = []
        if pixel is not None:
            chunks.append(self._disk(obs, self.camera, pixel, radius, limit))
        if pixel_back is not None and "backview" in self.matrices:
            chunks.append(self._disk(obs, "backview", pixel_back, radius, limit))
        chunks = [chunk for chunk in chunks if chunk is not None and len(chunk)]
        if not chunks:
            return None
        return np.concatenate(chunks)

    def _disk(self, obs, camera, pixel, radius, limit):
        from depth_locate import backproject_disk

        try:
            points = backproject_disk(self._depth(obs, camera), self.matrices[camera], pixel, radius)
        except Exception:
            return None
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            return None
        points = points[_in_workspace(points)]
        if len(points) > limit:
            points = points[:: int(np.ceil(len(points) / float(limit)))]
        return points

    def _compose(self, obs, rgb):
        self._ensure()
        cloud, color = self._scene_cloud(obs, rgb)
        eef = obs.get("robot0_eef_pos")
        if eef is not None:
            eef = np.asarray(eef, dtype=float).reshape(3)
        right = render_right(self.image_size, cloud, color, self.markers, eef)
        return side_by_side(draw_left(rgb, self.markers), right)

    def _scene_cloud(self, obs, front_rgb):
        stride = 2 if self.image_size >= 400 else 1
        points = []
        colors = []
        for name in self.cameras:
            if name not in self.inv:
                continue
            try:
                depth = self._depth(obs, name)
            except Exception:
                continue
            rgb = front_rgb if name == self.camera else self._rgb_or_none(obs, name)
            if rgb is not None and rgb.shape[:2] != depth.shape[:2]:
                rgb = None
            chunk, color = cloud_from_depth(depth, self.inv[name], rgb, stride)
            if len(chunk):
                points.append(chunk)
                colors.append(color)
        if not points:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        return np.concatenate(points), np.concatenate(colors)

    def _depth(self, obs, camera):
        from depth_locate import real_depth

        return real_depth(self.env, obs, camera)

    def _rgb(self, obs):
        image = self._rgb_or_none(obs, "frontview")
        if image is None:
            raise RuntimeError("缺少 frontview 图像")
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            raise RuntimeError("相机图像是 {}，测评要求 {:d}x{:d}".format(image.shape, self.image_size, self.image_size))
        return image.copy()

    def _rgb_or_none(self, obs, camera):
        key = camera + "_image"
        if key not in obs:
            return None
        return as_uint8_rgb(obs[key])


def _disk_radius(count, image_size):
    """圈的半径按 512 边长定。点少时和 depth_locate 量支撑面的 40 像素一致，点多时缩小。"""
    base = 40 if count <= 4 else (16 if count <= 10 else 8)
    return max(4, int(round(base * float(image_size) / 512.0)))


def _self_test():
    size = 128
    rgb = np.full((size, size, 3), 80, dtype=np.uint8)
    rgb[40:70, 30:60] = (180, 40, 40)
    world = np.array([0.04, -0.06, 0.86])
    samples = world + np.array(
        [[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0], [0.0, 0.01, 0.02], [0.0, -0.01, -0.01]]
    )
    marker = {
        "pixel": np.array([48.0, 55.0]),
        "world": world,
        "role": "object",
        "info": {"support_z": 0.80, "top_z": 0.90},
        "color": PALETTE[0],
        "radius": 10,
        "samples": samples,
        "ray_from": world + np.array([0.12, -0.12, 0.16]),
        "ray_to": world + np.array([0.0, 0.0, 0.02]),
        "ray_join": None,
    }
    xs = np.linspace(-0.25, 0.25, 40)
    ys = np.linspace(-0.25, 0.25, 40)
    xx, yy = np.meshgrid(xs, ys)
    table = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 0.80)])
    color = np.broadcast_to(np.array([170, 140, 110], dtype=np.uint8), (len(table), 3)).copy()
    right = render_right(size, table, color, [marker], np.array([0.04, -0.06, 1.05]))
    left = draw_left(rgb, [marker])
    frame = side_by_side(left, right)
    assert frame.shape == (size, size * 2 + GAP, 3)
    assert frame[55, 48, 0] > 200
    u, v, depth = project_view(world, size)
    assert depth[0] > 0
    x, y = int(round(u[0])), int(round(v[0]))
    patch = frame[y - 3 : y + 4, size + GAP + x - 3 : size + GAP + x + 4]
    assert patch[..., 0].max() > 200
    # 近处的点盖住远处投到同一像素的点。
    ui = np.array([3, 3], dtype=np.int32)
    vi = np.array([4, 4], dtype=np.int32)
    kept_u, kept_v, kept_z, kept_c = _keep_nearest(
        ui, vi, np.array([2.0, 0.5]), np.array([[0, 0, 255], [255, 0, 0]], dtype=np.uint8), 8, 8
    )
    assert len(kept_u) == 1 and kept_c[0, 0] == 255 and kept_z[0] == 0.5
    print("eval_video self-test ok", flush=True)
    return frame


if __name__ == "__main__":
    _self_test()
