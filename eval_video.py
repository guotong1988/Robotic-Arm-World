"""测评视频按采集输入的相机顺序并排，标出模型回答里的原始像素。

默认左边 frontview、右边 backview，单目时只保留 frontview。每个标记直接来自
subtasks 的 target_1、target_2、held、pivot、path；mask 的视图不画，三维定位、
深度采样和后续重定位都不会改写这些模型输出。
"""

import os

import numpy as np

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


def as_uint8_rgb(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(image)


def _font(size):
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
        return None
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


_FONT = _font(15)


def _draw():
    from PIL import Image, ImageDraw

    return Image, ImageDraw


def _text(draw, xy, text, color):
    if _FONT is None or not text:
        return
    x, y = xy
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=_FONT)
    draw.text((x, y), text, fill=tuple(int(c) for c in color), font=_FONT)


def _caption(canvas, text):
    try:
        Image, ImageDraw = _draw()
    except ImportError:
        return canvas
    image = Image.fromarray(np.ascontiguousarray(canvas))
    draw = ImageDraw.Draw(image)
    _text(draw, (8, 6), text, (255, 255, 255))
    return np.array(image)


def _pixel_key(pixel):
    if pixel is None:
        return None
    return tuple(np.round(np.asarray(pixel, dtype=float).reshape(2), 3))


def plan_markers(subtasks):
    """把解析后的模型计划展开成逐视图像素标记；完全相同的重复标注只画一次。"""
    markers = []
    seen = set()
    fields = (("pixel", "target_1"), ("pixel_2", "target_2"), ("held", "held"), ("pivot", "pivot"))
    for step in subtasks or []:
        entries = [(key, label, step.get(key), step.get(key + "_back")) for key, label in fields]
        back_path = step.get("path_back") or []
        for index, pixel in enumerate(step.get("path") or []):
            pixel_back = back_path[index] if index < len(back_path) else None
            entries.append(("path", "path[{}]".format(index), pixel, pixel_back))
        for _key, label, front, back in entries:
            if _key not in step:
                continue
            key = (label, _pixel_key(front), _pixel_key(back))
            if key in seen:
                continue
            seen.add(key)
            markers.append({"label": label, "pixel": front, "pixel_back": back, "color": PALETTE[len(markers) % len(PALETTE)]})
    return markers


def draw_view(rgb, markers, camera):
    """在一张相机图上画模型原始像素，不画深度采样圈或三维定位结果。"""
    canvas = np.array(rgb, copy=True)
    caption = "{} model pixels".format(camera)
    if not markers:
        return _caption(canvas, caption)
    try:
        Image, ImageDraw = _draw()
    except ImportError:
        return _caption(_dots(canvas, markers, camera), caption)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    placed = []
    for item in markers:
        pixel = item.get("pixel") if camera == "frontview" else item.get("pixel_back")
        if pixel is None:
            continue
        u, v = (float(value) for value in np.asarray(pixel).reshape(2))
        color = tuple(int(c) for c in item["color"])
        draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=(0, 0, 0), width=3)
        draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=color, width=2)
        draw.line((u - 8, v, u + 8, v), fill=color, width=1)
        draw.line((u, v - 8, u, v + 8), fill=color, width=1)
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=color)
        if all(abs(u - x) > 70 or abs(v - y) > 20 for x, y in placed):
            _text(draw, (u + 7, v + 6), item["label"], color)
            placed.append((u, v))
    return _caption(np.array(image), caption)


def _dots(canvas, markers, camera):
    height, width = canvas.shape[:2]
    for item in markers:
        pixel = item.get("pixel") if camera == "frontview" else item.get("pixel_back")
        if pixel is None:
            continue
        u, v = (int(round(float(value))) for value in np.asarray(pixel).reshape(2))
        if 1 <= u < width - 1 and 1 <= v < height - 1:
            canvas[v - 1 : v + 2, u - 1 : u + 2] = item["color"]
    return canvas


def side_by_side(left, right):
    gap = np.full((left.shape[0], GAP, 3), 12, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate([left, gap, right], axis=1))


class EvalVideo:
    """一条测评的相机画面；模型像素停在提问帧，视频图像随执行更新。"""

    def __init__(self, env, image_size):
        self.env = env
        self.image_size = int(image_size)
        self.frames = []
        self.markers = []
        self.cameras = None
        self._warned = False

    def _ensure(self):
        if self.cameras is None:
            from collect_lift_mllm import CAMERAS

            self.cameras = list(CAMERAS)

    def add(self, obs):
        if obs is None:
            return
        rgb = self._rgb(obs, "frontview")
        try:
            self.frames.append(self._compose(obs))
        except Exception as exc:
            if not self._warned:
                print("video overlay failed: {}".format(exc), flush=True)
                self._warned = True
            self.frames.append(_caption(rgb, "frontview"))

    def set_plan(self, subtasks, obs=None):
        """模型答案解析成功后立即记录原始像素，并重画提问那一帧。"""
        try:
            self._ensure()
            self.markers = plan_markers(subtasks)
            if obs is not None and self.frames:
                self.frames[-1] = self._compose(obs)
        except Exception as exc:
            print("video markers failed: {}".format(exc), flush=True)

    def _compose(self, obs):
        self._ensure()
        views = []
        for camera in self.cameras:
            rgb = self._rgb_or_none(obs, camera)
            if rgb is not None:
                views.append(draw_view(rgb, self.markers, camera))
        if not views:
            raise RuntimeError("缺少相机图像")
        if len(views) == 1:
            return views[0]
        return side_by_side(views[0], views[1])

    def _rgb(self, obs, camera):
        key = camera + "_image"
        if key not in obs:
            raise RuntimeError("缺少 {} 图像".format(camera))
        image = as_uint8_rgb(obs[key])
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            raise RuntimeError("相机图像是 {}，测评要求 {:d}x{:d}".format(image.shape, self.image_size, self.image_size))
        return image.copy()

    def _rgb_or_none(self, obs, camera):
        if camera + "_image" not in obs:
            return None
        return self._rgb(obs, camera)


def _self_test():
    size = 128
    front = np.full((size, size, 3), 80, dtype=np.uint8)
    back = np.full((size, size, 3), 60, dtype=np.uint8)
    subtasks = [
        {
            "type": "move_down",
            "pixel": np.array([48.0, 55.0]),
            "pixel_back": None,
            "pixel_2": np.array([68.0, 55.0]),
            "pixel_2_back": np.array([35.0, 45.0]),
            "held": np.array([80.0, 70.0]),
            "held_back": np.array([60.0, 70.0]),
        },
        {"type": "turn", "pivot": np.array([20.0, 90.0]), "pivot_back": np.array([90.0, 90.0])},
        {"type": "wipe", "path": [np.array([30.0, 100.0]), np.array([50.0, 100.0])]},
    ]
    markers = plan_markers(subtasks)
    assert [item["label"] for item in markers] == ["target_1", "target_2", "held", "pivot", "path[0]", "path[1]"]
    left = draw_view(front, markers, "frontview")
    right = draw_view(back, markers, "backview")
    frame = side_by_side(left, right)
    assert frame.shape == (size, size * 2 + GAP, 3)
    assert np.array_equal(frame[55, 48], PALETTE[0])
    assert np.array_equal(frame[45, size + GAP + 35], PALETTE[1])
    assert np.array_equal(draw_view(front, markers[:1], "backview")[55, 48], front[55, 48])
    video = EvalVideo(None, size)
    video.cameras = ["frontview", "backview"]
    obs = {"frontview_image": front, "backview_image": back}
    video.add(obs)
    video.set_plan(subtasks, obs)
    assert len(video.frames) == 1 and video.frames[0].shape == frame.shape
    assert np.array_equal(video.frames[0][55, 48], PALETTE[0])
    video.cameras = ["frontview"]
    assert video._compose(obs).shape == (size, size, 3)
    print("eval_video self-test ok", flush=True)
    return frame


if __name__ == "__main__":
    _self_test()
