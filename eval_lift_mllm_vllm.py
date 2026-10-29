"""用微调后的 Qwen 做 Lift 测评，直连 vLLM 的 OpenAI chat completions。

    python3 eval_lift_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20

vLLM 服务示例：

    python3 -m vllm.entrypoints.openai.api_server \\
        --model /the_model_path/ --host 0.0.0.0 --port 8000

    python3 eval_lift_mllm_vllm.py --oracle --episodes 5

每条测评做这些事：

1. 按 collect_lift_mllm.py 的方式摆好 Lift / Panda，渲染 frontview 和 backview（正对面机位）的 512x512 RGB
2. 按当前画面重算指令，和采集时一样：没对准就先移动到目标上方，对准后只写还要抬多少
3. 只把两张 RGB 和这条指令放进对话
4. 只在初始画面问一次模型，取出两个夹爪落点的像素。单目是 target_1 和 target_2（背面那个即使被挡住也要给）；
   对面双目时每个落点各写两台相机的像素：target_1、target_1_back 是第一个落点，target_2、target_2_back 是第二个，
   某台相机看不到那个落点时对应字段写 "mask"，不脑补这张图的像素。抬升高度用指令里已经写明的数字
5. 用这一帧的深度把像素换成两个落点的世界坐标。单目时背面那个点被方块挡住，用落点附近点云估计它的深度；
   双目时两张深度拼成一块点云，每个落点用看得到它的那台相机还原。之后不再问模型。
   按这两个点做逆运动学：走到落点中点上方、下降、合爪、抬起并稳住
6. 抓住方块，并且方块抬过桌面 4 厘米，才算成功

每条测评按控制频率收成 mp4，写到输出目录的 videos/。画面按采集输入的相机顺序并排，
逐视图标出模型回答的原始像素，不叠加深度点云。--no-video 可以关掉。

模型给的是像素，不是世界坐标。仿真里的落点只用于 --oracle 出答案和事后统计误差，不参与定位和控制。
move_above 在落点中点上方 12 厘米，这个高度落在采集时的悬停范围内。
"""

import argparse
import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import gl_backend

gl_backend.configure()

import numpy as np

from collect_lift_mllm import (
    BACK_CAMERA,
    CAMERAS,
    DEFAULT_CAMERA,
    INSTRUCTION_STYLES,
    MASK,
    STABILIZE_DURATION,
    above_cube,
    add_camera_arg,
    apply_cameras,
    arm_controller,
    back_enabled,
    format_answer,
    format_pixel,
    grip_landings,
    grip_view_pixels,
    hold,
    instruction_for,
    make_env,
    move_until,
    positions_from_obs,
    project_world,
    remaining_lift,
    round_pixel,
    sample_start_pos,
    unproject_pixel,
    world_to_pixel_matrices,
)
from depth_locate import fuse_estimates, make_views, recover_landings
from eval_video import EvalVideo
from grip_ik import GRIP_YAW, IKArm, choose_grasp, grasp_candidates, pad_offset, topdown_ori

# 采集时悬停高度在 8 到 15 厘米之间随机，这里取中间附近的固定值。
HOVER_CLEARANCE = 0.12
# 与 collect_lift_mllm.collect_episode 里的 move_until / hold 一致。
MOVE_ABOVE_TOL = 0.015
MOVE_ABOVE_STEPS = 160
MOVE_DOWN_TOL = 0.012
MOVE_DOWN_STEPS = 120
CLOSE_STEPS = 25
LIFT_TOL = 0.02
LIFT_STEPS = 160
# 与 collect_lift_mllm.make_env 的 control_freq 一致，播放接近实时。
VIDEO_FPS = 20
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
PIXEL_TEXT = re.compile(r"^\[\s*({n})(?:\s*,\s*|\s+)({n})\s*\]$".format(n=NUM))
THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
FENCE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
SUBTASK_TYPES = ("move_above", "move_down", "close_gripper", "lift", "stabilize")


def clip_workspace(pos):
    pos = np.array(pos, dtype=float)
    pos[0] = np.clip(pos[0], -0.35, 0.30)
    pos[1] = np.clip(pos[1], -0.35, 0.35)
    pos[2] = np.clip(pos[2], 0.75, 1.35)
    return pos


def plausible(pos):
    pos = np.asarray(pos, dtype=float)
    if pos.shape != (3,) or not np.all(np.isfinite(pos)):
        return False
    if abs(pos[0]) > 1.5 or abs(pos[1]) > 1.5:
        return False
    return 0.4 <= pos[2] <= 2.0


def parse_pixel(value, image_size):
    """采集写的是 \"[x y]\"。x 向右，y 向下，原点在左上角。也接受逗号分隔和长度为 2 的数组。"""
    if isinstance(value, str):
        match = PIXEL_TEXT.match(value.strip())
        if not match:
            raise ValueError("target 应为 [x y] 像素，收到 {}".format(value[:80]))
        pixel = np.array([float(item) for item in match.groups()], dtype=float)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        pixel = np.array([float(item) for item in value], dtype=float)
    else:
        raise ValueError("target 应为 [x y] 像素")
    if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
        raise ValueError("target 像素无效")
    limit = 4096.0 if image_size is None else float(image_size)
    if np.any(pixel < -limit) or np.any(pixel > 2.0 * limit):
        raise ValueError("target 像素超出画面：{}".format(pixel.tolist()))
    return pixel


# _view_field 的返回值哨兵：字段根本没写（区别于写了但被 mask 的 None）。
_MISSING = object()


def _view_field(item, keys, image_size):
    """按 keys 顺序取像素字段。字段没写返回 _MISSING；写了 "mask" 或 null 返回 None（这台相机看不到，不脑补）。"""
    for key in keys:
        if key not in item:
            continue
        value = item[key]
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() == MASK:
            return None
        return parse_pixel(value, image_size)
    return _MISSING


def normalize_subtasks(items, image_size):
    if not isinstance(items, list):
        raise ValueError("subtasks 应为数组")
    steps = []
    for item in items:
        if not isinstance(item, dict) or "type" not in item:
            raise ValueError("子任务缺少 type")
        kind = item["type"]
        if kind not in SUBTASK_TYPES:
            raise ValueError("未知子任务 {}".format(kind))
        step = {"type": kind}
        if kind in ("move_above", "move_down", "lift"):
            # 每个落点每视图一个字段：target_1、target_1_back 是第一个落点，target_2、target_2_back 是第二个。
            pixel = _view_field(item, ("target_1", "target"), image_size)
            if pixel is _MISSING:
                raise ValueError("{} 缺少 target_1".format(kind))
            step["pixel"] = pixel
            pixel_back = _view_field(item, ("target_1_back",), image_size)
            if pixel_back is not _MISSING:
                step["pixel_back"] = pixel_back
            pixel_2 = _view_field(item, ("target_2",), image_size)
            if pixel_2 is not _MISSING:
                step["pixel_2"] = pixel_2
            pixel_2_back = _view_field(item, ("target_2_back",), image_size)
            if pixel_2_back is not _MISSING:
                step["pixel_2_back"] = pixel_2_back
        if kind == "lift":
            if "delta_z" not in item:
                raise ValueError("lift 缺少 delta_z")
            delta = float(item["delta_z"])
            if not np.isfinite(delta) or not (0.0 <= delta <= 0.5):
                raise ValueError("delta_z 超出范围：{}".format(delta))
            step["delta_z"] = round(delta, 4)
        elif kind == "stabilize":
            duration = float(item.get("duration", STABILIZE_DURATION))
            if not np.isfinite(duration) or not (0.0 <= duration <= 5.0):
                raise ValueError("duration 超出范围：{}".format(duration))
            step["duration"] = round(duration, 4)
        steps.append(step)
    return steps


def _json_blobs(text):
    blobs = []
    for match in FENCE.findall(text):
        blobs.append(match.strip())
    blobs.append(text.strip())
    return blobs


def _load_object(text):
    start = text.find("{")
    if start < 0:
        raise ValueError("没有 JSON 对象")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("JSON 无法解析") from exc
    if not isinstance(obj, dict):
        raise ValueError("JSON 不是对象")
    return obj


def _plan_from_object(obj, image_size):
    if "subtasks" not in obj:
        raise ValueError("缺少 subtasks")
    steps = normalize_subtasks(obj["subtasks"], image_size)
    target = None
    target_back = None
    target_2 = None
    target_2_back = None
    for step in steps:
        if "pixel" in step:
            target = step["pixel"]
            target_back = step.get("pixel_back")
            target_2 = step.get("pixel_2")
            target_2_back = step.get("pixel_2_back")
            break
    return {
        "target": target,
        "target_back": target_back,
        "target_2": target_2,
        "target_2_back": target_2_back,
        "subtasks": steps,
    }


def _parse_plan(text, image_size):
    last_error = None
    for blob in _json_blobs(text):
        try:
            return _plan_from_object(_load_object(blob), image_size)
        except ValueError as exc:
            last_error = exc
    raise ValueError(str(last_error) if last_error else "没有抓取计划")


def parse_answer(text, image_size=None):
    """从模型回答里取出 subtasks。target 是第一个带像素的步骤，抬到目标后可以为空。"""
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
        try:
            return _parse_plan(candidate, image_size)
        except ValueError as exc:
            last_error = exc
    raise ParseError("无法解析抓取计划：{}".format(raw[:300])) from last_error


def pos_error(pred, gt):
    """预测减真值。x 远离机械臂为正，y 是另一水平轴，z 向上为正。"""
    delta = np.asarray(pred, dtype=float) - np.asarray(gt, dtype=float)
    return {
        "x": float(delta[0]),
        "y": float(delta[1]),
        "z": float(delta[2]),
        "xy": float(np.linalg.norm(delta[:2])),
        "l2": float(np.linalg.norm(delta)),
    }


def as_uint8_rgb(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(image)


def encode_png(image):
    image = as_uint8_rgb(image)
    buf = io.BytesIO()
    try:
        import imageio.v2 as imageio

        imageio.imwrite(buf, image, format="png")
    except ImportError:
        from PIL import Image

        Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def read_image(path):
    try:
        import imageio.v2 as imageio

        return as_uint8_rgb(imageio.imread(path))
    except ImportError:
        from PIL import Image

        return as_uint8_rgb(np.array(Image.open(path).convert("RGB")))


def message_text(message):
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        content = "".join(parts)
    content = "" if content is None else str(content)
    reasoning = message.get("reasoning_content") or ""
    return content, str(reasoning)


class ServerError(RuntimeError):
    pass


class ParseError(ValueError):
    pass


def user_text(instruction, text_template):
    """微调样本的指令以每台相机一个 <image> 占位开头。图走 image_url 后，正文去掉这些占位标记。"""
    text = text_template.format(instruction=instruction)
    while True:
        stripped = text.lstrip()
        if not stripped.startswith("<image>"):
            break
        text = stripped[len("<image>") :]
    if not text.startswith("\n"):
        text = "\n" + text
    return text


def image_content(image):
    png = encode_png(image)
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")},
    }


def chat_messages(images, instruction, system, text_template):
    """RGB 图（一张或一个列表：frontview、backview）加指令。"""
    if not isinstance(images, (list, tuple)):
        images = [images]
    text = user_text(instruction, text_template)
    content = [image_content(image) for image in images]
    content.append({"type": "text", "text": text})
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


def completion_text(data):
    """从 OpenAI chat 响应里取出模型文本。"""
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content, reasoning = message_text(message)
        if content.strip():
            return content
        if reasoning.strip():
            return reasoning
    response = data.get("response")
    if isinstance(response, str) and response.strip():
        return response
    raise ServerError("服务返回了空内容：{}".format(json.dumps(data, ensure_ascii=False)[:500]))


def post_chat(request, model, messages, temperature, max_tokens, drop_thinking):
    """发一次 chat completions。服务不认 chat_template_kwargs 时去掉再试一次。"""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not drop_thinking[0]:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        data = request(payload)
    except ServerError as exc:
        if drop_thinking[0] or "chat_template_kwargs" not in str(exc):
            raise
        drop_thinking[0] = True
        payload.pop("chat_template_kwargs", None)
        data = request(payload)
    return completion_text(data)


class OracleClient:
    """不访问模型。真值计划在 predict 里用 format_answer 生成，用来确认控制器能完成 Lift。"""

    def complete(self, image, instruction):
        raise RuntimeError("oracle 不调用模型")


class VLLMClient:
    def __init__(self, api_base, model, timeout, max_tokens, temperature, system, text_template, retries=2):
        self.api_base = api_base.rstrip("/")
        if not self.api_base.endswith("/v1"):
            self.api_base += "/v1"
        self.model = model or ""
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system = system
        self.text_template = text_template
        self.retries = max(0, int(retries))
        self._drop_thinking_flag = False

    def _request(self, method, path, payload=None):
        url = self.api_base + path
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise ServerError("HTTP {} {}\n{}".format(exc.code, url, detail[:800])) from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                reason = exc.reason if isinstance(exc, urllib.error.URLError) else "等待响应超时"
                if attempt >= self.retries:
                    raise ServerError(
                        "请求 {} 失败（已尝试 {} 次，单次超时 {} 秒）：{}".format(
                            url, attempt + 1, self.timeout, reason
                        )
                    ) from exc
                delay = 2 * (attempt + 1)
                print(
                    "请求 {} 第 {}/{} 次失败（{}），{} 秒后重试".format(
                        url, attempt + 1, self.retries + 1, reason, delay
                    ),
                    flush=True,
                )
                time.sleep(delay)

    def resolve_model(self):
        if self.model:
            return self.model
        data = self._request("GET", "/models")
        models = data.get("data") or []
        if not models:
            raise ServerError("{} 没有返回可用模型".format(self.api_base))
        self.model = models[0]["id"]
        return self.model

    def complete(self, images, instruction):
        """images 可以是一张 RGB，也可以是按 frontview、backview 顺序的列表。"""
        messages = chat_messages(images, instruction, self.system, self.text_template)
        drop_thinking = [self._drop_thinking_flag]

        def request(payload):
            return self._request("POST", "/chat/completions", payload)

        text = post_chat(
            request,
            self.model,
            messages,
            self.temperature,
            self.max_tokens,
            drop_thinking,
        )
        self._drop_thinking_flag = drop_thinking[0]
        return text


def real_eef(env):
    arm = arm_controller(env)
    arm.update(force=True)
    return np.array(arm.ref_pos, dtype=float)


def camera_size(env):
    """测评相机是正方形。边长来自创建环境时的 --image-size。"""
    height = int(np.asarray(env.camera_heights).reshape(-1)[0])
    width = int(np.asarray(env.camera_widths).reshape(-1)[0])
    if height != width:
        raise RuntimeError("测评相机应为正方形，当前 {}x{}".format(height, width))
    return height


def camera_image(obs, size, camera=DEFAULT_CAMERA):
    image = as_uint8_rgb(obs[camera + "_image"])
    if image.shape[0] != size or image.shape[1] != size:
        raise RuntimeError("相机图像是 {}，测评要求 {:d}x{:d}".format(image.shape, size, size))
    return image


def camera_images(obs, size):
    """按 CAMERAS 顺序的所有视图：frontview、backview。"""
    return [camera_image(obs, size, name) for name in CAMERAS]


def grab_frame(obs, size):
    """这一步的 frontview。仿真会复用同一块数组，必须拷一份。"""
    return camera_image(obs, size).copy()


def write_mp4(path, frames, fps=VIDEO_FPS):
    """把一条测评的画面写成 mp4。没有帧时不写文件。"""
    if not frames:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    # quality / pixelformat 只有 FFMPEG 插件认识。没有 imageio-ffmpeg 时，
    # imageio 会改用 PyAV，PyAVPlugin.write() 收到这些参数会直接报错。
    try:
        writer = imageio.get_writer(
            str(path),
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            pixelformat="yuv420p",
        )
    except (ImportError, ValueError, OSError):
        writer = imageio.get_writer(str(path), fps=fps, codec="libx264")
    # PyAV 走的是 v3 接口，write() 默认 is_batch=True。一张 (512, 512, 3)
    # 会被拆成 512 行 (512, 3)，再写进 (512, 3, 3)。
    if hasattr(writer, "write_args"):
        writer.write_args["is_batch"] = False
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return path


def predict(client, env, obs, instruction, oracle, phase, above, delta_z, episode):
    size = camera_size(env)
    images = camera_images(obs, size)
    gt_cube, gt_eef = positions_from_obs(obs)
    matrices = world_to_pixel_matrices(env, size)
    # 真值只用来出 oracle 答案和事后统计，不进定位和控制。
    gt_front, gt_back = grip_landings(env)
    gt_pixel, _ = project_world(gt_front, matrices[DEFAULT_CAMERA])
    gt_pixel_2, _ = project_world(gt_back, matrices[DEFAULT_CAMERA])
    gt_pixel_back = gt_pixel_2_back = None
    if BACK_CAMERA in matrices:
        gt_pixel_back, _ = project_world(gt_front, matrices[BACK_CAMERA])
        gt_pixel_2_back, _ = project_world(gt_back, matrices[BACK_CAMERA])
    started = time.perf_counter()
    if oracle:
        # 和采集同一条路径：遮挡判定、mask、双 mask 时 frontview 兜底都在 grip_view_pixels 里。
        pixels = grip_view_pixels(env, gt_front, gt_back, matrices)
        text = format_answer(pixels, gt_cube, gt_eef, phase, above, delta_z)
    else:
        text = client.complete(images, instruction)
    print("episode {:02d} instruction: {}".format(episode, instruction), flush=True)
    print("episode {:02d} answer ({:.3f}s):\n{}".format(episode, time.perf_counter() - started, text), flush=True)
    plan = parse_answer(text, image_size=size)
    if plan["target"] is None and plan["target_back"] is None:
        raise ParseError("target_1 在所有视图都被 mask 或缺失")
    if back_enabled():
        if plan["target_2"] is None and plan["target_2_back"] is None:
            raise ParseError("target_2 在所有视图都被 mask 或缺失")
    elif plan["target_2"] is None:
        raise ParseError("缺少 target_2")
    views = make_views(env, obs, matrices)
    front, back, located = recover_landings(
        {
            "front": plan["target"],
            "front_back": plan["target_back"],
            "same": plan["target_2"],
            "back": plan["target_2_back"],
        },
        views,
        size,
    )
    elapsed = time.perf_counter() - started
    return {
        "answer": text,
        "plan": {
            "pixel": plan["target"],
            "pixel_back": plan["target_back"],
            "pixel_2": plan["target_2"],
            "pixel_2_back": plan["target_2_back"],
            "front": front,
            "back": back,
            "target": 0.5 * (front + back),
            "subtasks": plan["subtasks"],
        },
        "gt_cube_pos": gt_cube,
        "gt_eef_pos": gt_eef,
        "gt_front": gt_front,
        "gt_back": gt_back,
        "gt_pixel": gt_pixel,
        "gt_pixel_2": gt_pixel_2,
        "gt_pixel_back": gt_pixel_back,
        "gt_pixel_2_back": gt_pixel_2_back,
        "located": located,
        "front_err": pos_error(front, gt_front),
        "back_err": pos_error(back, gt_back),
        "cube_err": pos_error(0.5 * (front + back), 0.5 * (gt_front + gt_back)),
        "seconds": round(elapsed, 3),
    }


def prepare_scene(env, rng, on_step=None):
    """和采集脚本一样：方块落地后，手臂先移到随机出发点。这段不交给模型。

    环境用关节位置控制器，移动走逆运动学。夹爪转到两指沿世界 x 合拢。
    """
    if on_step is None:
        on_step = lambda _obs: None
    obs = env.reset()
    arm = IKArm(env)
    arm.reset()
    on_step(obs)
    obs = arm.hold(-1.0, 20, on_step)
    finger_offset = pad_offset(env)
    rest_z = float(positions_from_obs(obs)[0][2])
    ori = topdown_ori(GRIP_YAW)
    home, _ = arm.pose()
    safe_z = max(float(home[2]), rest_z + 0.18)
    start = sample_start_pos(rng, positions_from_obs(obs)[0], rest_z)
    obs, _, _ = arm.move(np.array([home[0], home[1], safe_z]), ori, -1.0, 0.02, 120, on_step)
    obs, _, _ = arm.move(np.array([start[0], start[1], safe_z]), ori, -1.0, 0.02, 200, on_step)
    obs, _, _ = arm.move(np.asarray(start, dtype=float), ori, -1.0, 0.02, 120, on_step)
    return obs, arm, rest_z, finger_offset


def track(env, target, hold_ori, grip, tol, max_steps, on_step=None):
    if on_step is None:
        on_step = lambda _obs: None
    return move_until(env, target, hold_ori, grip, tol, max_steps, on_step)


def is_grasped(env):
    return bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cube))


def steps_to_run(subtasks):
    """先做计划里的第一步。这一步后面如果只剩 stabilize，就一起做完。"""
    if not subtasks:
        return []
    if subtasks[0]["type"] == "stabilize":
        return [subtasks[0]]
    if len(subtasks) == 2 and subtasks[1]["type"] == "stabilize":
        return list(subtasks)
    return [subtasks[0]]


def control_steps(delta_z):
    """方块位置已知后的固定流程。抬多高来自指令，不来自模型。"""
    steps = [
        {"type": "move_above"},
        {"type": "move_down"},
        {"type": "close_gripper"},
    ]
    if float(delta_z) >= 0.005:
        steps.append({"type": "lift", "delta_z": round(float(delta_z), 4)})
    steps.append({"type": "stabilize", "duration": STABILIZE_DURATION})
    return steps


def execute_subtask(env, step, target, hold_ori, grip, obs, on_step):
    """按采集时的容差和步数执行一个子任务。target 是计划里的方块位置。"""
    kind = step["type"]
    target = clip_workspace(target)
    if kind == "move_above":
        dest = clip_workspace(target + np.array([0.0, 0.0, HOVER_CLEARANCE]))
        obs, ok, err = track(env, dest, hold_ori, grip, MOVE_ABOVE_TOL, MOVE_ABOVE_STEPS, on_step)
        return obs, bool(ok), err, grip
    if kind == "move_down":
        obs, ok, err = track(env, target, hold_ori, grip, MOVE_DOWN_TOL, MOVE_DOWN_STEPS, on_step)
        return obs, bool(ok), err, grip
    if kind == "close_gripper":
        obs = hold(env, target, hold_ori, 1.0, CLOSE_STEPS, on_step)
        return obs, True, 0.0, 1.0
    if kind == "lift":
        dest = clip_workspace(target + np.array([0.0, 0.0, float(step["delta_z"])]))
        obs, ok, err = track(env, dest, hold_ori, 1.0, LIFT_TOL, LIFT_STEPS, on_step)
        return obs, bool(ok), err, 1.0
    if kind == "stabilize":
        steps = int(round(float(step["duration"]) * VIDEO_FPS))
        if steps <= 0:
            return obs, True, 0.0, grip
        obs = hold(env, clip_workspace(real_eef(env)), hold_ori, grip, steps, on_step)
        return obs, True, 0.0, grip
    raise ParseError("未知子任务 {}".format(kind))


def run_episode(env, client, rng, episode, height_min, height_max, oracle, video_path=None):
    size = camera_size(env)
    video = None if video_path is None else EvalVideo(env, size)

    def on_step(obs):
        if video is not None:
            video.add(obs)

    record = None
    try:
        obs, arm, rest_z, finger_offset = prepare_scene(env, rng, on_step)
        gt_cube, gt_eef = positions_from_obs(obs)
        target_lift = float(rng.uniform(height_min, height_max))
        style = INSTRUCTION_STYLES[int(rng.integers(len(INSTRUCTION_STYLES)))]
        lifted, remaining = remaining_lift(gt_cube[2], rest_z, target_lift)
        above = above_cube(gt_eef, gt_cube)
        opening_instruction, opening_spoken = instruction_for(style, remaining, lifted, above)
        queries = []
        hover_ok = grasp_ok = lift_ok = False
        hover_err = grasp_err = lift_err = None
        grip = -1.0
        success = False
        grasped = False
        error = None
        try:
            pred = predict(
                client,
                env,
                obs,
                opening_instruction,
                oracle,
                "initial",
                above,
                opening_spoken,
                episode,
            )
            if video is not None:
                video.set_plan(pred["plan"]["subtasks"], obs)
            front_w = pred["plan"]["front"]
            back_w = pred["plan"]["back"]
            try:
                pose, ori, _yaw, _q = choose_grasp(arm.ik, grasp_candidates([front_w, back_w], finger_offset))
            except ValueError as exc:
                raise ParseError(str(exc)) from exc
            print(
                "episode {:02d} front gt={} pred={} {}   back {} {}".format(
                    episode,
                    format_xyz(pred["gt_front"]),
                    format_xyz(front_w),
                    format_err_axes(pred["front_err"]),
                    format_xyz(back_w),
                    format_err_axes(pred["back_err"]),
                ),
                flush=True,
            )
            for step in control_steps(opening_spoken):
                kind = step["type"]
                if kind == "move_above":
                    obs, ok, err = arm.move(
                        pose + np.array([0.0, 0.0, HOVER_CLEARANCE]), ori, grip, MOVE_ABOVE_TOL, MOVE_ABOVE_STEPS, on_step
                    )
                elif kind == "move_down":
                    obs, ok, err = arm.move(pose, ori, grip, MOVE_DOWN_TOL, MOVE_DOWN_STEPS, on_step)
                elif kind == "close_gripper":
                    obs = arm.hold(1.0, CLOSE_STEPS, on_step)
                    ok, err, grip = True, 0.0, 1.0
                elif kind == "lift":
                    obs, ok, err = arm.move(
                        pose + np.array([0.0, 0.0, float(step["delta_z"])]), ori, 1.0, LIFT_TOL, LIFT_STEPS, on_step
                    )
                    grip = 1.0
                elif kind == "stabilize":
                    ticks = int(round(float(step["duration"]) * VIDEO_FPS))
                    if ticks > 0:
                        obs = arm.hold(grip, ticks, on_step)
                    ok, err = True, 0.0
                else:
                    raise ParseError("未知子任务 {}".format(kind))
                queries.append(pack_query(step["type"], opening_instruction, pred, ok, err))
                if step["type"] == "move_above":
                    hover_ok, hover_err = ok, err
                elif step["type"] == "move_down":
                    grasp_ok, grasp_err = ok, err
                elif step["type"] == "lift":
                    lift_ok, lift_err = ok, err
            grasped = is_grasped(env)
            success = bool(env._check_success()) and grasped
        except ParseError as exc:
            success = False
            grasped = is_grasped(env)
            error = str(exc)
        gt_cube, _ = positions_from_obs(obs)
        record = {
            "episode": episode,
            "success": success,
            "grasped": grasped,
            "instruction": opening_instruction,
            "lift_height_m": round(float(opening_spoken), 4),
            "target_lift_m": round(float(target_lift), 4),
            "above_at_start": bool(above),
            "hover_reached": bool(hover_ok),
            "grasp_reached": bool(grasp_ok),
            "lift_reached": bool(lift_ok),
            "hover_err_m": None if hover_err is None else round(float(hover_err), 4),
            "grasp_err_m": None if grasp_err is None else round(float(grasp_err), 4),
            "lift_err_m": None if lift_err is None else round(float(lift_err), 4),
            "final_cube_z": round(float(gt_cube[2]), 4),
            "queries": queries,
        }
        if error:
            record["error"] = error
        return record
    finally:
        if video is not None and video.frames:
            try:
                saved = write_mp4(video_path, video.frames)
            except Exception as exc:
                print("video failed: {}".format(exc), flush=True)
                if record is not None:
                    record["video_error"] = str(exc)
            else:
                rel = "videos/" + saved.name
                if record is not None:
                    record["video"] = rel
                else:
                    print("wrote partial video {}".format(saved), flush=True)


def pack_query(stage, instruction, pred, reached, pos_err):
    cube_err = pred["cube_err"]
    target = pred["plan"]["target"]
    record = {
        "stage": stage,
        "instruction": instruction,
        "answer": pred["answer"],
        "seconds": pred["seconds"],
        "subtasks": [
            {
                key: round_pixel(value)
                if key in ("pixel", "pixel_2", "pixel_back", "pixel_2_back") and value is not None
                else value
                for key, value in step.items()
            }
            for step in pred["plan"]["subtasks"]
        ],
        "target": [round(float(v), 4) for v in target],
        "gt_cube_pos": [round(float(v), 4) for v in pred["gt_cube_pos"]],
        "gt_eef_pos": [round(float(v), 4) for v in pred["gt_eef_pos"]],
        "cube_err_m": round(cube_err["l2"], 4),
        "cube_xy_err_m": round(cube_err["xy"], 4),
        "cube_x_err_m": round(cube_err["x"], 4),
        "cube_y_err_m": round(cube_err["y"], 4),
        "cube_z_err_m": round(cube_err["z"], 4),
        "reached": None if reached is None else bool(reached),
        "pos_err_m": None if pos_err is None else round(float(pos_err), 4),
    }
    pixel = pred["plan"].get("pixel")
    gt_pixel = pred.get("gt_pixel")
    if pixel is not None and gt_pixel is not None:
        delta = np.asarray(pixel, dtype=float) - np.asarray(gt_pixel, dtype=float)
        record["pixel"] = round_pixel(pixel)
        record["gt_pixel"] = round_pixel(gt_pixel)
        record["pixel_err"] = round(float(np.linalg.norm(delta)), 2)
        if pred.get("cube_depth_m") is not None:
            record["cube_depth_m"] = round(float(pred["cube_depth_m"]), 6)
    for name, pred_key, gt_key in (
        ("pixel_2", "pixel_2", "gt_pixel_2"),
        ("pixel_back", "pixel_back", "gt_pixel_back"),
        ("pixel_2_back", "pixel_2_back", "gt_pixel_2_back"),
    ):
        pixel = pred["plan"].get(pred_key)
        gt_pixel_n = pred.get(gt_key)
        if pixel is not None and gt_pixel_n is not None:
            delta = np.asarray(pixel, dtype=float) - np.asarray(gt_pixel_n, dtype=float)
            record[name] = round_pixel(pixel)
            record["gt_" + name] = round_pixel(gt_pixel_n)
            record[name + "_err"] = round(float(np.linalg.norm(delta)), 2)
    for name in ("front", "back"):
        err = pred.get(name + "_err")
        if err is not None:
            record[name + "_l2_m"] = round(err["l2"], 4)
    return record


def mean_or_none(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def grasp_query(record):
    """下降到方块时的那次预测最接近抓取定位。没有的话退回到别的步骤。"""
    for stage in ("move_down", "close_gripper", "lift", "move_above", "done"):
        for query in record["queries"]:
            if query["stage"] == stage:
                return query
    return record["queries"][-1] if record["queries"] else None


def format_cm(meters):
    if meters is None:
        return "n/a"
    return "{:.1f}cm".format(meters * 100.0)


def format_signed_cm(meters):
    if meters is None:
        return "n/a"
    return "{:+.1f}cm".format(meters * 100.0)


def format_axis_cm(x, y, z):
    """单次误差，带符号，单位厘米。"""
    if x is None or y is None or z is None:
        return "n/a"
    return "x={} y={} z={}".format(format_signed_cm(x), format_signed_cm(y), format_signed_cm(z))


def format_err_axes(err):
    return format_axis_cm(err["x"], err["y"], err["z"])


def query_pos_error(query):
    """从一条 query 还原有符号的三轴误差。旧结果没有这三列时返回 None。"""
    x = query.get("cube_x_err_m")
    y = query.get("cube_y_err_m")
    z = query.get("cube_z_err_m")
    if x is None or y is None or z is None:
        return None
    return pos_error(
        [float(x), float(y), float(z)],
        [0.0, 0.0, 0.0],
    )


def axis_means(errors):
    """mae 是各轴绝对误差的平均，bias 是预测减真值的平均。"""
    if not errors:
        return None
    out = {"n": len(errors), "l2_m": float(np.mean([err["l2"] for err in errors]))}
    for name in ("x", "y", "z"):
        values = np.array([float(err[name]) for err in errors], dtype=float)
        out[name] = {
            "mae_m": float(np.mean(np.abs(values))),
            "bias_m": float(np.mean(values)),
        }
    return out


def axis_hit_rate(errors, tol):
    """各轴绝对误差不超过 tol 的比例。"""
    if not errors:
        return None
    return {
        name: float(np.mean([abs(float(err[name])) <= tol for err in errors]))
        for name in ("x", "y", "z")
    }


def format_axis_means(stats):
    if not stats:
        return "n/a"
    parts = []
    for name in ("x", "y", "z"):
        item = stats[name]
        parts.append("|{}|={} bias={}".format(name, format_cm(item["mae_m"]), format_signed_cm(item["bias_m"])))
    parts.append("l2={}".format(format_cm(stats["l2_m"])))
    return "  ".join(parts)


def format_hit_rates(rates, tol):
    if not rates:
        return "n/a"
    body = " ".join("{}={:.0%}".format(name, rates[name]) for name in ("x", "y", "z"))
    return "{}内 {}".format(format_cm(tol), body)


def format_xyz(pos):
    values = [float(v) for v in np.asarray(pos, dtype=float).reshape(-1)[:3]]
    return "[{:.4f} {:.4f} {:.4f}]".format(*values)


def subtasks_match(plan, answer_text):
    """子任务类型要一致。lift 的高度和 stabilize 的时间也要接近采集标签。"""
    try:
        gt = json.loads(answer_text)
        gt_steps = gt["subtasks"]
    except (TypeError, ValueError, KeyError):
        return False
    pred_steps = plan["subtasks"]
    if [step["type"] for step in pred_steps] != [step["type"] for step in gt_steps]:
        return False
    for pred, gt_step in zip(pred_steps, gt_steps):
        if pred["type"] == "lift" and abs(float(pred["delta_z"]) - float(gt_step["delta_z"])) > 0.011:
            return False
        if pred["type"] == "stabilize" and abs(float(pred["duration"]) - float(gt_step["duration"])) > 0.05:
            return False
    return True


def load_camera_matrices(dataset):
    """采集时写下的世界到像素矩阵，{相机名: 4x4}。旧数据是单个 4x4，当作 frontview。"""
    meta_path = Path(dataset) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = meta.get("camera_world_to_pixel")
    if raw is None:
        raise FileNotFoundError("{} 没有 camera_world_to_pixel，需要用当前采集脚本重新采数据".format(meta_path))
    if isinstance(raw, dict):
        matrices = {name: np.asarray(matrix, dtype=float) for name, matrix in raw.items()}
        for name, matrix in matrices.items():
            if matrix.shape != (4, 4):
                raise ValueError("{} 的 camera_world_to_pixel 应为 4x4，收到 {}".format(name, matrix.shape))
    else:
        matrix = np.asarray(raw, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("camera_world_to_pixel 应为 4x4，收到 {}".format(matrix.shape))
        matrices = {DEFAULT_CAMERA: matrix}
    size = meta.get("image_size") or [512, 512]
    return matrices, int(size[0])


def probe_dataset(client, dataset, count, seed):
    path = Path(dataset) / "samples.jsonl"
    root = Path(dataset)
    matrices, image_size = load_camera_matrices(dataset)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(rows), size=min(count, len(rows)), replace=False)
    cube_errs = []
    pixel_errs = []
    by_phase = {}
    matches = 0
    print("probe {} samples from {}".format(len(picked), path), flush=True)
    for index in picked:
        row = rows[int(index)]
        images = [read_image(root / row["image"])]
        for key in ("image_back",):
            if row.get(key) and (root / row[key]).is_file():
                images.append(read_image(root / row[key]))
        text = client.complete(images, row["instruction"])
        try:
            plan = parse_answer(text, image_size=image_size)
        except ParseError as exc:
            print("{} parse failed: {}".format(row["id"], exc), flush=True)
            continue
        if plan["target"] is None and plan["target_back"] is None:
            print("{} 没有像素".format(row["id"]), flush=True)
            continue
        # 数据集没有深度图。新样本用两个落点的真值深度分别反投影；旧样本仍按方块中心。
        if "grip_front" in row and "grip_back" in row:
            # 每个落点各选一个看得见的视图：落点1 优先 frontview，落点2 优先 backview。
            picks = (
                ((plan["target"], DEFAULT_CAMERA), (plan["target_back"], BACK_CAMERA)),
                ((plan["target_2_back"], BACK_CAMERA), (plan["target_2"], DEFAULT_CAMERA)),
            )
            worlds = []
            pred_pixel = None
            gt_pixel = None
            for candidates, gt in zip(picks, (row["grip_front"], row["grip_back"])):
                chosen = None
                for pixel, camera in candidates:
                    if pixel is not None and camera in matrices:
                        chosen = (pixel, camera)
                        break
                if chosen is None:
                    worlds.append(None)
                    continue
                pixel, camera = chosen
                if pred_pixel is None:
                    pred_pixel = pixel
                    gt_pixel = round_pixel(project_world(gt, matrices[camera])[0])
                _, view_depth = project_world(gt, matrices[camera])
                try:
                    worlds.append(unproject_pixel(pixel, view_depth, matrices[camera]))
                except ValueError as exc:
                    print("{} {} {}".format(row["id"], camera, exc), flush=True)
                    worlds.append(None)
            if any(item is None for item in worlds) or gt_pixel is None:
                continue
            world = 0.5 * (np.asarray(worlds[0]) + np.asarray(worlds[1]))
            gt_world = 0.5 * (np.asarray(row["grip_front"], dtype=float) + np.asarray(row["grip_back"], dtype=float))
        elif "cube_pos" in row:
            estimates = {}
            pred_pixel = pred_camera = None
            for camera, pixel in ((DEFAULT_CAMERA, plan["target"]), (BACK_CAMERA, plan["target_back"])):
                if pixel is None or camera not in matrices:
                    continue
                if pred_pixel is None:
                    pred_pixel, pred_camera = pixel, camera
                _, view_depth = project_world(row["cube_pos"], matrices[camera])
                try:
                    estimates[camera] = (unproject_pixel(pixel, view_depth, matrices[camera]), {})
                except ValueError as exc:
                    print("{} {} {}".format(row["id"], camera, exc), flush=True)
            if not estimates:
                continue
            world, _ = fuse_estimates("object", estimates)
            gt_world = row["cube_pos"]
            # cube_pixel 是 frontview 的；预测用了 backview 时就按那台相机的矩阵投影真值。
            gt_pixel = row.get("cube_pixel") if pred_camera == DEFAULT_CAMERA else None
            if gt_pixel is None:
                gt_pixel, _ = project_world(row["cube_pos"], matrices[pred_camera])
                gt_pixel = round_pixel(gt_pixel)
        else:
            print("{} 没有落点或方块坐标，跳过".format(row["id"]), flush=True)
            continue
        if not plausible(world):
            print("{} 反投影超出工作空间 {}".format(row["id"], np.round(world, 4).tolist()), flush=True)
            continue
        plan = {
            "pixel": pred_pixel,
            "target": world,
            "subtasks": plan["subtasks"],
        }
        cube_err = pos_error(world, gt_world)
        pixel_err = float(
            np.linalg.norm(np.asarray(plan["pixel"], dtype=float) - np.asarray(gt_pixel, dtype=float))
        )
        matched = subtasks_match(plan, row.get("answer", ""))
        cube_errs.append(cube_err)
        pixel_errs.append(pixel_err)
        phase = row.get("phase") or "unknown"
        by_phase.setdefault(phase, []).append(cube_err)
        matches += int(matched)
        print(
            "{} pixel gt={} pred={} cube gt={} pred={} {} l2={} subtasks={}".format(
                row["id"],
                format_pixel(gt_pixel),
                format_pixel(plan["pixel"]),
                format_xyz(gt_world),
                format_xyz(world),
                format_err_axes(cube_err),
                format_cm(cube_err["l2"]),
                "ok" if matched else "mismatch",
            ),
            flush=True,
        )
    print(
        "probe {}  pixel={:.1f}px  subtask_match={}/{}".format(
            format_axis_means(axis_means(cube_errs)),
            float(np.mean(pixel_errs)) if pixel_errs else float("nan"),
            matches,
            len(cube_errs),
        ),
        flush=True,
    )
    print(
        "probe {}  {}".format(
            format_hit_rates(axis_hit_rate(cube_errs, 0.012), 0.012),
            format_hit_rates(axis_hit_rate(cube_errs, 0.03), 0.03),
        ),
        flush=True,
    )
    for phase in ("initial", "hover", "grasp", "close", "lift"):
        phase_errs = by_phase.get(phase)
        if not phase_errs:
            continue
        print(
            "probe {} n={} {}".format(phase, len(phase_errs), format_axis_means(axis_means(phase_errs))),
            flush=True,
        )


def summarize(records):
    grasp_errs = []
    pixel_errs = []
    for record in records:
        query = grasp_query(record)
        if query is None:
            continue
        err = query_pos_error(query)
        if err is not None:
            grasp_errs.append(err)
        if query.get("pixel_err") is not None:
            pixel_errs.append(float(query["pixel_err"]))
    stats = axis_means(grasp_errs)
    total = len(records)
    successes = sum(int(record["success"]) for record in records)
    grasps = sum(int(record["grasped"]) for record in records)
    return {
        "episodes": total,
        "successes": successes,
        "success_rate": None if total == 0 else successes / total,
        "grasps": grasps,
        "grasp_rate": None if total == 0 else grasps / total,
        "grasp_cube_err_m": None if stats is None else stats["l2_m"],
        "grasp_pixel_err": mean_or_none(pixel_errs),
        "grasp_cube_axis_m": stats,
        "grasp_cube_within_1_2cm": axis_hit_rate(grasp_errs, 0.012),
        "grasp_cube_within_3cm": axis_hit_rate(grasp_errs, 0.03),
    }


def print_episode(record):
    query = grasp_query(record)
    axes = "n/a" if query is None else format_axis_cm(
        query.get("cube_x_err_m"), query.get("cube_y_err_m"), query.get("cube_z_err_m")
    )
    final_z = record["final_cube_z"]
    z_text = "n/a" if final_z is None else "{:.3f}".format(final_z)
    print(
        "episode {:02d} success={} grasped={} height={:.2f}m 落点中心 {} final_z={}".format(
            record["episode"],
            record["success"],
            record["grasped"],
            record["lift_height_m"],
            axes,
            z_text,
        ),
        flush=True,
    )
    if query is not None:
        print(
            "  落点中心 gt={} pred={}".format(
                format_xyz(query["gt_cube_pos"]),
                format_xyz(query["target"]),
            ),
            flush=True,
        )
        if query.get("pixel") is not None:
            print(
                "  pixel gt={} pred={} err={:.1f}px".format(
                    format_pixel(query["gt_pixel"]),
                    format_pixel(query["pixel"]),
                    query["pixel_err"],
                ),
                flush=True,
            )
    if record.get("error"):
        print("  error: {}".format(record["error"]), flush=True)
    if record.get("video"):
        print("  video {}".format(record["video"]), flush=True)


def print_summary(summary):
    total = summary["episodes"]
    if total == 0:
        print("没有完成任何一条测评", flush=True)
        return
    print(
        "成功率 {}/{} = {:.1%}    抓住 {}/{} = {:.1%}".format(
            summary["successes"],
            total,
            summary["success_rate"],
            summary["grasps"],
            total,
            summary["grasp_rate"],
        ),
        flush=True,
    )
    print("抓取时落点中心 {}".format(format_axis_means(summary.get("grasp_cube_axis_m"))), flush=True)
    if summary.get("grasp_pixel_err") is not None:
        print("抓取时像素误差 {:.1f}px".format(summary["grasp_pixel_err"]), flush=True)
    print(
        "抓取时落点中心 {}    {}".format(
            format_hit_rates(summary.get("grasp_cube_within_1_2cm"), 0.012),
            format_hit_rates(summary.get("grasp_cube_within_3cm"), 0.03),
        ),
        flush=True,
    )
    print("x 远离机械臂为正，z 向上为正；bias 是预测减真值", flush=True)


def write_results(out_dir, records, summary):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonl_path, summary_path


def reset_episode_rng(env, seed):
    """把环境和所有物体摆放采样器恢复到同一个 seed，并返回本条测评的任务随机数。"""
    env.rng = np.random.default_rng(seed)

    def reset_sampler(sampler):
        if sampler is None:
            return
        if hasattr(sampler, "rng"):
            sampler.rng = np.random.default_rng(seed)
        for child in getattr(sampler, "samplers", {}).values():
            reset_sampler(child)

    reset_sampler(getattr(env, "placement_initializer", None))
    return np.random.default_rng(seed)


def add_common_args(parser):
    parser.add_argument("--model", default="", help="模型名。留空则用服务 /v1/models 的第一个")
    parser.add_argument("--episodes", type=int, default=20, help="测评多少条")
    parser.add_argument("--seed", type=int, default=1000, help="和采集用的 seed 0 错开，避免同一批初始场景")
    parser.add_argument("--fixed-initial", action="store_true", help="每条测评都使用同一个 seed 对应的初始场景")
    parser.add_argument("--image-size", type=int, default=1024, help="测评相机、模型输入和视频的边长")
    parser.add_argument("--height-min", type=float, default=0.06)
    parser.add_argument("--height-max", type=float, default=0.18)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system", default="", help="可选的 system prompt，默认不发")
    parser.add_argument("--text-template", default="{instruction}", help="拼在图片后面的文本，默认就是指令原文")
    parser.add_argument("--out", default="data/lift_eval")
    parser.add_argument("--no-video", action="store_true", help="不把每条测评录成并排 mp4")
    parser.add_argument("--oracle", action="store_true", help="用环境真值代替模型，检查控制器")
    parser.add_argument("--probe", type=int, default=0, help="正式测评前，先在采集样本上测方块位置和子任务")
    parser.add_argument("--dataset", default="data/lift_mllm")
    parser.add_argument("--self-test", action="store_true", help="只检查解析和坐标换算")
    add_camera_arg(parser)
    return parser


def run_eval(args, client, backend, extra_summary=None):
    """跑 probe 和正式测评，并把结果写到 args.out。oracle 时跳过 probe。"""
    apply_cameras(args)
    if args.oracle:
        print("oracle mode, skip model server", flush=True)
    elif args.probe > 0:
        probe_dataset(client, args.dataset, args.probe, args.seed)

    if args.episodes <= 0:
        return

    rng = np.random.default_rng(args.seed)
    env = make_env(args.image_size, args.seed, camera_depths=True, use_ik=True)
    video_dir = None if args.no_video else Path(args.out) / "videos"
    records = []
    try:
        for episode in range(args.episodes):
            if args.fixed_initial:
                rng = reset_episode_rng(env, args.seed)
            video_path = None if video_dir is None else video_dir / "ep{:04d}.mp4".format(episode)
            try:
                record = run_episode(
                    env,
                    client,
                    rng,
                    episode,
                    args.height_min,
                    args.height_max,
                    args.oracle,
                    video_path,
                )
            except ServerError as exc:
                print(exc, flush=True)
                break
            records.append(record)
            print_episode(record)
    finally:
        summary = summarize(records)
        summary["oracle"] = bool(args.oracle)
        summary["seed"] = args.seed
        summary["fixed_initial"] = bool(args.fixed_initial)
        summary["backend"] = "oracle" if args.oracle else backend
        if not args.oracle:
            summary["model"] = client.model
            if extra_summary:
                summary.update(extra_summary(client))
        jsonl_path, summary_path = write_results(args.out, records, summary)
        env.close()
        print_summary(summary)
        print("wrote {} and {}".format(jsonl_path, summary_path), flush=True)
        if video_dir is not None:
            print("videos in {}".format(video_dir), flush=True)


def self_test():
    instruction = "将方块向上抬起 14 厘米。"
    assert user_text(instruction, "{instruction}") == "\n" + instruction
    assert user_text("<image>\n" + instruction, "{instruction}") == "\n" + instruction
    assert user_text("<image><image>\n" + instruction, "{instruction}") == "\n" + instruction
    views_note = "第一张图是正面相机（frontview），第二张图是背面相机（backview）。"
    assert user_text("<image>\n<image>\n" + views_note + instruction, "{instruction}") == "\n" + views_note + instruction
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    messages = chat_messages(rgb, instruction, "", "{instruction}")
    assert len(messages[-1]["content"]) == 2
    assert messages[-1]["content"][0]["type"] == "image_url"
    assert messages[-1]["content"][1]["type"] == "text"
    dual = chat_messages([rgb, rgb], instruction, "", "{instruction}")
    assert [part["type"] for part in dual[-1]["content"]] == ["image_url", "image_url", "text"]
    seen = {}

    def capture(payload):
        seen["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    assert post_chat(capture, "m", messages, 0.0, 8, [True]) == "ok"
    assert "depth_meters" not in seen["payload"]
    assert len(seen["payload"]["messages"][-1]["content"]) == 2
    matrix = np.array(
        [
            [100.0, 0.0, 256.0, 0.0],
            [0.0, 100.0, 256.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    world = np.array([0.12, -0.07, 0.83])
    pixel, depth_z = project_world(world, matrix)
    assert np.allclose(unproject_pixel(pixel, depth_z, matrix), world)
    assert np.allclose(unproject_pixel(round_pixel(pixel), depth_z, matrix), world, atol=1e-2)
    text = format_answer(
        {"target_1": "[251 189]", "target_2": "[268 191]"},
        [-0.0974, 0.0752, 0.8216],
        [-0.0840, 0.0895, 0.9939],
        "initial",
        False,
        0.14,
    )
    plan = parse_answer("<think>先看图</think>\n" + text, image_size=512)
    assert np.allclose(plan["target"], [251, 189])
    assert np.allclose(plan["target_2"], [268, 191])
    assert np.allclose(plan["subtasks"][0]["pixel"], [251, 189])
    assert np.allclose(plan["subtasks"][0]["pixel_2"], [268, 191])
    assert '"[251 189]"' in text
    raw = json.loads(text)
    assert "desc" not in raw and all("desc" in step for step in raw["subtasks"])
    assert raw["subtasks"][0]["target_1"] == "[251 189]"
    assert [step["type"] for step in plan["subtasks"]] == [
        "move_above",
        "move_down",
        "close_gripper",
        "lift",
        "stabilize",
    ]
    assert plan["subtasks"][3]["delta_z"] == 0.14
    assert plan["subtasks"][4]["duration"] == 0.5
    assert subtasks_match(plan, text)
    comma = '{"subtasks": [{"type": "move_down", "target": "[320, 180]"}]}'
    plan = parse_answer("```json\n" + comma + "\n```", image_size=512)
    assert np.allclose(plan["target"], [320.0, 180.0])
    assert plan["subtasks"][0]["type"] == "move_down"
    two_views = '{"subtasks": [{"type": "move_down", "target": "[282 314]", "target_1_back": "[150 262]"}, {"type": "close_gripper"}]}'
    plan = parse_answer(two_views, image_size=512)
    assert np.allclose(plan["target"], [282, 314])
    assert np.allclose(plan["target_back"], [150, 262])
    assert np.allclose(plan["subtasks"][0]["pixel_back"], [150, 262])
    assert plan["target_2"] is None and plan["target_2_back"] is None
    # 被 mask 的视图解析成 None：不脑补这张图的像素，执行时用另一台相机。
    masked = (
        '{"subtasks": [{"type": "move_down", "target_1": "mask", "target_1_back": "[150 262]", '
        '"target_2": "[301 318]", "target_2_back": "mask"}, {"type": "close_gripper"}]}'
    )
    plan = parse_answer(masked, image_size=512)
    assert plan["target"] is None
    assert np.allclose(plan["target_back"], [150, 262])
    assert np.allclose(plan["target_2"], [301, 318])
    assert plan["target_2_back"] is None
    assert plan["subtasks"][0]["pixel"] is None
    assert plan["subtasks"][0]["pixel_2_back"] is None
    empty = '{"subtasks": []}'
    assert parse_answer(empty, image_size=512)["target"] is None
    prose = '计划：\n{"subtasks": [{"type": "lift", "target": "[200 180]", "delta_z": 0.1}, {"type": "stabilize", "duration": 0.5}]}'
    plan = parse_answer(prose)
    assert steps_to_run(plan["subtasks"])[0]["type"] == "lift"
    assert steps_to_run(plan["subtasks"])[1]["type"] == "stabilize"
    ruled = control_steps(0.14)
    assert [step["type"] for step in ruled] == [
        "move_above",
        "move_down",
        "close_gripper",
        "lift",
        "stabilize",
    ]
    assert ruled[3]["delta_z"] == 0.14
    assert control_steps(0.0)[3]["type"] == "stabilize"
    above = format_answer(
        {"target_1": "[200 180]", "target_2": "[220 184]"}, [0.0, 0.0, 0.82], [0.0, 0.0, 0.95], "initial", True, 0.09
    )
    assert parse_answer(above, image_size=512)["subtasks"][0]["type"] == "move_down"
    holding = format_answer(
        {"target_1": "[200 180]", "target_2": "[220 184]"}, [0.0, 0.0, 0.90], [0.0, 0.0, 0.90], "lift", True, 0.09
    )
    assert [step["type"] for step in parse_answer(holding, image_size=512)["subtasks"]] == ["lift", "stabilize"]
    try:
        parse_answer('{"subtasks": [{"type": "move_down"}]}', image_size=512)
        raise AssertionError("空间步骤缺少 target 应拒绝")
    except ParseError:
        pass
    try:
        parse_answer(
            '{"subtasks": [{"type": "move_down", "target": "[0.1 -0.2 0.82]"}]}',
            image_size=512,
        )
        raise AssertionError("三维 target 应拒绝")
    except ParseError:
        pass
    err = pos_error([0.01, -0.02, 0.83], [0.0, 0.0, 0.82])
    assert abs(err["x"] - 0.01) < 1e-9 and abs(err["y"] + 0.02) < 1e-9 and abs(err["z"] - 0.01) < 1e-9
    stats = axis_means([err, pos_error([0.03, 0.0, 0.82], [0.0, 0.0, 0.82])])
    assert abs(stats["x"]["mae_m"] - 0.02) < 1e-9 and abs(stats["x"]["bias_m"] - 0.02) < 1e-9
    assert abs(stats["y"]["bias_m"] + 0.01) < 1e-9
    assert axis_hit_rate([err], 0.012)["x"] == 1.0
    assert axis_hit_rate([err], 0.012)["y"] == 0.0
    assert "x=+1.0cm" in format_err_axes(err)
    assert "|x|=2.0cm" in format_axis_means(stats)
    packed = pack_query(
        "move_down",
        "将方块向上抬起 14 厘米。",
        {
            "answer": text,
            "plan": {
                "pixel": [251, 189],
                "target": [-0.0974, 0.0752, 0.8216],
                "subtasks": plan["subtasks"],
            },
            "gt_cube_pos": [0.0, 0.0, 0.82],
            "gt_eef_pos": [0.0, 0.0, 0.95],
            "gt_pixel": [250.0, 188.0],
            "cube_depth_m": 1.6,
            "cube_err": err,
            "seconds": 0.1,
        },
        True,
        0.01,
    )
    assert packed["cube_x_err_m"] == 0.01 and packed["cube_y_err_m"] == -0.02 and packed["cube_z_err_m"] == 0.01
    assert packed["pixel"] == [251, 189] and packed["gt_pixel"] == [250, 188]
    assert packed["pixel_err"] > 0.0
    backed = parse_answer(two_views, image_size=512)
    packed = pack_query(
        "move_down",
        "将方块向上抬起 14 厘米。",
        {
            "answer": two_views,
            "plan": {
                "pixel": backed["target"],
                "pixel_back": backed["target_back"],
                "target": [-0.0974, 0.0752, 0.8216],
                "subtasks": backed["subtasks"],
            },
            "gt_cube_pos": [0.0, 0.0, 0.82],
            "gt_eef_pos": [0.0, 0.0, 0.95],
            "gt_pixel": [250.0, 188.0],
            "gt_pixel_back": [149.0, 261.0],
            "cube_depth_m": 1.6,
            "cube_err": err,
            "seconds": 0.1,
        },
        True,
        0.01,
    )
    assert isinstance(backed["subtasks"][0]["pixel_back"], np.ndarray)
    assert packed["subtasks"][0]["pixel"] == [282, 314]
    assert packed["subtasks"][0]["pixel_back"] == [150, 262]
    assert packed["pixel_back"] == [150, 262] and packed["gt_pixel_back"] == [149, 261]
    json.dumps(packed)
    assert query_pos_error(packed)["y"] == -0.02
    assert completion_text({"choices": [{"message": {"content": text}}]}) == text
    assert completion_text({"response": comma}) == comma
    # 新命名：每个落点各写两台相机，target_1/target_1_back 是一个，target_2/target_2_back 是另一个。
    named = (
        '{"desc": "target_1 是夹爪1落点的像素", "subtasks": ['
        '{"type": "move_down", "desc": "降到方块处", "target_1": "[282 314]", "target_1_back": "[150 262]", '
        '"target_2": "[301 318]", "target_2_back": "[166 268]"}, '
        '{"type": "close_gripper", "desc": "合爪"}]}'
    )
    plan = parse_answer(named, image_size=512)
    assert np.allclose(plan["target"], [282, 314])
    assert np.allclose(plan["target_back"], [150, 262])
    assert np.allclose(plan["target_2"], [301, 318])
    assert np.allclose(plan["target_2_back"], [166, 268])
    print("self-test ok", flush=True)


def main():
    parser = argparse.ArgumentParser(description="直连 vLLM，按抓取计划在 robosuite Lift 上测成功率")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI 接口地址")
    add_common_args(parser)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.oracle:
        client = OracleClient()
    else:
        client = VLLMClient(
            args.api_base,
            args.model,
            args.timeout,
            args.max_tokens,
            args.temperature,
            args.system,
            args.text_template,
        )
        model = client.resolve_model()
        print("api {} model {}".format(client.api_base, model), flush=True)

    run_eval(
        args,
        client,
        "vllm",
        extra_summary=lambda item: {"api_base": item.api_base},
    )


if __name__ == "__main__":
    main()

