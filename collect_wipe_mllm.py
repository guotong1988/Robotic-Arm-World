"""采集 robosuite Wipe：用末端的擦板把桌上的一条污渍擦干净。

污渍是 100 个小圆点连成的一条曲线，擦板压着桌面从圆点上方经过，圆点就消失。
步骤是：移到第一处污渍上方、降到桌面、沿污渍压着走（wipe）、抬起、停住。
机械臂初始朝向偏了 7 度，擦板斜着只有一角着地，所以第一步带 yaw 0，把擦板摆平。
wipe 的 path 是还没擦掉的污渍，按曲线顺序每隔约 3 厘米取一点，最后一点一定写上。
偶尔会漏掉几点，这时从当前画面重新规划一遍，只擦剩下的。全部擦完后 subtasks 为空。

Wipe 换的是擦板，没有夹爪，计划里不会出现 close_gripper、open_gripper。

输入是一张 frontview RGB，对应样本的 image 字段；指令以一个 <image> 占位开头，
话术后面带 answer 各字段的说明：

    <image>
    把桌上的污渍擦干净。回答的 JSON 中，target_1 和 path 是对应点在正面相机图上的像素。

输出例子（还没开始擦时；path 按曲线每隔约 3 厘米取点，这里只示意几个点）：

    {"subtasks": [{"type": "move_above", "desc": "移到目标点上方", "target_1": "[220 310]", "target_kind": "table", "clearance": 0.1, "yaw": 0}, {"type": "move_down", "desc": "降到目标点处", "target_1": "[220 310]", "target_kind": "table", "offset_z": 0.0}, {"type": "wipe", "desc": "沿路径擦拭桌面", "path": ["[220 310]", "[250 300]", "[280 292]", "[310 286]"], "offset_z": -0.01, "path_kind": "table"}, {"type": "lift", "desc": "竖直抬起物体", "delta_z": 0.1, "target_1": "[310 286]", "target_kind": "table"}, {"type": "stabilize", "desc": "停在原地保持稳定", "duration": 0.5}]}

每条成功轨迹只存开局那一帧。污渍在桌面上换成横线、竖线、斜线、弧线或两段分开的痕迹，
长短和位置每条都不一样。中途画面不存。

例如凑满 50 条成功擦拭再停，图片和样本写到 data/mllm：

python collect_wipe_mllm.py --episodes 50 --out data/mllm
"""

import argparse

import numpy as np

from collect_common import add_common_args, run_collection
from task_plan import WIPE_PRESS

# 先验知识：擦板初始偏了 7 度，第一步 yaw 固定为 0 把它摆平；污渍按约 3 厘米取点。不解关节角。
WAYPOINT_SPACING = 0.03
HOVER_CLEARANCE = 0.10
LIFT_HEIGHT = 0.10


def _stroke(rng, n, limit, along):
    """一条痕迹。along 为 x 或 y 时大致沿桌边，否则沿随机方向，偶尔弯成弧。"""
    if along == "x":
        heading = float(rng.choice([0.0, np.pi])) + float(rng.uniform(-0.25, 0.25))
        length = float(rng.uniform(0.16, 0.36))
    elif along == "y":
        heading = float(rng.choice([np.pi / 2.0, -np.pi / 2.0])) + float(rng.uniform(-0.2, 0.2))
        length = float(rng.uniform(0.22, 0.50))
    else:
        heading = float(rng.uniform(-np.pi, np.pi))
        length = float(rng.uniform(0.20, 0.42))
    center = np.array(
        [
            rng.uniform(-limit[0] * 0.35, limit[0] * 0.35),
            rng.uniform(-limit[1] * 0.40, limit[1] * 0.40),
        ]
    )
    travel = np.linspace(-0.5 * length, 0.5 * length, n)
    lateral = np.zeros(n)
    if along is None and rng.random() < 0.7:
        amp = float(rng.uniform(0.02, 0.05))
        turns = float(rng.choice([1.0, 2.0]))
        lateral = amp * np.sin(np.linspace(0.0, turns * 2.0 * np.pi, n))
    cosine, sine = np.cos(heading), np.sin(heading)
    return np.stack(
        [center[0] + cosine * travel - sine * lateral, center[1] + sine * travel + cosine * lateral],
        axis=1,
    )


def _dirt_path(rng, n, half):
    """n 个污渍点。横、竖、斜、弧，或者桌面上分开的两段。出界就重抽。"""
    limit = np.asarray(half[:2], dtype=float) * 0.82
    kinds = ("x", "y", None, "two")
    for _ in range(24):
        kind = kinds[int(rng.integers(len(kinds)))]
        if kind == "two":
            first = _stroke(rng, n // 2, limit, ("x", "y")[int(rng.integers(2))])
            second = _stroke(rng, n - n // 2, limit, ("x", "y")[int(rng.integers(2))])
            if np.linalg.norm(first[-1] - second[0]) < 0.08:
                continue
            pts = np.concatenate([first, second], axis=0)
        else:
            pts = _stroke(rng, n, limit, kind)
        if np.all(np.abs(pts) <= limit):
            return pts
    travel = np.linspace(-0.12, 0.12, n)
    return np.stack([travel, np.zeros(n)], axis=1)


STYLES = (
    ("把桌上的污渍擦干净。", "把桌上剩下的污渍擦干净。", "桌面已经擦干净了。"),
    ("擦掉桌面上的污渍。", "继续擦掉剩下的污渍。", "污渍已经擦完了。"),
    ("把桌上那道脏印子擦掉。", "把剩下的脏印子擦掉。", "桌上已经没有污渍了。"),
)


class WipeTask:
    env_name = "Wipe"
    task = "wipe"
    primitives = ("move_above", "move_down", "wipe", "lift", "stabilize")
    max_rounds = 2
    settle_steps = 5
    styles = STYLES
    instruction_note = "还没开始擦时，指令是把污渍擦干净；擦掉一部分之后改成擦剩下的；擦完说明已经完成。同一条轨迹里用同一套说法。"

    def env_kwargs(self):
        return {}

    def vary_initial(self, env, rng):
        """覆盖环境自己撒下的污渍，改成一种选定的痕迹。点距和原来接近，擦板仍能顺着擦完。"""
        arena = env.model.mujoco_arena
        markers = arena.markers
        half = np.asarray(arena.table_half_size[:2], dtype=float)
        pts = _dirt_path(rng, len(markers), half)
        for marker, xy in zip(markers, pts):
            body_id = env.sim.model.body_name2id(marker.root_body)
            env.sim.model.body_pos[body_id] = np.array([xy[0], xy[1], arena.table_half_size[2]])
            geom_id = env.sim.model.geom_name2id(marker.visual_geoms[0])
            env.sim.model.geom_rgba[geom_id][3] = 1.0

    def markers(self, env):
        model, data = env.sim.model, env.sim.data
        return [np.array(data.body_xpos[model.body_name2id(marker.root_body)], dtype=float) for marker in env.model.mujoco_arena.markers]

    def remaining(self, env):
        wiped = set(id(marker) for marker in env.wiped_markers)
        return [k for k, marker in enumerate(env.model.mujoco_arena.markers) if id(marker) not in wiped]

    def wipe_path(self, env):
        positions = self.markers(env)
        left = self.remaining(env)
        chosen = []
        for k in left:
            if not chosen or np.linalg.norm(positions[k][:2] - positions[chosen[-1]][:2]) >= WAYPOINT_SPACING:
                chosen.append(k)
        if left and chosen[-1] != left[-1]:
            chosen.append(left[-1])
        return ["dirt{}".format(k) for k in chosen]

    def points(self, env):
        positions = self.markers(env)
        out = {"dirt{}".format(k): pos for k, pos in enumerate(positions)}
        path = self.wipe_path(env)
        if path:
            out["path_first"] = out[path[0]]
            out["path_last"] = out[path[-1]]
        return out

    def landmarks(self, env):
        return self.markers(env)

    def point_kind(self, name):
        """污渍都贴在桌面上。"""
        return "table"

    def record_points(self, env):
        return [name for name in ("path_first", "path_last") if name in self.points(env)]

    def build_plan(self, env, rng):
        table_z = float(self.markers(env)[0][2])
        return [
            {"type": "move_above", "ref": "path_first", "clearance": HOVER_CLEARANCE, "yaw": 0.0},
            {"type": "move_down", "ref": "path_first", "offset_z": 0.0},
            {"type": "wipe", "offset_z": -WIPE_PRESS},
            {"type": "lift", "ref": "path_last", "goal_z": table_z + LIFT_HEIGHT},
            {"type": "stabilize"},
        ]

    def instruction(self, env, plan, index, style, done):
        if done:
            return style[2]
        return style[1] if env.wiped_markers else style[0]

    def success(self, env):
        return bool(env._check_success())

    def start_pose(self, env, rng):
        table = np.asarray(env.table_offset, dtype=float)
        start = np.array(
            [table[0] + rng.uniform(-0.15, 0.05), rng.uniform(-0.20, 0.20), table[2] + rng.uniform(0.10, 0.22)]
        )
        return start, float(rng.uniform(-np.pi / 2.0, np.pi / 2.0)), table[2] + 0.22

    def record_extra(self, env):
        return {"wiped": len(env.wiped_markers), "num_markers": int(env.num_markers)}

    def episode_info(self, env):
        return {"wiped": len(env.wiped_markers)}

    def meta_extra(self):
        return {
            "waypoint_spacing_m": WAYPOINT_SPACING,
            "press_m": WIPE_PRESS,
            "point_names": "dirtK 是第 K 个污渍点；path_first、path_last 是还没擦掉的污渍里按顺序的第一点和最后一点",
        }

    def describe(self, env):
        return "wiped={}/{}".format(len(env.wiped_markers), env.num_markers)


TASKS = {"Wipe": WipeTask}


def main():
    parser = argparse.ArgumentParser(description="采集 Wipe / Panda 的 MLLM 图文步骤")
    add_common_args(parser, "data/wipe_mllm")
    args = parser.parse_args()
    run_collection(WipeTask(), args)


if __name__ == "__main__":
    main()
