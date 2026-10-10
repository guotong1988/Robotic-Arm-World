"""采集 robosuite Wipe：用末端的擦板把桌上的一条污渍擦干净。

污渍是 100 个小圆点连成的一条曲线，擦板压着桌面从圆点上方经过，圆点就消失。
步骤是：移到第一处污渍上方、降到桌面、沿污渍压着走（wipe）、抬起、停住。
机械臂初始朝向偏了 7 度，擦板斜着只有一角着地，所以第一步带 yaw 0，把擦板摆平。
wipe 的 path 是还没擦掉的污渍，按曲线顺序每隔约 3 厘米取一点，最后一点一定写上。
偶尔会漏掉几点，这时从当前画面重新规划一遍，只擦剩下的。全部擦完后 subtasks 为空。

Wipe 换的是擦板，没有夹爪，计划里不会出现 close_gripper、open_gripper。

    python collect_wipe_mllm.py --episodes 50 --out data/mllm
"""

import argparse

import numpy as np

from collect_common import add_common_args, run_collection
from task_plan import WIPE_PRESS

WAYPOINT_SPACING = 0.03
HOVER_CLEARANCE = 0.10
LIFT_HEIGHT = 0.10
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
