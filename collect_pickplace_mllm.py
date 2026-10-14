"""采集 robosuite PickPlace 系列：把左边箱子里的物品放进右边箱子对应的格子。

--env 可选 PickPlace（四样都放）、PickPlaceSingle（每次随机一样）、
PickPlaceMilk、PickPlaceBread、PickPlaceCereal、PickPlaceCan（固定一样）。
右边箱子每个格子里有对应物品的半透明影子，指令只说放进对应的格子。

每样物品的步骤：移到物品上方、下降、合爪、抬过箱壁、移到对应格子上方、下降、松爪、抬开。
全部放完后停住。麦片盒 10 厘米宽、3 厘米厚，只能沿窄边合拢，所以 move_above 带 yaw。
牛奶、面包两条边都夹得下，选和最近邻居方向垂直的那条，免得手指碰到旁边的东西；
易拉罐是圆的，旁边有东西时同样按邻居定 yaw，单独一个时不写。
抓靠上的位置，move_down 的 offset_z 是相对物品中心的高度，顶面按碰撞网格算。
四样都放时，先拿离其他物品最远的那样。

    python collect_pickplace_mllm.py --env PickPlaceCan --episodes 50 --out data/mllm
"""

import argparse

import numpy as np

from collect_common import add_common_args, body_yaw_deg, choose_yaws, is_grasping, run_collection, wrist_yaw
from task_plan import DEFAULT_CLEARANCE

ENVS = ("PickPlace", "PickPlaceSingle", "PickPlaceMilk", "PickPlaceBread", "PickPlaceCereal", "PickPlaceCan")
OBJECT_CN = {"Milk": "牛奶", "Bread": "面包", "Cereal": "麦片盒", "Can": "易拉罐"}
# 物品绕竖直轴的对称周期，度；None 表示圆柱，两指朝哪个方向合拢都行。
YAW_PERIOD = {"Milk": 90.0, "Bread": 180.0, "Cereal": 180.0, "Can": None}
# 两指能夹住的物品边长上限，米。夹爪最大张开约 8 厘米。
MAX_GRASP_WIDTH = 0.06
# 指尖停在物品顶面以下这么深，米。
GRASP_DEPTH = 0.03
# 箱壁顶面相对箱子原点的高度，米。
WALL_HEIGHT = 0.10
# 运输时物品底部比箱壁再高出这么多，米。
CARRY_MARGIN = 0.05
# 松爪时物品离格子底面还有这么高，米。
DROP_HEIGHT = 0.03
SINGLE_STYLES = (
    ("把{0}放进右边箱子里对应的格子。", "把手里的{0}放进右边箱子里对应的格子。", "{0}已经放好了。"),
    ("将{0}从左边箱子拿到右边箱子的对应格子里。", "将拿着的{0}放到右边箱子的对应格子里。", "{0}已经放进对应的格子。"),
    ("把左边箱子里的{0}放到右边对应的格子中。", "把拿着的{0}放到右边对应的格子中。", "{0}已经放到位了。"),
)
ALL_STYLES = (
    (
        "把左边箱子里的物品都放进右边箱子对应的格子。",
        "把左边箱子里剩下的物品放进右边箱子对应的格子。",
        "先把手里的{0}放进对应的格子，再放剩下的物品。",
        "物品都已经放好了。",
    ),
    (
        "将左边箱子里的四样物品分别放到右边箱子的对应格子里。",
        "将左边箱子里剩下的物品放到右边箱子的对应格子里。",
        "先把拿着的{0}放到对应格子里，再处理剩下的物品。",
        "四样物品都已经放好了。",
    ),
)


class PickPlaceTask:
    primitives = ("move_above", "move_down", "close_gripper", "lift", "open_gripper", "stabilize")
    max_rounds = 1
    settle_steps = 20

    def __init__(self, env_name):
        self.env_name = env_name
        self.task = env_name.lower()
        self._extents = {}
        self.all_objects = env_name == "PickPlace"
        self.styles = ALL_STYLES if self.all_objects else SINGLE_STYLES
        self.instruction_note = (
            "物品还没抓住时，指令是把它放进右边箱子对应的格子；抓住之后改成把手里的物品放好；放完说明已经完成。"
            "四样都放时，没拿东西的指令只说把（剩下的）物品放好，拿着东西时点名手里那样。同一条轨迹里用同一套说法。"
        )

    def env_kwargs(self):
        return {}

    def _index(self, env, name):
        return env.object_to_id[name.lower()]

    def active(self, env):
        if env.single_object_mode == 0:
            return [obj.name for obj in env.objects]
        return [env.objects[env.object_id].name]

    def placed(self, env):
        env._check_success()
        return [name for name in self.active(env) if env.objects_in_bins[self._index(env, name)]]

    def points(self, env):
        out = {}
        for name in self.active(env):
            out[name] = np.array(env.sim.data.body_xpos[env.obj_body_id[name]], dtype=float)
            out[name + "_bin"] = np.array(env.target_bin_placements[self._index(env, name)], dtype=float)
        return out

    def landmarks(self, env):
        return list(self.points(env).values())

    def record_points(self, env):
        return list(self.points(env).keys())

    def extent(self, env, name):
        """物品碰撞网格在自身坐标里的上下界。top_offset 不可靠，麦片盒写的是 3 厘米，实际顶面高 7.5 厘米。"""
        if name not in self._extents:
            model, data = env.sim.model, env.sim.data
            body = env.obj_body_id[name]
            mat = np.asarray(data.body_xmat[body], dtype=float).reshape(3, 3)
            points = []
            for geom in range(model.ngeom):
                if model.geom_bodyid[geom] != body or model.geom_contype[geom] == 0 or model.geom_type[geom] != 7:
                    continue
                mesh = model.geom_dataid[geom]
                start = model.mesh_vertadr[mesh]
                verts = np.asarray(model.mesh_vert[start : start + model.mesh_vertnum[mesh]], dtype=float)
                rot = np.asarray(data.geom_xmat[geom], dtype=float).reshape(3, 3)
                points.append(verts @ rot.T + data.geom_xpos[geom])
            local = (np.concatenate(points) - data.body_xpos[body]) @ mat
            self._extents[name] = (local.min(axis=0), local.max(axis=0))
        return self._extents[name]

    def grasp_yaw(self, env, name, points, neighbors):
        """两指合拢方向尽量和最近的邻居方向垂直，免得手指碰到旁边的物品。
        方的物品只能沿自身 x 或 y 合拢，而且那条边要夹得下；易拉罐是圆的，方向随意，没有邻居时不写 yaw。
        topdown yaw 为 0 时两指沿世界 y 合拢，所以 yaw 是合拢方向的角度减 90 度。"""
        near = None
        if neighbors:
            closest = min(neighbors, key=lambda other: np.linalg.norm(points[other][:2] - points[name][:2]))
            near = points[closest][:2] - points[name][:2]
            near = near / max(float(np.linalg.norm(near)), 1e-6)
        if YAW_PERIOD[name] is None:
            if near is None:
                return None
            return float(np.degrees(np.arctan2(near[0], -near[1]))) - 90.0
        body_yaw = np.radians(body_yaw_deg(env, env.obj_body_id[name]))
        low, high = self.extent(env, name)
        size = high - low
        axes = [k for k in (0, 1) if size[k] <= MAX_GRASP_WIDTH] or [int(np.argmin(size[:2]))]
        angles = [body_yaw + (0.0 if k == 0 else np.pi / 2.0) for k in axes]
        if near is None:
            angle = angles[int(np.argmin([size[k] for k in axes]))]
        else:
            angle = max(angles, key=lambda a: abs(np.cos(a) * near[1] - np.sin(a) * near[0]))
        return float(np.degrees(angle)) - 90.0

    def order(self, env):
        points = self.points(env)
        done = set(self.placed(env))
        left = [name for name in self.active(env) if name not in done]
        ordered = []
        while left:
            def room(name):
                others = [np.linalg.norm(points[name][:2] - points[other][:2]) for other in left if other != name]
                return min(others) if others else 0.0

            best = max(left, key=room)
            ordered.append(best)
            left.remove(best)
        return ordered

    def build_plan(self, env, rng):
        points = self.points(env)
        floor = float(env.bin1_pos[2])
        bin_z = float(env.bin2_pos[2])
        order = self.order(env)
        bases = [self.grasp_yaw(env, name, points, order[k + 1 :]) for k, name in enumerate(order)]
        chosen = iter(choose_yaws(wrist_yaw(env), [(base, 0.0) for base in bases if base is not None]))
        yaws = [None if base is None else next(chosen)[0] for base in bases]
        plan = []
        for name, yaw in zip(order, yaws):
            top = float(self.extent(env, name)[1][2])
            rest = float(points[name][2]) - floor
            grasp = max(0.0, top - GRASP_DEPTH)
            carry_z = bin_z + WALL_HEIGHT + CARRY_MARGIN + grasp + rest
            above = {"type": "move_above", "ref": name, "clearance": max(DEFAULT_CLEARANCE, top + 0.06), "obj": name}
            if yaw is not None:
                above["yaw"] = yaw
            target = name + "_bin"
            plan += [
                above,
                {"type": "move_down", "ref": name, "offset_z": grasp, "obj": name},
                {"type": "close_gripper", "obj": name},
                {"type": "lift", "ref": name, "goal_z": carry_z, "obj": name},
                {"type": "move_above", "ref": target, "clearance": carry_z - bin_z, "obj": name},
                {"type": "move_down", "ref": target, "offset_z": rest + grasp + DROP_HEIGHT, "obj": name},
                {"type": "open_gripper", "obj": name},
                {"type": "move_above", "ref": target, "clearance": carry_z - bin_z, "obj": name},
            ]
        plan.append({"type": "stabilize"})
        return plan

    def holding(self, env, plan, index):
        """正在处理的物品，以及是不是已经抓在手里。"""
        if not 0 <= index < len(plan) or "obj" not in plan[index]:
            return None, False
        name = plan[index]["obj"]
        kinds = [(k, step["type"]) for k, step in enumerate(plan) if step.get("obj") == name]
        closed = [k for k, kind in kinds if kind == "close_gripper"]
        opened = [k for k, kind in kinds if kind == "open_gripper"]
        between = bool(closed) and closed[0] < index <= opened[0]
        obj = env.objects[self._index(env, name)]
        return name, between and is_grasping(env, obj)

    def instruction(self, env, plan, index, style, done):
        name, held = self.holding(env, plan, index)
        if not self.all_objects:
            label = OBJECT_CN[self.active(env)[0]]
            if done:
                return style[2].format(label)
            return style[1 if held else 0].format(label)
        if done:
            return style[3]
        later = [step["obj"] for step in plan[index + 1 :] if step.get("obj") not in (None, name)]
        if held:
            if later:
                return style[2].format(OBJECT_CN[name])
            return SINGLE_STYLES[0][1].format(OBJECT_CN[name])
        return style[1] if self.placed(env) else style[0]

    def success(self, env):
        return bool(env._check_success())

    def start_pose(self, env, rng):
        bin1 = np.asarray(env.bin1_pos, dtype=float)
        start = np.array(
            [bin1[0] + rng.uniform(-0.12, 0.10), bin1[1] + rng.uniform(-0.15, 0.20), rng.uniform(1.0, 1.12)]
        )
        return start, float(rng.uniform(-np.pi / 2.0, np.pi / 2.0)), 1.12

    def record_extra(self, env):
        return {"placed": self.placed(env)}

    def episode_info(self, env):
        return {"objects": self.active(env)}

    def meta_extra(self):
        return {
            "objects_cn": OBJECT_CN,
            "grasp_depth_m": GRASP_DEPTH,
            "drop_height_m": DROP_HEIGHT,
            "point_names": "物品名是物品中心，物品名_bin 是右边箱子里对应格子底面的中心",
        }

    def describe(self, env):
        return "placed={}".format("+".join(self.placed(env)) or "-")


TASKS = {name: (lambda name=name: PickPlaceTask(name)) for name in ENVS}


def main():
    parser = argparse.ArgumentParser(description="采集 PickPlace / Panda 的 MLLM 图文步骤")
    parser.add_argument("--env", choices=ENVS, default="PickPlaceCan")
    add_common_args(parser, "data/pickplace_mllm")
    args = parser.parse_args()
    run_collection(PickPlaceTask(args.env), args)


if __name__ == "__main__":
    main()
