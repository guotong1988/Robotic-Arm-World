"""采集 robosuite PickPlace 系列：把左边箱子里的物品放进右边箱子对应的格子。

--env 可选 PickPlace（四样都放）、PickPlaceSingle（每次随机一样）、
PickPlaceMilk、PickPlaceBread、PickPlaceCereal、PickPlaceCan（固定一样）。
右边箱子每个格子里有对应物品的半透明影子，指令只说放进对应的格子。

每样物品的步骤：移到物品上方、下降、合爪、抬过箱壁、移到对应格子上方、下降、松爪、抬开。
全部放完后停住。麦片盒 10 厘米宽、3 厘米厚，只能沿窄边合拢；牛奶、面包选和最近邻居方向垂直的那条，
免得手指碰到旁边的东西；易拉罐旁边有东西时同样按邻居选方向。采集器内部用 yaw 生成两指落点，
标签不重复写 yaw，执行时由落点连线恢复。
抓靠上的位置，两个指腹落点的中点已含抓取高度，所以 move_down 的 offset_z 为 0。
四样都放时，先拿离其他物品最远的那样。

输入例子。每台相机一张 RGB，默认 frontview 加 backview，对应样本的 image、image_back 字段；
指令开头的 <image> 每行一个，和图片一一对应，多张图时占位后面注明每张来自哪台相机；话术后面带 answer 各字段的说明：

    <image>
    <image>
    第一张图是正面相机（frontview），第二张图是背面相机（backview）。把易拉罐放进右边箱子里对应的格子。回答的 JSON 中，抓物品的步骤里 target_1、target_2 是两指落点，每个落点各给两台相机的像素；某台相机看不到那个落点时对应字段写 "mask"，不要猜它的像素。两落点的中点是抓取位置；放进箱格的步骤只写格子中心。

单目只开 frontview，一个占位后面接话术和字段说明：

    <image>
    把易拉罐放进右边箱子里对应的格子。回答的 JSON 中，抓物品的步骤里 target_1、target_2 是两指落点，都在正面相机图上；另一个落点被挡住时也根据物品形状和朝向补出它的像素，不写 mask。放进箱格的步骤只写格子中心。

输出例子（易拉罐单独一个、还没抓住时；抓取步骤写两个指腹落点，中点是抓取位置，连线决定 yaw，
不再单独输出 yaw。双目时每个落点各给两台相机的像素，某台相机看不到那个点时对应字段写 "mask"）：

    {"subtasks": [{"type": "move_above", "desc": "移到易拉罐上方", "target_1": "[180 290]", "target_1_back": "mask", "target_2": "mask", "target_2_back": "[220 250]", "target_kind": "grip", "clearance": 0.12}, {"type": "move_down", "desc": "降到易拉罐处准备夹", "target_1": "[180 290]", "target_1_back": "mask", "target_2": "mask", "target_2_back": "[220 250]", "target_kind": "grip", "offset_z": 0.0}, {"type": "close_gripper", "desc": "合爪夹住易拉罐"}, {"type": "lift", "desc": "抬起易拉罐", "delta_z": 0.17, "target_1": "[180 290]", "target_1_back": "mask", "target_2": "mask", "target_2_back": "[220 250]", "target_kind": "grip"}, {"type": "move_above", "desc": "移到格子正上方", "target_1": "[360 270]", "target_1_back": "[155 255]", "target_kind": "bin", "clearance": 0.22}, {"type": "move_down", "desc": "把易拉罐放进格子", "target_1": "[360 270]", "target_1_back": "[155 255]", "target_kind": "bin", "offset_z": 0.1}, {"type": "open_gripper", "desc": "松爪放下易拉罐"}, {"type": "move_above", "desc": "向上抬开夹爪", "target_1": "[360 270]", "target_1_back": "[155 255]", "target_kind": "bin", "clearance": 0.22}, {"type": "stabilize", "desc": "停在原地保持稳定", "duration": 0.5}]}

每条成功轨迹只存开局那一帧。左边箱子里的物品会换位置和朝向，
右边格子里的半透明目标也会在格子内部挪开中心。中途画面不存。

例如固定采易拉罐，凑满 50 条成功轨迹再停，图片和样本写到 data/mllm：

python collect_pickplace_mllm.py --env PickPlaceCan --episodes 50 --out data/mllm
"""

import argparse

import numpy as np

from collect_common import (
    add_common_args,
    body_yaw_deg,
    choose_yaws,
    is_grasping,
    order_landings_at,
    run_collection,
    wrist_yaw,
)
from collect_lift_mllm import DEFAULT_CAMERA, back_enabled, set_free_xyaw
from grip_ik import box_contacts
from task_plan import DEFAULT_CLEARANCE

ENVS = ("PickPlace", "PickPlaceSingle", "PickPlaceMilk", "PickPlaceBread", "PickPlaceCereal", "PickPlaceCan")
OBJECT_CN = {"Milk": "牛奶", "Bread": "面包", "Cereal": "麦片盒", "Can": "易拉罐"}
# 先验知识：哪条边夹得下、指尖停多深、要抬过多高的箱壁。只决定末端到哪、朝哪，不解关节角。
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

    def _bin_center(self, env, index):
        """右边箱子第 index 个格子的底面中心。和环境里 target_bin_placements 的算法一致。"""
        bin_x = float(env.bin2_pos[0])
        bin_y = float(env.bin2_pos[1])
        if index == 0 or index == 2:
            bin_x -= float(env.bin_size[0]) / 2.0
        if index < 2:
            bin_y -= float(env.bin_size[1]) / 2.0
        bin_x += float(env.bin_size[0]) / 4.0
        bin_y += float(env.bin_size[1]) / 4.0
        return np.array([bin_x, bin_y, float(env.bin2_pos[2])])

    def vary_initial(self, env, rng):
        """物品在左箱里换位置，右箱格子里的目标影子离开格子中心。影子和标注用同一个点。"""
        for name in self.active(env):
            index = self._index(env, name)
            center = self._bin_center(env, index)
            center[0] += float(rng.uniform(-0.025, 0.025))
            center[1] += float(rng.uniform(-0.03, 0.03))
            env.target_bin_placements[index] = center
            visual = env.obj_body_id["Visual" + name]
            pos = np.array(env.sim.model.body_pos[visual], dtype=float)
            pos[:2] = center[:2]
            env.sim.model.body_pos[visual] = pos
        names = self.active(env)
        if len(names) == 1:
            self._place_one(env, rng, names[0])
        else:
            self._shift_group(env, rng, names)

    def _place_one(self, env, rng, name):
        """单件物品放到左箱的九宫格之一，再随机转一个朝向。"""
        bin1 = np.asarray(env.bin1_pos, dtype=float)
        slot = np.array(
            [
                float(rng.choice([-0.07, 0.0, 0.07])),
                float(rng.choice([-0.10, 0.0, 0.10])),
            ]
        )
        slot += rng.uniform(-0.015, 0.015, size=2)
        obj = env.objects[self._index(env, name)]
        set_free_xyaw(env, obj, bin1[:2] + slot, yaw=float(rng.uniform(0.0, 2.0 * np.pi)))

    def _shift_group(self, env, rng, names):
        """多件物品保持彼此间距，整组挪到左箱的另一处，避免重新摆放时叠在一起。"""
        bin1 = np.asarray(env.bin1_pos, dtype=float)
        x_lim, y_lim = 0.125, 0.165
        pts = []
        for name in names:
            obj = env.objects[self._index(env, name)]
            qpos = np.array(env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float)
            pts.append(qpos[:2])
        pts = np.stack(pts)
        desired = bin1[:2] + np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.07, 0.07)])
        shift = desired - pts.mean(axis=0)
        low = bin1[:2] - np.array([x_lim, y_lim])
        high = bin1[:2] + np.array([x_lim, y_lim])
        shift_low = low - pts.min(axis=0)
        shift_high = high - pts.max(axis=0)
        for axis in (0, 1):
            if shift_low[axis] <= shift_high[axis]:
                shift[axis] = np.clip(shift[axis], shift_low[axis], shift_high[axis])
            else:
                shift[axis] = 0.0
        for name, xy in zip(names, pts):
            set_free_xyaw(env, env.objects[self._index(env, name)], xy + shift)

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
            # 双落点中点的真值，只用于标注记录和评测误差统计；计划仍引用物品名。
            grasp = out[name].copy()
            grasp[2] += max(0.0, float(self.extent(env, name)[1][2]) - GRASP_DEPTH)
            out[name + "_grasp"] = grasp
            out[name + "_bin"] = np.array(env.target_bin_placements[self._index(env, name)], dtype=float)
        return out

    def landmarks(self, env):
        return list(self.points(env).values())

    def point_kind(self, name):
        """物品中心走桌面上物体的量法，格子底面走箱格的量法。"""
        return "bin" if name.endswith("_bin") else "object"

    def point_body(self, env, name):
        """遮挡判定：物品中心属于物品自己（命中它自己不算挡）；格子底面不豁免，箱壁挡住就是真被挡。"""
        if name.endswith("_bin"):
            return None
        return env.obj_body_id[name]

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

    def grip_points(self, env, name):
        """物品两侧的两指接触点，中点和原中心抓取目标一致。"""
        if name not in self.active(env):
            return None
        points = self.points(env)
        order = self.order(env)
        if name not in order:
            return None
        index = order.index(name)
        yaw = self.grasp_yaw(env, name, points, order[index + 1 :])
        angle = np.radians(0.0 if yaw is None else yaw)
        axis = np.array([-np.sin(angle), np.cos(angle), 0.0])
        center = np.array(points[name], dtype=float)
        low, high = self.extent(env, name)
        center[2] += max(0.0, float(high[2]) - GRASP_DEPTH)
        body = env.obj_body_id[name]
        rot = np.asarray(env.sim.data.body_xmat[body], dtype=float).reshape(3, 3)
        if name == "Can":
            radius = 0.25 * float(high[0] - low[0] + high[1] - low[1])
            pair = (center + axis * radius, center - axis * radius)
        else:
            pair = box_contacts(center, 0.5 * (high - low), rot, axis)
        camera = env.sim.data.cam_xpos[env.sim.model.camera_name2id(DEFAULT_CAMERA)]
        return order_landings_at(center, camera, pair)

    def plan_desc(self):
        """抓取写两指落点；箱格仍写单个中心点。单目被挡的落点按形状补出。"""
        if back_enabled():
            return (
                "抓物品的步骤里 target_1、target_2 是贴住物品两侧的两指落点，每个落点各给两台相机的像素："
                "target_1 在正面相机图上，target_1_back 在背面相机图上，target_2、target_2_back 同理；"
                "某台相机看不到那个落点时对应字段写 \"mask\"，不要猜它的像素。两落点的中点是抓取位置；"
                "放进箱格的步骤只写 target_1、target_1_back（格子中心）"
            )
        return (
            "抓物品的步骤里 target_1、target_2 是贴住物品两侧的两指落点，都在正面相机图上；"
            "另一个落点被物品挡住时也根据物品形状和朝向补出它的像素，不写 mask。两落点的中点是抓取位置；"
            "放进箱格的步骤只写 target_1（格子中心）"
        )

    def grasp_yaw(self, env, name, points, neighbors):
        """先验知识，不是逆运动学：只给出目标 yaw，关节角留给后面的控制器。
        两指合拢方向尽量和最近的邻居方向垂直，免得手指碰到旁边的物品。
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
        """先验知识：先拿离其他物品最远的那样，给后面的抓取留出空隙。"""
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
            label = OBJECT_CN[name]
            # 双落点中点已经包含 grasp 高度，above/down 只写相对这个中点的剩余位移。
            above = {
                "type": "move_above",
                "ref": name,
                "clearance": max(DEFAULT_CLEARANCE, top + 0.06) - grasp,
                "obj": name,
                "desc_hint": "移到{}上方".format(label),
            }
            if yaw is not None:
                above["yaw"] = yaw
            target = name + "_bin"
            plan += [
                above,
                {"type": "move_down", "ref": name, "offset_z": 0.0, "obj": name, "desc_hint": "降到{}处准备夹".format(label)},
                {"type": "close_gripper", "obj": name, "desc_hint": "合爪夹住{}".format(label)},
                {"type": "lift", "ref": name, "goal_z": carry_z, "obj": name, "desc_hint": "抬起{}过箱壁".format(label)},
                {"type": "move_above", "ref": target, "clearance": carry_z - bin_z, "obj": name, "desc_hint": "移到格子正上方"},
                {"type": "move_down", "ref": target, "offset_z": rest + grasp + DROP_HEIGHT, "obj": name, "desc_hint": "把{}放进格子".format(label)},
                {"type": "open_gripper", "obj": name, "desc_hint": "松爪放下{}".format(label)},
                {"type": "move_above", "ref": target, "clearance": carry_z - bin_z, "obj": name, "desc_hint": "向上抬开夹爪"},
            ]
        plan.append({"type": "stabilize", "desc_hint": "停在原地保持稳定"})
        return plan

    def step_desc(self, env, step):
        return step.get("desc_hint")

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
            "point_names": "物品名是物品中心，物品名_grasp 是两指落点中点，物品名_bin 是右边箱子里对应格子底面的中心",
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
