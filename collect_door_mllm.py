"""采集 robosuite Door：转开门把手，把门拉开。

门带锁。把手是一根绕门板法向转的横杆，要转过约 75 度锁舌才离开门框。
从上往下压做不到这么大的角度，而且两指沿门法向时手掌会撞上门板，
所以夹爪横着伸向把手（reach，两指上下合拢夹住横杆），合爪后：

1. 绕门锁转轴（pivot 是把手根部，axis 是门板法向）转，把手往上翻约 83 度；
2. 绕门的铰链（pivot 是铰链那根门框立柱，axis 竖直向上）转约 23 度，把门拉开。

turn 的 angle 写的是还要转的角度，按当前把手和门的角度算。往上翻而不是往下压，
是因为这台 Panda 横着抓之后，手腕最后一个关节往这个方向还有余量。
门打开（铰链角度超过 0.3 弧度）之后 subtasks 为空。

输入例子。每台相机一张 RGB，默认 frontview 加 backview，对应样本的 image、image_back 字段；
指令开头的 <image> 每行一个，和图片一一对应，多张图时占位后面注明每张来自哪台相机；话术后面带 answer 各字段的说明：

    <image>
    <image>
    第一张图是正面相机（frontview），第二张图是背面相机（backview）。把门打开。回答的 JSON 中，target_1 是要对准的点在正面相机图上的像素，target_1_back 是同一点在背面相机图上的像素；某台相机看不到那个点时对应字段写 "mask"，不要猜它的像素。

单目只开 frontview，一个占位后面接话术和字段说明：

    <image>
    把门打开。回答的 JSON 中，target_1 是要对准的点在正面相机图上的像素。

输出例子（把手还没转开时；每个 subtask 的 desc 说明这一步做什么。
双目时每个空间字段各给两台相机的像素，某台相机看不到那个点时对应字段写 "mask"）：

    {"subtasks": [{"type": "reach", "desc": "伸向目标点", "target_1": "[390 220]", "target_1_back": "[95 205]", "approach": [0.0, -1.0, 0.0], "standoff": 0.1, "target_kind": "door_handle"}, {"type": "reach", "desc": "伸向目标点", "target_1": "[390 220]", "target_1_back": "[95 205]", "approach": [0.0, -1.0, 0.0], "standoff": 0.0, "target_kind": "door_handle"}, {"type": "close_gripper", "desc": "合爪夹住"}, {"type": "turn", "desc": "绕转轴转动", "pivot": "[400 215]", "pivot_back": "[90 200]", "axis": [0.0, -1.0, 0.0], "angle": -83, "pivot_kind": "door_axis"}, {"type": "turn", "desc": "绕转轴转动", "pivot": "[430 240]", "pivot_back": "[70 220]", "axis": [0.0, 0.0, 1.0], "angle": 23, "pivot_kind": "door_axis"}, {"type": "stabilize", "desc": "停在原地保持稳定", "duration": 0.5}]}

每条成功轨迹只存开局那一帧。门在桌面上换位置，朝向在默认朝向附近偏一个角度。
门锁和门扇仍从关上的状态开始。中途画面不存。

例如凑满 50 条成功开门再停，图片和样本写到 data/mllm：

python collect_door_mllm.py --episodes 50 --out data/mllm
"""

import argparse

import numpy as np

from collect_common import add_common_args, run_collection

# 先验知识：锁舌转到约 -1.45 弧度才开，门拉开到 0.4 弧度；抓在把手靠外处，沿门板法向横着伸。不解关节角。
LATCH_GOAL = -1.45
HINGE_GOAL = 0.40
# 抓取点在把手根部到末端的这个比例处。
GRASP_FRACTION = 0.85
STANDOFF = 0.10
UNLATCHED = 1.2
STYLES = (
    ("把门打开。", "门锁已经转开，把门拉开。", "门已经打开了。"),
    ("转动门把手，把门拉开。", "接着把门拉开。", "门已经开了。"),
    ("打开这扇门。", "把手已经转到位，把门拉开。", "门已经拉开了。"),
)


class DoorTask:
    env_name = "Door"
    task = "door"
    primitives = ("reach", "close_gripper", "turn", "stabilize")
    max_rounds = 1
    settle_steps = 5
    styles = STYLES
    instruction_note = "把手还没转开时，指令是把门打开；把手转开之后改成把门拉开；门打开后说明已经完成。同一条轨迹里用同一套说法。"

    def env_kwargs(self):
        return {"use_latch": True}

    def prepare_reset(self, env, rng):
        """门的采样范围按这一条收窄到一个点，位置和朝向仍落在桌子上、把手朝向机器人一侧。"""
        sampler = env.placement_initializer
        x = float(rng.uniform(0.02, 0.12))
        y = float(rng.uniform(-0.04, 0.04))
        yaw = -np.pi / 2.0 + float(rng.uniform(-0.30, 0.15))
        sampler.x_range = (x, x)
        sampler.y_range = (y, y)
        sampler.rotation = yaw

    def _latch(self, env):
        joint = env.sim.model.joint_name2id(env.door.joints[1])
        return np.array(env.sim.data.xanchor[joint], dtype=float), np.array(env.sim.data.xaxis[joint], dtype=float)

    def points(self, env):
        pivot, _ = self._latch(env)
        handle = np.array(env._handle_xpos, dtype=float)
        hinge = env.sim.model.joint_name2id(env.door.joints[0])
        anchor = np.array(env.sim.data.xanchor[hinge], dtype=float)
        return {
            "handle": pivot + GRASP_FRACTION * (handle - pivot),
            "latch": pivot,
            "hinge": np.array([anchor[0], anchor[1], handle[2]]),
        }

    def landmarks(self, env):
        return list(self.points(env).values())

    def point_kind(self, name):
        """把手上的抓取点和门板中面上的转轴、铰链是两种量法。"""
        return "door_handle" if name == "handle" else "door_axis"

    def point_body(self, env, name):
        """遮挡判定：把手上的抓取点是表面点，不豁免；转轴、铰链贴着各自所属的 body，命中自身不算挡。"""
        if name == "handle":
            return None
        joints = {"latch": 1, "hinge": 0}
        joint = env.sim.model.joint_name2id(env.door.joints[joints[name]])
        return int(env.sim.model.jnt_bodyid[joint])

    def record_points(self, env):
        return list(self.points(env).keys())

    def joint_angle(self, env, joint):
        addr = env.handle_qpos_addr if joint == "latch" else env.hinge_qpos_addr
        return float(env.sim.data.qpos[addr])

    def build_plan(self, env, rng):
        _, axis = self._latch(env)
        axis = axis / np.linalg.norm(axis)
        return [
            {"type": "reach", "ref": "handle", "approach": axis, "standoff": STANDOFF},
            {"type": "reach", "ref": "handle", "approach": axis, "standoff": 0.0},
            {"type": "close_gripper"},
            {"type": "turn", "pivot": "latch", "axis": axis, "joint": "latch", "goal": LATCH_GOAL},
            {"type": "turn", "pivot": "hinge", "axis": np.array([0.0, 0.0, 1.0]), "joint": "hinge", "goal": HINGE_GOAL},
            {"type": "stabilize"},
        ]

    def turn_remaining(self, env, step):
        return float(np.degrees(float(step["goal"]) - self.joint_angle(env, step["joint"])))

    def instruction(self, env, plan, index, style, done):
        if done:
            return style[2]
        return style[1] if abs(self.joint_angle(env, "latch")) >= UNLATCHED else style[0]

    def success(self, env):
        return bool(env._check_success())

    def start_pose(self, env, rng):
        # 离机器人太近或太低时，夹爪从朝下转成水平会把手腕拧到关节极限。
        start = np.array([rng.uniform(-0.15, 0.0), rng.uniform(-0.12, 0.12), rng.uniform(1.05, 1.2)])
        return start, float(rng.uniform(-0.4, 0.4)), 1.15

    def record_extra(self, env):
        return {
            "latch_qpos": round(self.joint_angle(env, "latch"), 4),
            "hinge_qpos": round(self.joint_angle(env, "hinge"), 4),
        }

    def episode_info(self, env):
        return {"hinge_qpos": round(self.joint_angle(env, "hinge"), 4)}

    def meta_extra(self):
        return {
            "latch_goal_rad": LATCH_GOAL,
            "hinge_goal_rad": HINGE_GOAL,
            "point_names": "handle 是把手上的抓取点，latch 是把手转轴（根部），hinge 是门铰链所在的门框立柱，取把手高度",
        }

    def describe(self, env):
        return "latch={:.2f} hinge={:.2f}".format(self.joint_angle(env, "latch"), self.joint_angle(env, "hinge"))


TASKS = {"Door": DoorTask}


def main():
    parser = argparse.ArgumentParser(description="采集 Door / Panda 的 MLLM 图文步骤")
    add_common_args(parser, "data/door_mllm")
    args = parser.parse_args()
    run_collection(DoorTask(), args)


if __name__ == "__main__":
    main()
