"""采集 robosuite Door：转开门把手，把门拉开。

门带锁。把手是一根绕门板法向转的横杆，要转过约 75 度锁舌才离开门框。
从上往下压做不到这么大的角度，而且两指沿门法向时手掌会撞上门板，
所以夹爪横着伸向把手（reach，两指上下合拢夹住横杆），合爪后：

1. 绕门锁转轴（pivot 是把手根部，axis 是门板法向）转，把手往上翻约 83 度；
2. 绕门的铰链（pivot 是铰链那根门框立柱，axis 竖直向上）转约 23 度，把门拉开。

turn 的 angle 写的是还要转的角度，按当前把手和门的角度算。往上翻而不是往下压，
是因为这台 Panda 横着抓之后，手腕最后一个关节往这个方向还有余量。
门打开（铰链角度超过 0.3 弧度）之后 subtasks 为空。

    python collect_door_mllm.py --episodes 50 --out data/mllm
"""

import argparse

import numpy as np

from collect_common import add_common_args, run_collection

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
