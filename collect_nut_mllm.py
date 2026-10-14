"""采集 robosuite NutAssembly 系列：把螺母套到对应的柱子上。

--env 可选 NutAssembly（两个都套）、NutAssemblySingle（每次随机一个）、
NutAssemblySquare、NutAssemblyRound（固定一个）。方形螺母套方柱，圆形螺母套圆柱。

螺母只有把手能抓。夹爪按螺母朝向转到两指横跨把手（move_above 带 yaw），
抓的位置比把手中心再往外 1.2 厘米，免得指头压到螺母的环上。
柱子 20 厘米高，抬起后螺母底部要高过柱顶。放的时候要对准柱子的是螺母中心，不是夹爪，
所以往柱子去的几步带 held，是螺母中心的像素；柱子的像素是柱顶中心。
往柱子去的那一步还带 yaw，路上把螺母转到把手朝机器人，柱子在桌子远端，这样手臂不用伸直。
螺母沿柱子往下套，到离桌面 3.5 厘米时松爪，再竖直抬开。

    python collect_nut_mllm.py --env NutAssemblySquare --episodes 50 --out data/mllm
"""

import argparse

import numpy as np

from collect_common import (
    add_common_args,
    body_yaw_deg,
    choose_yaws,
    is_grasping,
    run_collection,
    wrap_angle,
    wrist_yaw,
)

ENVS = ("NutAssembly", "NutAssemblySingle", "NutAssemblySquare", "NutAssemblyRound")
NUT_CN = {"SquareNut": "方形螺母", "RoundNut": "圆形螺母"}
PEG_CN = {"SquareNut": "方柱", "RoundNut": "圆柱"}
PEG_BODY = {"SquareNut": "peg1", "RoundNut": "peg2"}
PEG_HALF_HEIGHT = 0.10
GRASP_OUTWARD = 0.012
HOVER_CLEARANCE = 0.10
# 抬起后夹爪比柱顶高这么多，螺母底部离柱顶约 6 厘米。
CARRY_MARGIN = 0.07
# 移到柱子上方时螺母中心比柱顶高这么多。
PEG_CLEARANCE = 0.05
# 松爪时螺母中心离桌面的高度。
SEAT_HEIGHT = 0.035
RETREAT_CLEARANCE = 0.06
# 运输时把把手转向机器人，最多转这么多度，度。
MAX_HANDLE_TURN = 120.0
SINGLE_STYLES = (
    ("把{0}套到{1}上。", "把手里的{0}套到{1}上。", "{0}已经套好了。"),
    ("将{0}拿起来，套进{1}。", "将拿着的{0}套进{1}。", "{0}已经套在{1}上了。"),
    ("把桌上的{0}放到{1}上套好。", "把拿着的{0}放到{1}上套好。", "{0}已经放好了。"),
)
ALL_STYLES = (
    ("把方形螺母套到方柱上，圆形螺母套到圆柱上。", "先把手里的{0}套到{1}上，再套剩下的螺母。", "两个螺母都已经套好了。"),
    ("把两个螺母分别套到形状对应的柱子上。", "先把拿着的{0}套进{1}，再处理另一个螺母。", "两个螺母都套好了。"),
)


class NutTask:
    primitives = ("move_above", "move_down", "close_gripper", "lift", "open_gripper", "stabilize")
    max_rounds = 1
    settle_steps = 20

    def __init__(self, env_name):
        self.env_name = env_name
        self.task = env_name.lower()
        self.all_nuts = env_name == "NutAssembly"
        self.styles = ALL_STYLES if self.all_nuts else SINGLE_STYLES
        self.instruction_note = (
            "螺母还没抓住时，指令是把它套到对应的柱子上；抓住之后改成把手里的螺母套好；套好说明已经完成。"
            "两个都套时，只剩一个螺母就按单个螺母的说法。同一条轨迹里用同一套说法。"
        )

    def env_kwargs(self):
        return {}

    def _index(self, env, name):
        return env.nut_to_id["square" if name == "SquareNut" else "round"]

    def _nut(self, env, name):
        return env.nuts[self._index(env, name)]

    def active(self, env):
        if env.single_object_mode == 0:
            return [nut.name for nut in env.nuts]
        return [env.nuts[env.nut_id].name]

    def placed(self, env):
        env._check_success()
        return [name for name in self.active(env) if env.objects_on_pegs[self._index(env, name)]]

    def points(self, env):
        data, model = env.sim.data, env.sim.model
        out = {}
        for name in self.active(env):
            nut = self._nut(env, name)
            body = env.obj_body_id[name]
            center = np.array(data.body_xpos[body], dtype=float)
            mat = np.asarray(data.body_xmat[body], dtype=float).reshape(3, 3)
            handle_x = float(model.site_pos[model.site_name2id(nut.important_sites["handle"])][0])
            out[name] = center
            out[name + "_grasp"] = center + mat @ np.array([handle_x + GRASP_OUTWARD, 0.0, 0.0])
            peg = np.array(data.body_xpos[model.body_name2id(PEG_BODY[name])], dtype=float)
            out[name + "_peg"] = peg + np.array([0.0, 0.0, PEG_HALF_HEIGHT])
        return out

    def landmarks(self, env):
        return list(self.points(env).values())

    def record_points(self, env):
        return list(self.points(env).keys())

    def build_plan(self, env, rng):
        points = self.points(env)
        table_z = float(env.table_offset[2])
        done = set(self.placed(env))
        names = [name for name in self.active(env) if name not in done]
        yaws = choose_yaws(wrist_yaw(env), [self.grasp_turn(env, name, points) for name in names])
        plan = []
        for name, (yaw, carry_yaw) in zip(names, yaws):
            grasp, peg = name + "_grasp", name + "_peg"
            peg_z = float(points[peg][2])
            plan += [
                {"type": "move_above", "ref": grasp, "clearance": HOVER_CLEARANCE, "yaw": yaw, "obj": name},
                {"type": "move_down", "ref": grasp, "offset_z": 0.0, "obj": name},
                {"type": "close_gripper", "obj": name},
                {"type": "lift", "ref": grasp, "goal_z": peg_z + CARRY_MARGIN, "obj": name},
                {
                    "type": "move_above",
                    "ref": peg,
                    "clearance": PEG_CLEARANCE,
                    "held": name,
                    "yaw": carry_yaw,
                    "obj": name,
                },
                {
                    "type": "move_down",
                    "ref": peg,
                    "offset_z": table_z + SEAT_HEIGHT - peg_z,
                    "held": name,
                    "obj": name,
                },
                {"type": "open_gripper", "obj": name},
                {"type": "move_above", "ref": peg, "clearance": RETREAT_CLEARANCE, "held": name, "obj": name},
            ]
        plan.append({"type": "stabilize"})
        return plan

    def grasp_turn(self, env, name, points):
        """抓取 yaw（两指横跨把手），以及抓住后要转多少度。柱子在桌子远端，手臂几乎伸直，
        运输途中把把手转到朝机器人（世界 -x），夹爪就不用伸那么远；把手朝左右时肘关节会被拉到伸直的极限。
        要转超过 120 度时，改转到朝左或朝右里近的那个，免得转动接近半圈。
        方柱和世界坐标轴对齐，这几个朝向下方螺母都正好对齐。"""
        handle = points[name + "_grasp"][:2] - points[name][:2]
        current = np.degrees(np.arctan2(handle[1], handle[0]))
        turn = wrap_angle(180.0 - current, 360.0)
        if abs(turn) > MAX_HANDLE_TURN:
            turn = min((wrap_angle(goal - current, 360.0) for goal in (90.0, -90.0)), key=abs)
        return body_yaw_deg(env, env.obj_body_id[name]), turn

    def holding(self, env, plan, index):
        if not 0 <= index < len(plan) or "obj" not in plan[index]:
            return None, False
        name = plan[index]["obj"]
        kinds = [(k, step["type"]) for k, step in enumerate(plan) if step.get("obj") == name]
        closed = [k for k, kind in kinds if kind == "close_gripper"]
        opened = [k for k, kind in kinds if kind == "open_gripper"]
        between = bool(closed) and closed[0] < index <= opened[0]
        return name, between and is_grasping(env, self._nut(env, name))

    def instruction(self, env, plan, index, style, done):
        name, held = self.holding(env, plan, index)
        left = [nut for nut in self.active(env) if nut not in self.placed(env)]
        if self.all_nuts:
            if done:
                return style[2]
            if held and len(left) > 1:
                return style[1].format(NUT_CN[name], PEG_CN[name])
            if not held and len(left) > 1:
                return style[0]
            single = SINGLE_STYLES[0]
            nut = name if held else left[0]
            return single[1 if held else 0].format(NUT_CN[nut], PEG_CN[nut])
        nut = self.active(env)[0]
        if done:
            return style[2].format(NUT_CN[nut], PEG_CN[nut])
        return style[1 if held else 0].format(NUT_CN[nut], PEG_CN[nut])

    def success(self, env):
        return bool(env._check_success())

    def start_pose(self, env, rng):
        start = np.array([rng.uniform(-0.20, 0.05), rng.uniform(-0.20, 0.20), rng.uniform(0.95, 1.10)])
        return start, float(rng.uniform(-np.pi / 2.0, np.pi / 2.0)), 1.10

    def record_extra(self, env):
        return {"placed": self.placed(env)}

    def episode_info(self, env):
        return {"nuts": self.active(env)}

    def meta_extra(self):
        return {
            "nuts_cn": NUT_CN,
            "pegs_cn": PEG_CN,
            "point_names": "螺母名是螺母中心，_grasp 是把手上的抓取点，_peg 是对应柱子的柱顶中心",
        }

    def describe(self, env):
        return "placed={}".format("+".join(self.placed(env)) or "-")


TASKS = {name: (lambda name=name: NutTask(name)) for name in ENVS}


def main():
    parser = argparse.ArgumentParser(description="采集 NutAssembly / Panda 的 MLLM 图文步骤")
    parser.add_argument("--env", choices=ENVS, default="NutAssemblySquare")
    add_common_args(parser, "data/nut_mllm")
    args = parser.parse_args()
    run_collection(NutTask(args.env), args)


if __name__ == "__main__":
    main()
