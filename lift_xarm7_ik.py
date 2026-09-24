"""Lift 任务：用差分逆运动学让 XArm7 抓起方块。

robosuite 自带的 IK_POSE 只支持 Baxter / Sawyer / Panda，不支持 XArm7。
这里直接用夹爪 grip site 的雅可比做阻尼最小二乘 IK，算出关节角，
再用 JOINT_POSITION（绝对角度）去跟踪。
"""

import argparse
import time

import numpy as np
import robosuite as suite

MAX_FR = 25


def top_down_orientation(yaw=0.0):
    """夹爪局部 z 朝下，手指开合沿水平方向，yaw 绕世界 z 旋转。"""
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def orientation_error(target_mat, current_mat):
    """当前姿态转到目标姿态的轴角误差，世界系，和 MuJoCo 旋转雅可比一致。"""
    return 0.5 * (
        np.cross(current_mat[:, 0], target_mat[:, 0])
        + np.cross(current_mat[:, 1], target_mat[:, 1])
        + np.cross(current_mat[:, 2], target_mat[:, 2])
    )


def damped_least_squares(jacobian, twist, damping):
    eye = np.eye(jacobian.shape[0])
    return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + (damping**2) * eye, twist)


def joint_position_config():
    """绝对关节角。IK 解出的 q 可以直接作为动作，不会被缩放到 [-1, 1]。"""
    return {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "JOINT_POSITION",
                "input_type": "absolute",
                "impedance_mode": "fixed",
                "kp": 80,
                "damping_ratio": 1,
                "input_max": 1,
                "input_min": -1,
                "output_max": 0.05,
                "output_min": -0.05,
                "kp_limits": [0, 300],
                "damping_ratio_limits": [0, 10],
                "qpos_limits": None,
                "interpolation": None,
                "ramp_ratio": 0.2,
                "gripper": {"type": "GRIP"},
            }
        },
    }


class DifferentialIK:
    def __init__(self, arm, rest_qpos, damping=0.05, nullspace_gain=0.15):
        self.arm = arm
        self.rest_qpos = np.array(rest_qpos, dtype=float)
        self.damping = damping
        self.nullspace_gain = nullspace_gain
        limits = arm.sim.model.jnt_range[arm.joint_index]
        self.q_low = limits[:, 0]
        self.q_high = limits[:, 1]

    def solve(self, target_pos, target_mat, pos_step=0.04, ori_step=0.25):
        self.arm.update(force=True)
        pos_err = target_pos - self.arm.ref_pos
        ori_err = orientation_error(target_mat, self.arm.ref_ori_mat)

        pos_norm = np.linalg.norm(pos_err)
        if pos_norm > pos_step:
            pos_err = pos_err * (pos_step / pos_norm)
        ori_norm = np.linalg.norm(ori_err)
        if ori_norm > ori_step:
            ori_err = ori_err * (ori_step / ori_norm)

        twist = np.concatenate([pos_err, ori_err])
        jacobian = np.array(self.arm.J_full)
        dq = damped_least_squares(jacobian, twist, self.damping)

        # 零空间把多余自由度拉回初始姿态，避免腕部拧死
        jjt = jacobian @ jacobian.T + (self.damping**2) * np.eye(6)
        j_pinv = jacobian.T @ np.linalg.solve(jjt, np.eye(6))
        nullspace = np.eye(jacobian.shape[1]) - j_pinv @ jacobian
        q_err = np.clip(self.rest_qpos - self.arm.joint_pos, -0.4, 0.4)
        dq = dq + nullspace @ (self.nullspace_gain * q_err)
        dq = np.clip(dq, -0.12, 0.12)

        q_cmd = np.clip(self.arm.joint_pos + dq, self.q_low + 0.02, self.q_high - 0.02)
        pose_err = (pos_norm, ori_norm)
        return q_cmd, pose_err


def make_env(render):
    env = suite.make(
        env_name="Lift",
        robots="XArm7",
        controller_configs=joint_position_config(),
        initialization_noise=None,
        has_renderer=render,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=2000,
    )
    env.reset()
    if render:
        env.viewer.set_camera(camera_id=0)
    return env


def cube_position(env):
    return np.array(env.sim.data.body_xpos[env.cube_body_id])


def fingers_touching_cube(env):
    """XArm7Gripper 登记的指腹名字和 XML 不一致，_check_grasp 会一直返回 False。"""
    left = right = False
    for i in range(env.sim.data.ncon):
        contact = env.sim.data.contact[i]
        names = (
            env.sim.model.geom_id2name(contact.geom1) or "",
            env.sim.model.geom_id2name(contact.geom2) or "",
        )
        if not any("cube" in name for name in names):
            continue
        if any("left_finger_pad" in name for name in names):
            left = True
        if any("right_finger_pad" in name for name in names):
            right = True
    return left and right


def send_action(env, q_cmd, grip):
    robot = env.robots[0]
    action = robot.composite_controller.create_action_vector(
        {
            "right": q_cmd,
            "right_gripper": np.array([grip]),
        }
    )
    obs, reward, done, info = env.step(action)
    if env.has_renderer:
        env.render()
        elapsed = time.time() - send_action.t0
        send_action.t0 = time.time()
        # render() 本身不记账，用上一帧起点限帧
        delay = 1.0 / MAX_FR - elapsed
        if delay > 0:
            time.sleep(delay)
    return obs, reward, done, info


send_action.t0 = time.time()


def move_until(env, ik, target_pos, target_mat, grip, pos_tol, ori_tol, max_steps):
    last_err = None
    for _ in range(max_steps):
        q_cmd, (pos_err, ori_err) = ik.solve(target_pos, target_mat)
        last_err = (pos_err, ori_err)
        send_action(env, q_cmd, grip)
        if pos_err < pos_tol and ori_err < ori_tol:
            return True, last_err
    return False, last_err


def hold(env, ik, target_pos, target_mat, grip, steps):
    for _ in range(steps):
        q_cmd, _ = ik.solve(target_pos, target_mat)
        send_action(env, q_cmd, grip)


def grasp_cube(env):
    robot = env.robots[0]
    arm = robot.part_controllers["right"]
    arm.update(force=True)
    rest_qpos = arm.joint_pos.copy()
    ik = DifferentialIK(arm, rest_qpos)

    cube = cube_position(env)
    target_mat = top_down_orientation(yaw=0.0)
    print("cube position: {}".format(np.array2string(cube, precision=4)))
    print("eef position:  {}".format(np.array2string(arm.ref_pos, precision=4)))

    # 先张开夹爪，同时把末端抬到方块正上方，避免横着扫过方块
    hover = cube + np.array([0.0, 0.0, 0.16])
    reached, err = move_until(env, ik, hover, target_mat, grip=-1.0, pos_tol=0.015, ori_tol=0.15, max_steps=250)
    print("hover reached={} pos_err={:.4f} ori_err={:.4f}".format(reached, err[0], err[1]))

    # 竖直下降。grip site 大约在指腹中间，对准方块中心再略高一点，避免指尖戳桌子
    grasp = np.array([cube[0], cube[1], cube[2] + 0.005])
    reached, err = move_until(env, ik, grasp, target_mat, grip=-1.0, pos_tol=0.012, ori_tol=0.2, max_steps=200)
    print("grasp reached={} pos_err={:.4f} ori_err={:.4f}".format(reached, err[0], err[1]))
    print("eef at grasp: {}".format(np.array2string(arm.ref_pos, precision=4)))

    # +1 是闭合。XArm7Gripper 每个控制步只走 0.2，需要多停几步才能合上
    hold(env, ik, grasp, target_mat, grip=1.0, steps=40)
    grasped = fingers_touching_cube(env)
    print("fingers closed, grasped={}".format(grasped))

    lift = grasp + np.array([0.0, 0.0, 0.18])
    reached, err = move_until(env, ik, lift, target_mat, grip=1.0, pos_tol=0.02, ori_tol=0.3, max_steps=200)
    if env.has_renderer:
        hold(env, ik, lift, target_mat, grip=1.0, steps=40)
    success = bool(env._check_success()) and fingers_touching_cube(env)
    print(
        "lift reached={} pos_err={:.4f} success={} cube_z={:.4f}".format(
            reached, err[0], success, cube_position(env)[2]
        )
    )
    return success


def main():
    parser = argparse.ArgumentParser(description="XArm7 differential-IK grasp for robosuite Lift")
    parser.add_argument("--headless", action="store_true", help="不打开 MuJoCo 窗口，只打印结果")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    env = make_env(render=not args.headless)
    try:
        success = grasp_cube(env)
    finally:
        env.close()
    print("done, success={}".format(success))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
