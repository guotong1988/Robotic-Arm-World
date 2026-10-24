"""夹爪两指落点和逆运动学执行。

落点是夹爪合拢在物体上时两块指腹贴住物体表面的那两点。两点的中点是抓取中心，
连线是两指合拢方向，高度差抵消掉指腹相对末端 site 的偏移就是末端要到的位置。
所以模型只要给出两个落点，抓取位姿（位置和绕竖直轴的 yaw）就定了。

执行时末端位姿先用阻尼最小二乘逆运动学解成 7 个关节角，再用关节位置控制器
（绝对角度）跟踪。直线路径按控制步切成小段，每一段都在上一段的解附近重新解，
零空间把多余的自由度拉回初始姿态，手腕不会越转越拧。

两指对称，yaw 和 yaw + 180 度是同一种抓法。grasp_candidates 两种都给，
choose_grasp 用逆运动学挑能解出来、离关节限位远、关节转动少的那个。
"""

import mujoco
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.transform_utils import mat2quat, quat2axisangle, quat2mat, quat_slerp

# 关节位置控制器的刚度。目标每步只挪 1 厘米左右，跟踪滞后一两个控制步。
JOINT_KP = 150
# 逆运动学：阻尼、每次迭代的步长上限、零空间回拉增益、离关节限位留的余量（弧度）。
IK_DAMPING = 0.05
IK_STEP = 0.2
IK_NULLSPACE_GAIN = 0.1
IK_LIMIT_MARGIN = 0.03
IK_ITERS = 200
IK_POS_TOL = 5e-4
IK_ORI_TOL = 5e-3
# 末端沿直线走时每个控制步最多挪这么多：米、弧度。
PATH_SPEED = 0.01
PATH_TURN = 0.06
# 走到终点后用实测误差修正目标：每步修正比例、最多修正多少米。
BIAS_GAIN = 0.3
BIAS_LIMIT = 0.03
# 两指最大张开约 8 厘米，落点间距超过这个值说明模型给错了。
MAX_GRIP_WIDTH = 0.085
MIN_GRIP_WIDTH = 0.005
# 竖直朝下、两指沿世界 y 合拢的夹爪朝向，和 task_plan.TOPDOWN 一致。各列是末端 x、y、z 轴，x 是两指合拢方向。
TOPDOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# 采集和评测都用这个朝向：两指沿世界 x 合拢，一个落点朝 frontview，另一个朝对面的 backview。
GRIP_YAW = 90.0


def joint_position_config():
    """Panda 的关节位置控制器，输入是绝对关节角。夹爪仍是 GRIP，-1 张开、+1 合拢。"""
    config = load_composite_controller_config(robot="Panda")
    arm = config["body_parts"]["right"]
    gripper = arm.get("gripper", {"type": "GRIP"})
    config["body_parts"]["right"] = {
        "type": "JOINT_POSITION",
        "input_type": "absolute",
        "input_max": 2.0 * np.pi,
        "input_min": -2.0 * np.pi,
        "output_max": 0.05,
        "output_min": -0.05,
        "kp": JOINT_KP,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": gripper,
    }
    return config


def rotation_z(angle_rad):
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def topdown_ori(yaw_deg):
    return rotation_z(np.radians(float(yaw_deg))) @ TOPDOWN


def box_contacts(center, half, rot, direction):
    """方块表面沿 direction 的两个接触点。

    half 是物体自身坐标的半尺寸，rot 的列是自身轴在世界系的方向，direction 是世界系方向。
    返回 (direction 一侧的点, 反方向的点)。

    交点距离是 1/max(|local_i| / half_i)：射线 x = center + t*direction 与盒面
    max(|x_local_i| / half_i) = 1 的交点。不能用 sum——方块旋转后 sum 给出的点
    缩在盒内部，不在表面上，做可见性判断时会被方块自己挡住。
    """
    center = np.asarray(center, dtype=float).reshape(3)
    half = np.asarray(half, dtype=float).reshape(3)
    rot = np.asarray(rot, dtype=float).reshape(3, 3)
    direction = np.asarray(direction, dtype=float).reshape(3)
    direction = direction / np.linalg.norm(direction)
    local = rot.T @ direction
    denom = float(np.max(np.abs(local) / np.maximum(half, 1e-6)))
    if denom < 1e-8:
        raise ValueError("方块尺寸无效")
    dist = 1.0 / denom
    return center + direction * dist, center - direction * dist


def grasp_eef(front, back, offset, yaw=GRIP_YAW):
    """两个落点的中点，再扣掉指腹相对末端的偏移，就是夹爪要到的位置。"""
    mid = 0.5 * (np.asarray(front, dtype=float) + np.asarray(back, dtype=float))
    ori = topdown_ori(yaw)
    return mid - ori @ np.asarray(offset, dtype=float), ori


def closing_yaw(axis):
    """两指合拢方向（水平）对应的 yaw，度。topdown_ori(yaw) 的 x 轴是 (-sin yaw, cos yaw, 0)。"""
    axis = np.asarray(axis, dtype=float)
    return float(np.degrees(np.arctan2(-axis[0], axis[1])))


def wrap_deg(deg):
    return (float(deg) + 180.0) % 360.0 - 180.0


def eef_site(env):
    return env.robots[0].eef_site_id["right"]


def eef_pose(env):
    site = eef_site(env)
    pos = np.array(env.sim.data.site_xpos[site], dtype=float)
    mat = np.array(env.sim.data.site_xmat[site], dtype=float).reshape(3, 3)
    return pos, mat


def pad_geom_ids(env):
    gripper = env.robots[0].gripper["right"]
    names = [gripper.important_geoms[key][0] for key in ("left_fingerpad", "right_fingerpad")]
    return [env.sim.model.geom_name2id(name) for name in names]


def pad_positions(env):
    return [np.array(env.sim.data.geom_xpos[gid], dtype=float) for gid in pad_geom_ids(env)]


def pad_offset(env):
    """两块指腹中点相对末端 site 的位置，末端坐标系。手指对称开合，中点和开度无关。"""
    pos, mat = eef_pose(env)
    mid = 0.5 * (pad_positions(env)[0] + pad_positions(env)[1])
    offset = mat.T @ (mid - pos)
    offset[0] = 0.0
    return offset


def landing_points(center, axis, half_width):
    """抓取中心沿合拢方向两侧各 half_width 的两点。"""
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return [center - axis * float(half_width), center + axis * float(half_width)]


def grasp_candidates(points, offset):
    """两个落点换成末端位姿。返回 [(末端位置, 朝向, yaw)]，yaw 和 yaw + 180 两种。"""
    p1, p2 = [np.asarray(point, dtype=float).reshape(3) for point in points]
    axis = p2 - p1
    axis[2] = 0.0
    width = float(np.linalg.norm(axis))
    if not np.isfinite(width) or width < MIN_GRIP_WIDTH or width > MAX_GRIP_WIDTH:
        raise ValueError("两个落点的水平间距 {:.3f} 米，夹爪夹不了".format(width))
    mid = 0.5 * (p1 + p2)
    yaw = closing_yaw(axis / width)
    out = []
    for candidate in (yaw, yaw + 180.0):
        ori = topdown_ori(wrap_deg(candidate))
        out.append((mid - ori @ np.asarray(offset, dtype=float), ori, wrap_deg(candidate)))
    return out


def orientation_error(target, current):
    """当前朝向转到目标朝向的轴角，世界系。"""
    return quat2axisangle(mat2quat(np.asarray(target) @ np.asarray(current).T))


class PandaIK:
    """阻尼最小二乘逆运动学。在一份独立的 MjData 上迭代，不碰仿真本身的状态。"""

    def __init__(self, env):
        robot = env.robots[0]
        self.env = env
        self.model = env.sim.model._model
        self.data = mujoco.MjData(self.model)
        self.qpos_idx = np.array(robot._ref_joint_pos_indexes)
        self.dof_idx = np.array(robot._ref_joint_vel_indexes)
        self.site = eef_site(env)
        joint_ids = [self.model.joint(name).id for name in robot.robot_joints]
        ranges = np.array(self.model.jnt_range[joint_ids], dtype=float)
        self.low = ranges[:, 0] + IK_LIMIT_MARGIN
        self.high = ranges[:, 1] - IK_LIMIT_MARGIN
        self.rest = np.array(robot.init_qpos, dtype=float)

    def current(self):
        return np.array(self.env.sim.data.qpos[self.qpos_idx], dtype=float)

    def fk(self, q):
        self.data.qpos[:] = self.env.sim.data.qpos
        self.data.qpos[self.qpos_idx] = q
        mujoco.mj_kinematics(self.model, self.data)
        return np.array(self.data.site_xpos[self.site]), np.array(self.data.site_xmat[self.site]).reshape(3, 3)

    def solve(self, pos, ori, q_init=None):
        """返回 (关节角, 位置误差米, 朝向误差弧度)。"""
        q = self.current() if q_init is None else np.array(q_init, dtype=float)
        pos = np.asarray(pos, dtype=float)
        ori = np.asarray(ori, dtype=float)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        pos_err = ori_err = np.inf
        for _ in range(IK_ITERS):
            cur_pos, cur_ori = self.fk(q)
            e_pos = pos - cur_pos
            e_ori = orientation_error(ori, cur_ori)
            pos_err, ori_err = float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_ori))
            if pos_err < IK_POS_TOL and ori_err < IK_ORI_TOL:
                break
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
            jac = np.vstack([jacp[:, self.dof_idx], jacr[:, self.dof_idx]])
            twist = np.concatenate([e_pos, e_ori])
            jjt = jac @ jac.T + (IK_DAMPING**2) * np.eye(6)
            pinv = jac.T @ np.linalg.inv(jjt)
            dq = pinv @ twist
            nullspace = np.eye(len(q)) - pinv @ jac
            dq = dq + nullspace @ (IK_NULLSPACE_GAIN * (self.rest - q))
            norm = float(np.linalg.norm(dq))
            if norm > IK_STEP:
                dq = dq * (IK_STEP / norm)
            q = np.clip(q + dq, self.low, self.high)
        return q, pos_err, ori_err

    def limit_margin(self, q):
        """离最近的关节限位还有多远，弧度。"""
        q = np.asarray(q, dtype=float)
        return float(np.min(np.minimum(q - self.low, self.high - q)))


def choose_grasp(ik, candidates, q_init=None):
    """yaw 和 yaw + 180 里挑一个：先看逆运动学能不能解到，再看离关节限位多远，最后看关节转动多少。"""
    q0 = ik.current() if q_init is None else np.asarray(q_init, dtype=float)
    best = None
    for pos, ori, yaw in candidates:
        q, pos_err, ori_err = ik.solve(pos, ori, q0)
        solved = pos_err < 0.005 and ori_err < 0.05
        key = (not solved, ik.limit_margin(q) < 0.15, float(np.linalg.norm(q - q0)))
        if best is None or key < best[0]:
            best = (key, (pos, ori, yaw, q))
    return best[1]


class IKArm:
    """末端位姿走逆运动学，再用关节位置控制器执行。"""

    def __init__(self, env):
        self.env = env
        self.ik = PandaIK(env)
        self.q_cmd = None

    def reset(self):
        self.ik = PandaIK(self.env)
        self.q_cmd = self.ik.current()

    def pose(self):
        return eef_pose(self.env)

    def _send(self, q, grip):
        robot = self.env.robots[0]
        action = robot.create_action_vector({"right": np.asarray(q, dtype=float), "right_gripper": np.array([grip])})
        self.q_cmd = np.array(q, dtype=float)
        obs, _, _, _ = self.env.step(action)
        return obs

    def move(self, target_pos, target_ori, grip, tol, max_steps, on_step, max_speed=None):
        """末端沿直线走到目标位姿，朝向按球面插值。返回 (obs, 到没到, 位置误差米)。"""
        if self.q_cmd is None:
            self.reset()
        speed = PATH_SPEED if max_speed is None else min(PATH_SPEED, float(max_speed))
        start_pos, start_ori = self.pose()
        target_pos = np.asarray(target_pos, dtype=float)
        target_ori = np.asarray(target_ori, dtype=float)
        dist = float(np.linalg.norm(target_pos - start_pos))
        angle = float(np.linalg.norm(orientation_error(target_ori, start_ori)))
        pieces = max(1, int(np.ceil(max(dist / speed, angle / PATH_TURN))))
        q0, q1 = mat2quat(start_ori), mat2quat(target_ori)
        obs = None
        err = dist
        q = self.q_cmd
        bias = np.zeros(3)
        for k in range(1, int(max_steps) + 1):
            frac = min(1.0, k / float(pieces))
            way_pos = start_pos + frac * (target_pos - start_pos)
            way_ori = quat2mat(quat_slerp(q0, q1, frac))
            q, _, _ = self.ik.solve(way_pos + bias, way_ori, q)
            obs = self._send(q, grip)
            on_step(obs)
            cur_pos, _ = self.pose()
            err = float(np.linalg.norm(cur_pos - target_pos))
            if frac >= 1.0:
                if err < tol:
                    return obs, True, err
                # 关节 PD 带着负载有稳态误差，到终点后按实测误差慢慢把目标往反方向推
                bias = np.clip(bias + BIAS_GAIN * (target_pos - cur_pos), -BIAS_LIMIT, BIAS_LIMIT)
        return obs, False, err

    def hold(self, grip, steps, on_step):
        """关节目标不动，只改夹爪。"""
        if self.q_cmd is None:
            self.reset()
        obs = None
        for _ in range(int(steps)):
            obs = self._send(self.q_cmd, grip)
            on_step(obs)
        return obs
