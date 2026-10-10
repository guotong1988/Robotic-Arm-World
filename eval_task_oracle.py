"""不加载模型，执行初始画面生成的步骤，确认 PickPlace、NutAssembly、Door、Wipe 的计划本身能做成。

和采集是同一份标注、同一个执行器，只是不渲染图像、不写盘。
像素在提问这一帧反投影，之后机械臂只按世界坐标走，不再看图像。

    python eval_task_oracle.py --env PickPlaceCan --episodes 20
"""

import argparse

import gl_backend

gl_backend.configure()

import numpy as np

from collect_common import make_env, run_episode
from collect_door_mllm import TASKS as DOOR_TASKS
from collect_lift_mllm import world_to_pixel_matrix
from collect_nut_mllm import TASKS as NUT_TASKS
from collect_pickplace_mllm import TASKS as PICKPLACE_TASKS
from collect_wipe_mllm import TASKS as WIPE_TASKS
from task_plan import check_parser

TASKS = {}
for group in (PICKPLACE_TASKS, NUT_TASKS, DOOR_TASKS, WIPE_TASKS):
    TASKS.update(group)


def main():
    parser = argparse.ArgumentParser(description="用真值步骤执行新任务，检查计划是否够用")
    parser.add_argument("--env", choices=sorted(TASKS), required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    check_parser()
    task = TASKS[args.env]()
    rng = np.random.default_rng(args.seed)
    env = make_env(task, args.image_size, args.seed, use_camera=False)
    matrix = world_to_pixel_matrix(env, args.image_size)
    successes = 0
    try:
        for episode in range(args.episodes):
            style = task.styles[int(rng.integers(len(task.styles)))]
            successes += int(run_episode(env, task, episode, rng, style, matrix, args.image_size))
    finally:
        env.close()
    print(
        "oracle {} {}/{} succeeded ({:.0%})".format(
            args.env, successes, args.episodes, successes / float(args.episodes) if args.episodes else 0.0
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
