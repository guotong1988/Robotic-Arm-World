"""不加载模型，执行 Stack 初始画面上的步骤，确认这套计划能叠成功。

和采集用同一个标注：按当前画面生成步骤，再解析成 task_plan 的执行列表。
像素在提问这一帧反投影，之后机械臂只按世界坐标走，不再看图像。
"""

import argparse

import gl_backend

gl_backend.configure()

import numpy as np

from collect_stack_mllm import (
    INSTRUCTION_STYLES,
    is_grasped,
    make_env,
    place_offset,
    plan_for,
    positions_from_obs,
    prepare_scene,
)
from collect_lift_mllm import world_to_pixel_matrix
from task_plan import ParseError, check_parser, execute_plan, parse_plan


def oracle_episode(env, rng, episode, image_size):
    obs, hold_ori, rest_z = prepare_scene(env, rng)
    style = INSTRUCTION_STYLES[int(rng.integers(len(INSTRUCTION_STYLES)))]
    matrix = world_to_pixel_matrix(env, image_size)
    offset = place_offset(env)
    grasped = is_grasped(env)
    done = bool(env._check_success())
    plan = plan_for(obs, rest_z, offset, matrix, style, grasped, done)
    parsed = parse_plan(plan["answer"], image_size=image_size)
    cube_a, cube_b, _ = positions_from_obs(obs)
    kinds = [step["type"] for step in parsed["subtasks"]]
    print(
        "episode {:02d} instruction={} steps={}".format(episode, plan["instruction"], ">".join(kinds)),
        flush=True,
    )
    final = obs
    try:
        if parsed["subtasks"]:
            final, _trace = execute_plan(
                env, parsed["subtasks"], hold_ori, [cube_a, cube_b], matrix, grip=-1.0
            )
    except ParseError as exc:
        print("episode {:02d} parse/execute failed: {}".format(episode, exc), flush=True)
        return False
    if final is None:
        final = obs
    cube_a, cube_b, _ = positions_from_obs(final)
    success = bool(env._check_success())
    horiz = float(np.linalg.norm(cube_a[:2] - cube_b[:2]))
    print(
        "episode {:02d} success={} grasped={} horiz={:.3f} zA={:.3f} zB={:.3f}".format(
            episode,
            success,
            is_grasped(env),
            horiz,
            cube_a[2],
            cube_b[2],
        ),
        flush=True,
    )
    return success


def main():
    parser = argparse.ArgumentParser(description="用真值步骤执行 Stack，检查计划是否够用")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    check_parser()
    rng = np.random.default_rng(args.seed)
    env = make_env(args.image_size, args.seed, use_camera=False)
    successes = 0
    try:
        for episode in range(args.episodes):
            try:
                successes += int(oracle_episode(env, rng, episode, args.image_size))
            except Exception as exc:
                print("episode {:02d} error: {}".format(episode, exc), flush=True)
    finally:
        env.close()
    print(
        "oracle {}/{} succeeded ({:.0%})".format(
            successes, args.episodes, successes / float(args.episodes) if args.episodes else 0.0
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
