"""用微调后的 Qwen 做 Door 测评，直连 vLLM 的 OpenAI chat completions。

    python3 eval_door_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20
    python3 eval_door_mllm_vllm.py --oracle --episodes 5

一张 frontview RGB 加一句「把门打开。」，模型写出 reach 抓把手、close_gripper、
两次 turn（先绕把手根部转开门锁，再绕铰链把门拉开）、stabilize。
把手标注点在杆的内部，可见表面比它近十几厘米，不能直接读深度：先用 reach 的
approach 当门板法向，从深度图量出门板所在的竖直平面，转轴取视线和门板中面的交点，
把手取和中面外移把手距离的那个平面的交点。流程详见 eval_task_core.py 和 depth_locate.py。
"""

import argparse

from collect_door_mllm import DoorTask
from eval_task_core import add_common_args, make_client, run_eval, self_test


def main():
    parser = argparse.ArgumentParser(description="直连 vLLM，在 robosuite Door 上测开门成功率")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI 接口地址")
    add_common_args(parser, "data/door_eval")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_eval(DoorTask(), args, make_client(args))


if __name__ == "__main__":
    main()
