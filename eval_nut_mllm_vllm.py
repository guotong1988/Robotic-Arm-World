"""用微调后的 Qwen 做 NutAssembly 系列测评，直连 vLLM 的 OpenAI chat completions。

    python3 eval_nut_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --env NutAssemblySquare --episodes 20
    python3 eval_nut_mllm_vllm.py --oracle --env NutAssembly --episodes 5

一张 frontview RGB 加一句「把方形螺母套到方柱上。」，模型写出带 yaw 的抓取、
带 held 的对准柱子、松爪退开的全部步骤。抓取点、held 的螺母中心
用深度图量出的桌面加半个螺母厚，柱顶用柱子上量到的顶面，都取视线交点。
流程详见 eval_task_core.py 和 depth_locate.py。
"""

import argparse

from collect_nut_mllm import ENVS, NutTask
from eval_task_core import add_common_args, make_client, run_eval, self_test


def main():
    parser = argparse.ArgumentParser(description="直连 vLLM，在 robosuite NutAssembly 系列上测套螺母成功率")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI 接口地址")
    parser.add_argument("--env", choices=ENVS, default="NutAssemblySquare")
    add_common_args(parser, "data/nut_eval")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_eval(NutTask(args.env), args, make_client(args))


if __name__ == "__main__":
    main()
