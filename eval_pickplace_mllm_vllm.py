"""用微调后的 Qwen 做 PickPlace 系列测评，直连 vLLM 的 OpenAI chat completions。

    python3 eval_pickplace_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --env PickPlaceCan --episodes 20
    python3 eval_pickplace_mllm_vllm.py --oracle --env PickPlace --episodes 5

一张 frontview RGB 加一句「把牛奶放进右边箱子里对应的格子。」，模型写出每样物品
移到上方、下降、合爪、抬过箱壁、放到格子里、松爪退开的全部步骤，长方物品带 yaw。
物品中心用深度图量出它的底面和顶面，取半高处的视线交点；格子底面被箱壁挡着看不到，
用量到的箱沿高度减去壁高。流程详见 eval_task_core.py 和 depth_locate.py。
"""

import argparse

from collect_pickplace_mllm import ENVS, PickPlaceTask
from eval_task_core import add_common_args, make_client, run_eval, self_test


def main():
    parser = argparse.ArgumentParser(description="直连 vLLM，在 robosuite PickPlace 系列上测归置物品成功率")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI 接口地址")
    parser.add_argument("--env", choices=ENVS, default="PickPlaceCan")
    add_common_args(parser, "data/pickplace_eval")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_eval(PickPlaceTask(args.env), args, make_client(args))


if __name__ == "__main__":
    main()
