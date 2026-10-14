"""用微调后的 Qwen 做 Wipe 测评，直连 vLLM 的 OpenAI chat completions。

    python3 eval_wipe_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20
    python3 eval_wipe_mllm_vllm.py --oracle --episodes 5

一张 frontview RGB 加一句「把桌上的污渍擦干净。」，模型写出移到污渍上方、降到桌面、
wipe 沿剩下的污渍走一遍、抬起、stabilize。污渍点的像素用深度图量出桌面高度，取视线和桌面的交点。
和采集一样最多问两轮：第一遍漏了几个点时，按当前画面再问一次，只擦剩下的。
流程详见 eval_task_core.py。
"""

import argparse

from collect_wipe_mllm import WipeTask
from eval_task_core import add_common_args, make_client, run_eval, self_test


def main():
    parser = argparse.ArgumentParser(description="直连 vLLM，在 robosuite Wipe 上测擦桌子成功率")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI 接口地址")
    add_common_args(parser, "data/wipe_eval")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_eval(WipeTask(), args, make_client(args))


if __name__ == "__main__":
    main()
