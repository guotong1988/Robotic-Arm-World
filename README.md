# Robotic Arm World

https://space.bilibili.com/447278957/lists

## Hello World

~/.pyenv/versions/3.11.13/bin/mjpython demo_random_action.py

~/.pyenv/versions/3.11.13/bin/mjpython lift_xarm7_ik.py

~/.pyenv/versions/3.11.13/bin/python3 lift_xarm7_ik.py --headless

## 准备MLLM训练数据

采集 Lift / Panda 的 MLLM 图文数据。每条样本是 frontview 的一张图、一条抬升指令，以及这一帧的方块位置和夹爪末端位置（世界坐标，米）。macOS 上脚本会自己设 `MUJOCO_GL=cgl` 做离屏渲染，用 `python3` 跑，不要用 `mjpython`。

```
~/.pyenv/versions/3.11.13/bin/python3 collect_lift_mllm.py --episodes 50 --out data/lift_mllm
```

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--episodes` | 50 | 采集多少条抓取 |
| `--out` | `data/lift_mllm` | 输出目录 |
| `--seed` | 0 | 随机种子 |
| `--image-size` | 256 | 图像边长 |
| `--height-min` | 0.06 | 抬升高度下限，米 |
| `--height-max` | 0.18 | 抬升高度上限，米 |

输出在 `--out` 目录：`images/` 存 frontview RGB，`samples.jsonl` 每行一条样本，`meta.json` 记环境和采集统计。夹爪不在方块正上方时，指令是先移到目标上方再抬剩余高度；对准之后只写还要抬多少。