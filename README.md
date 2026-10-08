# Robotic Arm World

https://space.bilibili.com/447278957/lists

## Hello World

~/.pyenv/versions/3.11.13/bin/mjpython demo_random_action.py

~/.pyenv/versions/3.11.13/bin/mjpython lift_xarm7_ik.py

~/.pyenv/versions/3.11.13/bin/python3 lift_xarm7_ik.py --headless

## 准备MLLM训练数据

采集 Lift / Panda 的 MLLM 图文数据。每条样本的输入是 frontview 的 RGB 和一条抬升指令。输出是一条抓取计划：`task` 为 `grasp`，`target` 是这一帧方块中心的整数像素 `"[x y]"`（x 向右、y 向下，原点在左上角），`subtasks` 是从这一帧还要做的步骤。评测时用现场的方块深度把预测像素反投影成世界坐标，再交给机械臂。macOS 上脚本会自己设 `MUJOCO_GL=cgl` 做离屏渲染，用 `python3` 跑，不要用 `mjpython`。

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

输出在 `--out` 目录：`images/` 存 frontview RGB，`samples.jsonl` 每行一条样本，`meta.json` 记环境和采集统计，其中 `camera_world_to_pixel` 是评测反投影用的相机矩阵。夹爪不在方块正上方时，指令是先移到目标上方再抬剩余高度，计划从 `move_above` 开始；对准之后指令只写还要抬多少，计划从还没做的步骤开始。`lift.delta_z` 与指令里的剩余高度一致，最后一步是 `stabilize`（0.5 秒）。

## 无窗口评测
安装 OSMesa。Mesa 25.1 起和当前 MuJoCo 的 OSMesa 接口不兼容，conda 包要卡在 25.1 之前。装到跑评测的那个环境里，不要设 `MUJOCO_GL=egl`。
```
conda install -y -c conda-forge 'mesalib<25.1'
```
没有 conda-forge 时：
```
sudo yum install -y mesa-libOSMesa mesa-libOSMesa-devel
```
启动模型服务
```
python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/qwen38_27b/ \
  --host 0.0.0.0 \
  --port 8000
```
评测
```
python3 eval_lift_mllm_vllm.py --api-base http://127.0.0.1:8000/v1
```
