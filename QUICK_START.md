# Quick Start

## Hello World

```
~/.pyenv/versions/3.11.13/bin/mjpython hello_world/demo_random_action.py
~/.pyenv/versions/3.11.13/bin/mjpython hello_world/lift_xarm7_ik.py
~/.pyenv/versions/3.11.13/bin/python3 hello_world/lift_xarm7_ik.py --headless
```

## 任务一览

每个任务有一个采集脚本和一个评测脚本。采集脚本生成 MLLM 图文数据，评测脚本把初始画面和指令发给模型，按模型给出的步骤执行。macOS 上脚本会自己设 `MUJOCO_GL=cgl` 做离屏渲染，用 `python3` 跑，不要用 `mjpython`。

| 任务 | 采集脚本 | 评测脚本 | 内容 |
| --- | --- | --- | --- |
| Lift | `collect_lift_mllm.py` | `eval_lift_mllm_vllm.py` | 抓起方块，抬到指令给的高度 |
| Stack | `collect_stack_mllm.py` | `eval_stack_mllm_vllm.py` | 抓起红块放到绿块上 |
| PickPlace | `collect_pickplace_mllm.py` | `eval_pickplace_mllm_vllm.py` | 把左箱的物品放进右箱对应的格子 |
| NutAssembly | `collect_nut_mllm.py` | `eval_nut_mllm_vllm.py` | 把螺母套到形状对应的柱子上 |
| Door | `collect_door_mllm.py` | `eval_door_mllm_vllm.py` | 转开门把手，把门拉开 |
| Wipe | `collect_wipe_mllm.py` | `eval_wipe_mllm_vllm.py` | 用擦板擦掉桌上的一道污渍 |

## 数据格式

每条样本的输入是 frontview 的 RGB 和一条指令，输出是一条只有 `subtasks` 的 JSON。空间步骤各自带像素 `"[x y]"`（x 向右、y 向下，原点在左上角）和参数，执行器 `task_plan.py` 按这个列表做，它不认识具体任务。

每一帧的标注是从这一帧起还要做的步骤：正在做的移动到位了就不写，`lift`、`turn` 写剩下的量，`wipe` 只写还没擦掉的污渍，任务完成后为空。没做成的轨迹整条丢掉。

输出在 `--out` 目录：`images/` 存 frontview RGB，`samples.jsonl` 每行一条样本，`meta.json` 记环境和采集统计，其中 `camera_world_to_pixel` 是评测反投影用的相机矩阵。样本的 `task` 是任务名（PickPlace、NutAssembly 用小写的环境名，如 `pickplacecan`），样本 id 和图片名带任务名前缀。所有任务可以写进同一个 `--out`，`samples.jsonl` 按 task 合并，后跑的采集不会盖掉其他任务。

所有采集脚本都有这几个参数：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--episodes` | 50 | 采集多少条成功的轨迹 |
| `--out` | 各脚本不同 | 输出目录 |
| `--seed` | 0 | 随机种子 |
| `--image-size` | 512 | 图像边长 |

## 步骤原语

| 原语 | 参数 | 含义 |
| --- | --- | --- |
| `move_above` | `clearance`，可选 `yaw`、`held` | 移到目标上方 |
| `move_down` | `offset_z`，可选 `yaw`、`held` | 下降到目标 |
| `lift` | `delta_z` | 竖直抬起 |
| `close_gripper` | | 合爪 |
| `open_gripper` | | 松爪 |
| `stabilize` | | 停住 |
| `reach` | `target`、`approach`、`standoff` | 夹爪沿 `approach`（世界坐标单位向量）伸向 `target`，停在 `standoff` 米处，两指上下合拢 |
| `turn` | `pivot`、`axis`、`angle` | 夹爪连同手里的东西绕过 `pivot` 像素、方向为 `axis` 的转轴转 `angle` 度 |
| `wipe` | `path`、`offset_z` | 压着桌面依次走过 `path` 里的像素，`offset_z` 是往下压的量 |

- `yaw`：夹爪竖直朝下，两指合拢方向从世界 y 轴绕竖直轴转过的角度，度。用在长方物品、螺母把手上。
- `held`：手里物体上要对准目标的那一点的像素。螺母要对准柱子的是螺母中心，不是夹爪。抓住后夹爪再转，这一点跟着一起转。

## 评测时的坐标换算

像素换世界坐标只用深度相机（`depth_locate.py`），仿真里的物体位置只用于事后统计误差，不进模型输入，也不参与控制。一个像素只定一条视线，还差一个数。标注点常被挡住或在物体内部，不能直接读该像素的深度，而是按像素在计划里的角色，从它周围的深度点量出标注点所在的平面，再取视线和平面的交点：

| 角色 | 量什么 |
| --- | --- |
| 要抓的物品（Lift、Stack、PickPlace） | 底下支撑面和物体顶面，取半高 |
| 箱里的格子 | 箱沿高度减壁高（底面被箱壁挡住） |
| 螺母 | 桌面加半个螺母厚 |
| 柱顶 | 柱子上的顶面 |
| 污渍 | 桌面 |
| 门的转轴、把手 | 竖直的门板平面，法向取模型给的 `approach` |

角色只看步骤顺序：合爪前指向要抓的东西，合爪后到松爪前指向放置目标。多物体时，每抓下一个之前用当前帧的深度重新量它，开局挡在它前面的物体可能已经搬走了。

评测时初始画面只问一次，模型给出全部步骤。所有评测脚本都有 `--api-base`（vLLM 服务地址）和 `--episodes`；`--oracle` 不加载模型，用真值步骤代替模型、其余流程不变，用来检查计划本身和执行器能不能做成。

## Lift

夹爪不在方块正上方时，指令是先移到目标上方再抬剩余高度，计划从 `move_above` 开始；对准之后指令只写还要抬多少，计划从还没做的步骤开始。`lift.delta_z` 与指令里的剩余高度一致，最后一步是 `stabilize`（0.5 秒）。

```
~/.pyenv/versions/3.11.13/bin/python3 collect_lift_mllm.py --episodes 50 --out data/lift_mllm
python3 eval_lift_mllm_vllm.py --api-base http://127.0.0.1:8000/v1
python3 eval_lift_mllm_vllm.py --oracle
```

采集额外参数：`--height-min`（抬升高度下限，默认 0.06 米）、`--height-max`（上限，默认 0.18 米）。

## Stack

红块还在桌上时，步骤是移到红块上方、下降、合爪、抬高、移到绿块上方、下降、松爪、抬开、停住，前几步的像素指向红块，后几步指向绿块。已经抓住并且对准绿块时，前面的步骤不再写。

```
~/.pyenv/versions/3.11.13/bin/python3 collect_stack_mllm.py --episodes 50 --out data/stack_mllm
python3 eval_stack_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20
python3 eval_stack_mllm_vllm.py --oracle --episodes 20
```

## PickPlace

`--env` 可选 `PickPlace`、`PickPlaceSingle`、`PickPlaceMilk`、`PickPlaceBread`、`PickPlaceCereal`、`PickPlaceCan`（默认）。长方物品用 `yaw` 对齐。

```
~/.pyenv/versions/3.11.13/bin/python3 collect_pickplace_mllm.py --env PickPlaceCan --episodes 50 --out data/mllm
python3 eval_pickplace_mllm_vllm.py --env PickPlaceCan --api-base http://127.0.0.1:8000/v1 --episodes 20
python3 eval_pickplace_mllm_vllm.py --env PickPlaceCan --oracle --episodes 20
```

## NutAssembly

`--env` 可选 `NutAssembly`、`NutAssemblySingle`、`NutAssemblySquare`（默认）、`NutAssemblyRound`。抓螺母把手用 `yaw`，套柱子时用 `held` 对准螺母中心。

```
~/.pyenv/versions/3.11.13/bin/python3 collect_nut_mllm.py --env NutAssemblySquare --episodes 50 --out data/mllm
python3 eval_nut_mllm_vllm.py --env NutAssembly --api-base http://127.0.0.1:8000/v1 --episodes 20
python3 eval_nut_mllm_vllm.py --env NutAssembly --oracle --episodes 20
```

## Door

用 `reach` 从侧面抓把手，先 `turn` 绕把手根部转开门锁，再 `turn` 绕铰链把门拉开。

```
~/.pyenv/versions/3.11.13/bin/python3 collect_door_mllm.py --episodes 50 --out data/mllm
python3 eval_door_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20
python3 eval_door_mllm_vllm.py --oracle --episodes 20
```

## Wipe

末端装的是擦板，没有夹爪，计划里不会出现 `close_gripper`、`open_gripper`。步骤是移到第一处污渍上方、降到桌面、`wipe` 压着桌面沿污渍走、抬起、停住。

```
~/.pyenv/versions/3.11.13/bin/python3 collect_wipe_mllm.py --episodes 50 --out data/mllm
python3 eval_wipe_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20
python3 eval_wipe_mllm_vllm.py --oracle --episodes 20
```

PickPlace、NutAssembly、Door、Wipe 的采集脚本还有 `--bucket-cm`：夹爪大约每移动多少厘米存一帧，默认 2。

## 未支持：ToolHang

先要把挂架抓起来竖直插进底座（这一步加上插入时降低控制器刚度已经能做成），再把扳手翻过来挂到挂架的横杆上。横杆朝向被底座方孔限定，扳手挂上去时夹爪离机器人底座太近，Panda 的肘关节和腕关节都会顶到极限，按这套笛卡尔空间的步骤做不出来。

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
之后按上面各任务小节的评测命令跑，`--api-base` 指向这个服务。
