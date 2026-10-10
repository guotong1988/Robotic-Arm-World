# Robotic Arm World

https://space.bilibili.com/447278957/lists

## Hello World

~/.pyenv/versions/3.11.13/bin/mjpython demo_random_action.py

~/.pyenv/versions/3.11.13/bin/mjpython lift_xarm7_ik.py

~/.pyenv/versions/3.11.13/bin/python3 lift_xarm7_ik.py --headless

## 准备MLLM训练数据

采集 Lift / Panda 的 MLLM 图文数据。每条样本的输入是 frontview 的 RGB 和一条抬升指令。输出只有 `subtasks`：要动位置的步骤各自带方块中心像素 `"[x y]"`（x 向右、y 向下，原点在左上角）。评测时用现场的方块深度把预测像素反投影成世界坐标，再交给机械臂。macOS 上脚本会自己设 `MUJOCO_GL=cgl` 做离屏渲染，用 `python3` 跑，不要用 `mjpython`。

```
~/.pyenv/versions/3.11.13/bin/python3 collect_lift_mllm.py --episodes 50 --out data/lift_mllm
```

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--episodes` | 50 | 采集多少条抓取 |
| `--out` | `data/lift_mllm` | 输出目录 |
| `--seed` | 0 | 随机种子 |
| `--image-size` | 512 | 图像边长 |
| `--height-min` | 0.06 | 抬升高度下限，米 |
| `--height-max` | 0.18 | 抬升高度上限，米 |

输出在 `--out` 目录：`images/` 存 frontview RGB，`samples.jsonl` 每行一条样本，`meta.json` 记环境和采集统计，其中 `camera_world_to_pixel` 是评测反投影用的相机矩阵。夹爪不在方块正上方时，指令是先移到目标上方再抬剩余高度，计划从 `move_above` 开始；对准之后指令只写还要抬多少，计划从还没做的步骤开始。`lift.delta_z` 与指令里的剩余高度一致，最后一步是 `stabilize`（0.5 秒）。

## 其他任务的执行步骤

Lift 评测只取模型给出的方块像素，后面的「到上方、下降、合爪、抬起」是固定流程。Stack 要先抓红色方块，再放到绿色方块上，然后松开并把夹爪抬开，步骤和 Lift 不一样。

采集时每条样本的输出都是一条只有 `subtasks` 的 JSON。空间步骤各自带像素和参数，执行器按这个列表做。`move_above` 带 `clearance`，`move_down` 带 `offset_z`，`lift` 带 `delta_z`，另外有 `close_gripper`、`open_gripper`、`stabilize`。Lift 的这些步骤指向同一个方块，Stack 的前几步指向红块、后几步指向绿块。评测仍是像素加该物体中心的相机深度，反投影成世界坐标后再走。

```
~/.pyenv/versions/3.11.13/bin/python3 collect_stack_mllm.py --episodes 50 --out data/stack_mllm
```

不加载模型，直接执行初始画面生成的步骤，用来确认计划本身能叠成功：

```
~/.pyenv/versions/3.11.13/bin/python3 eval_stack_oracle.py --episodes 20
```

用模型测 Stack：初始画面只问一次，模型给出全部步骤（两个方块的像素、抬升量、放置高度等）。像素换世界坐标只用深度相机：像素定视线，深度图估物体中心离桌面多高，取视线与该水平面的交点。仿真里的物体位置只用于事后统计误差，不进模型输入，也不参与控制。`--oracle` 用真值步骤代替模型，其余流程不变。

```
python3 eval_stack_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20
python3 eval_stack_mllm_vllm.py --oracle --episodes 20
```

## 其余单臂任务

PickPlace、NutAssembly、Door、Wipe 共用 `collect_common.py` 的采集流程和 `task_plan.py` 的执行器，输出格式和 Lift、Stack 一样，可以写进同一个 `--out`。样本的 `task` 是小写的环境名（如 `pickplacecan`），`samples.jsonl` 按 task 合并。每一帧的标注是从这一帧起还要做的步骤：正在做的移动到位了就不写，`lift`、`turn` 写剩下的量，`wipe` 只写还没擦掉的污渍，完成后为空。没做成的轨迹整条丢掉。

| 脚本 | `--env` | 任务 |
| --- | --- | --- |
| `collect_pickplace_mllm.py` | `PickPlace`、`PickPlaceSingle`、`PickPlaceMilk`、`PickPlaceBread`、`PickPlaceCereal`、`PickPlaceCan` | 把左箱的物品放进右箱对应的格子 |
| `collect_nut_mllm.py` | `NutAssembly`、`NutAssemblySingle`、`NutAssemblySquare`、`NutAssemblyRound` | 把螺母套到形状对应的柱子上 |
| `collect_door_mllm.py` | 无 | 转开门把手，把门拉开 |
| `collect_wipe_mllm.py` | 无 | 用擦板擦掉桌上的一道污渍 |

```
~/.pyenv/versions/3.11.13/bin/python3 collect_pickplace_mllm.py --env PickPlaceCan --episodes 50 --out data/mllm
~/.pyenv/versions/3.11.13/bin/python3 collect_nut_mllm.py --env NutAssemblySquare --episodes 50 --out data/mllm
~/.pyenv/versions/3.11.13/bin/python3 collect_door_mllm.py --episodes 50 --out data/mllm
~/.pyenv/versions/3.11.13/bin/python3 collect_wipe_mllm.py --episodes 50 --out data/mllm
```

这几个脚本的参数都是 `--episodes`（成功的条数）、`--out`、`--seed`、`--image-size`、`--bucket-cm`（夹爪大约每移动多少厘米存一帧，默认 2）。

在 Lift、Stack 的步骤之外新加的原语和参数：

- `move_above`、`move_down` 可带 `yaw`：夹爪竖直朝下，两指合拢方向从世界 y 轴绕竖直轴转过的角度，度。用在长方物品、螺母把手上。
- `move_above`、`move_down` 可带 `held`：手里物体上要对准目标的那一点的像素。螺母要对准柱子的是螺母中心，不是夹爪。抓住后夹爪再转，这一点跟着一起转。
- `reach`：夹爪沿 `approach`（世界坐标单位向量）伸向 `target`，停在 `standoff` 米处，两指上下合拢。Door 用它从侧面抓把手。
- `turn`：夹爪连同手里的东西绕过 `pivot` 像素、方向为 `axis` 的转轴转 `angle` 度。Door 先绕把手根部转开门锁，再绕铰链把门拉开。
- `wipe`：压着桌面依次走过 `path` 里的像素，`offset_z` 是往下压的量。

不加载模型、只执行真值步骤，检查计划本身能不能做成：

```
~/.pyenv/versions/3.11.13/bin/python3 eval_task_oracle.py --env NutAssembly --episodes 20
```

ToolHang 没有做：先要把挂架抓起来竖直插进底座（这一步加上插入时降低控制器刚度已经能做成），再把扳手翻过来挂到挂架的横杆上。横杆朝向被底座方孔限定，扳手挂上去时夹爪离机器人底座太近，Panda 的肘关节和腕关节都会顶到极限，按这套笛卡尔空间的步骤做不出来。

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
