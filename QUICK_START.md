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

每条样本的输入是一到两张 RGB 加一条指令：`image` 是 frontview（桌面正前方，robosuite 自带）。默认再加 `image_back`，即 backview。这台相机不是 robosuite 自带机位，采集脚本加在 frontview 正对面、往 +y 偏到约 (-1.25, 0.48, 1.80)，视线对着桌面中心 (0, 0, 0.82)，fovy 45 度；精确镜像的 -x 轴机位会被机身挡住。`--cameras frontview` 退回单目，所有任务都用同一套相机。指令以每台相机一个 `<image>` 占位开头，多张图时接着说明各图来自哪个相机，再接任务话术，话术后面带字段说明（写在指令里，不写进 answer）。输出是只含 `subtasks` 的 JSON，每个 subtask 带一句 `desc`（20 字以内）说明这一步做什么。空间字段的像素格式是 `"[x y]"`（x 向右、y 向下，原点在左上角）：`target_1` 对应 `image`，`target_1_back` 对应 `image_back`；`held`、`pivot`、`path` 同理各有 `held_back`、`pivot_back`、`path_back`。某台相机看不到那个点时对应字段写 `"mask"`，定位只用看得到的视图；两台都看不到时 frontview 保留像素兜底。单目没有 `*_back`，挡住也照写，不写 `"mask"`。要抓的物体（Lift、Stack 的方块，NutAssembly 的螺母把手）写两指落点：`target_1` 是朝 frontview 那一侧，`target_2` 是朝后那一侧，每个落点各给两台相机的像素（`target_1_back`、`target_2_back`），`target_kind` 为 `"grip"`，执行取两点的中点。放置目标和其余空间点仍是同一个点的两个视图。`task_plan.py` 按这个列表做，它不认识具体任务；Lift 的评测不走这个执行器，见 Lift 一节。

每条成功轨迹只保存开局那一帧（`phase` 为 `initial`），标注是从开局起的整条步骤。泛化靠开局布局，不靠轨迹中途的画面：方块、物品、螺母的位置和朝向，以及叠放目标、格子、柱子、门、污渍的位置都会变。没做成的轨迹整条丢掉。

输出在 `--out` 目录：`images/` 存各视图的 RGB（`样本id.png` 是 frontview，双目再加 `样本id_back.png` 是 backview），`samples.jsonl` 每行一条样本，`meta.json` 记环境和采集统计，其中 `camera_world_to_pixel` 是各视图的世界坐标到像素矩阵，评测反投影用。样本的 `task` 是任务名（PickPlace、NutAssembly 用小写的环境名，如 `pickplacecan`），样本 id 和图片名带任务名前缀。所有任务可以写进同一个 `--out`，`samples.jsonl` 按 task 合并，后跑的采集不会盖掉其他任务。

所有采集脚本都有这几个参数：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--episodes` | 50 | 采集多少条成功的轨迹 |
| `--out` | 各脚本不同 | 输出目录 |
| `--seed` | 0 | 随机种子 |
| `--episode-offset` | 0 | 首个 episode 编号；并行分片使用不同偏移，避免样本 ID 冲突 |
| `--episode-stride` | 1 | 每次尝试后的 episode 编号步长；并行分片使用 worker 总数，失败补采也不会与其他分片重叠 |
| `--image-size` | 512 | 图像边长 |
| `--cameras` | `frontview,backview` | 渲染并写入的相机。`frontview` 单目，被挡也照写像素；加上 `backview` 时被挡的视图写 `"mask"` |

批量运行 `collect.sh` 时可用 `EPISODES`（默认 100）、`WORKERS`（默认 2）、`SEED`、`CAMERAS`、`PY` 和 `OUT` 环境变量调整配置。每个任务再启动 `WORKERS` 个独立进程，episode 按 worker 编号交错分配，失败补采不会与其他分片重叠；分片全部采满并校验通过后才合并发布，任一 worker 失败会保留该任务的原有数据。单目采集和评测可统一使用 `CAMERAS=frontview`。

## 步骤原语

| 原语 | 参数 | 含义 |
| --- | --- | --- |
| `move_above` | `clearance`，可选 `yaw`、`held` | 移到目标上方 |
| `move_down` | `offset_z`，可选 `yaw`、`held` | 下降到目标 |
| `lift` | `delta_z` | 竖直抬起 |
| `close_gripper` | | 合爪 |
| `open_gripper` | | 松爪 |
| `stabilize` | | 停住 |
| `reach` | `target_1`、`approach`、`standoff` | 夹爪沿 `approach`（世界坐标单位向量）伸向 `target_1`，停在 `standoff` 米处，两指上下合拢 |
| `turn` | `pivot`、`axis`、`angle` | 夹爪连同手里的东西绕过 `pivot` 像素、方向为 `axis` 的转轴转 `angle` 度 |
| `wipe` | `path`、`offset_z` | 压着桌面依次走过 `path` 里的像素，`offset_z` 是往下压的量 |

- `yaw`：夹爪竖直朝下，两指合拢方向从世界 y 轴绕竖直轴转过的角度，度。只保留在无法由双落点确定的单点步骤中，例如 Nut 搬运到柱子上方、Wipe 初始朝向；双落点抓取步骤不写，执行器由两点连线推导。
- `held`：手里物体上要对准目标的那一点的像素。螺母要对准柱子的是螺母中心，不是夹爪。抓住后夹爪再转，这一点跟着一起转。
- `target_kind: "grip"`：`target_1`、`target_2` 是两指合拢时贴在物体两侧的两个落点，不是同一个点的两个视图。Lift、Stack、PickPlace 的竖直抓取和 NutAssembly 的螺母把手都这样写；Stack 的 Unstack/Reverse 也复用这一格式。两点的中点是抓取位置，连线决定合拢方向和 yaw，不再重复输出 `yaw`。只有 frontview 时，两个落点都写在这张图上，被物体挡住的另一个落点按形状和朝向补出像素，不写 `"mask"`；双目时每个落点再写 `target_1_back`、`target_2_back`，被挡住的那台写 `"mask"`。放置目标和其他空间字段仍是一点两视图。

## 评测时的坐标换算

像素换世界坐标只用深度相机（`depth_locate.py`），仿真里的物体位置只用于事后统计误差，不进模型输入，也不参与控制。一个像素只定一条视线，还差一个数。标注点常被挡住或在物体内部，不能直接读该像素的深度，而是按像素在计划里的角色，从它周围的深度点量出标注点所在的平面，再取视线和平面的交点。角色优先看步骤里的 `target_kind`、`held_kind`、`pivot_kind`、`path_kind`；没写或不认识时，按任务名和步骤顺序推断。

每个标注点在看得到的视图里各有一个像素，被 `"mask"` 的视图不参与。两个视图都有像素时，定位分两层：

- 测量级融合（`bin`、`nut`、`peg`、`table`）：两个视图的邻域点云都反投影到世界坐标系合成一片，在合并点云上量支撑面、顶面、中位数。单个视图被手臂等遮挡时，污染点是少数，被多数派统计自动投出去。模型在两个视图里指的不是同一个点（两条视线最近距离超过 1 厘米）时不合并，退回逐视图融合。
- 决策级融合（`object`、`cube_center`、`door_axis`、`door_handle`）：两个视图各自量平面、求交点，再融合。两台相机都是平视，主视图统一是 frontview。`object` 和 `cube_center` 的顶面沿主视线窄柱搜索，柱内遮挡不会被多数派投出，所以不合并点云；门板竖直，侧向视线和门板接近平行，求交不稳。主视图被 mask 或量不出来时用另一视图兜底；两个视图的分歧超过 3 厘米时取 z 低的那一路（手臂等遮挡都在桌面上方，被挡的视图量出来的点偏高）。

定位记录里带 `view`（实际用的视图；测量级合并时是两台相机名按字母序拼起来，如 `backview+frontview`）、`disagreement_m`（两视图独立定位的距离）和 `ray_gap_m`（两条视线的距离）。

| 角色 | 量什么 |
| --- | --- |
| 物体中心（没标类别时按结构推断的单点抓取目标） | 底下支撑面和物体顶面，取半高 |
| 方块中心（Stack 放绿块，`cube_center`） | 和物体中心同一套量法 |
| 箱里的格子 | 箱沿高度减壁高（底面被箱壁挡住） |
| 螺母 | 桌面加半个螺母厚 |
| 柱顶 | 柱子上的顶面 |
| 污渍 | 桌面 |
| 门的转轴、把手 | 竖直的门板平面，法向取模型给的 `approach` |
| 两指落点对（`grip`） | 成对还原两个落点（见下） |

两指落点成对还原（`recover_landings`）：双目且两台相机各看到一侧时，分别用自己的深度还原；单目或另一视图被 mask 时，Nut 用支撑面加半厚、PickPlace 结合物品顶面先验和可见指腹表面确定共同水平面，再求第二条像素射线与这个平面的交点。没有高度先验的任务退回落点附近点云和固定距离估计。执行取两落点的中点，统计误差也用中点配真值，合拢方向和 yaw 统一由恢复后的两点连线确定。深度恢复时 Stack 用开局保持的夹爪朝向作为初始连线方向，其他任务用世界 x 轴；单目分支再用两个像素的射线校正方向，旧数据里显式 `yaw` 仍兼容。

合爪前指向要抓的东西，合爪后到松爪前指向放置目标。多物体时，每抓下一个之前用当前帧的深度重新量它，开局挡在它前面的物体可能已经搬走了；两指落点也成对重测。

评测时初始画面只问一次，模型给出全部步骤。Wipe 没擦干净可以再问一轮。所有评测脚本都有 `--api-base`（vLLM 服务地址）、`--episodes` 和 `--cameras`；`--oracle` 不加载模型，用真值步骤代替模型、其余流程不变，用来检查计划本身和执行器能不能做成。

## Lift

夹爪不在方块正上方时，指令是先移到目标上方再抬剩余高度，计划从 `move_above` 开始；对准之后指令只写还要抬多少，计划从还没做的步骤开始。`lift.delta_z` 与指令里的剩余高度一致，最后一步是 `stabilize`（0.5 秒）。抓取步骤写两指落点对，采集时夹爪竖直朝下、两指沿世界 x 轴合拢。评测用这一帧的深度还原两个落点，再按落点做逆运动学（`grip_ik.py`）：中点扣掉指腹相对末端的偏移得到要到的位置，两种相差 180 度的抓法里挑能解、离关节限位远的那个，用关节位置控制器跟踪。抬升高度用指令里的数字。采集本身仍用世界坐标 OSC 走出这条轨迹。

```
~/.pyenv/versions/3.11.13/bin/python3 collect_lift_mllm.py --episodes 50 --out data/lift_mllm
python3 eval_lift_mllm_vllm.py --api-base http://127.0.0.1:8000/v1
python3 eval_lift_mllm_vllm.py --oracle
```

采集额外参数：`--height-min`（抬升高度下限，默认 0.06 米）、`--height-max`（上限，默认 0.18 米）。

## Stack

红块还在桌上时，步骤是移到红块上方、下降、合爪、抬高、移到绿块上方、下降、松爪、抬开、停住。抓红块写两指落点对（`target_kind` 为 `grip`，合拢轴是开局保持的夹爪朝向），放绿块写绿块中心（`target_kind` 为 `cube_center`）加距离。已经抓住并且对准绿块时，前面的步骤不再写。评测保持开局朝向，走 `task_plan.execute_plan`。

```
~/.pyenv/versions/3.11.13/bin/python3 collect_stack_mllm.py --episodes 50 --out data/stack_mllm
python3 eval_stack_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20
python3 eval_stack_mllm_vllm.py --oracle --episodes 20
```

另有两条复用同一场景的复合指令实验，`--seed` 和 Stack 相同时可逐条对照：`eval_stack_unstack_mllm_vllm.py`（叠上去再拿下来）、`eval_stack_reverse_mllm_vllm.py`（反向指令：绿叠红，统计模型是否无视指令照做原任务）。

## PickPlace

`--env` 可选 `PickPlace`、`PickPlaceSingle`、`PickPlaceMilk`、`PickPlaceBread`、`PickPlaceCereal`、`PickPlaceCan`（默认）。四类物品的抓取步骤都写两指落点（`grip`），中点包含顶面下 3 cm 的抓取高度；采集器内部仍用 `yaw` 选择长方物品的合拢方向，但标签只写落点，由连线恢复朝向。箱格放置仍写单个中心点（`bin`）。

```
~/.pyenv/versions/3.11.13/bin/python3 collect_pickplace_mllm.py --env PickPlaceCan --episodes 50 --out data/mllm
python3 eval_pickplace_mllm_vllm.py --env PickPlaceCan --api-base http://127.0.0.1:8000/v1 --episodes 20
python3 eval_pickplace_mllm_vllm.py --env PickPlaceCan --oracle --episodes 20
```

## NutAssembly

`--env` 可选 `NutAssembly`、`NutAssemblySingle`、`NutAssemblySquare`（默认）、`NutAssemblyRound`。抓螺母把手只写两指落点对，不再重复写抓取 `yaw`；套柱子时用 `held` 对准螺母中心，柱顶是单点（`peg`），搬运到柱子上方仍保留控制把手方向的 `yaw`。

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

采集脚本仍接受 `--bucket-cm`，但不再按夹爪位移抽帧。现在每条成功轨迹只存开局一帧。

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
