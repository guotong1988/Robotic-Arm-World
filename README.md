# Robotic Arm World

https://space.bilibili.com/447278957/lists

robosuite + Panda 上的单臂操作。微调后的 Qwen 只看一张 frontview RGB 和一句指令，输出 `subtasks` 步骤列表，空间步骤带像素；评测时只用深度相机把像素换成世界坐标，交给 `task_plan.py` 的执行器去做。

采集、评测、步骤原语和深度定位的说明见 [quick_start.md](quick_start.md)。

## 测评结果

| 任务 | 成功率 | 计划一致 | 备注 |
| --- | ---: | ---: | --- |
| Lift | **18/20 = 90%** | 不适用 | Lift 不比整条计划，成功率就是抓住并抬起的比例；失败 2 条未抓住 |
| Door | **17/20 = 85%** | 20/20 | 失败 3 条门闩几乎没动 |
| Stack | **16/20 = 80%** | 14/20 | 失败多为叠偏 |
| Wipe | **15/20 = 75%** | 24/38 | 失败多为擦不干净；1 次解析失败 |
| PickPlaceCan | **11/20 = 55%** | 20/20 | 单物体最好 |
| PickPlaceBread | **9/20 = 45%** | 20/20 | |
| PickPlaceCereal | **6/20 = 30%** | 18/20 | |
| PickPlaceMilk | **6/20 = 30%** | 19/20 | |
| NutAssemblySquare | **5/20 = 25%** | 20/20 | |
| NutAssemblyRound | **4/20 = 20%** | 20/20 | |
| NutAssembly（双螺母） | **2/20 = 10%** | 20/20 | |
| PickPlace（四物体） | **0/20 = 0%** | 1/20 | 常只放进 0–1 件 |

合计（12 个设置）：109/240 ≈ 45.4%
