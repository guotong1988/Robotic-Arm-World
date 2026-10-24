# Robotic Arm World

https://space.bilibili.com/447278957/lists

robosuite + Panda 上的单臂操作。微调后的 Qwen 看 frontview 和 backview（采集脚本加在 frontview 正对面、往 +y 偏一截的背面机位，视线对着桌面中心；`--cameras frontview` 退回单目）。指令以每台相机一个 `<image>` 占位开头，注明各图来自哪台相机，话术后面带字段说明。模型输出 `subtasks`，每步一句 `desc`。同一个点在正面相机写 `target_1`、背面相机写 `target_1_back`，看不到的视图写 `"mask"`。要抓的物体（Lift 和 Stack 的方块、螺母把手）写两指落点，每个落点各给两台相机的像素（`target_1`、`target_1_back`、`target_2`、`target_2_back`），`target_kind` 为 `grip`，中点是抓取位置。评测只用深度把像素换成世界坐标：单点按角色融合，主视图是 frontview；两指落点成对还原。Lift 按两个落点做逆运动学，其余任务交给 `task_plan.py` 的执行器。

采集、评测、步骤原语和深度定位的说明见 [QUICK_START.md](QUICK_START.md)。

## 测评结果

| 任务 | 成功率 | 计划一致 | 备注 |
| --- | ---: | ---: | --- |
| Lift | **20/20 = 100%** | 不适用 | Lift 不比整条计划，成功率就是抓住并抬起的比例 |
| PickPlaceBread | **19/20 = 95%** | 20/20 |  |
| PickPlaceCan | **17/20 = 85%** | 20/20 |  |
| PickPlaceCereal | **17/20 = 85%** | 20/20 |  |
| Stack | **15/20 = 75%** | 16/20 |  |
| PickPlaceMilk | **13/20 = 65%** | 20/20 |  |
| Door | **12/20 = 60%** | 20/20 |  |
| Wipe | **8/20 = 40%** | 31/35 | 4 次解析失败 |
| NutAssemblySquare | **4/20 = 20%** | 20/20 |  |
| NutAssemblyRound | **4/20 = 20%** | 20/20 |  |
| PickPlace（四物体） | **2/20 = 10%** | 4/20 |  |
| NutAssembly（双螺母） | **0/20 = 0%** | 20/20 |  |

合计（12 个设置）：131/240 ≈ 54.6%
