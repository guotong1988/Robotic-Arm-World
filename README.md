# Robotic Arm World

https://space.bilibili.com/447278957/lists

本项目基于 robosuite + Panda 实现 VLM 驱动的单臂操作。模型根据单张 frontview RGB 和指令生成操作计划，再由控制器驱动机械臂执行。空间标签支持两种模式：默认 `pixel` 使用 2D 像素并结合深度恢复三维位置；`xyz` 直接预测机器人基座坐标系中的 XYZ 整数厘米，评测不经过像素反投影或深度定位。

```bash
# 采集：默认输出分别为 mllm_data 和 mllm_data_xyz
bash collect.sh pixel --fg
bash collect.sh xyz --fg

# 评测：random 也可换成 fixed
bash eval.sh random pixel --fg
bash eval.sh random xyz --fg
```
去掉 --fg 是后台评估。

两种模式的默认输出目录彼此隔离，也可用 `OUT` 覆盖。XYZ 数据不写 `target_kind`、`align_point_kind`、`pivot_kind`、`path_kind`，这些角色字段只在 pixel 模式用于选择深度量法。

采集、评测、步骤原语和坐标定位的说明见 [QUICK_START.md](QUICK_START.md)。

## 测评结果

### pixel 方案

| 任务 | 成功率 | 计划一致 | 备注 |
| --- | ---: | ---: | --- |
| Stack | **20/20 = 100%** | 17/20 |  |
| PickPlaceBread | **20/20 = 100%** | 20/20 |  |
| PickPlaceCereal | **19/20 = 95%** | 20/20 |  |
| PickPlaceMilk | **19/20 = 95%** | 20/20 |  |
| NutAssemblySquare | **19/20 = 95%** | 20/20 |  |
| NutAssemblyRound | **19/20 = 95%** | 20/20 |  |
| Lift | **18/20 = 90%** | 不适用 | Lift 不比整条计划，成功率就是抓住并抬起的比例；失败 2 条未抓住 |
| Wipe | **18/20 = 90%** | 24/25 |  |
| Door | **14/20 = 70%** | 20/20 |  |
| PickPlaceCan | **14/20 = 70%** | 20/20 |  |
| PickPlace（四物体） | **13/20 = 65%** | 17/20 |  |
| NutAssembly（双螺母） | **9/20 = 45%** | 20/20 |  |

合计（12 个设置）：202/240 ≈ 84.2%

### XYZ 方案

| 任务 | 成功率 | 计划一致 | 备注 |
| --- | ---: | ---: | --- |
| Lift | **20/20 = 100%** | 不适用 | Lift 不比整条计划，成功率就是抓住并抬起的比例 |
| Door | **20/20 = 100%** | 20/20 |  |
| PickPlaceCan | **20/20 = 100%** | 20/20 |  |
| PickPlaceBread | **20/20 = 100%** | 20/20 |  |
| PickPlaceCereal | **20/20 = 100%** | 20/20 |  |
| PickPlaceMilk | **20/20 = 100%** | 20/20 |  |
| Stack | **19/20 = 95%** | 16/20 |  |
| PickPlace（四物体） | **18/20 = 90%** | 17/20 |  |
| NutAssemblyRound | **17/20 = 85%** | 20/20 |  |
| Wipe | **16/20 = 80%** | 32/33 |  |
| NutAssemblySquare | **16/20 = 80%** | 20/20 |  |
| NutAssembly（双螺母） | **15/20 = 75%** | 20/20 |  |

合计（12 个设置）：221/240 ≈ 92.1%

