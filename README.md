# Robotic Arm World

https://space.bilibili.com/447278957/lists

本项目基于 robosuite + Panda 实现 VLM 驱动的单臂操作。模型根据单张 frontview RGB 和指令生成操作计划，再由控制器驱动机械臂执行。空间标签支持两种模式：默认 `pixel` 使用 2D 像素并结合深度恢复三维位置；`xyz` 直接预测机器人基座坐标系中的 XYZ 整数厘米，评测不经过像素反投影或深度定位。最后通过逆运动学（IK）驱动机械臂执行操作。

```bash
# 采集：默认输出分别为 mllm_data 和 mllm_data_xyz
bash collect.sh pixel --fg
bash collect.sh xyz --fg

# 评测：random 也可换成 fixed
bash eval.sh random pixel --fg
bash eval.sh random xyz --fg
```
去掉 --fg 是后台跑。

采集、评测、步骤原语和坐标定位的说明见 [QUICK_START.md](QUICK_START.md)。

## 测评结果

### pixel 方案（预测2D再用深度）

| 任务 | 成功率 | 计划一致 |
| --- | ---: | ---: |
| Stack | **20/20 = 100%** | 17/20 |
| PickPlaceBread | **20/20 = 100%** | 20/20 |
| PickPlaceCereal | **19/20 = 95%** | 20/20 |
| PickPlaceMilk | **19/20 = 95%** | 20/20 |
| NutAssemblySquare | **19/20 = 95%** | 20/20 |
| NutAssemblyRound | **19/20 = 95%** | 20/20 |
| Lift | **18/20 = 90%** | 不适用 |
| Wipe | **18/20 = 90%** | 24/25 |
| Door | **14/20 = 70%** | 20/20 |
| PickPlaceCan | **14/20 = 70%** | 20/20 |
| PickPlace（四物体） | **13/20 = 65%** | 17/20 |
| NutAssembly（双螺母） | **9/20 = 45%** | 20/20 |

合计（12 个设置）：202/240 ≈ 84.2%

### XYZ 方案（预测3D）

| 任务 | 成功率 | 计划一致 |
| --- | ---: | ---: |
| Lift | **20/20 = 100%** | 不适用 |
| Door | **20/20 = 100%** | 20/20 |
| Stack | **20/20 = 100%** | 16/20 |
| PickPlaceCan | **20/20 = 100%** | 20/20 |
| PickPlaceMilk | **20/20 = 100%** | 20/20 |
| PickPlaceBread | **19/20 = 95%** | 20/20 |
| PickPlaceCereal | **19/20 = 95%** | 20/20 |
| PickPlace（四物体） | **19/20 = 95%** | 17/20 |
| NutAssemblySquare | **18/20 = 90%** | 20/20 |
| NutAssemblyRound | **17/20 = 85%** | 20/20 |
| Wipe | **14/20 = 70%** | 29/32 |
| NutAssembly（双螺母） | **14/20 = 70%** | 20/20 |

合计（12 个设置）：220/240 ≈ 91.7%

