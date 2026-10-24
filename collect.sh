#!/usr/bin/env bash
# 所有任务写进同一个目录。每条成功轨迹只存 phase=initial 的一帧，泛化靠开局布局。
# 图片和样本 id 带任务名前缀，samples.jsonl 按 task 合并。
# 各采集程序并行后台执行，日志各自写到 $OUT/logs；任一任务失败则脚本以非零退出。
# 直接运行会整体转到后台（nohup），立即返回，总日志是 $LOG/collect.log。
# 调试时用 bash collect.sh --fg 前台执行，能看到实时输出和退出码。
set -e
PY=/opt/conda/envs/python3.10.13/bin/python3
OUT=/data/oss_bucket_3/guotong/mllm_data/
CAMERAS=frontview,backview
LOG=$OUT/logs
mkdir -p $LOG

if [ "${1:-}" != "--fg" ]; then
    nohup bash "$0" --fg > $LOG/collect.log 2>&1 &
    echo "后台执行 pid $!，总日志 $LOG/collect.log，各任务日志在 $LOG"
    exit 0
fi

pids=()
run() {
  name=$1
  shift
  "$@" > $LOG/$name.log 2>&1 &
  pids+=($!)
}

run stack $PY collect_stack_mllm.py --episodes 100 --out $OUT --cameras $CAMERAS
run lift $PY collect_lift_mllm.py --episodes 100 --out $OUT --cameras $CAMERAS
for env in PickPlaceMilk PickPlaceBread PickPlaceCereal PickPlaceCan PickPlace; do
    run pickplace_$env $PY collect_pickplace_mllm.py --env $env --episodes 100 --out $OUT --cameras $CAMERAS
done
for env in NutAssemblySquare NutAssemblyRound NutAssembly; do
    run nut_$env $PY collect_nut_mllm.py --env $env --episodes 100 --out $OUT --cameras $CAMERAS
done
run door $PY collect_door_mllm.py --episodes 100 --out $OUT --cameras $CAMERAS
run wipe $PY collect_wipe_mllm.py --episodes 100 --out $OUT --cameras $CAMERAS

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
exit $fail
