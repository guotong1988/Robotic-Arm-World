#!/usr/bin/env bash
# 各任务测评并行后台执行，结果写到 $OUT/<任务>/（results.jsonl、summary.json、videos/）。
# 视频是左右并排：左边 frontview 标出模型给的像素，右边是深度反投影的三维点云和定位点。
# 日志各自写到 $OUT/logs；任一任务失败则脚本以非零退出。
# 所有任务连同一个 vLLM 服务，地址用 API_BASE 环境变量覆盖。
# 直接运行会整体转到后台（nohup），立即返回，总日志是 $LOG/eval.log。
# 全部任务结束后，按各目录的 summary.json 追加一张成功率表到这份总日志。
# 调试时用 bash eval.sh --fg 前台执行，能看到实时输出和退出码。
# ORACLE=1 bash eval.sh 不连模型，用真值步骤检查深度定位和执行器。
set -e
PY=${PY:-/opt/conda/envs/python3.10.13/bin/python3}
API_BASE=${API_BASE:-http://127.0.0.1:8000/v1}
EPISODES=${EPISODES:-20}
TIMEOUT=${TIMEOUT:-600}
OUT=${OUT:-data/eval}
CAMERAS=frontview,backview
LOG=$OUT/logs
mkdir -p $LOG

if [ "${1:-}" != "--fg" ]; then
    nohup bash "$0" --fg > $LOG/eval.log 2>&1 &
    echo "后台执行 pid $!，总日志 $LOG/eval.log，各任务日志在 $LOG"
    exit 0
fi

EXTRA=""
if [ "${ORACLE:-}" = "1" ]; then
    EXTRA="--oracle"
fi

pids=()
run() {
  name=$1
  shift
  "$@" > $LOG/$name.log 2>&1 &
  pids+=($!)
}

run lift $PY eval_lift_mllm_vllm.py --api-base $API_BASE --timeout $TIMEOUT --episodes $EPISODES --out $OUT/lift --cameras $CAMERAS $EXTRA
run stack $PY eval_stack_mllm_vllm.py --api-base $API_BASE --timeout $TIMEOUT --episodes $EPISODES --out $OUT/stack --cameras $CAMERAS $EXTRA
run door $PY eval_door_mllm_vllm.py --api-base $API_BASE --timeout $TIMEOUT --episodes $EPISODES --out $OUT/door --cameras $CAMERAS $EXTRA
run wipe $PY eval_wipe_mllm_vllm.py --api-base $API_BASE --timeout $TIMEOUT --episodes $EPISODES --out $OUT/wipe --cameras $CAMERAS $EXTRA
for env in PickPlaceMilk PickPlaceBread PickPlaceCereal PickPlaceCan PickPlace; do
    run pickplace_$env $PY eval_pickplace_mllm_vllm.py --api-base $API_BASE --timeout $TIMEOUT --env $env --episodes $EPISODES --out $OUT/pickplace_$env --cameras $CAMERAS $EXTRA
done
for env in NutAssemblySquare NutAssemblyRound NutAssembly; do
    run nut_$env $PY eval_nut_mllm_vllm.py --api-base $API_BASE --timeout $TIMEOUT --env $env --episodes $EPISODES --out $OUT/nut_$env --cameras $CAMERAS $EXTRA
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

# 表写进总日志。后台时 stdout 已经是这份日志，不要再往 stdout 打一遍，避免写偏位置。
summary_file=$LOG/summary.txt
set +e
"$PY" "$(dirname "$0")/eval_summary.py" "$OUT" > "$summary_file"
table_status=$?
set -e
if [ "$table_status" -eq 0 ]; then
    printf '\n' >> "$LOG/eval.log"
    cat "$summary_file" >> "$LOG/eval.log"
    if [ -t 1 ]; then
        printf '\n'
        cat "$summary_file"
    fi
else
    printf '\n汇总表生成失败\n' >> "$LOG/eval.log"
fi
rm -f "$summary_file"
exit $fail
