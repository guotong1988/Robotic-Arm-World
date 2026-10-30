#!/usr/bin/env bash
# 各任务测评并行后台执行，结果写到 $OUT/<任务>/（results.jsonl、summary.json、videos/）。
# 视频与 collect.sh 的单目输入对齐：frontview 标出模型原始像素，不画点云。
# 日志各自写到 $OUT/logs；任一任务失败则脚本以非零退出。
# 所有任务连同一个 vLLM 服务，地址用 API_BASE 环境变量覆盖。
# 直接运行会整体转到后台（nohup），立即返回，总日志是 $LOG/eval.log。
# 全部任务结束后，按各目录的 summary.json 追加一张成功率表到这份总日志。
# 两种模式：bash eval.sh random 保持每次初始场景随机；bash eval.sh fixed 让 20 次使用相同初始。
# 调试时追加 --fg 前台执行，例如 bash eval.sh fixed --fg，能看到实时输出和退出码。
# SEED 可换固定初始场景（默认 1000）；ORACLE=1 不连模型，用真值步骤检查深度定位和执行器。
set -e
PY=${PY:-/opt/conda/envs/python3.10.13/bin/python3}
API_BASE=${API_BASE:-http://127.0.0.1:8000/v1}
EPISODES=${EPISODES:-20}
TIMEOUT=${TIMEOUT:-600}
SEED=${SEED:-1000}
IMAGE_SIZE=${IMAGE_SIZE:-1024}
MODE=${MODE:-random}
FG=0
for arg in "$@"; do
    case "$arg" in
        random|fixed) MODE=$arg ;;
        --fg) FG=1 ;;
        *) echo "用法: bash eval.sh [random|fixed] [--fg]" >&2; exit 2 ;;
    esac
done
if [ "$MODE" != "random" ] && [ "$MODE" != "fixed" ]; then
    echo "MODE 只能是 random 或 fixed" >&2
    exit 2
fi
case $IMAGE_SIZE in ''|*[!0-9]*) echo "IMAGE_SIZE 必须是正整数" >&2; exit 2;; esac
if [ "$IMAGE_SIZE" -lt 1 ]; then
    echo "IMAGE_SIZE 必须大于 0" >&2
    exit 2
fi
if [ "$MODE" = "fixed" ]; then
    DEFAULT_OUT=data/eval_fixed
else
    DEFAULT_OUT=data/eval
fi
OUT=${OUT:-$DEFAULT_OUT}
LOG=$OUT/logs
mkdir -p $LOG

if [ "$FG" -ne 1 ]; then
    nohup env PY="$PY" API_BASE="$API_BASE" EPISODES="$EPISODES" TIMEOUT="$TIMEOUT" SEED="$SEED" \
        IMAGE_SIZE="$IMAGE_SIZE" OUT="$OUT" ORACLE="${ORACLE:-}" \
        bash "$0" "$MODE" --fg > $LOG/eval.log 2>&1 &
    echo "后台执行 pid $!，模式 $MODE，总日志 $LOG/eval.log，各任务日志在 $LOG"
    exit 0
fi

EXTRA="--seed $SEED --image-size $IMAGE_SIZE"
if [ "$MODE" = "fixed" ]; then
    EXTRA="$EXTRA --fixed-initial"
fi
if [ "${ORACLE:-}" = "1" ]; then
    EXTRA="$EXTRA --oracle"
fi

pids=()
run() {
  name=$1
  shift
  "$@" > $LOG/$name.log 2>&1 &
  pids+=($!)
}

run lift $PY eval_lift_mllm_vllm.py --api-base $API_BASE --timeout $TIMEOUT --episodes $EPISODES --out $OUT/lift $EXTRA
run stack $PY eval_stack_mllm_vllm.py --api-base $API_BASE --timeout $TIMEOUT --episodes $EPISODES --out $OUT/stack $EXTRA
run door $PY eval_door_mllm_vllm.py --api-base $API_BASE --timeout $TIMEOUT --episodes $EPISODES --out $OUT/door $EXTRA
run wipe $PY eval_wipe_mllm_vllm.py --api-base $API_BASE --timeout $TIMEOUT --episodes $EPISODES --out $OUT/wipe $EXTRA
for env in PickPlaceMilk PickPlaceBread PickPlaceCereal PickPlaceCan PickPlace; do
    run pickplace_$env $PY eval_pickplace_mllm_vllm.py --api-base $API_BASE --timeout $TIMEOUT --env $env --episodes $EPISODES --out $OUT/pickplace_$env $EXTRA
done
for env in NutAssemblySquare NutAssemblyRound NutAssembly; do
    run nut_$env $PY eval_nut_mllm_vllm.py --api-base $API_BASE --timeout $TIMEOUT --env $env --episodes $EPISODES --out $OUT/nut_$env $EXTRA
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
