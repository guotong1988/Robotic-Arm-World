#!/usr/bin/env bash
# 每个任务再拆成 WORKERS 个独立分片并行采集；全部成功后校验合并，残缺分片不发布。
# 坐标模式用 pixel（默认）或 xyz 区分；XYZ 是机器人基座整数厘米，默认输出目录与像素数据隔离。
# 直接运行会整体转到后台（nohup）；调试时用 bash collect.sh xyz --fg 前台执行。
set -e
PY=${PY:-/opt/conda/envs/python3.10.13/bin/python3}
COORD_MODE=${COORD_MODE:-pixel}
FG=0
for arg in "$@"; do
    case "$arg" in
        pixel|xyz) COORD_MODE=$arg ;;
        --fg) FG=1 ;;
        *) echo "用法: bash collect.sh [pixel|xyz] [--fg]" >&2; exit 2 ;;
    esac
done
if [ "$COORD_MODE" = "xyz" ]; then
    DEFAULT_OUT=/data/oss_bucket_3/guotong/mllm_data_xyz/
else
    DEFAULT_OUT=/data/oss_bucket_3/guotong/mllm_data/
fi
OUT=${OUT:-$DEFAULT_OUT}
EPISODES=${EPISODES:-1000}
WORKERS=${WORKERS:-24}
MAX_WORKERS=${MAX_WORKERS:-$WORKERS}
SEED=${SEED:-0}
IMAGE_SIZE=${IMAGE_SIZE:-1024}
SHARDS=${SHARDS:-$OUT/.collect_shards}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
LOG=$OUT/logs
ensure_log_dir() {
    if ! mkdir -p "$LOG"; then
        echo "无法创建日志目录 $LOG，请检查输出目录或 OSS 挂载" >&2
        exit 1
    fi
}
ensure_log_dir
mkdir -p "$SHARDS"

case $EPISODES in ''|*[!0-9]*) echo "EPISODES 必须是正整数" >&2; exit 2;; esac
case $WORKERS in ''|*[!0-9]*) echo "WORKERS 必须是正整数" >&2; exit 2;; esac
case $MAX_WORKERS in ''|*[!0-9]*) echo "MAX_WORKERS 必须是正整数" >&2; exit 2;; esac
case $IMAGE_SIZE in ''|*[!0-9]*) echo "IMAGE_SIZE 必须是正整数" >&2; exit 2;; esac
if (( EPISODES < 1 || WORKERS < 1 || MAX_WORKERS < 1 || IMAGE_SIZE < 1 )); then
    echo "EPISODES、WORKERS、MAX_WORKERS 和 IMAGE_SIZE 必须大于 0" >&2
    exit 2
fi
if (( WORKERS > EPISODES )); then
    WORKERS=$EPISODES
fi

if [ "$FG" -ne 1 ]; then
    ensure_log_dir
    nohup env PY="$PY" OUT="$OUT" EPISODES="$EPISODES" WORKERS="$WORKERS" SEED="$SEED" \
        IMAGE_SIZE="$IMAGE_SIZE" SHARDS="$SHARDS" COORD_MODE="$COORD_MODE" MAX_WORKERS="$MAX_WORKERS" \
        OMP_NUM_THREADS="$OMP_NUM_THREADS" \
        MKL_NUM_THREADS="$MKL_NUM_THREADS" OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
        NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" bash "$0" "$COORD_MODE" --fg > "$LOG/collect.log" 2>&1 &
    echo "后台执行 pid $!，坐标模式 $COORD_MODE，每任务 $WORKERS 个分片，全局最多 $MAX_WORKERS 个 worker，每任务采满 $EPISODES 条才发布"
    echo "总日志 $LOG/collect.log，各 worker 日志在 $LOG"
    exit 0
fi

pids=()
pid_tasks=()
tasks=()
run() {
    name=$1
    shift
    # WORKERS 是每任务分片数；这里限制所有任务合计的运行进程数，避免 CPU 和软件渲染过载。
    while (( $(jobs -pr | wc -l) >= MAX_WORKERS )); do
        sleep 0.2
    done
    ensure_log_dir
    "$@" > "$LOG/$name.log" 2>&1 &
    pids+=($!)
}

launch_task() {
    task=$1
    log_name=$2
    shift 2
    # 启动本任务前清掉旧分片，避免上次采集残留混入本次合并。
    rm -rf -- "$SHARDS/$task"
    mkdir -p "$SHARDS/$task"
    tasks+=("$task")
    base=$((EPISODES / WORKERS))
    extra=$((EPISODES % WORKERS))
    for (( worker=0; worker<WORKERS; worker++ )); do
        count=$base
        if (( worker < extra )); then
            count=$((count + 1))
        fi
        shard=$SHARDS/$task/worker_$worker
        name=${log_name}.worker_$worker
        run "$name" "$@" --episodes "$count" --episode-offset "$worker" \
            --episode-stride "$WORKERS" --seed "$((SEED + worker))" --out "$shard" \
            --image-size "$IMAGE_SIZE" --coord-mode "$COORD_MODE"
        pid_tasks+=("$task")
    done
}

launch_task stack stack "$PY" collect_stack_mllm.py
launch_task lift lift "$PY" collect_lift_mllm.py
for env in PickPlaceMilk PickPlaceBread PickPlaceCereal PickPlaceCan PickPlace; do
    task=$(printf '%s' "$env" | tr '[:upper:]' '[:lower:]')
    launch_task "$task" pickplace_$env "$PY" collect_pickplace_mllm.py --env "$env"
done
for env in NutAssemblySquare NutAssemblyRound NutAssembly; do
    task=$(printf '%s' "$env" | tr '[:upper:]' '[:lower:]')
    launch_task "$task" nut_$env "$PY" collect_nut_mllm.py --env "$env"
done
launch_task door door "$PY" collect_door_mllm.py
launch_task wipe wipe "$PY" collect_wipe_mllm.py

fail=0
failed_tasks=" "
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        failed_tasks+="${pid_tasks[$index]} "
        fail=1
    fi
done

for task in "${tasks[@]}"; do
    case $failed_tasks in
        *" $task "*)
            echo "$task 有 worker 失败，跳过发布并保留原有数据" >&2
            continue
            ;;
    esac
    ensure_log_dir
    shards=()
    for (( worker=0; worker<WORKERS; worker++ )); do
        shards+=("$SHARDS/$task/worker_$worker")
    done
    if ! "$PY" mllm_dataset.py merge-task --out "$OUT" --task "$task" \
        --episodes "$EPISODES" --shards "${shards[@]}" > "$LOG/$task.merge.log" 2>&1; then
        echo "$task 分片校验或合并失败，详见 $LOG/$task.merge.log" >&2
        fail=1
    else
        echo "$task 已采满并发布 $EPISODES 条成功轨迹"
    fi
done
exit $fail
