# 所有任务写进同一个目录。图片和样本 id 带任务名前缀，samples.jsonl 按 task 合并。
set -e
PY=/opt/conda/envs/python3.10.13/bin/python3
OUT=/data/oss_bucket_3/guotong/mllm_data/
$PY collect_stack_mllm.py --episodes 50 --out $OUT
$PY collect_lift_mllm.py --episodes 50 --out $OUT
for env in PickPlaceMilk PickPlaceBread PickPlaceCereal PickPlaceCan PickPlace; do
  $PY collect_pickplace_mllm.py --env $env --episodes 50 --out $OUT
done
for env in NutAssemblySquare NutAssemblyRound NutAssembly; do
  $PY collect_nut_mllm.py --env $env --episodes 50 --out $OUT
done
$PY collect_door_mllm.py --episodes 50 --out $OUT
$PY collect_wipe_mllm.py --episodes 50 --out $OUT
