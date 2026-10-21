#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs/logs artifacts/training

ir_seeds=(274 282 283)
no_ir_seeds=(287 288 289)
transition_samples=604569600
pids=()

for index in 0 1 2; do
  gpu="${index}"
  seed="${ir_seeds[$index]}"
  base_run="runs/fig7a_ir_seed${seed}_base"
  stable_run="runs/fig7a_ir_seed${seed}_stable"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" python3 -m sensorimotor.train_teacher \
      --config configs/single_gap_roll.yaml \
      --seed "${seed}" \
      --model-seed 274 \
      --total-samples "${transition_samples}" \
      --run-dir "${base_run}"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 -m sensorimotor.train_teacher \
      --config configs/single_gap_roll.yaml \
      --seed "${seed}" \
      --resume "${base_run}/checkpoints/teacher_final.pt" \
      --total-samples 1000000000 \
      --actor-learning-rate 0.0001 \
      --normalizer-freeze-samples 0 \
      --run-dir "${stable_run}"
  ) > "runs/logs/fig7a_ir_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

for index in 0 1 2; do
  gpu=$((index + 3))
  seed="${no_ir_seeds[$index]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -m sensorimotor.train_teacher \
    --config configs/single_gap_roll_no_ir.yaml \
    --seed "${seed}" \
    --run-dir "runs/fig7a_no_ir_seed${seed}" \
    > "runs/logs/fig7a_no_ir_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if ((status == 0)); then
  python3 scripts/plot_fig7a_reproduction.py \
    --ir-runs \
      runs/fig7a_ir_seed274_base,runs/fig7a_ir_seed274_stable \
      runs/fig7a_ir_seed282_base,runs/fig7a_ir_seed282_stable \
      runs/fig7a_ir_seed283_base,runs/fig7a_ir_seed283_stable \
    --no-ir-runs \
      runs/fig7a_no_ir_seed287 \
      runs/fig7a_no_ir_seed288 \
      runs/fig7a_no_ir_seed289 \
    --output artifacts/training/fig7a_reproduced_final.png \
    --csv-output artifacts/training/fig7a_reproduced_final.csv
fi

exit "${status}"
