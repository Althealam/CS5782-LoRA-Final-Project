#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/CS5782-LoRA-Final-Project
PY=$PROJECT/lora_env/bin/python
OUT=/root/autodl-tmp/exp_merge_mnli
LOG=$PROJECT/log/merge_exp
TASK=mnli
MODEL=roberta-base
RANK=8
ALPHA=16
BS=32
LR=1e-4
EPOCHS=3
SEED=42

mkdir -p "$OUT" "$LOG"
cd "$PROJECT"

# 1) train + get merged model
$PY -u scripts/train_glue.py \
  --task "$TASK" \
  --method lora \
  --model_name "$MODEL" \
  --lora_rank "$RANK" \
  --lora_alpha "$ALPHA" \
  --batch_size "$BS" \
  --learning_rate "$LR" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --output_dir "$OUT/lora_unmerged" \
  --merge_adapter \
  --merged_output_dir "$OUT/lora_merged" \
  > "$LOG/train_and_merge.log" 2>&1

# 2) evaluate unmerged (only evaluation, no training)
$PY -u scripts/train_glue.py \
  --task "$TASK" \
  --method lora \
  --model_name "$OUT/lora_unmerged" \
  --lora_rank "$RANK" \
  --lora_alpha "$ALPHA" \
  --batch_size "$BS" \
  --learning_rate "$LR" \
  --epochs 0 \
  --seed "$SEED" \
  --output_dir "$OUT/eval_unmerged" \
  > "$LOG/eval_unmerged.log" 2>&1

# 3) evaluate merged (only evaluation, no training)
$PY -u scripts/train_glue.py \
  --task "$TASK" \
  --method full \
  --model_name "$OUT/lora_merged" \
  --batch_size "$BS" \
  --learning_rate "$LR" \
  --epochs 0 \
  --seed "$SEED" \
  --output_dir "$OUT/eval_merged" \
  > "$LOG/eval_merged.log" 2>&1

echo "DONE"
echo "train log: $LOG/train_and_merge.log"
echo "unmerged eval log: $LOG/eval_unmerged.log"
echo "merged eval log: $LOG/eval_merged.log"
echo "results:"
echo "  $OUT/eval_unmerged/summary.json"
echo "  $OUT/eval_merged/summary.json"