## Log
### Frozen
1. SST2: python scripts/train_glue.py --task sst2 --method frozen --output_dir results/sst2_frozen
{
  "task": "sst2",
  "method": "frozen",
  "trainable_parameters": 592130,
  "total_parameters": 124647170,
  "runtime_seconds": 383.67638087272644,
  "eval_metrics": {
    "eval_loss": 0.46183428168296814,
    "eval_accuracy": 0.8061926605504587,
    "eval_runtime": 1.2416,
    "eval_samples_per_second": 702.338,
    "eval_steps_per_second": 11.276,
    "epoch": 3.0
  },
  "lora_targets": [
    "query",
    "value"
  ],
  "lora_rank": 8,
  "lora_alpha": 16
}

2. mrpc: python scripts/train_glue.py --task mrpc --method frozen --output_dir results/mrpc_frozen
{
  "task": "mrpc",
  "method": "frozen",
  "trainable_parameters": 592130,
  "total_parameters": 124647170,
  "runtime_seconds": 24.861077547073364,
  "eval_metrics": {
    "eval_loss": 0.617306649684906,
    "eval_accuracy": 0.6838235294117647,
    "eval_f1": 0.8122270742358079,
    "eval_runtime": 0.6039,
    "eval_samples_per_second": 675.662,
    "eval_steps_per_second": 11.592,
    "epoch": 3.0
  },
  "lora_targets": [
    "query",
    "value"
  ],
  "lora_rank": 8,
  "lora_alpha": 16
}

3. qqp: python scripts/train_glue.py --task qqp --method frozen --output_dir results/qqp_frozen



### Full Fine-tune
1. SST2: python scripts/train_glue.py --task sst2 --method full --output_dir results/sst2_full
2. mrpc: python scripts/train_glue.py --task mrpc --method full --output_dir results/mrpc_full
{
  "task": "mrpc",
  "method": "full",
  "trainable_parameters": 124647170,
  "total_parameters": 124647170,
  "runtime_seconds": 173.254154920578,
  "eval_metrics": {
    "eval_loss": 0.5415898561477661,
    "eval_accuracy": 0.8725490196078431,
    "eval_f1": 0.9097222222222222,
    "eval_runtime": 1.7085,
    "eval_samples_per_second": 238.8,
    "eval_steps_per_second": 4.097,
    "epoch": 3.0
  },
  "lora_targets": [
    "query",
    "value"
  ],
  "lora_rank": 8,
  "lora_alpha": 16
}
3. qqp: python scripts/train_glue.py --task qqp --method full --output_dir results/qqp_full