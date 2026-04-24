import argparse
import json
import os
import sys
import time
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
# Get the project root directory so that we can import files from src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lora import count_trainable_parameters, mark_only_lora_and_classifier_trainable, replace_linear_with_lora

# Each GLUE task has different input column names.
# SST-2 uses one sentence, while MRPC and QQP use sentence question pairs.
TASK_TO_KEYS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    # Dataset/model settings
    parser.add_argument("--task", type=str, default="sst2", choices=sorted(TASK_TO_KEYS))
    parser.add_argument("--model_name", type=str, default="roberta-base")
    
    # Training method: baseline frozen, full, or LoRA
    parser.add_argument("--method", type=str, default="lora", choices=["frozen", "full", "lora"])
    # Output and hyperparameters
    parser.add_argument("--output_dir", type=str, default="results/sst2_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    # LoRA-specific hyperparameters
    # lora_rank controls the low-rank dimension r
    # lora_alpha controls the scaling of the LoRA update
    # lora_targets decides which linear modules are replaced by LoRA
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_targets", nargs="+", default=["query", "value"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for more reproducible experiments.
    torch.manual_seed(args.seed)
    
    # Load the selected GLUE dataset from Hugging Face.
    # For example, task="sst2"
    dataset = load_dataset("glue", args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    metric = evaluate.load("glue", args.task)
    
    # binary classification tasks.
    num_labels = 2
    # load a pretrained RoBERTa model
    # sequence classification head.
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    if args.method == "frozen":
        freeze_all_but_classifier(model)        # Frozen baseline: only the classifier head is trainable.
    elif args.method == "lora": # LoRA method: replace selected linear layers with LoRA versions.
        replace_linear_with_lora(
            model,
            target_module_names=args.lora_targets,
            r=args.lora_rank,
            alpha=args.lora_alpha,
        )        
        
        # After inserting LoRA modules, freeze the original backbone weights.
        # Only LoRA parameters and the classifier head remain trainable.
        mark_only_lora_and_classifier_trainable(model)

    sentence1_key, sentence2_key = TASK_TO_KEYS[args.task]    # Get the correct input column names for the selected task.

    def tokenize(batch):
        return tokenizer(
            batch[sentence1_key],
            batch[sentence2_key] if sentence2_key else None,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")    # Hugging Face Trainer expects the target column to be named "labels".
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        seed=args.seed,
        report_to="none",
    )    # TrainingArguments controls the Trainer's optimization/evaluation behavior.
    

    trainer = Trainer(    # Trainer handles the training loop, evaluation loop, checkpoint, and logging.
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, args.task, metric),
    )

    start_time = time.time()    # Measure training time so that we can compare efficiency across methods.
    trainer.train()
    runtime_seconds = time.time() - start_time
    eval_metrics = trainer.evaluate()    # Evaluate the final model on the validation split.

    trainable, total = count_trainable_parameters(model)     # Count trainable parameters to compare LoRA with full fine-tuning.
    result = {
        "task": args.task,
        "method": args.method,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "runtime_seconds": runtime_seconds,
        "eval_metrics": eval_metrics,
        "lora_targets": args.lora_targets,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


def compute_metrics(eval_pred, task_name, glue_metric):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    results = {"accuracy": accuracy_score(labels, preds)}
    if task_name == "mrpc":    # MRPC commonly reports both accuracy and F1.
        results["f1"] = f1_score(labels, preds)

    try:
        results.update(glue_metric.compute(predictions=preds, references=labels))
    except Exception:
        pass

    return results


def freeze_all_but_classifier(model) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False
    # First freeze every parameter. Then unfreeze classifier parameters only.
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True


if __name__ == "__main__":
    main()
