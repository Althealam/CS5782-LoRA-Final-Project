# Proposal Draft

## Title
Re-implementation of LoRA: Low-Rank Adaptation of Large Language Models

## Group Members
- Yifei Zhang
- Kahei Lam

## Paper
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.

## Brief Summary of the Paper
This paper studies parameter-efficient fine-tuning for large pretrained language models. Instead of updating all pretrained weights during downstream adaptation, LoRA freezes the original model parameters and learns a low-rank update for selected weight matrices. Concretely, for a pretrained linear layer with weight matrix W, LoRA represents the update as BA where B and A are trainable low-rank matrices and the original weight W remains frozen. This greatly reduces the number of trainable parameters and optimizer states while preserving performance close to full fine-tuning.

## Why We Chose This Paper
We chose this paper for three reasons. First, the core idea is conceptually simple and suitable for a faithful re-implementation in a course project. Second, the method can be evaluated on publicly available benchmarks without requiring very large-scale infrastructure. Third, LoRA remains one of the most influential parameter-efficient fine-tuning methods, so reproducing it gives both practical engineering experience and a good understanding of modern LLM adaptation methods.

## Project Goal
Our goal is to reproduce the central empirical finding of the LoRA paper: LoRA can achieve performance close to full fine-tuning while using far fewer trainable parameters. To keep the project feasible on limited compute, we will focus on RoBERTa-base and GLUE text classification tasks rather than the larger GPT-3 setting in the original paper.

## Chosen Results to Reproduce
We will reproduce a scaled but faithful version of the paper's RoBERTa experiments on GLUE.

Primary target:
- Compare frozen-backbone baseline, full fine-tuning, and LoRA on SST-2.

Secondary target:
- Extend the comparison to one or two additional GLUE tasks such as MRPC and QQP.

Optional extension:
- Study the effect of LoRA rank r and target attention weights on MNLI or SST-2, inspired by the paper's rank analysis.

## Data
We will use datasets from the GLUE benchmark through the Hugging Face `datasets` library.

Primary dataset:
- SST-2 for sentiment classification

Additional datasets:
- MRPC for paraphrase classification
- QQP for duplicate question detection

These datasets are public, easy to download in Colab, and widely used in the original paper's RoBERTa evaluation.

## Re-implementation Plan
We will implement the LoRA module ourselves rather than relying on an external PEFT package for the core method. Our implementation plan is:

1. Start from a pretrained `roberta-base` sequence classification model.
2. Implement a custom `LoRALinear` layer with frozen base weights and trainable low-rank matrices A and B.
3. Replace selected self-attention projection layers in RoBERTa, beginning with query and value projections.
4. Train and compare three settings:
   - Frozen backbone with only the classification head trained
   - Full fine-tuning of all parameters
   - LoRA with frozen backbone and trainable LoRA parameters plus classification head
5. Measure accuracy or F1 on validation/test splits and compare trainable parameter counts, training time, and GPU memory usage.

## Metrics
We will evaluate both task performance and efficiency.

Task metrics:
- Accuracy for SST-2 and QQP
- Accuracy and F1 for MRPC if included

Efficiency metrics:
- Number of trainable parameters
- Training time per epoch or total runtime
- Peak GPU memory usage in Colab

## Expected Outcome
We expect LoRA to significantly reduce the number of trainable parameters while achieving performance reasonably close to full fine-tuning. We also expect LoRA to outperform the frozen-backbone baseline by a meaningful margin.

## Feasibility
This project is feasible in Google Colab using a single T4, L4, or A100 runtime. RoBERTa-base with short sequence lengths and GLUE datasets can be trained within the limits of a course project. The implementation scope is also manageable because we only need to modify linear layers inside the attention blocks rather than re-implement the entire Transformer.

## Timeline
Week 1:
- Read the LoRA paper and inspect prior implementations
- Set up GitHub repo and Colab environment
- Implement baseline training pipeline on SST-2

Week 2:
- Implement LoRA and verify trainable parameter counts
- Run frozen-head, full fine-tuning, and LoRA experiments on SST-2

Week 3:
- Run experiments on MRPC and/or QQP
- Add rank ablation if time permits
- Collect plots, summarize results, and write report
