# LoRA on GLUE: Re-Implementation Summary

![Course](https://img.shields.io/badge/CS5782-Final%20Project-6f42c1?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-Reproduction%20%26%20Analysis-0ea5e9?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-BERT--base%20%2B%20LoRA-16a34a?style=for-the-badge)

This project evaluates **LoRA (Low-Rank Adaptation)** on GLUE with three goals:
- Compare `LoRA` vs `Full Fine-tuning` vs `Frozen Backbone`.
- Measure the **performance-efficiency tradeoff**.
- Study MNLI sensitivity to LoRA settings (`rank`, `alpha`, target modules, and merge behavior).

---

## Quick Navigation
- [A) Results Snapshot](#a-results-snapshot)
- [B) 2-Page Report Writing Guide (Based on Course Requirement)](#b-2-page-report-writing-guide-based-on-course-requirement)
- [C) Reproduction Workflow](#c-reproduction-workflow)
- [D) Final Takeaways](#d-final-takeaways)

---

## A) Results Snapshot

### 1) Overall Performance on GLUE

![Performance Comparison](assets/performance%20comparison.png)

Key findings:
- LoRA is close to, and sometimes better than, Full Fine-tuning on multiple tasks.
- LoRA is best on `MRPC`, `QQP`, and `SST-2`.
- Full Fine-tuning is slightly better on `MNLI`, `QNLI`, and `CoLA`, but margins are small.
- Frozen Backbone underperforms, showing the limit of training only the classifier head.

### 2) Parameter Efficiency

![Parameter Efficiency](assets/parameter%20efficiency.png)

Key findings:
- Full Fine-tuning: `124.6M` trainable params (100%).
- LoRA: `0.89M` trainable params (`0.71%`), around `140x` fewer than Full.
- Frozen Backbone: `0.59M` (`0.47%`) but with clear performance drop.
- LoRA gives the strongest balance of effectiveness and efficiency.

### 3) LoRA Hyperparameters Used

![Hyperparameters](assets/hyperparameters.png)

Main settings:
- `rank r = 8`
- `alpha = 16`
- `epochs = 3`
- `batch size = 16`
- `max sequence length = 128`
- `target modules = query + value`

### 4) Effect of Rank on MNLI

![Effect of Rank](assets/effect%20of%20rank.png)

Key findings:
- `r=16` performs best in this sweep (MNLI accuracy `85.94`).
- `r=4/8/32` are very close, so rank impact is limited in this range.
- Larger rank does not guarantee monotonic gains.

### 5) Effect of Alpha on MNLI

![Effect of Alpha](assets/effect%20of%20alpha.png)

Key findings:
- Accuracy improves as alpha increases from `8` to `64`.
- `alpha=64` is best in this experiment (`0.8628`).
- Scaling factor matters, but per-task tuning is still important.

### 6) Weight Merging Consistency (MNLI)

![Weight Merging](assets/weight%20merging.png)

Key findings:
- Unmerged and merged models produce identical core metrics: `Accuracy=0.8566`, `F1=0.8557`, `Loss=0.3767`.
- This supports metric consistency before/after LoRA merge.
- Runtime is slightly higher for merged in this record (`46.83s` vs `31.20s`), but the main conclusion is unchanged.

### 7) Effect of Target Modules on MNLI

![Effect of Target Modules](assets/lora_target_modules_performance.png)

Key findings:
- `query` only is weakest (`81.43%` accuracy).
- Adding `value` gives a strong improvement (`84.92%`).
- Best result uses `query + value + key + dense` (`85.92%` accuracy, `85.83%` F1).
- Adapting more attention-related modules helps, with diminishing returns.

### 8) Parameter and Runtime Cost of Target Modules

![Target Module Cost](assets/lora_target_modules_efficiency.png)

Key findings:
- `Q` is lightest (`0.74M`) but weakest in performance.
- `Q+V` increases params to `0.89M` with meaningful accuracy gain.
- `Q+V+K+D` is largest (`1.18M`) but still under `1%` of full model.
- Runtime increases modestly with more adapted modules.
- In practice: `Q+V` is a strong middle ground; `Q+V+K+D` is best if quality is the priority.

---

## B) 2-Page Report Writing Guide (Based on Course Requirement)

> For your submission, emphasize **Methodology**, **Results & Analysis**, and **Reflections**.

### 1. Introduction (about 10-15%)
- Problem: parameter-efficient adaptation of large language models for sentence-level tasks.
- Motivation: reach near full fine-tuning quality with far fewer trainable parameters.
- Paper context: include original paper title/authors and the method contribution (LoRA as low-rank adapters).
- One-sentence project scope: "We re-implemented LoRA on GLUE and analyzed sensitivity on MNLI."

### 2. Chosen Result (about 10-15%)
- Chosen result suggestion: **LoRA vs Full Fine-tuning tradeoff** on GLUE + MNLI ablations.
- Why this result: it directly tests the main claim that LoRA preserves performance while reducing trainable parameters.
- Link to original paper artifact: cite the relevant figure/table from the paper that reports task-level quality and efficiency.

### 3. Methodology (about 25-30%)  **<- focus**
- Model: base transformer backbone + LoRA modules on selected attention/projection layers.
- Dataset: GLUE benchmark; highlight MNLI for controlled ablation.
- Metrics: accuracy/F1/loss; add runtime/parameter count for efficiency.
- Experimental settings: rank/alpha sweeps, module-target sweeps, merge-vs-unmerged validation.
- Repro details: seed strategy, hardware constraints, and deviations from original setup.

### 4. Results & Analysis (about 30-35%)  **<- focus**
- Compare reproduction results to paper trends, not only absolute numbers.
- Explain discrepancy sources:
  - scale differences (budget/hardware),
  - implementation detail gaps,
  - tuning scope and random seed variance.
- Highlight robust findings from your runs:
  - LoRA has strong efficiency gains with small quality loss,
  - alpha is more sensitive than rank in tested ranges,
  - weight merging preserves metrics.
- Include at least one visual that couples quality and cost.

### 5. Reflections (about 15-20%)  **<- focus**
- Lessons learned:
  - where reproduction is fragile,
  - which hyperparameters matter most,
  - what design decisions improved reliability.
- What you would do next:
  - larger-scale sweep,
  - additional tasks/models,
  - better regularization or search strategies.
- Tie back to broader implication: practical fine-tuning for constrained compute.

### 6. References
- Original LoRA paper (full citation).
- Any toolkit/frameworks (`Transformers`, `PEFT`, `PyTorch`, GLUE).
- Dataset and benchmark references.

---

## C) Reproduction Workflow

```mermaid
flowchart LR
    A[Select Baselines<br/>Full / LoRA / Frozen] --> B[Train on GLUE]
    B --> C[Run MNLI Ablations<br/>rank, alpha, target modules]
    C --> D[Evaluate Accuracy/F1/Loss]
    D --> E[Compute Cost Metrics<br/>Params + Runtime]
    E --> F[Analyze Tradeoffs & Reflections]
```

---

## D) Final Takeaways

- LoRA delivers near-Full Fine-tuning performance on GLUE with dramatically fewer trainable parameters.
- On MNLI, `alpha` shows a clearer impact than `rank` in the tested ranges.
- Merged and unmerged LoRA weights are metric-consistent in this setup, supporting deployment with merged weights.
