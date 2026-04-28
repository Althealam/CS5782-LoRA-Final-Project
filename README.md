# LoRA on GLUE: Results Summary

This project focuses on evaluating LoRA on GLUE tasks, with three main goals:
- Compare LoRA against Full Fine-tuning and a Frozen Backbone baseline
- Measure parameter efficiency
- Analyze the impact of MNLI hyperparameters (`rank`, `alpha`) and weight merging

## 1) Overall Performance on GLUE

![Performance Comparison](assets/performance%20comparison.png)

Key findings:
- LoRA achieves performance close to, and sometimes better than, Full Fine-tuning on multiple tasks.
- LoRA is the best method on `MRPC`, `QQP`, and `SST-2`.
- Full Fine-tuning is slightly better on `MNLI`, `QNLI`, and `CoLA`, but the gap is small.
- Frozen Backbone consistently underperforms, showing the limits of training only the classifier head.

## 2) Parameter Efficiency

![Parameter Efficiency](assets/parameter%20efficiency.png)

Key findings:
- Full Fine-tuning uses about `124.6M` trainable parameters (100%).
- LoRA uses only about `0.89M` trainable parameters (`0.71%`), a roughly `140x` reduction vs Full.
- Frozen Backbone is even smaller (`0.59M`, `0.47%`) but suffers clear drops in accuracy and F1.
- Overall, LoRA provides the best balance between effectiveness and efficiency.

## 3) LoRA Hyperparameters Used

![Hyperparameters](assets/hyperparameters.png)

Main settings used in this study:
- `rank r = 8`
- `alpha = 16`
- `epochs = 3`
- `batch size = 16`
- `max sequence length = 128`
- `target weights = Query + Value`

## 4) Effect of Rank on MNLI

![Effect of Rank](assets/effect%20of%20rank.png)

Key findings:
- In this sweep, `r=16` performs best (MNLI accuracy `85.94`).
- Results for `r=4/8/32` are very close, suggesting rank has limited impact in this range.
- Increasing rank does not guarantee monotonic improvement.

## 5) Effect of Alpha on MNLI

![Effect of Alpha](assets/effect%20of%20alpha.png)

Key findings:
- Accuracy improves steadily as alpha increases from `8` to `64`.
- `alpha=64` gives the best MNLI accuracy in this experiment (`0.8628`).
- A larger LoRA scaling factor can help, but task-level tuning is still recommended.

## 6) Weight Merging Consistency (MNLI)

![Weight Merging](assets/weight%20merging.png)

Key findings:
- Unmerged and merged models have identical core metrics: `Accuracy=0.8566`, `F1=0.8557`, `Loss=0.3767`.
- This confirms metric consistency before and after LoRA weight merging in this setup.
- Runtime is slightly higher for the merged run in this record (`46.83s` vs `31.20s`), but the main conclusion (metric equivalence) still holds.

## 7) Effect of Target Modules on MNLI

![Effect of Target Modules](assets/lora_target_modules_performance.png)

Key findings:
- Using LoRA on `query` alone gives the weakest result, with `81.43%` accuracy.
- Adding `value` leads to a noticeable jump, bringing accuracy up to `84.92%`.
- The best result comes from adapting `query + value + key + dense`, which reaches `85.92%` accuracy and `85.83%` F1.
- Overall, the pattern is pretty clear: adapting more attention-related modules helps on MNLI, though the gains get smaller as the setup becomes larger.


## 8) Parameter and Runtime Cost of Target Modules

![Target Module Cost](assets/lora_target_modules_efficiency.png)

Key findings:
- `Q` is the lightest setup, with `0.74M` trainable parameters, but it also gives the weakest performance.
- Moving to `Q+V` increases the trainable parameters to `0.89M`, but the improvement in accuracy is large enough to make that tradeoff worthwhile.
- `Q+V+K+D` is the largest of the three settings at `1.18M` trainable parameters, but that is still less than `1%` of the full model.
- Training and evaluation both get a little slower as more modules are added, but the increase is fairly modest.
- In practice, `Q+V` looks like a strong middle ground, while `Q+V+K+D` gives the best MNLI result if performance is the main priority.


## Final Takeaway

- LoRA delivers near-Full-Fine-tuning performance on GLUE with far fewer trainable parameters.
- On MNLI, `alpha` has a clearer impact than `rank` within the tested ranges.
- Metrics remain consistent before and after merging, supporting merged-weight deployment.
