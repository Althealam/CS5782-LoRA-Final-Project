# Execution Plan

## Final Scope
Use `roberta-base` on GLUE to reproduce the core LoRA idea with a self-implemented LoRA module.

Required experiments:
- Frozen backbone + classifier head
- Full fine-tuning
- LoRA fine-tuning

Required dataset:
- SST-2

Recommended extra dataset:
- MRPC or QQP

Optional extension:
- Rank ablation with `r in {1, 2, 4, 8}`
- Compare `query` vs `query+value`

## Recommended Timeline

### Day 1
- Create GitHub repository
- Create Colab notebook
- Clone repo in Colab
- Install dependencies
- Verify GPU runtime

### Day 2
- Load SST-2 with Hugging Face `datasets`
- Run a frozen-backbone baseline
- Log accuracy, runtime, and trainable parameters

### Day 3
- Run full fine-tuning baseline
- Save checkpoints and metrics
- Compare with frozen baseline

### Day 4
- Implement `LoRALinear`
- Patch RoBERTa attention layers
- Verify only LoRA weights and classifier head are trainable

### Day 5
- Run LoRA on SST-2
- Compare against full fine-tuning
- Record memory use and runtime

### Day 6
- Repeat on MRPC or QQP
- Decide whether results are strong enough

### Day 7
- If time allows, run rank ablation
- Start report tables and plots

## Deliverables Checklist
- Working GitHub repository
- Colab notebook or scripts that reproduce the experiments
- Saved experiment outputs
- Table comparing the three methods
- Short explanation of LoRA math
- Final report and slides/poster

## Minimal Success Criterion
Even if time gets tight, you can still finish a good project if you complete:
- One dataset: SST-2
- Three methods: frozen, full FT, LoRA
- Three comparisons: accuracy, trainable params, runtime

That is already a coherent and defensible final project.
