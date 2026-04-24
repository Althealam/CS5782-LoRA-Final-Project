import math
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive.")

        self.r = r
        self.alpha = alpha if alpha is not None else r
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(self.dropout(x))) * self.scaling


def replace_linear_with_lora(
    model: nn.Module,
    target_module_names: Iterable[str] = ("query", "value"),
    r: int = 8,
    alpha: int | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    target_module_names = set(target_module_names)

    for name, module in model.named_modules():
        if "encoder.layer" not in name or not hasattr(module, "attention"):
            continue

        attention = module.attention
        self_attn = attention.self

        if "query" in target_module_names and hasattr(self_attn, "query"):
            self_attn.query = _copy_linear_to_lora(self_attn.query, r, alpha, dropout)

        if "key" in target_module_names and hasattr(self_attn, "key"):
            self_attn.key = _copy_linear_to_lora(self_attn.key, r, alpha, dropout)

        if "value" in target_module_names and hasattr(self_attn, "value"):
            self_attn.value = _copy_linear_to_lora(self_attn.value, r, alpha, dropout)

        if "dense" in target_module_names and hasattr(attention.output, "dense"):
            attention.output.dense = _copy_linear_to_lora(attention.output.dense, r, alpha, dropout)

    return model


def mark_only_lora_and_classifier_trainable(model: nn.Module) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if ".A.weight" in name or ".B.weight" in name:
            param.requires_grad = True
        if "classifier" in name:
            param.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def _copy_linear_to_lora(old_linear: nn.Linear, r: int, alpha: int | None, dropout: float) -> LoRALinear:
    new_linear = LoRALinear(
        in_features=old_linear.in_features,
        out_features=old_linear.out_features,
        r=r,
        alpha=alpha,
        dropout=dropout,
        bias=old_linear.bias is not None,
    )
    new_linear.base.weight.data.copy_(old_linear.weight.data)
    if old_linear.bias is not None and new_linear.base.bias is not None:
        new_linear.base.bias.data.copy_(old_linear.bias.data)
    return new_linear
