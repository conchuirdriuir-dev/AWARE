"""A removable LoRA graft (the graft rule, 2 Nov: base weights untouched; removing the graft restores the base model
exactly). Written here rather than taken from a library: a few lines, and the switch that turns it off for reading
the senses (`off`) must be exact, which the tests check.

    lora = attach(model, rank=16, alpha=32)       # wraps q, k, v, o, gate, up and down in every layer
    with off(lora):                               # the base model, bit for bit
        ...
    detach(model, lora)                           # the original modules back
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base, self.rank, self.scale, self.enabled = base, rank, alpha / rank, True
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        d = (x.to(self.A.dtype) @ self.A.T) @ self.B.T
        return y + (self.scale * d).to(y.dtype)


def attach(model: nn.Module, rank: int = 16, alpha: float = 32.0, targets=TARGETS) -> dict[str, LoRALinear]:
    """Wrap every target linear layer; the base's own parameters are frozen. Returns name -> wrapper."""
    for p in model.parameters():
        p.requires_grad_(False)
    out = {}
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                w = LoRALinear(child, rank, alpha)
                setattr(module, child_name, w)
                out[f"{name}.{child_name}"] = w
    return out


def detach(model: nn.Module, lora: dict[str, LoRALinear]) -> None:
    for full, w in lora.items():
        parent_name, child_name = full.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child_name, w.base)


@contextlib.contextmanager
def off(lora: dict[str, LoRALinear]):
    before = [w.enabled for w in lora.values()]
    for w in lora.values():
        w.enabled = False
    try:
        yield
    finally:
        for w, b in zip(lora.values(), before):
            w.enabled = b


def parameters(lora: dict[str, LoRALinear]) -> list[nn.Parameter]:
    return [p for w in lora.values() for p in (w.A, w.B)]


def state(lora: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    return {f"{k}.{n}": getattr(w, n).detach().cpu() for k, w in lora.items() for n in ("A", "B")}


def load_state(lora: dict[str, LoRALinear], sd: dict[str, torch.Tensor]) -> None:
    for k, w in lora.items():
        w.A.data.copy_(sd[f"{k}.A"])
        w.B.data.copy_(sd[f"{k}.B"])
