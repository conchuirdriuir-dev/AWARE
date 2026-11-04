"""Peak GPU memory of one LoRA training step, by sequence length (Body Readout spec v0.2, open question "training
memory"; PLAN.md, "Babbling: training memory on 24 GB"). The probe that produced that section's numbers.

    python probes/lora_memory.py --model Qwen/Qwen3-8B --layers 2 --loss-tokens 64 --lengths 2048,4096,8192 --lora-compute bf16

--layers truncates the real architecture so the 8B's per-token costs can be measured on a small card: the gap between
two layer counts is the stored checkpoint per layer; the rest is one layer's recompute in the backward pass.

LoRA rank r on q,k,v,o,gate,up,down of every layer, base frozen, gradient checkpointing per layer, the loss's logits
computed only on the last `loss_tokens` positions (the actor's own steps are a small share of an episode).
Prints one JSON line per length: peak allocated and reserved, in GiB.
"""
import argparse, json, math, sys
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--four-bit", action="store_true")
p.add_argument("--lengths", default="512,1024,2048,4096")
p.add_argument("--rank", type=int, default=16)
p.add_argument("--loss-tokens", type=int, default=512)
p.add_argument("--no-checkpointing", action="store_true")
p.add_argument("--layers", type=int)
p.add_argument("--lora-compute", choices=["fp32", "bf16"], default="fp32")
a = p.parse_args()


class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__()
        self.base = base
        dev = next(base.parameters()).device
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))

    def forward(self, x):
        if a.lora_compute == "bf16":                       # fp32 master weights, the product in the input's dtype
            return self.base(x) + x @ self.A.t().to(x.dtype) @ self.B.t().to(x.dtype)
        return self.base(x) + (x.to(self.A.dtype) @ self.A.t() @ self.B.t()).to(x.dtype)


kw = dict(dtype=torch.bfloat16, device_map={"": 0})
if a.four_bit:
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
if a.layers:
    kw["num_hidden_layers"] = a.layers
model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
model.config.use_cache = False
for q in model.parameters():
    q.requires_grad_(False)
names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
for layer in model.model.layers:
    for parent in (layer.self_attn, layer.mlp):
        for n in names:
            if hasattr(parent, n):
                setattr(parent, n, LoRA(getattr(parent, n), a.rank))
lora = [q for q in model.parameters() if q.requires_grad]
if not a.no_checkpointing:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()
opt = torch.optim.AdamW(lora, lr=1e-4)
torch.cuda.synchronize()
base_alloc = torch.cuda.memory_allocated() / 2**30
print(json.dumps({"model": a.model, "four_bit": a.four_bit, "layers": model.config.num_hidden_layers, "lora_compute": a.lora_compute, "lora_params_M": round(sum(q.numel() for q in lora) / 1e6, 1),
                  "after_load_GiB": round(base_alloc, 2)}), flush=True)
V = model.config.vocab_size
for T in [int(x) for x in a.lengths.split(",")]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        ids = torch.randint(0, V, (1, T), device=0)
        h = model.model(input_ids=ids).last_hidden_state[:, -a.loss_tokens:]
        logits = model.lm_head(h).float()
        loss = nn.functional.cross_entropy(logits[0, :-1], ids[0, -a.loss_tokens + 1:])
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        out = {"T": T, "peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
               "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 3)}
    except torch.cuda.OutOfMemoryError:
        out = {"T": T, "oom": True}
    del ids
    h = logits = loss = None
    print(json.dumps(out), flush=True)
