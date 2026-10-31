"""Reading the model's states (PLAN.md, step 1): the residual stream after chosen layers, mean-pooled over a text's
tokens, through nnterp's standardized layer names so the same code reads Llama and Qwen.

Special tokens are left out of the mean: Llama's begin-of-text token carries a state far larger than any other and
would swamp it. States are derived, never the record; `info` is what the record keeps so they can be rebuilt,
including how the model was loaded: where its layers sit, in what precision, and whether it was quantised (4-bit is a
quick check only: it adds noise to D).

Loading the 8B on a card smaller than its weights: `device_map="auto"` with `gpu_memory` (GiB) puts as many layers
on the GPU as fit and the rest in system RAM. Slower, but exact.
"""
from __future__ import annotations

import numpy as np


class Reader:
    def __init__(self, model, tokenizer=None, revision: str | None = None, device: str | None = None,
                 dtype: str | None = None, gpu_memory: float | None = None, quantize: str | None = None):
        """`model` is a Hugging Face name, or an already-built model with its `tokenizer` (the tests' tiny one).
        `device`: "cuda", "cpu" or "auto" (split across GPU and RAM, capped at `gpu_memory` GiB on the GPU).
        `dtype`: "bfloat16" or "float32" (default: bfloat16 on a GPU, float32 on CPU). `quantize`: None, "4bit" or
        "8bit" (needs bitsandbytes and a GPU)."""
        import torch
        from nnterp import StandardizedTransformer

        self.load = {"device": device, "dtype": dtype, "gpu_memory": gpu_memory, "quantize": quantize}
        if isinstance(model, str):
            cuda = torch.cuda.is_available()
            kwargs = {"revision": revision,
                      "dtype": getattr(torch, dtype) if dtype else (torch.bfloat16 if cuda else torch.float32),
                      "device_map": device or ("cuda" if cuda else "cpu")}
            if device == "auto" and gpu_memory is not None:
                kwargs["max_memory"] = {0: f"{gpu_memory}GiB", "cpu": "64GiB"}
            if quantize is not None:
                if not cuda:
                    raise ValueError("quantised loading needs a GPU (bitsandbytes)")
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                                                 if quantize == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
            self.model = StandardizedTransformer(model, **kwargs)
            self.name = model
        else:
            self.model = StandardizedTransformer(model, tokenizer=tokenizer)
            self.name = getattr(model.config, "name_or_path", "") or type(model).__name__
        self.tokenizer = self.model.tokenizer
        self.special = torch.tensor(sorted(set(self.tokenizer.all_special_ids)))
        config = self.model._model.config        # nnsight leaves .config unset for a model passed in built
        self.info = {"model": self.name, "revision": revision,
                     "commit": getattr(config, "_commit_hash", None),
                     "layers": self.model.num_layers, "width": config.hidden_size,
                     "dtype": str(next(self.model._model.parameters()).dtype), "load": self.load}

    def states(self, texts: list[str], layers: list[int], skip_first: bool = True) -> np.ndarray:
        """One state per text and layer: an array of shape (texts, layers, width), float32. With `skip_first`, the
        sequence's first token is also left out: a model without a begin-of-sequence token (Qwen3) puts an attention
        sink there, whose massive activations otherwise swamp the mean."""
        import torch

        out = np.zeros((len(texts), len(layers), self.info["width"]), dtype=np.float32)
        with torch.no_grad():
            for i, text in enumerate(texts):
                ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
                keep = ~torch.isin(ids[0], self.special)
                if skip_first:
                    keep[0] = False
                saved = []
                with self.model.trace(ids):
                    for layer in layers:
                        saved.append(self.model.layers_output[layer].save())
                for j, s in enumerate(saved):
                    out[i, j] = s[0, keep.to(s.device)].float().mean(0).cpu().numpy()
        return out

    def states_in_context(self, chunks: list[str], layers: list[int], skip_first: bool = True) -> np.ndarray:
        """One state per chunk and layer, each chunk read inside the whole sequence (one forward pass): what the model
        makes of the chunk once deep in, against `states`, which reads each chunk on its own. Chunks are joined as
        given, so they carry their own separators or chat formatting."""
        import torch

        spans, ids = [], []
        first = self.tokenizer("", return_tensors="pt")["input_ids"][0].tolist()    # what the tokenizer prepends
        ids.extend(first)
        for chunk in chunks:
            piece = self.tokenizer(chunk, add_special_tokens=False)["input_ids"]
            spans.append((len(ids), len(ids) + len(piece)))
            ids.extend(piece)
        if skip_first and spans and spans[0][0] == 0:
            spans[0] = (1, spans[0][1])                 # the sequence's own first token: the sink
        saved = []
        with torch.no_grad():
            with self.model.trace(torch.tensor([ids])):
                for layer in layers:
                    saved.append(self.model.layers_output[layer].save())
        out = np.zeros((len(chunks), len(layers), self.info["width"]), dtype=np.float32)
        for j, s in enumerate(saved):
            for i, (a, b) in enumerate(spans):
                out[i, j] = s[0, a:b].float().mean(0).cpu().numpy()
        return out
