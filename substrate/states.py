"""Reading the model's states (PLAN.md, step 1): the residual stream after chosen layers, mean-pooled over a text's
tokens, through nnterp's standardized layer names so the same code reads Llama and Qwen.

Special tokens are left out of the mean: Llama's begin-of-text token carries a state far larger than any other and
would swamp it. States are derived, never the record; `info` is what the record keeps so they can be rebuilt,
including how the model was loaded: where its layers sit, in what precision, and whether it was quantised (4-bit is a
quick check only: it adds noise to D).

Loading the 8B on a card smaller than its weights: `device_map="auto"` with `gpu_memory` (GiB) puts as many layers
on the GPU as fit and the rest in system RAM. Slower, but exact.

Attention is read in chunks (DECISIONS.md, 2026-11-01): the model runs its own fused attention, and for the layers
asked for, the causal softmax is recomputed from the post-rotary queries and keys a block of query rows at a time and
averaged over heads, so no full heads x T x T matrix is ever held (eager attention's peak was ~17 GB a layer for the
8B at 8,192 tokens). Exact: the same softmax eager attention takes, in float32.
"""
from __future__ import annotations

import numpy as np

READOUT = "aware_readout"
SINK_FACTOR = 5.0               # a sink's norm over the median of the others (PLAN.md, the sink by norm)
ROW_BUDGET = 1 << 26            # score elements per block (heads x rows x T): 256 MB in float32
_capture: dict | None = None    # layer index -> (T, T) float32 array being filled, during attention_in_context
_base: list = []                # the model's own attention function, which the readout passes through to


def _readout(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """The model's own attention, plus, for a captured layer, its head-averaged probabilities computed in row
    blocks. query is (1, heads, T, d), after rotary embedding; key and value (1, kv heads, T, d)."""
    import torch
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    out = _base[0](module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    target = None if _capture is None else _capture.get(module.layer_idx)
    if target is not None:
        if getattr(module, "sliding_window", None):
            raise NotImplementedError("the attention readout assumes full causal attention")
        with torch.no_grad():
            q = query[0].float()
            k = repeat_kv(key, module.num_key_value_groups)[0].float()
            H, T, _ = q.shape
            rows = max(1, ROW_BUDGET // (H * T))
            kpos = torch.arange(T, device=q.device)
            for r0 in range(0, T, rows):
                r1 = min(T, r0 + rows)
                scores = torch.matmul(q[:, r0:r1], k.transpose(1, 2)) * scaling
                later = kpos[None, :] > torch.arange(r0, r1, device=q.device)[:, None]
                scores.masked_fill_(later, float("-inf"))
                target[r0:r1] = torch.softmax(scores, dim=-1).mean(0).cpu().numpy()
    return out


def sink_tokens(norms: np.ndarray, factor: float = SINK_FACTOR) -> np.ndarray:
    """A mask of the tokens that are the attention sink: norm above `factor` times the median of the others, the
    median taken without the largest token so that a two-token text can still flag. A lone token never flags."""
    norms = np.asarray(norms, dtype=float)
    if len(norms) < 2:
        return np.zeros(len(norms), dtype=bool)
    return norms > factor * np.median(np.sort(norms)[:-1])


class Reader:
    def __init__(self, model, tokenizer=None, revision: str | None = None, device: str | None = None,
                 dtype: str | None = None, gpu_memory: float | None = None, quantize: str | None = None,
                 sink: str = "norm"):
        """`model` is a Hugging Face name, or an already-built model with its `tokenizer` (the tests' tiny one).
        `device`: "cuda", "cpu" or "auto" (split across GPU and RAM, capped at `gpu_memory` GiB on the GPU).
        `dtype`: "bfloat16" or "float32" (default: bfloat16 on a GPU, float32 on CPU). `quantize`: None, "4bit" or
        "8bit" (needs bitsandbytes and a GPU). `sink`: how `skip_first` finds the attention sink: "norm", or "first",
        by position, the rule earlier records were read under (DECISIONS.md, 2026-11-02)."""
        import torch
        from nnterp import StandardizedTransformer

        if sink not in ("norm", "first"):
            raise ValueError(f"unknown sink rule {sink!r}")
        self.sink = sink
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
                     "dtype": str(next(self.model._model.parameters()).dtype), "load": self.load, "sink": sink,
                     "sink_layer": self.model.num_layers // 3 if sink == "norm" else None}

    def _read(self, ids, layers: list[int]) -> dict:
        """The residual stream at each layer asked for, plus the layer the sink is found at. Layers are visited in
        order, as the tracer requires."""
        wanted = set(layers) | ({self.info["sink_layer"]} if self.sink == "norm" else set())
        saved = {}
        with self.model.trace(ids):
            for layer in sorted(wanted):
                saved[layer] = self.model.layers_output[layer].save()
        return saved

    def _sink(self, saved: dict, among) -> "np.ndarray":
        """The sink's mask over a sequence's tokens, found among the tokens in the boolean mask `among`."""
        norms = saved[self.info["sink_layer"]][0].float().norm(dim=-1).cpu().numpy()
        among = np.asarray(among)
        mask = np.zeros(len(norms), dtype=bool)
        mask[among] = sink_tokens(norms[among])
        return mask

    def states(self, texts: list[str], layers: list[int], skip_first: bool = True) -> np.ndarray:
        """One state per text and layer: an array of shape (texts, layers, width), float32. With `skip_first`, the
        attention sink is also left out: a model without a begin-of-sequence token (Qwen3) puts massive activations on
        one early token, which otherwise swamp the mean. Under the "norm" rule it is found by its norm, wherever it
        landed; under "first" it is assumed to be the first token, which is not always so (PLAN.md, the D jump)."""
        import torch

        out = np.zeros((len(texts), len(layers), self.info["width"]), dtype=np.float32)
        with torch.no_grad():
            for i, text in enumerate(texts):
                ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
                keep = ~torch.isin(ids[0], self.special)
                saved = self._read(ids, layers)
                if skip_first and self.sink == "first":
                    if keep[1:].any():                      # a one-token text keeps its token: no empty mean
                        keep[0] = False
                elif skip_first:
                    sink = torch.from_numpy(self._sink(saved, keep.numpy()))
                    if (keep & ~sink).any():                # never an empty mean
                        keep = keep & ~sink
                for j, layer in enumerate(layers):
                    s = saved[layer]
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
        if skip_first and self.sink == "first" and spans and spans[0][0] == 0 and spans[0][1] > 1:
            spans[0] = (1, spans[0][1])                 # the sequence's own first token: the sink
        with torch.no_grad():
            saved = self._read(torch.tensor([ids]), layers)
        sink = np.zeros(len(ids), dtype=bool)
        if skip_first and self.sink == "norm":
            sink = self._sink(saved, ~np.isin(ids, self.special.numpy()))
        out = np.zeros((len(chunks), len(layers), self.info["width"]), dtype=np.float32)
        for j, layer in enumerate(layers):
            s = saved[layer]
            for i, (a, b) in enumerate(spans):
                keep = np.arange(a, b)[~sink[a:b]]
                keep = keep if len(keep) else np.arange(a, b)   # never an empty mean
                out[i, j] = s[0, torch.as_tensor(keep, device=s.device)].float().mean(0).cpu().numpy()
        return out

    def attention_in_context(self, chunks: list[str], layers: list[int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Attention probabilities over one sequence of chunks, averaged over heads: an array (layers, T, T), float32,
        and each chunk's token span. Read in row blocks beside the model's own attention (see the module notes)."""
        global _capture
        import torch
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

        spans, ids = [], []
        ids.extend(self.tokenizer("", return_tensors="pt")["input_ids"][0].tolist())
        for chunk in chunks:
            piece = self.tokenizer(chunk, add_special_tokens=False)["input_ids"]
            spans.append((len(ids), len(ids) + len(piece)))
            ids.extend(piece)
        if not getattr(self.model, "dispatched", True):    # nnsight keeps the weights on meta until first used
            self.model.dispatch()
        hf = self.model._model
        config = hf.config
        base = config._attn_implementation
        _base[:] = [ALL_ATTENTION_FUNCTIONS.get_interface(base, eager_attention_forward)]
        ALL_ATTENTION_FUNCTIONS.register(READOUT, _readout)
        ALL_MASK_ATTENTION_FUNCTIONS.register(READOUT, ALL_MASK_ATTENTION_FUNCTIONS[base])
        T = len(ids)
        _capture = {layer: np.zeros((T, T), dtype=np.float32) for layer in layers}
        try:
            config._attn_implementation = READOUT
            with torch.no_grad():
                hf(torch.tensor([ids], device=hf.device), use_cache=False, logits_to_keep=1)   # no T x vocab logits
            return np.stack([_capture[layer] for layer in layers]), spans
        finally:
            config._attn_implementation = base
            _capture = None
