"""Model access, on a tiny randomly initialised Qwen3 built here (no download): it tests the plumbing, never a
result. The real model's test is marked `model` and runs with `pytest -m model` where Hugging Face is reachable."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nnterp")

from substrate.measure_d import measure  # noqa: E402
from substrate.record import Record, read  # noqa: E402
from substrate.states import Reader  # noqa: E402

WORDS = "the a plan garden budget write design for of to and family river lesson".split()


@pytest.fixture(scope="module")
def tiny():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    vocab = {"[UNK]": 0, "<bos>": 1, "<pad>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 1)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", bos_token="<bos>", pad_token="<pad>")
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                         num_attention_heads=4, num_key_value_heads=2, head_dim=8)).eval()
    return model, tok


def test_states_are_the_residual_stream_pooled_without_special_tokens(tiny):
    model, tok = tiny
    reader = Reader(model, tokenizer=tok)
    text = "plan a garden budget"
    got = reader.states([text], [0, 2])
    ids = tok(text, return_tensors="pt")
    assert ids["input_ids"][0, 0].item() == 1                       # the tokenizer adds <bos>
    with torch.no_grad():
        hidden = model(**ids, output_hidden_states=True).hidden_states
    for j, layer in enumerate([0, 2]):                              # hidden_states[0] is the embeddings
        want = hidden[layer + 1][0, 1:].mean(0).numpy()
        assert np.allclose(got[0, j], want, atol=1e-5)
    assert got.shape == (1, 2, 32)
    assert reader.info["layers"] == 4


def test_measure_d_refuses_unfrozen_sets_and_records_a_draft_run(tiny, tmp_path):
    model, tok = tiny
    reader = Reader(model, tokenizer=tok)
    sets = [{"id": "a", "texts": ["plan a garden", "a garden plan", "plan the garden"], "frozen": False, "by": "model"},
            {"id": "b", "texts": ["write a lesson", "a lesson to write", "write the lesson"], "frozen": False,
             "by": "model"}]
    record = Record(tmp_path / "record.jsonl")
    with pytest.raises(SystemExit):
        measure(reader, sets, [0, 3], record)
    report = measure(reader, sets, [0, 3], record, draft=True)
    rows = read(tmp_path / "record.jsonl")
    assert [r["kind"] for r in rows] == ["run", "set", "set", "report"]
    assert rows[0]["draft"] is True and rows[1]["by"] == "model"
    assert report["repeat_max_abs_diff"] == 0.0
    assert {r["layer"] for r in report["layers"]} == {0, 3}
    assert np.load(tmp_path / "states.npy").shape == (2, 3, 2, 32)


@pytest.mark.model
def test_qwen3_small_reads_states_on_cpu():
    sets = [json.loads(line) for line in open("material/paraphrases.jsonl")]
    reader = Reader("Qwen/Qwen3-0.6B")
    S = reader.states(sets[0]["texts"][:2] + sets[0]["texts"][:1], [0, reader.info["layers"] - 1])
    assert S.shape[:2] == (3, 2) and np.isfinite(S).all()
    assert np.array_equal(S[0], S[2])


def test_a_chunk_read_in_context_differs_from_the_same_chunk_read_alone(tiny):
    model, tok = tiny
    reader = Reader(model, tokenizer=tok)
    chunks = ["plan a garden ", "write the budget ", "plan a garden "]
    inside = reader.states_in_context(chunks, [1, 3])
    alone = reader.states(chunks, [1, 3])
    assert np.allclose(inside[0], alone[0], atol=1e-5)      # the first chunk has nothing before it but <bos>
    assert not np.allclose(inside[2], alone[2], atol=1e-3)  # the same words, read deep in


def test_skip_first_leaves_out_the_first_token_only_when_no_bos_precedes_it(tiny):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    model, tok = tiny
    with_bos = Reader(model, tokenizer=tok)
    text = "plan a garden budget"
    assert np.allclose(with_bos.states([text], [1], skip_first=True), with_bos.states([text], [1]), atol=1e-6)
    vocab = {"[UNK]": 0, "<pad>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    no_bos = Reader(model, tokenizer=PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="<pad>"))
    ids = no_bos.tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**ids, output_hidden_states=True).hidden_states
    got = no_bos.states([text], [1], skip_first=True)
    assert np.allclose(got[0, 0], hidden[2][0, 1:].mean(0).numpy(), atol=1e-5)
    inside = no_bos.states_in_context(["plan a garden ", "write the budget "], [1], skip_first=True)
    alone = no_bos.states(["plan a garden "], [1], skip_first=True)
    assert np.allclose(inside[0], alone[0], atol=1e-5)          # the first chunk, sink left out either way


def test_a_one_token_text_keeps_its_token_rather_than_averaging_nothing(tiny):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    model, _ = tiny
    vocab = {"[UNK]": 0, "<pad>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    reader = Reader(model, tokenizer=PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="<pad>"))
    assert np.isfinite(reader.states(["plan"], [1])).all()
    assert np.isfinite(reader.states_in_context(["plan", "write the budget"], [1])).all()


def test_attention_in_context_matches_transformers_own_head_mean(monkeypatch):
    import copy

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from substrate import states

    vocab = {"[UNK]": 0, "<pad>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="<pad>")
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    eager = Qwen3ForCausalLM(copy.deepcopy(config)).eval()
    eager.set_attn_implementation("eager")
    fused = copy.deepcopy(eager)
    fused.set_attn_implementation("sdpa")                                   # the model runs its fused attention
    monkeypatch.setattr(states, "ROW_BUDGET", 4 * 6 * 2)                     # blocks of two query rows
    reader = Reader(fused, tokenizer=tok)
    A, spans = reader.attention_in_context(["plan a garden ", "write the budget "], [1, 3])
    ids = torch.tensor([tok("plan a garden write the budget", add_special_tokens=False)["input_ids"]])
    with torch.no_grad():
        ref = eager(ids, output_attentions=True).attentions
        assert torch.allclose(fused(ids).logits, eager(ids).logits, atol=1e-5)
    assert spans == [(0, 3), (3, 6)] and A.shape == (2, 6, 6)
    assert np.allclose(A[0], ref[1][0].mean(0).numpy(), atol=1e-6)
    assert np.allclose(A[1], ref[3][0].mean(0).numpy(), atol=1e-6)
    assert fused.config._attn_implementation == "sdpa"                     # restored after the read
