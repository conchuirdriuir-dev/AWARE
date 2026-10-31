"""Model access, on a tiny randomly initialised Qwen3 built here (no download): it tests the plumbing, never a
result. The real model's test is marked `model` and runs with `pytest -m model` where Hugging Face is reachable."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nnterp")

from measure_d import measure  # noqa: E402
from record import Record, read  # noqa: E402
from states import Reader  # noqa: E402

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
