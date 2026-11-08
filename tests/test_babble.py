"""Step 2's code (body/babble.py, body/lora.py) on a tiny random Qwen3 with Qwen3's own tokenizer: plumbing, never a
result."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _tiny():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3's tokenizer is not cached")
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                         head_dim=8)).eval()
    return tok, model


TURNS = [{"turn": t, "action": f"go {d}", "reply": f"You are in room {t}."} for t, d in enumerate("nsew" * 3)]


def test_lora_off_is_the_base_model_exactly_and_detach_restores_it():
    from body import lora

    tok, model = _tiny()
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        before = model(x).logits
    w = lora.attach(model, rank=4)
    for m in w.values():
        m.B.data.normal_()
    with torch.no_grad():
        on = model(x).logits
        with lora.off(w):
            off = model(x).logits
    assert torch.equal(off, before) and not torch.allclose(on, before)
    lora.detach(model, w)
    with torch.no_grad():
        assert torch.equal(model(x).logits, before)


def test_render_puts_the_sense_places_before_each_action_and_marks_its_tokens():
    from body.babble import K, render, windows

    tok, _ = _tiny()
    r = render(tok, "sys", "You wake up.", TURNS[:3])
    ids = r["ids"]
    assert sum(1 for i in ids if i == -1) == 3 * K
    for x in r["turns"]:
        assert all(ids[p] == -1 for p in x["sense"])
        a, b = x["lm"]
        assert tok.decode(ids[a:b - 1]) == TURNS[x["turn"]]["action"]
        assert tok.decode(ids[b - 1:b]) == "<|im_end|>"
        assert x["last"] == b - 2
    p = render(tok, "sys", "You wake up.", TURNS[:3], prompt=True)
    assert p["ids"][: len(ids)] == ids and len(p["turns"]) == 4
    assert render(tok, "sys", "You wake up.", TURNS[:3], k=0)["ids"].count(-1) == 0
    ws = windows(tok, "sys", "You wake up.", TURNS, max_len=len(ids) + 5)
    assert len(ws) > 1 and [x["turn"] for w in ws for x in w["turns"]] == list(range(12))
    assert all(len(w["ids"]) <= len(ids) + 5 for w in ws)


def test_a_training_step_moves_the_graft_and_the_zeroed_arm_ignores_the_rows():
    from body.babble import SPACE, Graft, _losses, render, sense_rows

    tok, model = _tiny()
    model.config.use_cache = False
    g = Graft(model)
    g.fresh(seed=0)
    win = render(tok, "sys", "You wake up.", TURNS[:4])
    rng = np.random.default_rng(0)
    Z = np.hstack([rng.normal(size=(4, 2)), np.array([[0], [0], [1], [1]])])
    d = {"Z": Z, "target": rng.normal(size=(4, SPACE)), "index": {t: t for t in range(4)}}
    rows = sense_rows(Z, TURNS[:4])
    assert rows[0].tolist() == [0, 0, 0] and rows[2].tolist() == Z[1].tolist()
    with torch.no_grad():
        a = g.hidden_states(win["ids"], rows, "zeroed")
        b = g.hidden_states(win["ids"], rows * 5, "zeroed")
        c = g.hidden_states(win["ids"], rows * 5, "live")
    assert torch.equal(a, b) and not torch.allclose(a, c)
    opt = g.optimizer()
    first = None
    for _ in range(30):
        l_self, l_fm, l_lm = _losses(torch, g, model, win, d, rows, "live")
        loss = l_self + l_fm + 0.1 * l_lm
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first
    assert all(float(p.grad.abs().sum()) >= 0 for p in g.params() if p.grad is not None)
    assert not any(p.requires_grad for n, p in model.named_parameters() if ".A" not in n and ".B" not in n)


def test_the_ratio_takes_the_lower_denominator_in_each_draw():
    from body.babble import ratio

    num = {i: 1.0 for i in range(20)}
    r = ratio(num, {i: 2.0 for i in range(20)}, den2={i: 4.0 for i in range(20)}, draws=50)
    assert r["ratio"] == 0.5 and r["interval_90"] == [0.5, 0.5]


def test_a_constant_column_in_training_does_not_explode_on_other_rows():
    from body.babble import _ridge_fit, _ridge_predict

    rng = np.random.default_rng(0)
    X = np.hstack([rng.normal(size=(50, 2)), np.ones((50, 1))])
    fit = _ridge_fit(X, X[:, :1] * 2)
    assert abs(_ridge_predict(fit, [[1.0, 0.0, 0.0]])[0, 0] - 2.0) < 0.5
