"""Step −1C's off-structure theme (body/offtheme.py): same words as house, other templates, usable, length-matched."""
import numpy as np
import pytest

from body import offtheme as T


def test_tokens_placeholders_and_inform7():
    assert T.tokens("Door, (dir): #i7_closed/open# (name) [if x]y[end if].") == ["door", "<x>", "<s>", "<x>", "y"]


def test_ngrams_ignore_placeholder_only_runs():
    r = {"a": ["(name) #b# (dir) (val)", "you are in the (name)"]}
    assert T.ngrams(r) == {("you", "are", "in", "the"), ("are", "in", "the", "<x>")}


def test_check_catches_new_words_and_shared_templates(tmp_path):
    pytest.importorskip("textworld")
    bad = tmp_path / "bad_room.twg"
    bad.write_text("dec:You are now in the (name), zebra.\n")
    out = T.check(bad)
    assert "zebra" in out["words_not_in_house"]
    assert out["missing_symbols"] and not out["ok"]


def test_theme_passes_its_checks():
    pytest.importorskip("textworld")
    out = T.check()
    assert out == {**out, "words_not_in_house": [], "shared_4grams": [], "missing_symbols": [], "ok": True}


def test_describe_matches_length_and_is_seeded():
    pytest.importorskip("textworld")
    import textworld
    from textworld.generator import make_game
    opts = textworld.GameOptions()
    opts.seeds, opts.nb_rooms, opts.nb_objects, opts.quest_length = 5, 4, 8, 3
    game = make_game(opts)
    matched = 0
    for room in game.world.rooms:
        a, b = T.describe(room, game, seed=11), T.describe(room, game, seed=11)
        assert a == b
        if a["desc"] is not None:
            matched += 1
            assert abs(a["length"] - a["house_length"]) <= T.TOLERANCE * a["house_length"]
            assert a["desc"] != game.infos[room.id].desc
    assert matched >= len(game.world.rooms) - 1
