import json
from pathlib import Path

import pytest

from rung2 import builder, store
from rung2.arms import ARMS, VIEW, Arms
from rung2.model import Fake
from rung2.run import main
from rung2.text import chunk, fit, fit_end

STORY = Path(__file__).parent / "fixtures" / "story"
count = Fake().tokens


def passage(conn, chapter, seq, text):
    p = {"id": f"c{chapter:02d}-p{seq:03d}", "chapter": chapter, "seq": seq, "text": text, "tokens": count(text)}
    store.add_passages(conn, [p])
    return p


@pytest.fixture
def conn(tmp_path):
    return store.open_store(tmp_path / "store.sqlite")


def empty(**kw):
    reply = {k: [] for k in ("new_entities", "states", "knowledge", "events", "rules")}
    reply.update(kw)
    return reply


def test_chunk_and_fit():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 50 for i in range(20))
    ps = chunk(3, text, count, target=200)
    assert len(ps) > 1 and all(p["tokens"] <= 200 for p in ps)
    assert ps[0]["id"] == "c03-p001" and [p["seq"] for p in ps] == list(range(1, len(ps) + 1))
    items = ["one two", "three four", "five six"]
    assert fit(items, 8, count) == ["one two", "three four"]          # each item: 3 tokens + 1 for its separator
    assert fit_end(items, 8, count) == ["three four", "five six"]
    assert fit(items, 7, count) == ["one two"]


def test_quotes_aliases_and_dropped_records(conn):
    p1 = passage(conn, 1, 1, "Mara Vell had grey eyes. Teodor, whom she called Teo, kept the key.")
    r = store.apply(conn, empty(
        new_entities=[{"name": "Mara Vell", "kind": "character", "aliases": ["Mara"], "quote": "Mara Vell had grey eyes"},
                      {"name": "Teodor", "kind": "character", "aliases": ["Teo"], "quote": "whom she called Teo"},
                      {"name": "Ghost", "kind": "character", "aliases": [], "quote": "a line that is not there"}],
        states=[{"entity": "Mara", "attribute": "Eye colour", "value": "grey", "quote": "had grey eyes"},
                {"entity": "Nobody", "attribute": "x", "value": "y", "quote": "kept the key"}]), p1)
    assert r["written"]["entities"] == 2 and r["written"]["states"] == 1
    whys = sorted(d["why"] for d in r["dropped"])
    assert whys == ["quote not in passage", "unresolved entity 'Nobody'"]
    row = conn.execute("SELECT * FROM state").fetchone()
    assert (row["entity_id"], row["attribute"], row["passage_id"]) == ("mara-vell", "eye colour", "c01-p001")
    assert store.resolve(conn, "teo", {}) == "teodor"


def test_states_supersede_and_repeats_skip(conn):
    p1 = passage(conn, 1, 1, "Mara hid the locket under the board.")
    p2 = passage(conn, 3, 1, "She moved the locket to her coat pocket. It was under the board no longer.")
    store.apply(conn, empty(new_entities=[{"name": "the locket", "kind": "object", "aliases": [], "quote": "the locket"}],
                            states=[{"entity": "the locket", "attribute": "location", "value": "under the board",
                                     "quote": "hid the locket under the board"}]), p1)
    store.apply(conn, empty(states=[{"entity": "the-locket", "attribute": "location", "value": "under the board",
                                     "quote": "It was under the board"},
                                    {"entity": "the-locket", "attribute": "location", "value": "her coat pocket",
                                     "quote": "moved the locket to her coat pocket"}]), p2)
    rows = conn.execute("SELECT id, value, supersedes, from_chapter FROM state ORDER BY id").fetchall()
    assert [(r["value"], r["supersedes"]) for r in rows] == [("under the board", None), ("her coat pocket", 1)]
    lines2 = builder.entity_lines(conn, "the-locket", 2, builder.Counter())
    lines3 = builder.entity_lines(conn, "the-locket", 3, builder.Counter())
    assert any("under the board" in x for x in lines2) and not any("coat" in x for x in lines2)
    assert any("coat pocket" in x for x in lines3) and not any("under the board" in x for x in lines3)


def test_knowledge_correction(conn):
    p1 = passage(conn, 2, 1, "Mara believed her father wrote the letter.")
    p2 = passage(conn, 3, 1, "Mara learned that Teodor had written the letter himself.")
    store.apply(conn, empty(new_entities=[{"name": "Mara", "kind": "character", "aliases": [], "quote": "Mara"},
                                          {"name": "the letter", "kind": "object", "aliases": [], "quote": "the letter"}],
                            knowledge=[{"character": "Mara", "proposition": "her father wrote the letter",
                                        "about": "the letter", "stance": "believes", "true_in_story": None,
                                        "quote": "believed her father wrote the letter"}]), p1)
    store.apply(conn, empty(knowledge=[{"character": "mara", "proposition": "Teodor wrote the letter",
                                        "about": "the-letter", "stance": "knows", "true_in_story": True,
                                        "quote": "learned that Teodor had written the letter"}]), p2)
    first, second = conn.execute("SELECT id, corrected_by FROM knowledge ORDER BY id").fetchall()
    assert first["corrected_by"] == second["id"] and second["corrected_by"] is None
    as_of_2 = builder.entity_lines(conn, "mara", 2, builder.Counter())
    assert any("believes that her father" in x for x in as_of_2) and not any("revised" in x for x in as_of_2)
    assert any("revised in ch. 3" in x for x in builder.entity_lines(conn, "mara", 3, builder.Counter()))


def test_loops_as_of_chapter(conn):
    p1 = passage(conn, 1, 1, "Who had Teodor locked out of the archive?")
    p2 = passage(conn, 2, 1, "The archive stayed locked all spring.")
    p3 = passage(conn, 3, 1, "Mara found the archive door open at last.")
    def chapter(n, p, **kw):
        return store.apply_loops(conn, {"opened": [], "advanced": [], "resolved": [], **kw}, n, [p])
    chapter(1, p1, opened=[{"description": "who Teodor locked out", "entities": [], "quote": "Who had Teodor locked out"}])
    r = chapter(2, p2, advanced=[{"loop_id": 1, "quote": "The archive stayed locked"},
                                 {"loop_id": 99, "quote": "The archive stayed locked"},
                                 {"loop_id": 1, "quote": "a line from another chapter"}])
    assert sorted(d["why"] for d in r["dropped"]) == ["no open loop 99", "quote not in chapter"]
    r = chapter(3, p3, resolved=[{"loop_id": 1, "quote": "found the archive door open"}])
    assert r["dropped"] == [] and r["written"]["resolved"] == 1
    assert conn.execute("SELECT closed_passage FROM loop").fetchone()[0] == "c03-p001"
    assert [x["touched"] for x in builder.open_loops(conn, 1)] == [1]
    assert [x["touched"] for x in builder.open_loops(conn, 2)] == [2]
    assert builder.open_loops(conn, 3) == []


def test_pipeline_with_fake_model(tmp_path):
    run = tmp_path / "run"
    main(["build", "--story", str(STORY), "--run", str(run), "--fake"])
    conn = store.open_store(run / "store.sqlite")
    assert conn.execute("SELECT COUNT(*) FROM chapter_line").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM entity").fetchone()[0] > 0
    assert set(json.loads((run / "summaries.json").read_text())) == {"1", "2", "3"}

    questions = tmp_path / "q.jsonl"
    questions.write_text("\n".join(json.dumps(q) for q in [
        {"id": "q1", "type": 1, "after": 3, "question": "What colour were Mara's eyes?"},
        {"id": "q2", "type": 4, "after": 3, "question": "Which setups are still unresolved?"},
        {"id": "q3", "type": 6, "after": 2, "question": "Write the next scene."}]))
    evidence = tmp_path / "e.jsonl"
    evidence.write_text(json.dumps({"id": "q1", "quotes": ["grey eyes"]}))
    main(["ask", "--run", str(run), "--questions", str(questions), "--evidence", str(evidence)])
    answers = [json.loads(x) for x in (run / "answers.jsonl").read_text().splitlines()]
    assert {a["arm"] for a in answers} == set(ARMS)
    assert all(a["view_tokens"] <= VIEW for a in answers)
    assert len([a for a in answers if a["id"] == "q3" and a["arm"] == "A"]) == 3

    key = tmp_path / "key.jsonl"
    key.write_text("\n".join(json.dumps(k) for k in [{"id": "q1", "accept": ["grey"]},
                                                      {"id": "q2", "items": [["locket"], ["lighthouse"]]},
                                                      {"id": "q3"}]))
    main(["score", "--run", str(run), "--key", str(key)])
    text = (run / "report.md").read_text()
    assert "| 1 |" in text and "Continuations" in text
    calls = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    assert any(c["kind"] == "call" and c["role"] == "keeper" for c in calls)
    assert any(c["kind"] == "view" and c["arm"] == "D" for c in calls)


def test_views_stay_in_budget_on_a_long_story(tmp_path):
    story = tmp_path / "story"
    story.mkdir()
    for n in range(1, 7):
        paras = [f"Mara walked to Tallow Street in chapter {n}, part {i}. " + "The rain kept on. " * 40 for i in range(12)]
        (story / f"ch{n:02d}.md").write_text("\n\n".join(paras))
    run = tmp_path / "run"
    main(["build", "--story", str(story), "--run", str(run), "--fake", "--window", "32768"])
    conn = store.open_store(run / "store.sqlite")
    arms = Arms(conn, run, count, Fake().embed)
    q = {"id": "q", "type": 1, "after": 6, "question": "Where did Mara walk?"}
    for arm in "ABCD":
        assert count(arms.view(arm, q)) <= VIEW, arm


def test_dry_run_split_and_show(tmp_path, capsys):
    from rung2.model import strip_thinking
    assert strip_thinking("<think>\nhmm\n</think>\n\nMara hid it.") == "Mara hid it."
    para = "Mara walked on. " * 120
    book = ("The Project Gutenberg eBook\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
            "CONTENTS\nCHAPTER I. The Shop\nCHAPTER II. The Key\n\n"
            f"CHAPTER I. The Shop\n\n{para}\n\nCHAPTER II. The Key\n\n{para}\n\nCHAPTER III. Ness\n\n{para}\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicence text")
    (tmp_path / "book.txt").write_text(book)
    main(["split", "--text", str(tmp_path / "book.txt"), "--out", str(tmp_path / "story"), "--chapters", "2"])
    assert sorted(p.name for p in (tmp_path / "story").iterdir()) == ["ch01.md", "ch02.md"]
    assert "licence" not in (tmp_path / "story" / "ch02.md").read_text()

    run = tmp_path / "run"
    main(["build", "--story", str(STORY), "--run", str(run), "--fake", "--only", "keeper"])
    assert not (run / "summaries.json").exists() and not (run / "vectors.npy").exists()
    capsys.readouterr()
    main(["show", "--run", str(run)])
    out = capsys.readouterr().out
    assert "KEEPER: 3 calls" in out and "ENTITIES" in out and "CHAPTER LINES" in out


def test_server_errors_carry_the_reason_and_schema_falls_back():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from rung2.keeper import SCHEMA
    from rung2.model import Server, ServerError, check

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, {"data": [{"id": "tiny-0.8b"}, {"id": "stand-in"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if body.get("model") not in ("tiny-0.8b", "stand-in"):
                return self.reply(400, {"error": {"message": "model name is missing from the request"}})
            if self.path == "/tokenize":
                return self.reply(200, {"tokens": body["content"].split()})
            if self.path == "/v1/embeddings":
                return self.reply(501, {"error": {"message": "This server does not support embeddings"}})
            if "response_format" in body:
                return self.reply(400, {"error": {"message": "unsupported response_format"}})
            content = '{"new_entities": []}' if "json_schema" in body else "<think>x</think>hello there friend"
            self.reply(200, {"choices": [{"message": {"content": content}}]})

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_port}"
    try:
        with pytest.raises(ServerError, match="several models.*tiny-0.8b, stand-in"):
            Server(url)
        s = Server(url, model="stand-in")
        assert s.name == "stand-in" and s.chat("hi") == "hello there friend"
        assert s.chat("hi", schema=SCHEMA) == '{"new_entities": []}' and s.schema_style == "json_schema"
        with pytest.raises(ServerError, match="HTTP 501.*does not support embeddings"):
            s.embed(["x"])
        assert check(url, None, SCHEMA)[0][1] is False
        results = {name: ok for name, ok, _ in check(url, None, SCHEMA, model="stand-in")}
        assert results["schema: response_format"] is False and results["schema: json_schema field"] is True
        assert results["token count (POST /tokenize)"] and results["plain chat, thinking off"]
    finally:
        httpd.shutdown()


def test_lenient_parse_names_from_the_passage_and_loop_merge(conn):
    from rung2.keeper import parse
    assert parse('```json\n{"states": []}\n```') == {"states": []}
    with pytest.raises(ValueError):
        parse('{"states": [')                                  # cut off at the token limit
    p = passage(conn, 2, 1, "Aunt Polly sent Tom to the fence on Main Street. Who had taken the jam?")
    r = store.apply(conn, empty(
        events=[{"summary": "Polly sends Tom to the fence", "place": "Main Street", "participants": ["Aunt Polly", "Tom"],
                 "happened": True, "quote": "Aunt Polly sent Tom to the fence"},
                {"summary": "Ben arrives", "place": None, "participants": ["Ben"], "happened": True,
                 "quote": "sent Tom to the fence"}]), p)
    assert sorted(c["name"] for c in r["created"]) == ["Aunt Polly", "Main Street", "Tom"]
    assert [d["why"] for d in r["dropped"]] == ["unresolved entity 'Ben'"]   # not in the passage: never created
    kinds = dict(conn.execute("SELECT name, kind FROM entity").fetchall())
    assert kinds == {"Aunt Polly": "character", "Tom": "character", "Main Street": "place"}
    opened = {"opened": [{"description": "who took the jam", "entities": ["Tom"], "quote": "Who had taken the jam?"}],
              "advanced": [], "resolved": []}
    assert store.apply_loops(conn, opened, 2, [p])["written"]["opened"] == 1
    assert conn.execute("SELECT entity_id FROM loop_entity").fetchone()[0] == "tom"
    opened["opened"][0]["description"] = "Who took the jam"
    r = store.apply_loops(conn, opened, 2, [p])
    assert r["written"]["opened"] == 0 and r["written"]["advanced"] == 1


def test_second_dry_run_fixes(conn):
    from rung2.keeper import SCHEMA
    assert store.quoted("he had a very good time", "and he had a really good time of it")    # 1 word of 6 differs
    assert not store.quoted("had a bad time", "and he had a really good time of it")        # under 5 words: exact only
    assert store.proper_name("Aunt Polly", "aunt polly") and store.proper_name("Ben", "Then Ben came by.")
    assert not store.proper_name("the shade", "They sat in the shade.") and not store.proper_name("Ben", "Ben came.")
    assert SCHEMA["properties"]["states"]["items"]["properties"]["attribute"]["enum"][0] == "location"
    p = passage(conn, 1, 1, "Tom and the new boy fought. The boy cried. Tom laughed, and Tom laughed again.")
    r = store.apply(conn, empty(
        new_entities=[{"name": "Tom", "kind": "character", "aliases": ["Tom", "the boy"], "quote": "Tom and the new boy"},
                      {"name": "the new boy", "kind": "character", "aliases": ["the boy"], "quote": "the new boy"}],
        knowledge=[{"character": "Tom", "proposition": "the new boy cried", "about": None, "stance": "knows",
                    "true_in_story": None, "quote": "The boy cried"}] * 2,
        events=[{"summary": "the boy cries", "place": None, "participants": ["the boy"], "happened": True,
                 "quote": "The boy cried"}]), p)
    assert json.loads(conn.execute("SELECT aliases FROM entity WHERE id = 'tom'").fetchone()[0]) == ["the boy"]
    assert r["written"]["knowledge"] == 1
    assert [d["why"] for d in r["dropped"]] == ["ambiguous entity 'the boy': tom, the-new-boy"]


def test_third_dry_run_fixes(conn):
    p = passage(conn, 1, 1, "Aunt Polly caught the new boy near the fence. She knew Tom had whitewashed it. "
                            "Later she believed Tom could work when he wanted to. Tom kept the stone.")
    r = store.apply(conn, empty(
        new_entities=[{"name": "the stone", "kind": "object", "aliases": [], "quote": "Tom kept the stone"},
                      {"name": "Tom", "kind": "character", "aliases": [], "quote": "Tom kept the stone"}],
        events=[{"summary": "Polly catches the boy", "place": "near the fence", "participants": ["Aunt Polly",
                 "the new boy"], "happened": True, "quote": "Aunt Polly caught the new boy"},
                {"summary": "nothing", "place": "unknown", "participants": [], "happened": True,
                 "quote": "Tom kept the stone"}],
        knowledge=[{"character": "Aunt Polly", "proposition": "Tom whitewashed the fence", "about": "Tom",
                    "stance": "knows", "true_in_story": True, "quote": "She knew Tom had whitewashed it"},
                   {"character": "Aunt Polly", "proposition": "Tom can work when he wants to", "about": "Tom",
                    "stance": "believes", "true_in_story": None, "quote": "believed Tom could work"}],
        states=[{"entity": "Aunt Polly", "attribute": "held_by", "value": "cat", "quote": "Aunt Polly caught"},
                {"entity": "the stone", "attribute": "held_by", "value": "Tom", "quote": "Tom kept the stone"}]), p)
    assert {c["name"] for c in r["created"]} == {"Aunt Polly", "the new boy"}
    assert [d["why"] for d in r["dropped"]] == ["held_by on something not an object"]
    places = conn.execute("SELECT place_id, place_text FROM event ORDER BY id").fetchall()
    assert [tuple(x) for x in places] == [(None, "near the fence"), (None, None)]
    rows = conn.execute("SELECT about_id, corrected_by FROM knowledge").fetchall()
    assert [r["about_id"] for r in rows] == ["tom", "tom"] and all(r["corrected_by"] is None for r in rows)
    assert any("(near the fence)" in x for x in builder.entity_lines(conn, "the-new-boy", 1, builder.Counter()))


def test_negation_test_scores_kept_inverted_and_missing(conn, tmp_path, capsys):
    from rung2 import negation
    p1 = passage(conn, 1, 1, "No iron passed that gate. The latch was one that no one had fastened. Maren kept bees.")
    store.apply(conn, empty(
        new_entities=[{"name": "Maren", "kind": "character", "aliases": [], "quote": "Maren kept bees"}],
        rules=[{"statement": "No iron may pass the gate", "quote": "No iron passed that gate"}],
        events=[{"summary": "Someone fastened the latch", "participants": [], "happened": True,
                 "quote": "no one had fastened"}]), p1)
    probes = tmp_path / "key_probes.py"
    probes.write_text('GENDER = {"Maren": "f"}\nPROBES = {\n'
                      '  "R4": [dict(label="no iron passes the gate", find="No iron passed that gate",\n'
                      '              lemmas=["iron", "pass", "gate"], neg=True)],\n'
                      '  "L1": [dict(label="latch never fastened", find="no one had fastened",\n'
                      '              lemmas=["latch", "fasten"], neg=True)],\n'
                      '  "S1": [dict(label="Maren keeps bees", find="Maren kept bees", lemmas=["bee"], entity="Maren")],\n'
                      '  "X": [dict(label="not in the text", find="a sentence nowhere", lemmas=["x"])],\n}\n')
    got = {i: negation.score_probe(conn, ps[0])["status"] for i, ps in negation.load_probes(probes).items()}
    assert got == {"R4": "ok", "L1": "inverted", "S1": "partial", "X": "sentence not found"}    # S1: no row holds bees

    erg = tmp_path / "ergonly_report.txt"
    rows = [("R4", "no iron passes the gate", "ok", "erg", ""),
            ("L1", "latch never fastened", "ok", "erg", "")]
    erg.write_text("item      probe\n" + "".join(f"{a:<10}{b[:33]:<34}{c:<20}{d:<16}{e}\n" for a, b, c, d, e in rows))
    reply = {"events": [{"summary": "Someone fastened the latch", "quote": "no one had fastened"}]}
    (tmp_path / "log.jsonl").write_text(json.dumps({"kind": "call", "role": "keeper", "ref": p1["id"],
                                                    "reply": json.dumps(reply)}) + "\n")
    main(["negation", "--run", str(tmp_path), "--probes", str(probes), "--erg", str(erg)])
    out = capsys.readouterr().out
    assert "kept 1, stored inverted 1, not stored 0" in out
    assert "the ERG kept the negation and the keeper inverted it on 1" in out
    assert "quote: 'no one had fastened', a polarity check catches it" in out
    assert "would catch 1 of 1" in out and "points to: no decision: 2 negated facts comparable" in out
    assert (tmp_path / "negation.txt").read_text() == out
    assert negation.negated(negation.words("the latch is unfastened")) and negation.has(negation.words("unfastened"), "fasten")
    assert not negation.negated(negation.words("she understood the united bees"))
    assert negation.row_negated({"field": "Hanno set foot in the garth", "happened": False})
    assert not negation.row_negated({"field": "Maren stood in the garth", "happened": True})


def test_negation_test_reading_follows_the_pass_criteria():
    from rung2.negation import reading
    base = {"comparable": 7, "erg_lost": 0, "not stored": 0, "inverted": 0, "added": 0, "uncaught": 0,
            "uncaught_erg_kept": 0}
    assert reading(base, erg=False).startswith("no reading")
    assert reading({**base, "comparable": 4}, True).startswith("no decision")
    assert "does not test negation" in reading({**base, "erg_lost": 4}, True)
    assert "coverage" in reading({**base, "not stored": 4}, True)
    assert "ERG stays out" in reading(base, True)
    assert "goes into store.apply" in reading({**base, "inverted": 1}, True)
    assert "ERG returns" in reading({**base, "inverted": 1, "uncaught": 1, "uncaught_erg_kept": 1}, True)
    assert "read it before deciding" in reading({**base, "inverted": 1, "uncaught": 1}, True)


def test_told_hives_dry_run_fixes(conn, tmp_path):
    """The negation test's dry run (DECISIONS.md): a state about an entity the keeper named but never listed lost the
    whole passage; a passage spanning a scene break came back empty."""
    p = passage(conn, 1, 1, "The yard was a square of cobbles. No wagon crossed the yard, and none ever had. "
                            "Old Brindle was the oldest horse there. She slept.")
    r = store.apply(conn, empty(states=[
        {"entity": "the-yard", "attribute": "condition", "value": "No wagon crossed the yard",
         "quote": "No wagon crossed the yard"},
        {"entity": "the yard", "attribute": "appearance", "value": "a square of cobbles", "quote": "a square of cobbles"},
        {"entity": "Old Brindle", "attribute": "appearance", "value": "the oldest horse", "quote": "the oldest horse"},
        {"entity": "she", "attribute": "condition", "value": "asleep", "quote": "She slept."},
        {"entity": "the-barn", "attribute": "condition", "value": "empty", "quote": "a square of cobbles"}]), p)
    assert r["written"]["states"] == 3
    assert r["created"] == [{"name": "the yard", "kind": "object", "from": "states", "kind_inferred_from": "condition"},
                            {"name": "Old Brindle", "kind": "object", "from": "states",
                             "kind_inferred_from": "appearance"}]
    assert sorted(d["why"] for d in r["dropped"]) == ["unresolved entity 'she'", "unresolved entity 'the-barn'"]
    assert conn.execute("SELECT value FROM state WHERE entity_id = 'the-yard' AND attribute = 'condition'"
                        ).fetchone()[0] == "No wagon crossed the yard"

    text = "Supper was quiet.\n\nNo one spoke.\n\n---\n\nLater the yard was dark.\n\n* * *\n\nMorning came."
    ps = chunk(1, text, count, target=400)
    assert [x["text"] for x in ps] == ["Supper was quiet.\n\nNo one spoke.", "Later the yard was dark.", "Morning came."]

    from rung2.keeper import EMPTY_MIN_TOKENS, keep_passage
    from rung2.record import Recorder

    class EmptyFirst(Fake):
        def __init__(self):
            self.calls = []

        def chat(self, prompt, schema=None, max_tokens=1024, temperature=0.0, seed=0):
            self.calls.append((temperature, seed))
            if len(self.calls) == 1:
                return json.dumps(empty())
            return json.dumps(empty(rules=[{"statement": "No one spoke at supper", "quote": "No one spoke"}]))

    long_text = "Supper was quiet. No one spoke. " * (EMPTY_MIN_TOKENS // 4)
    q = passage(conn, 1, 2, long_text)
    model = EmptyFirst()
    rec = Recorder(model, tmp_path / "log.jsonl")
    out = keep_passage(conn, rec, q, "")
    assert model.calls == [(0.0, 0), (0.7, 1)] and out["written"]["rules"] == 1
    events = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in events] == ["call", "call", "keeper_empty", "keeper_applied"]
    assert events[2]["used_retry"] is True
    class AlwaysEmpty(EmptyFirst):
        def chat(self, prompt, schema=None, max_tokens=1024, temperature=0.0, seed=0):
            self.calls.append((temperature, seed))
            return json.dumps(empty())

    model = AlwaysEmpty()
    rec = Recorder(model, tmp_path / "log2.jsonl")
    keep_passage(conn, rec, passage(conn, 1, 3, "She slept."), "")         # short: an empty reply is believed
    assert model.calls == [(0.0, 0)]
    keep_passage(conn, rec, passage(conn, 1, 4, long_text), "")            # long, empty twice: recorded, not retried again
    assert model.calls == [(0.0, 0), (0.0, 0), (0.7, 1)]
    last = [json.loads(x) for x in (tmp_path / "log2.jsonl").read_text().splitlines() if '"keeper_empty"' in x]
    assert last[-1]["used_retry"] is False and last[-1]["why_not"] == "empty again"
