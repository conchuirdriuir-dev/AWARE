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
    reply = {k: [] for k in ("new_entities", "states", "knowledge", "events", "rules", "loops_opened",
                             "loops_touched", "loops_closed")}
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
    store.apply(conn, empty(loops_opened=[{"description": "who Teodor locked out", "entities": [],
                                           "quote": "Who had Teodor locked out"}]), p1)
    store.apply(conn, empty(loops_touched=[1, 99]), p2)
    r = store.apply(conn, empty(loops_closed=[{"loop_id": 1, "quote": "found the archive door open"}]), p3)
    assert r["dropped"] == [] and r["written"]["loops_closed"] == 1
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
            self.reply(200, {"data": [{"id": "stand-in"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
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
        s = Server(url)
        assert s.name == "stand-in" and s.chat("hi") == "hello there friend"
        assert s.chat("hi", schema=SCHEMA) == '{"new_entities": []}' and s.schema_style == "json_schema"
        with pytest.raises(ServerError, match="HTTP 501.*does not support embeddings"):
            s.embed(["x"])
        results = {name: ok for name, ok, _ in check(url, None, SCHEMA)}
        assert results["schema: response_format"] is False and results["schema: json_schema field"] is True
        assert results["token count (POST /tokenize)"] and results["plain chat, thinking off"]
    finally:
        httpd.shutdown()
