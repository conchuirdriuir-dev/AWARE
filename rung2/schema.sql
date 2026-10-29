-- The store for arm D (SPEC.md, "The store"). Every row but chapter_line cites its passage.

CREATE TABLE passage (
  id        TEXT PRIMARY KEY,          -- 'c03-p012'
  chapter   INTEGER NOT NULL,
  seq       INTEGER NOT NULL,          -- order within the chapter
  text      TEXT NOT NULL,
  tokens    INTEGER NOT NULL           -- counted with the test model's tokenizer
);

CREATE TABLE entity (
  id        TEXT PRIMARY KEY,          -- 'mara', 'locket-silver'
  name      TEXT NOT NULL,
  kind      TEXT NOT NULL CHECK (kind IN ('character','object','place','group')),
  aliases   TEXT NOT NULL DEFAULT '[]', -- JSON list: 'the girl', 'Teo'
  first_passage TEXT NOT NULL REFERENCES passage(id)
);

CREATE TABLE state (
  id          INTEGER PRIMARY KEY,
  entity_id   TEXT NOT NULL REFERENCES entity(id),
  attribute   TEXT NOT NULL,           -- 'eye colour', 'location', 'holder'
  value       TEXT NOT NULL,
  from_chapter INTEGER NOT NULL,
  supersedes  INTEGER REFERENCES state(id),
  passage_id  TEXT NOT NULL REFERENCES passage(id)
);

CREATE TABLE knowledge (
  id          INTEGER PRIMARY KEY,
  character_id TEXT NOT NULL REFERENCES entity(id),
  proposition TEXT NOT NULL,           -- 'Teodor took the map'
  about_id    TEXT REFERENCES entity(id),
  stance      TEXT NOT NULL CHECK (stance IN ('knows','believes')),
  true_in_story INTEGER,               -- 0 for a false belief, NULL if not yet clear
  from_chapter INTEGER NOT NULL,
  corrected_by INTEGER REFERENCES knowledge(id),
  passage_id  TEXT NOT NULL REFERENCES passage(id)
);

CREATE TABLE event (
  id          INTEGER PRIMARY KEY,
  chapter     INTEGER NOT NULL,
  summary     TEXT NOT NULL,           -- one line
  place_id    TEXT REFERENCES entity(id),  -- when the place is a known entity
  place_text  TEXT,                        -- where, as the keeper said it: 'near the fence'
  happened    INTEGER NOT NULL DEFAULT 1, -- 0: planned, threatened or cancelled
  passage_id  TEXT NOT NULL REFERENCES passage(id)
);

CREATE TABLE event_entity (
  event_id    INTEGER NOT NULL REFERENCES event(id),
  entity_id   TEXT NOT NULL REFERENCES entity(id),
  PRIMARY KEY (event_id, entity_id)
);

CREATE TABLE rule (
  id          INTEGER PRIMARY KEY,
  statement   TEXT NOT NULL,
  chapter     INTEGER NOT NULL,
  passage_id  TEXT NOT NULL REFERENCES passage(id)
);

CREATE TABLE loop (
  id          INTEGER PRIMARY KEY,
  description TEXT NOT NULL,           -- 'the locket Mara hid'
  opened_chapter INTEGER NOT NULL,
  opened_passage TEXT NOT NULL REFERENCES passage(id),
  closed_chapter INTEGER,              -- NULL while open
  closed_passage TEXT REFERENCES passage(id)
);

CREATE TABLE loop_touch (              -- every passage that opens, mentions or closes a loop: "as of chapter N" needs them
  loop_id     INTEGER NOT NULL REFERENCES loop(id),
  chapter     INTEGER NOT NULL,
  passage_id  TEXT NOT NULL REFERENCES passage(id),
  PRIMARY KEY (loop_id, passage_id)
);

CREATE TABLE loop_entity (
  loop_id     INTEGER NOT NULL REFERENCES loop(id),
  entity_id   TEXT NOT NULL REFERENCES entity(id),
  PRIMARY KEY (loop_id, entity_id)
);

CREATE TABLE chapter_line (
  chapter     INTEGER PRIMARY KEY,
  line        TEXT NOT NULL            -- written from the chapter's raw text, never from earlier lines
);
