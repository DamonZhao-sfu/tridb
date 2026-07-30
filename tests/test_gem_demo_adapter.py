"""Offline tests for the GEM wiki demo adapter. No database, no network.

The adapter is where the demo's modelling decisions live — edge orientation and
edit-comment parsing — so these tests pin the decisions rather than the code
shape. Comments used here are verbatim from the live Wikidata history of Q7251
(Alan Turing) and Q21198 (computer science), fetched 2026-07-30.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.agent_memory.demo.adapter import (
    ARTICLE_REF,
    CLASS_REF,
    REL_HYPERLINK,
    REL_MEMBER_OF,
    REL_SUBCLASS_OF,
    WikiArticle,
    WikiClass,
    WikiRevision,
    WikiSlice,
    parse_revisions,
    parse_wikidata_comment,
    slice_plan_ops,
    summarize,
    supersessions,
)


def _slice() -> WikiSlice:
    """Two articles, two classes with a subsumption edge, one hyperlink."""
    return WikiSlice(
        articles=(
            WikiArticle(
                title="Alan Turing",
                pageid=1208,
                qid="Q7251",
                extract="Alan Mathison Turing was an English mathematician. He was highly influential.",
                label="Alan Turing",
                description="English computer scientist (1912-1954)",
                classes=("Q5", "Q82594"),
            ),
            WikiArticle(
                title="Turing machine",
                pageid=30012,
                qid="Q163310",
                extract="A Turing machine is a mathematical model of computation.",
                label="Turing machine",
                description="model of computation",
                classes=("Q82594",),
            ),
        ),
        classes=(
            WikiClass(qid="Q5", label="human", parents=()),
            WikiClass(qid="Q82594", label="computer scientist", parents=("Q901",)),
            WikiClass(qid="Q901", label="scientist", parents=()),
        ),
        links=(("Alan Turing", "Turing machine"), ("Alan Turing", "Not in slice")),
    )


def _ops_of(kind: str, ops):
    return [op for op in ops if op["kind"] == kind]


def _links_of(ops, edge_kind: str, rel: str):
    return [
        op
        for op in _ops_of("link", ops)
        if op["edge_kind"] == edge_kind and op["rel"] == rel
    ]


class TestEdgeTyping:
    """The decision the demo rests on: which relation grants propagation."""

    def test_hyperlinks_are_association_never_extension(self):
        ops = slice_plan_ops(_slice(), scope_id="s", valid_from="2026-07-30T00:00:00Z")
        hyperlinks = _links_of(ops, "association", REL_HYPERLINK)
        assert len(hyperlinks) == 1
        assert hyperlinks[0]["src_ref"] == ARTICLE_REF + "Alan Turing"
        assert hyperlinks[0]["dst_ref"] == ARTICLE_REF + "Turing machine"
        assert not [
            op
            for op in _ops_of("link", ops)
            if op["rel"] == REL_HYPERLINK and op["edge_kind"] == "extension"
        ]

    def test_membership_is_extension_oriented_class_to_member(self):
        """`revise` walks OUT-edges, and the change originates at the class."""
        ops = slice_plan_ops(_slice(), scope_id="s", valid_from="2026-07-30T00:00:00Z")
        members = _links_of(ops, "extension", REL_MEMBER_OF)
        pairs = {(op["src_ref"], op["dst_ref"]) for op in members}
        assert (CLASS_REF + "Q82594", ARTICLE_REF + "Alan Turing") in pairs
        assert (CLASS_REF + "Q82594", ARTICLE_REF + "Turing machine") in pairs
        # The reverse orientation would make C3 vacuous.
        assert (ARTICLE_REF + "Alan Turing", CLASS_REF + "Q82594") not in pairs

    def test_subclass_is_extension_oriented_parent_to_child(self):
        ops = slice_plan_ops(_slice(), scope_id="s", valid_from="2026-07-30T00:00:00Z")
        pairs = {
            (op["src_ref"], op["dst_ref"])
            for op in _links_of(ops, "extension", REL_SUBCLASS_OF)
        }
        assert (CLASS_REF + "Q901", CLASS_REF + "Q82594") in pairs

    def test_edges_to_units_outside_the_slice_are_dropped(self):
        ops = slice_plan_ops(_slice(), scope_id="s", valid_from="2026-07-30T00:00:00Z")
        refs = {op["ref"] for op in _ops_of("upsert_unit", ops)}
        for op in _ops_of("link", ops):
            assert op["src_ref"] in refs
            assert op["dst_ref"] in refs


class TestUnitsAndFields:
    def test_one_unit_per_article_carrying_its_embedding_source(self):
        ops = slice_plan_ops(_slice(), scope_id="s", valid_from="2026-07-30T00:00:00Z")
        articles = [
            op
            for op in _ops_of("upsert_unit", ops)
            if op["metadata"]["kind"] == "article"
        ]
        assert len(articles) == 2
        for op in articles:
            # A run must never silently compare lead vectors with chunk vectors.
            assert op["metadata"]["embedding_source"] == "article_lead"
            assert op["metadata"]["strategy_variant"] == "wiki_article"
            assert op["embed_text"] and op["embedding"] is None

    def test_abstract_label_description_become_fields(self):
        ops = slice_plan_ops(_slice(), scope_id="s", valid_from="2026-07-30T00:00:00Z")
        fields = {(op["ref"], op["field"]) for op in _ops_of("append_field_value", ops)}
        assert (ARTICLE_REF + "Alan Turing", "abstract") in fields
        assert (ARTICLE_REF + "Alan Turing", "label") in fields
        assert (ARTICLE_REF + "Alan Turing", "description") in fields

    def test_ingest_writes_are_not_supersessions(self):
        """First write of a field is an insert; C1's index guards the rest."""
        ops = slice_plan_ops(_slice(), scope_id="s", valid_from="2026-07-30T00:00:00Z")
        assert all(
            op["supersede_current"] is False
            for op in _ops_of("append_field_value", ops)
        )

    def test_unused_classes_are_not_materialised(self):
        wiki = _slice()
        wiki.classes = wiki.classes + (WikiClass(qid="Q999", label="unused"),)
        ops = slice_plan_ops(wiki, scope_id="s", valid_from="2026-07-30T00:00:00Z")
        titles = {op["title"] for op in _ops_of("upsert_unit", ops)}
        assert "unused" not in titles
        # Reachable as an ancestor, so it stays. QID qualification prevents a
        # class label from colliding with an article title in the live store.
        assert "Wikidata class Q901: scientist" in titles

    def test_class_and_article_with_the_same_label_remain_distinct_units(self):
        wiki = WikiSlice(
            articles=(
                WikiArticle(
                    title="human",
                    pageid=1,
                    qid="QArticle",
                    extract="Human is also an article title.",
                    classes=("Q5",),
                ),
            ),
            classes=(WikiClass(qid="Q5", label="human"),),
        )
        ops = slice_plan_ops(wiki, scope_id="s", valid_from="2026-07-30T00:00:00Z")
        titles = [op["title"] for op in _ops_of("upsert_unit", ops)]
        assert titles == ["human", "Wikidata class Q5: human"]

    def test_summarize_prefers_a_sentence_end_inside_the_budget(self):
        text = "First sentence here. " + ("padding words " * 60)
        assert summarize(text, max_chars=80) == "First sentence here."

    def test_summarize_falls_back_to_a_word_boundary_never_mid_word(self):
        text = "padding words " * 60  # no sentence end at all
        summary = summarize(text, max_chars=80)
        normalized = " ".join(text.split())
        assert len(summary) <= 80
        assert normalized.startswith(summary)
        # The character right after the cut is a space, i.e. no word was split.
        assert normalized[len(summary)] == " "


class TestCommentParsing:
    """Verbatim comments from live Wikidata history."""

    def test_description_edit_extracts_language_scoped_field_exactly(self):
        edit = parse_wikidata_comment(
            WikiRevision(
                qid="Q7251",
                revid=2518790778,
                timestamp="2026-07-17T19:06:36Z",
                user="x",
                comment="/* wbsetdescription-set:1|ca */ informàtic anglès (1912–1954)",
            )
        )
        assert edit is not None
        assert edit.field == "description@ca"
        assert edit.value == "informàtic anglès (1912–1954)"
        assert edit.confidence == "exact"

    def test_claim_edit_trims_the_tool_summary_and_says_it_guessed(self):
        edit = parse_wikidata_comment(
            WikiRevision(
                qid="Q7251",
                revid=2508935897,
                timestamp="2026-06-21T22:04:58Z",
                user="x",
                comment=(
                    "/* wbsetclaim-create:1||1 */ [[Property:P13772]]: "
                    "alan-mathison-turing, Matched to [[:toollabs:something]]"
                ),
            )
        )
        assert edit is not None
        assert edit.field == "P13772"
        assert edit.value == "alan-mathison-turing"
        assert edit.confidence == "heuristic"

    def test_claim_without_a_tool_summary_is_exact(self):
        edit = parse_wikidata_comment(
            WikiRevision(
                qid="Q7251",
                revid=2503731597,
                timestamp="2026-06-09T10:41:30Z",
                user="x",
                comment="/* wbsetclaim-create:2||1 */ [[Property:P5739]]: 156220",
            )
        )
        assert edit is not None and edit.confidence == "exact"
        assert edit.value == "156220"

    @pytest.mark.parametrize(
        "comment",
        [
            "/* wbeditentity-update:0| */ QuickStatements 3.0 [[:toollabs:qs/batch/1|#1]]:",
            "restored an old revision",
            "",
        ],
    )
    def test_unparseable_comments_return_none_rather_than_a_guess(self, comment):
        assert (
            parse_wikidata_comment(
                WikiRevision(
                    qid="Q1", revid=1, timestamp="t", user="u", comment=comment
                )
            )
            is None
        )

    def test_parse_order_is_oldest_first_per_entity(self):
        revisions = [
            WikiRevision("Q7251", 20, "t2", "u", "/* wbsetdescription-set:1|ca */ new"),
            WikiRevision("Q7251", 10, "t1", "u", "/* wbsetdescription-set:1|ca */ old"),
        ]
        parsed = parse_revisions(revisions)
        assert [edit.value for edit in parsed] == ["old", "new"]

    def test_supersessions_finds_the_repeated_field(self):
        revisions = [
            WikiRevision("Q7251", 10, "t1", "u", "/* wbsetdescription-set:1|ca */ old"),
            WikiRevision("Q7251", 20, "t2", "u", "/* wbsetdescription-set:1|ca */ new"),
            WikiRevision("Q7251", 30, "t3", "u", "/* wbsetlabel-add:1|en */ once"),
        ]
        assert supersessions(parse_revisions(revisions)) == [
            ("Q7251", "description@ca", 2)
        ]


class TestSliceRoundTrip:
    def test_load_reads_what_wiki_source_writes(self, tmp_path: Path):
        (tmp_path / "manifest.json").write_text(json.dumps({"counts": {}}))
        (tmp_path / "articles.jsonl").write_text(
            json.dumps(
                {
                    "title": "Alan Turing",
                    "pageid": 1208,
                    "qid": "Q7251",
                    "extract": "text",
                    "label": "Alan Turing",
                    "description": "d",
                    "classes": ["Q5"],
                }
            )
            + "\n"
        )
        (tmp_path / "classes.jsonl").write_text(
            json.dumps({"qid": "Q5", "label": "human", "parents": []}) + "\n"
        )
        (tmp_path / "links.tsv").write_text("Alan Turing\tTuring machine\n")
        (tmp_path / "revisions.jsonl").write_text(
            json.dumps(
                {
                    "qid": "Q7251",
                    "revid": 1,
                    "timestamp": "t",
                    "user": "u",
                    "comment": "/* wbsetlabel-add:1|en */ Alan Turing",
                }
            )
            + "\n"
        )

        wiki = WikiSlice.load(tmp_path)
        assert wiki.articles[0].title == "Alan Turing"
        assert wiki.by_qid["Q7251"].classes == ("Q5",)
        assert wiki.class_by_qid["Q5"].label == "human"
        assert wiki.links == (("Alan Turing", "Turing machine"),)
        assert parse_revisions(wiki.revisions)[0].field == "label@en"


class TestRevisionStrategy:
    def test_one_qid_updates_both_article_and_class_representations(self):
        from types import SimpleNamespace

        from bench.agent_memory.demo.strategies import (
            WikidataRevisionStrategy,
            revision_events,
        )

        edit = parse_revisions(
            [
                WikiRevision(
                    "Q5",
                    1,
                    "2026-01-01T00:00:00Z",
                    "u",
                    "/* wbsetlabel-set:1|en */ human",
                )
            ]
        )

        class View:
            def units(self, scope_id, *, limit):
                return [
                    SimpleNamespace(id=10, metadata={"qid": "Q5", "kind": "article"}),
                    SimpleNamespace(id=20, metadata={"qid": "Q5", "kind": "class"}),
                ]

        ops = WikidataRevisionStrategy().plan(
            revision_events(edit, scope_id="s"), View()
        )
        assert [op["unit_id"] for op in ops] == [10, 20]


class TestHotpotResolution:
    """Coverage bookkeeping — the number that must travel with act 2's metric."""

    @staticmethod
    def _questions():
        from bench.agent_memory.demo.hotpot_link import HotpotQuestion

        return [
            HotpotQuestion(
                "q1", "both in slice?", "yes", ("Alan Turing", "Turing machine")
            ),
            HotpotQuestion(
                "q2", "half in slice?", "maybe", ("Alan Turing", "Bletchley Park")
            ),
            HotpotQuestion("q3", "neither?", "no", ("Cricket", "Tea")),
        ]

    def test_only_questions_with_every_gold_title_count_as_resolved(self):
        from bench.agent_memory.demo.hotpot_link import coverage, resolve

        resolved = resolve(self._questions(), {"Alan Turing", "Turing machine"})
        stats = coverage(resolved)
        assert stats["fully_resolved"] == 1
        assert stats["anchored_but_incomplete"] == 1
        assert stats["questions_considered"] == 3

    def test_seed_suggestions_complete_anchored_questions_only(self):
        from bench.agent_memory.demo.hotpot_link import resolve, suggest_seeds

        resolved = resolve(self._questions(), {"Alan Turing", "Turing machine"})
        seeds = suggest_seeds(resolved)
        # q2 is anchored (Alan Turing present) so its missing gold is wanted;
        # q3 touches the slice nowhere and must not drag the corpus off-domain.
        assert seeds == ["Bletchley Park"]

    def test_gold_titles_are_normalized_on_load(self, tmp_path: Path):
        from bench.agent_memory.demo.hotpot_link import load_questions

        path = tmp_path / "dev.json"
        path.write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "id": "q1",
                            "question": "q",
                            "answer": "a",
                            # Underscored form, and the same article cited for
                            # two sentences. MediaWiki only case-folds the FIRST
                            # character, so the rest of the title is left alone.
                            "supporting_facts": [
                                ["Alan_Turing", 0],
                                ["Alan_Turing", 2],
                            ],
                        }
                    ]
                }
            )
        )
        loaded = load_questions(path)
        # Same article cited twice is one gold title, and the MediaWiki key rule
        # applies so it matches what wiki_source wrote.
        assert loaded[0].gold_titles == ("Alan Turing",)


class TestExpansionTerminates:
    """Regression: the co-citation walk must not spin on unresolvable titles.

    A link target that redirects onto an article already held, or that comes
    back with no lead extract, never enters ``collected``. Filtering the
    frontier on ``collected`` alone re-proposes such a title every round, and
    the walk loops forever at a fixed count — observed live at 387/400.
    """

    def test_titles_that_never_resolve_are_not_re_requested(self, monkeypatch):
        from bench.agent_memory.demo import wiki_source

        asked: list[list[str]] = []

        def fake_fetch_articles(fetcher, titles, *, batch=20, max_link_pages=12):
            asked.append(list(titles))
            out = []
            for title in titles:
                if title == "Seed":
                    out.append(
                        wiki_source.Article(
                            title="Seed",
                            pageid=1,
                            qid=None,
                            extract="text",
                            # Ghost never comes back; Seed is its own redirect
                            # target and is already collected.
                            links=("Ghost", "Seed"),
                        )
                    )
            return out

        monkeypatch.setattr(wiki_source, "fetch_articles", fake_fetch_articles)
        collected = wiki_source.expand(
            wiki_source.Fetcher(cache_dir=Path("/nonexistent")),
            ["Seed"],
            target=50,
            log=lambda _msg: None,
        )

        assert set(collected) == {"Seed"}
        flat = [title for round_titles in asked for title in round_titles]
        assert len(flat) == len(set(flat)), f"a title was requested twice: {flat}"
