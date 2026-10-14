"""EvoTrace trajectory store v0.2 loader (trimem/data/evotrace*.py).

Unit tests need nothing. The integration test loads a 4-session heilbronn_triangle slice (one
session per backend, including a gepa run with merges) into a PRIVATE, throw-away PG16 cluster
built from .pgprefix16; it is skipped when the prefix or the dataset is absent. It never connects
to 55440 / 55441.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from trimem.data import evotrace as ev
from trimem.data import evotrace_text as tx

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / ".pgprefix16/usr/lib/postgresql/16/bin"
SLICE = (
    "349117b0:openevolve_native/ablation/heilbronn_triangle_dpsk-chat_t0.0_100_1305e2",
    "349117b0:gepa_native/heilbronn_triangle_deepseek-deepseek-reasoner_100_683ca7",
    "349117b0:shinkaevolve/heilbronn_triangle_dpsk-reasoner_100_373173",
    "349117b0:evox/heilbronn_triangle_deepseek-deepseek-reasoner_100_6f7f16",
)
A1FD = SLICE[0] + "#a1fd3e6b-67a2-4d63-9244-10133caf3e8d"

# --------------------------------------------------------------------------- unit


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://u@127.0.0.1:55441/trimem_evo",
        "host=/tmp/trimem-sock port=55440 dbname=x",
        "postgresql://u@127.0.0.1/x",  # libpq default port 5432 = system cluster
    ],
)
def test_guard_refuses_production(dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGPORT", raising=False)
    with pytest.raises(SystemExit):
        ev.guard_dsn(dsn)


def test_guard_honours_pgport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGPORT", "55441")
    with pytest.raises(SystemExit):
        ev.guard_dsn("host=/tmp/x dbname=y")
    monkeypatch.setenv("PGPORT", "55450")
    assert ev.guard_dsn("host=/tmp/x dbname=y")["dbname"] == "y"


def test_extract_intent_formats() -> None:
    diff = "I will vectorise the loop.\n\n<<<<<<< SEARCH\nfor i in x:\n=======\nnp.sum(x)\n>>>>>>> REPLACE"
    assert tx.extract_intent(diff) == "I will vectorise the loop."
    shinka = (
        "<NAME>fix</NAME>\n<DESCRIPTION>\n  Break collinearity.\n</DESCRIPTION>\n<<<"
    )
    assert tx.extract_intent(shinka) == "Break collinearity."
    fence = "Rewrite with a grid.\n```python\nprint(1)\n```"
    assert tx.extract_intent(fence) == "Rewrite with a grid."
    assert tx.extract_intent("<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE") is None
    assert tx.extract_intent("plain text with no edit block") is None
    assert tx.extract_intent(None) is None


def test_pair_links_exact_and_duplicates() -> None:
    a, b, c = tx.sha1("A"), tx.sha1("B"), tx.sha1("C")
    links = tx.pair_links([a, None, b, b, c, b], {a: [10], b: [20, 21], "zz": [30]})
    assert links[0] == tx.Link(10, "response_sha1", (10,))
    assert links[1] == links[4] == tx.Link(None, None, ())
    # Repeated identical responses do not prove which call produced which node.
    for i in (2, 3, 5):
        assert links[i] == tx.Link(None, "response_sha1_ambiguous", (20, 21))
    assert tx.sha1(" A \n") == a  # whitespace-insensitive


def test_membership_rle_uses_checkpoint_sequence() -> None:
    rle = tx.MembershipRLE()

    def ev_(it: int, node: str, value: object = None) -> dict:
        return {
            "session_uid": "s",
            "iteration": it,
            "role": "archive",
            "slot_key": 0,
            "node_uid": node,
            "value": value,
        }

    # checkpoints every 10 iterations; x present at 10,20,30 then 50 (gap at 40)
    for it, node in [(10, "x"), (20, "x"), (30, "x"), (40, "y"), (50, "x")]:
        rle.add(ev_(it, node))
    rle.add(ev_(10, "z", 1.0))
    rle.add(ev_(20, "z", 2.0))  # value change -> new interval
    got = sorted((n, lo, hi, v) for _, _, _, n, lo, hi, v in rle.intervals())
    assert got == [
        ("x", 10, 30, None),
        ("x", 50, 50, None),
        ("y", 40, 40, None),
        ("z", 10, 10, "1.0"),
        ("z", 20, 20, "2.0"),
    ]
    assert list(rle.checkpoint_rows()) == [
        ("s", 10),
        ("s", 20),
        ("s", 30),
        ("s", 40),
        ("s", 50),
    ]


def test_shinka_init_has_no_intent(tmp_path: Path) -> None:
    prog = {
        "metadata": {"patch_type": "init", "patch_description": "Initial program setup"}
    }
    assert tx.node_text({}, tmp_path, prog, None) is None
    prog = {
        "metadata": {
            "patch_type": "fix",
            "patch_description": "Break ties.",
            "llm_result": {"content": "<NAME>a</NAME>", "msg": "u", "system_msg": "s"},
        }
    }
    t = tx.node_text({}, tmp_path, prog, None)
    assert (t.intent, t.intent_source, t.user, t.source) == (
        "Break ties.",
        "shinka_patch_description",
        "u",
        "shinka_llm_result",
    )


# --------------------------------------------------------------------------- fixture slice


def _filter_jsonl(src: Path, dst: Path, keep) -> int:
    n = 0
    with src.open() as f, dst.open("w") as out:
        for line in f:
            if keep(json.loads(line)):
                out.write(line)
                n += 1
    return n


def build_slice(out: Path, sessions: tuple[str, ...] = SLICE) -> tuple[Path, Path]:
    """A self-consistent normalized + raw slice of the real dataset (raw runs are symlinked)."""
    src, raw = ev.SRC, ev.RAW
    norm, rawo = out / "normalized", out / "raw"
    norm.mkdir(parents=True)
    keep = set(sessions)
    by_s = lambda r: r["session_uid"] in keep
    for f in (
        "sessions",
        "nodes",
        "lineage_edges",
        "context_edges",
        "prompts",
        "llm_calls",
        "state_events",
    ):
        _filter_jsonl(src / f"{f}.jsonl", norm / f"{f}.jsonl", by_s)
    tasks = {json.loads(line)["task_uid"] for line in (norm / "sessions.jsonl").open()}
    _filter_jsonl(
        src / "tasks.jsonl", norm / "tasks.jsonl", lambda r: r["task_uid"] in tasks
    )
    arts = {json.loads(line)["artifact_uid"] for line in (norm / "nodes.jsonl").open()}
    _filter_jsonl(
        src / "artifacts.jsonl",
        norm / "artifacts.jsonl",
        lambda r: r["artifact_uid"] in arts,
    )
    uids = {
        json.loads(line)["node_uid"] for line in (norm / "nodes.jsonl").open()
    } | tasks
    z = np.load(src / "vectors.npz", allow_pickle=True)
    m = np.array([u in uids for u in z["uids"]])
    np.savez(
        norm / "vectors.npz",
        uids=z["uids"][m],
        kinds=z["kinds"][m],
        sources=z["sources"][m],
        vectors=z["vectors"][m],
    )
    for s in sessions:
        rel = s.split(":", 1)[1]
        (rawo / rel).parent.mkdir(parents=True, exist_ok=True)
        (rawo / rel).symlink_to(raw / rel)
    return norm, rawo


def fake_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        rng = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16))
        v = rng.standard_normal(1024)
        out.append((v / np.linalg.norm(v)).tolist())
    return out


@pytest.fixture(scope="module")
def private_pg() -> Iterator[str]:
    if not (BIN / "initdb").exists() or not (ev.SRC / "nodes.jsonl").exists():
        pytest.skip("needs .pgprefix16 and dataset/evotrace")
    root = Path(tempfile.mkdtemp(prefix="trj"))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert port not in ev.FORBIDDEN_PORTS
    data = root / "data"
    subprocess.run(
        [BIN / "initdb", "-D", data, "-A", "trust", "--no-locale", "-E", "UTF8"],
        check=True,
        capture_output=True,
    )
    opts = (
        f"-k {root} -p {port} -c listen_addresses= -c shared_buffers=128MB "
        "-c shared_preload_libraries=tjs_pg,trimem_pg -c max_parallel_workers=0 "
        "-c autovacuum=off"
    )
    subprocess.run(
        [BIN / "pg_ctl", "-D", data, "-l", root / "log", "-o", opts, "-w", "start"],
        check=True,
        capture_output=True,
    )
    try:
        subprocess.run(
            [BIN / "createdb", "-h", root, "-p", str(port), "traj"], check=True
        )
        yield f"host={root} port={port} dbname=traj"
    finally:
        subprocess.run(
            [BIN / "pg_ctl", "-D", data, "-m", "fast", "-w", "stop"],
            capture_output=True,
            check=False,
        )
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def loaded(
    private_pg: str, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, Path, dict]:
    norm, raw = build_slice(tmp_path_factory.mktemp("slice"))
    r = ev.load(private_pg, norm, raw, similar_k=4, embed=fake_embed)
    return private_pg, norm, r


def _q(dsn: str, sql: str, *args: object) -> list[tuple]:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as c:
        return c.execute(sql, args or None).fetchall()


def _lines(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()]


# --------------------------------------------------------------------------- integration


def test_receipt_matches_slice(loaded: tuple[str, Path, dict]) -> None:
    dsn, norm, r = loaded
    assert r["ok"], r["checks"]
    nodes = _lines(norm / "nodes.jsonl")
    assert r["attempt"] == len(nodes) == r["attempt_emb"] == r["attempt_with_code"]
    assert r["task"] == r["task_emb"] == 1 and r["session"] == 4
    e = r["edges_mirrored"]
    assert e["lineage"] == len(_lines(norm / "lineage_edges.jsonl"))
    assert e["merge_parent"] == 6  # the gepa 683ca7 run's 'Merge Parent B' labels
    assert e["context"] + e["merge_parent"] == len(_lines(norm / "context_edges.jsonl"))
    assert r["llm_call"] == len(_lines(norm / "llm_calls.jsonl"))
    assert r["intent_emb"] == r["intent"] > 0
    # every node with a resolved response links to exactly one call (exact sha1 only)
    # every resolved response has an exact-content call in its session; a response shared by
    # more nodes than calls links only as many nodes as there are calls
    orphan = _q(
        dsn,
        """SELECT count(*) FROM prompt p JOIN attempt a ON a.id = p.attempt_id
                        WHERE p.response_sha1 IS NOT NULL AND NOT EXISTS (
                          SELECT 1 FROM llm_call c WHERE c.session = a.session
                          AND c.content_sha1 = p.response_sha1)""",
    )
    assert orphan == [(0,)]
    ((linked, expect),) = _q(
        dsn,
        """
        WITH n AS (SELECT a.session, p.response_sha1 h, count(*) k FROM prompt p
                   JOIN attempt a ON a.id = p.attempt_id WHERE p.response_sha1 IS NOT NULL
                   GROUP BY 1, 2),
             c AS (SELECT session, content_sha1 h, count(*) k FROM llm_call GROUP BY 1, 2)
        SELECT (SELECT count(attempt_id) FROM llm_call),
               (SELECT count(*) FROM n JOIN c USING (session, h) WHERE n.k=1 AND c.k=1)""",
    )
    assert linked == expect
    assert set(r["llm_call_link_methods"]) <= {"response_sha1"}


def test_native_edges_and_reverse_symmetry(loaded: tuple[str, Path, dict]) -> None:
    dsn, _, r = loaded
    assert r["edges_native"]["context"] == r["edges_mirrored"]["context"]
    asym = _q(
        dsn,
        """
        SELECT count(*) FROM edge_prop f JOIN graph_store.edge_type tf ON tf.id = f.edge_type
        WHERE tf.name IN ('lineage','context','merge_parent','similar') AND NOT EXISTS (
          SELECT 1 FROM edge_prop u JOIN graph_store.edge_type tu ON tu.id = u.edge_type
          WHERE tu.name = tf.name || '_up' AND u.src = f.dst AND u.dst = f.src)""",
    )
    assert asym == [(0,)]
    # context never duplicates the parent edge; similar never stays inside a session
    assert _q(
        dsn,
        """SELECT count(*) FROM edge_prop e JOIN graph_store.edge_type t
                      ON t.id = e.edge_type JOIN attempt d ON d.id = e.dst
                      WHERE t.name = 'context' AND d.parent_id = e.src""",
    ) == [(0,)]
    assert _q(
        dsn,
        """SELECT count(*) FROM edge_prop e JOIN graph_store.edge_type t
                      ON t.id = e.edge_type JOIN attempt a ON a.id = e.src JOIN attempt b
                      ON b.id = e.dst WHERE t.name = 'similar' AND a.session = b.session""",
    ) == [(0,)]


def test_context_traversal_through_expand(loaded: tuple[str, Path, dict]) -> None:
    dsn, _, _ = loaded
    sql = """SELECT d.id FROM attempt s, LATERAL trimem.expand(s.vid, 'context_up', 8) e
             JOIN attempt d ON d.vid = e.dst WHERE s.uid = %s"""
    got = {i for (i,) in _q(dsn, sql, A1FD)}
    want = {
        i
        for (i,) in _q(
            dsn,
            """SELECT e.dst FROM edge_prop e JOIN graph_store.edge_type t
            ON t.id = e.edge_type JOIN attempt a ON a.id = e.src
            WHERE a.uid = %s AND t.name = 'context_up'""",
            A1FD,
        )
    }
    assert got == want and len(got) >= 1


def test_intent_and_llm_call_join(loaded: tuple[str, Path, dict]) -> None:
    dsn, _, _ = loaded
    ((intent, src, fw),) = _q(
        dsn,
        "SELECT intent, intent_source, framework_changes FROM attempt WHERE uid = %s",
        A1FD,
    )
    assert src == "response_prose" and intent.startswith(
        "Looking at the current solution"
    )
    assert "Very slow execution" in intent and fw.startswith("Change 1:")
    rows = _q(
        dsn,
        """SELECT c.purpose, c.completion_tokens, p.response_text = c.content
                      FROM attempt a JOIN llm_call c ON c.attempt_id = a.id
                      JOIN prompt p ON p.attempt_id = a.id WHERE a.uid = %s""",
        A1FD,
    )
    assert rows and rows[0][0] == "mutation" and rows[0][2]
    # every shinka node but the iteration-0 'init' root and its island copies has an intent
    shinka = _q(
        dsn,
        "SELECT count(*) FILTER (WHERE intent_source = 'shinka_patch_description'), "
        "count(*) FILTER (WHERE iteration > 0) FROM attempt "
        "WHERE backend = 'shinkaevolve'",
    )
    assert shinka[0][0] == shinka[0][1] > 0


def test_not_selected_excludes_parent_and_context(
    loaded: tuple[str, Path, dict],
) -> None:
    dsn, _, _ = loaded
    ((aid, parent),) = _q(dsn, "SELECT id, parent_id FROM attempt WHERE uid = %s", A1FD)
    cands = {
        c for (c,) in _q(dsn, "SELECT DISTINCT candidate_id FROM not_selected(%s)", aid)
    }
    ctx = {
        s
        for (s,) in _q(
            dsn,
            """SELECT e.src FROM edge_prop e JOIN graph_store.edge_type t
                                    ON t.id = e.edge_type WHERE e.dst = %s AND t.name = 'context'""",
            aid,
        )
    }
    assert cands and parent not in cands and aid not in cands and not (cands & ctx)


def test_rerun_is_idempotent(loaded: tuple[str, Path, dict], tmp_path: Path) -> None:
    dsn, norm, first = loaded
    again = ev.load(dsn, norm, norm.parent / "raw", similar_k=4, embed=fake_embed)
    assert again["ok"] and again["edges_native"] == first["edges_native"]
    assert again["inserted"] == {}  # no new native edges on the second run
    for k in ("attempt", "prompt", "llm_call", "search_membership", "intent_emb"):
        assert again[k] == first[k]


def test_refuses_a_different_dataset(
    loaded: tuple[str, Path, dict], tmp_path: Path
) -> None:
    dsn, _, _ = loaded
    norm, raw = build_slice(tmp_path / "other", SLICE[:2])
    with pytest.raises(SystemExit):
        ev.load(dsn, norm, raw, similar_k=4)


def test_trimodal_plan_streams(loaded: tuple[str, Path, dict]) -> None:
    """HNSW on intent_emb -> native context expand -> relational filter, no blocking node."""
    import psycopg

    dsn, _, _ = loaded
    q = "[" + ",".join(map(repr, fake_embed(["vectorise the loop"])[0])) + "]"
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SET enable_seqscan = off")
        plan = "\n".join(
            r[0]
            for r in c.execute(
                "EXPLAIN SELECT d.id, d.fitness FROM (SELECT vid FROM attempt "
                "WHERE intent_emb IS NOT NULL ORDER BY intent_emb <=> %s::vector LIMIT 8) s, "
                "LATERAL trimem.expand(s.vid, 'context', 8) e JOIN attempt d ON d.vid = e.dst "
                "WHERE d.is_valid LIMIT 5",
                (q,),
            )
        )
    assert "attempt_intent_hnsw" in plan and "TriMEM Expand" in plan, plan
    assert "Sort" not in plan and "Materialize" not in plan and "Hash" not in plan, plan


def test_unified_context_preserves_merge_labels(loaded):
    from trimem.data import evotrace_graph as gr

    _, norm, _ = loaded
    ds = ev.Dataset(norm, norm.parent / "raw")
    split = gr.context_edges(ds)
    unified = gr.context_edges(ds, merge_as_context=True)
    assert len(unified) == len(split)
    assert {e.etype for e in unified} == {"context", "context_up"}
    assert (
        sum(e.label == gr.MERGE_LABEL for e in unified)
        == sum(e.etype in ("merge_parent", "merge_parent_up") for e in split)
        > 0
    )
    assert {(e.src, e.dst, e.label) for e in split} == {
        (e.src, e.dst, e.label) for e in unified
    }


def test_refuses_changed_graph_policy_before_truncate(loaded):
    dsn, norm, first = loaded
    with pytest.raises(SystemExit, match="graph policy"):
        ev.load(dsn, norm, norm.parent / "raw", similar_k=4, merge_as_context=True)
    assert _q(dsn, "SELECT count(*) FROM attempt") == [(first["attempt"],)]


def test_new_schema_semantic_input_and_fallback(loaded):
    from trimem.trajectory_query import evidence, query

    dsn, _, _ = loaded
    expression = evidence("a", "p", "s")
    rows = _q(
        dsn,
        "SELECT " + expression + ", a.intent, a.framework_changes FROM attempt a "
        "JOIN attempt p ON p.id=a.parent_id JOIN session s ON s.session=a.session "
        "WHERE a.intent IS NOT NULL LIMIT 1",
    )
    text, intent, changes = rows[0]
    assert intent[:2400] in text
    assert (changes or "")[:1200] in text
    assert all(
        label in text
        for label in ("intent_source=", "framework_changes=", "parent_fitness=")
    )
    assert "a.code" not in expression
    for plan in ("SemRankOnly", "P1+Rank", "P2+Rank"):
        sql = query(plan, "intent_emb")
        assert "trimem.sem_score(%(score_predicate)s," in sql
        assert ".changes" not in sql and "framework_changes" in sql
    for bad in ("DROP TABLE attempt", "other_column"):
        with pytest.raises(ValueError):
            query("P1", bad)
    fallback = _q(
        dsn,
        "SELECT " + evidence("a", "p", "s") + " FROM "
        "(SELECT NULL::text AS intent,NULL::text AS intent_source,'actual diff'::text AS framework_changes,1.0 AS fitness,'python'::text AS language) a, "
        "(SELECT 0.5 AS fitness) p, (SELECT 'test'::text AS backend) s",
    )[0][0]
    assert "intent_source=missing" in fallback and "intent=actual diff" in fallback


def test_context_sql_text_equals_preview(loaded):
    import psycopg

    from trimem.trajectory_query import judge_input, render_judge_input

    dsn, _, _ = loaded
    params = {
        "current_context": '{"objective":"fix boundary","failure_reason":"outside"}',
        "candidate": "intent=move inward",
    }
    with psycopg.connect(dsn) as conn:
        got = conn.execute(
            "SELECT " + judge_input("%(candidate)s::text"), params
        ).fetchone()[0]
        assert got == render_judge_input(params["current_context"], params["candidate"])
        with pytest.raises(psycopg.ProgrammingError):
            conn.execute(
                "SELECT " + judge_input("%(candidate)s::text"), {"candidate": "alone"}
            )
