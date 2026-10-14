"""Rebuild the isolated trajectory corpus, with resumable full-code embeddings.

Native adjacency is updated transactionally together with code vectors. Lineage
and context are preserved. The old vectors and similarity properties are backed
up before mutation. Never targets legacy experiment databases.
"""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psycopg

from trimem.code_embedding import REVISION, VERSION, FullCodeEmbedder
from trimem.data.evotrace_graph import insert_edges, similar_edges


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dsn", default="host=/tmp/trimem-traj port=55450 dbname=trimem_traj"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with psycopg.connect(args.dsn, autocommit=True) as c:
        db, port = c.execute(
            "select current_database(), current_setting('port')"
        ).fetchone()
        if db != "trimem_traj" or port != "55450":
            raise ValueError("rebuild is restricted to isolated trimem_traj:55450")
        rows = c.execute(
            "select id,uid,task,session,code,emb::text from attempt order by id"
        ).fetchall()
        digest = hashlib.sha256()
        for r in rows:
            digest.update(json.dumps([r[0], r[4]], ensure_ascii=False).encode())
        corpus_sha = digest.hexdigest()
        manifest = args.out / "input.json"
        identity = {
            "corpus_sha256": corpus_sha,
            "embedding_version": VERSION,
            "revision": REVISION,
        }
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError(
                "staging files belong to a different corpus or embedding contract"
            )
        manifest.write_text(json.dumps(identity, indent=2))
        backup = args.out / "before.json.gz"
        if not backup.exists():
            edges = c.execute(
                "select e.src,e.dst,t.name,e.label,e.ordinal,e.weight,e.provenance from edge_prop e join graph_store.edge_type t on t.id=e.edge_type where t.name in ('similar','similar_up') order by src,t.name,ordinal"
            ).fetchall()
            with gzip.open(backup, "wt") as f:
                json.dump({"vectors": [(r[0], r[5]) for r in rows], "edges": edges}, f)
        vecdir = args.out / "vectors"
        vecdir.mkdir(exist_ok=True)
        model = None
        receipts = []
        for n, r in enumerate(rows):
            path = vecdir / f"{r[0]}.npz"
            if not path.exists():
                if model is None:
                    model = FullCodeEmbedder()
                vector = model.embed(r[4])
                tmp = path.with_suffix(".tmp.npz")
                np.savez(tmp, vector=vector, receipt=json.dumps(model.last_receipt))
                tmp.replace(path)
            with np.load(path) as z:
                receipt = json.loads(str(z["receipt"]))
                if receipt["sha256"] != hashlib.sha256(r[4].encode()).hexdigest():
                    raise ValueError(f"stale vector {r[0]}")
                receipts.append(receipt)
            if n % 100 == 0:
                print(
                    f"embedded {n + 1}/{len(rows)} elapsed={time.monotonic() - start:.1f}s",
                    flush=True,
                )
        # Release GPU before native graph construction.
        del model
        vectors = {
            ("node", r[1]): np.load(vecdir / f"{r[0]}.npz")["vector"] for r in rows
        }
        ds = SimpleNamespace(
            nodes=[
                {"node_uid": r[1], "task_uid": r[2], "session_uid": r[3]} for r in rows
            ],
            ids={r[1]: r[0] for r in rows},
            vectors=vectors,
        )
        edges = similar_edges(ds, 16)
        edges = [e._replace(provenance=f"derived:knn_{VERSION}") for e in edges]
        with c.transaction():
            c.execute("LOCK TABLE attempt, edge_prop IN EXCLUSIVE MODE")
            current = c.execute("select id,code from attempt order by id").fetchall()
            check = hashlib.sha256()
            for item in current:
                check.update(json.dumps(list(item), ensure_ascii=False).encode())
            if check.hexdigest() != corpus_sha:
                raise ValueError(
                    "corpus changed during embedding; refusing publication"
                )
            c.execute("CREATE TEMP TABLE code_vectors(id int, emb vector(1024))")
            with c.cursor().copy("COPY code_vectors FROM STDIN") as cp:
                for r in rows:
                    cp.write_row(
                        (r[0], "[" + ",".join(map(str, vectors["node", r[1]])) + "]")
                    )
            c.execute(
                "UPDATE attempt a SET emb=v.emb FROM code_vectors v WHERE a.id=v.id"
            )
            preserved = c.execute(
                "select count(*) from edge_prop e join graph_store.edge_type t on t.id=e.edge_type where t.name not in ('similar','similar_up')"
            ).fetchone()[0]
            c.execute(
                "select count(graph_store.gph_tombstone_edge(s.vid,d.vid,e.edge_type)) from edge_prop e join graph_store.edge_type t on t.id=e.edge_type join attempt s on s.id=e.src join attempt d on d.id=e.dst where t.name in ('similar','similar_up')"
            ).fetchone()
            c.execute(
                "delete from edge_prop where edge_type in (select id from graph_store.edge_type where name in ('similar','similar_up'))"
            )
            inserted = insert_edges(c, edges)
            c.execute(
                "ALTER TABLE load_meta ADD COLUMN IF NOT EXISTS code_embedding jsonb"
            )
            meta = identity | {
                "rows": len(rows),
                "max_tokens": max(r["tokens"] for r in receipts),
                "truncated": 0,
                "format": "raw full code",
                "similar_k": 16,
                "same_task": True,
                "cross_session": True,
            }
            c.execute(
                "UPDATE load_meta SET code_embedding=%s::jsonb", (json.dumps(meta),)
            )
            after = c.execute(
                "select count(*) from edge_prop e join graph_store.edge_type t on t.id=e.edge_type where t.name not in ('similar','similar_up')"
            ).fetchone()[0]
            if preserved != after:
                raise ValueError("lineage/context changed")
        c.execute("ANALYZE attempt")
        c.execute("SELECT trimem.analyze_graph()")
        result = meta | {
            "inserted": inserted,
            "preserved_non_similarity_edges": preserved,
            "elapsed_seconds": time.monotonic() - start,
        }
        (args.out / "receipt.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
