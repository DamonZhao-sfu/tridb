"""EvoTrace dataset access for the trajectory-store loader: the normalized files that are small
enough to hold, per-session raw run access, completeness expectations and input fingerprints.

Missing inputs fail loudly: a raw `programs.jsonl`, a prompt blob that prompts.jsonl declares
present, or a code blob that a node references raises instead of being silently skipped. What
the dataset itself declares absent (prompts.jsonl `blob_present: false`, a node without an
artifact or vector) is counted per backend and gated in the receipt.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from trimem.data import evotrace_text as tx


def clean(x: object) -> object:
    """Postgres text cannot hold NUL; the dataset has a few."""
    return x.replace("\x00", "") if isinstance(x, str) else x


def as_float(x: object) -> float | None:
    try:
        v = None if x is None else float(x)
    except (TypeError, ValueError):
        return None
    return v if v is None or np.isfinite(v) else None


def vec(v: np.ndarray | None) -> str | None:
    return None if v is None else "[" + ",".join(map(repr, v.tolist())) + "]"


def jsonb(v: object) -> str | None:
    return None if v is None else json.dumps(v)


def _sha_file(h: hashlib._Hash, path: Path) -> None:
    h.update(path.name.encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)


class Dataset:
    def __init__(self, src: Path, raw: Path) -> None:
        self.src, self.raw = src, raw
        self.tasks = list(tx.read_jsonl(src / "tasks.jsonl"))
        self.sessions = list(tx.read_jsonl(src / "sessions.jsonl"))
        self.nodes = list(tx.read_jsonl(src / "nodes.jsonl"))
        self.ids = {n["node_uid"]: i + 1 for i, n in enumerate(self.nodes)}
        self.by_session: dict[str, list[dict]] = defaultdict(list)
        for n in self.nodes:
            self.by_session[n["session_uid"]].append(n)
        self.run_rel = {s["session_uid"]: s["run_rel"] for s in self.sessions}
        self.backend = {s["session_uid"]: s["backend"] for s in self.sessions}
        prompts = list(tx.read_jsonl(src / "prompts.jsonl"))
        self.prompt_rel = {
            p["node_uid"]: p["blob_rel"] for p in prompts if p["blob_present"]
        }
        self.prompt_declared_missing = Counter(
            self.node_backend(p["node_uid"]) for p in prompts if not p["blob_present"]
        )
        self.artifacts = {
            a["artifact_uid"]: a for a in tx.read_jsonl(src / "artifacts.jsonl")
        }
        z = np.load(src / "vectors.npz", allow_pickle=True)
        self.vectors = {
            (str(k), str(u)): v
            for u, k, v in zip(z["uids"], z["kinds"], z["vectors"], strict=True)
        }
        report = src / "integrity_report.json"
        self.integrity = json.loads(report.read_text()) if report.is_file() else None
        self.llm_result_expected: Counter = Counter()  # filled by each texts() pass

    def node_backend(self, node_uid: str) -> str:
        return self.backend[node_uid.split("#", 1)[0]]

    def run_dir(self, session: str) -> Path:
        return self.raw / self.run_rel[session]

    def raw_programs(self, session: str) -> dict[str, dict]:
        path = self.run_dir(session) / "programs.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"raw programs.jsonl missing: {path}")
        return {r["id"]: r for r in tx.read_jsonl(path)}

    def code(self, node: dict) -> str | None:
        """None only when the node references no artifact (counted); a referenced artifact
        whose blob is in none of its sessions' run dirs raises."""
        if not node.get("artifact_uid"):
            return None
        art = self.artifacts[node["artifact_uid"]]
        for s in dict.fromkeys([node["session_uid"], *art.get("sessions", [])]):
            if s in self.run_rel and (p := self.run_dir(s) / art["blob_rel"]).is_file():
                return p.read_text(errors="replace")
        raise FileNotFoundError(
            f"code blob missing for {node['node_uid']}: {art['blob_rel']}"
        )

    def texts(self) -> Iterator[tuple[dict, tx.NodeText | None]]:
        """(node, resolved prompt/response) in session order, one raw programs file at a time."""
        self.llm_result_expected = Counter()
        for s in self.run_rel:
            shinka = self.backend[s] == "shinkaevolve"
            progs = self.raw_programs(s) if shinka else {}
            for n in self.by_session.get(s, []):
                if shinka and n["program_id"] not in progs:
                    raise KeyError(f"{n['node_uid']} not in raw programs.jsonl")
                md = (progs.get(n["program_id"]) or {}).get("metadata") or {}
                self.llm_result_expected[self.backend[s]] += isinstance(
                    md.get("llm_result"), dict
                )
                yield (
                    n,
                    tx.node_text(
                        n,
                        self.run_dir(s),
                        progs.get(n["program_id"]),
                        self.prompt_rel.get(n["node_uid"]),
                    ),
                )

    def expectations(self) -> dict[str, dict[str, int]]:
        """Per backend, what the normalized files say must be loaded."""
        exp: dict[str, Counter] = defaultdict(Counter)
        for n in self.nodes:
            b = exp[self.backend[n["session_uid"]]]
            b["nodes"] += 1
            b["code"] += bool(n.get("artifact_uid"))
            b["emb"] += ("node", n["node_uid"]) in self.vectors
            b["prompt_blob"] += n["node_uid"] in self.prompt_rel
        for b, c in self.prompt_declared_missing.items():
            exp[b]["prompt_blob_declared_missing"] = c
        for b, c in self.llm_result_expected.items():
            exp[b]["prompt_llm_result"] = c
        return {b: dict(c) for b, c in exp.items()}

    def graph_fingerprint(self, similar_k: int) -> str:
        """Every input that shapes the native graph: lineage / context edge files, the node set,
        the vectors (kNN), every raw programs.jsonl (merge labels, context order) and k."""
        h = hashlib.sha256(f"similar_k={similar_k}".encode())
        for name in (
            "nodes.jsonl",
            "lineage_edges.jsonl",
            "context_edges.jsonl",
            "vectors.npz",
        ):
            _sha_file(h, self.src / name)
        for s in self.run_rel:
            h.update(s.encode())
            _sha_file(h, self.run_dir(s) / "programs.jsonl")
        return h.hexdigest()
