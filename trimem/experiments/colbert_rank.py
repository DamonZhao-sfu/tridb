"""ColBERT ranker client: the shared "Rank" operator for VGR+Rank and VGR+SemFilter+Rank (P2+Rank)
on every dataset. Satisfies the ``semrank`` contract of ``session_retrieval.retrieve``
(cards -> {id: score}); no LLM tokens. ``calls`` records the server latency per request.
"""

from __future__ import annotations

import time

import httpx

VERSION = "colbert-rank-v1:lightonai/GTE-ModernColBERT-v1"


def rank_base_for(llm_base: str) -> str:
    """Each evaluation half ranks on its own GPU: 8030 -> 8040, 8031 -> 8041."""
    port = int(llm_base.rstrip("/").removesuffix("/v1").rsplit(":", 1)[1])
    return f"http://127.0.0.1:{port + 10}"


class ColBERTRank:
    def __init__(
        self,
        query: str,
        render,
        base: str = "http://127.0.0.1:8040",
        max_chars: int = 12000,
    ):
        self.query, self.render, self.base, self.max_chars = (
            query,
            render,
            base.rstrip("/"),
            max_chars,
        )
        self.calls: list[dict] = []

    def __call__(self, cards: list[dict]) -> dict:
        if not cards:
            return {}
        docs = [
            {"id": str(c["id"]), "text": self.render(c)[: self.max_chars]}
            for c in cards
        ]
        t0 = time.monotonic()
        r = httpx.post(
            f"{self.base}/rank",
            json={"query": self.query, "documents": docs},
            timeout=600,
        )
        r.raise_for_status()
        j = r.json()
        self.calls.append(
            {
                "usage": {
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
                "seconds": round(time.monotonic() - t0, 3),
                "server_ms": j.get("ms"),
                "model": j.get("model"),
                "kind": "colbert",
            }
        )
        return {c["id"]: j["scores"][str(c["id"])] for c in cards}
