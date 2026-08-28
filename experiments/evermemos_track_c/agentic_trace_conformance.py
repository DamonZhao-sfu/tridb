"""Verify the Figure-10 EverMemOS agentic round instrumentation."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from everalgo.rank.agentic import aagentic_retrieve
from everalgo.types import Candidate


class _ProbeLLM:
    def __init__(self, *, sufficient: bool) -> None:
        self.sufficient = sufficient

    async def chat(self, *, messages: list[object], temperature: float = 0.0) -> object:
        prompt = str(getattr(messages[0], "content"))
        if "is_sufficient" in prompt:
            value = "true" if self.sufficient else "false"
            return SimpleNamespace(
                content=(
                    '{"is_sufficient": '
                    + value
                    + ', "reasoning":"probe", '
                    '"key_information_found":[], '
                    '"missing_information":["x"]}'
                )
            )
        return SimpleNamespace(
            content='{"queries":["q2"],"reasoning":"probe"}'
        )


async def _retrieve(_query: str, _top_k: int) -> list[Candidate]:
    return [
        Candidate(
            id="c1",
            score=1.0,
            source="vector",
            metadata={
                "episode": {"subject": "subject", "content": "content"},
                "timestamp": 0,
            },
        )
    ]


async def _run() -> dict[str, object]:
    _, one = await aagentic_retrieve(
        "q", base_retrieve=_retrieve, llm=_ProbeLLM(sufficient=True)
    )
    _, two = await aagentic_retrieve(
        "q",
        base_retrieve=_retrieve,
        llm=_ProbeLLM(sufficient=False),
        multi_query_count=1,
    )
    assert one.retrieval_rounds == 1
    assert two.retrieval_rounds == 2
    assert len(one.round_boundaries_ns) == 1
    assert len(two.round_boundaries_ns) == 2
    for decision in (one, two):
        for boundary in decision.round_boundaries_ns:
            assert int(boundary["started_ns"]) <= int(boundary["completed_ns"])
    return {"one_round": one.model_dump(), "two_round": two.model_dump()}


def main() -> int:
    print(json.dumps(asyncio.run(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
