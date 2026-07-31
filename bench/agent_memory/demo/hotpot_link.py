"""Resolve HotpotQA gold evidence onto slice units, so act 2 is graded.

WHY THIS EXISTS
---------------
Retrieval quality judged by reading the top-5 is not a measurement. HotpotQA
supplies real multi-hop questions whose gold supporting paragraphs are named, so
if the slice contains those articles the demo can report joint evidence recall@k
instead of an impression.

WHAT IT CAN AND CANNOT CLAIM
----------------------------
A question is **fully resolved** only when EVERY one of its gold supporting
titles is an article in the slice. Only on that subset is multi-hop evidence
recall well defined — a question missing one of its two gold paragraphs can
never score 1.0 and would silently drag the mean down. Coverage is therefore
always reported next to the metric, never folded into it.

The slice is **question-aware by construction**: :func:`suggest_seeds` returns
the gold titles that are missing for questions already anchored in the domain,
and those get added to the fetcher's seed list. This is the same move HotpotQA's
own distractor setting makes, and it is legitimate as long as it is stated — but
it means the demo measures retrieval over a candidate pool of a few hundred
in-domain articles, NOT retrieve-from-all-of-Wikipedia. The full-wiki setting is
a GX10-scale exercise (``bench/wiki_scale_report.py``), and nothing here may be
quoted as if it were that.

USAGE
-----
    python -m bench.agent_memory.demo.hotpot_link \
        --slice data/wiki_demo --hotpot data/hotpot/dev_slice.json
    python -m bench.agent_memory.demo.hotpot_link ... --suggest-seeds 120
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from bench.agent_memory.demo.adapter import WikiSlice
from bench.agent_memory.demo.wiki_source import normalize_title


@dataclass(frozen=True)
class HotpotQuestion:
    qid: str
    question: str
    answer: str
    gold_titles: tuple[str, ...]  # normalized, de-duplicated, order preserved
    level: str = ""
    type: str = ""


@dataclass(frozen=True)
class ResolvedQuestion:
    question: HotpotQuestion
    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def fully_resolved(self) -> bool:
        return not self.missing

    @property
    def anchored(self) -> bool:
        """At least one gold title is already in the slice.

        These are the questions worth pulling the rest of the gold in for: the
        domain already covers half the hop, so completing them keeps the slice
        topically coherent instead of scattering it across Wikipedia.
        """
        return bool(self.present)


def load_questions(path: Path | str) -> list[HotpotQuestion]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["questions"] if isinstance(payload, dict) else payload
    out: list[HotpotQuestion] = []
    for row in rows:
        titles = tuple(
            dict.fromkeys(
                normalize_title(title)
                for title, _sent in row.get("supporting_facts", [])
            )
        )
        if not titles:
            continue
        out.append(
            HotpotQuestion(
                qid=str(row["id"]),
                question=row["question"],
                answer=row.get("answer", ""),
                gold_titles=titles,
                level=row.get("level", ""),
                type=row.get("type", ""),
            )
        )
    return out


def resolve(
    questions: Sequence[HotpotQuestion], slice_titles: set[str]
) -> list[ResolvedQuestion]:
    resolved: list[ResolvedQuestion] = []
    for question in questions:
        present = tuple(t for t in question.gold_titles if t in slice_titles)
        missing = tuple(t for t in question.gold_titles if t not in slice_titles)
        resolved.append(
            ResolvedQuestion(question=question, present=present, missing=missing)
        )
    return resolved


def coverage(resolved: Sequence[ResolvedQuestion]) -> dict[str, Any]:
    """The numbers that must travel with any retrieval metric from this demo."""
    total = len(resolved)
    full = [r for r in resolved if r.fully_resolved]
    anchored = [r for r in resolved if r.anchored]
    gold_hops = Counter(len(r.question.gold_titles) for r in full)
    return {
        "questions_considered": total,
        "fully_resolved": len(full),
        "fully_resolved_fraction": (len(full) / total) if total else 0.0,
        "anchored_but_incomplete": len(anchored) - len(full),
        "gold_titles_per_resolved_question": dict(sorted(gold_hops.items())),
        "note": (
            "metrics are defined only on the fully_resolved subset; this is an "
            "in-domain candidate pool, not retrieve-from-all-of-Wikipedia"
        ),
    }


def gold_seed_titles(
    questions: Sequence[HotpotQuestion], *, limit_questions: int
) -> list[str]:
    """Gold supporting titles of the first N questions, in question order.

    This is the corpus-construction direction that actually works. Seeding from
    a topic (say, computing) and hoping HotpotQA overlaps it does not: measured
    on this slice, 400 in-domain articles fully resolved 0 of 1500 dev
    questions, because HotpotQA is mostly films, bands and athletes. Choosing
    the questions first and fetching their gold guarantees the subset exists.

    What it costs is stated plainly wherever the metric appears: the candidate
    pool is built to contain the gold, so this measures ranking within a few
    hundred in-domain articles, NOT retrieve-from-all-of-Wikipedia.
    """
    seen: dict[str, None] = {}
    for question in questions[:limit_questions]:
        for title in question.gold_titles:
            seen.setdefault(title, None)
    return list(seen)


def suggest_seeds(
    resolved: Sequence[ResolvedQuestion], *, limit: int = 120, min_present: int = 1
) -> list[str]:
    """Missing gold titles for anchored questions, most-wanted first.

    Ranked by how many anchored questions need each title, ties broken by title
    so a re-run proposes the same seeds. Feed these back to ``wiki_source
    --seed`` and the previously-anchored questions become fully resolved.
    """
    wanted: Counter[str] = Counter()
    for row in resolved:
        if len(row.present) >= min_present and row.missing:
            for title in row.missing:
                wanted[title] += 1
    ranked = sorted(wanted.items(), key=lambda kv: (-kv[1], kv[0]))
    return [title for title, _count in ranked[:limit]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slice", type=Path, default=Path("data/wiki_demo"))
    parser.add_argument(
        "--hotpot", type=Path, default=Path("data/hotpot/dev_slice.json")
    )
    parser.add_argument(
        "--suggest-seeds",
        type=int,
        default=0,
        help="print N missing gold titles for anchored questions and exit",
    )
    parser.add_argument(
        "--seeds-for-questions",
        type=int,
        default=0,
        help=(
            "print the gold titles of the first N questions and exit — feed to "
            "wiki_source --seed-file so those questions become fully resolved"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the resolved question set as JSON (default: <slice>/questions.json)",
    )
    args = parser.parse_args(argv)

    questions = load_questions(args.hotpot)

    # Seed selection needs the question set only — it runs BEFORE a slice
    # exists, which is the whole point of building the corpus from questions.
    if args.seeds_for_questions:
        for title in gold_seed_titles(
            questions, limit_questions=args.seeds_for_questions
        ):
            print(title)
        return 0

    wiki = WikiSlice.load(args.slice)
    titles = {article.title for article in wiki.articles}
    resolved = resolve(questions, titles)

    if args.suggest_seeds:
        for title in suggest_seeds(resolved, limit=args.suggest_seeds):
            print(title)
        return 0

    stats = coverage(resolved)
    out_path = args.out or (args.slice / "questions.json")
    out_path.write_text(
        json.dumps(
            {
                "coverage": stats,
                "hotpot_source": str(args.hotpot),
                "questions": [
                    {
                        "id": row.question.qid,
                        "question": row.question.question,
                        "answer": row.question.answer,
                        "level": row.question.level,
                        "type": row.question.type,
                        "gold_titles": list(row.question.gold_titles),
                    }
                    for row in resolved
                    if row.fully_resolved
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))
    print(f"resolved question set -> {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
