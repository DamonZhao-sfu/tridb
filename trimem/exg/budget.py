"""Prompt-budget auditing, including attempts blocked before generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from trimem.exg.runtime import read, restrict_resources, save


def record_budget(
    path, context, input_tokens, retrieval, context_limit=32768, output_reserve=8192
):
    skipped = retrieval.get("skipped_for_token_budget", [])
    audit = dict(
        **context,
        prompt_tokens=input_tokens,
        context_limit=context_limit,
        output_tokens_reserved=output_reserve,
        prompt_tokens_allowed=context_limit - output_reserve,
        context_excess_tokens=max(0, input_tokens + output_reserve - context_limit),
        context_limit_exceeded=input_tokens + output_reserve > context_limit,
        hint_count=len(retrieval["hints"]),
        hint_tokens=retrieval.get("accepted_hint_tokens", 0),
        hint_token_limit=retrieval.get("hint_token_limit"),
        memory_budget_omission=bool(skipped),
        skipped_hint_count=len(skipped),
        skipped_hint_ids=skipped,
        skipped_token_details=retrieval.get("skipped_token_details", []),
        prompt_actually_truncated=False,
        prompt_tokens_actually_removed=0,
        policy="Skip over-budget memory candidates; reject over-context prompts; no text truncation",
    )
    # Save before generation/rejection so incomplete tasks are included in the audit.
    save(path, audit)
    return audit


def summarize(root):
    root = Path(root)
    groups = {}
    for path in root.glob("cells/*/*/r*/*/tasks/*/*/prompt_budget.json"):
        row = read(path)
        key = "/".join(str(row[k]) for k in ("dataset", "mode", "rep", "arm"))
        groups.setdefault(key, []).append(row)

    def counts(rows):
        n = len(rows)
        omitted = sum(r["memory_budget_omission"] for r in rows)
        overflow = sum(r["context_limit_exceeded"] for r in rows)
        return dict(
            prompts_checked=n,
            prompts_with_memory_omission=omitted,
            memory_omission_rate=omitted / n if n else None,
            skipped_hint_occurrences=sum(r["skipped_hint_count"] for r in rows),
            prompts_exceeding_context=overflow,
            context_overflow_rate=overflow / n if n else None,
            prompts_actually_truncated=sum(
                r["prompt_actually_truncated"] for r in rows
            ),
            prompt_tokens_actually_removed=sum(
                r["prompt_tokens_actually_removed"] for r in rows
            ),
        )

    return dict(
        counting_unit="One generation attempt; repeated candidates in different retrievals count as separate occurrences",
        includes_blocked_attempts=True,
        overall=counts([r for rows in groups.values() for r in rows]),
        by_cell={key: counts(rows) for key, rows in groups.items()},
    )


def main():
    restrict_resources()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    save(args.root / "prompt_budget_summary.json", summarize(args.root))


if __name__ == "__main__":
    main()
