"""Whitespace-tolerant SEARCH/REPLACE application for OpenEvolve diffs (opt-in: `run_arm --diff-ws-tolerant`).

OpenEvolve 0.3.2 applies a block only when every SEARCH line equals a parent line exactly, and silently
drops blocks that do not match. ALE seeds have trailing spaces on ~20% of their lines; the LLM strips them
when it copies code into a SEARCH block, so 17% of all ALE blocks were dropped (77% of those match once
trailing whitespace is ignored) -- typically the block that defines a constant while the block that uses
it applies, giving "was not declared" compile errors.

Here a line matches when the two lines are equal after `rstrip()`; the replacement text is unchanged.
Workers are spawned (Python 3.11 + max_tasks_per_child), so the patch is installed in each worker by
wrapping OpenEvolve's worker initializer, which is pickled by reference.
"""

from __future__ import annotations

from typing import Any

DEFAULT_PATTERN = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"


def _find(lines: list[str], search: list[str]) -> int:
    key = [s.rstrip() for s in search]
    stripped = [s.rstrip() for s in lines]
    for i in range(len(lines) - len(search) + 1):
        if stripped[i : i + len(search)] == key:
            return i
    return -1


def apply_diff_blocks(original_text: str, diff_blocks: list[tuple[str, str]]) -> tuple[str, int]:
    lines = original_text.split("\n")
    applied = 0
    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        i = _find(lines, search_lines)
        if i >= 0:
            lines[i : i + len(search_lines)] = replace_text.split("\n")
            applied += 1
    return "\n".join(lines), applied


def apply_diff(original_code: str, diff_text: str, diff_pattern: str = DEFAULT_PATTERN) -> str:
    from openevolve.utils.code_utils import extract_diffs

    return apply_diff_blocks(original_code, extract_diffs(diff_text, diff_pattern))[0]


def install() -> None:
    """Replace OpenEvolve's matchers in this process (the worker imports them from code_utils per call)."""
    import openevolve.utils.code_utils as cu

    cu.apply_diff = apply_diff
    cu.apply_diff_blocks = apply_diff_blocks


def worker_init(*args: Any) -> Any:
    install()
    from openevolve import process_parallel

    return process_parallel._worker_init(*args)  # the original: a spawned worker imports it fresh


def enable() -> None:
    """Patch this process and every OpenEvolve worker it starts."""
    install()
    from openevolve import process_parallel

    process_parallel._worker_init = worker_init
