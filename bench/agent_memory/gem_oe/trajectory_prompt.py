"""Task-native prompts with run history and code-plus-intent retrieval context.

Installed only by --trajectory-store. No changes to the installed OpenEvolve
package or to legacy experiment workers. Prompt assembly never calls a model.
"""

import glob
import inspect
import json
import math
import os
import random
from functools import lru_cache
from pathlib import Path

from openevolve.process_parallel import ProcessParallelController
from openevolve.process_parallel import _run_iteration_worker as _base_worker
from openevolve.prompt.sampler import PromptSampler

from bench.agent_memory.gem_oe.memory_database import is_external

TEMPLATE_DIR = Path(__file__).parent / "prompts" / "trajectory"
SECTION = "## Inspiration Programs\n\nThese programs represent diverse approaches and creative solutions that may inspire new ideas:\n\n"
TOP_TOKEN_BUDGET = 8192
INSPIRATION_TOKEN_BUDGET = 16384
PROMPT_TOKEN_BUDGET = 32768
MAX_INSPIRATIONS = 4
_base_snapshot = ProcessParallelController._create_database_snapshot
FEEDBACK_FIELDS = (
    "error",
    "error_signature",
    "stderr",
    "standard_error",
    "stdout",
    "message",
    "judge_result",
    "is_valid",
    "validity",
    "evaluation_feedback",
)


@lru_cache(maxsize=1)
def generation_tokenizer():
    """Load only the local generation tokenizer, never model weights or a network."""
    from transformers import AutoTokenizer

    paths = sorted(
        glob.glob(
            "/localhome/hza214/hf-cache/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/tokenizer.json"
        )
    )
    if not paths:
        raise RuntimeError(
            "Local Qwen3.8 generation tokenizer is required for prompt budgeting"
        )
    return AutoTokenizer.from_pretrained(
        str(Path(paths[0]).parent), local_files_only=True
    )


def generation_prompt_tokens(prompt, *, enable_thinking=False):
    """Count chat-template tokens including the assistant generation prefix."""
    tokens = generation_tokenizer().apply_chat_template(
        [
            {"role": role, "content": prompt[key]}
            for role, key in (("system", "system"), ("user", "user"))
        ],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        **({"enable_thinking": enable_thinking} if enable_thinking is not None else {}),
    )
    return len(tokens)


def evaluation_feedback(program, artifacts=None, *, include_metrics=True):
    """Render recorded feedback verbatim, without inventing missing diagnostics."""
    metrics = program.get("metrics") or {}
    metadata = program.get("metadata") or {}
    selected = (
        metrics
        if include_metrics
        else {k: metrics[k] for k in FEEDBACK_FIELDS if k in metrics}
    )
    payload = {}
    if selected:
        payload["evaluation_metrics"] = selected
    recorded = {k: metadata[k] for k in FEEDBACK_FIELDS if k in metadata}
    if recorded:
        payload["recorded_feedback"] = recorded
    if artifacts:
        payload["evaluation_artifacts"] = artifacts
    if not payload:
        return "No additional evaluation/error feedback recorded."
    return (
        "````json\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=lambda value: (
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else str(value)
            ),
        )
        + "\n````"
    )


def render_inspiration(number, code, intent, language):
    return (
        (TEMPLATE_DIR / "inspiration_program.txt")
        .read_text()
        .format(
            program_number=number,
            intent=intent or "Not recorded.",
            language=language or "python",
            program_snippet=code,
        )
        .rstrip()
    )


def score(program):
    value = program.get("metrics", {}).get("combined_score")
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def outcome(program):
    current = score(program)
    parent = score({"metrics": program.get("metadata", {}).get("parent_metrics", {})})
    if current is None or parent is None:
        return "Comparison unavailable: combined_score or parent score not recorded"
    if current > parent:
        return "Improvement in combined_score"
    if current < parent:
        return "Decline in combined_score"
    return "No change in combined_score"


def own_context(programs, parent_id, iteration):
    """Use only completed current-run records, never retrieved/future programs."""
    own = [
        p
        for p in programs
        if not is_external(p.get("id", ""))
        and not p.get("metadata", {}).get("provenance", {}).get("memory")
        and p.get("iteration_found", 0) < iteration
    ]
    previous = sorted(
        (p for p in own if p.get("parent_id") is not None),
        key=lambda p: (p.get("iteration_found", 0), p.get("id", "")),
    )[-3:]
    parent_code = next(
        (p.get("code") for p in programs if p.get("id") == parent_id), None
    )
    top = [
        p
        for p in own
        if p.get("id") != parent_id
        and p.get("code") != parent_code
        and score(p) is not None
        and p.get("metrics", {}).get("is_valid") is not False
        and p.get("metrics", {}).get("validity") != 0
        and not p.get("metrics", {}).get("error")
        and p.get("metrics", {}).get("judge_result") in (None, "AC", "ACCEPTED")
    ]
    top.sort(key=lambda p: (-score(p), p.get("id", "")))
    return previous, top


class TrajectoryPromptSampler(PromptSampler):
    """Reuse the native system/user template, overriding context formatting only."""

    def __init__(
        self,
        config,
        *,
        own_programs=None,
        parent_id=None,
        iteration=None,
        artifacts_by_program=None,
        parent_record=None,
        prompt_token_budget=None,
        inspiration_token_budget=None,
    ):
        super().__init__(config)
        self.own_programs = own_programs
        self.parent_id = parent_id
        self.iteration = iteration
        self.context_receipt = {}
        self.artifacts_by_program = artifacts_by_program or {}
        self.parent_record = parent_record
        self.prompt_token_budget = int(
            prompt_token_budget
            if prompt_token_budget is not None
            else os.environ.get(
                "TRIMEM_TRAJECTORY_PROMPT_TOKEN_BUDGET", PROMPT_TOKEN_BUDGET
            )
        )
        self.inspiration_token_budget = int(
            inspiration_token_budget
            if inspiration_token_budget is not None
            else os.environ.get(
                "TRIMEM_TRAJECTORY_MEMORY_TOKEN_BUDGET", INSPIRATION_TOKEN_BUDGET
            )
        )
        thinking = os.environ.get("TRIMEM_TRAJECTORY_ENABLE_THINKING", "false").lower()
        self.enable_thinking = {"true": True, "false": False}.get(thinking)
        if min(self.prompt_token_budget, self.inspiration_token_budget) <= 0:
            raise ValueError(
                "Trajectory prompt and inspiration budgets must be positive"
            )

    def build_prompt(self, *args, **kwargs):
        from trimem.objective.store import count_tokens

        # Normalize positional calls too, without mutating caller-owned records.
        bound = inspect.signature(PromptSampler.build_prompt).bind_partial(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        kwargs = dict(bound.arguments)
        kwargs.pop("self")
        kwargs.update(kwargs.pop("kwargs", {}))
        if self.own_programs is not None:
            previous, top = own_context(
                self.own_programs, self.parent_id, self.iteration
            )
            kwargs.update(previous_programs=previous, top_programs=top)
        parent = self.parent_record or next(
            (p for p in (self.own_programs or []) if p.get("id") == self.parent_id), {}
        )
        parent = dict(
            parent, metrics=kwargs.get("program_metrics", parent.get("metrics", {}))
        )
        parent_artifacts = kwargs.get(
            "program_artifacts"
        ) or self.artifacts_by_program.get(self.parent_id)
        kwargs["current_evaluation_feedback"] = evaluation_feedback(
            parent, parent_artifacts
        )
        # Feedback has its own section. Bypass the default artifact renderer,
        # which truncates diagnostics and would otherwise duplicate them.
        kwargs["program_artifacts"] = None
        kwargs["previous_programs"] = list(kwargs.get("previous_programs", []))[-3:]
        kwargs["top_programs"] = list(kwargs.get("top_programs", []))
        kwargs["inspirations"] = list(kwargs.get("inspirations", []))
        if self.config.num_top_programs == 0 and self.config.num_diverse_programs == 0:
            # Strict no-memory control: preserve current parent and its feedback,
            # but never reintroduce own history from the database snapshot.
            for section in ("previous_programs", "top_programs", "inspirations"):
                kwargs[section] = []
        removed = []
        # Reuse the same stochastic template choice during budget reductions.
        random_state = random.getstate()

        def render():
            random.setstate(random_state)
            return super(TrajectoryPromptSampler, self).build_prompt(**kwargs)

        result = render()
        original_tokens = generation_prompt_tokens(
            result, enable_thinking=self.enable_thinking
        )
        top_skips = self.context_receipt["top_skipped_for_budget"]
        # Freeze the visible top records, so dropping one cannot promote another.
        shown = set(self.context_receipt["top_program_ids"])
        kwargs["top_programs"] = [
            p for p in kwargs["top_programs"] if p.get("id") in shown
        ]
        inspirations = kwargs["inspirations"]
        for p in inspirations[MAX_INSPIRATIONS:]:
            removed.append(
                {"section": "inspirations", "id": p.get("id"), "reason": "max_programs"}
            )
        selected = []
        for p in inspirations[:MAX_INSPIRATIONS]:
            section = self._format_inspirations_section(
                selected + [p], kwargs["language"]
            )
            if count_tokens(section) > self.inspiration_token_budget:
                removed.append(
                    {
                        "section": "inspirations",
                        "id": p.get("id"),
                        "reason": "inspiration_token_budget",
                    }
                )
            else:
                selected.append(p)
        kwargs["inspirations"] = selected
        result = render()
        tokens = generation_prompt_tokens(result, enable_thinking=self.enable_thinking)
        while tokens > self.prompt_token_budget:
            section = next(
                (
                    key
                    for key in ("inspirations", "top_programs", "previous_programs")
                    if kwargs[key]
                ),
                None,
            )
            if section is None:
                break
            p = kwargs[section].pop(0 if section == "previous_programs" else -1)
            removed.append(
                {
                    "section": section,
                    "id": p.get("id"),
                    "reason": "full_prompt_token_budget",
                }
            )
            result = render()
            tokens = generation_prompt_tokens(
                result, enable_thinking=self.enable_thinking
            )
        body_tokens = sum(count_tokens(v) for v in result.values())
        self.context_receipt.update(
            {
                "iteration": kwargs.get("evolution_round"),
                "prompt_body_tokens": body_tokens,
                "prompt_input_tokens_original": original_tokens,
                "prompt_input_tokens": tokens,
                "prompt_chat_template_overhead_tokens": tokens - body_tokens,
                "prompt_input_token_budget": self.prompt_token_budget,
                "prompt_tokenizer_model": "Qwen/Qwen3.8-27B-FP8",
                "prompt_enable_thinking": self.enable_thinking,
                "inspiration_token_budget": self.inspiration_token_budget,
                "inspiration_tokens": count_tokens(
                    self._format_inspirations_section(
                        kwargs["inspirations"], kwargs["language"]
                    )
                ),
                "inspiration_ids": [p.get("id") for p in kwargs["inspirations"]],
                "inspiration_count": len(kwargs["inspirations"]),
                "max_inspiration_programs": MAX_INSPIRATIONS,
                "removed_records": removed,
                "removed_record_count": len(removed),
                "top_skipped_for_budget": top_skips,
                "prompt_reduced_for_budget": any(
                    p["reason"] != "max_programs" for p in removed
                ),
                "prompt_exceeded_budget": original_tokens > self.prompt_token_budget,
                "prompt_budget_error": tokens > self.prompt_token_budget,
                "assembly_llm_calls": 0,
            }
        )
        path = os.environ.get("TRIMEM_TRAJECTORY_PROMPT_LOG")
        if path:
            with Path(path).open("a") as f:
                f.write(json.dumps(self.context_receipt) + "\n")
        if tokens > self.prompt_token_budget:
            raise ValueError(
                f"Mandatory trajectory prompt requires {tokens} input tokens, exceeding budget {self.prompt_token_budget}; current code and feedback were not truncated"
            )
        return result

    def _format_inspirations_section(
        self, inspirations, language, feature_dimensions=None
    ):
        if not inspirations:
            return ""
        return SECTION + "\n\n".join(
            render_inspiration(
                i + 1,
                p.get("code", ""),
                p.get("changes_description", ""),
                p.get("language") or language,
            )
            for i, p in enumerate(inspirations)
        )

    def _format_evolution_history(
        self,
        previous_programs,
        top_programs,
        inspirations,
        language,
        feature_dimensions=None,
    ):
        from trimem.objective.store import count_tokens

        attempts = []
        for i, p in enumerate(reversed(previous_programs[-3:])):
            changes = (
                p.get("changes_description")
                or p.get("metadata", {}).get("changes")
                or "Unknown changes"
            )
            metrics = ", ".join(
                f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                for k, v in p.get("metrics", {}).items()
            )
            number = p.get("iteration_found", len(previous_programs) - i)
            attempts.append(
                f"### Attempt {number}\n- Changes: {changes}\n- Metrics: {metrics}\n- Outcome: {outcome(p)}"
                + "\n\n#### Evaluation Feedback\n"
                + evaluation_feedback(
                    p, self.artifacts_by_program.get(p.get("id")), include_metrics=False
                )
            )
        blocks, used_ids, seen_code = [], [], set()
        skipped = 0
        for p in top_programs:
            if len(blocks) >= self.config.num_top_programs:
                break
            code = p.get("code", "")
            if not code or code in seen_code:
                continue
            seen_code.add(code)
            value = score(p)
            label = f"{value:.4f}" if value is not None else "unknown"
            block = f"### Program {len(blocks) + 1} (combined_score: {label})\n```{p.get('language') or language}\n{code}\n```"
            if (
                count_tokens(
                    "## Top Performing Programs\n\n" + "\n\n".join(blocks + [block])
                )
                > TOP_TOKEN_BUDGET
            ):
                skipped += 1
                continue
            blocks.append(block)
            used_ids.append(p.get("id"))
        self.context_receipt = {
            "previous_attempt_ids": [p.get("id") for p in previous_programs[-3:]],
            "top_program_ids": used_ids,
            "top_skipped_for_budget": skipped,
            "top_token_budget": TOP_TOKEN_BUDGET,
            "truncated_records": 0,
        }
        return (
            "## Previous Attempts\n\n"
            + ("\n\n".join(attempts) or "No completed previous attempts in this run.")
            + "\n\n## Top Performing Programs\n\n"
            + (
                "\n\n".join(blocks)
                or "No additional evaluated programs available within the context budget."
            )
            + "\n\n"
            + self._format_inspirations_section(
                inspirations, language, feature_dimensions
            )
        )


def trajectory_iteration_worker(iteration, db_snapshot, parent_id, inspiration_ids):
    """Top-level callable works with fork and spawn ProcessPoolExecutor workers."""
    import openevolve.process_parallel as parallel

    parallel._lazy_init_worker_components()
    original = parallel._worker_prompt_sampler
    # Preserve the explicit zero-context injection gate.
    own = list(db_snapshot["programs"].values()) if any(db_snapshot["islands"]) else []
    parallel._worker_prompt_sampler = TrajectoryPromptSampler(
        parallel._worker_config.prompt,
        own_programs=own,
        parent_id=parent_id,
        iteration=iteration,
        artifacts_by_program=db_snapshot.get("artifacts", {}),
        parent_record=db_snapshot["programs"].get(parent_id),
    )
    try:
        return _base_worker(iteration, db_snapshot, parent_id, inspiration_ids)
    finally:
        parallel._worker_prompt_sampler = original


def trajectory_database_snapshot(controller):
    """Include selected-parent/recent feedback even beyond the base snapshot cap."""
    snapshot = _base_snapshot(controller)
    programs = list(snapshot["programs"].values())
    trace = getattr(controller.database, "injection_trace", [])
    parent_id = trace[-1].get("parent_id") if trace else None
    next_iteration = max((p.get("iteration_found", 0) for p in programs), default=0) + 1
    previous, _ = own_context(programs, parent_id, next_iteration)
    for pid in {parent_id, *(p.get("id") for p in previous)} - {None}:
        artifacts = controller.database.get_artifacts(pid)
        if artifacts:
            snapshot["artifacts"][pid] = artifacts
    return snapshot


def install():
    import openevolve.process_parallel as parallel

    parallel._run_iteration_worker = trajectory_iteration_worker
    parallel.ProcessParallelController._create_database_snapshot = (
        trajectory_database_snapshot
    )
