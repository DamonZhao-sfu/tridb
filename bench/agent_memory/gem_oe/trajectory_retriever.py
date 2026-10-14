"""OpenEvolve adapter for the rebuilt trajectory store, with per-round state.

Old experiment retrievers remain reproducible. This adapter requires the full-code
embedding receipt, carries the entire current parent to semantic operators, and
renders evidence cards without a summarization LLM. Semantic scores are pointwise
annotations, not a global sorting operator.
"""

import json
import math
from copy import copy
from dataclasses import asdict
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from bench.agent_memory.gem_oe.memory_database import RetrievedProgram
from trimem.code_embedding import VERSION, FullCodeEmbedder
from trimem.trajectory_query import (
    EVIDENCE_VERSION,
    FILTER_CRITERION,
    SCORE_CRITERION,
    CurrentAgentContext,
    bind_context,
    check_request_size,
    query,
)


def parent_context(
    task_uid, specification, parent, iteration, remaining_budget=None, objective=None
):
    metrics = dict(getattr(parent, "metrics", {}) or {})
    metadata = dict(getattr(parent, "metadata", {}) or {})
    errors = {
        k: metrics[k]
        for k in ("error", "error_signature", "stderr", "judge_result")
        if metrics.get(k)
    }
    errors.update(
        {
            k: metadata[k]
            for k in ("error", "error_signature", "evaluation_feedback")
            if metadata.get(k)
        }
    )
    fitness = metrics.get("combined_score")
    fitness = float(fitness) if fitness is not None else None
    if fitness is not None and not math.isfinite(fitness):
        fitness = None
    failed = bool(
        metrics.get("error")
        or metrics.get("error_signature")
        or metadata.get("error")
        or metadata.get("error_signature")
    )
    judge_status = str(
        metrics.get("judge_result") or metadata.get("status") or ""
    ).lower()
    failed = (
        failed
        or metrics.get("is_valid") is False
        or judge_status
        in {
            "wa",
            "re",
            "tle",
            "mle",
            "ce",
            "failed",
            "invalid",
            "compile_error",
            "runtime_error",
        }
    )
    status = (
        "failed"
        if failed
        else str(
            metrics.get("judge_result")
            or metadata.get("status")
            or "validity not explicitly recorded"
        )
    )
    return CurrentAgentContext(
        task_id=task_uid,
        problem=specification,
        objective=objective
        or metadata.get("retrieval_objective")
        or (
            "Repair the recorded failure while preserving the task constraints, then improve fitness."
            if failed
            else "Improve the selected parent's fitness while preserving validity and task constraints."
        ),
        program_code=parent.code,
        program_status=status,
        failure_reason=json.dumps(
            {"observed_errors": errors, "evaluation_metrics": metrics},
            ensure_ascii=False,
            default=str,
        )
        if metrics or errors
        else "No evaluation feedback recorded; failure state unknown.",
        constraints="Respect all constraints in the full problem statement. Remaining token budget is null when not configured.",
        provenance=f"OpenEvolve selected parent {parent.id}, retrieval round {iteration}; no candidate outcome used as current feedback",
        fitness=fitness,
        iteration=iteration,
        remaining_token_budget=remaining_budget,
    )


def evidence_card(row, include_code=False):
    """Agent-facing change prose only. Provenance stays in retrieval records.

    Reuse historical intent, falling back to recorded framework changes. No new
    summary is generated and no missing description is invented.
    """
    summary = (row.get("intent") or "").strip() or (
        row.get("framework_changes") or ""
    ).strip()
    if include_code:
        return "\n\n".join(part for part in (summary, row.get("code") or "") if part)
    return summary


def streaming_plan_nodes(plan):
    """Fail closed if statistics changes introduce a blocking physical plan."""
    nodes = []

    def walk(node):
        kind = node.get("Node Type")
        if kind in {
            "Sort",
            "Incremental Sort",
            "Hash",
            "Hash Join",
            "Materialize",
            "Aggregate",
            "Gather Merge",
        }:
            raise ValueError(f"trajectory plan violates streaming contract: {kind}")
        nodes.append(
            {
                k: node[k]
                for k in ("Node Type", "Index Name", "Custom Plan Provider")
                if k in node
            }
        )
        for child in node.get("Plans", []):
            walk(child)

    walk(plan)
    if not any(n.get("Index Name") == "attempt_hnsw" for n in nodes):
        raise ValueError("full-code trajectory plan must use the HNSW code index")
    return nodes


class TrajectoryRetriever:
    def __init__(
        self,
        dsn,
        plan="P2+Rank",
        out=None,
        total_token_budget=None,
        objective=None,
        memory_tokens=16384,
        include_code=True,
        embedder=None,
    ):
        if memory_tokens <= 0 or (
            total_token_budget is not None and total_token_budget < 0
        ):
            raise ValueError(
                "memory token budget must be positive and total budget nonnegative"
            )
        self.conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        meta = self.conn.execute("SELECT code_embedding FROM load_meta").fetchone()[
            "code_embedding"
        ]
        if not meta or meta.get("embedding_version") != VERSION:
            raise ValueError(
                "trajectory code embeddings must be rebuilt before using this adapter"
            )
        self.name = f"trajectory-{plan}"
        self.plan, self.out = plan, Path(out) if out else None
        self.total_token_budget, self.objective = total_token_budget, objective
        if not include_code:
            raise ValueError(
                "trajectory inspirations require complete code plus intent"
            )
        self.memory_tokens, self.include_code = memory_tokens, include_code
        self.embedder = embedder
        self.telemetry = []
        self.last_inspirations = []
        self.database = None
        self.conn.execute("SET jit=off")
        for setting, value in {
            "enable_hashjoin": "off",
            "enable_mergejoin": "off",
            "enable_material": "off",
            "enable_memoize": "off",
            "enable_seqscan": "off",
            "enable_sort": "off",
            "enable_bitmapscan": "off",
            "max_parallel_workers_per_gather": "0",
            "join_collapse_limit": "1",
            "from_collapse_limit": "1",
            "plan_cache_mode": "force_custom_plan",
            "hnsw.iterative_scan": "strict_order",
            "trimem.min_batch": "1",
            "trimem.max_batch": "1",
            "trimem.placement": "top",
            "trimem.force_placement": "",
        }.items():
            self.conn.execute("SELECT set_config(%s,%s,false)", (setting, value))
        self.conn.execute("SET trimem.timeout_ms=180000")
        self.conn.execute(
            "SELECT set_config('trimem.prompt_version',%s,false)", (EVIDENCE_VERSION,)
        )
        self.max_bytes = int(
            self.conn.execute("SHOW trimem.max_input_bytes").fetchone()[
                "trimem.max_input_bytes"
            ]
        )

    def _remaining(self):
        if self.total_token_budget is None:
            return None
        used = 0
        path = self.out / "llm_telemetry.jsonl" if self.out else None
        if path and path.exists():
            for line in path.read_text().splitlines():
                try:
                    used += int(json.loads(line).get("total_tokens") or 0)
                except json.JSONDecodeError:
                    # A concurrent writer may have an incomplete last line.
                    continue
        return max(0, self.total_token_budget - used)

    def retrieve(self, *, task_uid, parent, k, iteration):
        if k <= 0:
            return []
        requested_k = k
        k = min(k, 4)
        task = self.conn.execute(
            "SELECT specification FROM task WHERE task=%s", (task_uid,)
        ).fetchone()
        if not task or not task["specification"]:
            raise ValueError("trajectory retrieval needs the actual task specification")
        if self.database is not None:
            artifacts = self.database.get_artifacts(parent.id)
            if artifacts:
                parent = copy(parent)
                parent.metadata = dict(parent.metadata or {}) | {
                    "evaluation_feedback": artifacts
                }
        context = parent_context(
            task_uid,
            task["specification"],
            parent,
            iteration,
            self._remaining(),
            self.objective,
        )
        if self.embedder is None:
            self.embedder = FullCodeEmbedder()
        vector = self.embedder.embed(context.program_code)
        params = bind_context(
            {
                "task": task_uid,
                "exclude_session": None,
                "q": "[" + ",".join(map(str, vector)) + "]",
                "predicate": FILTER_CRITERION,
                "score_predicate": SCORE_CRITERION,
            },
            context,
        )
        # Check current context now. The native operator checks the actual full
        # predicate + context + candidate at execution and rejects oversize input.
        check_request_size(
            params["score_predicate"],
            params["current_context"],
            self.max_bytes,
        )
        if self.out:
            folder = self.out / "retrieval_contexts"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{iteration:04d}.json").write_text(
                json.dumps(asdict(context), ensure_ascii=False, indent=2)
            )
        from trimem.objective.store import count_tokens

        sql = query(self.plan, "emb")
        planned = self.conn.execute(
            "EXPLAIN (FORMAT JSON) DECLARE trajectory_check CURSOR FOR " + sql, params
        ).fetchone()["QUERY PLAN"][0]["Plan"]
        physical_nodes = streaming_plan_nodes(planned)

        selected, seen, used, skipped, rows_seen = [], set(), 0, 0, 0
        rendered_blocks = []
        from bench.agent_memory.gem_oe.trajectory_prompt import (
            SECTION,
            render_inspiration,
        )

        # Named cursor provides early termination without fetching a full result.
        with (
            self.conn.transaction(),
            self.conn.cursor(name=f"trajectory_{iteration}") as cur,
        ):
            cur.itersize = 1
            cur.execute(sql, params)
            for row in cur:
                rows_seen += 1
                if row["id"] in seen:
                    continue
                seen.add(row["id"])
                card = evidence_card(row)
                if not row.get("code"):
                    continue
                language = row.get("language") or (
                    "cpp" if task_uid.startswith("ale:") else "python"
                )
                block = render_inspiration(
                    len(selected) + 1,
                    row["code"],
                    card,
                    language,
                )
                tokens = count_tokens(SECTION + "\n\n".join(rendered_blocks + [block]))
                if tokens > self.memory_tokens:
                    skipped += 1
                    continue
                selected.append(
                    RetrievedProgram(
                        uid=row["uid"],
                        code=row["code"],
                        language=language,
                        metrics={"combined_score": float(row["fitness"])},
                        changes_description=card,
                        provenance={
                            "memory": "trajectory",
                            "attempt_id": row["id"],
                            "context_sha256": params["current_context_sha256"],
                            "assembly": "complete-code-plus-stored-intent",
                            "retrieved_fields": {
                                key: value
                                for key, value in row.items()
                                if key != "code"
                            },
                        },
                    )
                )
                rendered_blocks.append(block)
                used = tokens
                if len(selected) >= k:
                    break
        self.last_inspirations = selected
        receipt = {
            "iteration": iteration,
            "parent_id": parent.id,
            "plan": self.plan,
            "returned": len(selected),
            "requested_k": requested_k,
            "effective_k": k,
            "max_inspiration_programs": 4,
            "context_sha256": params["current_context_sha256"],
            "embedding": self.embedder.last_receipt,
            "memory_tokens": used,
            "memory_token_budget": self.memory_tokens,
            "skipped_for_budget": skipped,
            "truncated_records": 0,
            "post_retrieval_summary_llm_calls": 0,
            "rows_seen": rows_seen,
            "physical_plan_nodes": physical_nodes,
            "remaining_token_budget": context.remaining_token_budget,
            "remaining_budget_scope": "generation LLM usage; excludes semantic retrieval calls",
            "semantic_rank_contract": "pointwise score, streaming order; no global sort",
        }
        self.telemetry.append(receipt)
        if self.out:
            with (self.out / "trajectory_retrieval.jsonl").open("a") as f:
                f.write(json.dumps(receipt, ensure_ascii=False) + "\n")
        return selected

    def close(self):
        self.conn.close()
