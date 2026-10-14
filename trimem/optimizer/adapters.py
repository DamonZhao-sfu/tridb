"""Explicit bridge from isolated continuation callbacks to native retrieval.

This module does not create a DB or start a worker. Each branch must supply its
own connection and exact token-usage collector. No old all-plan pool is invoked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .compiler import QueryContext, compile_candidate
from .evaluator import EvalContext
from .executor import ExecutionReceipt, NativeExecutor
from .ledger import MissingUsageError, TokenBudgetExhausted
from .packing import GainRankedPool, PackedContext, Tokenizer, pack_stream
from .schema import CandidateSpec


@dataclass
class RetrievalResult:
    packed: PackedContext
    receipt: ExecutionReceipt

    @property
    def text(self) -> str:
        return self.packed.text

    def to_dict(self) -> dict:
        return asdict(self)


class NativeRetrievalAdapter:
    """The `retrieve` callback of ContinuationAdapter, with exact prompt packing.

    Factories are called only for the selected action. `query_context` may embed
    the query; it must record embedding usage in context.ledger itself. `executor`
    must return a connection bound to context.isolated_branch; a shared mutable
    connection is not sufficient to establish branch isolation.
    """

    def __init__(
        self,
        *,
        executor: Callable[[EvalContext], NativeExecutor],
        query_context: Callable[[EvalContext, Any], QueryContext],
        tokenizer: Tokenizer,
        tokenizer_id: str,
        render_prompt: Callable[[EvalContext, Any, str], str],
        model_context_window: int,
        max_output_tokens: int,
    ):
        self.executor_factory = executor
        self.context_factory = query_context
        self.tokenizer, self.tokenizer_id = tokenizer, tokenizer_id
        self.render_prompt = render_prompt
        self.model_context_window = model_context_window
        self.max_output_tokens = max_output_tokens

    def __call__(
        self,
        action: CandidateSpec,
        context: EvalContext,
        state: Any,
    ) -> RetrievalResult:
        if context.protocol.tokenizer_version != self.tokenizer_id:
            raise ValueError("retrieval tokenizer differs from evaluation protocol")
        if context.ledger.remaining_tokens == 0:
            raise TokenBudgetExhausted("retrieval has no remaining token budget")
        query = (
            None
            if action.plan_id == "no_memory_v1"
            else self.context_factory(context, state)
        )
        plan = compile_candidate(action, query)
        # A no-memory action must not even ask the factory for a DB connection.
        executor = (
            NativeExecutor(None) if plan.sql is None else self.executor_factory(context)
        )
        # The exclusive worker processes semantic calls serially in its own
        # ledger. Freeze the available budget before dispatch, then import the
        # physical request events exactly once after Close.
        collector = getattr(executor, "usage_collector", None)
        set_budget = getattr(collector, "set_token_budget", None)
        if callable(set_budget):
            set_budget(context.ledger.remaining_tokens)
        cursor = executor.open_selected(plan)
        try:
            with cursor:
                # ordering="gain": bounded pool (<= candidate_visit_cap rows),
                # ranked client-side by fitness gain, then the same packing.
                source = (
                    GainRankedPool(cursor)
                    if plan.sql is not None and plan.action.ordering == "gain"
                    else cursor
                )
                packed = pack_stream(
                    source,
                    self.tokenizer,
                    top_k=0 if plan.sql is None else plan.action.top_k,
                    memory_token_cap=0
                    if plan.sql is None
                    else plan.action.memory_token_cap,
                    tokenizer_id=self.tokenizer_id,
                    model_context_window=self.model_context_window,
                    render_prompt=lambda memory: self.render_prompt(
                        context, state, memory
                    ),
                    max_output_tokens=self.max_output_tokens,
                )
        finally:
            # Include requests made even if fetching/packing failed. Incomplete
            # attribution remains an explicit failed request, not zero cost.
            try:
                for event in cursor.receipt.usage_events:
                    context.ledger.record_event(event)
            except Exception:
                cursor.receipt.usage_complete = False
                raise
            finally:
                if not cursor.receipt.usage_complete:
                    context.ledger.record(
                        "unattributed-" + cursor.receipt.execution_id,
                        "retrieval",
                        None,
                        None,
                        "failure",
                    )
        if not cursor.receipt.usage_complete:
            raise MissingUsageError("retrieval receipt has incomplete request usage")
        return RetrievalResult(packed, cursor.receipt)
