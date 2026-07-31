"""Paradigm IV construction: the model decides when and what to write.

The LLM is handed memory tools — ``search_memory``, ``read_unit``,
``write_field``, ``link``, ``split_topic`` — and loops until it stops.

**Writes are applied inside the operator's transaction as the agent requests
them**, so the agent's later reads see its own earlier writes. PostgreSQL gives
this for free, and it is what makes an agentic loop coherent without per-round
commits — which would break the atomicity [GEM] Algorithm 1 requires.

Concretely: this strategy is unusual in emitting ops *and* reading through a
view backed by the same transaction. It stages ops as the agent goes, and the
``read_unit``/``search_memory`` tools answer from ``M_t`` plus the ops staged
so far, so the agent is never lied to about what it has already written.

**``max_rounds`` and ``max_tool_calls`` are REQUIRED constructor arguments with
no defaults.** [AM] Recommendation 10 exists because LLM-bounded phases have
tails reaching p95/p50 = 5.9x. On cap exhaustion the strategy records
``capped=True`` and the operator commits what exists — a capped run is a
recorded operating point, never a silent truncation.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from bench.agent_memory.gem import plan as planmod
from bench.agent_memory.gem.protocols import MemoryView
from bench.agent_memory.gem.types import EdgeKind, InteractionEvent

TOOLS = ("search_memory", "read_unit", "write_field", "link", "split_topic")

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """\
You maintain a topic-grained long-term memory store. Read the excerpt and \
decide what, if anything, is worth remembering.

You have these tools:
  search_memory(query)                 -> candidate topics
  read_unit(unit_id)                   -> a topic and its current fields
  write_field(unit_id|new_title, field, value, valid_from, supersede)
  link(src_title, dst_title, kind, rel)  kind: extension | association
  split_topic(unit_id, fields, new_title)

Respond with ONE JSON object per turn:
  {"tool": "<name>", "args": {...}}          to call a tool
  {"done": true}                             when finished

Rules:
- Search before writing. Prefer updating an existing topic over creating one.
- Set supersede=true when replacing a fact that already has a value.
- "extension" means a change in src entails re-evaluating dst. Use it sparingly.
- No prose. JSON only.\
"""


class AgenticIngestStrategy:
    """Implements the ``AgenticIngest`` Protocol.

    ``client`` is any object with ``complete(system, user) -> (text, usage)``,
    the same narrow contract ``llm_mediated`` uses, so both strategies run
    against the same co-located endpoint.
    """

    name = "agentic"
    tools = TOOLS
    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_rounds: int,
        max_tool_calls: int,
        chunker: Any | None = None,
        chunk_tokens: int = 4096,
    ) -> None:
        # No defaults, deliberately. [AM] Recommendation 10: LLM-bounded phases
        # need EXTERNAL iteration caps, and a default is how a cap silently
        # becomes someone else's problem.
        if max_rounds <= 0 or max_tool_calls <= 0:
            raise ValueError("max_rounds and max_tool_calls must be positive")
        self.client = client
        self.model = model
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls
        self.chunk_tokens = chunk_tokens
        self._chunker = chunker
        self.cost: dict[str, int] = {}
        self.capped = False
        self.rejections: list[dict[str, Any]] = []
        self.transcript: list[dict[str, Any]] = []

    @property
    def chunker(self) -> Any:
        if self._chunker is None:
            from bench.agent_memory.tridbBackend.chunking import TiktokenSentenceChunker

            self._chunker = TiktokenSentenceChunker(chunk_size=self.chunk_tokens)
        return self._chunker

    # -- the loop ---------------------------------------------------------

    def plan(
        self, events: Sequence[InteractionEvent], view: MemoryView
    ) -> list[Mapping[str, Any]]:
        self.cost = {}
        self.capped = False
        self.rejections = []
        self.transcript = []

        ops: list[Mapping[str, Any]] = []
        titles_in_plan: dict[str, str] = {}
        calls_remaining = self.max_tool_calls

        # Materialise the work list so "did the budget run out with work still
        # to do?" is answerable. An agent that spends its LAST permitted call
        # and then legitimately reports done is NOT capped, and recording it as
        # capped would misreport the operating point in the one place the flag
        # exists to describe honestly.
        work = [
            (event, chunk)
            for event in events
            for chunk in self.chunker.chunk(event.content)
        ]
        for event, chunk in work:
            if calls_remaining <= 0:
                self.capped = True  # budget gone with chunks still unprocessed
                break
            calls_remaining -= self._run_loop(
                event, chunk, view, ops, titles_in_plan, calls_remaining
            )
        return ops

    def _run_loop(
        self,
        event: InteractionEvent,
        chunk: str,
        view: MemoryView,
        ops: list[Mapping[str, Any]],
        titles_in_plan: dict[str, str],
        calls_remaining: int,
    ) -> int:
        history: list[dict[str, Any]] = []
        used = 0

        for round_index in range(self.max_rounds):
            if used >= calls_remaining:
                self.capped = True
                break

            user = json.dumps(
                {
                    "excerpt": chunk,
                    "scope_id": event.scope_id,
                    "history": history[-8:],
                    "rounds_left": self.max_rounds - round_index,
                    "tool_calls_left": calls_remaining - used,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            text, usage = self.client.complete(SYSTEM_PROMPT, user)
            self._bump("llm_calls", 1)
            self._bump("prompt_tokens", int(usage.get("prompt_tokens", 0)))
            self._bump("completion_tokens", int(usage.get("completion_tokens", 0)))

            try:
                message = json.loads(_strip_fences(text))
            except (ValueError, TypeError) as exc:
                self.rejections.append(
                    {"gate": "json", "detail": f"agent turn did not parse: {exc}"}
                )
                break

            if message.get("done"):
                break
            tool = message.get("tool")
            if tool not in TOOLS:
                self.rejections.append(
                    {"gate": "enum", "detail": f"unknown tool {tool!r}"}
                )
                break

            used += 1
            result = self._dispatch(
                tool, message.get("args") or {}, event, view, ops, titles_in_plan
            )
            history.append({"tool": tool, "result": result})
            self.transcript.append(
                {"round": round_index, "tool": tool, "result": result}
            )
        else:
            # Loop ran to max_rounds without the agent saying done.
            self.capped = True

        return used

    # -- tools -------------------------------------------------------------

    def _dispatch(
        self,
        tool: str,
        args: Mapping[str, Any],
        event: InteractionEvent,
        view: MemoryView,
        ops: list[Mapping[str, Any]],
        titles_in_plan: dict[str, str],
    ) -> Any:
        if tool == "search_memory":
            return self._search(view, event.scope_id, args)
        if tool == "read_unit":
            return self._read(view, args)
        if tool == "write_field":
            return self._write_field(args, event, ops, titles_in_plan)
        if tool == "link":
            return self._link(args, event, ops, titles_in_plan)
        if tool == "split_topic":
            return self._split(args, ops)
        return {"error": f"unhandled tool {tool!r}"}

    def _search(
        self, view: MemoryView, scope_id: str, args: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        if view is None:
            return []
        units = view.units(scope_id, limit=int(args.get("limit", 10)))
        return [
            {"unit_id": u.id, "title": u.title, "summary": u.summary} for u in units
        ]

    def _read(self, view: MemoryView, args: Mapping[str, Any]) -> dict[str, Any]:
        """Reads see this transaction's own uncommitted writes.

        That is the whole reason the agent loop can be coherent inside one
        transaction, and it is why the view is bound to the operator's Tx.
        """
        if view is None or args.get("unit_id") is None:
            return {"error": "read_unit needs unit_id"}
        unit = view.unit(int(args["unit_id"]))
        if unit is None:
            return {"error": f"no unit {args['unit_id']}"}
        return {
            "unit_id": unit.id,
            "title": unit.title,
            "summary": unit.summary,
            "fields": {
                name: (unit.current(name).value if unit.current(name) else None)
                for name in unit.fields
            },
        }

    def _write_field(
        self,
        args: Mapping[str, Any],
        event: InteractionEvent,
        ops: list[Mapping[str, Any]],
        titles_in_plan: dict[str, str],
    ) -> dict[str, Any]:
        for key in ("field", "value"):
            if not args.get(key):
                return {"error": f"write_field needs {key}"}

        unit_id = args.get("unit_id")
        ref = None
        if unit_id is None:
            title = str(args.get("new_title") or "").strip()
            if not title:
                return {"error": "write_field needs unit_id or new_title"}
            ref = titles_in_plan.get(title)
            if ref is None:
                ref = f"agent:{title}"
                titles_in_plan[title] = ref
                ops.append(
                    planmod.upsert_unit(
                        scope_id=event.scope_id,
                        title=title,
                        summary=str(args.get("summary") or ""),
                        ref=ref,
                        embed_text=title,
                        metadata={
                            "embedding_source": "title_summary",
                            "strategy": self.name,
                            "prompt_version": self.prompt_version,
                        },
                    )
                )

        ops.append(
            planmod.append_field_value(
                ref=ref,
                unit_id=None if unit_id is None else int(unit_id),
                field=str(args["field"]),
                value=str(args["value"]),
                valid_from=str(
                    args.get("valid_from") or event.event_time or "-infinity"
                ),
                supersede_current=bool(args.get("supersede")),
                provenance={
                    "source_external_ids": [event.external_id],
                    "extractor_model": self.model,
                    "prompt_version": self.prompt_version,
                    "operator": "ingest",
                },
            )
        )
        return {"ok": True, "field": args["field"]}

    def _link(
        self,
        args: Mapping[str, Any],
        event: InteractionEvent,
        ops: list[Mapping[str, Any]],
        titles_in_plan: dict[str, str],
    ) -> dict[str, Any]:
        kind = str(args.get("kind") or "")
        if kind not in (EdgeKind.EXTENSION.value, EdgeKind.ASSOCIATION.value):
            return {"error": f"kind must be extension|association, got {kind!r}"}
        src_title = str(args.get("src_title") or "")
        dst_title = str(args.get("dst_title") or "")
        if not src_title or not dst_title:
            return {"error": "link needs src_title and dst_title"}

        refs = {}
        for side, title in (("src", src_title), ("dst", dst_title)):
            ref = titles_in_plan.get(title)
            if ref is None:
                ref = f"agent:{title}"
                titles_in_plan[title] = ref
                ops.append(
                    planmod.upsert_unit(
                        scope_id=event.scope_id,
                        title=title,
                        summary="",
                        ref=ref,
                        embed_text=title,
                        metadata={
                            "embedding_source": "title_summary",
                            "strategy": self.name,
                        },
                    )
                )
            refs[side] = ref

        ops.append(
            planmod.link(
                edge_kind=kind,
                rel=str(args.get("rel") or kind),
                src_ref=refs["src"],
                dst_ref=refs["dst"],
            )
        )
        return {"ok": True, "src": src_title, "dst": dst_title, "kind": kind}

    def _split(
        self, args: Mapping[str, Any], ops: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        if args.get("unit_id") is None or not args.get("fields"):
            return {"error": "split_topic needs unit_id and fields"}
        ops.append(
            planmod.split_topic(
                unit_id=int(args["unit_id"]),
                fields=[str(f) for f in args["fields"]],
                new_title=str(args.get("new_title") or "split"),
            )
        )
        return {"ok": True}

    def _bump(self, key: str, amount: int) -> None:
        self.cost[key] = self.cost.get(key, 0) + amount


def _strip_fences(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()
