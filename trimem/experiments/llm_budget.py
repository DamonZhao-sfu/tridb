"""Token budgets shared by the trajectory-memory experiments, as versioned protocols.

User decision 2026-10-15 (protocol v2): the served model context, the retrieved-memory injection cap
and the per-call output cap are the same for every dataset and arm. Change them only by adding a
NEW protocol here; never edit a published one (results of different protocols are not mixed).

- ``llm-budget-v2-2026-10-15``: context 131,072 / memory 32,768 / output 8,192. Still the
  module-level default (``PROTOCOL``, ``CONTEXT_TOKENS``, ... below), so every runner that does not
  ask for another protocol reproduces its earlier runs unchanged.
- ``llm-budget-v3-2026-10-18`` (user default for the 5-repo SWE memory experiment, plan
  docs/trimem_swe_memory_plan_v0.1.0.md): context 262,144 / memory 131,072 / output 8,192; vLLM
  8030/8031 serve --max-model-len 262144 (scripts/serve_qwen38_ctx256k.sh). The SWE runners
  (``trimem.experiments.swebench``) use ``SWE``.

Measured before v2 (32,768 context / 8,192 memory): SWE-QA-Pro No-Memory calls reached 30,480
prompt tokens; every SWE-QA memory card was truncated (card median 16,382, p90 28,282 tokens);
ALFWorld / ScienceWorld calls peak near 7k.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Budget:
    protocol: str
    context_tokens: int  # vLLM --max-model-len of the Qwen3.8 endpoints
    memory_tokens: int  # total retrieved-memory tokens injected per episode / question
    output_tokens: int  # max new tokens per LLM call

    def per_card(self, top_k: int) -> int:
        """Equal split of the memory cap across the selected cards."""
        return self.memory_tokens // max(1, top_k)

    def as_dict(self) -> dict:
        return asdict(self)


V2 = Budget("llm-budget-v2-2026-10-15", 131_072, 32_768, 8_192)
V3 = Budget("llm-budget-v3-2026-10-18", 262_144, 131_072, 8_192)
PROTOCOLS = {b.protocol: b for b in (V2, V3)}
SWE = V3  # every SWE memory runner (agent_driver, retrieval, rebench_run, native_run, swe5_*)


def get(protocol: str) -> Budget:
    if protocol not in PROTOCOLS:
        raise KeyError(
            f"unknown LLM budget protocol {protocol!r}; known: {sorted(PROTOCOLS)}"
        )
    return PROTOCOLS[protocol]


# module-level default = v2, unchanged for the non-SWE runners
PROTOCOL = V2.protocol
CONTEXT_TOKENS = V2.context_tokens
MEMORY_TOKENS = V2.memory_tokens
OUTPUT_TOKENS = V2.output_tokens


def per_card(top_k: int) -> int:
    """Equal split of the v2 memory cap across the selected cards."""
    return V2.per_card(top_k)


def as_dict() -> dict:
    return {
        "protocol": PROTOCOL,
        "context_tokens": CONTEXT_TOKENS,
        "memory_tokens": MEMORY_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
    }
