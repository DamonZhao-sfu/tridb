"""``trimem_memory_agent``: the official τ² ``LLMAgent`` plus a ``<memory>`` block after its system prompt.

Nothing else changes: same instruction and policy text, tools, ``generate`` call, user simulator,
orchestrator and evaluator. The orchestrator sends the hard-coded greeting, so the agent's first LLM
call happens on the customer's first message; at that moment (once per episode) the agent asks its
memory provider for memory with that message as the retrieval query -- the agent never sees the
task's ``reason_for_call`` / instructions -- and rebuilds its system message as
``official system prompt + "\\n\\n" + <memory>...</memory>``. With an empty retrieval the prompt is
byte-identical to the official one.

The provider is ``provider(task, first_user_message) -> {"memory": str, ...}``; ``task`` is passed
only so the provider can apply isolation (query identity). Register with ``register()``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.registry import registry

AGENT_NAME = "trimem_memory_agent"
MemoryProvider = Callable[[object, str], dict]
_PROVIDER: dict[str, Optional[MemoryProvider]] = {"fn": None}


class TrimemMemoryAgent(LLMAgent):
    def __init__(
        self,
        tools,
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        task=None,
        provider: Optional[MemoryProvider] = None,
    ):
        super().__init__(
            tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args
        )
        self.task = (
            task  # query identity for isolation only; never rendered into the prompt
        )
        self.provider = provider
        self.memory = ""
        self.memory_meta: Optional[dict] = None
        self.query_text: Optional[str] = None
        self.retrieved = False

    @property
    def official_system_prompt(self) -> str:
        return LLMAgent.system_prompt.fget(self)

    @property
    def system_prompt(self) -> str:
        base = self.official_system_prompt
        return f"{base}\n\n{self.memory}" if self.memory else base

    def _retrieve(self, text: str, state) -> None:
        self.retrieved = True
        self.query_text = text
        if self.provider is None:
            return
        result = self.provider(
            self.task, text
        )  # may raise (e.g. quality reuse ends the episode)
        self.memory_meta = result
        self.memory = result.get("memory") or ""
        state.system_messages = [
            SystemMessage(role="system", content=self.system_prompt)
        ]

    def _generate_next_message(self, message, state):
        if (
            not self.retrieved
            and isinstance(message, UserMessage)
            and isinstance(message.content, str)
            and message.content.strip()
        ):
            self._retrieve(message.content.strip(), state)
        return super()._generate_next_message(message, state)


def set_memory_provider(fn: Optional[MemoryProvider]) -> None:
    """Process-wide provider used by the registered factory (the runner sets it once)."""
    _PROVIDER["fn"] = fn


def create_trimem_memory_agent(tools, domain_policy, **kwargs) -> TrimemMemoryAgent:
    return TrimemMemoryAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
        provider=kwargs.get("provider", _PROVIDER["fn"]),
    )


def register() -> str:
    if AGENT_NAME not in registry.get_agents():
        registry.register_agent_factory(create_trimem_memory_agent, AGENT_NAME)
    return AGENT_NAME
