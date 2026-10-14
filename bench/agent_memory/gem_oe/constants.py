"""Identifiers shared between the injector and the tools that audit it.

Separate from `memory_database` so an auditing tool does not have to import
`openevolve` -- the gates run under the repo venv, the runner under `.venv-e0`.
"""

from __future__ import annotations

#: Prefix on every injected program id. Also the marker the prompt assertion greps for,
#: and what keeps injected ids from ever colliding with OpenEvolve's own uuid4s.
EXTERNAL_PREFIX = "gemext:"

#: Metadata key set on every injected program.
EXTERNAL_FLAG = "tridb_external"


def is_external(program_id: str) -> bool:
    return program_id.startswith(EXTERNAL_PREFIX)
