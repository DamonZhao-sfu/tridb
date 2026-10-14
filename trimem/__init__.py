"""TriMEM: tri-modal agent memory with planner-controlled LLM semantic operators.

The C extension ``src/trimem_pg`` plans ``trimem.sem_filter`` as a ``TriMEM SemFilter`` node,
``trimem.sem_score`` / ``trimem.sem_extract`` as ``TriMEM SemScore`` / ``TriMEM SemExtract`` nodes
and ``trimem.expand`` as a ``TriMEM Expand`` node. This package holds the out-of-process side: the
LLM worker that answers the SemFilter mailbox, the cache-key function, and the LLM clients.
"""

DEFAULT_MODEL = "Qwen/Qwen3.8-27B-FP8"
DEFAULT_PROMPT_VERSION = "trimem-filter-v1"
OP_FILTER = "sem_filter"
OP_SCORE = "sem_score"
OP_EXTRACT = "sem_extract"
