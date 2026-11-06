"""AppWorld memory arms for TriMEM (default: HEAD agent, appworld 0.2 / data 0.2.0; also 0.1.3.post1 / 0.1.0).

Modules (plan ``docs/trimem_three_dataset_plan_v0.1.0.md`` steps A1-A3):

- ``common``: task ids, splits, specs, ground-truth API calls, path guards (no appworld import).
- ``store_build``: the ``trimem_appworld`` store (task -> session -> step, native graph edges) from
  train 90: ground-truth API-call sessions plus, later, the official agent's own successful runs.
- ``adapter``: ``AppWorldTaskAdapter`` (``SessionAdapter``), memory cards, L1 scenario isolation.
- ``agent_head``: glue to the official ``vendor/appworld`` HEAD ReAct code agent (``appworld_agents``
  simplified); imported only in ``vendor/appworld-venv-0.2`` (default, ``run.py --agent_version head``).
- ``agent``: glue to the official v0.1.3.post1 ReAct agent (recoma); imported only in
  ``vendor/appworld-venv`` (``run.py --agent_version 0.1.3``).
- ``run``: per-task runner (retrieve once, inject, run, official evaluation, opt row).
- ``build_panel``: frozen profiling panel train 90 + dev 57.

AppWorld plaintext (instructions, ground truth, agent outputs) must stay under ``/data/`` (gitignored)
and is never committed or published.
"""
