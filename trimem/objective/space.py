"""Plan search space v2: graph-traversal configuration × plan family × k (plan v0.3.0).

A graph configuration fixes how the candidate pool is reached from the ANN seeds:

* ``edge``    — ``lineage`` (parent → child, successful edits) or ``similar`` (cross-session kNN);
* ``dir``     — ``down`` follows stored edges, ``up`` follows the reverse edge types
  (``lineage_up`` = child → parent, ``similar_up`` = reverse kNN), both native in the graph store;
* ``hops``    — 1 or 2 chained TriMEM Expand nodes (union of both frontiers for 2);
* ``fanout``  — per-seed cap of each Expand (4 / 8 / 16).

The v1 plans are the single configuration ``lineage/down/1/8``; the constraint objective's
reference candidate ``rank:5`` is taken under it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphConfig:
    edge: str = "lineage"
    dir: str = "down"
    hops: int = 1
    fanout: int = 8

    @property
    def edge_type(self) -> str:
        return self.edge if self.dir == "down" else f"{self.edge}_up"

    @property
    def hop_types(self) -> tuple[str, ...]:
        """Edge type of each chained Expand. `mixed` = cross-session + within-session:
        down = similar (into another session) then `hops` lineage steps (its successful descendants);
        up = `hops` lineage_up steps (the ancestors) then similar (their look-alikes elsewhere)."""
        if self.edge == "mixed":
            return ("similar",) + ("lineage",) * self.hops if self.dir == "down" else \
                ("lineage_up",) * self.hops + ("similar",)
        return (self.edge_type,) * self.hops

    @property
    def name(self) -> str:
        return f"{self.edge}/{self.dir}/{self.hops}/{self.fanout}"

    @classmethod
    def parse(cls, name: str) -> GraphConfig:
        e, d, h, f = name.split("/")
        return cls(e, d, int(h), int(f))


DEFAULT = GraphConfig()
GRAPH_SPACE: tuple[GraphConfig, ...] = tuple(
    GraphConfig(e, d, h, f)
    for e, d, h, f in itertools.product(("lineage", "similar"), ("down", "up"), (1, 2), (4, 8, 16))
)
# v3 (superseded by depth_space): + mixed path (similar -> lineage, down) at every fanout
GRAPH_SPACE_V3: tuple[GraphConfig, ...] = GRAPH_SPACE + tuple(GraphConfig("mixed", "down", 1, f) for f in (4, 8, 16))


def depth_space(depths: tuple[int, ...]) -> tuple[GraphConfig, ...]:
    """Traversal-depth space (2026-10-12): lineage and mixed paths take `depths` = the task's
    {1, mean lineage depth, 90th-percentile lineage depth}; similar keeps {1, 2} (16-NN: deeper
    covers the whole task graph). Direction down/up, fanout 4/8/16 -> 48 configurations."""
    out = []
    for f in (4, 8, 16):
        for d in ("down", "up"):
            out += [GraphConfig("lineage", d, h, f) for h in depths]
            out += [GraphConfig("similar", d, h, f) for h in (1, 2)]
            out += [GraphConfig("mixed", d, h, f) for h in depths]
    return tuple(dict.fromkeys(out))


DEPTH_SQL = """
WITH RECURSIVE dd AS (
  SELECT id, 0 AS depth FROM attempt WHERE task = %s AND parent_id IS NULL AND cell IS NULL
  UNION ALL
  SELECT a.id, dd.depth + 1 FROM attempt a JOIN dd ON a.parent_id = dd.id WHERE a.cell IS NULL AND dd.depth < 500)
SELECT round(avg(depth))::int, percentile_disc(0.9) WITHIN GROUP (ORDER BY depth) FROM dd
"""


def task_depths(conn, task: str) -> tuple[int, ...]:
    """{1, mean, p90} of the corpus lineage depth of `task` (distinct, ascending)."""
    mean, p90 = conn.execute(DEPTH_SQL, (task,)).fetchone()
    return tuple(sorted({1, max(1, int(mean or 1)), max(1, int(p90 or 1))}))
REFERENCE = ("rank", 5, DEFAULT)  # constraint objective: be at least as good as rank:5 on the v1 graph
POOL_M_V2 = 8  # rows per configuration (k <= 8); semantic passes run over the union


def pool_sql(g: GraphConfig) -> str:
    """Tri-modal pool for one configuration. Gain = improvement over the row's OWN parent, which
    equals the v1 'improvement over the seed' for lineage/down/1 and stays meaningful for the
    other directions (an ancestor is rarely better than the seed it was reached from)."""
    types = g.hop_types
    # one CTE per hop, each expanding from the DISTINCT frontier of the previous hop (fanout-capped);
    # the pool is the union of every frontier, except for mixed/down where the similar hop is only a
    # bridge into another session (the pool = its descendants) and mixed/up where it is the target.
    ctes = [f"h1 AS (SELECT DISTINCT e.dst FROM s, LATERAL trimem.expand(s.vid, '{types[0]}', {g.fanout}) e)"]
    for i, t in enumerate(types[1:], start=2):
        ctes.append(f"h{i} AS (SELECT DISTINCT e.dst FROM h{i - 1}, LATERAL trimem.expand(h{i - 1}.dst, '{t}', {g.fanout}) e)")
    n = len(types)
    if g.edge == "mixed" and g.dir == "down":
        keep = range(2, n + 1)
    elif g.edge == "mixed":
        keep = range(n, n + 1)
    else:
        keep = range(1, n + 1)
    union = " UNION ".join(f"SELECT dst FROM h{i}" for i in keep)
    hop2 = ",\n" + ",\n".join(ctes[1:] + [f"x AS ({union})"]) if n > 1 else f",\nx AS ({union})"
    return f"""
WITH s AS (SELECT vid FROM attempt
           WHERE task = %(task)s AND cell IS NULL AND emb IS NOT NULL
           ORDER BY emb <=> %(q)s::vector LIMIT 24),
{ctes[0]}{hop2}
SELECT d.id, d.fitness, p.fitness AS parent_fitness, d.changes, d.code, coalesce(d.tokens, 0),
       1 - (d.emb <=> %(q)s::vector) AS sim, d.session
FROM x JOIN attempt d ON d.vid = x.dst JOIN attempt p ON p.id = d.parent_id
WHERE d.is_valid AND d.fitness > p.fitness AND d.fitness >= %(floor)s
  AND d.cell IS NULL AND d.code IS NOT NULL
ORDER BY d.fitness - p.fitness DESC, d.id LIMIT %(m)s
"""
