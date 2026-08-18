# E1 组合代价与模态消融 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在同一批 query 上，测出 TriDB 单进程与 Polyglot-Tuned（Milvus + Neo4j + pgvector）的等质量延迟差距与其机理归因，并在 TriDB 内部通过模态消融证明三模态的必要性。

**Architecture:** 复用 E0 已有的 `experiments/e0/plan_spread` 骨架（`QuerySpec` / `PlanSpec` / checkpoint runner / `parquet_reference` oracle）。新增两个 runner：交错 A/B runner（同进程内随机顺序轮流执行两个 backend，消除机器漂移）与消融 runner（在 TriDB 内关断模态腿）。模态关断通过给 benchmark-only 的 `tjs_e0_open` 增加 `graph_off` / `vector_off` 两个布尔参数实现，不触碰 `graph_query()` 公开 v1 模板。

**Tech Stack:** Python 3.13（`.venv-e0`）、pytest、pyarrow、psycopg 3、pymilvus、neo4j-python-driver、PostgreSQL C 扩展（stock PG 16）、yaml。

**Spec:** `docs/superpowers/specs/2026-08-18-e1-composition-design.md`

## Global Constraints

- **venv 选择（预检时写反了，已更正）**：`.venv/bin/python`（Makefile 的 `$(PY)`）用于所有
  接触 `live_backend.py` 或 `tridb_backend.py` 的命令，以及所有 `tests/test_e1_*.py`。
  `.venv/bin/python`（`$(E0_PY)`）只用于数据准备与 `parquet_reference` 路径。
  依据是仓库既有约定：Makefile 里 `e0-plan-live`、`e0-openevolve-polyglot-load`、
  `e0-tridb-load`、`e0-tridb-run`、`e0-tridb-analyze` 全部用 `$(PY)`，只有
  `e0-plan-reference-*` 和数据准备用 `$(E0_PY)`。`.venv-e0` 由 `requirements-e0.lock`
  固定，刻意不含 `neo4j` / `pymilvus` / `psycopg`。
- TriDB 连接：host `/localhome/hza214/tridb/.tridb-pgdata`（unix socket 目录），port `55432`，db `tridb_e0_stark` / `tridb_e0_openevolve`。
- Polyglot 连接：Milvus `127.0.0.1:19530`、Neo4j `bolt://127.0.0.1:7688`（auth `neo4j/testpassword`）、pgvector `127.0.0.1:5434` db `tridb_wiki`。
- Polyglot 三件套用 `scripts/baseline_up_podman.sh` 起，镜像固定为 `milvusdb/milvus:v2.4.5`、`neo4j:5.20`、`pgvector/pgvector:pg16`，`--network=host`。
- 质量等价规则（冻结，不得改）：先保留达到该 query 最佳 `hit_at_1`（容差 0.02）的计划，再在其中保留达到最佳 `mrr`（容差 0.02）的计划。`hit_at_5` / `recall_at_20` 只记录。
- C 代码同时面向 PG 13.4 fork 与 stock PG 16/17 访问方法 API。本计划的 C 改动只在 stock PG 16 上验证，**不得**声称 GX10 / ARM64 / CUDA / 32 KiB fork / 128 GB sign-off。
- 所有新产物写入 `results/e1/<run>/`，禁止覆盖 `results/e0/` 下的任何既有目录。
- 每次测量用全新输出目录；checkpoint runner 会 resume 已有 observation 而不是重测。
- 提交信息格式：`type(scope): summary`。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `experiments/e0/plan_spread/live_backend.py` (修改) | Polyglot 后端：补 `same_parent` 谓词、修 OpenEvolve 断点、补跨边界行数计数 |
| `src/tjs_pg/tjs_e0.c` (修改) | benchmark-only 算子：新增 `graph_off` / `vector_off` |
| `experiments/e0/tridb_schema.sql` (修改) | `tjs_e0_open` 函数签名 |
| `experiments/e0/plan_spread/tridb_backend.py` (修改) | 传递两个新布尔参数 |
| `test/e0_tridb_internal_test.sql` (修改) | stock-PG SQL 套件覆盖新参数 |
| `experiments/e1/__init__.py` | 包声明 |
| `experiments/e1/variants.py` | 七个消融变体的定义与到 (shape, predicate, vector_off, graph_off) 的映射 |
| `experiments/e1/plan_selection.py` | 从 E0 结果中挑出每 query 的 5 个计划 |
| `experiments/e1/interleaved_runner.py` | 交错 A/B runner |
| `experiments/e1/ablation_runner.py` | 消融 runner |
| `experiments/e1/analyze_h2h.py` | 等质量比较 + 代价分解 |
| `experiments/e1/analyze_ablation.py` | 消融的三个数 + 分层 |
| `configs/e1/composition_v0.1.yaml` | 冻结契约 |
| `tools/e1/parity_referee.py` | oracle 对账 |
| `tests/test_e1_variants.py` | 变体映射与合法性 |
| `tests/test_e1_plan_selection.py` | 计划挑选 |
| `tests/test_e1_interleave.py` | 交错顺序与 checkpoint |
| `tests/test_e1_analyze.py` | 等质量比较、代价分解、消融分析 |
| `tests/test_e1_parity.py` | 对账判责逻辑 |
| `docs/benchmark_e1_composition_v0.1.0.md` | 最终报告 |

---

## Task 1: 修复 Polyglot 在 OpenEvolve 上的全空结果

**Files:**
- Create: `tools/e1/probe_openevolve.py`
- Modify: `experiments/e0/plan_spread/live_backend.py`

**Interfaces:**
- Produces: 一份可复现的逐 leg 探针报告，确认 Polyglot 后端在 OpenEvolve 上返回非空 `result_ids`。

**背景（不要跳过）：** `results/e0/plan_space/polyglot_live_v0.2/observations.jsonl` 中 OpenEvolve 的
1,010 个 cell 全部 `result_ids == []`，而 `tridb_live_v0.3` 有 727 个非空。已排除的假设：Neo4j
关系名（loader 建 `evolved_to` / `evolved_to_reverse`，`_relationship("evolved_to:reverse")` 正好得
到 `evolved_to_reverse`）、label 名、表名、UUID 字符串类型。**尚未验证**的假设需要容器起来才能测。

- [ ] **Step 1: 起 baseline 三件套**

```bash
cd /localhome/hza214/tridb
scripts/baseline_up_podman.sh
```

Expected: neo4j / pgvector / milvus 三行 ready 日志。若报 subuid/subgid 错误，见脚本内 PREREQUISITES 注释。

- [ ] **Step 2: 写逐 leg 探针**

`tools/e1/probe_openevolve.py`：

```python
"""Probe each Polyglot leg for oe-000 with the exact parameters the E0 run used.

Run:  .venv/bin/python -m tools.e1.probe_openevolve
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("data/e0/openevolve/normalized/seed_42")
LABEL = "E0OpenEvolveNode"
TABLE = "e0_openevolve_node"
COLLECTION = "e0_openevolve"


def main() -> int:
    from neo4j import GraphDatabase
    from pymilvus import Collection, connections
    import psycopg

    query = json.loads(ROOT.joinpath("queries.jsonl").read_text().splitlines()[0])
    anchors = list(query["anchor_ids"])
    print("query_id", query["query_id"], "anchors", anchors)
    print("edge_types", query["edge_types"], "target", query["target_entity_type"])

    qemb = pq.read_table(ROOT / "query_embeddings.parquet")
    vec_by_id = dict(
        zip(
            [str(v) for v in qemb["query_id"].to_pylist()],
            qemb["embedding"].to_pylist(),
        )
    )
    vector = np.asarray(vec_by_id[query["query_id"]], dtype=np.float32)

    connections.connect(alias="probe", host="127.0.0.1", port="19530")
    collection = Collection(COLLECTION, using="probe")
    collection.load()
    hits = collection.search(
        data=[vector.tolist()],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=10,
    )
    milvus_ids = [hit.id for hit in hits[0]]
    print("LEG milvus:", len(milvus_ids), milvus_ids[:3])

    driver = GraphDatabase.driver(
        "bolt://127.0.0.1:7688", auth=("neo4j", "testpassword")
    )
    with driver.session() as session:
        present = session.run(
            f"MATCH (a:{LABEL}) WHERE a.node_id IN $anchors RETURN count(a) AS n",
            anchors=anchors,
        ).single()["n"]
        print("LEG neo4j anchors present:", present)
        rels = session.run(
            f"MATCH (:{LABEL})-[r]->(:{LABEL}) "
            "RETURN type(r) AS t, count(*) AS n ORDER BY t"
        ).data()
        print("LEG neo4j relationship types:", rels)
        reached = session.run(
            f"MATCH (a:{LABEL}) WHERE a.node_id IN $anchors "
            f"MATCH (a)-[:evolved_to_reverse*1..2]->(b:{LABEL}) "
            "RETURN DISTINCT b.node_id AS node_id",
            anchors=anchors,
        ).data()
        print("LEG neo4j reached:", len(reached), [r["node_id"] for r in reached][:3])
    driver.close()

    with psycopg.connect(
        host="127.0.0.1", port=5434, dbname="tridb_wiki",
        user="postgres", password="postgres",
    ) as pg:
        with pg.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {TABLE}")
            print("LEG pg rows:", cursor.fetchone()[0])
            cursor.execute(
                f"SELECT count(*) FROM {TABLE} WHERE entity_type = %s",
                (query["target_entity_type"],),
            )
            print("LEG pg predicate matches:", cursor.fetchone()[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: 运行探针，定位断点**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m tools.e1.probe_openevolve
```

Expected: 恰好一条 `LEG ...` 行报告 0 或空。把该行记进提交信息。

若 `LEG neo4j anchors present: 0` → OpenEvolve 数据未装进 Neo4j，执行 `make e0-openevolve-polyglot-load` 后重跑探针。
若 `LEG neo4j relationship types` 里没有 `evolved_to_reverse` → loader 的建边语句没跑到，同上。
若三条 leg 都非空 → 断点在 `live_backend.py` 的组合逻辑，进入 Step 4。

- [ ] **Step 4: 写失败测试锁住修复**

在 `tests/test_e1_parity.py` 中新增（该文件由本任务创建）：

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.integration
def test_polyglot_returns_nonempty_on_openevolve():
    """Regression: the E0 v0.2 run produced empty result_ids on all 1010 OE cells."""
    from experiments.e0.plan_spread.config import load_config
    from experiments.e0.plan_spread.live_backend import PolyglotLiveDataset
    from experiments.e0.plan_spread.model import PlanSpec

    config = load_config(Path("configs/e0/plan_space_v0.3.yaml"))
    spec = config["datasets"]["openevolve"]
    backend = PolyglotLiveDataset("openevolve", spec)
    queries = backend.load_queries(Path(spec["queries"]))
    plan = PlanSpec(shape="traverse_first", k=20, hops=2, predicate_placement="post")
    nonempty = 0
    for query in queries:
        backend.prepare_query(query, {2})
        result = backend.execute(query, plan, top_n=20)
        if result["result_ids"]:
            nonempty += 1
    assert nonempty > 0, "every OpenEvolve query still returns an empty result set"
```

- [ ] **Step 5: 运行测试，确认它失败**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_parity.py::test_polyglot_returns_nonempty_on_openevolve -v
```

Expected: FAIL，`assert 0 > 0`。

- [ ] **Step 6: 按 Step 3 的定位实施最小修复**

修改 `experiments/e0/plan_spread/live_backend.py`。修复必须针对 Step 3 定位到的那一条 leg，
**不要**顺手重构其它部分——本文件同时服务 STARK 的既有结果，任何语义改动都会使 STARK 侧
不可比。若修复改变了 STARK 侧行为，必须在提交信息中说明并重跑 STARK。

- [ ] **Step 7: 运行测试，确认通过**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_parity.py::test_polyglot_returns_nonempty_on_openevolve -v
```

Expected: PASS。

- [ ] **Step 8: 提交**

```bash
cd /localhome/hza214/tridb
git add tools/e1/probe_openevolve.py experiments/e0/plan_spread/live_backend.py tests/test_e1_parity.py
git commit -m "fix(e0): polyglot backend returned empty results on every OpenEvolve query"
```

**时间盒：** 本任务限 1 天。超时则在 `configs/e1/composition_v0.1.yaml` 中把 `datasets` 限定为
`stark_prime`，并在最终报告的 Material Passport 明写 OpenEvolve 未纳入及原因。不要无限期挖下去。

---

## Task 2: Polyglot 侧补齐 `same_parent` 谓词

**Files:**
- Modify: `experiments/e0/plan_spread/live_backend.py`
- Test: `tests/test_e1_parity.py`

**Interfaces:**
- Consumes: Task 1 修复后的 `PolyglotLiveDataset`
- Produces: `PolyglotLiveDataset._predicate_sql` 与 `._neo_predicate` 支持 `same_parent`

**背景：** `data/e0/openevolve/normalized/seed_42/queries.jsonl` 中 10 条 query 有 3 条带
`structured_predicate = {"same_parent": true}`。`tridb_backend.py:110-126` 已实现该谓词（解析
anchor 的 `parent_vid`，要求候选同父且排除 anchor 自身），而 `live_backend.py` 的
`_predicate_sql` / `_neo_predicate` 只处理 `generation_lte` / `generation_gte`，**静默忽略**
`same_parent`。这使 Polyglot 在这 3 条 query 上解一个更宽松的问题，比较不成立。

- [ ] **Step 1: 写失败测试**

在 `tests/test_e1_parity.py` 追加：

```python
def test_polyglot_predicate_sql_honours_same_parent():
    from experiments.e0.plan_spread.model import QuerySpec

    query = QuerySpec(
        query_id="q", dataset="openevolve", query_text="t",
        anchor_ids=("a1",), answer_ids=("x",), edge_types=("evolved_to",),
        hop_limit=2, structured_predicate={"same_parent": True},
        target_entity_type="program", template="t", annotation_status="s",
        require_each_anchor=False,
    )
    from experiments.e0.plan_spread import live_backend

    sql, _params = live_backend.PolyglotLiveDataset._predicate_sql(
        _StubBackend(), query
    )
    assert "parent_id" in sql, f"same_parent not applied, got: {sql}"
```

配套 stub（同文件）：

```python
class _StubBackend:
    table = "e0_openevolve_node"
    _parent_cache: dict[str, list[str]] = {}

    def _parents_of(self, anchor_ids):
        return ["p1"]
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_parity.py::test_polyglot_predicate_sql_honours_same_parent -v
```

Expected: FAIL，`same_parent not applied`。

- [ ] **Step 3: 确认 Polyglot 的 pg 表有 parent 列**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -c "
import psycopg
with psycopg.connect(host='127.0.0.1',port=5434,dbname='tridb_wiki',user='postgres',password='postgres') as c:
    with c.cursor() as cur:
        cur.execute(\"select column_name from information_schema.columns where table_name='e0_openevolve_node'\")
        print([r[0] for r in cur])
"
```

Expected: 列出列名。若没有 parent 列，先在 `tools/e0/load_openevolve_polyglot.py` 的建表与
COPY 中加入 `parent_id text`（取自 `edges.parquet` 的 `src_id → dst_id` 映射，`evolved_to` 方向
的 src 即 parent），重跑 `make e0-openevolve-polyglot-load`。

- [ ] **Step 4: 实现 `same_parent`**

在 `live_backend.py` 的 `_predicate_sql` 中，`generation_gte` 分支之后加入：

```python
        if predicate.get("same_parent"):
            parents = self._parents_of(query.anchor_ids)
            if not parents:
                clauses.append("FALSE")
            else:
                clauses.append("parent_id = ANY(%s)")
                params.append(list(parents))
                clauses.append("NOT (node_id = ANY(%s))")
                params.append(list(query.anchor_ids))
```

并新增取父方法：

```python
    def _parents_of(self, anchor_ids: tuple[Any, ...]) -> list[Any]:
        key = tuple(anchor_ids)
        cached = self._parent_cache.get(key)
        if cached is not None:
            return cached
        with self.pg.cursor() as cursor:
            cursor.execute(
                f"SELECT DISTINCT parent_id FROM {self.table} "
                "WHERE node_id = ANY(%s) AND parent_id IS NOT NULL",
                (list(anchor_ids),),
            )
            parents = [row[0] for row in cursor.fetchall()]
        self._parent_cache[key] = parents
        return parents
```

在 `__init__` 中初始化 `self._parent_cache: dict[tuple[Any, ...], list[Any]] = {}`。

在 `_neo_predicate` 中作对应处理：

```python
        if predicate.get("same_parent"):
            parents = self._parents_of(query.anchor_ids)
            if not parents:
                clauses.append("FALSE")
            else:
                clauses.append("b.parent_id IN $same_parents")
                params["same_parents"] = list(parents)
                clauses.append("NOT b.node_id IN $anchors")
```

Neo4j 侧需要节点带 `parent_id` 属性；若缺，在 `tools/e0/load_openevolve_polyglot.py` 的
`MERGE ... SET` 语句里补 `n.parent_id = r.parent_id` 并重跑装载。

- [ ] **Step 5: 运行测试确认通过**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_parity.py -v
```

Expected: 全部 PASS。

- [ ] **Step 6: 提交**

```bash
cd /localhome/hza214/tridb
git add experiments/e0/plan_spread/live_backend.py tools/e0/load_openevolve_polyglot.py tests/test_e1_parity.py
git commit -m "fix(e0): apply the same_parent predicate on the polyglot backend"
```

---

## Task 3: 补跨边界中间结果行数计数

**Files:**
- Modify: `experiments/e0/plan_spread/live_backend.py`
- Test: `tests/test_e1_parity.py`

**Interfaces:**
- Produces: 每条 observation 新增 `rows_shipped`（整数），与既有 `bytes_shipped` 并列

**背景：** spec §3.3 要求报告跨边界中间结果的**行数**与字节数。现有 `ship()` 只累加字节。

- [ ] **Step 1: 写失败测试**

```python
def test_ship_counts_rows_and_bytes():
    from experiments.e0.plan_spread import live_backend

    counter = live_backend.ShipCounter()
    counter.add(["a", "b", "c"])
    counter.add(["d"])
    assert counter.rows == 4
    assert counter.bytes > 0
    assert counter.serialization_ms >= 0.0
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_parity.py::test_ship_counts_rows_and_bytes -v
```

Expected: FAIL，`AttributeError: module ... has no attribute 'ShipCounter'`。

- [ ] **Step 3: 实现 `ShipCounter`**

在 `live_backend.py` 中，`_bytes` 定义之后加入：

```python
class ShipCounter:
    """Counts rows, bytes, and serialization time crossing a store boundary."""

    def __init__(self) -> None:
        self.rows = 0
        self.bytes = 0
        self.serialization_ms = 0.0

    def add(self, values: list[Any]) -> None:
        started = time.perf_counter_ns()
        self.rows += len(values)
        self.bytes += _bytes(values)
        self.serialization_ms += _ms(started)
```

在 `execute()` 中把局部 `ship()` 闭包替换为 `counter = ShipCounter()` 并把所有 `ship(x)` 调用
改为 `counter.add(x)`；返回字典中把 `bytes_shipped` 改为 `counter.bytes`，
`serialization_ms` 改为 `counter.serialization_ms`，并新增 `"rows_shipped": counter.rows`。

- [ ] **Step 4: 运行确认通过**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_parity.py -v
```

Expected: PASS。

- [ ] **Step 5: TriDB 侧补同名字段（恒为 0）**

在 `experiments/e0/plan_spread/tridb_backend.py` 的 `execute()` 返回字典中，`"bytes_shipped": 0`
旁边加 `"rows_shipped": 0`。TriDB 不跨进程边界，0 是事实而非占位。

- [ ] **Step 6: 提交**

```bash
cd /localhome/hza214/tridb
git add experiments/e0/plan_spread/live_backend.py experiments/e0/plan_spread/tridb_backend.py tests/test_e1_parity.py
git commit -m "feat(e0): record rows crossing a store boundary alongside bytes"
```

---

## Task 4: oracle 对账裁判与公平性门禁

**Files:**
- Create: `tools/e1/parity_referee.py`
- Test: `tests/test_e1_parity.py`

**Interfaces:**
- Produces: `python -m tools.e1.parity_referee --tridb <dir> --polyglot <dir> --reference <dir> --out results/e1/parity_report.json`
- Produces: `referee_verdicts(tridb_rows, poly_rows, ref_rows) -> list[dict]`，每条含
  `query_id`、`plan_id`、`verdict ∈ {"agree","tridb_wrong","polyglot_wrong","both_wrong"}`、
  `tridb_jaccard`、`polyglot_jaccard`

**背景：** 现有两个 E0 run 在 STARK 上有 173 个 cell 双方 `hit_at_1` / `mrr` 不一致。必须用
`parquet_reference` backend（精确 cosine + 精确可达性，无 ANN 近似）判责，否则质量对比不可发表。

- [ ] **Step 1: 写失败测试**

```python
def test_referee_assigns_blame_against_the_oracle():
    from tools.e1.parity_referee import referee_verdicts

    ref = [{"query_id": "q", "plan_id": "p", "result_ids": ["a", "b", "c"]}]
    tridb = [{"query_id": "q", "plan_id": "p", "result_ids": ["a", "b", "c"]}]
    poly = [{"query_id": "q", "plan_id": "p", "result_ids": ["x", "y", "z"]}]

    verdicts = referee_verdicts(tridb, poly, ref)
    assert len(verdicts) == 1
    assert verdicts[0]["verdict"] == "polyglot_wrong"
    assert verdicts[0]["tridb_jaccard"] == 1.0
    assert verdicts[0]["polyglot_jaccard"] == 0.0
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_parity.py::test_referee_assigns_blame_against_the_oracle -v
```

Expected: FAIL，`ModuleNotFoundError: tools.e1.parity_referee`。

- [ ] **Step 3: 实现裁判**

`tools/e1/parity_referee.py`：

```python
"""Referee TriDB and Polyglot results against the exact parquet oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.e0.common import write_json

AGREEMENT_FLOOR = 0.98


def _jaccard(left: list[Any], right: list[Any]) -> float:
    a, b = set(left[:20]), set(right[:20])
    if not a and not b:
        return 1.0
    return len(a & b) / float(len(a | b))


def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        out.setdefault((row["query_id"], row["plan_id"]), row)
    return out


def referee_verdicts(
    tridb_rows: list[dict[str, Any]],
    poly_rows: list[dict[str, Any]],
    ref_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tridb, poly, ref = _index(tridb_rows), _index(poly_rows), _index(ref_rows)
    verdicts = []
    for key in sorted(set(tridb) & set(poly) & set(ref)):
        truth = ref[key]["result_ids"]
        tj = _jaccard(tridb[key]["result_ids"], truth)
        pj = _jaccard(poly[key]["result_ids"], truth)
        if tj >= AGREEMENT_FLOOR and pj >= AGREEMENT_FLOOR:
            verdict = "agree"
        elif tj >= AGREEMENT_FLOOR:
            verdict = "polyglot_wrong"
        elif pj >= AGREEMENT_FLOOR:
            verdict = "tridb_wrong"
        else:
            verdict = "both_wrong"
        verdicts.append(
            {
                "query_id": key[0],
                "plan_id": key[1],
                "verdict": verdict,
                "tridb_jaccard": tj,
                "polyglot_jaccard": pj,
                "shape": ref[key].get("shape"),
                "k": ref[key].get("k"),
                "dataset": ref[key].get("dataset"),
            }
        )
    return verdicts


def _load(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("repetition") == 0 and row.get("status") == "ok":
                rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tridb", type=Path, required=True)
    parser.add_argument("--polyglot", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    verdicts = referee_verdicts(
        _load(args.tridb / "observations.jsonl"),
        _load(args.polyglot / "observations.jsonl"),
        _load(args.reference / "observations.jsonl"),
    )
    counts: dict[str, int] = {}
    for row in verdicts:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    tj = sorted(row["tridb_jaccard"] for row in verdicts)
    pj = sorted(row["polyglot_jaccard"] for row in verdicts)
    median = lambda xs: xs[len(xs) // 2] if xs else 0.0  # noqa: E731
    report = {
        "schema_version": "e1-parity-v0.1.0",
        "agreement_floor": AGREEMENT_FLOOR,
        "cells": len(verdicts),
        "counts": counts,
        "tridb_jaccard_median": median(tj),
        "polyglot_jaccard_median": median(pj),
        "gate_passed": (
            median(tj) >= AGREEMENT_FLOOR and median(pj) >= AGREEMENT_FLOOR
        ),
        "verdicts": verdicts,
    }
    write_json(args.out, report)
    print(json.dumps({k: report[k] for k in
                      ("cells", "counts", "gate_passed")}, sort_keys=True))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_parity.py -v
```

Expected: PASS。

- [ ] **Step 5: 跑参照 run 并执行门禁**

**已由控制器预先跑完**：`results/e1/reference_full_v0.3` 存在且 `complete: true`
（3,450 cell、0 error、61 s），与两个 live run 的 cell 三方重合 3,450/3,450。若该目录已存在
且 manifest `complete` 为 true，**跳过本命令**，直接做判责。否则用它重建（注意这是唯一仍用
`$(E0_PY)` 的命令）：

```bash
cd /localhome/hza214/tridb
.venv-e0/bin/python -m experiments.e0.plan_spread.runner \
  --backend parquet_reference --config configs/e0/plan_space_v0.3.yaml \
  --output-dir results/e1/reference_full_v0.3 --repetitions 1
```

然后判责：

```bash
mkdir -p results/e1
.venv/bin/python -m tools.e1.parity_referee \
  --tridb results/e0/plan_space/tridb_live_v0.3 \
  --polyglot results/e0/plan_space/polyglot_live_v0.2 \
  --reference results/e1/reference_full_v0.3 \
  --out results/e1/parity_report.json
```

Expected: 打印 `gate_passed`。**门禁未通过则不得进入 Task 6 之后的测量任务**——先按 verdict
分布定位是哪一侧错，回到 Task 1/2 修复，或证明差异属 ANN 近似（Step 6）。

- [ ] **Step 6: 若判为 ANN 近似，出示收敛证据**

对被判 `*_wrong` 的 cell，把 TriDB 的 `hnsw_ef_search` 与 Milvus 的 `ef` 提到 1000 重跑该 cell，
确认结果收敛到 oracle。收敛则该 cell 归类为合法 ANN 近似，在报告中单列。不收敛则是 bug。

- [ ] **Step 7: 提交**

```bash
cd /localhome/hza214/tridb
git add tools/e1/parity_referee.py tests/test_e1_parity.py results/e1/parity_report.json
git commit -m "feat(e1): referee TriDB and Polyglot against the exact oracle before comparing"
```

---

## Task 5: 标注 E0 中被污染的 OpenEvolve 结论

**Files:**
- Modify: `docs/benchmark_e0_plan_space_v0.1.0.md`

**背景：** `results/e0/plan_space/polyglot_live_v0.2/summary.json` 报 OpenEvolve plan spread
中位 9.38×。但该数据集全部 1,010 个 cell 的 `result_ids` 为空 → 全部 `hit_at_1 = 0` → 质量等价集
退化为整个计划网格 → spread 变成整个网格的 max/min，与「质量等价前提下」的定义不符。这个数
必须撤回，不能留在文档里被后续引用。

- [ ] **Step 1: 在文档顶部加撤回声明**

在 `docs/benchmark_e0_plan_space_v0.1.0.md` 的 TL;DR 之前插入：

```markdown
> **⚠️ 部分结论已撤回（2026-08-18）。** 本文中一切依赖 Polyglot 在 **OpenEvolve** 上质量指标
> 的数字均无效：`results/e0/plan_space/polyglot_live_v0.2` 的 1,010 个 OpenEvolve cell 全部返回
> 空 `result_ids`，导致 `hit_at_1` 恒为 0、质量等价集退化成整个计划网格，其 spread（中位
> 9.38×）不满足「质量等价前提下」的定义。根因与修复见
> `docs/superpowers/plans/2026-08-18-e1-composition.md` Task 1。**STARK-PRIME 的结论不受影响。**
```

- [ ] **Step 2: 检查文档中是否还有其它引用**

```bash
cd /localhome/hza214/tridb && grep -n "openevolve\|OpenEvolve\|9.38" docs/benchmark_e0_plan_space_v0.1.0.md
```

Expected: 逐条确认每处引用都在撤回范围内或已加注。

- [ ] **Step 3: 提交**

```bash
cd /localhome/hza214/tridb
git add docs/benchmark_e0_plan_space_v0.1.0.md
git commit -m "docs(e0): retract the OpenEvolve polyglot numbers, every cell returned empty"
```

---

## Task 6: `tjs_e0_open` 增加 `graph_off` / `vector_off`

**Files:**
- Modify: `src/tjs_pg/tjs_e0.c`
- Modify: `experiments/e0/tridb_schema.sql`
- Modify: `test/e0_tridb_internal_test.sql`

**Interfaces:**
- Produces: `tjs_e0_open(regclass, integer, integer, integer, text, text, vector, bigint[], integer[], boolean, text, text, boolean, boolean)`
  —— 末尾两个新参数依次为 `graph_off`、`vector_off`

**背景（关键设计约束）：**
1. **不能用零向量关断向量腿。** pgvector 的 `<=>` 计算 `1 - (a·b)/(|a||b|)`；零向量令 `|a| = 0`
   返回 **NaN**，排序未定义。因此 `vector_off = true` 时把种子查询的 `ORDER BY <vec> <op> $1`
   换成 `ORDER BY <id_col>`，traverse-first 的 top-k 换成先到先得的插入序。
2. **`graph_off` 与 `traverse_first` 不可组合。** traverse-first 的候选集由遍历生成，关掉图就
   没有候选来源。该组合必须 `ereport(ERROR)`，不得静默退化。

- [ ] **Step 1: 写失败的 SQL 测试**

在 `test/e0_tridb_internal_test.sql` 末尾追加：

```sql
-- graph_off skips the reachability constraint; candidates come from ANN only.
SELECT count(*) AS graph_off_returns_rows
FROM tjs_e0_open('e0_node'::regclass, 20, 20, 2, 'id', 'TRUE',
                 (SELECT embedding FROM e0_node ORDER BY id LIMIT 1),
                 ARRAY[1]::bigint[], ARRAY[1]::integer[], false,
                 'vector_first', 'post', true, false);

-- vector_off makes the ordering deterministic by id, never NaN.
SELECT count(*) AS vector_off_returns_rows
FROM tjs_e0_open('e0_node'::regclass, 20, 20, 2, 'id', 'TRUE',
                 (SELECT embedding FROM e0_node ORDER BY id LIMIT 1),
                 ARRAY[1]::bigint[], ARRAY[1]::integer[], false,
                 'traverse_first', 'post', false, true);

-- graph_off does not compose with traverse_first and must raise.
DO $$
BEGIN
    PERFORM * FROM tjs_e0_open('e0_node'::regclass, 20, 20, 2, 'id', 'TRUE',
                 (SELECT embedding FROM e0_node ORDER BY id LIMIT 1),
                 ARRAY[1]::bigint[], ARRAY[1]::integer[], false,
                 'traverse_first', 'post', true, false);
    RAISE EXCEPTION 'expected graph_off + traverse_first to be rejected';
EXCEPTION WHEN others THEN
    IF SQLERRM LIKE '%expected graph_off%' THEN RAISE; END IF;
    RAISE NOTICE 'graph_off + traverse_first correctly rejected: %', SQLERRM;
END $$;
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /localhome/hza214/tridb && make -C src/tjs_pg clean all && scripts/pg17_graph_test.sh
```

Expected: FAIL，函数参数个数不匹配（现有签名 12 参，测试给 14 参）。

- [ ] **Step 3: 改 C 签名与参数解析**

`src/tjs_pg/tjs_e0.c`，在 `placement = text_to_cstring(PG_GETARG_TEXT_PP(11));` 之后加入：

```c
	bool		graph_off = PG_GETARG_BOOL(12);
	bool		vector_off = PG_GETARG_BOOL(13);
```

（`graph_off` / `vector_off` 的声明放到函数开头的变量声明区，与 `shape` / `placement` 同处，
以符合本文件的 C89 风格声明位置。）

在既有的 shape/placement 校验之后追加：

```c
	if (graph_off && strcmp(shape, "traverse_first") == 0)
		ereport(ERROR, (errmsg("tjs_e0_open: graph_off does not compose with "
							   "traverse_first; traversal generates the candidates")));
```

- [ ] **Step 4: 实现 `vector_off` 的种子排序**

在 ANN 分支构造 SQL 的位置（`appendStringInfo(&sql, " ORDER BY %s %s $1 LIMIT %d", ...)`）改为：

```c
		if (vector_off)
			appendStringInfo(&sql, " ORDER BY %s LIMIT %d",
							 qident(id_col), candidate_k);
		else
			appendStringInfo(&sql, " ORDER BY %s %s $1 LIMIT %d",
							 qident(vec_col), opname, candidate_k);
```

同时 `SELECT` 列表里的距离表达式在 `vector_off` 时会被忽略，但仍需返回一个非 NULL 的 dist，
否则既有的 NULL 检查会报错。把距离列改为：

```c
		if (vector_off)
			appendStringInfo(&sql, "SELECT %s, 0.0::float8 AS dist FROM %s.%s",
							 qident(id_col), qident(nspname), qident(relname));
		else
			appendStringInfo(&sql, "SELECT %s, %s %s $1 AS dist FROM %s.%s",
							 qident(id_col), qident(vec_col), opname,
							 qident(nspname), qident(relname));
```

注意 `vector_off` 时 `$1` 不再出现在 SQL 中，但 `SPI_execute_with_args` 仍传 1 个参数——
PostgreSQL 允许未被引用的参数，无需改调用。

- [ ] **Step 5: 实现 `graph_off` 跳过遍历**

在 ANN 分支中 `if (pending > 0)` 那一段外层包一层：

```c
		if (graph_off)
		{
			for (i = 0; i < ncand; i++)
				matched[i] = passes[i];
			tjs_e0_termination = "graph_disabled";
			tjs_graph_examined = 0;
			tjs_graph_censored = false;
		}
		else if (pending > 0)
		{
			/* ... 既有的遍历确认逻辑，原样保留 ... */
		}
		else
			tjs_e0_termination = "predicate_empty";
```

- [ ] **Step 6: 实现 traverse-first 分支的 `vector_off`**

traverse-first 分支用 `Topk` 按距离维护 top-k。`vector_off` 时改为先到先得：在把 reached id
放入 topk 之前，若 `vector_off` 为真，则用递增序号代替距离作为排序键：

```c
		double sort_key = vector_off ? (double) emitted_seq++ : dist;
```

其中 `emitted_seq` 是本次调用内初始化为 0 的 `int64`。这样 topk 保留最先到达的 k 个，顺序确定。

- [ ] **Step 7: 更新 SQL 函数声明**

`experiments/e0/tridb_schema.sql` 中：

```sql
CREATE OR REPLACE FUNCTION public.tjs_e0_open(
    regclass, integer, integer, integer, text, text, vector,
    bigint[], integer[], boolean, text, text, boolean, boolean
) RETURNS SETOF bigint
AS '@TJS_LIB@', 'tjs_e0_open_pg'
LANGUAGE C VOLATILE;
```

旧的 12 参声明必须 `DROP FUNCTION IF EXISTS` 掉，否则会与新声明并存造成重载歧义。在
`CREATE OR REPLACE` 之前加：

```sql
DROP FUNCTION IF EXISTS public.tjs_e0_open(
    regclass, integer, integer, integer, text, text, vector,
    bigint[], integer[], boolean, text, text);
```

- [ ] **Step 8: 重建并运行 SQL 套件确认通过**

```bash
cd /localhome/hza214/tridb && make -C src/tjs_pg clean all && scripts/pg17_graph_test.sh
```

Expected: PASS，包括新加的三条断言。

- [ ] **Step 9: 更新 Python 后端传参**

`experiments/e0/plan_spread/tridb_backend.py` 的 `execute()` 中，SQL 占位符串加两个 `%s`，
参数元组末尾加 `plan_graph_off, plan_vector_off`。默认值来自 `PlanSpec`——若 `plan` 没有这两个
属性（E0 的 `PlanSpec` 没有），用 `getattr(plan, "graph_off", False)` /
`getattr(plan, "vector_off", False)`，以保证 E0 既有调用路径不变：

```python
                    plan.shape,
                    plan.predicate_placement,
                    getattr(plan, "graph_off", False),
                    getattr(plan, "vector_off", False),
                ),
```

- [ ] **Step 10: 跑 E0 回归确认既有结果不变**

```bash
cd /localhome/hza214/tridb
.venv/bin/python -m experiments.e0.plan_spread.runner --backend tridb_live \
  --config configs/e0/plan_space_v0.3.yaml \
  --output-dir results/e1/tridb_regression_check \
  --dataset openevolve --query-limit 2 --plan-limit 6 --repetitions 1
```

Expected: 0 errors。把该目录的 `result_ids` 与 `results/e0/plan_space/tridb_live_v0.3` 中对应
cell 比对，必须逐条相同——两个新参数默认 false 时不得改变任何既有行为。

- [ ] **Step 11: 提交**

```bash
cd /localhome/hza214/tridb
git add src/tjs_pg/tjs_e0.c experiments/e0/tridb_schema.sql test/e0_tridb_internal_test.sql experiments/e0/plan_spread/tridb_backend.py
git commit -m "feat(tjs): add graph_off and vector_off to the E0 benchmark operator

Ablation needs to disable one modality leg at a time. The vector leg cannot be
disabled with a zero vector because pgvector's <=> returns NaN at zero norm, so
vector_off switches the seed ordering to the id column instead. graph_off is
rejected for traverse_first, where traversal is the candidate generator."
```

---

## Task 7: 消融变体定义

**Files:**
- Create: `experiments/e1/__init__.py`
- Create: `experiments/e1/variants.py`
- Test: `tests/test_e1_variants.py`

**Interfaces:**
- Produces: `VARIANTS: tuple[Variant, ...]`，七个元素
- Produces: `Variant` 冻结 dataclass，字段 `name`、`shape`、`use_predicate: bool`、`vector_off: bool`、`graph_off: bool`
- Produces: `variant_plan(variant: Variant, k: int, hops: int) -> AblationPlanSpec`
- Produces: `AblationPlanSpec`，继承 `PlanSpec` 的字段并附加 `graph_off` / `vector_off` / `variant`

- [ ] **Step 1: 写失败测试**

`tests/test_e1_variants.py`：

```python
from __future__ import annotations

import pytest

from experiments.e1.variants import VARIANTS, variant_plan


def test_seven_variants_cover_every_modality_combination():
    names = {v.name for v in VARIANTS}
    assert names == {"V", "G", "R", "V+R", "V+G", "G+R", "V+G+R"}


def test_graph_off_never_uses_traverse_first():
    for variant in VARIANTS:
        if variant.graph_off:
            assert variant.shape != "traverse_first", variant.name


def test_full_variant_uses_all_three_legs():
    full = next(v for v in VARIANTS if v.name == "V+G+R")
    assert full.use_predicate and not full.vector_off and not full.graph_off


def test_variant_plan_carries_flags_and_stable_id():
    variant = next(v for v in VARIANTS if v.name == "V+R")
    plan = variant_plan(variant, k=20, hops=2)
    assert plan.graph_off is True
    assert plan.vector_off is False
    assert plan.shape == "filter_first"
    assert plan.plan_id.startswith("V+R-")
    assert variant_plan(variant, k=20, hops=2).plan_id == plan.plan_id
    assert variant_plan(variant, k=50, hops=2).plan_id != plan.plan_id
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_variants.py -v
```

Expected: FAIL，`ModuleNotFoundError: experiments.e1.variants`。

- [ ] **Step 3: 实现**

`experiments/e1/__init__.py`：空文件。

`experiments/e1/variants.py`：

```python
"""The seven modality-ablation variants and their execution parameters.

Shape is NOT a free variable. traverse-first generates its candidates by
traversal, so a graph-off variant cannot use it; each variant's shape is the
one shape capable of expressing that modality combination.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from experiments.e0.plan_spread.model import canonical_json


@dataclass(frozen=True)
class Variant:
    name: str
    shape: str
    use_predicate: bool
    vector_off: bool
    graph_off: bool
    predicate_placement: str


VARIANTS: tuple[Variant, ...] = (
    Variant("V", "vector_first", False, False, True, "post"),
    Variant("G", "traverse_first", False, True, False, "post"),
    Variant("R", "filter_first", True, True, True, "pre"),
    Variant("V+R", "filter_first", True, False, True, "pre"),
    Variant("V+G", "traverse_first", False, False, False, "post"),
    Variant("G+R", "traverse_first", True, True, False, "during"),
    Variant("V+G+R", "traverse_first", True, False, False, "during"),
)


@dataclass(frozen=True)
class AblationPlanSpec:
    variant: str
    shape: str
    k: int
    hops: int
    predicate_placement: str
    graph_off: bool
    vector_off: bool
    use_predicate: bool
    is_default: bool = False

    @property
    def plan_id(self) -> str:
        payload = {
            "graph_off": self.graph_off,
            "hops": self.hops,
            "k": self.k,
            "predicate_placement": self.predicate_placement,
            "shape": self.shape,
            "use_predicate": self.use_predicate,
            "variant": self.variant,
            "vector_off": self.vector_off,
        }
        digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:12]
        return f"{self.variant}-{digest}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "variant": self.variant,
            "shape": self.shape,
            "k": self.k,
            "hops": self.hops,
            "predicate_placement": self.predicate_placement,
            "graph_off": self.graph_off,
            "vector_off": self.vector_off,
            "use_predicate": self.use_predicate,
            "is_default": self.is_default,
        }


def variant_plan(variant: Variant, k: int, hops: int) -> AblationPlanSpec:
    return AblationPlanSpec(
        variant=variant.name,
        shape=variant.shape,
        k=k,
        hops=hops,
        predicate_placement=variant.predicate_placement,
        graph_off=variant.graph_off,
        vector_off=variant.vector_off,
        use_predicate=variant.use_predicate,
    )
```

- [ ] **Step 4: 运行确认通过**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_variants.py -v
```

Expected: 4 passed。

- [ ] **Step 5: 提交**

```bash
cd /localhome/hza214/tridb
git add experiments/e1/__init__.py experiments/e1/variants.py tests/test_e1_variants.py
git commit -m "feat(e1): define the seven modality-ablation variants"
```

---

## Task 8: 从 E0 结果挑选 E1 的计划集

**Files:**
- Create: `experiments/e1/plan_selection.py`
- Test: `tests/test_e1_plan_selection.py`

**Interfaces:**
- Consumes: `experiments.e0.plan_spread.model.PlanSpec`
- Produces: `best_equivalent_plan(rows) -> dict | None`，按冻结的质量等价规则挑最快计划
- Produces: `select_plans(tridb_rows, poly_rows, dataset, query_id) -> list[PlanSpec]`，
  返回三 shape 代表 + TriDB 最优 + Polyglot 最优 + 默认计划，去重后 4–6 个

- [ ] **Step 1: 写失败测试**

`tests/test_e1_plan_selection.py`：

```python
from __future__ import annotations

from experiments.e1.plan_selection import best_equivalent_plan, select_plans


def _row(plan_id, shape, k, hops, p50, hit1, mrr, is_default=False):
    return {
        "query_id": "q", "dataset": "d", "plan_id": plan_id, "shape": shape,
        "k": k, "hops": hops, "predicate_placement": "post",
        "is_default": is_default, "p50_latency_ms": p50,
        "hit_at_1": hit1, "mrr": mrr,
    }


def test_best_equivalent_plan_ignores_a_faster_but_worse_plan():
    rows = [
        _row("slow", "traverse_first", 20, 2, 10.0, 1.0, 1.0),
        _row("fast", "vector_first", 5, 1, 1.0, 0.0, 0.0),
    ]
    assert best_equivalent_plan(rows)["plan_id"] == "slow"


def test_best_equivalent_plan_takes_the_fastest_within_tolerance():
    rows = [
        _row("slow", "traverse_first", 20, 2, 10.0, 1.0, 1.0),
        _row("quick", "filter_first", 10, 2, 3.0, 1.0, 0.99),
    ]
    assert best_equivalent_plan(rows)["plan_id"] == "quick"


def test_select_plans_includes_both_systems_optima_and_the_default():
    tridb = [
        _row("t-trav", "traverse_first", 20, 2, 1.0, 1.0, 1.0),
        _row("t-vec", "vector_first", 5, 1, 9.0, 1.0, 1.0),
        _row("t-filt", "filter_first", 10, 2, 8.0, 1.0, 1.0),
        _row("t-def", "vector_first", 8, 2, 7.0, 0.0, 0.0, is_default=True),
    ]
    poly = [
        _row("t-trav", "traverse_first", 20, 2, 5.0, 1.0, 1.0),
        _row("t-vec", "vector_first", 5, 1, 2.0, 1.0, 1.0),
        _row("t-filt", "filter_first", 10, 2, 8.0, 1.0, 1.0),
        _row("t-def", "vector_first", 8, 2, 7.0, 0.0, 0.0, is_default=True),
    ]
    plans = select_plans(tridb, poly, "d", "q")
    ids = {p.plan_id for p in plans}
    shapes = {p.shape for p in plans}
    assert shapes == {"traverse_first", "vector_first", "filter_first"}
    assert any(p.is_default for p in plans)
    assert 4 <= len(ids) <= 6
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_plan_selection.py -v
```

Expected: FAIL，`ModuleNotFoundError`。

- [ ] **Step 3: 实现**

`experiments/e1/plan_selection.py`：

```python
"""Pick the E1 plan set for each query from the completed E0 runs.

E1 trades plan-space breadth for repetition count: E0 already mapped the full
grid, so E1 keeps five plans per query and runs each 31 times so p95 is
reportable.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.e0.plan_spread.model import PlanSpec

EPS_HIT = 0.02
EPS_MRR = 0.02


def best_equivalent_plan(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Fastest plan among those tied for best quality, per the frozen rule."""
    if not rows:
        return None
    best_hit = max(row["hit_at_1"] for row in rows)
    tied = [row for row in rows if row["hit_at_1"] >= best_hit - EPS_HIT]
    best_mrr = max(row["mrr"] for row in tied)
    tied = [row for row in tied if row["mrr"] >= best_mrr - EPS_MRR]
    return min(tied, key=lambda row: row["p50_latency_ms"])


def _to_plan(row: dict[str, Any]) -> PlanSpec:
    return PlanSpec(
        shape=row["shape"],
        k=int(row["k"]),
        hops=int(row["hops"]),
        predicate_placement=row["predicate_placement"],
        is_default=bool(row.get("is_default", False)),
    )


def select_plans(
    tridb_rows: list[dict[str, Any]],
    poly_rows: list[dict[str, Any]],
    dataset: str,
    query_id: str,
) -> list[PlanSpec]:
    tridb = [r for r in tridb_rows if r["dataset"] == dataset
             and r["query_id"] == query_id]
    poly = [r for r in poly_rows if r["dataset"] == dataset
            and r["query_id"] == query_id]
    chosen: dict[str, PlanSpec] = {}

    # One representative per shape: the fastest quality-equivalent plan of that
    # shape on TriDB, so no shape is represented by an accidentally bad point.
    by_shape: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tridb:
        if not row.get("is_default"):
            by_shape[row["shape"]].append(row)
    for shape in sorted(by_shape):
        pick = best_equivalent_plan(by_shape[shape])
        if pick is not None:
            plan = _to_plan(pick)
            chosen[plan.plan_id] = plan

    for rows in (tridb, poly):
        grid = [r for r in rows if not r.get("is_default")]
        pick = best_equivalent_plan(grid)
        if pick is not None:
            plan = _to_plan(pick)
            chosen.setdefault(plan.plan_id, plan)

    default = next((r for r in tridb if r.get("is_default")), None)
    if default is not None:
        plan = _to_plan(default)
        chosen[plan.plan_id] = plan

    return [chosen[key] for key in sorted(chosen)]


def load_plan_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Reduce an E0 observations file to one row per (dataset, query, plan)."""
    latencies: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line in (run_dir / "observations.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        key = (row["dataset"], row["query_id"], row["plan_id"])
        latencies[key].append(float(row["latency_ms"]))
        meta[key] = row
    out = []
    for key, values in latencies.items():
        row = meta[key]
        out.append(
            {
                "dataset": key[0],
                "query_id": key[1],
                "plan_id": key[2],
                "shape": row["shape"],
                "k": int(row["k"]),
                "hops": int(row["hops"]),
                "predicate_placement": row["predicate_placement"],
                "is_default": bool(row["is_default"]),
                "p50_latency_ms": st.median(values),
                "hit_at_1": float(row["quality"]["hit_at_1"]),
                "mrr": float(row["quality"]["mrr"]),
            }
        )
    return out
```

- [ ] **Step 4: 运行确认通过**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_plan_selection.py -v
```

Expected: 3 passed。

- [ ] **Step 5: 在真实 E0 数据上核对规模**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -c "
from pathlib import Path
from experiments.e1.plan_selection import load_plan_rows, select_plans
t = load_plan_rows(Path('results/e0/plan_space/tridb_live_v0.3'))
p = load_plan_rows(Path('results/e0/plan_space/polyglot_live_v0.2'))
qs = sorted({(r['dataset'], r['query_id']) for r in t})
sizes = [len(select_plans(t, p, d, q)) for d, q in qs]
print('queries', len(qs), 'plans per query min/median/max',
      min(sizes), sorted(sizes)[len(sizes)//2], max(sizes))
print('total cells', sum(sizes))
"
```

Expected: 50 条 query，每 query 4–6 个计划，总 cell 数约 200–300。若某 query 只有 3 个计划，
说明该 query 三个 shape 中有 shape 全无质量等价计划——记录下来，报告中单列。

- [ ] **Step 6: 提交**

```bash
cd /localhome/hza214/tridb
git add experiments/e1/plan_selection.py tests/test_e1_plan_selection.py
git commit -m "feat(e1): select the per-query plan set from the completed E0 runs"
```

---

## Task 9: 冻结 E1 配置

**Files:**
- Create: `configs/e1/composition_v0.1.yaml`
- Test: `tests/test_e1_interleave.py`

**Interfaces:**
- Produces: `configs/e1/composition_v0.1.yaml`，schema_version `e1-composition-v0.1.0`

- [ ] **Step 1: 写失败测试**

`tests/test_e1_interleave.py`：

```python
from __future__ import annotations

from pathlib import Path

import yaml


def test_e1_config_is_frozen_with_the_declared_contract():
    config = yaml.safe_load(
        Path("configs/e1/composition_v0.1.yaml").read_text(encoding="utf-8")
    )
    assert config["schema_version"] == "e1-composition-v0.1.0"
    assert config["repetitions"] == 31
    assert config["quality"]["eps_hit"] == 0.02
    assert config["quality"]["eps_mrr"] == 0.02
    assert config["backends"] == ["tridb_live", "polyglot_live"]
    assert config["falsification"]["composition_ratio_median_below"] == 1.2
    assert config["falsification"]["ablation_quality_fraction_above"] == 0.95
    assert config["source_runs"]["tridb"] == "results/e0/plan_space/tridb_live_v0.3"
    assert config["source_runs"]["polyglot"] == (
        "results/e0/plan_space/polyglot_live_v0.2"
    )
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_interleave.py -v
```

Expected: FAIL，文件不存在。

- [ ] **Step 3: 写配置**

`configs/e1/composition_v0.1.yaml`：

```yaml
# E1 contract. Frozen before measurement. Do not edit after the first run.
schema_version: e1-composition-v0.1.0

# The two systems compared on identical queries. Polyglot-Tuned is
# Milvus 2.4.5 + Neo4j 5.20 + pgvector pg16, co-located, connection-pooled.
backends: [tridb_live, polyglot_live]

# E0 supplies the plan set; E1 does not re-enumerate the grid.
source_runs:
  tridb: results/e0/plan_space/tridb_live_v0.3
  polyglot: results/e0/plan_space/polyglot_live_v0.2

# Dataset specs are inherited verbatim so connection details stay in one place.
inherit_datasets_from: configs/e0/plan_space_v0.3.yaml
datasets: [stark_prime, openevolve]

repetitions: 31
warmups: 1
timeout_seconds: 120
top_n: 20

quality:
  eps_hit: 0.02
  eps_mrr: 0.02
  primary: [hit_at_1, mrr]
  recorded_only: [hit_at_5, recall_at_20]

# n=31 supports p50 and p95. p99 is NOT reportable at this sample size and must
# not appear in the report.
percentiles: [50, 95]

falsification:
  # Sub-proposition C fails if the equal-quality latency ratio median is below this.
  composition_ratio_median_below: 1.2
  # Sub-proposition N fails if some two-modality variant reaches this fraction of
  # the three-modality Hit@1 on at least this share of queries without the
  # candidate set blowing up.
  ablation_quality_fraction_above: 0.95
  ablation_query_share_above: 0.95

interleave:
  # Deterministic per-cell backend order, seeded so a rerun reproduces it.
  seed: 20260818
```

- [ ] **Step 4: 运行确认通过**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_interleave.py -v
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
cd /localhome/hza214/tridb
git add configs/e1/composition_v0.1.yaml tests/test_e1_interleave.py
git commit -m "feat(e1): freeze the composition contract before measuring"
```

---

## Task 10: 交错 A/B runner

**Files:**
- Create: `experiments/e1/interleaved_runner.py`
- Test: `tests/test_e1_interleave.py`

**Interfaces:**
- Consumes: `experiments.e1.plan_selection.select_plans` / `load_plan_rows`、
  `experiments.e0.plan_spread.backend.build_backend`
- Produces: `backend_order(seed, dataset, query_id, plan_id, repetition) -> tuple[str, str]`
- Produces: `run(config_path, output_dir, *, datasets, query_limit, repetitions) -> dict`
- Produces: `results/e1/<run>/observations.jsonl`，schema_version `e1-observation-v0.1.0`

- [ ] **Step 1: 写失败测试（追加到 `tests/test_e1_interleave.py`）**

```python
def test_backend_order_is_deterministic_and_balanced():
    from experiments.e1.interleaved_runner import backend_order

    first = backend_order(20260818, "d", "q", "p", 0)
    assert first == backend_order(20260818, "d", "q", "p", 0)
    assert set(first) == {"tridb_live", "polyglot_live"}

    orders = [
        backend_order(20260818, "d", "q", "p", rep)[0] for rep in range(200)
    ]
    tridb_first = orders.count("tridb_live")
    assert 70 <= tridb_first <= 130, tridb_first


def test_observation_key_includes_the_backend():
    from experiments.e1.interleaved_runner import observation_key

    a = observation_key("tridb_live", "d", "q", "p", 0)
    b = observation_key("polyglot_live", "d", "q", "p", 0)
    assert a != b
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_interleave.py -v
```

Expected: FAIL，`ModuleNotFoundError: experiments.e1.interleaved_runner`。

- [ ] **Step 3: 实现**

`experiments/e1/interleaved_runner.py`：

```python
"""Interleaved A/B runner for TriDB vs Polyglot-Tuned.

The two completed E0 runs were measured 13 hours apart, so machine drift could
not be ruled out. Here both backends execute inside one process and, for every
(query, plan, repetition) cell, in an order derived from a seeded hash. Drift
therefore becomes noise shared by both systems rather than a bias favouring the
one that ran on a quieter machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from experiments.e0.plan_spread.backend import build_backend
from experiments.e0.plan_spread.config import load_config
from experiments.e0.plan_spread.model import QuerySpec, canonical_json
from experiments.e1.plan_selection import load_plan_rows, select_plans
from tools.e0.common import artifact_record, environment_record, sha256_file, write_json

SCHEMA_VERSION = "e1-observation-v0.1.0"
BACKENDS = ("tridb_live", "polyglot_live")


def backend_order(
    seed: int, dataset: str, query_id: str, plan_id: str, repetition: int
) -> tuple[str, str]:
    payload = f"{seed}|{dataset}|{query_id}|{plan_id}|{repetition}"
    digest = hashlib.sha256(payload.encode()).digest()
    return BACKENDS if digest[0] % 2 == 0 else tuple(reversed(BACKENDS))


def observation_key(
    backend: str, dataset: str, query_id: str, plan_id: str, repetition: int
) -> tuple[str, str, str, str, int]:
    return backend, dataset, query_id, plan_id, repetition


def completed_keys(path: Path, config_sha256: str) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    done = set()
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("config_sha256") != config_sha256:
            raise RuntimeError(
                f"{path}:{lineno}: config hash differs; use a new output directory"
            )
        done.add(
            observation_key(
                row["backend"], row["dataset"], row["query_id"],
                row["plan_id"], int(row["repetition"]),
            )
        )
    return done


def _append(handle: Any, row: dict[str, Any]) -> None:
    handle.write(canonical_json(row) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _query_record(query: QuerySpec) -> dict[str, Any]:
    return {
        "query_id": query.query_id,
        "query_hop_limit": query.hop_limit,
        "template": query.template,
        "annotation_status": query.annotation_status,
        "answer_count": len(query.answer_ids),
    }


def run(
    config_path: Path,
    output_dir: Path,
    *,
    datasets: list[str] | None = None,
    query_limit: int | None = None,
    repetitions: int | None = None,
) -> dict[str, Any]:
    e1 = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if e1.get("schema_version") != "e1-composition-v0.1.0":
        raise ValueError("unsupported E1 schema_version")
    base = load_config(Path(e1["inherit_datasets_from"]))
    config_sha256 = sha256_file(config_path)
    selected = datasets or list(e1["datasets"])
    repetitions = repetitions or int(e1["repetitions"])
    top_n = int(e1["top_n"])
    seed = int(e1["interleave"]["seed"])
    timeout_ms = float(e1["timeout_seconds"]) * 1000.0

    tridb_rows = load_plan_rows(Path(e1["source_runs"]["tridb"]))
    poly_rows = load_plan_rows(Path(e1["source_runs"]["polyglot"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "observations.jsonl"
    done = completed_keys(raw_path, config_sha256)
    counts = {"written": 0, "resumed": len(done), "errors": 0}
    started = time.time()
    inputs = [artifact_record(config_path)]

    with raw_path.open("a", encoding="utf-8") as handle:
        for dataset_name in selected:
            spec = base["datasets"][dataset_name]
            for field in ("nodes", "edges", "embeddings",
                          "query_embeddings", "queries"):
                inputs.append(artifact_record(Path(spec[field])))
            engines = {
                name: build_backend(name, dataset_name, spec) for name in BACKENDS
            }
            queries = engines["tridb_live"].load_queries(Path(spec["queries"]))
            if query_limit is not None:
                queries = queries[:query_limit]

            for query in queries:
                plans = select_plans(
                    tridb_rows, poly_rows, dataset_name, query.query_id
                )
                if not plans:
                    raise RuntimeError(
                        f"{dataset_name}/{query.query_id}: no plans selected"
                    )
                hops = {plan.hops for plan in plans}
                for engine in engines.values():
                    engine.prepare_query(query, hops)
                    for _ in range(int(e1["warmups"])):
                        engine.execute(query, plans[0], top_n=top_n)

                for plan in plans:
                    for repetition in range(repetitions):
                        order = backend_order(
                            seed, dataset_name, query.query_id,
                            plan.plan_id, repetition,
                        )
                        for backend_name in order:
                            key = observation_key(
                                backend_name, dataset_name, query.query_id,
                                plan.plan_id, repetition,
                            )
                            if key in done:
                                continue
                            try:
                                result = engines[backend_name].execute(
                                    query, plan, top_n=top_n
                                )
                                if result["latency_ms"] > timeout_ms:
                                    result["status"] = "timeout"
                            except Exception as exc:
                                counts["errors"] += 1
                                raise RuntimeError(
                                    f"{backend_name}/{dataset_name}/"
                                    f"{query.query_id}/{plan.plan_id}: {exc}"
                                ) from exc
                            row = {
                                "schema_version": SCHEMA_VERSION,
                                "config_sha256": config_sha256,
                                "dataset": dataset_name,
                                "backend_order": list(order),
                                **_query_record(query),
                                **plan.as_dict(),
                                "repetition": repetition,
                                **result,
                            }
                            _append(handle, row)
                            counts["written"] += 1
                            done.add(key)
            for engine in engines.values():
                close = getattr(engine, "close", None)
                if close is not None:
                    close()

    manifest = {
        "schema_version": "e1-run-v0.1.0",
        "environment": environment_record(),
        "backends": list(BACKENDS),
        "valid_for_system_latency_claims": True,
        "config": artifact_record(config_path),
        "inputs": inputs,
        "output": artifact_record(raw_path),
        "datasets": selected,
        "repetitions": repetitions,
        "counts": counts,
        "seconds": round(time.time() - started, 3),
        "complete": counts["errors"] == 0,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/e1/composition_v0.1.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args(argv)
    manifest = run(
        args.config,
        args.output_dir,
        datasets=args.datasets,
        query_limit=args.query_limit,
        repetitions=args.repetitions,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_interleave.py -v
```

Expected: 3 passed。

- [ ] **Step 5: 冒烟跑**

```bash
cd /localhome/hza214/tridb
.venv/bin/python -m experiments.e1.interleaved_runner \
  --output-dir results/e1/h2h_smoke \
  --dataset openevolve --query-limit 2 --repetitions 2
```

Expected: `{"errors": 0, ...}`，`observations.jsonl` 中每个 cell 有两个 backend 各一行。

- [ ] **Step 6: 提交**

```bash
cd /localhome/hza214/tridb
git add experiments/e1/interleaved_runner.py tests/test_e1_interleave.py
git commit -m "feat(e1): interleaved A/B runner so machine drift is shared noise"
```

---

## Task 11: 消融 runner

**Files:**
- Create: `experiments/e1/ablation_runner.py`
- Test: `tests/test_e1_variants.py`

**Interfaces:**
- Consumes: `experiments.e1.variants.VARIANTS` / `variant_plan`、
  `experiments.e1.plan_selection.best_equivalent_plan` / `load_plan_rows`
- Produces: `run(config_path, output_dir, *, datasets, query_limit, repetitions) -> dict`
- Produces: 每条 observation 除常规字段外，附加 `variant`、`candidate_contains_answer`（bool）、
  `candidate_cardinality`（int）

**关键：** 「能力上界」`candidate_contains_answer` 必须在**排序之前**判定——它回答的是「该模态
组合的候选集里到底有没有答案」，与排序无关。实现方式是把 `top_n` 临时提到候选集大小，
取回全部候选后判断是否含 answer；这次调用不计入延迟。

- [ ] **Step 1: 写失败测试（追加到 `tests/test_e1_variants.py`）**

```python
def test_capability_upper_bound_is_independent_of_ranking():
    from experiments.e1.ablation_runner import capability_upper_bound

    assert capability_upper_bound(["x", "a", "y"], ("a",)) is True
    assert capability_upper_bound(["x", "y"], ("a",)) is False
    # order must not matter
    assert capability_upper_bound(["a", "x"], ("a",)) is True
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_variants.py::test_capability_upper_bound_is_independent_of_ranking -v
```

Expected: FAIL，`ModuleNotFoundError`。

- [ ] **Step 3: 实现**

`experiments/e1/ablation_runner.py`：

```python
"""Modality ablation inside TriDB.

Each variant disables one or two of the three legs. Three numbers are recorded
per variant, and all three are needed to read the result honestly:

  1. realized quality   -- Hit@1 / MRR / Recall@20 under a deterministic order
  2. capability upper   -- does the candidate set contain the answer AT ALL,
                           independent of ranking; a variant whose candidate set
                           lacks the answer cannot be rescued by a better ranker
  3. candidate size     -- what the variant costs when it does not collapse in
                           quality; this is the "cost explodes" half of N
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from experiments.e0.plan_spread.backend import build_backend
from experiments.e0.plan_spread.config import load_config
from experiments.e0.plan_spread.model import QuerySpec, canonical_json
from experiments.e1.plan_selection import best_equivalent_plan, load_plan_rows
from experiments.e1.variants import VARIANTS, variant_plan
from tools.e0.common import artifact_record, environment_record, sha256_file, write_json

SCHEMA_VERSION = "e1-ablation-v0.1.0"
UPPER_BOUND_TOP_N = 100_000


def capability_upper_bound(
    candidate_ids: list[Any], answer_ids: tuple[Any, ...]
) -> bool:
    return bool(set(candidate_ids) & set(answer_ids))


def _append(handle: Any, row: dict[str, Any]) -> None:
    handle.write(canonical_json(row) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _query_record(query: QuerySpec) -> dict[str, Any]:
    return {
        "query_id": query.query_id,
        "query_hop_limit": query.hop_limit,
        "template": query.template,
        "annotation_status": query.annotation_status,
        "answer_count": len(query.answer_ids),
    }


class _PredicateMask:
    """Wraps a QuerySpec so a variant can drop the relational predicate."""

    @staticmethod
    def without_predicate(query: QuerySpec) -> QuerySpec:
        from dataclasses import replace

        return replace(query, structured_predicate={}, target_entity_type="")


def run(
    config_path: Path,
    output_dir: Path,
    *,
    datasets: list[str] | None = None,
    query_limit: int | None = None,
    repetitions: int | None = None,
) -> dict[str, Any]:
    e1 = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if e1.get("schema_version") != "e1-composition-v0.1.0":
        raise ValueError("unsupported E1 schema_version")
    base = load_config(Path(e1["inherit_datasets_from"]))
    config_sha256 = sha256_file(config_path)
    selected = datasets or list(e1["datasets"])
    repetitions = repetitions or int(e1["repetitions"])
    top_n = int(e1["top_n"])
    tridb_rows = load_plan_rows(Path(e1["source_runs"]["tridb"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "observations.jsonl"
    counts = {"written": 0, "errors": 0}
    started = time.time()
    inputs = [artifact_record(config_path)]

    with raw_path.open("a", encoding="utf-8") as handle:
        for dataset_name in selected:
            spec = base["datasets"][dataset_name]
            for field in ("nodes", "edges", "embeddings",
                          "query_embeddings", "queries"):
                inputs.append(artifact_record(Path(spec[field])))
            engine = build_backend("tridb_live", dataset_name, spec)
            queries = engine.load_queries(Path(spec["queries"]))
            if query_limit is not None:
                queries = queries[:query_limit]

            for query in queries:
                grid = [
                    row for row in tridb_rows
                    if row["dataset"] == dataset_name
                    and row["query_id"] == query.query_id
                    and not row["is_default"]
                ]
                anchor_plan = best_equivalent_plan(grid)
                if anchor_plan is None:
                    raise RuntimeError(
                        f"{dataset_name}/{query.query_id}: no anchor plan"
                    )
                k = int(anchor_plan["k"])
                hops = int(anchor_plan["hops"])
                engine.prepare_query(query, {hops})

                for variant in VARIANTS:
                    plan = variant_plan(variant, k, hops)
                    masked = (
                        query if variant.use_predicate
                        else _PredicateMask.without_predicate(query)
                    )
                    # tridb_backend caches the compiled predicate by query_id, and
                    # the masked query keeps the same id. Without this the
                    # predicate-off variants would silently reuse the predicate.
                    engine._predicate_cache.pop(query.query_id, None)
                    engine._predicate_count_cache.pop(query.query_id, None)
                    engine.prepare_query(masked, {hops})

                    # Capability upper bound: the candidate set before ranking.
                    probe = engine.execute(masked, plan, top_n=UPPER_BOUND_TOP_N)
                    contains = capability_upper_bound(
                        probe["result_ids"], query.answer_ids
                    )
                    cardinality = len(probe["result_ids"])

                    for repetition in range(repetitions):
                        # tridb_backend already scores against query.answer_ids,
                        # which masking does not change.
                        result = engine.execute(masked, plan, top_n=top_n)
                        row = {
                            "schema_version": SCHEMA_VERSION,
                            "config_sha256": config_sha256,
                            "dataset": dataset_name,
                            **_query_record(query),
                            **plan.as_dict(),
                            "anchor_plan_id": anchor_plan["plan_id"],
                            "repetition": repetition,
                            "candidate_contains_answer": contains,
                            "candidate_cardinality": cardinality,
                            **result,
                        }
                        _append(handle, row)
                        counts["written"] += 1
            close = getattr(engine, "close", None)
            if close is not None:
                close()

    manifest = {
        "schema_version": "e1-ablation-run-v0.1.0",
        "environment": environment_record(),
        "backend": "tridb_live",
        "valid_for_system_latency_claims": True,
        "config": artifact_record(config_path),
        "inputs": inputs,
        "output": artifact_record(raw_path),
        "datasets": selected,
        "repetitions": repetitions,
        "variants": [v.name for v in VARIANTS],
        "counts": counts,
        "seconds": round(time.time() - started, 3),
        "complete": counts["errors"] == 0,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/e1/composition_v0.1.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args(argv)
    manifest = run(
        args.config,
        args.output_dir,
        datasets=args.datasets,
        query_limit=args.query_limit,
        repetitions=args.repetitions,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_variants.py -v
```

Expected: 5 passed。

- [ ] **Step 5: 冒烟跑**

```bash
cd /localhome/hza214/tridb
.venv/bin/python -m experiments.e1.ablation_runner \
  --output-dir results/e1/ablation_smoke \
  --dataset openevolve --query-limit 2 --repetitions 2
```

Expected: `{"errors": 0, "written": 28}`（2 query × 7 变体 × 2 rep）。检查 `V+G+R` 的
`candidate_contains_answer` 为 true 而 `V` 为 false——若 `V` 也为 true，说明该 query 太小，
消融看不出差别，记录下来。

- [ ] **Step 6: 提交**

```bash
cd /localhome/hza214/tridb
git add experiments/e1/ablation_runner.py tests/test_e1_variants.py
git commit -m "feat(e1): ablation runner recording realized quality, capability bound, and cost"
```

---

## Task 12: H2H 分析器与代价分解

**Files:**
- Create: `experiments/e1/analyze_h2h.py`
- Test: `tests/test_e1_analyze.py`

**Interfaces:**
- Produces: `equal_quality_ratio(tridb_rows, poly_rows) -> list[dict]`，每 query 一条，含
  `ratio_p50`、`ratio_p95`、两侧质量
- Produces: `cost_decomposition(poly_rows, tridb_rows) -> dict`，含
  `round_trip_ms`、`extra_work_ms`、`serialization_ms`、`unexplained_ms` 及各自占比
- Produces: `analyze(raw_path, config_path, output_dir) -> dict`，写 `summary.json`、
  `metrics_query.csv`、`figures/`

- [ ] **Step 1: 写失败测试**

`tests/test_e1_analyze.py`：

```python
from __future__ import annotations

from experiments.e1.analyze_h2h import cost_decomposition, equal_quality_ratio


def _obs(backend, plan_id, latency, hit1, mrr, **extra):
    row = {
        "backend": backend, "dataset": "d", "query_id": "q", "plan_id": plan_id,
        "status": "ok", "latency_ms": latency,
        "quality": {"hit_at_1": hit1, "mrr": mrr,
                    "hit_at_5": hit1, "recall_at_20": hit1},
        "round_trips": 0, "rows_shipped": 0, "bytes_shipped": 0,
        "serialization_ms": 0.0,
        "intermediate_cardinality": {"candidates": 0},
    }
    row.update(extra)
    return row


def test_equal_quality_ratio_uses_each_system_best_quality_equivalent_plan():
    tridb = [_obs("tridb_live", "fast_bad", 1.0, 0.0, 0.0),
             _obs("tridb_live", "good", 10.0, 1.0, 1.0)]
    poly = [_obs("polyglot_live", "fast_bad", 1.0, 0.0, 0.0),
            _obs("polyglot_live", "good", 20.0, 1.0, 1.0)]
    rows = equal_quality_ratio(tridb, poly)
    assert len(rows) == 1
    assert rows[0]["ratio_p50"] == 2.0
    assert rows[0]["tridb_hit_at_1"] == 1.0
    assert rows[0]["polyglot_hit_at_1"] == 1.0


def test_cost_decomposition_splits_the_gap_and_reports_the_remainder():
    tridb = [_obs("tridb_live", "p", 10.0, 1.0, 1.0,
                  intermediate_cardinality={"candidates": 10})]
    poly = [_obs("polyglot_live", "p", 16.0, 1.0, 1.0,
                 round_trips=3, serialization_ms=0.5,
                 intermediate_cardinality={"candidates": 20})]
    parts = cost_decomposition(poly, tridb)
    assert parts["gap_ms"] == 6.0
    assert parts["serialization_ms"] == 0.5
    assert parts["round_trip_ms"] > 0
    assert parts["extra_work_ms"] > 0
    total = (parts["round_trip_ms"] + parts["extra_work_ms"]
             + parts["serialization_ms"] + parts["unexplained_ms"])
    assert abs(total - parts["gap_ms"]) < 1e-9
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_analyze.py -v
```

Expected: FAIL，`ModuleNotFoundError`。

- [ ] **Step 3: 实现**

`experiments/e1/analyze_h2h.py`：

```python
"""Equal-quality comparison and cost decomposition for the E1 H2H run."""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.e0.common import write_json

EPS_HIT = 0.02
EPS_MRR = 0.02
# Provisional; Task 14 replaces this with the value measured against the live
# stack. It must be > 0 or the decomposition silently drops the round-trip term.
ROUND_TRIP_MS = 0.1


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[index]


def _by_plan(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latencies: dict[str, list[float]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        latencies[row["plan_id"]].append(float(row["latency_ms"]))
        meta[row["plan_id"]] = row
    out = {}
    for plan_id, values in latencies.items():
        row = meta[plan_id]
        out[plan_id] = {
            "plan_id": plan_id,
            "p50": st.median(values),
            "p95": _percentile(values, 0.95),
            "n": len(values),
            "hit_at_1": float(row["quality"]["hit_at_1"]),
            "mrr": float(row["quality"]["mrr"]),
            "recall_at_20": float(row["quality"]["recall_at_20"]),
            "row": row,
        }
    return out


def _best(plans: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not plans:
        return None
    values = list(plans.values())
    best_hit = max(p["hit_at_1"] for p in values)
    tied = [p for p in values if p["hit_at_1"] >= best_hit - EPS_HIT]
    best_mrr = max(p["mrr"] for p in tied)
    tied = [p for p in tied if p["mrr"] >= best_mrr - EPS_MRR]
    return min(tied, key=lambda p: p["p50"])


def equal_quality_ratio(
    tridb_rows: list[dict[str, Any]], poly_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    keys = sorted(
        {(r["dataset"], r["query_id"]) for r in tridb_rows}
        & {(r["dataset"], r["query_id"]) for r in poly_rows}
    )
    out = []
    for dataset, query_id in keys:
        t = _best(_by_plan([r for r in tridb_rows
                            if r["dataset"] == dataset
                            and r["query_id"] == query_id]))
        p = _best(_by_plan([r for r in poly_rows
                            if r["dataset"] == dataset
                            and r["query_id"] == query_id]))
        if t is None or p is None or t["p50"] <= 0:
            continue
        out.append(
            {
                "dataset": dataset,
                "query_id": query_id,
                "template": t["row"].get("template"),
                "annotation_status": t["row"].get("annotation_status"),
                "hops": t["row"].get("hops"),
                "tridb_plan_id": t["plan_id"],
                "polyglot_plan_id": p["plan_id"],
                "tridb_p50": t["p50"],
                "polyglot_p50": p["p50"],
                "tridb_p95": t["p95"],
                "polyglot_p95": p["p95"],
                "ratio_p50": p["p50"] / t["p50"],
                "ratio_p95": (
                    None if not t["p95"] else p["p95"] / t["p95"]
                ),
                "tridb_hit_at_1": t["hit_at_1"],
                "polyglot_hit_at_1": p["hit_at_1"],
                "tridb_mrr": t["mrr"],
                "polyglot_mrr": p["mrr"],
                "tridb_recall_at_20": t["recall_at_20"],
                "polyglot_recall_at_20": p["recall_at_20"],
            }
        )
    return out


def cost_decomposition(
    poly_rows: list[dict[str, Any]], tridb_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Split the Polyglot-minus-TriDB gap into round trips, extra work, and bytes.

    `unexplained_ms` is reported, never absorbed. A large remainder means the
    three named mechanisms do not account for the gap, and the report must say so.
    """
    t = _by_plan(tridb_rows)
    p = _by_plan(poly_rows)
    shared = sorted(set(t) & set(p))
    gap = sum(p[k]["p50"] - t[k]["p50"] for k in shared)
    serialization = sum(
        float(p[k]["row"].get("serialization_ms") or 0.0) for k in shared
    )
    round_trips = sum(int(p[k]["row"].get("round_trips") or 0) for k in shared)
    round_trip_ms = round_trips * ROUND_TRIP_MS
    extra_work = 0.0
    for k in shared:
        t_cand = float(
            (t[k]["row"].get("intermediate_cardinality") or {}).get("candidates") or 0
        )
        p_cand = float(
            (p[k]["row"].get("intermediate_cardinality") or {}).get("candidates") or 0
        )
        if t_cand > 0 and p_cand > t_cand:
            extra_work += t[k]["p50"] * (p_cand - t_cand) / t_cand
    named = round_trip_ms + extra_work + serialization
    return {
        "cells": len(shared),
        "gap_ms": gap,
        "round_trip_ms": round_trip_ms,
        "round_trips": round_trips,
        "extra_work_ms": extra_work,
        "serialization_ms": serialization,
        "unexplained_ms": gap - named,
        "round_trip_share": round_trip_ms / gap if gap else None,
        "extra_work_share": extra_work / gap if gap else None,
        "serialization_share": serialization / gap if gap else None,
        "unexplained_share": (gap - named) / gap if gap else None,
    }


def _load(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def analyze(raw_path: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = _load(raw_path)
    tridb = [r for r in rows if r["backend"] == "tridb_live"]
    poly = [r for r in rows if r["backend"] == "polyglot_live"]
    per_query = equal_quality_ratio(tridb, poly)
    ratios = [r["ratio_p50"] for r in per_query]
    config = json.loads(json.dumps({"path": str(config_path)}))

    def stratify(field: str) -> dict[str, Any]:
        groups: dict[str, list[float]] = defaultdict(list)
        for row in per_query:
            groups[str(row.get(field))].append(row["ratio_p50"])
        return {
            key: {"n": len(values), "median": st.median(values)}
            for key, values in sorted(groups.items())
        }

    summary = {
        "schema_version": "e1-h2h-summary-v0.1.0",
        "config": config,
        "queries": len(per_query),
        "ratio_p50": {
            "median": st.median(ratios) if ratios else None,
            "p95": _percentile(ratios, 0.95),
            "min": min(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
        },
        "quality": {
            "tridb_hit_at_1": st.mean([r["tridb_hit_at_1"] for r in per_query])
            if per_query else None,
            "polyglot_hit_at_1": st.mean([r["polyglot_hit_at_1"] for r in per_query])
            if per_query else None,
            "tridb_recall_at_20": st.mean(
                [r["tridb_recall_at_20"] for r in per_query]
            ) if per_query else None,
            "polyglot_recall_at_20": st.mean(
                [r["polyglot_recall_at_20"] for r in per_query]
            ) if per_query else None,
        },
        "by_dataset": stratify("dataset"),
        "by_hops": stratify("hops"),
        "by_template": stratify("template"),
        "by_annotation_status": stratify("annotation_status"),
        "cost_decomposition": cost_decomposition(poly, tridb),
        "falsification": {
            "threshold": 1.2,
            "triggered": bool(ratios) and st.median(ratios) < 1.2,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "metrics_query.json", per_query)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/e1/composition_v0.1.yaml")
    )
    args = parser.parse_args(argv)
    summary = analyze(
        args.run_dir / "observations.jsonl", args.config, args.run_dir
    )
    print(json.dumps(
        {"queries": summary["queries"], "ratio_p50": summary["ratio_p50"]},
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_analyze.py -v
```

Expected: 2 passed。第二个测试断言 `round_trip_ms > 0`，所以 Step 3 的占位常量 `0.1` 是必须
的——设为 `0.0` 会让代价分解静默丢掉 round-trip 项。Task 14 会用实测值替换它。

- [ ] **Step 5: 提交**

```bash
cd /localhome/hza214/tridb
git add experiments/e1/analyze_h2h.py tests/test_e1_analyze.py
git commit -m "feat(e1): equal-quality comparison and cost decomposition"
```

---

## Task 13: 消融分析器

**Files:**
- Create: `experiments/e1/analyze_ablation.py`
- Test: `tests/test_e1_analyze.py`

**Interfaces:**
- Produces: `ablation_summary(rows) -> dict`，按 variant 给 realized 质量、能力上界命中率、
  候选集基数分位、延迟分位，并按 `template` / `annotation_status` 分层
- Produces: `necessity_verdict(summary, *, quality_fraction, query_share) -> dict`，
  判定 N 是否被证伪

- [ ] **Step 1: 写失败测试（追加到 `tests/test_e1_analyze.py`）**

```python
def _ab(variant, query_id, hit1, contains, cardinality, latency=1.0):
    return {
        "variant": variant, "dataset": "d", "query_id": query_id,
        "template": "t", "annotation_status": "s", "status": "ok",
        "latency_ms": latency, "candidate_contains_answer": contains,
        "candidate_cardinality": cardinality,
        "quality": {"hit_at_1": hit1, "mrr": hit1,
                    "hit_at_5": hit1, "recall_at_20": hit1},
    }


def test_ablation_summary_reports_all_three_numbers_per_variant():
    from experiments.e1.analyze_ablation import ablation_summary

    rows = [
        _ab("V+G+R", "q1", 1.0, True, 20),
        _ab("V", "q1", 0.0, False, 100),
    ]
    summary = ablation_summary(rows)
    assert summary["variants"]["V+G+R"]["hit_at_1_mean"] == 1.0
    assert summary["variants"]["V"]["capability_hit_rate"] == 0.0
    assert summary["variants"]["V"]["candidate_cardinality_median"] == 100


def test_necessity_verdict_triggers_when_a_pair_matches_the_full_variant():
    from experiments.e1.analyze_ablation import ablation_summary, necessity_verdict

    rows = []
    for i in range(10):
        rows.append(_ab("V+G+R", f"q{i}", 1.0, True, 20))
        rows.append(_ab("V+G", f"q{i}", 1.0, True, 20))
        rows.append(_ab("V", f"q{i}", 0.0, False, 100))
    verdict = necessity_verdict(
        ablation_summary(rows), quality_fraction=0.95, query_share=0.95
    )
    assert verdict["triggered"] is True
    assert "V+G" in verdict["offending_variants"]
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_analyze.py -v
```

Expected: 新增的两个测试 FAIL。

- [ ] **Step 3: 实现**

`experiments/e1/analyze_ablation.py`：

```python
"""Summarise the modality ablation and evaluate sub-proposition N."""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.e0.common import write_json

FULL = "V+G+R"
PAIRS = ("V+R", "V+G", "G+R")


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def ablation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        by_variant[row["variant"]].append(row)

    variants: dict[str, Any] = {}
    per_query: dict[str, dict[str, float]] = defaultdict(dict)
    for name, group in by_variant.items():
        hits = [float(r["quality"]["hit_at_1"]) for r in group]
        contains = [bool(r["candidate_contains_answer"]) for r in group]
        cards = [float(r["candidate_cardinality"]) for r in group]
        lats = [float(r["latency_ms"]) for r in group]
        variants[name] = {
            "observations": len(group),
            "hit_at_1_mean": st.mean(hits),
            "mrr_mean": st.mean([float(r["quality"]["mrr"]) for r in group]),
            "recall_at_20_mean": st.mean(
                [float(r["quality"]["recall_at_20"]) for r in group]
            ),
            "capability_hit_rate": sum(contains) / float(len(contains)),
            "candidate_cardinality_median": st.median(cards),
            "candidate_cardinality_p95": _percentile(cards, 0.95),
            "latency_p50": st.median(lats),
            "latency_p95": _percentile(lats, 0.95),
        }
        for row in group:
            per_query[row["query_id"]][name] = float(row["quality"]["hit_at_1"])

    strata: dict[str, Any] = {}
    for field in ("template", "annotation_status", "dataset"):
        groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in ok:
            groups[(str(row.get(field)), row["variant"])].append(
                float(row["quality"]["hit_at_1"])
            )
        strata[field] = {
            f"{key[0]}|{key[1]}": {"n": len(v), "hit_at_1_mean": st.mean(v)}
            for key, v in sorted(groups.items())
        }

    return {
        "schema_version": "e1-ablation-summary-v0.1.0",
        "variants": variants,
        "per_query_hit_at_1": {q: dict(v) for q, v in per_query.items()},
        "strata": strata,
    }


def necessity_verdict(
    summary: dict[str, Any], *, quality_fraction: float, query_share: float
) -> dict[str, Any]:
    """N fails if a two-modality variant matches the full one nearly everywhere."""
    per_query = summary["per_query_hit_at_1"]
    offending = []
    details = {}
    for pair in PAIRS:
        matched = 0
        counted = 0
        for scores in per_query.values():
            if FULL not in scores or pair not in scores:
                continue
            counted += 1
            full = scores[FULL]
            if full <= 0.0:
                matched += 1  # both at zero: the pair is not worse
            elif scores[pair] >= quality_fraction * full:
                matched += 1
        share = matched / float(counted) if counted else 0.0
        cards = summary["variants"].get(pair, {})
        full_card = summary["variants"].get(FULL, {}).get(
            "candidate_cardinality_median"
        )
        pair_card = cards.get("candidate_cardinality_median")
        blows_up = bool(
            full_card and pair_card and pair_card > 10.0 * full_card
        )
        details[pair] = {
            "query_share_matching": share,
            "candidate_cardinality_median": pair_card,
            "cost_blows_up": blows_up,
        }
        if share >= query_share and not blows_up:
            offending.append(pair)
    return {
        "quality_fraction": quality_fraction,
        "query_share": query_share,
        "offending_variants": offending,
        "triggered": bool(offending),
        "details": details,
    }


def _load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze(run_dir: Path, config_path: Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = _load(run_dir / "observations.jsonl")
    summary = ablation_summary(rows)
    summary["necessity"] = necessity_verdict(
        summary,
        quality_fraction=float(
            config["falsification"]["ablation_quality_fraction_above"]
        ),
        query_share=float(config["falsification"]["ablation_query_share_above"]),
    )
    write_json(run_dir / "ablation_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/e1/composition_v0.1.yaml")
    )
    args = parser.parse_args(argv)
    summary = analyze(args.run_dir, args.config)
    print(json.dumps(summary["necessity"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_analyze.py -v
```

Expected: 4 passed。

- [ ] **Step 5: 提交**

```bash
cd /localhome/hza214/tridb
git add experiments/e1/analyze_ablation.py tests/test_e1_analyze.py
git commit -m "feat(e1): ablation summary and the sub-proposition N verdict"
```

---

## Task 14: 测出单次 round-trip 的固定开销并回填常量

**Files:**
- Create: `tools/e1/measure_round_trip.py`
- Modify: `experiments/e1/analyze_h2h.py`

**Interfaces:**
- Produces: `results/e1/round_trip_cost.json`，含 Milvus / Neo4j / pgvector 三者的空查询往返
  中位耗时
- Produces: `analyze_h2h.ROUND_TRIP_MS` 被替换为实测值

**背景：** Task 12 的 `cost_decomposition` 用 `ROUND_TRIP_MS` 把 round-trip 折算成毫秒。这个常量
必须实测，不能拍脑袋——否则代价分解是循环论证。

- [ ] **Step 1: 写测量脚本**

`tools/e1/measure_round_trip.py`：

```python
"""Measure the fixed per-round-trip cost of each Polyglot store.

A round trip is charged even when the query returns nothing, so the floor is
measured with the cheapest possible request to each store: a trivial Cypher
RETURN, a one-row SELECT, and a top-1 Milvus search over one vector.
"""

from __future__ import annotations

import json
import statistics as st
import time
from pathlib import Path

import numpy as np

from tools.e0.common import write_json

TRIALS = 200


def _ms(started: int) -> float:
    return (time.perf_counter_ns() - started) / 1e6


def main() -> int:
    import psycopg
    from neo4j import GraphDatabase
    from pymilvus import Collection, connections

    samples: dict[str, list[float]] = {"milvus": [], "neo4j": [], "pgvector": []}

    connections.connect(alias="rt", host="127.0.0.1", port="19530")
    collection = Collection("e0_stark_prime", using="rt")
    collection.load()
    vector = np.zeros(1024, dtype=np.float32)
    vector[0] = 1.0
    for _ in range(TRIALS):
        started = time.perf_counter_ns()
        collection.search(
            data=[vector.tolist()], anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}}, limit=1,
        )
        samples["milvus"].append(_ms(started))

    driver = GraphDatabase.driver(
        "bolt://127.0.0.1:7688", auth=("neo4j", "testpassword")
    )
    with driver.session() as session:
        for _ in range(TRIALS):
            started = time.perf_counter_ns()
            session.run("RETURN 1 AS x").single()
            samples["neo4j"].append(_ms(started))
    driver.close()

    with psycopg.connect(
        host="127.0.0.1", port=5434, dbname="tridb_wiki",
        user="postgres", password="postgres",
    ) as pg:
        with pg.cursor() as cursor:
            for _ in range(TRIALS):
                started = time.perf_counter_ns()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                samples["pgvector"].append(_ms(started))

    report = {
        "schema_version": "e1-round-trip-v0.1.0",
        "trials": TRIALS,
        "per_store_median_ms": {k: st.median(v) for k, v in samples.items()},
        "per_store_p95_ms": {
            k: sorted(v)[int(0.95 * len(v))] for k, v in samples.items()
        },
        "mean_of_store_medians_ms": st.mean(
            [st.median(v) for v in samples.values()]
        ),
    }
    write_json(Path("results/e1/round_trip_cost.json"), report)
    print(json.dumps(report["per_store_median_ms"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 运行测量**

```bash
cd /localhome/hza214/tridb && mkdir -p results/e1 && .venv/bin/python -m tools.e1.measure_round_trip
```

Expected: 打印三个中位数（毫秒量级，多半在 0.1–1 ms）。

- [ ] **Step 3: 回填常量**

把 `experiments/e1/analyze_h2h.py` 中的

```python
# Provisional; Task 14 replaces this with the value measured against the live
# stack. It must be > 0 or the decomposition silently drops the round-trip term.
ROUND_TRIP_MS = 0.1
```

替换为实测的 `mean_of_store_medians_ms`，并把注释改为：

```python
# Fixed per-round-trip cost, measured by tools/e1/measure_round_trip.py against
# the live stack; see results/e1/round_trip_cost.json for the per-store split.
ROUND_TRIP_MS = <实测值>
```

- [ ] **Step 4: 重跑分析测试**

```bash
cd /localhome/hza214/tridb && .venv/bin/python -m pytest tests/test_e1_analyze.py -v
```

Expected: 4 passed。

- [ ] **Step 5: 提交**

```bash
cd /localhome/hza214/tridb
git add tools/e1/measure_round_trip.py experiments/e1/analyze_h2h.py results/e1/round_trip_cost.json
git commit -m "feat(e1): measure the fixed per-round-trip cost instead of assuming it"
```

---

## Task 15: Makefile 目标、完整测量与报告

**Files:**
- Modify: `Makefile`
- Create: `docs/benchmark_e1_composition_v0.1.0.md`

**Interfaces:**
- Produces: `make e1-h2h`、`make e1-ablation`、`make e1-analyze`

- [ ] **Step 1: 加 Makefile 目标**

在 `.PHONY` 行末尾追加 ` e1-h2h e1-ablation e1-analyze`，并在 `e0-tridb-analyze` 目标之后加入：

```makefile
E1_CONFIG ?= configs/e1/composition_v0.1.yaml
E1_H2H_OUT ?= results/e1/h2h_v0.1
E1_ABL_OUT ?= results/e1/ablation_v0.1

e1-h2h:
	$(PY) -m experiments.e1.interleaved_runner \
	  --config $(E1_CONFIG) --output-dir $(E1_H2H_OUT)

e1-ablation:
	$(PY) -m experiments.e1.ablation_runner \
	  --config $(E1_CONFIG) --output-dir $(E1_ABL_OUT)

e1-analyze:
	$(PY) -m experiments.e1.analyze_h2h --run-dir $(E1_H2H_OUT) --config $(E1_CONFIG)
	$(PY) -m experiments.e1.analyze_ablation --run-dir $(E1_ABL_OUT) --config $(E1_CONFIG)
```

- [ ] **Step 2: 确认门禁已通过**

```bash
cd /localhome/hza214/tridb && jq '.gate_passed, .counts' results/e1/parity_report.json
```

Expected: `true`。**若为 false，停止，回到 Task 4。**

- [ ] **Step 3: 跑完整 H2H**

```bash
cd /localhome/hza214/tridb && scripts/baseline_up_podman.sh && make e1-h2h
```

Expected: `{"errors": 0, "resumed": 0, "written": N}`，N 约为 2 × 50 × (4–6) × 31 ≈ 12,400–18,600。

- [ ] **Step 4: 跑完整消融**

```bash
cd /localhome/hza214/tridb && make e1-ablation
```

Expected: `{"errors": 0, "written": 10850}`（50 query × 7 变体 × 31 rep）。

- [ ] **Step 5: 分析**

```bash
cd /localhome/hza214/tridb && make e1-analyze
jq '{ratio: .ratio_p50, quality: .quality, decomposition: .cost_decomposition, falsification: .falsification}' results/e1/h2h_v0.1/summary.json
jq '.necessity' results/e1/ablation_v0.1/ablation_summary.json
```

- [ ] **Step 6: 独立重复一次 H2H**

```bash
cd /localhome/hza214/tridb && make e1-h2h E1_H2H_OUT=results/e1/h2h_v0.1_repeat
make e1-analyze E1_H2H_OUT=results/e1/h2h_v0.1_repeat
```

比较两次的 `ratio_p50.median`。若两次相差超过 1.2×，说明测量不稳定，报告必须写明而不是取
其中好看的那次。

- [ ] **Step 7: 写报告**

`docs/benchmark_e1_composition_v0.1.0.md`，结构照抄 `docs/benchmark_e0_tridb_v0.3.0.md`：

1. **Material Passport**：日期、实验、状态、两个 backend、冻结契约路径、scope、design size、
   平台、**claim boundary**（stock PostgreSQL 16.14、x86_64、8 KiB `BLCKSZ`；不是 GX10 /
   ARM64 / CUDA / 32 KiB fork / 128 GB sign-off）。
2. **公平性门禁**：`results/e1/parity_report.json` 的 verdict 分布与 Jaccard 中位数。
3. **子命题 C 结果**：等质量 `ratio_p50` 的 median / p95 / min / max；两侧 Hit@1、MRR、
   Recall@20；证伪条件是否触发。
4. **代价分解**：round-trip / 多做功 / 序列化 / **未解释残差** 四项占比。未解释残差必须列出，
   不得吸收进其它三项。
5. **必须写的三条不利证据**（spec §1.4）：比值不随 hop 或 k 增长（E0 实测 hop=1 为 1.78×、
   hop=2 为 1.20×；k 从 1 到 100 平稳在 1.26–1.47×）；序列化仅占约 1%；Recall@20 上 TriDB
   不占优。
6. **子命题 N 结果**：七变体的 realized 质量、能力上界命中率、候选集基数、延迟；
   `necessity.triggered` 与 offending variants；按 `template` / `annotation_status` 分层表，
   明确指出哪类查询不需要三模态。
7. **重复性**：两次 H2H 的 `ratio_p50.median` 对比。
8. **已知局限**：n=31 不支持 p99；消融的 shape 随变体变化（是模态组合的必然后果，不是自由
   选择）；OpenEvolve 若未纳入需说明原因。

- [ ] **Step 8: 全量测试**

```bash
cd /localhome/hza214/tridb && make test && make lint
```

Expected: 全部通过。

- [ ] **Step 9: 提交**

```bash
cd /localhome/hza214/tridb
git add Makefile docs/benchmark_e1_composition_v0.1.0.md results/e1/
git commit -m "feat(e1): composition cost and modality ablation result v0.1.0"
```

---

## Self-Review 记录

**Spec 覆盖检查：**

| Spec 章节 | 对应 Task |
|---|---|
| §1.3 证伪条件冻结 | Task 9（config）、Task 12（C 判定）、Task 13（N 判定） |
| §1.4 不利证据必须写进报告 | Task 15 Step 7 第 5 条 |
| §2.1 OpenEvolve 断点 | Task 1 |
| §2.2 `same_parent` | Task 2 |
| §2.3–2.4 oracle 对账与门槛 | Task 4 |
| §2.5 撤回 E0 污染结论 | Task 5 |
| §3.1 交错协议 | Task 10 |
| §3.2 网格收缩换重复数 | Task 8、Task 9 |
| §3.3 指标（含跨边界行数） | Task 3、Task 10 |
| §3.4 代价分解 | Task 12、Task 14 |
| §4.1 `graph_off` / `vector_off` | Task 6 |
| §4.2 七变体与 shape 映射 | Task 7 |
| §4.3 三个数 | Task 11、Task 13 |
| §4.4 分层 | Task 12、Task 13 |
| §5 产物 | Task 15 |

**已知遗留：** spec §4.2 末句要求 V+G+R 另跑一次「该 query 的等质量最优 shape」以确认
traverse-first 基线没有低估完整系统。本计划未单列任务——它由 Task 10 的 H2H run 覆盖，
因为那里 TriDB 就是用每 query 的等质量最优计划测的，两者可直接对照。报告中需明确写出这个
对照，见 Task 15 Step 7 第 6 条。
