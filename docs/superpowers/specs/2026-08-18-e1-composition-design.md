# E1 设计：跨模态组合查询的执行代价与模态必要性

- 日期：2026-08-18
- 阶段：E1，对应 `haikaidocs/trevillisPlan.md` §3 E1
- 打的子命题：**C**（组合惩罚）与 **N**（模态必要性）
- 本轮**不做**：Polyglot-Naive 三档收敛、PostgreSQL+pgvector+AGE 第三基线、E2/E3/E4
- 状态：设计已确认，待实施

---

## 1. 命题

### 1.1 子命题 C —— 组合惩罚

同一条 query、同一份数据、同一质量水平下，比较两个系统。**"同一质量水平"沿用 E0 冻结的
质量等价规则**：先保留达到该 query 最佳 Hit@1（容差 0.02）的计划，再在其中保留达到最佳
MRR（容差 0.02）的计划；Hit@5 与 Recall@20 只记录，不参与等价判定。

| 系统 | 构成 | 组合发生在哪 |
|---|---|---|
| **TriDB** | 单个 PostgreSQL 进程内的 pgvector + 关系存储 + 原生邻接表图访问方法 | 引擎内，一条语句，`tjs_e0_open` 算子 |
| **Polyglot-Tuned** | Milvus v2.4.5 + Neo4j 5.20 + pgvector/pg16，同机 rootless podman `--network=host` | 应用层 Python 编排，跨进程 |

Polyglot-Tuned 已消除网络因素、复用连接池、批量下推 ID。它不是稻草人；残余差距才是 C 要归因的东西。

### 1.2 子命题 N —— 模态必要性

在 TriDB 单引擎内关断模态腿，证明去掉任何一种模态"要么质量塌，要么成本爆"。

### 1.3 测量前冻结的证伪条件

| 条件 | 触发后的处置 |
|---|---|
| 等质量下 `p50(poly-tuned) / p50(tridb)` 中位数 **< 1.2×** | C 不成立，E1 降级为 characterization，如实报告 |
| 存在某双模态变体在 ≥95% 查询上达到三模态 95% 的 Hit@1 **且**候选集不膨胀 | N 不成立，主张缩小到特定 query template |

### 1.4 必须写进报告的既知不利证据

从已完成的 E0 两个 run（`tridb_live_v0.3`、`polyglot_live_v0.2`，3,450 个对齐的 plan cell）推导：

- 等质量延迟比中位数 **1.80×**（STARK-PRIME），质量 Hit@1 0.500 = 0.500、MRR 0.608 = 0.608、Recall@20 TriDB 0.728 **低于** Polyglot 0.761。
- **比值不随组合深度增长**：hop=1 为 1.78×，hop=2 为 1.20×；k 从 1 到 100 平稳在 1.26–1.47×。
- 序列化只占 Polyglot 延迟的 **0.93%**；round-trip 均值 2.41（最多 3），跨边界字节均值 2.5 KB、最大 167 KB。

因此 C 只能主张**恒定因子形式**，机理归因到 round-trip 固定开销，而非序列化，也非随深度放大的下推缺失。这三点必须主动出现在报告里，不能等审稿人发现。

---

## 2. Phase 0 — 公平性门禁

不通过不得进入 Phase 1。任何质量对比在门禁通过前都不可发表。

1. 起 baseline 容器，用 `oe-000` 的原始参数逐 leg 手打 Milvus search / Cypher / pgvector，定位 OpenEvolve 全空结果（1,010/1,010 cell 的 `result_ids` 为空）的具体断点。复现必须使用与原 run 完全一致的参数形式，不做等价改写。
2. 补 `same_parent` 谓词到 `experiments/e0/plan_spread/live_backend.py` 的 `_predicate_sql` 与 `_neo_predicate`。该谓词在 10 条 OpenEvolve query 中有 3 条用到，目前被 Polyglot 侧静默忽略而 TriDB 侧已实现，属既存的不公平。
3. 用 `parquet_reference` 作裁判，对 STARK 那 173 个双方质量分歧的 cell 逐条判责为 {tridb 错 / polyglot 错 / ANN 近似导致的合法差异}，产出 `results/e1/parity_report.json`，按 shape × k 分层给 exact-match 率与 top-20 Jaccard。
4. **门槛**：两侧对 oracle 的 top-20 Jaccard 中位数 ≥ 0.98，且剩余差异须通过拉高 `hnsw_ef_search` / Milvus `ef` 收敛到 oracle 来证明属于 ANN 近似。
5. 回头在 `docs/benchmark_e0_plan_space_v0.1.0.md` 标注 OpenEvolve 段落被污染：质量恒为 0 导致质量等价集退化成整个网格，其 spread 9.38× 无效。

**时间盒**：OpenEvolve 断点修复限 1 天。超时则 E1 限定 STARK-PRIME，并在 Material Passport 明写 OpenEvolve 未纳入及原因。

---

## 3. Phase 1 — H2H 交错测量（子命题 C）

### 3.1 交错协议

现有两个 E0 run 相隔 13 小时，非交错，机器漂移无法排除。E1 改为**同进程内交错**：对每个 `(query, plan, repetition)`，以随机顺序依次执行 TriDB 与 Polyglot-Tuned，两者共享同一份预先算好的 query embedding。漂移因此成为两系统共同的噪声而非偏置。

### 3.2 网格收缩换重复数

E0 的 `repetitions: 3` 只够 p50，而 trevillisPlan 要求 p50/p95/p99。E1 每 query 保留 5 个计划：

- 三个 plan shape 各一个代表（vector_first / filter_first / traverse_first）
- 该 query 在 **TriDB** 上的等质量最优计划（由 E0 `tridb_live_v0.3` 确定）
- 该 query 在 **Polyglot-Tuned** 上的等质量最优计划（由 E0 `polyglot_live_v0.2` 确定）
- 硬编码默认计划（vector_first, k=8, hop=2, post）

两系统的最优计划若相同则去重，该 query 的计划数降为 4。两个系统都执行这一整组计划，
不允许任一系统只跑对自己有利的那个——否则比较退化为不同计划之间的比较。

`repetitions: 31`，报 **p50 / p95**，并明写 n=31 不足以报 p99——不硬凑。

规模：2 系统 × 50 query × 5 计划 × 31 rep ≈ 15,500 observation。E0 的 10,350 条耗时 88 秒，成本可忽略。

### 3.3 指标

- 延迟 p50 / p95
- 质量 Hit@1 / MRR / Recall@20
- round-trip 次数
- 跨边界中间结果**行数**（现有实现只记字节，需补）与字节数
- 序列化耗时占比
- `intermediate_cardinality`（候选集、谓词命中、图边检查数）

### 3.4 代价分解

把 `tuned − tridb` 的差拆成三项并报告各自占比：

- (a) round-trip 固定开销 × 次数
- (b) 无跨模态下推导致的多搬运，用两侧候选集基数比值量化
- (c) 序列化

已知 (c) ≈ 1%，作为负面结果直接报。这一分解直接回应 trevillisPlan §5 中最强的那条审稿意见（SHADB 式质疑：收益能否被搬进通用引擎）。

---

## 4. Phase 2 — TriDB 模态消融（子命题 N）

### 4.1 关断方式

`tjs_e0_open(table, k, top_n, hops, id_col, predicate, qvec, anchors[], edge_types[], require_each, shape, placement)`：

| 关掉 | 实现 | 是否改 C |
|---|---|---|
| relational | `predicate` 传 `'TRUE'` | 否 |
| vector | 传零向量，所有 cosine 距离相等，退化为确定性 id tie-break | 否 |
| graph | 需一条跳过可达性约束的路径 | **是**，benchmark-only 面加 `graph_off` 布尔参数 |

`graph_off` 落在 `src/tjs_pg`，属 stock-PG 可 build 可测路径，不受 GX10 门禁。它不进入 `graph_query()` 公开 v1 模板。

### 4.2 七个变体

固定在每 query 在 TriDB 上的等质量最优 `k` / `hop` 上（与 §3.2 取同一个计划），
只变模态组合。plan shape 固定为该最优计划的 shape，使七个变体之间唯一的自变量是模态组合：

| 变体 | 对应现实系统能力 | 预期 |
|---|---|---|
| V | 纯向量 RAG | 塌 |
| G | 纯图遍历 | 大 reach 时塌 |
| R | 纯关系过滤 | 塌 |
| V+R | ≈ VBASE / ACORN 能力上限 | 需多跳的查询上塌 |
| V+G | ≈ GraphRAG / Zep 能力 | 需结构化谓词的查询上塌或成本爆 |
| G+R | 无语义排序 | 大 reach 时塌 |
| V+G+R | 完整 | 基线 |

规模：7 × 50 × 31 ≈ 10,850 observation。

### 4.3 每个变体必报三个数

1. **realized 质量**：Hit@1 / MRR / Recall@20，确定性 id tie-break，规则在 spec 中声明，以排除"塌是因为随机排序"的解释。
2. **能力上界**：答案是否**存在于**该变体的候选集中，不受排序影响。若某变体的候选集根本不含答案，再完美的排序也救不了——这是最强的论证形式。
3. **候选集基数与延迟**：量化"成本爆"。E0 实测参照：STARK reach hop1 p50=4 / hop2 p50=23，单条 query 的谓词命中 4,176 行，全库 129,375 行。

### 4.4 分层

按 `template` 与 `annotation_status` 分层报告，诚实指出哪类查询不需要三模态。E0 已证明人工审计的 10 条与自动派生的 30 条会给出相反结论（spread 3.15× vs 1.65×），E1 不得重犯这个错误。

---

## 5. 产物

`docs/benchmark_e1_composition_v0.1.0.md`，沿用 `docs/benchmark_e0_tridb_v0.3.0.md` 的 Material Passport 格式，含 claim boundary：stock PostgreSQL 16.14、x86_64、8 KiB `BLCKSZ`，**不是 GX10 / ARM64 / CUDA / 32 KiB fork / 128 GB 的 sign-off**。

图：等质量延迟分布对比、代价分解堆叠图、消融的质量-候选集双轴图、分层表、capability 上界表。

---

## 6. 代码改动清单

| 位置 | 改动 | 风险 |
|---|---|---|
| `experiments/e0/plan_spread/live_backend.py` | 补 `same_parent`；修 OpenEvolve 断点；补跨边界行数计数 | 低 |
| `src/tjs_pg/tjs_pg.c` | `tjs_e0_open` 增 `graph_off` 布尔参数 | 中，需过 stock-PG 测试套件 |
| `experiments/e0/tridb_schema.sql` | 更新函数签名 | 低 |
| `experiments/e1/interleaved_runner.py` | 交错 A/B runner，复用 `plan_spread` 的 model / config / checkpoint 骨架 | 新增 |
| `experiments/e1/ablation_runner.py` | 七变体消融 runner | 新增 |
| `experiments/e1/analyze.py` | 代价分解、等质量比较、消融分层分析 | 新增 |
| `configs/e1/composition_v0.1.yaml` | 测量前冻结的契约 | 新增 |
| `tools/e1/parity_referee.py` | oracle 对账 | 新增 |

## 7. 环境

| | |
|---|---|
| TriDB | stock PostgreSQL 16.14，socket `/localhome/hza214/tridb/.tridb-pgdata`，port 55432 |
| Milvus | v2.4.5，:19530 |
| Neo4j | 5.20，bolt :7688 |
| pgvector | pgvector/pgvector:pg16，:5434 |
| 部署 | 同机，rootless podman，`--network=host` |
| 平台 | Linux x86_64，Python 3.13.12（`.venv-e0`） |

## 8. 工期

| 阶段 | 天 |
|---|---|
| Phase 0 门禁 | 1–2 |
| Phase 1 交错 H2H | 2 |
| Phase 2 消融 | 2 |
| Phase 3 报告 | 1 |
| 合计 | 6–8 |
