# E0 执行计划 — Polyglot-Tuned 上的计划空间离散度（STARK-PRIME）v0.1.0

> **版本：** 0.1.0
> **日期：** 2026-08-18
> **状态：** 实现计划。**尚未执行，无任何测量结果。**
> **对应：** [`haikaidocs/trevillisPlan.md`](../haikaidocs/trevillisPlan.md) §3 E0 / §6 P0
> **依赖：** [`e0_data_preparation_v0.1.0.md`](e0_data_preparation_v0.1.0.md)（数据契约与 pin）

---

## 0. 结论先行：现在能跑，但有三个必须先修的缺口

**E0 不需要 TriDB 引擎。** 它测的是 polyglot 拼装架构自身的计划空间离散度——被测对象是
Polyglot-Tuned 基线，不是 TriDB。所以 MSVBASE fork 镜像尚未构建**不阻塞 E0**。这是"现在
能否跑"的直接答案。

已就绪（2026-08-18 实测）：

| 组件 | 状态 |
|---|---|
| STARK-PRIME normalized | ✅ nodes 129,375 / edges 8,100,498 / embeddings 129,375×1024，SHA 全部校验通过 |
| embedding 模型 | ✅ vLLM `Qwen/Qwen3-Embedding-0.6B` @ `127.0.0.1:8001`（1024 维） |
| 生成模型 | ✅ vLLM `Qwen/Qwen3-32B` @ `127.0.0.1:8000`（E0 不需要，E5 才用） |
| Polyglot-Tuned 底座 | ✅ Milvus 19530 / Neo4j bolt 7688 / pgvector 5434，**同机、host networking、无端口转发代理**（`scripts/baseline_up_podman.sh`） |
| E0 python 环境 | ✅ `.venv-e0`（3.11，stark-qa 1.1.0，numpy 1.26.4 硬 pin） |

未就绪，且**必须在出任何 plan spread 数字之前修掉**：

### G1. 查询数量：10 条撑不起 P0 的判停条件

`queries_v0.1.jsonl` 只有 10 条人工审计查询。而 P0 的判停条件是
**「spread 中位数 < 2× → 命题 O 不成立，立即重新 scope」**——用 10 个点的中位数去触发
或否决一个"立即重新 scope"的决定，是拿噪声做重大决策。trevillisPlan §3 E0 本身写的也是
30–50 条。

官方资源足够：`stark_qa.csv` 有 **11,204** 条带 `answer_ids` 的查询，test split 2,801 条。
扩到 40 条完全可行。

### G2. traverse-first 执行器不存在

`e0_data_preparation_v0.1.0.md` §5 明确写着：本阶段不含 traverse-first 执行器，
「当前系统层 pilot 仍只应比较已经存在的 vector-first 和 filter-first」。

但 E0 的 plan shape 维度是 **vector-first / filter-first / traverse-first** 三选。缺一种，
「最优 plan shape 随查询而变的比例」这个指标就只在二元空间里统计，结论强度显著下降。

在 polyglot 侧 traverse-first **是可表达的**（从 anchor 的 typed 出边开始扫，最后才做向量
排序），代价是候选集可能爆炸。这是一个需要你拍板的设计决策，见 §4 D1。

### G3. `ready_for_e0 = false`

`data/e0/preparation_manifest.json` 顶层是 `false`，原因是 **OpenEvolve 腿未 normalize/embed**
（`normalized_ready: false`）。STARK-PRIME 腿本身是 `ready: true`。

两条路二选一（§4 D2）：把 E0 的 scope 显式冻结为 stark-only 并让 verify 输出一个
scope-aware 的 ready 标志；或者先补完 OpenEvolve 腿。**不能**在 `ready_for_e0=false` 的
状态下直接开跑却不记录——那正是这套 manifest 机制要防的事。

---

## 1. 已核实的事实（会改变实验设计的那些）

### 1.1 STARK-PRIME 的跳数只有 1–2 跳 —— 深度扫描在这个数据集上做不了

trevillisPlan §7「待核实事项」要求先统计 STARK 查询的跳数分布。**已统计**（10 条 pilot）：

```
hop_limit dist: {1: 5, 2: 5}
template dist:  typed_chain 3 / neighbor_intersection 3 / typed_neighbors 2 / shared_neighbor_constraint 2
```

E0 设计里的 `hop depth ∈ {1, 2, 3, 5}` 在 STARK-PRIME 上**只能扫到 {1, 2}**。3 跳和 5 跳
在这个数据集上没有对应的真实查询语义，硬扫出来的延迟数字不对应任何用户意图。

**结论：** STARK-PRIME 负责"图密、真实、外部可比"这一端；**深度维度必须由 OpenEvolve 树补位**
（trevillisPlan §1.3 已经这么设计，§7 的怀疑得到证实）。E0 在 STARK 上报告的是
**shape × k × 谓词放置** 三个维度的离散度，hop 维度只有两档，报告时必须写明。

### 1.2 答案集极小 —— Recall@20 会饱和，Hit@1 才是有区分度的指标

10 条 pilot 的答案数：`[4, 1, 3, 4, 1, 1, 3, 1, 1, 1]`，中位数 1。

在这种分布下 Recall@20 几乎恒为 1.0（只要答案进了前 20），**无法区分计划质量**，于是
"质量等价"这个前提会退化成"全都等价"，plan spread 就变成了纯延迟比较——那不是 E0 想要的。

**结论：** 质量口径以 **Hit@1 / MRR** 为主，Recall@20 只作附录记录。§3 的 quality-equivalence
定义据此设定。

### 1.3 边是双向 arc，不是无向边

`stark-qa` 默认把每条 PrimeKG 关系展开成两个方向的 adjacency arc，Parquet 保留了该语义
（8,100,498 arcs）。加载到 Neo4j 时**必须原样保留方向**，否则遍历基数和官方 answer 的可达性
都会偏。同理，`edges.parquet` 不得退化成关系型 join table。

---

## 2. 被测对象：Polyglot-Tuned 的准确定义

trevillisPlan §2 对 Polyglot-Tuned 的定义是「同机部署（消除网络）+ 批处理 + 异步 + 连接池」。
逐条对照当前底座：

| 要求 | 当前状态 | 备注 |
|---|---|---|
| 同机部署 | ✅ 三个存储与 harness 同机 | |
| 消除网络 | ✅ **全部 `--network=host`** | rootless podman 的发布端口要过 pasta/slirp4netns 用户态代理，那个代理在请求路径上会给每次 round-trip 加延迟。映射端口 = 人为拖慢基线 = 制造假胜利。已规避。 |
| 连接池 | ⬜ 待实现 | Neo4j driver 自带；pymilvus 需显式复用 alias；psycopg 需 `ConnectionPool` |
| 批处理 | ⬜ 待实现 | id 列表一次性下推，禁止逐行往返 |
| 异步 | ⬜ 待实现 | 仅在**同一 plan 内存在可并行的独立子请求**时启用；不得用异步掩盖串行依赖 |

**诚实边界（必须写进最终报告）：** Polyglot-Tuned 是我们自己实现并自己调优的基线。
「我们已经尽力调优对手」这句话的可信度取决于调优细节是否公开。因此：每一项调优措施都要
在报告里逐条列出并给出开关，允许审稿人重跑 `--tuning=naive|tuned` 对照。

---

## 3. 度量口径（必须在跑之前冻结）

### 3.1 quality-equivalence 的定义

plan spread 只能在**质量等价的计划之间**计算，否则"最快的计划"永远是"什么都不做的计划"。

冻结定义：
```
计划 P 与 Q 质量等价  ⟺  |Hit@1(P) − Hit@1(Q)| ≤ ε₁  且  |MRR(P) − MRR(Q)| ≤ ε₂
ε₁ = 0.02, ε₂ = 0.02   (对齐 bench/wiki_fusion.py 的 matched-recall 协议，eps 同量级)
```
每条查询独立判定：先取该查询上所有计划中的**最高 Hit@1**，把落在 ε₁ 内的计划构成
"质量等价集"，`plan_spread = max_latency / min_latency` **只在这个集合内**计算。

### 3.2 要报告的核心数字（trevillisPlan §3 E0 原文三项）

| 指标 | 定义 | 判停 |
|---|---|---|
| **plan spread** | 质量等价集内 max/min p50 延迟，按查询给出分布 | **本轮唯一判停项：中位数 < 2× → 命题 O 不成立，停** |
| **默认计划次优比** | cost(硬编码默认) / cost(该查询最优)。默认取 GraphRAG 风格 top-k=8 + 固定 2-hop + post-filter | 分布，无判停 |
| **最优 shape 随查询变化比例** | argmin shape 不恒定的查询占比 | **降级为描述性统计**，不判停——shape 空间只有 2 个元素（D1），见 §4 |

### 3.3 机理插桩（E1 会复用，E0 顺手采集）

每个计划每次执行记录：`round_trips`、`bytes_shipped`（跨存储边界，行数 + 字节）、
每阶段耗时（ANN / traversal / filter / merge）、中间基数（`|seeds|`/`|reached|`/`|cand|`）、
序列化耗时占比。延迟一律报 **p50 / p95 / p99**，禁止均值。

---

## 4. 已冻结的决策（2026-08-18）

| # | 决策 | 取值 | 直接后果 |
|---|---|---|---|
| **D1** | traverse-first 做不做 | **不做。** shape 空间 = {vector-first, filter-first} | §3.2 的"最优 shape 随查询变化"指标降级，见下 |
| **D2** | scope | **stark-only。** `tools/e0/verify.py` 改为 per-dataset ready，E0 只 gate 在 `stark_prime.ready` | OpenEvolve 腿留给 P1，不阻塞 P0 |
| **D3** | 查询数 | **40 条**（从 test split 分层采样 + 人工审计） | 支撑得起 P0 判停的统计基础 |

### D1 带来的指标降级 —— 必须写进报告

shape 空间只有两个元素时，"最优 shape 随查询而变的比例" **不再能否证命题 O**：

- 若测得两种 shape **各有胜负**，这是命题 O 的**必要非充分**证据——它证明固定规则不够，但
  没有排除"存在第三种恒优 shape"（traverse-first 恰恰没测）。
- 若测得某一种 shape **恒优**，**不能**据此宣布命题 O 死亡——只能说"在这两种 shape 之间不需要
  选择"。真正的否证需要覆盖完整 shape 空间。

因此本轮 E0 的**判停条件收窄为只看 plan spread**（§3.2 第一行），shape 变化比例作为
**描述性统计**报告，不承担判停职责。traverse-first 补齐前，命题 O 的否证结论不得下达。
补齐时需重跑完整笛卡尔积（shape 是最外层维度，无法增量拼接）。

---

## 5. 分阶段实现

### Phase A — 前置（D1/D2/D3 已冻结，见 §4）

| # | 交付物 | 说明 |
|---|---|---|
| A1 | `tools/e0/query_expand.py` + `queries_v0.2.jsonl` | 从 `stark_qa.csv` test split 采样，按 template 分层，自动抽 anchor/typed path，产出候选标注供人工审计；沿用 `oracle_method` 与 `path_audit` 字段格式 |
| A2 | `tools/e0/verify.py` 改 per-dataset ready | 输出 `ready_for_e0.stark_prime` / `.openevolve`，顶层保留但不再是唯一门 |
| A3 | 冻结 `configs/e0/plan_space_v0.1.yaml` | shape / k / hop / 谓词放置 / beam 的笛卡尔积定义 + ε₁ε₂ + 默认计划定义 |

### Phase B — 加载进三个存储

| # | 交付物 | 要点 |
|---|---|---|
| B1 | Milvus collection `stark_prime_nodes` | 1024-d，HNSW/COSINE，`M=16, efConstruction=200`；`node_id` 为 PK |
| B2 | Neo4j `(:Node {node_id, entity_type})` + 8.1M typed arcs | **保方向**（§1.3）；`edge_type` 作为关系类型或属性——需实测两种表示的遍历性能差异并记录，因为这直接影响基线公平性 |
| B3 | pgvector `stark_prime_node(node_id, entity_type, name, attributes jsonb, embedding vector(1024))` | HNSW 索引；`entity_type` B-tree；`attributes` GIN |
| B4 | `tools/e0/load_verify.py` | 三边计数对账 + 抽样 oracle 可达性复验（官方 answer 在图里确实可达） |

### Phase C — Polyglot-Tuned 执行器

`experiments/e0/polyglot_tuned/` 下：

| # | 交付物 | 说明 |
|---|---|---|
| C1 | `executor.py` | **两种** shape（vector-first / filter-first，D1）的实现，统一 `PlanSpec → PlanResult` 接口。shape 作为最外层维度留出扩展位，但本轮不实现 traverse-first |
| C2 | `tuning.py` | 连接池 / 批处理 / 异步，`--tuning=naive|tuned` 双档可切（§2 的可审计性要求） |
| C3 | `instrument.py` | round_trips / bytes_shipped / 分阶段耗时 / 中间基数 |
| C4 | `run_plan_space.py` | 笛卡尔积驱动 + checkpoint 续跑（计划数 = shape×k×hop×placement，40 查询下是数千次执行，必须可断点续跑） |

### Phase D — 度量与产出

| # | 交付物 |
|---|---|
| D1 | `results/e0/plan_space/metrics.parquet` + `MANIFEST.json`（沿用 `results/agent_memory_characterization/` 的 manifest 惯例） |
| D2 | plan spread 分布图 / 默认计划次优比分布 / shape 随查询变化统计 |
| D3 | `docs/benchmark_e0_plan_space_v0.1.0.md` — 含判停结论，**包括命题 O 不成立时的诚实报告路径** |

---

## 6. 风险与诚实边界

- **这不是 TriDB 的数字。** E0 全程测的是 polyglot 基线的计划空间。任何"TriDB 更快"的结论都
  不在 E0 的射程内，报告标题与摘要必须防止这种误读。
- **底座是 rootless podman + host networking**，不是 docker。存储版本与
  `baseline/docker-compose.yml` 逐一对齐，但运行时不同，需在 Material Passport 里记录。
- **调优基线的自证问题**（§2）：我们既造对手又调对手。缓解手段是 `--tuning` 开关 + 逐条公开。
- **STARK 只有 1–2 跳**（§1.1）：深度维度的结论一律不得从 E0 得出。
- **答案集极小**（§1.2）：Recall@20 饱和，不作为质量等价判据。
- **命题 O 可能不成立。** 若 spread 中位数 < 2×，按 trevillisPlan §6 P0 立即停并重新
  scope——这个结果要照常写进文档，不得因为"不好看"而扩大搜索空间去凑。
- **shape 空间不完整（D1）：** 本轮只有 vector-first / filter-first。命题 O 的**否证**结论
  不得在 traverse-first 补齐之前下达，理由见 §4。反向的"成立"结论则不受影响——两种 shape
  之间已经出现显著 spread，加入第三种只会让空间更大。
