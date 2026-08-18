# E0 — 计划空间离散度，Polyglot-Tuned on STARK-PRIME（v0.1.0）

> **状态：EXECUTED。P0 判停条件触发 —— 本轮测量不支持命题 O。**
> **日期：** 2026-08-18 · 40 条查询 × 96 计划 × 3 次重复 = 3,840 次执行 / 319 s
> **原始数据：** `results/e0/plan_space/raw.jsonl` · 分析：`results/e0/plan_space/analysis.json`
> **口径：** `configs/e0/plan_space_v0.1.yaml`（**测量前**冻结）
> **被测对象：Polyglot-Tuned 基线，不是 TriDB。** 本文任何数字都不构成 TriDB 的性能主张。

## TL;DR

| 指标 | 结果 | 判停 |
|---|---|---|
| **plan spread 中位数** | **1.88×** | **< 2.0× → 命题 O 在本轮不成立** |
| plan spread 分布 | min 1.0× · **中位 1.88×** · p90 **9.4×** · max **1824.5×** | 重尾 |
| 最优 plan shape | **traverse_first 33 / 40**（vector_first 2、filter_first 1） | 接近恒优 |
| 硬编码默认计划 | **在 36 条可answered查询上全部达不到质量等价** | 见下 |
| 可答查询 | 36 / 40（4 条被**所有** 96 个计划漏掉） | |

按 `haikaidocs/trevillisPlan.md` §6 P0 的规定：**「spread 中位数 < 2× → 命题 O 不成立，立即重新 scope」**。本轮中位数 1.877×，低于阈值。按约定如实报告，不扩大搜索空间去凑。

---

## 1. 三个必须一起读的限定条件

### 1.1 两个分层给出相反结论 —— 这是最重要的一条

40 条查询里 10 条人工审计、30 条自动派生。分层看：

| 分层 | n | spread 中位数 | 结论 |
|---|---|---|---|
| `manually_audited_v0.1` | 10 | **3.15×** | ≥ 2.0× → 命题 O **成立** |
| `auto_derived_v0.2` | 26 | **1.65×** | < 2.0× → 命题 O **不成立** |

**只看人工审计的那 10 条，结论是反的。** 这正是当初把 `annotation_status` 带进每一行的原因。两种解释都还没被排除：

1. 自动派生偏向了「容易派生」的查询 —— 派生器要求 anchor 名出现在文本里、边类型集 ≤2、答案类型唯一，这些约束可能系统性地筛出了 reach 小、计划空间窄的查询；
2. 人工审计的 10 条本身是当初挑出来展示能力的，天然偏向有意思的查询。

在这一点被查清之前，**1.88× 这个数不应被当成定论**，同样 3.15× 也不行。

### 1.2 中位数掩盖了一条真实的重尾

p90 = **9.4×**、max = **1824.5×**（`stark-prime-036`：5.8 ms → 10,624.8 ms）。也就是说计划选择在**部分**查询上代价极大，只是这类查询在 STARK-PRIME 上不是多数。按 reach 大小切分：

| reach | n | spread 中位数 |
|---|---|---|
| ≤ 20 | 21 | **2.77×** |
| > 20 | 15 | 1.36× |

反直觉：**小 reach 的查询 spread 反而更大**。机理是小 reach 时质量等价集里既有 1 ms 级的 traverse 计划、也有恰好也命中的 ANN 计划（后者要付全库 ANN 的钱），比值被拉开；大 reach 时所有计划都被同一份内在工作量支配，比值收窄。

### 1.3 STARK-PRIME 的 reach 太小，计划空间本来就窄

实测 reach 分布（沿标注的 typed 边 + 类型过滤）：

| | min | p50 | p90 | max |
|---|---|---|---|---|
| hop 1 | 0 | **4** | 385 | 2,130 |
| hop 2 | 1 | **23** | 4,478 | 14,136 |

中位查询只有 4–23 个候选。这个规模下几乎任何计划都是 1–2 ms，被固定往返开销支配 —— **不是优化器无用，而是这个数据集在中位数附近没有给优化器留下余地**。这与 `docs/e0_plan_space_execution_v0.1.0.md` §1.1 已记录的「STARK 只有 1–2 跳」是同一个根因，也再次说明深度/扇出维度必须由 OpenEvolve 树补位（P1）。

---

## 2. 比 spread 更强的一个结果：硬编码默认计划是**错的**，不只是慢

默认计划取 GraphRAG 风格：`vector_first, k=8, hops=2, post-filter`。

**它在 36 条可答查询上，一条都没有进入质量等价集。** 因此「次优比」这个指标算不出来 —— 分母存在，分子却是个答错的计划，比下去没有意义。

机理已单独测过（这也是推翻 D1 的那次测量）：官方答案在**全局精确向量排名**（129,375 个节点）中排在 **4,589 / 11,543 / 15,913 / 18,772 / 55,160** 位。任何 k ≤ 100 的 ANN top-k 都不可能包含它们。anchor 给定时，向量腿是 reach 集**内部的排序器**，不是候选生成器。

这条对 trevillisPlan 的 motivation 而言，比 plan spread 更有力：现有系统的固定计划在这类负载上不是「离最优有多远」的问题，而是**根本答不对**。

---

## 3. 最优 plan shape：traverse_first 近乎恒优

| shape | 赢下的查询数 | 能进入质量等价集的查询数 |
|---|---|---|
| **traverse_first** | **33 / 36** | 34 |
| vector_first | 2 | 11 |
| filter_first | 1 | 6 |

`shape_invariant = false`（不是严格恒优），但 33/36 已经很接近。按 §4 冻结的口径，这项**只作描述性统计，不承担判停职责**。

同时注意：这三种 shape 是在 D1 被测量推翻之后才凑齐的。**如果当初真的只跑 vector_first + filter_first，可行计划数为 0，E0 会产出一张空表。** 决策记录见 `docs/e0_plan_space_execution_v0.1.0.md` §4 与 `configs/e0/plan_space_v0.1.yaml` 的 `shape` 注释。

质量等价计划数中位数为 **8 / 96** —— 96 个计划里只有约 8 个能答对。

---

## 4. 4 条查询被所有计划漏掉

`stark-prime-013` / `024` / `035` / `039`：96 个计划全部 MRR = 0。它们**没有**被计入 spread 分布（对一组全都答错的计划求延迟比值没有意义）。

这不是加载错误 —— Phase B 的对账已用 Cypher 逐条验证过全部 40 条查询的官方答案在图中可达（`tools/e0/load_polyglot.py verify`，6/6 PASS）。所以答案在 reach 集里，只是**向量排序没把它排进 top-1/top-20**。这是排序质量问题，不是计划空间问题，但它同时说明：本实验的质量口径（Hit@1/MRR）受限于 embedding 对这类生物医学文本的判别力。

---

## 5. 环境与方法

- **底座**：Milvus v2.4.5（19530，HNSW/COSINE，1024 维）+ Neo4j 5.20（bolt 7688，129,375 节点 / 8,100,498 关系）+ pgvector pg16（5434，`vector 0.8.6`，HNSW + entity_type B-tree）
- **容器运行时**：**rootless podman**（本机无 docker，且 Ubuntu 24.04 的 `kernel.apparmor_restrict_unprivileged_userns=1` 禁止非特权 user namespace；补 `/etc/subuid` 后可用）。镜像 tag 与 `baseline/docker-compose.yml` 逐一对齐，但运行时不同。
- **全部 `--network=host`**：rootless podman 的端口转发要过 pasta/slirp4netns 用户态代理，会给每次 round-trip 加延迟，映射端口等于人为拖慢基线、制造假胜利。
- **查询 embedding 预先算好并缓存**，任何 shape 都不为 embedding 付费。
- **三条腿全部 client 端计时**（pymilvus / neo4j / psycopg 的 Python wall clock），无服务端计时优势。
- **连接池复用**（`--tuning=tuned`）；`--tuning=naive` 每条腿重连，供对照，可审计。

## 6. 诚实边界

- **本实验不测 TriDB。** E0 的被测对象是 polyglot 基线自身的计划空间。
- **Polyglot-Tuned 是我们自己造、自己调的对手。** `--tuning` 开关保留以供审稿人重跑对照。
- **30/40 的查询是自动派生的**，仅经机器审计（官方答案在标注的 typed-hop 包络内可达，且用官方 `stark-qa` API 独立复验 40/40 通过），**未经人工审计**。§1.1 的分层分歧尚未解释。
- **命题 O 的否证结论不下达。** 本轮触发了 P0 的判停阈值，但 §1.1 的分层分歧、§1.2 的重尾、§1.3 的数据集局限，三者都指向「STARK-PRIME 的中位查询太小，撑不起这个测量」，而不是「计划空间不存在」。正确的下一步是按 trevillisPlan §6 重新 scope 到能提供深度与扇出的负载（OpenEvolve 树，P1），而不是宣布命题 O 死亡。
