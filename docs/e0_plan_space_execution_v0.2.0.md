# E0 双数据集 Plan Spread 实现计划 v0.2.0

## Material Passport

- 阶段：E0 / Optimization Opportunity
- 数据：STARK-PRIME 40 queries；OpenEvolve seed 42 的 10 queries
- 冻结配置：`configs/e0/plan_space_v0.2.yaml`
- headline backend：`polyglot_live`
- correctness backend：`parquet_reference`
- 状态：代码、reference smoke、polyglot live smoke 已完成；完整 10,350 条 run 尚未启动

本版本替代 v0.1 的 stark-only scope，但保留 v0.1 文件作为决策历史。它不修改 TriDB
执行器，也不声称 GX10 sign-off。E0 的目标是先测量计划空间是否存在值得优化的离散度。

截至本版本完成时，两个 live load gate 均已通过：STARK 为 129,375 Milvus rows、129,375
Neo4j nodes、8,100,498 relationships、129,375 pgvector rows，40 条 query 全部可达；
OpenEvolve 为 31 rows/nodes、60 条正反向 lineage relationship、31 pgvector rows。
平衡抽取三个 shape 的 live smoke 共 12 条 observation、0 error；它的
`design_complete=false`，因此不构成 P0 结果。

## 1. 冻结的研究问题与判据

对同一 query，在答案质量等价的计划中定义：

```text
plan_spread(q) = max(plan_p50_latency) / min(plan_p50_latency)
```

先保留达到该 query 最佳 Hit@1（容差 0.02）的计划，再在其中保留达到最佳 MRR（容差
0.02）的计划。Recall@20 和 Hit@5 只记录，不参与等价判定。若两个数据集合并后的 spread
中位数低于 2×，按 `trevillisPlan.md` 的 P0 条件停止并重新审视命题 O。

## 2. 数据与分层

| 数据集 | query | 作用 | 分层字段 |
|---|---:|---|---|
| STARK-PRIME | 40 | 图密、typed relation、外部真实性 | hop、template、annotation_status |
| OpenEvolve | 10 | lineage tree、深度与分支 | hop、template、generation predicate |

STARK 的 30 条自动推导 query 必须和 10 条人工审计 query 分层报告。它们已由独立的官方
`stark-qa` API 复核可达性，但自然语言解释仍不是人工审计，不能混写成同一证据等级。

## 3. 计划空间

- vector-first：全局 ANN 取 k 个候选，再做 typed graph constraint，关系谓词 during/post。
- filter-first：关系谓词先约束候选域，在域内按向量排序，再做 typed graph constraint。
- traverse-first：从 query anchor 做 typed traversal，再应用关系谓词并向量排序。
- STARK：k={1,5,10,20,50,100}，hop={1,2}。
- OpenEvolve：k={1,5,10,20,31}，hop={1,2,3,5}。

`pre` 只属于 filter-first，`during/post` 只属于 vector/traverse-first。不合法组合在枚举期
报错；不能把失败计划静默删除。默认计划是 vector-first、k=8、2-hop、post-filter，它作为
额外计划加入，不污染冻结的 k 梯度。

因此 STARK 每 query 为 60 个网格计划 + 1 个默认计划，共 7,320 条 observation
（40×61×3）；OpenEvolve 每 query 为 100 个网格计划 + 1 个默认计划，共 3,030 条
（10×101×3）。完整 reference/live run 的冻结总数是 10,350 条 observation。

## 4. 两层后端与诚实边界

`parquet_reference` 直接读取真实 Parquet、真实 embedding 和原生 adjacency CSR，用于：

- 验证三个 shape 的结果语义；
- 验证 quality-equivalence、checkpoint 和分析代码；
- 在 CI/开发机运行 smoke。

它的阶段耗时是 Python reference implementation 耗时，绝不能作为数据库系统延迟或论文
headline。只有连接同机 Milvus + Neo4j + pgvector、并记录实际 RPC/bytes/stage timing 的
`polyglot_live` 才能产出系统 plan-spread 结论。

## 5. 实现阶段

1. 查询向量：用 pinned Qwen3-Embedding-0.6B 对两份 query text 生成独立 Parquet 和 receipt。
2. 统一模型：`QuerySpec`、`PlanSpec`、稳定 plan ID、严格配置验证与计划枚举。
3. Reference executor：缓存每 query 的 exact cosine order 和 typed reachability，执行三种 shape。
4. Runner：warmup、repetition、append-only JSONL checkpoint、resume key、输入/配置 hash manifest。
5. Analyzer：raw → per-plan → per-query，计算 spread、default suboptimality 和 shape winner。
6. Figures：spread ECDF、default suboptimality、winner shape、dataset/hop breakdown；reference 图强制水印。
7. Tests：合成图覆盖三种 shape、质量过滤、spread 算法、resume；真实数据只做受限 smoke。

当前 v0.2 已完成 reference 与 `polyglot_live` 路径。live adapter 复用 Milvus/Neo4j/Postgres
连接、批量下推 ID，并记录实际 round trips、bytes shipped 与 stage timing。分析器另有
design-completeness gate：只有两个数据集、完整网格、3 repetitions、10,350 条 observation
全部齐全时才评估 P0；smoke 即使是真实延迟也不会触发判停。

## 6. 产物契约

```text
results/e0/plan_space/<run>/
├── observations.jsonl
├── run_manifest.json
├── metrics_plan.csv
├── metrics_query.csv
├── summary.json
└── figures/
    ├── plan_spread_ecdf.png
    ├── default_suboptimality.png
    ├── winner_shape.png
    └── spread_by_hop.png
```

每条 observation 必须含 backend、dataset、query、plan、repetition、status、quality、latency、
stage latency、intermediate cardinality、round trips 和 bytes shipped。manifest 固定所有输入
SHA256、配置 SHA256、环境和 `valid_for_system_latency_claims`。

## 7. 当前复现命令

```bash
make e0-query-expand
make e0-query-audit
make e0-query-embed
make e0-plan-reference-smoke
make e0-openevolve-polyglot-load
make e0-plan-live-smoke
```

只重新归约已有 checkpoint：

```bash
make e0-plan-reference-analyze
```

完整 10,350 条 live run（长任务、append-only checkpoint，可原命令续跑）：

```bash
make e0-plan-live
```
