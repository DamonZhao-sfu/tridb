# 实现 prompt 0：共享层（数据契约、特征探针、只跑检索）

> 先做这一份。离线和在线两份 prompt 都依赖这里产出的模块。

---

你在 `DamonZhao-sfu/tridb` 仓库的 `experiment-trajectory-retrieval-v4` 分支上工作。任务是实现 TRACE 的共享层：数据契约、特征探针、只跑检索的执行器。之后的离线 profiling 和在线路由都会调用这一层，所以同一个特征定义必须在两边逐字节一致。

## 先读这些

1. `CLAUDE.md`：黄金规则，尤其是 TR-1（算子必须是 Open/Next/Close 的流式算子，不得物化完整中间结果）、只用一个 PostgreSQL 进程、图拓扑走原生邻接 AM（`trimem.expand`），不用关系 join 表来建模。
2. `docs/trace_routing_aware_design_v0.2.0.md`：设计来源。重点是 §4.2 的 r(q,x) 与 §5 的 T0/T1/T2 特征。
3. `docs/decisions/0026-semantic-operator-placement.md`：LLM 语义算子放在进程外，结果缓存存在 PG 表里。
4. `trimem/optimizer/schema.py` 的 `CandidateSpec`、`stable_hash`，以及 `trimem/optimizer/space.py` 的 `SearchSpace.encode`。配置的身份统一用 `CandidateSpec.action_hash`。

## 第一步：找到环境类 agent 的实验代码，找不到就停

ALFWorld、AppWorld、SWE-smith 这类数据集的 profiling 代码和检索入口不在这个仓库里。结果页提到它们使用 `session_retrieval.retrieve`，并由 `trimem/experiments/render_atlas_results.py` 渲染，但这两个文件在本仓库都不存在。

- 在本机搜索 `session_retrieval`、`render_atlas_results`、BO profiling 的归档目录，以及 ALFWorld 经验库的 PG DSN。
- 用一段简短的报告写清找到的路径：检索入口函数的签名、profiling 归档的格式（每行有哪些字段）、经验库 schema（task、session、step 三张表与 has_session、has_step、precedes、similar 等边类型）、隔离规则（检索训练题时如何排除它自己的轨迹）。
- **如果找不到，就停下来报告你搜了哪里，不要猜 schema，也不要编数据。**

## 要实现的内容

新建包 `trimem/trace/`。先确认这个名字没有和现有模块冲突。

### 1. `trimem/trace/contract.py`：逐题结果的数据契约

- `ProfileRow`（frozen dataclass），字段如下：`dataset`、`split`（train 或 test）、`query_id`、`config_hash`（即 `CandidateSpec.action_hash`）、`config`（`to_dict()` 的结果）、`repeat`、`quality`、`agent_input_tokens`、`agent_output_tokens`、`retrieval_llm_tokens`、`retrieval_latency_s`、`generation_latency_s`、`injected_ids`（有序元组）、`injected_hash_ordered`、`injected_hash_set`、`status`、`protocol_id`。
- 两个哈希都要算。注入顺序会影响提示词，所以复用结果时以有序哈希为准；集合哈希只用于分析。
- 为第一步找到的真实归档格式写一个 adapter，把它转换成 `ProfileRow`。不认识的字段原样保留到 `extra`，不要丢弃。
- 校验函数，检查四件事：训练集是否稠密（每道训练题在每个配置上都有记录）、主键 (dataset, split, query_id, config_hash, repeat) 是否重复、usage 缺失（标记为 unavailable，不插补）、同一矩阵内 `protocol_id` 是否一致。
- 导出为 Parquet 或 CSV，同时写一个 manifest，里面记录源文件的 sha256、行数和是否稠密。

### 2. `trimem/trace/features.py`：一份特征定义，离线与在线共用

- 常量 `FEATURE_VERSION`，再加 `feature_schema_hash()`：对特征名列表、各个参数（τ、k、fanout cap）以及本模块源码求 sha256。这个哈希之后会写进路由器产物，在线加载时要逐一核对。
- **T0**（不访问库）：题干 embedding 降维，用 PCA 降到 16 到 32 维，维度做成参数。PCA 只能在训练集上拟合，拟合结果序列化保存。另外加 task family 的 one-hot 和题干长度。
- **T1**（一次 HNSW 探针）：在 task 或 session 的向量上取 top-k（默认 16）。输出 top-1 相似度、top-k 平均相似度、相似度 ≥ τ 的 session 数（τ 可配置，默认试 0.7 和 0.8 两档），以及 top-k 中成功 session 的比例。
- **T2**（关系过滤加一次 `trimem.expand`）：从 T1 的种子出发，沿一跳 `similar`（边类型做成参数）扩展，fanout 设上限。输出：关系过滤后的候选池大小、池内不同 task family 数、深度 2 与深度 1 的池大小之比（深度 2 也必须有上限）、种子平均出度。
- **硬约束**：
  - 探针必须用和正式检索完全相同的隔离规则，训练题的探针不能看到它自己的轨迹，否则特征会泄漏标签。
  - 只读事务，设 `statement_timeout`，所有扩展都带 LIMIT 或 fanout cap，绝不调用 LLM。
  - 结果确定：同一个库、同一个 query 必须得到同样的特征。
  - 每次探针都记录 T1 和 T2 各自的耗时。
- 唯一入口：`compute_features(conn, query, *, fitted) -> FeatureVector`，同时返回原始统计和最终特征。离线训练和在线路由都只能调用这一个函数。

### 3. `trimem/trace/retrieval_only.py`：只跑检索，不跑 agent

- `retrieve_only(conn, query, config) -> RetrievalOutcome`：调用第一步找到的真实检索入口，执行该配置的检索，但不启动 agent。
- 输出字段：有序注入 id、两个注入哈希、注入 token 数、注入 session 的平均相似度、注入 session 中成功的比例、检索耗时。
- `r_features(outcome, profiled_outcomes_same_query)`：在上述字段之外，再算与同一题上已 profile 配置的注入集合的最大 Jaccard 重合度。
- SemFilter 或 Rank 配置的处理：
  - 如果 judge 结果按 (query, candidate session) 缓存在 PG 里，就直接读缓存。
  - 缓存未命中时，不调用 LLM，返回过滤前的候选池特征，并标记 `semantic_unresolved=True`。
  - 只有显式传入 `allow_semantic_calls=True` 时，才允许调用进程外的 judge。

### 4. CLI

`python -m trimem.trace probe|retrieve-only|export-matrix ...`，用 argparse，每个子命令都要有 `--help`，并把输出写到用户指定的目录。

## 测试与验收

- `tests/test_trace_contract.py`、`tests/test_trace_features.py`、`tests/test_trace_retrieval_only.py`。用 fake connection 或现有的 private-PG 测试夹具，参考 `trimem/optimizer/private_pg.py`。
- 必须覆盖以下情况：
  - 稠密校验能抓出缺格。
  - 注入哈希稳定，且顺序敏感。
  - 训练题的探针看不到自己的轨迹，用一条隔离测试来证明。
  - 同样输入下特征结果相同。
  - `feature_schema_hash` 随参数变化而变化。
  - SemFilter 缓存未命中时不发出任何 LLM 请求。
- `make test` 和 `make lint` 都要通过。
- 如果能连上真实的 ALFWorld 经验库，在 50 道训练题上测探针耗时，报告 p50 和 p95，目标是 p95 低于 50 ms。连不上就如实写明未测。

## 不要做的事

- 不改变任何现有实验的默认行为。不碰 `trimem/optimizer/search.py` 的 qLogNEHVI 路径。
- 不启动 GPU 模型，不跑 agent episode，不调用生成 LLM。
- 不把任何 test split 的结果写进训练用的矩阵。

提交信息格式为 `feat(trace): ...`，并推送到 `experiment-trajectory-retrieval-v4`。最后用一段话报告：做了什么、测了什么、哪些没测以及原因。
