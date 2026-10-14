# TriMEM × EXG QA pilot：启动协议、SQL 与指标

日期：2026-10-11。承接 `trimem_exg_qwen38_experiment_plan_v0.1.2.md`，记录用户授权启动后实际采用的 pilot 设置；不改写原讨论草案。Origin Skill: academic-research-suite / experiment-agent。状态：真实模型 pilot 已启动，正式 500 题批次尚未启动；尚无有效性或性能结论。

## 数据和任务

MuSiQue Ans v1.0 train、HotpotQA distractor train 分别按 task ID 排序后以 seed 42 打乱，各预留 500 题。在线使用全部 500；离线前 350 题为 collection，后 150 题为 test。当前 pilot 各取随后 4 题，与正式题目不重叠。

输入只有 question 和给定 passages；输出为 `{"answer": "短答案"}`。标准答案及别名放在独立 evaluator 文件中，不进入生成、embedding、semantic judge 或经验图。评测器计算归一化 EM/F1，EM 正确即停止，错误或非法输出最多重试一次；反馈只含 incorrect，不含答案或 F1。MuSiQue 的完整官方 scorer 对照仍是正式运行前的检查项。

## 实际启动

- 实现：`trimem/experiments/exg_pilot.py`。
- 输出：`results/trimem_exg_qwen38_v0.1.0/pilot_20261011_01/`；启动 PID/命令在相邻 `launch.json`。
- 每个数据集 4 题 × 5 组，总计 40 个 task episodes，最多 80 次 solver 调用，semantic 调用另计。
- 组别：T（三模态）、N（无 memory）、SB/ST/SA（原生 SemFilter 的 base/top/auto 放置）。这些是 pilot 放置探针，尚不等同于草案正式 S0/S1/S2。
- 模型：本地 `Qwen/Qwen3.8-27B-FP8`，服务实际 context 32768；generation thinking on、temperature 0.7、top_p 0.95、seed 42、max_tokens 8192。未验证实际加载权重 revision，不把历史 262K alias 当作当前配置。
- Semantic 同一模型，thinking off、temperature 0、max_tokens 128；逐请求保存完整 request/response/usage。没有隐藏传输重试；HTTP 失败停止并留存失败记录。
- Embedding：本地 `Qwen/Qwen3-Embedding-0.6B`，1024 维，输入为 question；query embedding 复用于该题 case 写入，成本只记一次。
- 数据库：独立 PostgreSQL 16.14 实例，127.0.0.1:55442，x86_64、8KB page；每组新建数据库。已有 55441 实例的 TriMEM mailbox 在实例内共享，所以没有向该实例附加本轮 worker。每次仅运行一个本轮 semantic worker。
- GPU 服务仍共享现有双 RTX PRO 6000 服务；本轮延迟仅用于管线预检，不能作为正式性能比较，也不代表 GX10 fork 结果。

## 检索 SQL 的变化范围

所有题复用同一参数化主干：HNSW top-10 → 原生邻接扩展 → metadata lookup → 可选 SemFilter → 消费 hints。图拓扑由 `trimem.expand` 读取，SQL JOIN 连接的是 case 属性，不是关系边表。

每题替换 query vector、sequence 可见性边界以及 semantic predicate 中的问题文本。SB/ST/SA 的 SQL 逻辑相同，只改 `trimem.placement`；auto 目前选择的是语义算子位置，并不意味着已经实现下游生成成本预测。T 去掉 semantic predicate；N 无检索。

真实查询通过 named server cursor 逐行消费，在获得 5 条不重复 hints 或耗尽候选后关闭，hint 合计上限 2048 tokens。`EXPLAIN` 没有 ANALYZE，避免为测量重复调用模型；计划中出现 Sort、Hash Join、Hash、Aggregate、Materialize、Function Scan 即拒绝执行。当前 min/max semantic batch 均为 1。

SQL 审计快照：

- `results/trimem_exg_qwen38_v0.1.0/pilot_20261011_01/sql_audit/all_32_retrieval_queries.sql`：两套数据各 4 题 × 4 个 memory 组的全部计划语句；未计算的 vector 保留绑定占位符。
- `executed_retrieval_queries.sql`：快照时已完成检索的语句，包含实际向量值。
- `bindings_and_status.json`：题目、参数、状态、快照时间。pending 不是已执行记录；这些文件是快照，不会随后台进程自动刷新。

合成预检中，base 的 SemFilter 位于 metadata lookup 上方、Nested Loop 内侧；top 位于整个 Nested Loop 上方；auto 选择 top。这只验证位置开关生效，不证明正式数据上的速度收益。

## 与 EXG 及正式草案的差异

当前 pilot 使用流式遍历顺序，不做 EXG 的全局 relevance rerank；seed 本身也进入候选。LIMIT 30 约束的是去重前候选行数，消费者随后去重，不等同于 EXG 去重后 30 个 case。similar_to 连接最多 10 个过去近邻，没有相似度阈值；包含双向物理邻接。当前没有 anchor bridge 检索、failure-aware 向量重排、SemExtract 或 insights。

每次真实 attempt 都写 case 和 contain；第二次通过时，第一次失败到第二次成功写 fixed_by。当前 pilot 每题只检索一次；第二次尝试保留本题首次输出、错误反馈和首次 hints，没有再从图读取自身失败 case。正式协议是否采用逐 attempt 重检索需要在后续版本明确冻结。

这些边界意味着本轮是原生管线冒烟测试，不能标为论文算法的严格复现，或用于宣称等价物理计划的 quality 差异。

## 报告指标及口径

[EXG §4.1.2、§4.2–4.3、Appendix C](https://arxiv.org/pdf/2605.17721) 报告 pass@1、pass@2、平均 LLM calls、LLM inference latency、retrieval latency，并在附录分析 input/output token 和在线曲线。我们的核心指标与这些维度对齐，不要求数值相同。

| 指标 | 本实验口径 |
|---|---|
| pass@1 | 首次 attempt 成功的任务比例；QA 以 EM 为准 |
| pass@2 | 至多一次反馈重试后成功的任务比例；不是两份独立采样的无偏 pass@k 估计 |
| LLM calls/task | 全部真实 generation + semantic 调用数除以任务数；embedding 另列 |
| LLM latency | 分别记录 generation 与 semantic API 请求耗时；包含服务排队/网络，不能直接称纯 GPU 推理时间 |
| retrieval latency | 本轮含 query embedding、数据库/semantic 等待和 hint tokenization；正式表同时需要 DB-only 分项 |
| input/output/total tokens | 服务实际 usage，含 reasoning 被后端计入的 completion；按 generation/semantic 分项并汇总总量及每题均值 |
| retries | 每题重试次数、首次失败后的修复率；格式错误与错误答案分别保留 |
| QA EM/F1 | 补充 answer 质量，不声称 EXG 已单独报告同样 F1 表 |
| construction | 每次 case/向量/图关系发布耗时；后续加派生和索引构建，包含 collection 总成本及摊销成本 |
| end-to-end | 每题实际墙钟时间，涵盖读取、生成、评测、写入、编排；报告均值、p50、p95 |
| graph growth / operator work | 从写入收据统计 case/typed edges，并在正式版本补采样快照、ANN/Expand 消费量、SemFilter 行数及缓存命中 |

retrieval 已包含 semantic 等待，不能再把 semantic latency 加一次当 end-to-end；用实际 wall time 校验。不能只报 DB retrieval 节省而漏掉 semantic token、重试或构建成本。离线报告 collection/test 分开，以及全生命周期成本；本次在线 pilot 尚未启动离线 collection。

## 启动检查

9 项 Python 协议测试通过：首答正确停止、错误最多重试一次、gold 不进入请求、阻塞计划拒绝。独立数据库合成预检通过空图读取、两次 case 提交、fixed_by 边、流式 retrieval 和三种 native semantic placement；证据在相邻 `preflight.json`。真实 Qwen 的首个生成请求已返回并保存 usage，后续任务后台顺序执行。完整模型任务结果以 complete/failed 标记为准。
