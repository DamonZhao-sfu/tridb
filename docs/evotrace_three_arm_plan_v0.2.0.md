# EvoTrace × GEM × OpenEvolve — 三方对比实施方案 v0.2.0

- 日期：2026-08-24
- 状态：**方案，未实施**
- 前一版：`evotrace_three_arm_plan_v0.1.0.md`
- 依据：`haikaidocs/benchmarkAnalysis.md` 附录 C（OpenEvolve 源码审计）与附录 D（指标溯源）

## 相对 v0.1.0 的变更

| 变更 | 内容 | 后果 |
|---|---|---|
| **seed 数 3 → 1** | 每个 (task, arm) 只跑一次 | **失去误差棒**。arm 间差异不得称为显著，见 §6.1 |
| **删除 Arm D (Oracle)** | 不再注入历史最优 | **B ≈ A 时产生不可解歧义**，见 §6.2 |
| **删除 Arm E (Poisoned)** | 不再注入已知死枝 | 由 Phase 0 闸门 6 的 **prompt 断言**替代，见 §2 |
| 规模 | 99 runs → **24 runs** | 估算 wall-clock 14–17h → **4–5h**（两卡） |

---

## 0. 指标口径

**Agent 指标 —— 只报两个，均为 AlphaEvolve 原生：**

1. **best-so-far vs 计算预算 曲线**（AlphaEvolve Figure 8 形状）
   x = 计算预算（iteration 数；另出一版以累计 LLM token 为 x），
   y = 目标 metric，跨任务平均，每 arm 一条曲线。
   **单 seed，无 CI 阴影**（v0.2.0 变更）。
2. **per-task 最佳分**，跨任务聚合用**分类率**而非求平均：
   `超过历史最佳 / 达到（evaluator 容差内）/ 未达到` 三档各占多少任务。
   （照 AlphaEvolve 的 "75% rediscovered, 20% discovered new" 做法，回避不可通约的分数平均。）

**不报**：best-so-far AUC、iterations-to-threshold —— 非 AlphaEvolve 原生指标。

**检索指标**（第一类）不变：`gem_eg/metrics.py` 的 ndcg / oracle_recall / harmful@k / parity。
两类指标**分表报告**。

---

## 0.5 术语：什么是 task，什么是「一次查询」

本文里 **task = 一道题**，**查询 = agent 每改一次代码之前，问记忆库的那一次「有没有类似的」**。
两者常被混淆，此节固定口径。

### task

一道题。例如 `circle_packing`：

> 在单位正方形内放 26 个圆，最大化半径之和。（AlphaEvolve 达到 2.635）

Phase 1/2 用 **7 道 math 题**，即 7 个 task。

### 一次查询

OpenEvolve 解题的方式是把程序改 100 次。**每次改之前发一次查询**：

> 「我在做 circle_packing，当前程序 2.41 分，历史上有没有更好的类似做法？」

**输入**（三部分同时给）：

1. 该 task 的规格向量（`task_spec` track，用于 ANN 找相似 task）；
2. 起点 vertex（ANN 命中的 task）；
3. 下推谓词，例如 `kind='node' AND is_valid AND fitness >= 2.41 AND session_uid <> :held_out`。

**执行**：`tjs_open` 两段 —— seedless ANN 找相似 task，再 filter-first 从该 task 沿
`eg_hier` 走 2 跳（task → session → node），逐顶点求值谓词，bounded top-k、流式早停。

**输出**：N 段历史代码（`num_diverse_programs` 条），按向量距离排序，注入 prompt 的
`{inspirations_section}`。

**因此：1 道题 × 100 轮 = 100 次查询。**

### 两个 Phase 里「查询」的来源不同

| | **Phase L**（测延迟） | **Phase 1/2**（测 agent） |
|---|---|---|
| 查询来源 | **历史回放**。`decision_points.jsonl` 记录了 18,291 个真实决策点，每个都是当年 agent 真要改代码的时刻，重放为一次查询 | **实时**。agent 真在跑，每轮真发一次 |
| 是否跑 LLM | **否**，纯数据库，不占 GPU | 是 |
| 查询量 | 分层抽样（见 §1.4） | 100 轮 × 7 题 × 3 arm = **2,100** 次 |
| 观测 | first-row latency、做功量、parity | best-so-far 曲线、per-task 最佳分 |

**为什么分开**：Phase L 能量出「GEM 比 Polyglot 快多少」，但**快不等于 agent 解题更好**；
Phase 1/2 能回答后者，但慢、贵、且随机。两类结论分表报告（§7.4）。

---

## 1. Phase L —— 检索 latency 对比（不需要 GPU，先做）

### 1.1 三个被测系统

| 系统 | 实现 | 定位 |
|---|---|---|
| **TriDB/GEM** | `bench/agent_memory/gem_eg/query.py`，两次 `tjs_open`，一个事务 | 主张 |
| **Polyglot** | `experiments/e0/plan_spread/live_backend.py`（Milvus + Neo4j + pgvector） | **同检索语义、异物理系统** → 干净的系统对比 |
| **Cognee** | `bench/agent_memory/table5_track_c/adapters/cognee.py` | **异检索语义** → 现成方案基线，单独成行 |

> Polyglot 与 GEM 应产出**相同的结果集**（quality 相同），差异只在延迟与做功量。
> Cognee 会用 LLM 把代码抽成知识图谱，语义不同，quality 必然不同，**不得与前两者同表比 latency**。

### 1.2 指标

- **first-row latency（主）** —— 卡点在「第一个系统查完才能喂第二个」，总时长会掩盖它
- total latency to top-k
- 做功量：`tjs_open_candidates_examined() / graph_examined() / graph_censored() / termination_reason()`
- 等召回校验：Polyglot 与 GEM 的结果集必须一致（`parity()` 的 exact / recall / tie_equivalent）

### 1.3 前置闸门（**上次撤回就是因为缺这两条**）

2026-08-18 commit `b3b4e6d` 整体撤回了 e0 的 polyglot 数字：1,010 个 cell 全部返回空
`result_ids`（loader 在 run 结束后 2 分钟才完成），且产出该数字的 `live_backend.py` 从未提交进 git。

因此本次强制：

1. **加载完成回执**：每个系统必须先写出 `load_receipt.json`（行数、耗时、xid），
   测量脚本读到回执才允许开测；
2. **非空断言**：任何 arm 在任何 cell 返回空结果集 → **立即 fail**，不许静默计 0；
3. **代码版本固定**：测量前 `git rev-parse HEAD` 写进结果目录，工作区必须 clean。

### 1.4 规模与抽样

`decision_points.jsonl` 共 **18,291** 个决策点，四个查询形状：

```
W1.a  10,479   （reuse：改代码前找可复用的历史尝试）
W1.b   6,525   （stuck：卡住时找突破口）
W1.b-fail 668  （repair：失败后找修复路径）
W1.d     619   （path：重放整条谱系）
```

**其中属于 Phase 1/2 那 7 道 math 题的只有 8,317 个**，其余约一万个属于 ALE 的 10 道 C++ 题。
7 道题内部分布极不均衡：

```
second_autocorr_ineq   3,294        third_autocorr_ineq      625
heilbronn_triangle     2,763        heilbronn_convex_13      485
circle_packing           574        first_autocorr_ineq      363
                                    uncertainty_ineq         213
```

**前两道题占 73%。因此抽样必须按 (task, 查询形状) 分层**，否则聚合结果实质上只反映
`second_autocorr_ineq` 和 `heilbronn_triangle` 两道题。

目前 W1 只测过 60 个决策点，Phase L 应扩到每层至少数十个并报告每层的 n。

> Phase L 与 seed 数无关 —— 检索是确定性的，重复运行只影响延迟测量的稳定性，
> 按常规做多轮取分位数即可，不受 §6.1 的单 seed 限制影响。

### 1.5 Cognee 的额外成本

实测建库速率 1.17 条/秒 → 10,672 份代码约 **2.5 小时** + LLM 调用开销。
需在 Phase L 之前单独排期，且要占用 GPU（与 Phase 0 的 qwen3.8 部署冲突，二者串行）。

---

## 2. Phase 0 —— agent 实验的前置闸门

1. **qwen3.8 部署实测**
   `--max-model-len` 32,768 → **98,304**（实测 prompt+completion p99 = 73,702，
   当前配置会让 25.10% 的调用放不下）；`--kv-cache-dtype fp8`；
   tp=1 × 2 副本（无 NVLink，不用 tp=2）。用真实 prompt 长度分布回放 50 条，
   量实际 tok/s 与峰值显存。**先清空 GPU**（当前 GPU0 76 GB / GPU1 92 GB 被占）。
2. **evaluator 复现闸**
   取每个 math task 的历史最优程序，用其 run-local `evaluate.py` 重跑，**分数必须复现**。
   不复现的 task 出局 —— 评测契约不成立时任何 outcome 都无意义。
3. **cascade 一致性**
   确认所有 arm 的 `cascade_evaluation` / `cascade_thresholds` 相同，且 evaluator 确实定义了
   `evaluate_stage1`（否则会静默退化成直接评估，只打一条 warning）。
4. **泄漏审计**
   source/target artifact SHA 去重；target session 的所有 Node 从 memory 中排除。
   语料约 5% 跨 run 内容重复，不审计会直接泄漏答案。
5. **token 配平核对**
   记录每 arm 每轮实际 prompt token，两两之差须在 ±5% 内。
6. **注入通道断言（v0.2.0 新增，替代已删除的 Poisoned arm）**
   删掉 Arm E 之后，没有任何 outcome 证据能证明「注入的东西模型真的看到了」。
   改用一条**确定性断言**，成本近似为零且比看 outcome 更直接：

   - 打开 `prompt.log_prompts`（`DatabaseConfig.log_prompts` 默认已 True），落盘渲染后的 prompt；
   - 对 Arm B / C 的**每一轮**断言：本轮注入的 program id（或其代码的前 N 字节指纹）
     **确实出现在** `{inspirations_section}` 渲染结果中；
   - 任一轮断言失败 → **立即 fail**，不得继续。

   要防的三个真实失败模式：
   (a) prompt 超长被截断；
   (b) `sampler.py:449-455` 的去重把外部条目当成重复丢弃；
   (c) 外部 program 没进 `db_snapshot["programs"]`，worker 端
       `[programs[pid] for pid in inspiration_ids if pid in programs]` 静默过滤掉
       （注意这行的 `if pid in programs` 是**静默**的，不会报错）。

   > 这条断言证明的是「注入到达了 prompt」，**不能**证明「模型受其影响」。
   > 后者只有 Poisoned arm 能证明，本版已删除。

---

## 3. Arm 定义

所有 arm 共享同一 `num_diverse_programs = N`，**只换来源，不改条数**：

| Arm | inspiration 来源 | 作用 |
|---|---|---|
| **A** no-reuse | N 条全部来自本 run 的 database（stock OpenEvolve） | 基线。**注意这不是「无上下文」** |
| **B** GEM | N 条替换为 GEM 跨 session 检索（`tjs_open`） | 主张 |
| **C** Polyglot | N 条替换为 Polyglot 检索（同语义、异系统） | 系统对比 |

**必须是「替换」不是「追加」。** stock OpenEvolve 每轮已注入
`num_top_programs=3 + num_diverse_programs=2` 共 5 个自身 run 的程序。
若 B/C 为追加，prompt 变长，outcome 差异可能纯粹来自上下文长度。

**B 与 C 的预期关系**：二者检索语义相同，**结果集应当一致**。
因此 agent outcome 上 B ≈ C 是**预期结果，不是负面结果** —— 它们的差异应当出现在
Phase L 的延迟上，而非 agent 指标上。若 agent outcome 上 B ≠ C，
说明二者的结果集实际不同，应回到 Phase L 的 `parity()` 检查找原因。

### 3.1 接入点

`process_parallel.py:865`：

```python
parent, inspirations = self.database.sample_from_island(
    island_id=target_island,
    num_inspirations=self.config.prompt.num_diverse_programs)
```

做法：子类化 `ProgramDatabase` 覆写 `sample_from_island`，把外部 program 追加进返回的
inspirations，同时写进 `self.programs` 以便进入 `db_snapshot`。外部条目必须打 metadata 标记，保证：

1. **永远不被选为 parent**（否则演化直接从别人的代码继续 —— 那是 seeding，不是 memory）；
2. 不计入 island 成员 / archive / fitness 统计 / MAP-Elites 网格；
3. 在 evolution trace 里可区分，事后可审计注入来源与 receipt。

---

## 4. 任务集合

`find data/evotrace/raw -name evaluate.py` = 18 个，全在 `shinkaevolve/` 下，覆盖 **17/18 个 task**
（缺 `math:signal_processing`）。

| 域 | 数量 | 可跑性 |
|---|---:|---|
| **math** | 7 | ✅ 直接跑。仅依赖 numpy/sympy/stdlib；文件头 `sys.path.insert('/home/user/anon/skydiscover/...')` 是匿名化残留，实际未从该路径 import 任何符号。入口 `evaluate(program_path)` 即 OpenEvolve 标准接口；`circle_packing` 自带 `evaluate_stage1/stage2` |
| **ale** | 10 | ⚠️ 需 `ale_bench` 包 + `ale-bench-lite-problems/<task>` 测试数据 + C++ 工具链（来自 SkyDiscover，rev `59643ef8`） |

**Phase 1/2 只做 math 7 个**：`circle_packing`、`heilbronn_triangle`、`heilbronn_convex_13`、
`first/second/third_autocorr_ineq`、`uncertainty_ineq`。ale 放 Phase 3。

---

## 5. 分阶段

| Phase | 内容 | 规模 | GPU |
|---|---|---:|---|
| **L** | 检索 latency 三方对比（GEM / Polyglot / Cognee） | — | 仅 Cognee 建库需要 |
| **0** | 六道闸门（含新增的注入断言） | — | 需要（部署实测） |
| **1** | 冒烟，`circle_packing`，arm A/B/C × 1 seed | **3 runs** | 需要 |
| **2** | 主实验，7 task × arm A/B/C × 1 seed | **21 runs** | 需要 |
| **3** | ale 扩展（可选），需先验证 ale-bench 复现历史分数 | — | 需要 |

**Phase 1 门槛（全满足才进 Phase 2）**：

- 三个 arm 均跑完 100 iterations 且无 crash；
- 闸门 5 的 token 配平在 ±5% 内；
- **闸门 6 的注入断言逐轮通过**（B 与 C 各 100/100 轮）；
- B 与 C 的注入内容一致率报告出来（预期高；不一致说明 Phase L 的 parity 有问题）。

> Phase 1 在 v0.2.0 里从「manipulation check」降格为**纯冒烟测试** ——
> 删掉 Arm D/E 后它不再具备诊断能力，只验证管道通不通。

**预算估算（算术推演，未实测）**：Phase 1+2 = 24 runs × ~17–20 min，
两卡 data-parallel ≈ **4–5 小时 wall-clock**。此数建立在未实测的吞吐推算上，
Phase 0 实测后可能显著变化。

---

## 6. v0.2.0 引入的两个限制（必须写进报告）

### 6.1 单 seed —— 不得声称显著性

演化搜索本身随机（`Config.random_seed` / `DatabaseConfig.random_seed` /
`LLMModelConfig.random_seed` 三层均可固定，但固定 seed 只保证**可复现**，
不代表该 seed 的结果**有代表性**）。

单 seed 的后果：

- Figure-8 曲线**只有单条线，无 CI 阴影**；
- arm 间的任何差异**不得使用「显著」「优于」等推断性措辞**，
  只能写成「在本次单 seed 运行中观察到」；
- 若 A/B/C 三者差距小于任务本身的 run-to-run 方差，本实验**无法给出结论** ——
  而单 seed 恰恰无法估计该方差。

**补救路径（按成本排序）**：
1. 只对差距最大的 1–2 个 task 补跑 2 个额外 seed（+ 4–6 runs）；
2. 用 EvoTrace 历史数据估计 run-to-run 方差作为参照 ——
   语料中同 (task, backend) 有多个 session（如 `second_autocorr_ineq` 有 24 个），
   其 best_fitness 的离散程度可作为方差的**外部先验**，零额外计算成本。
   **注意**：那些 session 用的是 deepseek-reasoner 等模型，不是 qwen3.8，
   只能作为量级参照，不能当作我们的误差棒。

### 6.2 无 Oracle arm —— B ≈ A 时的不可解歧义

删除 Arm D 后，若观察到 B ≈ A，以下两种解释**无法区分**：

- 记忆对该任务本就无用（天花板低）；
- 记忆有用，但我们的检索没能捞到有用的东西。

这两个结论对项目方向的含义完全相反。

**触发条件与处置**：一旦 Phase 2 出现 B ≈ A，**第一优先动作是补 Arm D（Oracle）**，
而不是去调检索参数 —— 在天花板未知的情况下调参无法判断是否有效。
Arm D 的成本与 B 相同（7 runs），且不需要新代码：注入内容改为
「该 task 历史 best_fitness 最高的 N 个 valid 且非死枝节点，排除 held-out session」，
数据在 `gem_eg_node` 里直接可查。

---

## 7. 必须同时出现的声明

1. 我们的 fitness **不能与 EvoTrace / AlphaEvolve 论文数字对比** —— 原 trace 的模型是
   deepseek-reasoner(76%) / gemini-3-flash / claude / gpt-5，**无 qwen3.8**；
2. thinking 开关状态（EvoTrace 实测 reasoning tokens p50 1,941 / p99 16,384，影响极大）；
3. Cognee 与 GEM/Polyglot **检索语义不同**，不得同表比 latency；
4. 检索指标与 agent outcome **分表报告** —— 数据库更快推不出 agent 更聪明；
5. Arm A **不是「无上下文」**，它是「只有本 run 上下文」；
6. **单 seed**，无误差棒，差异不得称为显著（§6.1）；
7. **无 Oracle 上界**，记忆天花板未知（§6.2）。

---

## 8. 已知风险

| 风险 | 表现 | 处置 |
|---|---|---|
| **单 seed 无法判断差异真伪** | A/B/C 差距小于 run-to-run 方差 | §6.1 补救路径 |
| **B ≈ A 歧义** | 无法区分「记忆无用」与「检索不行」 | §6.2，补 Arm D |
| 注入通道惰性 | 外部条目未进 prompt | Phase 0 闸门 6 的逐轮断言 |
| evaluator 不复现 | 历史最优重跑分数不符 | Phase 0 闸门 2，该 task 出局 |
| polyglot 空结果复发 | 结果集为空却计 0 分 | Phase L 闸门 1+2 |
| cascade 静默退化 | evaluator 无 `evaluate_stage1` | Phase 0 闸门 3 |
| 泄漏 | 检索到的代码与 target session 逐字节相同 | `is_trivial_hit()` 已实现，单独计数并从 headline 剔除 |
| task 数偏小 | 7 个 math task，跨任务统计效力有限 | 如实声明；Phase 3 扩到 17 个 |

---

# 附录 A — Phase P 与 Phase 0 的执行记录（2026-08-25）

本节记录已实际执行的部分，以及执行中发现、原计划没有预料到的约束。
所有"发现"均为实测，不是推断。

## A.1 Phase P 完成，两处修复

| 修复 | 位置 | 效果 |
|---|---|---|
| 真题面替换占位串 | `tools/evotrace/normalize.py:task_specification()` | 18/18 task,114,270 字符,`specification_source=prompt_system_message`。跨 backend 一致性断言:`task_specification_divergence` reject = 0，即同一 task 的所有 backend 记录的 system message **逐字节相同** |
| node 顶点补下推列 | `bench/agent_memory/gem_eg/load.py:_load_node()` | 10,672 个 node 顶点的 `backend` / `domain` 之前**全为 NULL**，任何 `filter="backend=..."` 下推谓词都匹配不到任何东西 |

**loader 可复现性已证明**：干净装载进 `evotrace_eg_v2` 与原地 backfill 的 `evotrace_eg`
在 vertex（非向量列）/ task / session / node / edge 五张表上**逐行哈希相同**。

## A.2 向量修复的效果（W1.a，1,000 点抽样 vs 修复前全量）

| arm | 指标 | 修复前 | 修复后 |
|---|---|---:|---:|
| **no_graph**（纯 ANN） | harmful@k | **0.7151** | **0.2127** |
| | nDCG | **−0.0154** | **0.1409** |
| **fused** | nDCG | 0.2714 | 0.3157 |
| | harmful@k | 0.2084 | 0.1523 |
| no_vector / filter_only | 全部 | — | 基本不变（不用向量，应有的对照） |

`no_graph` 修复前 nDCG 为**负**，意味着占位串向量检索回来的东西**比不检索更糟**；
harmful 0.715 即 top-10 里七成是已知死枝。这是"占位串 embedding 不只是没用，而是有害"的直接证据。

> ⚠️ 抽样口径不同（1,000 vs 全量 10,479），差异混入了抽样噪声。
> 全量重测在 `results/e2/w1/phaseP_full/`。

## A.3 新增闸门：向量一致性（本轮发现的隐患）

引擎从 `gem_eg_vertex.embedding` 读向量，oracle 从 `data/evotrace/normalized/vectors.npz`
读向量。**同一份数据两个真相来源，没有任何东西绑定它们。**

实测后果：重新 embed 进库但忘记跑 `export_vectors.py`，W1.a 的
`parity_exact_rate` 从 0.9634 塌到 **0.000**、`oracle_recall` 从 0.996 塌到 **0.094**，
**全程没有任何报错**。而 parity gate 抓不到它——parity 正是被破坏的那个量。

处置：`experiments/e2/w1_runner.py:_assert_vectors_agree()`，测量前逐条比对并 fail closed。
已注入不一致自测通过。

## A.4 Phase 0 闸门执行状态

| 闸门 | 状态 | 结果 |
|---|---|---|
| 1 吞吐 | ✅ | prompt **3,448 tok/s**，p50 延迟 10.31 s，48/48 成功。**该负载 prefill 主导**：464,601 prompt token ÷ 3,448 = 134.7 s ≈ 整段 wall time，因此实测到的 148.8 completion tok/s **不是解码能力上限**，不可用作容量估算 |
| 2 evaluator 复现 | ⏳ | 见 A.5 |
| 3 cascade 一致 | ✅ | `run_arm.FROZEN["cascade_evaluation"]=False`，跨 arm 冻结 |
| 4 泄漏审计 | ❌ 未做 | |
| 5 token 配平 | ✅ | 代码保证 + 单测 `test_injection_replaces_rather_than_adds` |
| 6 注入断言 | ✅ 工具就绪 | `tools/evotrace/gate_injection.py`，待真实 run 的 trace |

## A.5 numpy 版本的三方冲突（原计划未预料）

| numpy | 2-D `np.cross`（语料程序需要） | jax（4 个 evaluator 需要） |
|---|---|---|
| 1.26.4（语料原生，`.venv-e0` 原状态） | ✅ | ❌ `numpy.dtypes has no attribute StringDType` |
| **2.4.6（已固定）** | ⚠️ 可用，仅 DeprecationWarning | ✅ |
| 2.5.1（仓库 `.venv`） | ❌ **已移除** | ✅ |

发现路径：`heilbronn_convex_13` 的评分返回 0.0，错误信息是
`Both input arrays must be (arrays of) 3-dimensional vectors`。真因**不在 evaluator，在被评的
历史程序里**——它调了 2-D `np.cross`。语料中 **210 / 5,403** 个 math 程序用到该写法。

**只有 numpy 2.4.6 能让 7 个 math evaluator 全部运行**，已固定在 `requirements-eval.txt`。

必须在报告中声明的推论：
1. 我们的复现分数产自与语料**不同的 numpy**，`gate_evaluator` 量化了漂移幅度；
2. 若将来升到 2.5.x，arm B 注入的历史程序会因 API 移除而失败，**arm B 会因为代码过时而
   显得更差，与记忆质量无关**。这是一个真实的潜在混淆项，靠版本固定规避。

## A.6 evaluator 复现的判定口径被修正

原口径是单一阈值 `rel_delta ≤ 1e-6`，注释写着"确定性构造器产出的 float64 分数"。
**该假设只对其中两个 evaluator 成立。**

| evaluator | 类型 | 实测 rel_delta |
|---|---|---:|
| `circle_packing` | 确定性几何构造 | **0.00e+00** |
| `heilbronn_triangle` | 确定性几何构造 | **0.00e+00** |
| `second_autocorr_ineq` | jax/optax 数值优化 | 1.13e-05 |
| `first_autocorr_ineq` | jax/optax 数值优化 | 4.97e-05 |
| `third_autocorr_ineq` | jax/optax 数值优化 | 1.00e-03 |
| `uncertainty_ineq` | jax/optax 数值优化 | 4.59e-03 |

优化器结果依赖库版本与浮点求和顺序，1e-5 量级漂移是数值噪声，不是"评测契约不成立"。
改为三分类：`reproduced`（精确）/ `reproduced_within_tolerance`（默认 1e-2）/ `drifted`。

**报告约束**：对后四个 task，我们的 fitness 与语料记录值的比较**分辨率不得优于其漂移量**。

## A.7 任务集合的实测收缩

| task | 状态 |
|---|---|
| `math:circle_packing` | ✅ |
| `math:heilbronn_triangle` | ✅ |
| `math:heilbronn_convex_13` | ⏳ 待 numpy 2.4.6 下复测 |
| `math:first/second/third_autocorr_ineq` | ⚠️ 可用，带 1e-5～1e-3 漂移 |
| `math:uncertainty_ineq` | ⚠️ 可用，带 4.6e-3 漂移 |
| **`math:signal_processing`** | ❌ **出局**：无任何 session 发布过 run-local `evaluate.py` |

**7 道 → 6 道**（若 `heilbronn_convex_13` 复测通过则为 7 道，含 4 道带漂移）。
这进一步削弱 §6.1 已声明的统计效力，须如实报告。

## A.8 种子程序的选择被修正

原方案取"该 task 最差的 valid 程序"。改为取**历史 run 自己的根节点**——语料自己定义的起点，
比某条搜索轨迹中途的低分程序站得住，且与语料记录的 run 可比。

实测：`circle_packing` 的 8 个根**代码完全相同**（3,873 字符、0.3642 分）。
断言"同一 task 的所有根为同一 artifact"抓到**唯一一个例外**：
`math:second_autocorr_ineq` 有 2 个根，因为语料带一个 `strong_seed` 消融组
（起点 0.9677 vs 基线 0.9549）。这是数据集的设计而非缺陷，改为**按多数票选根**
（base 18 + ablation 5 + nodiff 1 = 24 票对 strong_seed 8 票），
全部变体记入 receipt，**平票才失败**。

## A.9 已写好的代码清单

```
bench/agent_memory/gem_oe/
  constants.py         注入标记（不依赖 openevolve，供审计工具使用）
  memory_database.py   GemMemoryDatabase —— 唯一覆写 sample_from_island
  retrievers.py        GemRetriever / PolyglotRetriever / NullRetriever
  run_arm.py           单 (task, arm) cell，含 --dry-run
  run_matrix.py        task × arm 矩阵，按副本 round-robin
tools/evotrace/
  gate_evaluator.py / gate_throughput.py / gate_injection.py / gate_polyglot_parity.py
tests/test_gem_oe_memory.py   8 个单测
requirements-eval.txt         固定 numpy==2.4.6，理由见 A.5
```

矩阵 dry-run：**14/14 cells 通过**。单测：**8/8 通过**。

单测抓到的实现缺陷：原实现按"请求条数"切片，但 `super().sample_from_island()`
**常返回少于请求数**（island 太小、parent 被排除），导致 arm B 在小 island 上比 arm A
少渲染若干条，**恰好破坏 token 配平**。已改为"与 arm A 渲染条数相同"。
