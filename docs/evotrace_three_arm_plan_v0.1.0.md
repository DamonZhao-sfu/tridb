# EvoTrace × GEM × OpenEvolve — 三方对比实施方案 v0.1.0

- 日期：2026-08-24
- 状态：**方案，未实施**
- 依据：`haikaidocs/benchmarkAnalysis.md` 附录 C（OpenEvolve 源码审计）与附录 D（指标溯源）
- 决定：agent 指标**只报 AlphaEvolve 原生的两个**；latency 对比先行；agent 用现成 OpenEvolve 0.3.2

---

## 0. 指标口径（已收窄）

**Agent 指标 —— 只报两个，均为 AlphaEvolve 原生：**

1. **best-so-far vs 计算预算 曲线**（AlphaEvolve Figure 8 形状）
   x = 计算预算（iteration 数；另出一版以累计 LLM token 为 x），
   y = 目标 metric，跨任务平均，每 arm 一条曲线，阴影 = 3 seed 的 CI。
2. **per-task 最佳分**，跨任务聚合用**分类率**而非求平均：
   `超过历史最佳 / 达到（evaluator 容差内）/ 未达到` 三档各占多少任务。
   （照 AlphaEvolve 的 "75% rediscovered, 20% discovered new" 做法，回避不可通约的分数平均。）

**已删除**：best-so-far AUC、iterations-to-threshold。二者非 AlphaEvolve 原生指标，
本版不报，避免在论文里解释自定义指标。

**检索指标**（第一类）不变：`gem_eg/metrics.py` 的 ndcg / oracle_recall / harmful@k / parity。
两类指标**分表报告**。

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

### 1.4 规模

18,291 个 decision point 中取分层样本（目前 W1 只测了 60 个）。
四个查询形状 W1.a / W1.b / W1.b-fail / W1.d × 三个系统。

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

---

## 3. Arm 定义

所有 arm 共享同一 `num_diverse_programs = N`，**只换来源，不改条数**：

| Arm | inspiration 来源 | 作用 |
|---|---|---|
| **A** no-reuse | N 条全部来自本 run 的 database（stock OpenEvolve） | 基线。**注意这不是「无上下文」** |
| **B** GEM | N 条替换为 GEM 跨 session 检索 | 主张 |
| **C** Polyglot | N 条替换为 Polyglot 检索（同语义、异系统） | 系统对比 |
| **D** Oracle | N 条替换为该 task 历史最优（后见之明，排除 held-out） | **上界诊断** |
| **E** Poisoned | N 条替换为已知死枝（`is_dead_end()`） | **有效性检查** |

**D 的作用**（把二元对照变成三点诊断）：

| 观察 | 结论 |
|---|---|
| B ≈ D | 检索已足够好，瓶颈不在检索 |
| B ≈ A 但 D ≫ A | **记忆有用，是我们的检索不行** ← 唯一可行动的诊断 |
| D ≈ A | 记忆在此任务上根本无用，与检索质量无关 |

**E 的作用**：outcome 不下降 ⇒ 注入通道是惰性的（被截断 / 被去重吃掉 / 模型忽略），
此时 A/B/C/D 的所有结论都不成立。成本低但不可省。

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

`sampler.py:449-455` 已有去重：inspiration 若与 top/diverse 区块重复会被丢弃，
外部条目与自身高分程序撞车时不会重复计 token。

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
| **0** | 五道闸门 | — | 需要（部署实测） |
| **1** | Pilot + manipulation check，`circle_packing`，arm A/B/C/D/**E** × 3 seed | 15 runs | 需要 |
| **2** | 主实验，7 task × arm A/B/C/D × 3 seed | 84 runs | 需要 |
| **3** | ale 扩展（可选），需先验证 ale-bench 复现历史分数 | — | 需要 |

**Phase 1 门槛（全满足才进 Phase 2）**：
- E 显著劣于 A → 注入通道有效；**E ≈ A 则实验作废，先修注入路径**
- D 显著优于 A → 该任务上记忆有可测的上升空间

**预算估算（算术推演，未实测）**：Phase 1+2 ≈ 99 runs × ~17–20 min，
两卡 data-parallel ≈ **14–17 小时 wall-clock**。此数建立在未实测的吞吐推算上，
Phase 0 实测后可能显著变化。

---

## 6. 必须同时出现的声明

1. 我们的 fitness **不能与 EvoTrace / AlphaEvolve 论文数字对比** —— 原 trace 的模型是
   deepseek-reasoner(76%) / gemini-3-flash / claude / gpt-5，**无 qwen3.8**；
2. thinking 开关状态（EvoTrace 实测 reasoning tokens p50 1,941 / p99 16,384，影响极大）；
3. Cognee 与 GEM/Polyglot **检索语义不同**，不得同表比 latency；
4. 检索指标与 agent outcome **分表报告** —— 数据库更快推不出 agent 更聪明；
5. Arm A **不是「无上下文」**，它是「只有本 run 上下文」。

---

## 7. 已知风险

| 风险 | 表现 | 处置 |
|---|---|---|
| 注入通道惰性 | E ≈ A | Phase 1 门槛拦截 |
| 记忆天花板过低 | D ≈ A | 该 task 出局；**先于 B 的结论判定** |
| evaluator 不复现 | 历史最优重跑分数不符 | Phase 0 闸门 2，该 task 出局 |
| polyglot 空结果复发 | 结果集为空却计 0 分 | Phase L 闸门 1+2 |
| cascade 静默退化 | evaluator 无 `evaluate_stage1` | Phase 0 闸门 3 |
| 泄漏 | 检索到的代码与 target session 逐字节相同 | `is_trivial_hit()` 已实现，单独计数并从 headline 剔除 |
| task 数偏小 | 7 个 math task，跨任务统计效力有限 | 如实声明；Phase 3 扩到 17 个 |
