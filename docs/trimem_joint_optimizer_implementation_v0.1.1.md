# TriMEM 联合优化器：首版实现与运行说明

这是 `trimem_joint_optimizer_implementation_plan_v0.1.0.html` 的实现补充，不修改旧计划或旧 P1/P2 实验语义。按本次确认，目标中去掉 latency，先做 Math/ALE，先运行 1 个 Math task、每候选 5 轮 continuation。第二种固定总 token 实验使用独立协议和归档。

## 目标与搜索

优化 `(E[best_valid_fitness], -E[total_LLM_input+output_tokens])`，保留 Pareto 前沿，不用加权和预先替用户决定质量与成本的兑换比例。负 fitness 可以是有效程序，不能按正负号判定有效性。固定 5 轮得到 `Q_5_proxy`，不能称为完整任务的终局质量。

借鉴 RAG-Stack 的候选、评估、代理模型分离；实现独立编写，直接调用 BoTorch 的 `MixedSingleTaskGP`、`ModelListGP` 和 `qLogNoisyExpectedHypervolumeImprovement`。没有复制 RAG-Stack 源码，也没有将其自带任务标签当成 Math/ALE 的质量。参考 checkout 为 `64d82093b877c45a44164ed7534ffe651c5946b9`，本地路径见此前实现计划。

一次 action 是 `(plan_id, active_config)`。联合搜索同时改变计划和配置；固定计划搜索使用同一搜索空间的子集。候选提出阶段只读取元数据，不执行数据库查询、不调用 LLM，也不预先构建所有计划的结果池。计划可返回不同上下文，没有等价性保证或等价性准入条件。

默认网格有 1985 个合法候选。变量包括 seed count、明确的边类型序列/hop、fanout、最终 top-k、stream score threshold 和 unknown policy。不存在于计划中的变量不会改变 action hash。`SpaceGrid.score_admit_mins` 默认 `(0.3, 0.5, 0.7)`；旧单阈值参数仍可覆盖。SemFilter 的实际选择率是测量值，不能把它当作可以任意指定的执行结果。

| 新计划 ID | 实际执行顺序 |
| --- | --- |
| `no_memory_v1` | 不访问数据库或 embedding 服务 |
| `vgr_v1` | Vector seeds → Native Graph → relational telemetry → bounded packing |
| `vgr_score_v1` | Vector seeds → Native Graph → telemetry → StreamScore → packing |
| `seed_sem_expand_v1` | Vector seeds → seed SemFilter → Native Graph → telemetry → StreamScore → packing |
| `expand_row_sem_v1` | Vector seeds → Native Graph → telemetry → endpoint SemFilter → StreamScore → packing |

这里的 StreamScore 逐条评分并按阈值接纳，保持原输入顺序；它不是全局排序的 SemRank。全局 Sort 会阻塞，违反 TR-1，因此没有偷偷放入首版。评分开关目前由 VGR 与 VGR+Score 两个计划族表达；两种 SemFilter 计划末端固定有 StreamScore。返回明确 hop 深度的 endpoint，不声称复现旧实验的各层 frontier union。

模型使用预先冻结的任务尺度、权重和 hypervolume reference；reference 不是 token budget。重复观测保留原始记录，按同 cohort 聚合，方差不因重复使用相同任务/seed 而不当地缩小。启动按计划族分层覆盖，随后用 qLogNEHVI 提议昂贵评估；依赖或拟合失败时显式记录 fallback。

## 执行与计量

`compiler.py` 只编译选中的 action。`executor.py` 先做不执行查询的 `EXPLAIN`，拒绝 Sort、Hash、Materialize 等阻塞/未知节点，再使用 named cursor 逐行消费，实际检索只执行一次。达到 top-k 或 token cap 时 Close 上游。拓扑始终由同一 Postgres 进程中的 native `trimem.expand` 访问；关系访问只读取属性，没有关系边表或图 sidecar。

`edge_emit_cap` 是根据 seeds/fanout/hops 推导的输出边数量上界，不能冒充底层检查过的邻接槽数。实际物理槽访问量未测量时保留 `None`。

memory 上限是 **8192 tokens**，与完整生成 prompt 的 model context window 和输出预留分开。使用部署模型本地 tokenizer、完整 chat template、`enable_thinking=False`；每条截断/跳过、原始和保留 token 数、最终 memory/full prompt token 数写入 receipt。只统计已经消费的记录，不为了统计假想尾部而跑完整个查询。后续每轮重新计数实际 prompt，超出窗口会失败，不静默省略经验。

主目标的 token 类别是 construction、retrieval、semantic_filter、semantic_rank、generation、retry、reflection。失败请求和每个物理重试都保留；缺少 provider usage 就标记不完整，不能计成 0 或进入 GP。memory 注入的 token 已包含在 generation input，不能再加一次。embedding、纯研究 evaluator、optimizer 费用单列。首版 frozen graph 没有在线 construction LLM 调用；历史建图成本不能伪装成当前调用成本。

`clients.py` 默认不重试，显式配置的重试逐次计费。发送前用完整 prompt 和剩余预算决定输出 allowance，发送后核对真实 provider usage。供应方不返回完整 usage 或违反 allowance 的观测保留为失败。此保护依赖匹配的 tokenizer 和供应方协议，不保证异常供应方不会收费。

## 隔离与评估协议

首版运行方式为 `first_decision_then_frozen_policy` + `frozen_no_ingest`：在共同 checkpoint 选择一次检索动作，后续 continuation 复用返回的 memory，以相同策略改善 best valid program。每个候选使用相同 task/checkpoint/seed、初始图、模型、提示、评估器和结束条件。它还不是在线持续建图闭环，也不是每一轮重新自适应选计划。

当前 semantic mailbox 是 cluster 共享的，不能把同 cluster 的不同 database 当作可靠计量隔离。`private_pg.py` 每个检索分支创建全新临时 PostgreSQL cluster、独立 Unix socket和空语义缓存；no-memory 不启动 PG。`snapshot.py` 从冻结输入重建原生图，查询时不读取外部图文件。结束后停止并清理自己创建的实例，不连接已有 55440/55441。

`semantic_usage.py` 核验私有 cluster identity、预期 backend 和空 mailbox，启动自己的短生命周期 `Worker.step` 循环。semantic 请求串行计费，结束后一次性归入主 ledger。预算拒绝/worker 错误会唤醒 native 查询，不等待长超时；worker 未退出则保留 poisoned 状态，避免下一次错误归因。runtime 首版要求 semantic/generation 使用同一模型版本与 tokenizer。

两种协议不能混入同一归档：

1. **先运行固定 5 轮**：质量是第 5 轮结束的 best valid fitness，成本是实际总 tokens。400000 tokens 是预先冻结的保护上限；提前触及它会标记 `horizon_incomplete`，不会伪装完成 5 轮。
2. **之后固定总 tokens**：样例预算 100000，计入语义检索和生成等实际请求；没有空间容纳下一次合法请求即正常结束。100 轮仅是防失控保护，先触及轮数上限的观测不当成完成 token horizon。

`math_ale.py` 使用原 evaluator 的 metrics，区分程序无效与基础设施失败，保存每轮完整代码、请求、反馈和 best fitness。子进程限制 CPU 亲和性、数值库线程和超时；这不是执行任意不可信代码的安全沙箱。初始 seed、NumPy/random seed、PYTHONHASHSEED 记录在协议中；原 evaluator 自己另开的子进程是否确定性仍由其契约决定。

## 已准备的真实首轮输入

目录：`triMEM_optimizer/trajectory_optimizer_design/first_edition_evidence/`。

- `math_fixed5_config.json`：1 个 `math:circle_packing`，seed 42，17 个候选，5 轮；Qwen3.8-27B-FP8，thinking 关闭，generation 输出上限 8192。
- `math_token100k_config.json`：同一输入的第二种结束协议，单独 archive。
- `inputs/math_circle_frozen.json`：278 节点、3 个完整历史 session；排除 query checkpoint 所属 session 的全部 90 节点。lineage 271、similar 3845，外加各自精确反向边。similar 来自已有 native snapshot，未重算 kNN；向量来自已有 1024 维 embedding。
- `inputs/checkpoint_calibration.json`：原始 evaluator 单次 CPU 校准，`validity=1.0`、fitness `0.36423689449571406`，与历史记录一致。它是起点校准，不是新优化实验结果。
- `discovered_math_ale_inputs.json`：本地 5 Math / 10 ALE 输入清单。首轮只冻结 circle；后续 ALE 需固定 public cases/评估设置并按同一接口接入。

这些输入与核心源码均通过 SHA256 绑定协议。单 seed smoke 用于检查联调和计量，不支持统计显著性或普遍质量提升结论。当前 Claude/ALE 任务仍占用模型服务，真实 Qwen continuation 尚未启动，也没有启动后台竞争实验。

## 运行

独立可选环境使用 `requirements-trimem-optimizer.txt`，不改变原依赖环境：

```bash
uv venv --python python3 .venv-optimizer
uv pip install --python .venv-optimizer/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-optimizer/bin/python -r requirements-trimem-optimizer.txt
```

本次精确依赖版本见 evidence 中的 `environment.freeze.txt`。实际使用 CPU torch；不会启动本地生成模型。以下命令在仓库根目录运行。

```bash
# 只检查输入，不访问模型、数据库，也不运行 evaluator
.venv-optimizer/bin/python -m trimem.optimizer preflight \
  --config triMEM_optimizer/trajectory_optimizer_design/first_edition_evidence/math_fixed5_config.json

# 无真实任务/模型的 CPU BO 联调；结果明确标 synthetic
.venv-optimizer/bin/python -m trimem.optimizer demo --archive /tmp/trimem-bo-demo --steps 8

# 等既有 Claude 实验结束后，启动一个候选的五轮真实 continuation
taskset -c 16-19 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' .venv-optimizer/bin/python -m trimem.optimizer run \
  --factory trimem.optimizer.runtime:math_ale_factory \
  --config triMEM_optimizer/trajectory_optimizer_design/first_edition_evidence/math_fixed5_config.json \
  --archive triMEM_optimizer/trajectory_optimizer_design/first_edition_evidence/math_fixed5_archive \
  --steps 1
```

`--steps` 是 BO trial 数，每个 trial 内按协议生成 5 轮。后续重复相同 `run` 命令可追加 trial；同一 archive 拒绝不同协议。也可以 `ask`、外部评估、`tell`，但 tell 必须提供完整逐请求 ledger 和逐任务结果，通过重算校验才能入模。中断的 pending trial 不自动重复收费；`fail --reason ... --evidence ...` 显式保留证据再继续。

## 验证与边界

新增定向测试 **139 passed**，结果见 `optimizer_tests.xml`；覆盖真实 CPU qLogNEHVI、归档恢复/防串协议、token 会计、完整 Math/ALE 回调，以及独立 PG 中四种计划与真实 worker mailbox 路径。已有 objective/semantic 回归 **23 passed、3 skipped**，未设置共享 DB DSN。语义 HTTP 和生成 HTTP 在集成测试中使用 mock；不当成真实 LLM 质量证据。8 步 synthetic archive 中前 5 步分层初始化、后 3 步确实选择了 qLogNEHVI。

```bash
taskset -c 16-19 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' TRIMEM_OPTIMIZER_TEST_PG_PREFIX="$PWD/.pgprefix16" \
  .venv-optimizer/bin/python -m pytest -q tests/test_trimem_optimizer_*.py
.venv/bin/python -m ruff check trimem/optimizer tests/test_trimem_optimizer_*.py
.venv/bin/python -m ruff format --check trimem/optimizer tests/test_trimem_optimizer_*.py
```

Native 验证只覆盖当前安装的 **PG16.14 / TriMEM 0.2.0**。没有改动或重建 C；当前源码 0.3、PG13.4 ARM64 fork、GX10/CUDA 和 128GB benchmark 均不在本次验收范围。
