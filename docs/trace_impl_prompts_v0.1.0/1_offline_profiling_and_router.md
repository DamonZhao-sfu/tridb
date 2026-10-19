# 实现 prompt 1：离线部分（回放分析、为路由而探索、路由器训练与冻结）

> 在 prompt 0 的共享层合入之后再做。

---

你在 `DamonZhao-sfu/tridb` 仓库的 `experiment-trajectory-retrieval-v4` 分支上工作。任务是实现 TRACE 的离线部分，分三个阶段。阶段 A 不需要 GPU，必须先做完并报告结果，再继续 B 和 C。

## 先读这些

1. `CLAUDE.md`。
2. `docs/trace_routing_aware_design_v0.2.0.md`，全文。§4 是探索算法，§5 是路由器训练，§7 是验证计划。
3. 共享层 `trimem/trace/`：`contract.py`、`features.py`、`retrieval_only.py`。如果它们还不存在，就停下来，先完成 `docs/trace_impl_prompts_v0.1.0/0_shared_contract_and_features.md`。
4. 现有优化器：
   - `trimem/optimizer/search.py`：qLogNEHVI、按计划初始化、`SearchSettings`、`Proposal`。
   - `trimem/optimizer/router.py`：BRANE 风格的路由器。重点看 artifact 的 sha 校验、`deployment_invariants`、λ 校准，以及 sklearn 版本的严格检查。
   - `trimem/optimizer/archive.py`：`TrialArchive` 的 reserve、tell、fail 机制。

## 核心定义（摘自设计文档，实现时以它为准）

- S 是已 profile 的配置集合，始终包含默认配置 c₀。
- 标签：`g(q,c) = Q[q,c] − Q[q,c₀]`。
- 路由价值在特征分辨率上计算，使用交叉拟合的预测值 μ，而不是原始 0/1 结果：
  `V_route(S) = (1/n) Σ_q max_{s∈S} [ μ_s(z_q) − λ·c_s ]`
- 对照组是单点部署的价值：`V_mean(S) = max_{s∈S} (1/n) Σ_q [ μ_s(z_q) − λ·c_s ]`。
- 成本：`c_s = w_T·ln T(s) + w_L·ln L(s)`，T 和 L 分别是 profiling 得到的平均 token 和平均延迟，与原草稿的公式 (2) 一致。
- 路由超体积：在一组 λ 上画出路由后的（平均质量，平均成本）点，再求超体积。参考点在开始前固定，不能从当前数据的极值算出来。

## 阶段 A：用现有矩阵做回放分析，不用 GPU

把实现放在 `trimem/trace/replay.py`，CLI 为 `python -m trimem.trace replay`。

- **A1 路由空间（设计文档 E2）**：对每个数据集报告下面四个量，并在 query 上做 1000 次 bootstrap 给出置信区间。
  1. 最好单配置的质量。
  2. 原始 oracle：逐题取最大值。它会因噪声虚高，必须明确标注为上界。
  3. 特征分辨率上的路由上界：把 query 分成 5 折交叉拟合 μ，每题选 argmax μ，再在矩阵里查被选配置的真实结果。
  4. 上面三者之间的差值。
- **A2 探索策略回放（设计文档 E3）**：
  - 全集：矩阵中已有的全部配置（约 45 个）。
  - 策略：随机；均值超体积（复用 `search.py` 的 qLogNEHVI，或在 μ 上按 V_mean 贪心）；路由价值（新方法）；全知贪心（直接用真实矩阵，作为上界）。
  - 预算：k = 5、10、15、20 个配置，所有策略都从 c₀ 加按计划初始化的同一批起点开始。
  - 评估：query 外层 5 折。在 4 折内完成选配置和训练路由器（用阶段 C 的训练函数），在留出折上查矩阵得到路由后的质量、成本和路由超体积。随机性用至少 5 个种子。
  - 输出：JSON 汇总，外加两张图，分别是"预算对路由后质量"和"预算对路由超体积"。
  - 报告里写明偏置：这批全集本身是按均值 BO 选出的，所以结论偏保守。
- **A3 噪声地板**：如果矩阵里有同题同配置的重复运行，估计逐题方差；没有就写明缺失，并给出 E1 实验需要补跑的最小规模（ALFWorld 20 题 × 5 配置 × 3 次）。

**做完阶段 A 后先停下，报告数字，等用户确认再进入阶段 B。** 如果 A1 显示特征分辨率上的路由空间落在噪声之内，就直接报告这个结论。

## 阶段 B：为路由而探索

- `trimem/trace/surrogate.py`：
  - 联合代理模型 `f(z_q, enc(x), r(q,x)) → g`，用约 20 个 bootstrap 的 `GradientBoostingRegressor` 组成集成。sklearn 版本与 `router.py` 的要求一致，并加同样严格的版本检查。
  - 支持后验均值、按集成成员采样，以及对未 profile 配置的预测（此时 r(q,x) 来自 `retrieve_only`）。
- `trimem/trace/acquisition.py`：
  - 单个 λ 的路由 EI：`α_λ(x) = (1/n) Σ_q E[(f_x(z_q) − λ·c_x − b_S(q))⁺]`，其中 `b_S(q) = max_{s∈S} μ_s(z_q) − λ·c_s`。
  - 多个 λ 的路由超体积增量，用集成样本做蒙特卡洛估计。
  - profiling 成本：`cost(x) = (x 在训练题上产生的新有序注入哈希数) × 单个 episode 的平均成本`。注入哈希已经 profile 过（同一 `protocol_id`）的题，直接复用结果，并记录 `reused_from`。复用机制来自 RAG-Stack 的 quality reuse，代码注释和文档都要注明出处，不得称为新贡献。
  - 批量选择：按 α/cost 做 CELF 式的惰性成本收益贪心。每选中一个配置，用后验均值占位后再选下一个。
  - 停止规则，任一成立即停：max α/cost 低于阈值；或者路由超体积上界减去最好单配置超体积，落在噪声地板之内。后一种情况输出结论"该 workload 不需要路由"。
- 接入现有搜索：
  - 在 `search.py` 中新增一个显式开关，例如 `SearchSettings.acquisition = "qlognehvi" | "routed_hvi"`，默认值保持 `"qlognehvi"`，现有行为和现有测试都不能改变。
  - 新路径产出的 `Proposal.strategy` 为 `"routed_hvi"`，diagnostic 中写入 α、cost、b_S 摘要、复用的题数。
  - 候选池沿用 `_stratified_pool`。如果环境类实验的代码里另有四通道的提案器，就接入它；没有就不要临时编写。
- 实时 profiling 需要跑 agent episode 和 GPU。**没有用户的明确授权不得启动。** 先提供 `--dry-run` 模式（只做检索和打分，打印将要 profile 的配置及预计 episode 数），以及基于 fake evaluator 的端到端测试。

## 阶段 C：训练并冻结路由器

`trimem/trace/train_router.py`，CLI 为 `python -m trimem.trace train-router`：

1. **候选集**：模糊 Pareto 集，即与 BRANE 一样 τ_acc = 0.02、τ_cost = 0.10，再并上每个检索计划在训练集上的最优点和 c₀。
2. **模型**：
   - 主模型是共享 GBDT `f(z_q, enc(c)) → g`。
   - 消融一：逐配置的独立分类或回归器，按 BRANE 的做法在 logistic regression、决策树、随机森林、GBDT 中用内层 3 折交叉验证选模型。
   - 消融二：输入只用 T0 题干特征。
3. **λ 校准**：
   - 按 query 做 5 折交叉拟合，对 λ 网格上的每个值模拟路由，用矩阵中被选配置的真实结果计算质量和成本。
   - 选成本最低、且质量与最好值相差不超过 ε 的 λ。ε 默认取一个 bootstrap SE，做成参数。
   - 完整的 λ 表存进产物。
4. **噪声地板回退**：若 `max_c ĝ_c − ĝ_{c₀}` 不超过阈值就选 c₀。阈值来自阶段 A3 的逐题噪声估计；如果没有，就用 bootstrap SE，并在产物里标注来源。
5. **路由或单配置**：在训练集的交叉验证上比较路由与最好单配置。路由不显著更好时，产物记录 `mode = "single_config"`。
6. **产物（JSON）**：
   - 内容：`FEATURE_VERSION`、`feature_schema_hash`、拟合好的 T0 变换（PCA）、候选集（每项含 action_hash 与 config）、模型（pickle 后 base64 编码，各附 sha256）、每个配置的平均成本、λ 及完整 λ 表、回退规则与阈值、mode、训练数据 manifest 的 sha、sklearn 版本、代码哈希、整个产物的 `artifact_sha256`。
   - 发布方式和加载校验照搬 `router.py`：先写临时文件再原子地 hard-link 到目标位置，目标已存在就失败。
7. **永远不读取 test split**：训练入口检测到 test 行就直接报错。

## 测试与验收

- `tests/test_trace_replay.py`、`tests/test_trace_acquisition.py`、`tests/test_trace_train_router.py`，必须覆盖：
  - 合成数据上，单个 λ 的路由 EI 与 V_route 增量的蒙特卡洛结果一致。
  - 构造一个"均值较低、但在某个子群上占优"的配置：V_mean 给它的增益为 0，V_route 给它的增益为正。
  - μ 固定时，贪心选出的集合价值不低于最优解的 (1 − 1/e) 倍（在小规模问题上穷举验证）。
  - 注入哈希相同的题被复用，并且 cost 被正确扣减。
  - 产物的 sha 或 sklearn 版本不符时加载失败。
  - 训练数据中出现 test 行时报错。
  - `search.py` 原有测试全部不变且通过。
- `make test` 和 `make lint` 通过。

提交信息格式为 `feat(trace): ...`，按阶段分别提交，推送到 `experiment-trajectory-retrieval-v4`。报告里给出阶段 A 的实际数字，每个数字附上它所在的结果文件路径。没跑的部分写明没跑，不要估算。
