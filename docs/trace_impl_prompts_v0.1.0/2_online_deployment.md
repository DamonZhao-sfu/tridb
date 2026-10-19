# 实现 prompt 2：在线部分（加载冻结产物、逐题路由、执行与回执）

> 在 prompt 0 的共享层与 prompt 1 的阶段 C（冻结产物）合入之后再做。

---

你在 `DamonZhao-sfu/tridb` 仓库的 `experiment-trajectory-retrieval-v4` 分支上工作。任务是实现 TRACE 的在线部分：加载离线冻结的路由器产物，对每道新题先做库内探针，再选择检索配置、执行检索、交给 agent，并写下完整回执。在线部分只做预测、选择和执行，不更新任何模型。

## 先读这些

1. `CLAUDE.md`。
2. `docs/trace_routing_aware_design_v0.2.0.md`，重点是 §5 和 §6 的在线流程。
3. `trimem/trace/features.py`（`compute_features`、`feature_schema_hash`）、`trimem/trace/retrieval_only.py`、`trimem/trace/train_router.py` 的产物格式。
4. `trimem/optimizer/router.py` 的 `RoutedPolicy.from_artifact`、`check_deployment`、`choose`。在线加载的校验方式照搬这里。
5. 环境类 agent 的实验驱动，也就是 prompt 0 第一步找到的那份代码：ALFWorld、AppWorld、SWE-smith 的 arm 是如何定义、运行和记账的。

## 要实现的内容

### 1. `trimem/trace/deploy.py`

- `TracePolicy.load(path, *, expected_sha256=None)`，依次校验：
  1. 产物的 sha256。
  2. 每个模型 pickle 的 sha256。
  3. sklearn 版本完全一致。
  4. 产物里的 `feature_schema_hash` 等于当前代码的 `feature_schema_hash()`。
  5. 候选集中每个配置都能还原成合法的 `CandidateSpec`。

  任何一项不符就拒绝加载，并在报错中说明是哪一项不符。不得静默降级。
- `route(conn, query) -> Decision`：
  - 调用 `compute_features` 得到特征。这是唯一的特征入口，禁止在此另写一份特征逻辑。
  - 对候选集中每个配置计算 `ĝ_c(z) − λ·ĉ_c`。
  - 应用回退规则，确定所选配置。
  - `Decision` 记录：特征、每个候选的分数、所选配置的 action_hash、`fallback`（布尔值），以及 `fallback_reason`。原因取值为 `gain_below_noise_floor`、`probe_error`、`single_config_mode` 或 `None`。
  - 探针出错（超时或 DB 错误）时，回退到 c₀，并记录 `probe_error` 和错误摘要。不得丢掉这道题，也不得重试到成功为止。
- `execute(conn, query, decision) -> Memory`：调用真实的检索入口执行所选配置。
  - SemFilter 和 Rank 只在所选配置要求时才启用。judge 在进程外调用，结果写入 PG 缓存，遵循 ADR-0026。
  - 打包沿用现有的 token 预算逻辑。
- **可选优化：复用探针种子。** 当所选配置的种子数与查询向量都和探针相同时，可以直接复用探针的种子，省掉第二次 ANN。启用前必须有测试证明：开启与关闭复用时，注入集合逐字节相同。默认关闭。

### 2. 回执

每道题追加写一行 JSONL。字段如下：

- 标识：`query_id`、`split`、`artifact_sha256`、`feature_version`。
- 特征：`features`（原始统计和最终特征都要有）。
- 路由：`scores`、`chosen_config_hash`、`fallback`、`fallback_reason`。
- 耗时：`probe_latency_ms`（T1 和 T2 分别记）、`route_latency_ms`、`retrieval_latency_s`。
- 检索结果：`semantic_tokens`、`injected_ids`、`injected_hash_ordered`。
- 完成 episode 后补写：`quality`、agent 的输入与输出 token、生成耗时。

回执是只写的审计记录。**不得写入任何训练矩阵或优化器归档。** 测试集的结果绝不回流到训练。

### 3. 接入实验驱动

- 在环境类 agent 的实验驱动中新增一个 arm `TRACE-router`，与现有固定计划的 arm 并列，不修改其他 arm 的任何行为。
- 运行器增加参数 `--split test --policy <artifact.json> --policy-sha256 <sha>`。
- 汇总时与固定计划的 arm 做同题配对比较，报告质量、token、延迟的均值与配对 SE。格式与现有结果页一致，并单列路由开销：探针加路由的 p50 和 p95，以及它占 episode 时长的比例。
- 输出路由分布：每个配置被选中的题数与比例，以及回退率和各回退原因的占比。

## 测试与验收

- `tests/test_trace_deploy.py`，必须覆盖：
  - 同样的特征和产物，选择结果确定。
  - sha 不符、特征版本不符、sklearn 版本不符时都拒绝加载。
  - 三种回退路径都会触发，并被正确记录。
  - 回执字段齐全，且没有任何写入训练归档的代码路径（用 mock 断言）。
  - 在线与离线一致：给定同一组特征，`route` 的选择等于阶段 C 里 λ 校准模拟所做的选择。
  - 如果实现了探针种子复用，验证注入集合逐字节相同。
- `make test` 和 `make lint` 通过。

## 运行规则

- 没有用户的明确授权，不得在测试集上跑 agent episode，也不得启动 GPU。
- 获得授权后，先在 5 道测试题上做冒烟测试，检查回执完整、成本记账正确，再跑完整测试集。
- 报告每个数字时附上回执文件路径。冒烟测试不等于完成，也不能宣称有了结论。

提交信息格式为 `feat(trace): ...`，推送到 `experiment-trajectory-retrieval-v4`。最后用一段话报告：做了什么、测了什么、哪些没测以及原因。
