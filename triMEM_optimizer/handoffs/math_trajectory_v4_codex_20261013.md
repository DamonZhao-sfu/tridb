# Codex 执行计划：最新 Trajectory 检索的 Math 对照实验

日期：2026-10-13。用户已授权：计划完成后直接交给 Codex 执行实验，无需再次确认。

## 实验目标与矩阵

比较最新库、三类图边和最新任务 prompt 下，检索计划对 agent 的 quality、tokens 和 latency 的影响。先做固定轮数实验，不做 BO，不扩展到 ALE 或 QA。

任务：`math:heilbronn_convex_13`、`math:second_autocorr_ineq`、`math:heilbronn_triangle`。
每个任务跑 `VGR`、`SemRankOnly`、`P1`、`P2`，每个 cell 使用 seed 42、20 轮，共 12 cells / 240 次计划中的迭代。@10 和 @20 从同一次运行提取。失败迭代计入尝试次数，未执行迭代标记 pending。

重要实现边界：当前 `SemRankOnly` 是 Vector → SemScore 的逐条评分，没有按分数重新排序。结果必须标注 “score-only, no reranking”，不能宣称测量了真正 reranker 的效果。不要为了消除这一限制偷偷加入 blocking Sort 或更改所有计划。VGR 无 semantic 算子，P1 在 graph 前 SemFilter，P2 在 graph 后 SemFilter。

## 固定输入与公平性

- 使用隔离新库 `host=/tmp/trimem-traj port=55450 dbname=trimem_traj`，不要读写旧实验库 55441。
- 离线 trajectory corpus 固定，记录 attempt 数量、版本及 full-code embedding/similarity 构建 manifest 哈希。每个 cell 新建自己的 OpenEvolve population，不续跑旧 checkpoint，不把其他 arm 的新结果写入公共 graph。
- 对每个任务固定 seed 源码、evaluator 和原始 SYSTEM MESSAGE，记录 SHA256。各 cell 重新实际评测初始程序。不要以历史 fitness 冒充新测量。
- 原始历史对照在 `results/trimem_v0.1.0/math_sem/it20/`，其参数是审计来源，不复制旧 10 个 inspiration slots。
- 固定 Qwen/Qwen3.8-27B-FP8、temperature 0.7、top_p 0.95、seed 42、thinking=false、generation max_tokens=32768、evaluator timeout=240 秒、LLM timeout=1800 秒、injection frequency/rate=1.0、diff evolution。其余参数核对历史 receipt 后记录。
- Math 历史默认 simplification nudge 保持一致，不静默添加 ALE 的 `--no-simplify-nudge`。说明示例导出器和运行模板可能存在此差异，运行 prompt 是权威依据。
- 当前图 SQL：24 vector seeds，一层 native expansion，lineage/context_up/similar 各 fanout 4，candidate cap 64。固定这些参数，记录实际 SQL 与 EXPLAIN。不要把旧实验三跳参数误认为最新图默认值。

## Prompt 和预算合同（启动前必须实现并验证）

1. 保留 task 原始 SYSTEM MESSAGE，Current Evaluation Feedback 独立呈现当前反馈，最近最多 3 条 Previous Attempts 各带自己的错误反馈。
2. 检索 inspiration 最多 4 个，每个是完整源程序 + 已存 intent，拼接不调用摘要 LLM。独立 own-run Top Performing Program 最多 1 个，排除当前 parent、外部 memory、未来记录和明确无效程序。因此“4”指检索 inspiration，不包括 current parent 和 top 1。
3. Inspiration section 总预算 16384 tokens，完整 generation input 上限 32768 tokens，包含 system/user 和实际 Qwen chat template/assistant prefix。output 的 32768 是另一个独立预算。服务 context window 必须至少 65536。
4. 超限时按顺序整条删除 inspiration、top 参考程序、最旧 Previous Attempt。不截断任何保留程序或错误反馈。只有 mandatory current context 仍超限则明确失败并记录，不能默默裁剪当前代码。
5. 每轮记录预算前后完整 token 数、实际 inspiration 数、超限/跳过/移除记录数，区分 section budget 与 full-input budget。代码截断数应为 0。保留 prompt/messages 和 provenance，展示 prompt 时可隐藏内部审计 ID。
6. Semantic 输入使用 `trajectory-context-v4-candidate-code`：当前问题、parent 完整代码、fitness、真实反馈、当前改进目标、剩余预算，以及候选完整代码和 intent。超出 native 请求限制要报错，不能截断。

## 启动与资源

先完成 targeted tests、read-only EXPLAIN、服务/模型/window 检查，然后先启动一个 cell 验证首轮完整链路，成功后顺序跑剩余 cells。一个 cell 同时执行，BLAS/OMP/evaluator threads=1、nice=10，至少保留一半 CPU。不要中断现有 Claude/ALE/BO 的任何进程，也不要重新启动共享模型服务器。记录共用 GPU contention，latency 不宣称无竞争环境。

Semantic worker 必须指向 55450。先确认是否已有同库 worker。需要时以 concurrency=2、max-take=2 启动自己的 worker，并记录 PID/命令/日志。embedding 必须使用现有完整代码的 embedding contract，不能退回 code[:2000] 或换模型。

启动前写 manifest、preflight.json、status.json。使用独立 session/可靠 supervisor 的持久 launcher，不能用工具 timeout process group 中裸 `nohup &` 假装后台任务可靠。用锁阻止重复启动，状态日志只追加。仅管理本实验明确记录的 PID。

单 cell 命令模板（先核对当前 argparse，补齐端点等已核实参数）：

```bash
.venv-e0/bin/python -m bench.agent_memory.gem_oe.run_arm \
  --task math:heilbronn_convex_13 --arm trimem --task-source trimem \
  --trajectory-store --trimem-dsn 'host=/tmp/trimem-traj port=55450 dbname=trimem_traj' \
  --trajectory-plan VGR --trajectory-memory-tokens 16384 --trajectory-prompt-tokens 32768 \
  --inject-as code --num-diverse 4 --injection-policy fixed \
  --injection-rate 1.0 --injection-frequency 1.0 \
  --llm-base http://127.0.0.1:8000/v1 --llm-model Qwen/Qwen3.8-27B-FP8 \
  --max-iterations 20 --max-tokens 32768 --seed 42 --thinking false \
  --llm-telemetry --eval-timeout 240 --checkpoint-interval 10 \
  --out results/trimem_v0.1.0/math_trajectory_v4_20261013/convex13__VGR__s42
```

不要直接复用 `oe_matrix.py` 的旧 msem driver，它仍调用 legacy seed-* 检索。`--trajectory-prompt-tokens` 是本次 prerequisite，未实现验证前不能启动。

## 计量和报告

- Quality：各任务独立 best valid combined_score 曲线，含 seed iteration 0，@10/@20 及相对 seed 提升。validity 用真实 evaluator 状态，不用统一 score>0 假设。区分非法 diff、生成失败、程序运行错误和超时。
- Tokens：generation prompt/completion、SemFilter、SemScore 分项。尽可能使用实际 API usage，区分 retries/cache hits/估计值。全局 worker 累计数不能冒充 cell token 数，拿不到归属就写 unavailable。
- Latency：retrieval/embedding/semantic wait、generation、evaluation、end-to-end 分开，明确嵌套时间不能相加两遍。共享离线 construction 成本单独报告，不每轮重复记账。
- 每轮审计：task/arm/seed/iteration、parent、候选/保留 ID、三种边与 SQL/物理计划、prompt/context/hash、inspiration 数、token limits、跳过/移除数、LLM usage、fitness/validity/errors。尝试轮数和成功产生 child 的数量分开。
- 单 seed 是探索性结果，不宣称统计显著。

将新结果以独立标记 section 加到 `triMEM_optimizer/it20_summary/trimem_it20_summary_en.html`，保留所有历史内容。使用 task tabs、@10/@20 切换、quality 曲线、token/latency 对比和 per-round prompt 审计。现有 `trimem/experiments/render_msem_section.py` 硬编码旧 arm，不能原样运行覆盖。

输出：manifest/preflight/status、所有原始日志和 receipts、汇总 JSON/CSV、更新 HTML、简短结论。先验证流程再积累结果。遇到服务不可用/资源不足时记具体 blocker，不伪造启动或结果。

## 已完成的执行交接验证

- Codex 预算实现：76 项 targeted tests 通过，targeted Ruff、run_arm 编译及 CLI 检查通过。
- 保存的最新完整示例含 chat template 为 12603 tokens，inspiration section 为 5163 tokens。此次仅重新计数，没有重新检索或调用生成模型。
- 预检查：55450 新库包含 10672 attempts，full-code embedding contract 为 qwen3-full-code-v1。服务模型报告 max_model_len=262144。
- Codex 执行代理已接收此计划，并获准在其 launcher/preflight 验证通过后启动。启动 PID、实际进度和结果以实验输出 status.json 为准，本段不代表模型实验已经完成或已经启动。

## 实际启动确认

- 输出目录：`results/trimem_v0.1.0/math_trajectory_v4_20261013/`。
- 首个 cell：`heilbronn_convex_13__VGR__s42`，实际生成 API 已成功返回，thinking=false。
- 首轮完整输入本地计数/API usage 均为 3595 tokens，4 个 inspiration 合计 1748 tokens，无截断。首个完成的 API 输出 1160 tokens。
- 当前 supervisor PID 以 `status.json` 为准。进度统计必须以实际 generation dispatch / evaluator 完成为依据，OpenEvolve 提前做的 retrieval 不计为已执行生成轮次。
- 完整矩阵仍在进行，以上不是最终 quality 结论。

## 后续授权：GPU 并行执行（2026-10-13）

用户要求停止 optimizer 实验、并行本轮 Math 实验，并优先完成同一任务的不同计划。
此授权替代前文单 cell 串行资源限制。队列按任务优先，依次 VGR/SemRankOnly/P1/P2，
从 6 个并发 cell 起步，空闲槽可补入下一任务。保持 4 inspirations、16K section、32K
完整输入和 32K 输出以及其他模型/检索参数。保留至少半数 CPU，embedding 副本分配
到两张 GPU，共享现有双副本生成服务。已完成的 convex13/VGR 不重跑。

该 VGR cell 是串行阶段结果，后续并行阶段的 wall-clock/latency 含资源竞争，报告应记录
执行模式，不把两阶段的耗时差异全部归因于 query plan。完整真实 prompt 示例：
`triMEM_optimizer/handoffs/vgr_convex13_iteration1_prompt.txt`。
