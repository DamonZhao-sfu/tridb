受限于未提供完整 brain principles；以下仅按指定的数学、split、去重、预算与恢复 lens 审阅。结论：**REVISE**。

1. **[§6–§9，高] 连续 acquisition 与实际提交目标不一致。**  
   L-BFGS-B 优化的是平滑的 `ell(u)`，但可提交配置实际按 `ell(P(u))` 重评分；`P=A∘E∘D∘A⁻¹` 含 greedy argmax，且可能把大量端点投到同一 key。重评分保证不会用错误端点直接 profile，但不使梯度优化成为对实际可选配置的 BO 优化。文中虽承认这一点，却仍将其作为核心 latent refinement。  
   **补救：**明确将其表述为“连续 surrogate 引导的候选生成启发式”，而非直接优化可执行配置的 qLogNEHVI；加入硬性诊断/停止门槛（投影后最佳分数相对 anchor/pool 的改进率、端点→unique-key 比率、投影后排序相关性）。若该关联不足，应切换到离散合法池上的 acquisition 搜索，而非继续解释为连续优化有效。

2. **[§6–§7，高] GP 的目标坐标与 observation noise 未形成可执行的数学合同。**  
   文本同时出现 raw `y`、初始化冻结的归一化 `y`，以及 `Standardize` outcome transform；未明确 qLogNEHVI 的 `ref` 位于哪一个坐标系。若 ref 被传入内部标准化空间或对已归一化数据再次按错误尺度解释，HV 和 acquisition 将失真。另，“100 个固定 TRAIN tasks 的 task 明细方差”只有在目标是从 task population 估计期望时才是均值估计噪声；若目标就是这固定 panel 的均值，它不是 observation noise。  
   **补救：**冻结唯一合同：原始三目标 → 初始化归一化 → GP 输入/输出 → `Standardize` 的内部作用 → acquisition/ref/HV 的外部坐标，并用单元测试验证等价性。明确 estimand；固定 panel 时用重复执行/agent seed 的测量噪声或保守 nugget，若把 task 当抽样总体则需说明共享 panel 导致的跨配置相关性及其近似处理。

3. **[§8–§9，高] “pending-aware q=4” 的语义不足以防止伪批量多样性。**  
   伪代码把 `pending + selected` 传给 `frozen_qLogNEHVI`，但未定义是通过 `set_X_pending` 的后验 fantasization，还是把 pending 当作已观测点/仅在去重中排除。后两者都不能实现 qNEHVI 对未完成结果的不确定性处理，并可能重复选择高度相关的点。  
   **补救：**规定每 slot 都使用 q=1 acquisition，调用支持的 pending/fantasy 机制，以已预留和运行中配置的 canonical `u(c)` 设置 `X_pending`；不可用时明确改为 sequential greedy 的近似，并停止宣称 pending-aware qLogNEHVI。恢复后必须从 reservation 状态重建同一 pending 集。

4. **[§1、§7、§12，中] TRAIN/TEST 的最终选择与预算仍有歧义。**  
   §1 的“仅搜索结束后评测 TRAIN Pareto”与 §12 的“固定 200 TEST 仅用于 TRAIN Pareto 最终评价”表述冲突；尤其未规定对最终 TRAIN Pareto 的全部点、固定 top-k，还是一个由规则选出的代表点进行 TEST。不同 arm 的 Pareto 点数不同，会造成 TEST 工作量和报告口径不对称。  
   **补救：**冻结协议：TEST 完全不参与训练、模型选择、候选选择和阈值调参；明确最终从 TRAIN archive 选哪些点去跑固定 TEST-200、该次数是否另计预算、及所有 arm 相同的选择规则。

5. **[§11，中] 恢复协议缺少 profile 提交—完成—持久化之间的幂等状态机。**  
   `evaluation_key` 与“核验 pending 真实状态”不足以处理崩溃窗口：profile 已完成但 observation 尚未写入 checkpoint，或 reservation 仍是 pending 时重启，可能重复 profile、错误消耗 30 个新配置预算，或漏入 archive。  
   **补救：**将每个 evaluation key 设为持久化的状态机（reserved → submitted〈带外部 request/run ID〉→ completed_raw → archived/failed），恢复时先按外部 ID 查询或读取完成产物，再原子写 archive 和预算账本；只有确认未提交的 reservation 才释放重选。规定失败、超时、部分产物是否计入“新 profile”预算。