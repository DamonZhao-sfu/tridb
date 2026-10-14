# TriMEM 使用 Qwen3.8 复现 EXG 的在线与离线实验计划草案

本实验在 HumanEval、HumanEval+、MuSiQue 和 HotpotQA 上测量经验复用策略及 TriMEM 执行计划对质量、延迟和 token 的影响。在线实验从空经验图开始，逐 attempt 写入；离线实验先收集轨迹，再冻结源经验图。主要比较在同一 Qwen3.8 部署上完成，不将更换模型后的结果称为 EXG 表格数值的严格复现。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent。
- Origin Mode: plan；Version: v0.1.2；Origin Date: 2026-10-11。
- Verification Status: UNVERIFIED，讨论草案；当前工作只编写方案，不运行实验。
- 已确定：使用 Qwen3.8；同时做在线和离线实验；在线空图启动；QA evaluator 只返回答对或答错，不返回 gold answer；答错最多重试一次；MuSiQue 和 HotpotQA 首轮各使用 500 个任务；这一规模由用户明确选择，不声称恢复作者全部实验设置。
- 其余具体参数和计划矩阵均为建议，须在讨论后冻结为运行协议。

## 研究问题与比较边界

1. 跨任务经验是否提高首次成功率、两次尝试内成功率，并改变完成同一批任务所需的总成本？
2. 在三模态检索之上增加语义适用性判断和经验摘要，是否改善 quality–latency–token 的取舍？
3. 保持逻辑查询、候选宇宙、semantic 判断和输出契约相同时，物理计划能否减少检索内 LLM 工作与延迟？
4. eager、lazy 和成本驱动的派生时机，分别在什么读写比下有利？减少的是总工作还是只移动了工作发生的时间？

问题 1、2 涉及返回上下文变化，属于策略比较；问题 3 是等价物理计划比较。等价计划应返回相同有序 hints，不能把随机生成差异解释为它提高了质量。自适应 hint 数、模型、图深度等改变结果的选择，单独作为质量受约束的策略扩展。

主研究对象是 TriMEM 的规划收益，建议先固定为一个 EXG 式求解器：生成、评估、必要时根据反馈重试；不额外加入 Reflexion 的独立反思调用。EXG-Reflexion、SE-Agent 等机制不与执行计划同时变化。

## 与 EXG 的对应和差异

[EXG 原文](https://arxiv.org/html/2605.17721v1) §2–4 给出的参考预算为：每题最多两次尝试、向量种子 10、相似边一跳 fanout 5、anchor 最多取一个 case 并桥接 fanout 5、候选上限 30、相似度 prompt 权重 0.8、最多 5 条 hints。离线采用随机 7:3 收集与评估划分。

本实验保留这些预算作为初始参考，但明确记录以下差异：

- 生成模型换为用户指定的 Qwen3.8；存储和检索由 TriDB 提供。
- 原文没有明确在线初始图是否为空；空图是本实验显式协议。
- 原文未完整披露 QA 变体、题目清单、反馈与重试细节；本实验将这些固定并公开 manifest。
- LLM `SemFilter`、`SemExtract` 及其联合执行计划是 TriMEM 扩展，不称为原 EXG 已有算子。
- EXG 描述的候选全局重排与 TriDB 的 TR-1 流式约束需要专门对齐；不能直接复制“收集全部候选再排序”的阻塞实现。
- 原文离线主对比是 ExpeL 对 EXG-Reflexion，并使用 insights；本草案的离线主实验首先比较 TriMEM 计划。若要复现原文 Table 2，另加相应求解器和 insights 阶段，并计入其成本。

## 数据集与输入输出

文件已下载并校验；来源和 SHA-256 见 [数据清单](../exg/DATASETS.md)、[下载 manifest](../exg/download_manifest.json) 和 [准备记录](../exg/reproduction/20261011/preparation.json)。MuSiQue 和 HotpotQA 首轮各取 500 个任务；在线运行全部 500 个，离线沿用 7:3 协议划为 350 个收集任务和 150 个评估任务。在线、离线使用独立运行状态，不复用在线产生的图或答案。数据变体和输入处理仍为待讨论建议，不声称恢复作者题目子集。

| 数据集 | 建议主数据 | 模型可见输入 | 模型输出 | 质量指标 |
|---|---|---|---|---|
| HumanEval | 全部 164 题 | 原始函数签名、docstring、允许的本题反馈和 memory | 函数补全代码 | 首次测试通过率；两次内通过率 |
| HumanEval+ | 同一批 164 题，v0.1.10 | 同上，隐藏增强测试不进入 prompt | 函数补全代码 | 增强测试通过率，另报 base 结果 |
| MuSiQue | 首轮 500 个任务；在线 500，离线 350 收集＋150 测试；Ans v1.0 train 为建议来源 | question 和原始 paragraphs；移除 answer、aliases、decomposition、support 标签 | 短答案 JSON | answer EM、F1；EM 判定重试 |
| HotpotQA | 首轮 500 个任务；在线 500，离线 350 收集＋150 测试；distractor train 为建议来源 | question 和给定 context；移除 answer、supporting facts | 短答案 JSON | answer EM、F1；EM 判定重试 |

### 论文规模核验与修订记录

核验范围包括 PDF §4.1、§4.4、Appendix B/C、Figure 9，以及 arXiv 发布的 [实验 TeX](../exg/paper/20261011/source/s4-experiments.tex) 和 [附录 TeX](../exg/paper/20261011/source/s-appendix.tex)。

- MuSiQue 的 Figure 9（PDF 第 13 页）任务横轴及曲线到 500；它支持对齐这一在线学习曲线的规模，不能单凭此图认定所有模型/表格也使用 500。Table 4 是沿任务进程的图快照均值，其 491.8 cases 不是题目数。
- HotpotQA 的“over 100k”是公开 benchmark 总规模介绍；论文没有明确列出在线或离线所用子集题数。也不能从三位小数的通过率倒推出唯一分母。
- 离线正文只明确 7:3 比例，Table 2 报告 HumanEval 和 HotpotQA，没有 MuSiQue 离线结果。因此 MuSiQue 离线属于我们的扩展。用户现已选择 500 个任务，350/150 是本实验采用 7:3 的划分，不是论文原报数量。
- 原来的各 1,000 题、700/300 和相应全矩阵总量均撤销。v0.1.2 根据用户后续选择，将两种 QA 的首轮规模明确设为 500。未披露的作者 IDs、数据变体和采样配置继续标注差异，不阻止按已选规模讨论本实验。
- Ans/train 和 distractor/train 仅是先前建议，论文没有足够信息确认；保留下载的各公开变体，正式选集须记录差异。

QA 主实验使用给定事实段落，所有组事实输入一致；经验检索与事实检索分开。暂不引入 Wikipedia 检索器，避免把事实召回变化当成经验图收益。保留原始段落顺序；超长输入采用预先冻结的统一处理规则，不按实验组单独截断。

HumanEval 与 HumanEval+ 的题目 ID 在离线划分中绑定，运行状态相互隔离；不能跨评测器共享成功标签或已求解答案。四个 workload 分别建图，不让代码或 QA 数据跨 benchmark 混入。MuSiQue Ans/Full 不叠加当独立题目；HotpotQA 两种配置也不重复计数。公开隐藏标签 test 不用于本地质量主比较。

当前未恢复作者 IDs。首轮按用户选择的规模开展机制复现，在正式运行前固定选集、划分和顺序 manifest；不得将自行抽取的 500 题标为作者原始题目清单。不能根据模型表现选择“容易体现收益”的题目。将采样、任务顺序、生成随机性分别记录。主结果完成前，不用在线结果调参后再把同题的离线结果称为未见验证。

## 运行模型与环境

[历史本地运行协议](../exg/results/experience_graph_v0.1.0/protocol.json) 曾将 `qwen3.8-262k` 记录为 `Qwen/Qwen3.8-27B-FP8`，revision 为 `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`。这是历史证据，不是当前服务核验结果。正式运行前重新读取服务身份，冻结权重、tokenizer、量化、chat template、reasoning 配置、vLLM 版本与硬件；不只记录 model alias。

| 项目 | 草案默认值 |
|---|---|
| Generator | 同一 Qwen3.8 checkpoint；thinking on、medium；temperature 0.7、top_p 0.95 |
| 输出上限 | 每次生成 8,192 tokens，包含后端计入 completion 的 reasoning；pilot 若截断明显，在正式运行前统一修订 |
| Semantic 模型 | 同一 Qwen3.8；temperature 0；结构化 JSON；固定模板、输入字段和输出上限 |
| Semantic 输出上限 | extract 512、filter 128 tokens；thinking off 或后端支持的固定低成本配置，在 pilot 验证并冻结 |
| Embedding | 建议复用本地 Qwen3-Embedding-0.6B，固定 revision、维度、归一化和字段；这与 EXG MiniLM 有差异 |
| Context | 按已核验部署能力设置统一上限；主实验实际输入由任务材料和 memory token cap 约束，不将服务宣称上限当实测能力 |
| Hint 预算 | 最多 5 条，总 memory 最多 2,048 tokens；按固定顺序装配，记录丢弃和截断 |
| 并发 | 每个在线图只有一个顺序 writer；主延迟实验固定服务并发，记录排队时间；多 agent 吞吐另测 |

所有组使用相同 generator 配置、工具和任务反馈。资源分配按运行块轮换，避免一个计划始终运行在空闲服务、另一个始终与其他任务竞争。记录 CPU/GPU、显存、PG 版本、BLCKSZ、索引参数、存储、服务队列、prefix cache 和后台负载。stock PG 路径结果不构成 GX10 fork 验收。

## 成功判定与重试

QA 已确定采用 benchmark evaluator 反馈：首次回答后计算 EM/F1，agent 仅收到一个正确或错误信号；EM=0 或输出格式不合法时最多重试一次。首答正确则停止。F1 和正确答案不提供给求解器或 query-time semantic judge。因 EM 规则重试可能仍遗漏语义正确的别名，必须使用并校验数据集的官方 normalization/alias 规则。

代码任务建议沿用最多两次尝试。agent 只能收到成功状态和清理后的错误类别，不回传 canonical solution、隐藏测试源码、断言中的期望值。首次通过即停止。任何附加诊断调用都纳入 token 和调用计数。

这些运行属于使用 evaluator 反馈的受控实验，不能直接等同于未知真值的生产部署。No memory 组允许相同的本题历史和反馈，不使用跨任务经验。

工具基础设施错误、服务失败、解析失败和任务答案错误分别记录。基础设施失败不伪装成语义失败；保留不完整运行和可用性指标。预先固定超时及传输重试政策，所有真实调用计入成本；正式统计中不静默删除失败任务。

## 轨迹与经验图

原始任务只提供 Task。每次实际生成和评估后记录 Attempt，以及可观测的输入、输出、工具和反馈事件；一次 attempt 是 EXG case 的单位。

| 对象 | 必须保留的信息 |
|---|---|
| Task / anchor | dataset、原始 task ID、split、run、任务顺序、输入 hash |
| Attempt / case | attempt ID、序号、输出、成功标记、错误信号、版本、模型配置、时间、token |
| contain | anchor 到其 attempt；关系来源为确定的运行记录 |
| similar_to | case 间相似关系；向量和建边规则版本、权重、创建时间 |
| fixed_by | 同题先前失败到后续通过的尝试；需要明确反馈关联、相同 evaluator 和评分证据 |
| retrieval receipt | 查询快照、候选/返回 ID、顺序、实际 hints、选择的计划、各阶段工作量 |
| semantic derivation | case 版本、模板/模型 hash、lesson、判断、缓存依赖及 pending 状态 |

图拓扑使用 TriDB 原生邻接表，向量、属性和原始 case 在同一 PG 事务中发布。初版 lesson 只派生文本属性，不改变 similar_to/fixed_by 的定义，便于比较 eager/lazy。LLM 外部调用的等待不等于持有长事务：先计算或登记 pending，再用短事务做版本检查和原子发布；不得让读者看到不一致的图和属性。

similar_to 建边阈值未由论文明确给出。建议初始用新 case 到此前最多 10 个近邻的确定规则，阈值作为预先选定的敏感性参数；禁止沿用历史 0.75 阈值而不记录。去掉自环、保存双向物理邻接但按一个无向关系统计。不得因为图稀疏就临时补边以制造图收益。

## 在线协议

每个 dataset × arm × replicate 从空经验图、空 semantic cache 和空 pending 队列开始。模型常驻进程的预热与经验冷启动分开记录。

```text
读取任务 → 构造当前可见输入 → 读取已提交图快照
         → 检索并生成 hints → Qwen3.8 生成 → evaluator 反馈
         → 保存日志并发布新 case 和关系
         → 成功则进入下一题；失败且还有预算则进行第二次尝试
```

新 attempt 只能读取此前已经完成的经验，不能检索自己的未来结果。失败 attempt 写入后，可供同题第二次尝试检索。不同组的新经验彼此隔离，采用配对任务顺序和生成 seed；图随组而分化是在线策略效果的一部分。

记录每题开始前和每次提交后的 case/edge 数量、成功失败构成、累计成本。主图按已处理任务数呈现累计质量和成本；另外显示固定大小窗口的质量，避免累计平均掩盖局部变化。

## 离线协议

按 7:3 比例划分首轮题目池；代码建议 114/50，但其取整规则和作者 IDs 未恢复。MuSiQue 和 HotpotQA 各使用 500 个任务，分别划为 350 个收集任务和 150 个评估任务。这里的 500 是总题目池，不是 500 个测试题再另加收集题。MuSiQue 离线为新增实验；HotpotQA 500 题也不声称是作者原始规模。划分 seed 与顺序 seed 分离；第一轮使用固定划分、三个顺序/生成重复。不同划分的重复作为后续敏感性实验，不将重叠的测试题当独立新样本。

离线分成两个解释不同的比较：

1. **主比较使用共同源经验。** 每个 replicate 用冻结的 T 三模态策略在 collection 集上采集一次真实轨迹；各 memory arm 从相同原始轨迹建立自己的必要索引和派生状态。测试使用相同冻结 case、关系和反馈。No memory 不使用该经验。这样可以在相同经验内容下比较检索与规划。
2. **补充比较各自收集的完整系统。** 对选出的少量策略，分别从空图采集 70%，冻结各自图后评估 30%。它同时包含经验收集质量和测试检索效果，结果单独报告；这更接近比较不同 agent-memory 系统的两阶段表现。

离线测试不插入测试 case、关系、insights 或供后续测试使用的测试反馈。允许对冻结源 case 做版本确定的 lazy 派生，以及使用隔离的查询缓存；它们不得改变源图的逻辑内容。涉及测试答案和反馈的信息只用于本题允许的重试及事后评分。

共同采集成本、每组索引/派生成本、测试成本分开列出，再报告每组使用一次源经验的完整成本和摊销曲线。缓存、索引不能免费复制给一个组而让另一个组从零计算。主离线实验不是原文 ExpeL 对照的替代声明。

## Semantic 操作和执行计划

建议先只加入两个定义明确的操作，避免将自由的 agent reasoning 全部包装成数据库算子：

- `SemExtract(case)`：从已完成 attempt 的输入、输出和允许的反馈提取简短 lesson、适用条件与失败模式。不得读取 canonical solution 或未来 case。
- `SemFilter(query, case, lesson)`：判断该经验是否对当前求解有可迁移的帮助。不得访问当前题 gold 或未来评估结果。模型在固定 schema 下返回接受或拒绝；异常走固定处理规则。

返回顺序、字段和 hint 模板冻结。缓存键至少包含算子版本、模型与采样配置、query hash、case/lesson 版本和反馈可见性协议。温度 0 不保证绝对确定性；记录响应差异，使用独立校验轨迹上的固定 decision tape 检查物理等价性，正式运行仍报告实际模型调用及缓存成本。

| 组 | 检索与语义处理 | 构建方式 | 主要比较 |
|---|---|---|---|
| N | No memory，仅本题反馈 | 只记公共测量日志 | 无经验基线 |
| V | 向量检索＋必要属性限制，无图扩展 | 原始 case 与向量 | 是否需要图关系 |
| T | V/G/R 三模态，无 LLM semantic op | 原始 case、向量、图 | 三模态经验基线 |
| S0 | 三模态＋固定 semantic-first 的合法残余过滤顺序 | eager lesson | 固定 semantic 基线 |
| S1 | 同一逻辑查询，固定 cheap-filter-first | eager lesson | 手工优化基线，避免只击败明显低效计划 |
| S2 | 同一逻辑查询，成本驱动选择合法过滤顺序、消费批量和已验证的等价访问路径 | eager lesson | 物理 planner 收益 |

S0、S1、S2 都支持 Open/Next/Close、LIMIT 和取消传播；不设置先耗尽全部候选的阻塞基线。权限、dataset/split、可见性和版本隔离检查始终先执行，不作为可移到 LLM 后的过滤器。可交换的是已证明纯净、独立的残余属性谓词；失败 case 不能一律被 cheap filter 删除，因为 warning 本身就是经验。

S0–S2 必须使用同一候选宇宙 U、相同逻辑顺序和预算，返回 U 中最先满足过滤条件的最多 H 条 hints。不能让一种计划先 cap 再过滤，另一种先过滤再 cap。不能把会切断修复路径的节点过滤随意下推到图遍历内部。改变 ANN 访问路径若改变召回/顺序，归入策略实验而非等价计划。

**排序约束需要在实现前解决。** 初版建议采用原生游标的确定消费顺序，明确记录它与 EXG 全局 relevance sort 的差异。完整复现其排名，只有在能证明排名界限、正确停止且符合 TR-1 时才纳入主计划。候选上限 30 并不豁免阻塞排序。不得把离线全量排序审计工具当作线上算子。

完成基本比较后，再增加两个独立扩展，不立即做全参数笛卡尔积：

- **构建扩展：** 在 S1 或 S2 固定检索策略下比较 eager、lazy、adaptive derivation；lazy 的强读为真正需要的源 case 完成派生。写入确认、派生就绪、pending、后台及最终补齐成本分别计量。先确保所有模式在约定读语义下返回一致内容。
- **下游成本扩展：** 在 S2 上允许选择 H∈{0,1,3,5}、memory token cap 和其他已定义策略。用独立校准数据估计下游成本，或仅用当前运行此前已完成的事件更新；禁止窥视未来任务结果。它可能改变质量，必须验证质量约束或 Pareto 取舍，不再称为物理等价重写。

现有 [SemPlanner](../bench/agent_memory/semantic/planner.py) 主要选择谓词顺序，[semantic stream](../bench/agent_memory/semantic/stream.py) 从 PG portal 分批消费。它们不是已完成的全计划空间优化器。若 pilot 中所有查询都选择 S1，应该报告这一结果；不能为展示“自适应”强行切换计划。

## 成本与质量统计

| 层次 | 记录指标 | 计量边界 |
|---|---|---|
| 质量 | 首答通过率、两次内通过率、最终答案质量、QA EM/F1、格式失败率 | 首答和最终提交分别记；不做事后挑最好答案 |
| 检索 | p50/p95、首条结果延迟、完整 hints 就绪时间、rows/edges、semantic calls、cache hit、underfill、stop reason | 同时报原生 DB 时间、embedding、semantic、装配和总检索时间 |
| 构建 | commit latency、完整就绪延迟、case/s、向量/边写入量、派生调用与 tokens、存储/WAL 增量 | 原始写入和 semantic 派生分别记；lazy 后台与补齐工作纳入 |
| 生成 | 各 attempt TTFT、生成时间、输入/输出 tokens、实际重试、反思/诊断调用 | 排队、工具和模型时间分别记 |
| 整体 | 每题/全序列总 wall time、总 tokens、总调用、成功数、截止预算下质量 | graph 构建与测试采集成本说明摊销范围 |

token 总量为所有真实、计费口径的输入和输出之和，按 generator、semantic-read、semantic-write、额外诊断分桶；memory 是 generator input 的子集，不重复相加。reasoning 通常是 completion 的子集，按服务实际字段核对。embedding token 和资源成本单列，不直接与不同模型 token 等价计价。失败、超时和重发的实际模型消耗保留。

对每题记 `N_attempts`、`N_llm_calls` 和 `retry_trigger`，三者不能混同。端到端 wall time 从事件时间戳计算关键路径，不直接相加存在重叠的阶段均值。

主质量指标采用经验性的 first-attempt success 和 success within two attempts，避免与独立采样的组合式 pass@k 估计器混用。QA 同时报 final-answer F1。允许不增加质量但降低成本，也允许提高质量且增加成本，结论用实测 Pareto 关系表达。

固定任务集合下报告总成本；固定 token/time 预算下，同时报告完成覆盖率和按完整集合分母计算的成功率。未到达任务和超时保留为未完成，避免便宜计划只做容易题。原始进度曲线用于定位 cold-start 代价与回报；短序列无法展示回报时如实保留。

## 控制变量与分析

每个配对 replicate 的任务 IDs、顺序、模型、生成参数、反馈、工具预算、hint 格式与服务资源规则一致。不同组严格隔离图和 cache；所有主动预热单列。在线自适应统计仅使用已观测信息，禁止从其他组复用免费 semantic 结果或未来成功标签。

建议 3 个独立顺序/生成重复用于第一轮；不是统计功效保证。在线任务通过共享图相互依赖，不能把每次 attempt 当独立样本做普通 bootstrap。首先报告逐运行配对差值、跨运行变化和曲线；需要可靠显著性判断时根据 pilot 方差增加完整运行重复。离线条件于固定图的逐题配对区间，与图构建重复的变异分别展示。不同数据集质量指标不直接平均。

S2 同时报告选择频率、估计与实测成本、相对 S1 的收益。离线保存的相同快照查询可以执行候选计划审计，估计其相对实测最优计划的 regret；额外审计成本单列，不向在线运行提供这些未来测量结果。candidate/output hash 和 hints token 序列用于检查物理等价性。

质量非劣界限、目标延迟/成本及预算上限需要在正式结果前冻结；草案不凭空指定“必须节省 20%”。如果三模态或 semantic 组无收益，也是有效结论。输出计划空间不足、统计不足和模型截断等解释限制。

## 执行阶段与预计规模

1. **协议冻结和工程核验。** 固定任务 manifest、反馈模板、模型身份和指标定义；核验原生图、事务可见性、跨组隔离、评测实现、early stop/cancel 与非阻塞排序方案。当前尚无本实验完整可运行入口；不提供虚构命令。
2. **小规模 pilot。** QA 使用正式样本之外的 train 题；代码使用独立合成/示例题，避免调参消费 HumanEval 正式题。验证字段、截断、token 核算、semantic 选择率、图密度和成本量级，随后冻结正式配置。
3. **主矩阵。** N/V/T/S0/S1/S2，四个 workload、在线及共同源图离线、三个重复；先小批检查产物完整性，再推进已冻结矩阵。
4. **定向扩展。** 根据主结果选择少量组合做 construction 策略、完整系统离线和下游预算选择；不根据测试集结果反复调参后仍使用同一测试成绩作为最终证据。

按首轮 QA 各 500 题、代码各 164 题，若采用六组和三个重复：在线每个 arm/replicate 为 1,328 个 task episodes，共 23,904；离线测试每个 arm/replicate 为 400 个 episodes，共 7,200；共同 collection 每个 replicate 为 928 个，共 2,784。总计 33,888 个 episodes，最多 67,776 次 solver attempts；semantic 调用另外计算。上述规模不含 pilot、各组独立 collection 或扩展实验，六组/三重复仍为建议。pilot 用于估算预算与时长；当前只更新计划，不启动生成或构图。

超时建议初值：LLM 单请求 180 秒；代码评测使用已校验的 per-test 与总预算；单题总预算在 pilot 后统一冻结。大规模运行按任务边界做可恢复 checkpoint；恢复复用已提交输出，不重新生成后覆盖旧成本。监控进程存活、成功提交进度、队列、剩余磁盘与 pending；基础设施问题以可追踪事件记录。

## 预期产物

建议使用新目录 `results/trimem_exg_qwen38_v0.1.0/`，不覆盖历史 `exg/results/experience_graph_v0.1.0/`。

| 产物 | 内容 |
|---|---|
| protocol.json / environment.json | 冻结的所有条件、硬件、模型/代码版本、差异声明 |
| task_manifest.jsonl | 输入文件 hash、task ID、partition、group、顺序；gold 引用仅 evaluator 可读 |
| attempts.jsonl / llm_calls.jsonl | 真实生成和调用、反馈、token、时间、失败和重试归属 |
| query_receipts.jsonl / ingest_receipts.jsonl | 图快照、计划、候选/返回 ID、缓存与派生、阶段成本、停止原因 |
| graph_checkpoints / validation.json | 恢复点、节点关系计数、可见性与输出等价检查 |
| summary.csv / report.pdf | 各组质量/成本、运行差异、异常、可视化和复现限制 |

预期图包括：quality–token 与 quality–latency 散点；在线累计和窗口质量曲线；总 token 分解；构建/检索/生成延迟分解；图规模与查询成本；eager/lazy 的 pending 与总成本；planner 选择和 regret。图表来自实际 receipts，不填补未完成运行，不把估计画为实测。

## 接下来讨论的选择

QA 反馈和首轮各 500 个任务已确定。接下来讨论是否保留六组主矩阵、semantic 范围、当前 Qwen3.8 配置、具体题目 manifest 及离线共同源图协议。参数确认后形成 v0.2.0 冻结协议，之后再实现和运行。
