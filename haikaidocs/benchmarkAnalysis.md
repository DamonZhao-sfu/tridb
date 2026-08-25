# 自改进 / 自演化 LLM Agent 的实验负载与基准全景（截至 2026 年 8 月）——为 TriDB 跨模态查询优化器读者定制

## TL;DR

- **现有文献里没有任何一个现成负载能直接证明"图+向量+关系"三模态数据库作为 agent 经验底座的价值**：绝大多数论文的经验池只有几十到几百条（Evo-Memory 每个多轮环境 60–134 个任务、SWE-bench Verified 500 例、AlphaEvolve/DGM 种群仅几十到几百），暴力扫描或 top-k 向量检索就够用，索引 / 关系连接根本不是瓶颈。唯一天然带"树+分数"结构、且规模稍能撑起数据库论证的是 AlphaEvolve / DGM 这类演化搜索，但它们的存储层就是一堆代码目录 + JSON/pickle + checkpoint，没人把它当数据库问题来研究。
- **对无 API 预算、只有 2×RTX PRO 6000（192GB）的你**：非权重更新类里 AWM、ExpeL、ACE、ReasoningBank、Evo-Memory、OpenEvolve/ShinkaEvolve、LifelongAgentBench、G-Memory 都能用开源权重跑；权重更新类的 agentic RL（GRPO/SAO 等）在 SWE-bench 上单次训练动辄上万美元、需要大集群，本地不可行；而多数记忆基准（LoCoMo、MemoryArena、Evo-Memory）默认用 GPT-4o/Gemini/Claude 做 judge 或 backbone——**judge 这一环是最容易被"协议锁定"卡住的地方**，但通常可换成本地 gpt-oss-120b 或 Qwen 当 judge。
- **最接近你需求的现成工作**：ReasoningBank（2509.25140）+ Evo-Memory（2511.20857）+ G-Memory（NeurIPS'25, 2506.07398）三者合起来提供了"跨会话经验复用 + 图结构记忆 + 向量检索"的雏形，但都缺规模（万级以上节点）和关系型查询负载。要撑起 TriDB 论文，你需要把 Evo-Memory 的流式任务规模放大 1–2 个数量级，并把 G-Memory 的三层图 + AlphaEvolve 的评分节点显式建成需要"向量种子 + 图遍历 + 关系过滤"混合查询的负载。

---

## Key Findings

1. **实验规模普遍很小**。这一整个领域的"经验池"几乎都能塞进上下文或用暴力 top-k 向量检索搞定：
   - Voyager 技能库：top-5 检索，技能以 JS 代码存，key 是描述的 embedding（向量库），约 160 次迭代累积（原文未给最终技能整数）。
   - ExpeL：HotpotQA/ALFWorld/WebShop 三个域，经验池是"成功轨迹 + 自然语言 insights"扁平列表，规则数上限约 8–10 条。
   - AlphaEvolve / OpenEvolve：MAP-Elites + 岛屿模型，OpenEvolve 默认 population=500 / 5 islands；DeepEvolve 甚至只有 ≤25 程序 / 5 岛 + 10 格 archive。ShinkaEvolve 一次跑仅约 150 次评估（主打样本效率）。
   - DGM：80 次迭代 ≈ 80 个 agent 节点的演化树。
   - Evo-Memory：每个多轮环境 60–134 个任务。
   - **只有真实 AlphaEvolve 部署会到"数千"程序**——Google DeepMind《Building Production-Ready Probes For Gemini》(arXiv:2601.11516) 明确写"a single large AlphaEvolve job that generates and evaluates approximately 2500 probing architectures"——但这仍是万级以下。
2. **结构上，演化搜索类天然是"带分数的树/DAG"，记忆类多是"扁平列表或向量集合"，少数是"图"**。AlphaEvolve/DGM 每个节点存 fitness/score 并有 parent-child 血缘；G-Memory 是三层图（insight / query / interaction，节点带 Failed/Resolved 状态），做双向遍历检索；ReasoningBank/AWM/ACE 基本是可增删的结构化条目集合。
3. **检索机制以 embedding 相似度 top-k 为主**。真正做到"向量种子 + 结构遍历"（vector-seeded graph expansion）的只有 **G-Memory 的双向图遍历**最接近：先按 query 相似度找到相关历史 query 节点，再沿图取其高层 insight 和细粒度轨迹。没有任何公开负载同时要求"向量检索 + 图遍历 + 关系过滤"三者混合。值得注意的是 ReasoningBank 的消融显示 top-k 越大反而越差（k=1 成功率 49.7%，k=4 降到 44.4%），说明当前规模下"检索多"根本不是需求——这恰恰反衬出规模不足。
4. **测量的东西大多只有最终成功率/准确率**。少数（ACE、ReasoningBank、AWM）报告了 token 成本、adaptation 延迟、步数；MemoryArena 报 latency；AlphaEvolve 报 wall-clock 与 throughput。几乎没人把存储层 / 检索本身当作被测对象。
5. **存储基本不是数据库，而是文件 + 对象**。演化系统 = 一堆代码目录 + JSON/pickle 元数据 + 周期性 checkpoint（非 SQL）；记忆系统 = 向量库（FAISS/Chroma 级）+ JSON。只有综述 / 架构论文（如 Aeon 2601.15311、agent memory 综述 2603.07670）才把 FAISS/HNSW ANN 检索在大规模下退化（"Vector Haze"：语义相似但上下文无关）当成问题提出，但这些不是本文关心的自演化实验负载。
6. **确实出现了一批"专为跨会话/跨任务经验复用"而生的基准**（2025–2026）：Evo-Memory、MemoryArena、MemoryAgentBench、LifelongAgentBench、EvoMemBench、WorldMemArena、SEA-Eval 等。但它们规模都不大，且多数默认依赖闭源 API judge。

---

## Details

### A 类：非权重更新型自改进

| 论文 (arXiv / venue) | 基准 / 数据集 | 经验规模 | 结构 | 检索 | 测量 | 本地可跑? |
|---|---|---|---|---|---|---|
| **Voyager** (2305.16291) | Minecraft (MineDojo) | ~160 迭代累积技能；top-5 检索 | 技能=代码，vector-indexed 扁平库 | dense embedding top-5 | 独特物品数(3.3×)、里程碑速度(15.3×) | 环境可本地跑，原文用 GPT-4；换开源权重可跑 |
| **ExpeL** (2308.10144) | HotpotQA, ALFWorld, WebShop | 经验池 + ≤8–10 条 insight 规则 | 扁平：成功轨迹 + NL insights | task 相似度检索 + insight 注入 | 成功率 | 可，轻量 |
| **Reflexion** (NeurIPS'23) | ALFWorld, HotpotQA, HumanEval | 每任务口头反思记忆(小) | 扁平 episodic | 全上下文 | 成功率 / pass@1 | 可 |
| **AWM** (2409.07429, ICML'25) | WebArena, Mind2Web(1000+任务/200+域) | 在线/离线诱导 workflow；WebArena map 约 40 query 即收敛 | workflow 结构化条目 | 选择性注入 | 成功率(+51.1% 相对)、步数↓ | WebArena 自托管本地可跑；原文 GPT-4o |
| **ACE** (2510.04618) | AppWorld, 金融推理 | evolving playbook(增量条目) | 结构化"playbook"条目，增删改 | 上下文 + curation | 成功率(+10.6%)、adaptation 延迟、rollout 成本、token | 可，原文用 DeepSeek-V3.1 开源(671B，需量化或多卡) |
| **ReasoningBank** (2509.25140) | WebArena, Mind2Web, SWE-bench Verified(500) | 跨任务流累积 memory items(无公开总数)；默认检索 k=1 | {title, description, content} 结构化条目 | embedding 相似度 top-k | 成功率(up to +34.2% 相对)、步数↓16% | 原文 Gemini/Claude；逻辑可移植开源；WebArena 本地可跑 |
| **G-Memory** (2506.07398, NeurIPS'25) | ALFWorld, SciWorld, PDDL, HotpotQA, FEVER | 五基准；三层图 | **图(insight/query/interaction 三层，节点带 Failed/Resolved 状态)** | **双向图遍历(向量种子+结构)** | 成功率(具身 +20.89%，如 ALFWorld 58.21→79.10%；QA +10.12%) | 可 |
| **AFlow** (2410.10762, ICLR'25) | HumanEval, MBPP, MATH, GSM8K, HotpotQA, DROP | MCTS 树，节点=工作流代码 | **树，节点带分数** | MCTS 软最大父采样 | solve rate / pass@1 / F1；成本对比 | 可，executor 用小开源模型 |
| **ADAS** (2408.08435) | ARC, 阅读理解, 数学等 | meta-agent 迭代 archive | archive 列表，存代码+分数 | 线性启发式 | 准确率 | 可 |
| **Darwin Gödel Machine** (2505.22954) | SWE-bench Verified, Polyglot | ~80 迭代 ≈ 80 agent 节点 | **演化树，节点=agent 代码目录+JSON 元数据，存 SWE 分数** | 从 archive 采样(open-ended) | SWE 20→50%, Polyglot 14.2→30.7%；单 run 约 $22,000/约 2 周 | **不可**(成本/规模)；但存储结构对你有参考价值 |
| **AlphaEvolve** (2506.13131) | 矩阵乘法, Erdős 等数学 + Google 工程负载 | 数千代；真实作业约 2,500 程序 | **MAP-Elites + 岛屿，节点带多维 fitness，parent-child 血缘** | quality-weighted 采样 + embedding 去重 | 目标分数、wall-clock、throughput | 闭源(Gemini Flash+Pro)；用 OpenEvolve/ShinkaEvolve 开源替代 |
| **OpenEvolve / ShinkaEvolve** (2509.19349) | 圆填充、算法发现等 | pop=500/5岛(OE)；Shinka 约 150 评估 | MAP-Elites 网格 + 岛屿，存分数 | 岛屿 + MAP-Elites 采样 | 目标分数、样本效率 | **可**，开源；存储=目录+JSON/pickle+checkpoint |
| **AI Scientist v1/v2** (2408.06292) | 自定 ML 研究任务 | 研究轨迹(小) | 树/链 | — | 论文质量(LLM/人评) | 部分可；评审环节多用闭源 |

### B 类：权重更新型自改进（agentic RL / 自训练）

| 论文 (arXiv / venue) | 基准 | 经验/rollout 规模 | 结构 | 学习方式 | 测量 | 本地可跑? |
|---|---|---|---|---|---|---|
| **SAO** (2607.07508) | SWE-bench Verified, BeyondAIME, IMOAnswerBench | 千步训练，单 rollout/prompt | rollout 轨迹(瞬态) | single-rollout 异步 RL(value model) | 成功率、训练稳定性；部署于 GLM-5.2(750B) | **不可**(巨型模型/集群) |
| **DeepSWE / SWE-World** (2602.03419) | SWE-bench, R2E-Gym/SWE-Gym/SWE-rebench | Docker-free rollout 池 | 轨迹 | GRPO++ from self rollouts | 成功率 | 32B 级勉强，需大显存 |
| **通用 agentic RL** (GiGPO/ARPO/SkyRL 等) | ALFWorld, WebShop, tool use | 多轮 rollout | 轨迹 | GRPO 变体 | 成功率 | 小模型(≤7B)可，受显存限 |
| **Structured Agent Distillation** (2505.13820) | ALFWorld(8055 轨迹), WebShop, HotpotQA | 数千轨迹蒸馏 | 轨迹数据集 | SeqKD/token-KD | 成功率 | 可，蒸馏到小模型 |
| **Ornith-1.0/1.5** (DeepReinforce, 2026) | Terminal-Bench, SWE-bench Verified, DeepSWE | self-scaffolding RL rollout | 轨迹+自生成 harness | GRPO，自搭 scaffold | Terminal-Bench 86.1, SWE 86.0 | 9B/35B 变体本地可跑推理；训练不可 |

### 专为"跨会话/跨任务经验复用"而生的基准（重点）

| 基准 (arXiv) | 定位 | 规模 | 结构/存储 | 是否需闭源 API |
|---|---|---|---|---|
| **Evo-Memory** (2511.20857, UIUC+DeepMind) | 流式测试时学习+自演化记忆 | 10 数据集；多轮 ALFWorld 134 / BabyAI 112 / ScienceWorld 90 / PDDL 60 任务 | 任务重构成序列流；ExpRAG 存 ⟨x,ŷ,f⟩ 元组 | **是**(Gemini + Claude API，无开源权重) |
| **MemoryArena** (He et al., 2026) | 多会话相互依赖 agentic 记忆 | 766 个多会话任务 | Memory-Agent-Environment 闭环；Task Progress Score | judge 多用 GPT；可换 |
| **MemoryAgentBench** (He et al., 2025) | 检索/测试时学习/长程理解/冲突消解 | 512/4096 token 增量分块 | 四能力切分 | LLM judge |
| **LifelongAgentBench** (2505.11942) | 终身学习 | DB/OS/KG 三环境，技能接地互依赖任务 | Docker 化，自动标签验证 | **可本地**，自动验证不靠 judge |
| **EvoMemBench** (2605.18421) | 自演化视角的记忆 | 跨 episode：BFCL-Multiturn, WebWalkerQA(170样本), ALFWorld | 跨 episode 记忆演化 | 部分 |
| **WorldMemArena** (2605.29341) | 多模态 agent 记忆(写/维护/检索/用) | 分解式评估 | 定位失败来源 | 部分 |
| **SEA-Eval** (2604.08988) | 超越 episodic 的自演化评估 | — | — | — |

### 与 Meta Trellis / KernelEvolve（已知背景，不复述）访问模式的对照

你关心 Trellis 声称的四种访问模式在公开负载里被"跑到"了没有：

- **Frontier selection（前沿选择）**：被 AlphaEvolve/OpenEvolve/DGM 的 MAP-Elites+岛屿采样、DGM open-ended archive 采样直接练到了——这是它们的核心循环。但规模只有几十到几百节点。
- **Cross-session reuse as vector-seeded graph traversal（向量种子图遍历的跨会话复用）**：**只有 G-Memory 的双向图遍历真正接近**（向量找 query 节点 → 沿图取高分 insight/轨迹）；ReasoningBank/Evo-Memory 是纯向量 top-k，没有图遍历。没有一个公开负载把它做成大规模、需要索引优化的查询。
- **Training-data extraction as materialized view（训练数据抽取=物化视图）**：B 类的 trajectory distillation（SWE-World、Structured Agent Distillation）在概念上就是"从 rollout 池抽训练集"，但没人用物化视图 / 增量视图的数据库语义来表达或优化它。
- **As-of time travel（按时间旅行查询）**：公开负载里**基本没有**。DGM/AlphaEvolve 的 archive 有天然的迭代序 / 血缘，理论上支持"回到第 k 代"，但没有任何实验把"历史版本查询"当成负载来压测。

---

## Recommendations（给 TriDB 的分阶段建议）

**阶段 0 — 立即可做（本地零 API）：**
- 以 **LifelongAgentBench**（2505.11942）为主力功能负载：它三环境里就有 **Database 环境**，天然贴合 DB systems 叙事，且自动标签验证、不依赖闭源 judge，可完全用开源权重跑。
- 用 **OpenEvolve / ShinkaEvolve**（开源 AlphaEvolve）作为"带分数的树/DAG + 前沿选择"负载，直接把它们的目录+JSON 存储替换成 TriDB，测量 frontier-selection 查询延迟。这是唯一你能本地跑、又天然带 fitness 节点血缘的负载。

**阶段 1 — 构造你缺的混合查询负载：**
- 把 **G-Memory**（2506.07398）的三层图搬到 TriDB，把它的"双向遍历检索"实现成显式的 **vector-seeded graph expansion** 查询（向量找种子 → 图遍历取高分节点 → 关系过滤 status=Resolved），这正是你 cost-based 跨模态优化器要证明的场景。G-Memory 有开源实现、可本地跑。
- 把 **Evo-Memory**（2511.20857）的流式任务放大：原生每环境仅 60–134 任务，你需要合成 / 复制到**万级任务流**才能让"索引检索 vs 暴力扫描"出现拐点。把 backbone/judge 换成本地 gpt-oss-120b 或 Qwen3 以绕开 Gemini/Claude API 依赖。

**阶段 2 — 撑起 SIGMOD/VLDB/CIDR 论证：**
- 制造一个规模拐点实验：经验节点从 10² 扩到 10⁶，展示纯向量 top-k（FAISS 平表）出现 "Vector Haze"（相似但无关）退化，而 TriDB 的图+向量+关系混合查询在 recall 和延迟上都更优。这一退化现象在 agent memory 架构论文（Aeon 2601.15311、综述 2603.07670）里已被点名，但没人用真实自演化负载压测过——这是你的空位。
- 报告的指标要超越成功率：必须报 **检索延迟、召回质量、token 成本、收敛速度、存储层吞吐**，因为现有论文几乎不报这些，你天然有增量贡献。

**会改变建议的阈值 / 信号：**
- 如果出现一个**公开、开源权重可跑、且经验节点上万**的跨会话基准（目前不存在），优先直接采用而非自造。
- 如果 Evo-Memory 或 MemoryArena 后续版本放出**开源 judge + 大规模任务流**，则以其为标准负载可增强可比性。
- 如果本地 judge（gpt-oss-120b/Qwen）与 GPT-4 judge 的评分相关性 <0.8，需回退到有确定性验证器的负载（LifelongAgentBench、SWE-bench 单元测试、AlphaEvolve 打分器）以避免 judge 噪声污染结论。

---

## Caveats

- **规模是最大硬伤**：整个文献里"经验池够大到让索引 / 数据库论证成立"的负载基本不存在；AlphaEvolve 真实部署的约 2,500 程序是上限，且闭源。你几乎肯定要自己合成规模。
- **judge 协议锁定风险**：Evo-Memory、MemoryArena、多数 LoCoMo 系记忆基准默认用 GPT-4o/Gemini/Claude 当 judge 或 backbone。功能上可换本地模型（已有论文用 gpt-oss-120b 当 judge 的先例），但"官方可比分数"会失去，审稿人可能质疑。带确定性验证器的负载（LifelongAgentBench、SWE-bench、演化打分器）对你更安全。
- **B 类权重更新几乎全部本地不可行**：SWE-bench 上的 agentic RL 单次训练成本高（DGM 单 run 约 $22,000）、需大集群；你的 192GB 只够跑 ≤32B 级模型的推理或小规模蒸馏。
- **部分 arXiv 编号为 2026 年预印本**（如 2602/2604/2605/2606/2607 开头），这些是较新的 workshop/预印本，结论稳定性低于已被 ICLR/NeurIPS/ICML 正式接收的论文（AWM=ICML'25, AFlow=ICLR'25, G-Memory=NeurIPS'25, DGM=已接收；"A Survey of Self-Evolving Agents" 2507.21046 已被 TMLR 2026 接收）。
- **几个数字未在原文以单一整数给出**：DGM archive 大小（~80 是"80 迭代 × 1 agent"推算，仅计入功能正常的子代）、AlphaEvolve/ReasoningBank 累积条目总数、Voyager 最终技能数，均无干净的官方整数，已在正文标注。

---

## 专项深读：跨会话/跨任务经验复用 benchmark 对 GEM 的可复用设计（2026-08-24）

### 结论

“专为跨会话/跨任务经验复用而生”的七个 benchmark 不能被当作同一种负载。它们实际覆盖的是四个不同问题：

1. **显式跨 session 因果依赖**：MemoryArena；
2. **跨 task 的 procedural/skill transfer**：LifelongAgentBench、Evo-Memory 和 EvoMemBench 的 execution tracks；
3. **记忆底座的写入、更新、检索与使用正确性**：WorldMemArena、MemoryAgentBench；
4. **长期演化曲线与抗干扰能力**：SEA-Eval。

对 GEM 最有价值的组合不是选择其中一个，而是：

- 用 **EvoMemBench CrossEp-Know/CrossEp-Tool** 提供跨 episode 的端到端任务收益；
- 用 **MemoryArena** 提供 paper-protocol 的跨 session prefix dependency 诊断；
- 用 **LifelongAgentBench** 提供无闭源 judge 的跨 task outcome；
- 用 **WorldMemArena** 提供 Write → Maintain → Retrieve → Use 生命周期标注；
- 用 **SEA-Eval** 提供纵向收敛和干扰序列；
- 用 GEM 已有 **EvoTrace** 提供 reward、失败节点、谱系和 dead-end ground truth。

最终应当形成四条相互分离的 benchmark track，而不是把所有分数混成一个总分：因果复用、技能迁移、记忆生命周期正确性、数据库规模与性能。

### 1. 七个 benchmark 的实际适用边界

| Benchmark | 实际测量对象 | 对 GEM 最有用的部分 | 不应怎样使用 |
|---|---|---|---|
| [Evo-Memory](https://arxiv.org/abs/2511.20857) | 固定任务流中的 Search → Synthesize → Evolve | 顺序消融、成功/失败经验、memory budget、easy→hard / hard→easy 顺序测试 | 没有显式 dependency graph；不能用它验证图查询正确性 |
| [MemoryArena](https://arxiv.org/abs/2602.16313) | 后续 session 依赖早期 session 的事实、约束或中间结果 | 跨 session prefix dependency 诊断；Task Progress、SR@depth | 公开数据没有逐 decision 的最小 evidence edge，不能把 all-prior prefix 称为人工标注的最小 evidence |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257) | chunk 注入后的准确检索、test-time learning、长程理解和冲突处理 | 时间版本、supersession、stale-fact rejection、检索底座质量 | 多数任务是“先写完、后查询”，不能独自证明 experience reuse |
| [LifelongAgentBench](https://arxiv.org/abs/2505.11942) | 不同任务共享 SQL/Bash/KG 技能，复用历史成功轨迹 | 跨 task procedural reuse；自动验证；Task→Skill→Experience 建模 | 不能称为持续世界状态：DB/OS 环境按任务重置 |
| [EvoMemBench](https://arxiv.org/abs/2605.18421) | in/cross episode × knowledge/execution 六种设置 | CrossEp-Know/CrossEp-Tool 的端到端 reuse outcome、跨环境迁移、agent 与 memory token accounting | 没有 source-episode relevance 金标；官方仓库是六套 runtime，且固定版本没有顶层 LICENSE |
| [WorldMemArena](https://arxiv.org/abs/2605.29341) | Write、Maintain、Retrieve、Use 四阶段记忆生命周期 | Experience Graph schema、update/conflict/evidence ground truth、标准 adapter 接口 | 多模态全量实现不应成为 GEM 第一阶段的门槛 |
| [SEA-Eval](https://arxiv.org/abs/2604.08988) | 重复任务、相似任务和干扰任务组成的纵向序列 | token/step 收敛、迁移、抗干扰指标 | 官方仓库当前没有可用 benchmark 实现，只能借协议和指标 |

#### 1.1 Evo-Memory：适合借用流式协议，不适合提供 Experience Graph ground truth

Evo-Memory 将 memory agent 形式化为检索、更新、综合和控制组件，并在每个任务后把
`<input, output, feedback>` 经验写回 memory。ExpRAG 默认是扁平向量 top-k，ReMem 则增加
Think/Act/Refine 的记忆整理动作。它的强项是可以系统地测：

- memory-off 与 memory-on 的累计 outcome 曲线；
- easy→hard 与 hard→easy 的顺序效应；
- 只存成功经验、成功+失败经验和噪声经验的差异；
- top-k、memory budget、pruning 对准确率、成功率、step efficiency 的影响；
- task similarity 与 transfer gain 的关系。

但其数据通常只是把静态 dataset example 排成 stream，前后任务共享分布或策略，不存在可审计的
“session A 是 session B 的必要前置条件”边。因此它适合验证 **experience reuse 是否改善后续
outcome**，不适合验证 GEM 的 dependency traversal 或 evidence-chain recall。论文也没有给出
retrieval relevance label；这些标签需要 GEM 自己从 reward、dependency 或 paired outcome 构造。

截至本次核查，没有找到论文作者提供的正式官方实现链接；网络上可找到的 Evo-Memory 项目主要是
第三方复刻，不能作为 paper-faithful implementation baseline。

#### 1.2 MemoryArena：最强的真实跨 session 因果负载

MemoryArena 包含 bundled web shopping、group travel、progressive search 和 formal reasoning 等
多 session 任务。其关键不是 session 数量，而是数据构造时显式审核了依赖关系：

- shopping 后续选择依赖先前购买物的 compatibility；
- travel 需要跨 session 汇集 JOIN/RELATION 约束，dependency depth 可分层；
- progressive search 每个子查询增加新约束，且禁止使用未来 session 才给出的信息；
- formal reasoning 将 lemma/proposition 按依赖顺序排列，并剔除 forward dependency。

每个 task episode 从空 memory 开始；每个 subtask 是一个 session，session 结束后原 trace 不再直接
可见，只能通过 persistent memory 访问。官方协议在 agent action 前调用 retrieval，在 session 完成后
调用 update。因此它适合作为 GEM 的 cross-session dependency 诊断负载，并可直接生成：

- paper protocol 要求的 all-prior-session prefix；
- `PROTOCOL_PRECEDES/PROTOCOL_DEPENDS_ON` edge；
- session ordinal 与 prefix depth；
- cutoff 前的合法历史集合；
- Success Rate、Task Progress Score 和 SR@dependency-depth。

但截至固定的公开 Hugging Face release，数据只包含有序 `questions/answers`，没有逐 decision 的最小
source-session evidence edge。因此 all-prior prefix 只能报告 **Dependency Recall**，不能称为人工标注的
最小 Evidence Recall；后者必须由独立标注产物补齐。

[官方 MemoryArena 仓库](https://github.com/ZexueHe/MemoryArena)明确标注为 preview。当前实现中的
memory server 以 Python 进程内字典持有实例，RAG backend 多为小规模内存检索；GraphRAG 路径还存在
构造参数不匹配、每次 add 重建 workspace、subprocess 状态未充分检查和 placeholder query 返回等问题。
因此应复用其 **dataset、environment、session protocol 与 evaluator**，不应把其 storage backend 当作
可信的系统 baseline。GEM adapter 应直接落 PostgreSQL/TriDB，不复制这个 sidecar 设计。

#### 1.3 MemoryAgentBench：是 substrate benchmark，不是 cross-session reuse 主证据

MemoryAgentBench 将长文本或多 session history 切成 chunk 顺序注入 memory，然后集中执行 query。
它覆盖 Accurate Retrieval、Test-Time Learning、Long-Range Understanding、Conflict Resolution 等能力。
对 GEM 最有用的是：

- gold evidence/chunk 的 retrieval hit；
- LongMemEval 风格的长历史规模；
- FactConsolidation 中新事实覆盖旧事实的 serial order；
- stale fact rejection、temporal cutoff 和 supersession 正确性；
- memory construction time、query latency、输入输出 token。

但这类“inject once, query many”负载没有 agent action、environment feedback 和后续任务 outcome，不能单独
支持“跨 session 经验复用让 agent 变好”的主张。仓库已有 LongMemEval/MemoryAgentBench 路径时，应把它
保留为 **Track C 的 memory substrate correctness**，而不是 headline agent workload。

#### 1.4 LifelongAgentBench：最干净的跨 task procedural transfer

LifelongAgentBench 的 DB、OS 和 KG 环境把任务组织成严格序列，并为任务提供 SQL/Bash/KG skill
标注。其默认 experience replay 更接近“最近成功轨迹 FIFO”：保留最近若干 successful sessions，随后
把这些 session 文本注入新任务，并没有 similarity-aware 或 graph-aware retrieval。这给 GEM 留出了非常
干净的替换点：

```text
Target Task ANN
  → REQUIRES Skill
  → 历史 Task / Experience
  → success、cutoff、domain、environment compatibility filter
  → bounded top-k injection
```

DB 和 OS 任务能够用数据库最终状态或 shell 执行结果确定性验证，避免 LLM judge 噪声。官方 Session
对象也保留 task、sample、status、chat history、task output 和 evaluation record，适合直接规范化成
GEM Session/Attempt/Outcome。

必须控制 claim：DB task 通常使用新 MySQL container，完成后表会被删除；OS container 也按 task 重建。
所以这里测的是共享技能与程序化策略迁移，不是一个 world state 跨 session 持续演化。

#### 1.5 EvoMemBench：适合做应用层主 benchmark，但只接入两个 cross-episode track

[官方 EvoMemBench 仓库](https://github.com/DSAIL-Memory/EvoMemBench)把 benchmark 分成：

- In-Episode Knowledge；
- In-Episode Execution；
- Cross-Episode Knowledge；
- Cross-Episode Tool Use；
- Cross-Episode Web Search；
- Cross-Episode Embodied Execution。

cross-episode tracks 的共同协议是在早期 episode 后 update memory，并在相同 context、dataset subset 或
source→target pair 的后续 episode 中调用 retrieval/injection。execution tracks 同时统计 agent token、
memory extraction/retrieval token、latency、success 和 progress。

实现上并不存在一套统一 adapter：CrossEp-Know 使用 `retrieve/extract`，BFCL 使用 `utilize/update`，
ALFWorld 使用 `inject/update`，不同目录还有各自环境、依赖、API 和 evaluator。建议只选择一个
execution domain 接入：优先 BFCL tool-use，其确定性 checker 和 prefix progress metric 更适合作为
GEM workload；ALFWorld 可作为第二个 domain，而不是一次移植全部六套代码。

进一步核查公开实现后，CrossEp-Know 是很合适的第一阶段主负载：固定版本包含 120 个相互隔离的
context、884 个 episode，每个 context 有 5–12 个 episode；同 context 内保持文件顺序串行执行，不同
context 并行，回答当前 episode 后才 extract 新 memory。CrossEp-Tool 则先在四个 source environment
建立 memory bank，再以只读副本执行 12 个有向 source→target pair，适合测 procedural transfer 和
negative transfer。

它的局限也必须写进协议：CrossEp-Know 的 rubrics 是 output-quality judge 条件，不是历史 episode
relevance label；不能从最终 answer score 反推出某条图边正确。官方 GraphRAG 还是
NumPy + NetworkX：query 时堆叠全部 embedding 计算相似度，写入时与全部旧节点比较并 pickle 整张图。
这个实现可作为 agent-quality baseline，却不能作为 TriDB 系统性能 baseline，也不满足 TR-1。

#### 1.6 WorldMemArena：最适合补齐 GEM 的 memory lifecycle schema

WorldMemArena 把 agent memory 解释为 Action–World Interaction Loop，并明确划分四阶段：

1. Observe to Write：是否从 session trajectory 选择了未来有用的信息；
2. Update and Consolidate：是否正确 revise/merge/remove 已过期内容；
3. Retrieve for Decision：是否找回当前决策所需的 evidence；
4. Use and Act：是否在答案或动作中正确使用检索结果。

其数据为每个 session 提供 gold memory points、state updates、interference/distractor 和 checkpoint QA
evidence chain。指标也按阶段分解为 Memory Recall/Correctness/Hallucination/Irrelevance、Update
Handling、Interference Rejection、Retrieval Recall/NDCG/Coverage，以及最终 QA 的 Correct、
Hallucination、Omission、F1 和 BLEU-1。

[官方实现](https://github.com/UCSB-AI/WorldMemArena)已经定义了非常适合 GEM 的 adapter surface：

- `reset()`；
- `ingest_turn()`；
- `end_session()`；
- `snapshot_memories()`；
- `export_memory_delta()`；
- `retrieve()`。

runtime record 还保留 stable memory/session ID、delta op、linked previous memories、retrieval rank/score、
checkpoint、gold evidence 和分阶段 latency/token。GEM 可以实现这个 adapter，而不需要 fork 其 evaluator。
第一阶段可只跑 text/caption projection；图片作为 content-addressed Artifact URI/哈希保存，等后续再增加
视觉 embedding track。

需要冻结 dataset manifest：WorldMemArena 当前 arXiv 摘要、PDF正文和仓库说明之间存在 400/461
task 的版本漂移，正式实验必须同时记录 paper version、dataset snapshot 和 repo commit。

#### 1.7 SEA-Eval：借用纵向序列和指标，不依赖其代码

SEA-Eval 使用 32 个 atomic tasks 构造长度为 5 的序列，核心结构为：

```text
Correlated:   A1 → A1 → A1 → A2 → A3
Orth-Same:    A1 → B → C → D → A1
Orth-Similar: A1 → B → C → D → A2
```

A2/A3 保留相同执行逻辑但替换参数，用于区分策略泛化和死记答案；B/C/D 是无关干扰任务。它还设置
clean 与预加载无关 skill 的 noisy 条件。其最值得复用的指标是：

- Success Rate；
- Token Consumption；
- Execution Steps；
- Self-Correction Frequency；
- Efficiency Evolution Rate；
- SR Growth Slope；
- interference 后的 retention/generalization。

截至本次核查，[SEA-Eval 官方仓库](https://github.com/LeaperOvO/SEA-Eval)只有 IDE metadata，没有可用
dataset/runner/evaluator。因此 GEM 应自行在 MemoryArena/LifelongAgentBench task 上实例化这些 sequence
和 longitudinal metrics，不能把 SEA-Eval 列为“代码已接入”的 baseline。

### 2. 推荐的 GEM workload tracks

#### Track A：Causal Cross-Session Reuse

主数据：MemoryArena Progressive Search、Formal Reasoning，之后再加入 Travel。

每个 subtask 映射成 Session；每个需要历史信息的 decision point 都带 gold dependency/evidence chain。
主指标为 dependency coverage、Task Progress、SR@depth 和 downstream success。它回答的是：

> 如果后续 session 的成功确实需要早期 session 的信息，GEM 能否在严格 cutoff 下找回并使用它？

#### Track B：Cross-Task Skill Transfer

主数据：LifelongAgentBench DB/OS；补充 EvoMemBench BFCL。

用 GEM retrieval 替换 FIFO recent-success replay，分别测试 same-skill、related-skill、cross-domain 和
interference 条件。主指标为自动验证 Success、Progress、tokens/steps/time-to-target 与 transfer gain。

#### Track C：Memory Lifecycle Correctness

主数据：WorldMemArena text-first subset；补充 MemoryAgentBench Conflict Resolution/LongMemEval。

分阶段验证 write、update、retrieve、use，不将最终 QA accuracy 反推成所有阶段都正确。这个 track 还应
承载 as-of snapshot、supersession、interference rejection、policy/cutoff 和 stale-memory 查询。

#### Track D：Systems Scale and Query Execution

主数据：现有 EvoTrace 真实谱系 + 从 A/B/C 提取的 session/task/experience 分布；额外生成结构保持的
hard negatives。

这个 track 专门测 TriDB/GEM 的查询正确性、latency、early termination、mixed read/write throughput 和
规模拐点，不把合成数据上的 retrieval quality 当成 agent outcome。

### 3. Experience Graph 建模建议

当前 `bench/agent_memory/gem_eg/schema.sql` 已有 `Task/Session/Node/Prompt/Artifact`、logical-step event、
LLM usage、native adjacency topology 和 edge/load receipt，应保留为 core lineage layer。为了覆盖上述
benchmark，需要增加 decision/reuse 和语义经验层：

```text
Task ──REQUIRES──────────────→ Skill
 │                              ↑
 └─HAS_SESSION→ Session         │ DEMONSTRATES
                   │            │
                   └─HAS_ATTEMPT→ Attempt ──DERIVED_INTO→ Experience
                                      │
                                      └─HAS_STEP→ Step
                                                   ├─OBSERVED→ Artifact
                                                   ├─ACTED_WITH→ Action
                                                   └─RECEIVED→ Feedback

New Fact/Constraint ──SUPERSEDES→ Old Fact/Constraint
Evidence ──SUPPORTS→ Decision
Decision ──HAS_RETRIEVAL→ ReuseEvent ──SELECTED→ Experience
Outcome ──EVALUATES→ Attempt
```

建模原则：

- `Attempt/Step/Action/Observation/Feedback` 是原始发生过的经验；
- `Experience/Skill/Fact/Constraint` 是从轨迹蒸馏出的可复用对象，不能与 raw trace 混为一谈；
- `HELPED/HARMED/INFLUENCED` 等效用边只能在 target outcome 发生后生成，不能在检索时泄漏；
- Skill/Fact/Constraint 可以继续落在 GEM 的知识型 `gem_unit`，通过 native graph 的
  `DERIVED_FROM/REQUIRES/DEMONSTRATES/SUPERSEDES` 与 Experience Graph 连接；
- 图 topology 只放 native adjacency-list AM；关系表只存属性、provenance 和 receipt，不通过 edge-table
  join 模拟 traversal；
- Task 和 distilled Experience 需要 embedding；Session、Step、Prompt 通常保持 NULL embedding；
- 所有可以作为查询结果返回的 vertex 必须有 embedding，避免 `tjs_open` 对 NULL vector 静默跳过。

需要补充的关系属性：

- `availability_seq`、`valid_from_seq`、`valid_to_seq`；
- `source_session_uid`、`target_session_uid`；
- scope/ACL/visibility group；
- domain、environment version、evaluator hash、tool schema version；
- success、fitness、error signature、failure class；
- superseded/invalid/retracted；
- token/step/time cost；
- artifact SHA 和 derivation receipt。

没有可靠 wall-clock 的 corpus 使用 deterministic session/step ordinal，必须标记为 synthetic availability
order，不能伪称真实 chronology。

### 4. GEM 的 canonical cross-session query

主查询应冻结为一个 bounded、可提前终止的三模态流程：

```text
1. 对 target Task 或当前 Decision 做 ANN，得到相似历史 Task/Experience seeds；
2. 沿 Task→Skill→Experience、Task→Session→Attempt、lineage/evidence 边做有界 native traversal；
3. 下推 relational predicates：
     availability < target cutoff
     source_session != target_session
     policy/scope allowed
     valid AND NOT superseded
     evaluator/environment/tool-schema compatible
     outcome >= threshold
4. 按 query relevance、historical utility、recency 和 injection cost 做 bounded top-k；
5. 流式返回，并生成 immutable retrieval/injection receipt。
```

必须遵守 Open/Next/Close 与 LIMIT early termination，不能先物化所有历史候选再全局排序。每次 receipt
至少记录 query/cutoff/snapshot、seed IDs、traversed edge types/hops、候选 ID、similarity、reward、每个
filter 的接受/拒绝原因、最终 rank、注入 token 和 downstream decision/outcome。

现有 EvoTrace W1 已经给出一个关键负面结果：W1.a 用 **task-description vector 直接排序 code artifact**
时，fused nDCG 输给 no-vector；similarity top-10 与 reward top-10 的 overlap 约为 0.05。说明跨模态语义
并未自动对齐。GEM query 应优先采用：

- Task description → historical Task ANN；
- 再沿 graph 找 Experience；或
- 为 Experience 生成与 task/query 空间对齐的 distilled summary embedding。

不要用一个 task-description embedding 直接给 raw code、tool trace 或 observation 排序，也不能只按
similarity 代替 reward/utility ranking。

### 5. 实验臂、ground truth 与指标

#### 5.1 必备实验臂

每个 target 使用相同 task、model、sampling seed、environment、prompt template 和 injection token budget：

1. `memory-off`；
2. `full-context` 或 `recent-success FIFO`；
3. `vector-only`；
4. `graph/metadata-only`；
5. `GEM fused vector+graph+relational`；
6. `oracle evidence/experience`。

附加消融包括：成功经验 vs 成功+失败经验、不同 top-k/budget、clean vs noisy、same-task cross-session vs
cross-task transfer、with/without supersession、不同 dependency depth。所有 arm 的最终注入 token 必须配平，
否则 outcome 差异可能只是上下文长度差异。

#### 5.2 Ground truth 来源

| Ground truth | 数据来源 | 用途 |
|---|---|---|
| session dependency/evidence chain | MemoryArena | causal recall、SR@depth |
| Task→Skill | LifelongAgentBench `skill_list` | cross-task relevance |
| memory point/update/interference/evidence | WorldMemArena | write/update/retrieve correctness |
| supersession/serial fact order | MemoryAgentBench | stale rejection、as-of correctness |
| reward/lineage/dead-end/sibling | EvoTrace | utility grading、harmful@k、repair/path recall |
| paired target outcome | 新跑 memory-off/on target sessions | reuse 的最终因果证据 |

retrieval relevance 和 causal utility 必须分开：检索到 gold evidence 只能证明 alignment；只有 matched
target run 的 outcome 改善才能证明 reuse effectiveness。后者不能由“图已经加载”或 offline replay 替代。

#### 5.3 三层指标

**Retrieval quality**：

- Evidence Recall@k、NDCG@k、MRR；
- dependency-chain coverage；
- utility-weighted nDCG；
- `harmful@k`、`stale@k`；
- full-constraint-valid fraction；
- future/session/policy leakage，目标必须为 0。

**Database/system**：

- time-to-first-row、time-to-k；
- p50/p95/p99 retrieval latency；
- visited nodes/edges、ANN candidates、每一腿候选收缩率；
- LIMIT early-termination ratio；
- session commit/update throughput、WAL bytes、index/storage size；
- snapshot/as-of query latency；
- concurrent read/write 下的 throughput 和 tail latency。

**Agent outcome**：

- Success Rate、Task Progress；
- tokens/steps/wall-clock-to-target；
- fixed-threshold reward AUC；
- transfer gain 与 negative transfer；
- interference 后 retention/generalization；
- SEA-Eval 风格的 token/step convergence。

数据库更快不能自动推出 agent 更好。系统指标和 agent outcome 应分表报告；若要连接二者，应增加
fixed-SLA track，验证更低 retrieval latency 是否在同一 wall-clock budget 下换来更多有效 memory calls
或更高 target success。

### 6. 规模扩展：不能复制原任务

原建议中“把 Evo-Memory 任务复制到万级”的做法会制造 exact payload duplication、artifact leakage、
cache-friendly 重复和过于容易的 ANN positives。应将规模实验拆成：

- **原始规模**：只用于 agent reuse effectiveness；
- **扩展规模**：只用于 database scalability 和 retrieval robustness。

扩展到 `10² → 10³ → 10⁴ → 10⁵ → 10⁶` Experience/Step 节点时，增加结构保持的 hard negatives：

- 同 domain、语义相似但 skill 不同的 task；
- embedding 相似但 dependency chain 不满足的 experience；
- 已过期或被 supersede 的 facts/constraints；
- 高相似但 failed/dead-end 的 trajectories；
- 同 topology 但使用新 ID、新文本、新 artifact 的 session；
- 不同 evaluator/environment/tool schema 下不可复用的高 reward nodes。

每个 tier 保持原始 session length、branching factor、dependency depth、degree、success/failure ratio、
predicate selectivity 和 query mix。source/target artifact SHA、prompt/response 和 exact text 必须去重；每个
tier 都要保存 exact eligible set 和 top-k oracle。这样测到的是 Vector Haze、图约束和关系谓词的真实
贡献，而不是重复数据或缓存效应。

### 7. 推荐实施顺序

1. **LifelongAgentBench DB**：替换 FIFO callback，先得到无闭源 judge 的跨 task paired outcome；
2. **WorldMemArena adapter**：补齐 session delta、snapshot、supersession、interference 和 evidence schema；
3. **MemoryArena Progressive Search/Formal Reasoning**：建立真正的跨 session dependency track；
4. **EvoTrace scaled systems track**：继续使用现有 reward/lineage ground truth，加入 structure-preserving
   hard negatives 和 mixed read/write stream；
5. **EvoMemBench BFCL**：增加第二个确定性 execution domain；
6. 最后再考虑 WorldMemArena multimodal embedding 和 EvoMemBench ALFWorld/Web tracks。

第一阶段最小可发表组合应是：

```text
LifelongAgentBench DB     → cross-task outcome
MemoryArena subset        → causal cross-session outcome
WorldMemArena text subset → lifecycle correctness
EvoTrace scaled           → graph/reward/latency/early-termination
```

它同时避免了三个常见 claim 错误：把静态 retrieval 当经验复用、把 procedural transfer 当 persistent
world state、把数据库 latency 当 agent outcome。

### 8. 复现与 claim 边界

- 冻结 paper version、official repository commit、dataset checksum、model/judge version、prompt 和
  evaluator contract；
- 本地替换 judge 时单独报告，不与 paper 官方分数混称可复现；
- MemoryArena repo 只能称 preview implementation；
- Evo-Memory 当前只能按论文协议复刻，不能称使用官方代码；
- SEA-Eval 当前只能借 sequence/metrics；
- WorldMemArena 必须处理 400/461 task 的版本漂移；
- x86_64 stock-PG 可以完成 schema、adapter、correctness 和 stock-PG extension 测试，但不能声称 GX10
  ARM64 fork build 或 128 GB live benchmark 已 sign-off；
- 所有 GEM 查询必须留在同一个 PostgreSQL transaction/WAL 中，graph topology 使用 native adjacency
  AM，不退化成 relational edge joins；
- 所有 operators 必须满足 Open/Next/Close 和 early termination，任何需要完整物化中间结果的实现都不
  能作为 GEM canonical plan。

### 9. 2026-08-24 选型更新：EvoMemBench 是否适合下一步

**结论：适合，并且应升级为端到端应用层的主 benchmark；但不能单独承担 retrieval oracle 和数据库
scale benchmark。** 推荐组合改为：

```text
EvoMemBench CrossEp-Know  → 第一阶段主负载：跨 episode 知识/规则/过程复用
EvoMemBench CrossEp-Tool  → 第二阶段主负载：工具经验与 source→target transfer
MemoryArena Progressive   → prefix dependency 与 cutoff/leakage 诊断
EvoTrace scaled           → 数据库规模、谱系、失败经验和系统 tail latency
```

这比只跑 MemoryArena 更能回答“multi-modal database 作为 agent memory 有什么好处”：EvoMemBench 直接
测后续 episode 的 answer/success/progress；MemoryArena 则更适合隔离跨 session dependency。二者不能把
分数混在一起。

#### 9.1 已验证的数据与实现事实

- 官方仓库固定到 commit `aa4cea8fd936b76b2d3591d3ef897030617dc43a`；
- CrossEp-Know 文件 SHA-256 为
  `f4652ddcf954dd33653f91c9b40ed6138617b92e0f80d470cd1c92bb50890ce9`；
- 实际为 120 contexts / 884 episodes / 764 个非首 episode decision points；每个 context 5–12 episodes；
- 类别分布为 Procedural Task Execution 306、Domain Knowledge Reasoning 294、Rule System Application
  257、Empirical Discovery & Simulation 27；
- 同 context 内按文件顺序串行，memory 在当前 answer 生成后更新；不同 context 使用独立目录并行；
- CrossEp-Tool 支持 source bank 固定、target 只读的跨环境 transfer；
- 官方只提供 answer accuracy、success/progress、steps、token 和粗粒度 retrieve/extract latency；没有
  candidate receipt、future-leakage oracle、retrieval relevance、visited/pruned 或 WAL 指标；
- 固定 commit 没有顶层 LICENSE。内部研究可先做本地适配，但在确认条款前不要把上游 dataset/code
  复制进 GEM 发布物。

#### 9.2 GEM 的六臂公平实验

所有臂使用完全相同的 backbone、prompt、episode order、context isolation、top-k、injection token budget
和 snapshot cutoff：

1. `memory_off`：不检索、不注入、不写入可复用 memory；
2. `recent_fifo`：只取最近合法 episode；
3. `vector_only`：只按 embedding 相似度排序，但仍强制 scope 与 cutoff；
4. `graph_relational`：不用 vector，按 typed graph reachability + validity/scope/filter；
5. `gem_fused`：vector seed → native graph traversal → relational hard filter → bounded top-k；
6. `oracle`：只在有发布或独立标注 relevance 的 track 上运行。EvoMemBench 当前为 N/A，不能用
   “所有历史 episode”冒充 oracle；MemoryArena 可运行 protocol-prefix oracle。

cutoff、target exclusion、context/scope 和 target-readonly 是 governance rule，任何臂都不能消融。否则
测到的是未来信息或跨 context 污染，不是单模态 baseline。

#### 9.3 应报告的核心指标

| 层次 | 主指标 | 解释 |
|---|---|---|
| Agent outcome | Accuracy / Rubric Pass、Success、Progress、按 episode ordinal 的 cumulative AULC | memory 是否让后续 episode 真正变好 |
| Transfer | `memory_on - memory_off` paired gain、positive/negative transfer rate、source→target matrix | 哪类经验能迁移，何时反而伤害 |
| Retrieval | Dependency/Evidence Recall@10、NDCG@10、harmful@10、stale@10、constraint-valid fraction | relevance 未发布时 Recall/NDCG 必须报 N/A；不得报 0 |
| Leakage | future-session、target-session、cross-context、scope/policy violations | 必须全为 0；任一非零使该 run 无效 |
| Latency | TTFR、time-to-k、p50/p95/p99、open-loop achieved QPS | 区分首行流式收益与完整 top-k 延迟 |
| Search work | ANN candidates examined、visited nodes/edges、各 stage pruned、early-termination ratio | 验证 fused query 是否真的减少工作量 |
| Write path | episode commit/update throughput、extract latency、WAL bytes/episode、write amplification | agent memory 是读写系统，不能只测 query |
| Growth | index/storage bytes、latency/recall 随 episode/context/noise tier 的曲线 | 验证长期使用是否退化 |
| Cost | agent inference tokens、memory extraction tokens、retrieval embedding tokens、injection tokens | 所有 memory 模块成本都计入，不能只算 agent |
| Efficiency | successful episodes/s、success per 1M tokens、success under fixed wall-clock SLA | 把系统性能与 agent outcome 连起来 |

主统计单位应是 **context-level paired difference**，而不是把 884 个相互相关 episode 当独立样本。
报告 context bootstrap 95% CI；同时按类别、难度、episode ordinal 和历史长度分层。LLM judge 至少做固定
版本、盲化 arm label、固定 temperature，并对抽样集做双 judge/人工一致性校验。

#### 9.4 EvoMemBench 的 Experience Graph 映射

确定性、可审计的初始图只包含：

```text
Episode ──BELONGS_TO──→ Context
Episode(i) ──PRECEDES──→ Episode(i+1)
Episode ──EVALUATED_BY──→ Rubric
```

每个 episode 完成后，GEM extractor 才可创建：

```text
Episode ──DERIVED_INTO──→ Experience/Skill/Fact/Constraint
Evidence ──SUPPORTS─────→ Decision
New Fact ──CONTRADICTS/SUPERSEDES──→ Old Fact
Experience ──APPLIES_TO──→ Task/Tool/Environment
Retrieval ──SELECTED────→ Experience
Outcome ──EVALUATES─────→ Attempt
```

三种存储职责保持正交：vector 负责语义候选；native adjacency graph 负责 provenance、dependency、skill
和 supersession traversal；relational 负责 `context_id`、`ordinal < cutoff`、validity interval、ACL、
environment/tool-schema version 和 success/failure filter。`SUPPORTS/HELPED/HARMED` 是运行期推断或
target outcome 后标签，不得回填成上游 benchmark ground truth。

代码侧已经增加独立合同层：

- `bench/agent_memory/evomembench/dataset.py`：固定版本、context order、cutoff 和 relevance-N/A；
- `bench/agent_memory/evomembench/experience_graph.py`：只生成可审计结构边；
- `bench/agent_memory/memoryarena/`：六臂 reference、leakage gate、指标和内容寻址 receipt；
- `tools/fetch_memoryarena.py`：固定 Hugging Face revision 与 checksum manifest。

下一实现步是把 `CrossEp-Know` 的 `retrieve/extract` adapter 接到现有 `tjs_open` 流式路径，先做 12 个
context 的 local-model pilot；通过 parity、leakage 和 receipt 校验后再扩到 120 contexts。随后接
CrossEp-Tool，而不是一次移植 Web/ALFWorld 全套运行时。

### 10. 当前实现与 stock-PG pilot checkpoint

本轮已经完成以下可复现合同和 smoke evidence：

- MemoryArena 固定 release：Progressive Search 221 tasks / 1,641 sessions / 1,420 decisions；Formal Math
  40 / 354 / 314；Formal Physics 20 / 86 / 66。Progressive 的公开 task 数与论文所述 256 不一致，
  manifest 明确保留此 drift；
- exact reference 全集 gate：1,800 decisions × 6 arms = 10,800 个内容寻址 retrieval/injection receipts，
  future/target/cross-task/scope leakage 为 0；该 gate 不计时、不冒充 engine result；
- stock PostgreSQL 16.14 x86_64 上 `vector 0.8.0`、`graph_store_am 0.2.0`、`tjs_pg 0.2.0` live tests：
  GEM/C1–C6 与 MemoryArena adapter 共 33 passed；这不代表 GX10 ARM64 fork sign-off；
- `tjs_open` 使用 target-list + named server cursor + `itersize=1`，取到 top-k 即 close；receipt 记录真实
  TTFR、time-to-k、candidates/graph examined 和 termination reason；当前扩展没有独立 visited-edge
  counter，因此 `visited_edges=N/A`，不伪造为 0；
- Progressive 12-task hash-vector pilot 生成 456 receipts；Formal Math 40-task pilot 生成 1,884 receipts；
  两者 receipt digest 校验和 leakage gate 都通过。

这些 pilot 只使用 8 维 deterministic hash embedding、whitespace word budget 和 released gold response
回放，没有真实 agent generation。因此它们只证明 live query/update/cutoff/receipt wiring，**没有 agent
outcome 或 embedding quality claim**。

| Pilot / arm | decisions | TTFR p50 ms | time-to-k p95 ms | protocol Dependency Recall@10 mean |
|---|---:|---:|---:|---:|
| Progressive-12 recent/FIFO | 76 | 0.304 | 1.076 | 0.999 |
| Progressive-12 vector-only | 76 | 0.387 | 1.300 | 0.999 |
| Progressive-12 graph+relational, 1 hop | 76 | 0.511 | 1.094 | 0.227 |
| Progressive-12 GEM fused | 76 | 1.162 | 2.155 | 0.999 |
| Formal-Math-40 recent/FIFO | 314 | 0.313 | 1.161 | 0.989 |
| Formal-Math-40 vector-only | 314 | 0.400 | 1.156 | 0.989 |
| Formal-Math-40 graph+relational, 1 hop | 314 | 0.527 | 1.222 | 0.206 |
| Formal-Math-40 GEM fused | 314 | 2.174 | 3.381 | 0.989 |

表中的 Recall 是 all-prior protocol dependency，不是最小 evidence recall；当历史超过 top-10 时，oracle
本身也小于 1。graph-only 的低值是当前 `hops=1` 只覆盖相邻 session 的预期结果。更重要的是，小表上
fused 明显承担固定 operator 开销，并没有 latency 优势。这恰好说明报告不能预设“三模态一定更快”：
正式比较必须使用真实 embedding、足够大的 hard-negative history、相同质量点、hops/work-budget sweep
和 open-loop load，才能判断候选收缩是否抵消融合开销。

对应产物：

- `bench/agent_memory/memoryarena/manifests/protocol_gate_pinned.json`；
- `results/memoryarena/progressive_hash12_stockpg_v0.3/`；
- `results/memoryarena/formal_math_hash40_stockpg_v0.1/`。

### 11. 2026-08-24 真实 embedding 全量结果

在同一台 x86_64 stock-PG 主机上，用本地固定的
`Qwen/Qwen3-Embedding-0.6B`（1024 维）完成了两个 released split 的全量数据库路径回放：

- Progressive Search：221 tasks / 1,641 sessions / 1,420 decisions/arm，六臂共 8,520 receipts 和
  9,846 updates；
- Formal Reasoning–Math：40 tasks / 354 sessions / 314 decisions/arm，六臂共 1,884 receipts 和
  2,124 updates；
- 独立逐条复核结果：10,404 个 receipt digest 全部有效，future/target cutoff、cross-task scope、
  top-k/token budget 和 duplicate-selection 违规均为 0；
- embedding endpoint 的 advertised model 必须与 manifest 完全一致。Progressive 记录 9,846 次
  construction embedding、3,283 次 query embedding；Formal Math 分别为 2,124 和 709，失败调用为 0。

Progressive Search 的主结果如下（毫秒；TTFR 对 memory-off 为 N/A）：

| arm | decisions | TTFR p50 | time-to-k p50 | time-to-k p95 | protocol Dependency Recall@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| memory-off | 1,420 | N/A | 0.003 | 0.004 | 0.0000 | 0.0000 |
| recent/FIFO | 1,420 | 0.434 | 0.994 | 1.459 | 0.9978 | 0.9998 |
| vector-only | 1,420 | 13.097 | 13.741 | 16.929 | 0.9978 | 0.9998 |
| graph+relational, 1 hop | 1,420 | 0.690 | 1.058 | 1.634 | 0.2291 | 0.3404 |
| GEM fused | 1,420 | 24.575 | 25.687 | 28.031 | 0.9978 | 0.9998 |
| protocol-prefix oracle | 1,420 | 0.416 | 0.954 | 1.422 | 0.9979 | 0.9999 |

Formal Reasoning–Math 的结果具有相同方向：vector-only time-to-k p50/p95 为 12.668/15.364 ms，
GEM fused 为 29.190/31.420 ms；graph+relational 为 1.064/1.558 ms，但 protocol Dependency Recall@10
仅 0.2063。各有效 memory arm 的 update p50 为 15.8–17.4 ms；Progressive 为 28.1–29.3 ms。

这两个全量结果给出三个重要的**负向或限制性发现**：

1. 当前 released task 每个 scope 的历史很短，top-10 几乎容纳全部 prefix；因此 recent、vector、fused
   与 prefix oracle 的 protocol recall 都接近 1，无法区分 embedding quality，也不能证明图带来质量收益；
2. 1-hop 图只覆盖相邻 session，低 recall 是当前简单 `PRECEDES/FOLLOWS` 图的预期行为，不应解释为
   native graph store 本身质量差；下一步必须建立 Evidence/Skill/Supersedes 等有语义的 experience edges；
3. 当前规模上 GEM fused 比 vector-only 慢约 1.9–2.3 倍，固定融合开销尚未被候选收缩抵消。因此不能从
   本实验宣称 tri-modal DB 有 latency 优势；应在 structure-preserving hard-negative scale track 上做
   matched-quality 的 fused-vs-staged sweep。

此外，本轮仍是 released gold response 的 protocol replay：`agent_outcome=null`，没有生成式 agent 的
Success/Progress、tokens、steps 或 time-to-target。Recall 的 relevance basis 是论文协议要求的 all-prior
prefix，不是最小人工 evidence；公开数据也没有 stale/harmful 标签，所以这些指标在当前 run 应为 N/A，
不能从默认布尔字段汇总成 0。receipt 能证明选择、注入、cutoff 和内容哈希，但当前 live adapter 的
`semantic_score` 字段仍是确定性的候选序号占位，不是实际 pgvector distance；正式 quality run 前必须
将真实距离和 score provenance 写入新 schema 版本，不能用本轮 receipt 做 score calibration。

可复现产物：

- `results/memoryarena/progressive_qwen3emb221_stockpg_v0.1/summary.json`；
- `results/memoryarena/progressive_qwen3emb221_stockpg_v0.1/receipts.jsonl`；
- `results/memoryarena/formal_math_qwen3emb40_stockpg_v0.1/summary.json`；
- `results/memoryarena/formal_math_qwen3emb40_stockpg_v0.1/receipts.jsonl`。

这些结果把下一实验的优先级进一步收敛为：先完成 Progressive 的固定本地 answer-model paired outcome
track，再构造长历史 hard negatives 和语义 experience graph，最后做 GEM fused 与同质量分阶段执行的
系统比较。此时直接跑更多相同短 prefix task 不会回答“multi-modal database 是否更好”。

### 12. 2026-08-24/25 最终主实验：Progressive 全集 agent outcome、Formal 补充与 matched-quality 对照

本节是当前**权威结果**，取代第 10–11 节的历史 pilot/checkpoint 数字。旧目录仍保留用于说明协议和失败
演进，不能与本节的新 schema/run 混合汇总。特别是，第 11 节所说的 semantic-score 占位问题已在
`memoryarena_receipt_v0.3.0` 修复：本节 Progressive run 有 11,394 个实际
`pgvector_cosine_similarity_selected_top_k` provenance rows，独立 audit 未发现 score provenance 违规。

#### 12.1 固定协议与 claim boundary

- 主数据：MemoryArena Progressive Search release 全集，221 tasks、1,641 sessions、每臂 1,420 个
  cross-session decisions；Formal Reasoning–Math 全集 40/354/314，作为更长依赖链的 graph-heavy 补充；
- 六臂：`memory_off`、`recent_fifo`、`vector_only`、`graph_relational`、`gem_fused`、
  `protocol-prefix oracle`；每个 task/arm 独立 scope；
- Progressive answer model：本地 `qwen3.8`，revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`，temperature 0，seed 20260824；embedding 为
  `Qwen/Qwen3-Embedding-0.6B` revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`；每个 decision 的 injection budget 为 2,048 tokens；
- Progressive 的 session 0 使用 released gold answer 作为固定 source experience，后续每个 decision
  只调用一次本地模型做 direct-answer exact-match；因此这是“固定 source experience 能否跨 session
  帮助直接回答”的 paired evidence，**不是**上游 web-search agent、官方 judge 或完整 MemoryArena
  Task Progress evaluator 的复现；
- Formal 使用相同真实 embedding 做 released gold-response database-path replay，不生成 agent outcome；
- 执行环境为 PostgreSQL 16.14 x86_64 stock-PG，`vector 0.8.0`、`graph_store_am 0.2.0`、
  `tjs_pg 0.2.0`；不代表 GX10 ARM64 fork/128 GB benchmark sign-off。

实际查询路径为：`recent/oracle` 走 scope+state+cutoff 的 relational prefix access；`vector_only` 走带同一
关系谓词的 HNSW cosine top-k；`graph_relational` 先用关系属性确定 anchor，再执行 native adjacency
traversal，最后做 vertex-property filter；`gem_fused` 在一个 `tjs_open` 中执行 vector candidate stream →
native graph expansion → relational hard filter → bounded top-k。图 topology 从未物化为 relational edge
join table。

#### 12.2 完整性、parity、leakage 与 immutable receipt gate

Progressive `v0.5` runner 和独立 auditor 均通过：

| gate | 观测 | 违规 |
|---|---:|---:|
| outcomes / updates completeness | 9,846 / 9,846 | 0 |
| receipts completeness | 8,520 / 8,520 | 0 |
| answer-generation calls | 8,520；失败 0；missing prediction 0 | 0 |
| exact mandatory-eligibility parity | 8,520 queries | 0 |
| exact oracle-selection parity | 1,420 queries | 0 |
| future/target/cross-task/scope leakage | 全集 | 0 |
| receipt digest / injection content hash | 全集 | 0 |
| top-k / injection-token budget | 全集 | 0 |
| memory-off writes | 0 units | 0 |

数据库 post-state 也通过 scope↔receipt linkage：1,105 个非-off scopes、8,205 个 receipt-linked units，
memory-off units=0。Formal `v0.2` 的 1,884 receipts、2,124 updates 全部存在；1,884/1,884 exact
eligibility parity、314/314 oracle parity、leakage=0、memory-off write=0。这里的 eligibility oracle 是独立
从 dataset+cutoff 重算，不是仅检查 runner 自报的布尔值。

#### 12.3 Progressive retrieval 与 system metrics

单位为 ms；`R@10` 是 paper protocol 要求的 all-prior dependencies，不是人工标注的最小 evidence set。

| arm | R@10 | NDCG@10 | TTFR p50/p95 | time-to-10 p50/p95 | constraint valid | update p50/p95 | update/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| memory-off | 0.0000 | 0.0000 | N/A | 0.030 / 0.048 | 1.000 | N/A | N/A |
| recent/FIFO | 0.9981 | 1.0000 | 0.484 / 0.754 | 1.089 / 1.581 | 1.000 | 29.765 / 34.814 | 33.98 |
| vector-only | 0.9981 | 1.0000 | 13.874 / 15.939 | 15.352 / 17.584 | 1.000 | 29.132 / 33.548 | 35.80 |
| graph+relational | 0.2291 | 0.3404 | 0.773 / 1.077 | 1.194 / 1.719 | 1.000 | 29.341 / 34.385 | 35.36 |
| GEM fused | 0.9981 | 1.0000 | 52.188 / 63.007 | 54.389 / 65.498 | 1.000 | 29.192 / 33.162 | 35.81 |
| prefix oracle | 0.9981 | 1.0000 | 0.471 / 0.693 | 1.056 / 1.588 | 1.000 | 29.082 / 33.188 | 35.97 |

`gem_fused` 平均 examined 20,675.99 candidates，平均/p95 graph edge steps 为 4.275/7，按
`1-returned/examined` 计算的 candidate contraction 约 99.98%。这个高 contraction **不是效率优势**：它
来自“全局 HNSW stream 扫描大量其他 scope candidates，再由极窄 scope/cutoff predicate 丢弃”的
scan-and-prune 行为。冻结 `v0.5` receipts 所用的 installed `tjs_pg 0.2.0` 只暴露
`tjs_open_graph_examined()` edge-step counter，当时不暴露 distinct visited-node counter；因此主 receipt
中的 `visited_nodes=N/A`，不能把 edge steps 重命名为 nodes。随后使用 workspace-built、未替换系统安装
库的只读补充 probe，在当前 post-Formal snapshot 上重放全部查询：Progressive fused distinct graph-reach
nodes mean/p50/p95 为 4.725/5/8，edge steps 为 5.185/7/8；graph-relational 为 2.844/3/3 nodes、
1.844/2/2 edges，均无 graph censoring。这里 node 的定义是“graph leg reach hash 中的 distinct vertices，
包含 seeds”。后续 Formal 写入改变了 shared global HNSW 的 relaxed-order stream，因此这组补充 work 数
不能倒填或替换 v0.5 当时 receipt 的 4.275 edge-step mean；两个 artifact 分开保留。公开 release
没有 harmful/stale 标注，所以所有臂的 `harmful@10`、`stale@10` 也是 N/A，而不是 0。WAL bytes 是共享
Postgres WAL interval，只作为同机运行的描述性写放大信号，不当作严格 arm-local 因果量。

#### 12.4 Progressive paired agent outcome

| arm | decision exact match | task-mean cross-session accuracy | final-session success | Task Progress AULC | all-sessions success |
|---|---:|---:|---:|---:|---:|
| memory-off | 0.0014 | 0.0012 | 0.0045 | 0.0010 | 0.0000 |
| recent/FIFO | 0.3599 | 0.3831 | 0.8054 | 0.2754 | 0.1267 |
| vector-only | 0.3577 | 0.3806 | 0.8054 | 0.2740 | 0.1267 |
| graph+relational | 0.0915 | 0.1012 | 0.1629 | 0.0761 | 0.0000 |
| GEM fused | 0.3599 | 0.3828 | 0.8100 | 0.2747 | 0.1267 |
| prefix oracle | 0.3613 | 0.3839 | 0.8100 | 0.2771 | 0.1312 |

以 task 为 cluster 的 paired percentile bootstrap（2,000 iterations）显示，`gem_fused−memory_off`：

- task-mean cross-session accuracy `+0.3816`，95% CI `[+0.3425,+0.4225]`；
- final-session success `+0.8054`，95% CI `[+0.7511,+0.8552]`；
- Task Progress AULC `+0.2737`，95% CI `[+0.2294,+0.3199]`；
- prompt tokens/task `+5,357.5`，95% CI `[+5,022.5,+5,689.6]`；
- steps/task `+0.0000`，95% CI `[0,0]`，因为协议固定每 decision 一次 direct-answer call；
- inclusive time-to-target/task（retrieval + generation + post-answer update）`+454.55 ms`，95% CI
  `[+333.91,+582.10]`。

这证明在此 direct-answer track 中“注入历史经验”相对 memory-off 大幅提高结果，但**不证明三模态融合本身
带来增益**：recent、vector、fused、oracle 的 outcome 极接近，且它们看到的短 prefix 几乎都能完整装入
top-10；fused 比 recent/vector 多付出了系统延迟，却没有可分辨的 agent-quality lift。巨大 on/off 差异还
受“首 session 以 released gold answer 固定播种”这一 protocol 设计影响，不能外推为开放式 agent 自主积累
经验的收益。

#### 12.5 Formal Reasoning–Math graph-heavy 补充

| arm | R@10 | NDCG@10 | TTFR p50/p95 | time-to-10 p50/p95 | fused candidates / edge steps mean |
|---|---:|---:|---:|---:|---:|
| memory-off | 0.0000 | 0.0000 | N/A | 0.018 / 0.029 | N/A |
| recent/FIFO | 0.9894 | 1.0000 | 0.542 / 1.065 | 1.256 / 2.096 | N/A |
| vector-only | 0.9894 | 1.0000 | 19.237 / 26.817 | 21.088 / 28.849 | N/A |
| graph+relational | 0.2063 | 0.3210 | 0.834 / 1.753 | 1.329 / 2.543 | N/A |
| GEM fused | 0.9894 | 1.0000 | 99.306 / 108.228 | 102.565 / 111.539 | 25,665.56 / 4.904 |
| prefix oracle | 0.9894 | 1.0000 | 0.511 / 1.068 | 1.215 / 2.223 | N/A |

Formal 的更长依赖没有改变结论：简单 session-chain 的 1-hop graph arm 只能找邻近 session；它并没有
Evidence/Dependency typed edges，因而不能测试“图能否找到跨多步证明依赖”。这里的 graph-heavy 是图路径
调用更频繁/依赖序列更长，而不是已经具备高质量语义图 ground truth。

同一 post-Formal snapshot 的只读 node/work supplement 给出：Formal fused distinct reached nodes
mean/p50/p95 = 5.261/5/8、edge steps = 5.790/7/8；graph-relational 为 2.873/3/3 nodes、
1.873/2/2 edges，graph censoring 均为 0。probe 前后 source digest 与 global unit/vertex/visible-edge counts
完全一致。

#### 12.6 同快照 matched-quality：fused vs 分阶段执行

对 Progressive 完整 `gem_fused` scopes 做只读对照：每个 query 预计算一次 embedding（排除在两侧 DB
timing 之外），每 query 1 次 warm-up + 3 次 measured repetitions，执行顺序交错；共 1,420 queries、
4,260 paired measurements。staged 路径为 bounded vector window（36）→ 4 seeds 的 native
`gph_traverse_bounded` 1-hop、共享 65,536 edge-step budget → relational scope/state/cutoff filter + vector
final rank。中间集合上界为 vector window + graph budget，没有无界物化；这是一进程中的分阶段 baseline，
**不是**独立部署、含 RPC/serialization 的真实 polyglot 系统。

质量 gate 使用公开 release 实际支持的逐查询 Dependency Recall@10 与 NDCG@10，而不是要求 ID 顺序相同：
4,260/4,260 quality match，constraint violations=0；source scope SHA-256 digest 和全局 unit/vertex/visible-edge
counts 前后相同，独立 audit 再与 live DB 校验也通过。exact ID set 相同 4,224/4,260（99.15%），exact order
相同 1,740/4,260（40.85%）；这些只是诊断，因为 all-prior relevance 下，不同 prior ID/顺序仍可具有相同
官方可计算质量。

| matched-quality metric | GEM fused | staged |
|---|---:|---:|
| Dependency Recall@10 mean | 0.9981 | 0.9981 |
| NDCG@10 mean | 1.0000 | 1.0000 |
| TTFR p50 / p95 ms | 77.542 / 84.502 | 4.039 / 6.133 |
| time-to-10 p50 / p95 ms | 78.466 / 85.368 | 4.777 / 7.087 |
| candidates examined / bounded union mean | 26,549.92 | 4.95 |
| graph edge steps mean / p95 | 5.185 / 8 | 5.185 / 8 |

配对 task-cluster bootstrap 的 `staged−fused` mean time-to-10 为 `−71.923 ms`，95% CI
`[−72.452,−71.417]`。staged mean 分解为 vector 1.057 ms、graph 1.784 ms、final filter/rank 1.432 ms。
fused 的 4,260 次结束原因全部为 `stream_end_unknown`，平均检查约等于运行时全局 26,550 units；graph
没有 censoring。当前 workload 下，fused 的 shared global HNSW + 极窄 per-task filter 产生灾难性的
filtered-ANN drain，因此结果明确反对“融合算子天然更快”。要让 fused 有机会取胜，需要 scope-aware vector
physical design、显式 `tjs.vector_scan_budget`/quality sweep，或每 scope 足够大的 hard-negative history；
不能通过省略这次负结果来维护三模态论点。

#### 12.7 当前能回答与不能回答的问题

本实验支持：

1. GEM/TriDB 能在同一 Postgres transaction/WAL 边界内完成严格 cutoff、native graph traversal、关系过滤、
   vector retrieval、流式 top-k、更新和可审计 receipt；全量 leakage/parity gate 通过；
2. 在固定 gold source experience + 本地 direct-answer 模型的 Progressive track 中，cross-session memory
   injection 相对 memory-off 显著改善 outcome；
3. 当前 released short-history workload 上，recent/vector/fused/oracle 的质量与 outcome 无法区分；
4. 当前 `tjs_open` physical path 在极窄 scope filter 下显著慢于 bounded staged baseline，没有系统效率收益。

本实验不支持：

- “图”提高了 retrieval quality 或 agent outcome；当前图只是相邻 session chain；
- “三模态”优于 recent/FIFO 或 vector-only；没有观察到独有质量收益；
- multimodal content（image/audio/video）本身带来收益；这里的 multi-modal 指 vector+graph+relational data
  models，payload 仍是文本；
- harmful/stale rejection 或 semantic evidence-graph quality；公开标签缺失；冻结主 receipt 的 node 字段
  仍为 N/A，但独立只读 node/work replay 已提供当前快照的数值；
- GX10 性能、真实 polyglot RPC 对照、并发读写、10^5–10^6 scale 或开放式 web agent 的外推。

下一轮最有信息量的工作不是再重复短 prefix，而是构造 structure-preserving hard-negative histories，并加入
typed `SUPPORTS/SUPERSEDES/APPLIES_TO` experience edges、harmful/stale labels 和 scope-aware vector
access。distinct-node probe 还应正式 version/install 到 extension，而不是长期依赖 workspace-built
supplement。然后在相同 Recall/NDCG operating point 下 sweep history size、predicate
selectivity、graph depth/work budget、vector scan budget 和并发度，才能真正检验 multi-modal database 作为
agent memory 的质量/效率边界。

#### 12.8 可复现产物

- Progressive agent 主 run：
  `results/memoryarena/progressive_agent_qwen38_goldseed221_stockpg_v0.5/summary.json`、
  `outcomes.jsonl`、`receipts.jsonl`、`updates.jsonl`、`provenance_preflight.json`、`audit.json`；
- Formal 补充：`results/memoryarena/formal_math_qwen3emb40_stockpg_v0.2/summary.json`、
  `receipts.jsonl`、`updates.jsonl`、`audit.json`；
- matched-quality 对照：
  `results/memoryarena/progressive_fused_vs_staged_stockpg_v0.2/measurements.jsonl`、`summary.json`、
  `audit.json`；
- graph work/node supplements：`results/memoryarena/progressive_graph_work_probe_stockpg_v0.1/` 与
  `results/memoryarena/formal_math_graph_work_probe_stockpg_v0.1/`；每条 query 的 nodes/edges/candidates 和
  cutoff 验证在 `measurements.jsonl`，aggregate/read-only gate 在 `summary.json`；
- runner/auditor：`tools/run_memoryarena_agent_outcomes.py`、`tools/pilot_memoryarena_tridb.py`、
  `tools/audit_memoryarena_agent_run.py`、`tools/audit_memoryarena_replay_run.py`、
  `tools/benchmark_memoryarena_fused_vs_staged.py`、`tools/audit_memoryarena_fused_vs_staged.py`；node
  supplement runner 为 `tools/probe_memoryarena_graph_work.py`；
- 两个真实服务中断产生的 incomplete run 保留在 Progressive `v0.3/failure.json` 和
  `v0.4/failure.json`，不纳入指标；2-task matched-quality gate 的失败/通过目录分别为
  `progressive_fused_vs_staged_gate2_stockpg_v0.1` 和 `v0.2`。

---

# 附录 C — OpenEvolve 作为 agent-outcome 载体：源码审计与指标定义（2026-08-24）

## C.0 Material Passport

- 审计对象：**本地已安装的 `openevolve 0.3.2`**，路径
  `/localhome/hza214/tridb/.venv-e0/lib/python3.11/site-packages/openevolve`，
  入口 `.venv-e0/bin/openevolve-run`。
- 上游：<https://github.com/codelion/openevolve>（作者 Asankhaya Sharma）。
- Verification Status：**源码已读，未执行**。本附录的所有行号引用来自本地 0.3.2 安装；
  评估器行为、cascade 分流比例、token 消耗均为**读码推断**，尚未在真实负载上测量。
- 上游论文状态：**OpenEvolve 自身没有论文**，它是 DeepMind
  [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)（2025）的开源实现。
  同类开源工作另有 [CodeEvolve, arXiv:2510.14150](https://arxiv.org/html/2510.14150v1)。
- AlphaEvolve 的指标契约：solution → **一组标量 metric，约定为「越大越好」**；
  支持同时优化多个 metric，且论文明确指出即使主目标只有一个，附加 metric 也常常提升表现
  （鼓励高分解的多样性）。OpenEvolve 沿用这个契约。

## C.1 结论先行

| 问题 | 判定 |
|---|---|
| 需要外挂 GEM 做 memory-on/off 对照吗？ | **需要，而且这是唯一能给出因果结论的做法**。OpenEvolve **完全没有跨 session 记忆**，每个 run 从零开始，这个缺口就是 GEM 的实验位 |
| 需要自己写一个 evolve agent 吗？ | **不需要**。0.3.2 已装好，注入点是一个函数返回值，改动面很小 |
| 注入点在哪？ | `process_parallel.py:865` 的 `database.sample_from_island(...) -> (parent, inspirations)`；`inspirations` 一路流到 `prompt/sampler.py:build_prompt(inspirations=...)` 的 `{inspirations_section}` |
| 生成代码能测什么？ | evaluator 返回的 metric + artifacts 侧信道，**在我们自己的 run 里是 100% 覆盖的**，不受 EvoTrace 历史 corpus 日志残缺的限制 |
| 能直接对 EvoTrace 论文的数字吗？ | **不能**。原 trace 用 deepseek-reasoner(76%)/gemini-3-flash/claude/gpt-5，一个 qwen3.8 都没有 |

## C.2 evaluator 契约 —— buggy rate 的数据来源

`evaluation_result.py`：

```python
@dataclass
class EvaluationResult:
    metrics: Dict[str, float]                      # 必需
    artifacts: Dict[str, Union[str, bytes]] = {}   # 可选侧信道
```

- `combined_score` 由系统从 metrics 自动汇总，是主评价信号。
- **metrics 必须是原始连续值,不能是预先分好的 bin 序号** —— MAP-Elites 的分箱由 database 内部
  min-max 完成（`evaluation_result.py` 文件头有明确警告）。
- `artifacts` 约定键：`stderr`、`profiling_data`、`llm_feedback`、`build_warnings`。
  **artifacts 会自动进入下一轮的 prompt**（`prompt.include_artifacts=True`,
  `max_artifact_bytes = 20*1024`），构成 OpenEvolve 的错误反馈闭环。

失败是怎么记的（`evaluator.py`）：

| 情形 | 记录 | 行 |
|---|---|---|
| 超时 | `{"error": 0.0, "timeout": True}` + artifact `{"timeout": True, "timeout_duration": …, "error_type": "timeout"}` | :174-183, :253-265 |
| 重试耗尽 | `{"error": 0.0}` | :292-296 |
| 评估阶段异常 | artifact `failure_stage: "evaluation"` | :182, :261, :279 |

**cascade 是默认开的**（`EvaluatorConfig.cascade_evaluation = True`,
`cascade_thresholds = [0.5, 0.75, 0.9]`，对应 evaluator 模块里的
`evaluate_stage1/2/3`）。这直接影响 buggy rate 的分母：**被 stage1 淘汰的程序永远拿不到完整分数**。
因此必须逐程序记录「死在第几个 stage」，否则 buggy rate 会把「早筛」和「有 bug」混为一谈。

> 注：若 evaluator 未定义 `evaluate_stage1`，`cascade_evaluation: true` 会**静默退化**成直接评估
> （只打一条 warning，`evaluator.py:108-119`）。做对照实验前必须确认两个 arm 的 cascade 行为一致。

## C.3 GEM 的注入点 —— 精确位置

调用链（本地 0.3.2）：

```
process_parallel.py:865   parent, inspirations = self.database.sample_from_island(
                              island_id=target_island,
                              num_inspirations=self.config.prompt.num_diverse_programs)
process_parallel.py:~879  executor.submit(_run_iteration_worker, iteration, db_snapshot,
                                          parent.id, [insp.id for insp in inspirations])
process_parallel.py:146   inspirations = [programs[pid] for pid in inspiration_ids if pid in programs]
                          # programs 来自 db_snapshot["programs"]
process_parallel.py:187   build_prompt(..., inspirations=[p.to_dict() for p in inspirations], ...)
prompt/sampler.py:458     _format_inspirations_section(deduped_inspirations, ...)
prompt/templates.py:104   {inspirations_section}
```

**关键事实：`db_snapshot["programs"]` 只含当前 run 自己的程序。** OpenEvolve
没有任何跨 session 检索 —— `embedding.py` 和 `novelty_judge.py`（均改编自 SakanaAI/ShinkaEvolve,
Apache-2.0）只用于**去重**（`DatabaseConfig.similarity_threshold = 0.99`），不是检索。
所以「跨 session 经验复用」在 stock OpenEvolve 里是一个**空缺**，不是一个可调参数。

推荐接法：**子类化 `ProgramDatabase`，覆写 `sample_from_island`**，把 GEM 检索到的外部
program 追加进返回的 inspirations，同时写进 `self.programs` 以便进入 snapshot。外部条目必须打
metadata 标记，保证：

1. 永远不被选为 `parent`（否则演化会直接从别的 session 的代码继续，那不是 memory，是 seeding）；
2. 不计入 island 成员、archive、fitness 统计、MAP-Elites 网格；
3. 在 evolution trace 里可区分，事后能审计每次注入的来源和 receipt。

`sampler.py:449-455` 已有去重：inspiration 若与 top/diverse 区块重复会被丢弃，因此外部条目与
自身 run 的高分程序撞车时不会被重复计入 token。

## C.4 token 配平 —— 最容易做错的一步

| 旋钮 | 默认 | 含义 |
|---|---:|---|
| `prompt.num_top_programs` | 3 | 「以往尝试」区块的条数 |
| `prompt.num_diverse_programs` | 2 | **inspiration 条数**，即注入率的直接旋钮 |
| `prompt.max_artifact_bytes` | 20 KB | 错误反馈注入上限 |
| `prompt.include_artifacts` | True | 是否把 stderr 等回灌 prompt |

**memory-off ≠ 无上下文。** stock OpenEvolve 每轮已经注入 3+2 个自身 run 的程序。
因此正确的对照不是「加不加」，而是**同样注入 N 条，只换来源**：

```
Arm A (memory-off)  : N 条全部来自本 run 的 database
Arm B (memory-on)   : 其中 M 条替换为 GEM 跨 session 检索结果，N 不变
Arm C (vector-only) : 同 B，但检索退化为纯 ANN，无图遍历
```

若 memory-on 是**追加**而非**替换**，两个 arm 的 prompt 长度不同，outcome 差异可能纯粹来自
上下文长度，与记忆质量无关。这条在正文 §「不做 token 配平结论即无效」里已经写过，
在 OpenEvolve 上的具体落点就是 `num_diverse_programs`。

注入率扫描建议：`num_diverse_programs ∈ {0, 2, 5, 10}`，其中 0 是真正的「无 inspiration」下界。

## C.5 可复现性旋钮

`random_seed: Optional[int] = 42` 同时存在于 `Config`、`DatabaseConfig`、`LLMModelConfig` 三层。
**演化搜索本身随机**，单次 run 的 fitness 差异说明不了任何事 —— 每个配置必须多 seed 重复并报 CI。

其他必须在两个 arm 之间冻结的量：`max_iterations`、`diff_based_evolution`、`temperature`、
`top_p`、`max_tokens`、`num_islands`、`migration_interval`、`population_size`、`archive_size`、
`cascade_thresholds`、evaluator 版本与其 sha256。

## C.6 evolution_trace —— 新 run 可以直接进 Experience Graph

`EvolutionTraceConfig`（`config.py:403-412`）：

```python
enabled: bool = False        # 默认关，必须显式打开
format: str = "jsonl"        # jsonl | json | hdf5
include_code: bool = False   # 建议开
include_prompts: bool = True
output_path: Optional[str] = None
buffer_size: int = 10
compress: bool = False
```

`evolution_trace.py` 另提供 `extract_evolution_trace_from_checkpoint()` 和
`extract_full_lineage_traces()`，后者逐条产出 `{template, system_prompt, user_prompt, llm_response}`。

**这意味着我们自己跑出来的 run 与 EvoTrace 是同一套 schema**，可以直接经
`tools/evotrace/normalize.py` 进入 `gem_eg_*`。好处有二：

1. 新 run 可以和历史 121 个 run 放进同一张图，做「历史经验 → 新 run」的真实跨 session 复用；
2. 第一类（检索）指标和第二类（agent）指标建立在同一份数据上，不需要两套 loader。

必须打开 `include_code: true`，否则 artifact 层为空，无法做泄漏审计（source/target
artifact SHA 去重）。

## C.7 生成代码的指标定义

### C.7.1 A 组 —— 失败率（客观、确定性、我们自己的 run 里 100% 覆盖）

分母 = 本 run 生成的候选总数。分子按 evaluator 的确定性裁决分类：

| 类别 | 判据 |
|---|---|
| `timeout` | `metrics["timeout"] is True` |
| `execution_error` | 返回 `{"error": 0.0}` 且无 timeout 标记 |
| `compile_error` | 仅 C++/ALE 域；由 evaluator 的 build 阶段给出，落在 `artifacts["build_warnings"]`/`stderr` |
| `wrong_answer` | evaluator 判定答案错但程序正常退出 |
| `cascade_stage1/2/3_reject` | **单列**，记录死在第几阶段 |
| `non_improving` | 程序正常且有分，但未超过 parent —— **这不是 bug，必须与上面分开** |

> 这一组正是**历史 EvoTrace corpus 算不出来的东西**：`is_valid=false` 仅 gepa_native 有
> （其余三 backend 全为 0，是不记录而非无 bug）；`error_signature` 覆盖率
> gepa 80.25% / evox 15.23% / openevolve 9.73% / shinkaevolve 0.00%。
> 跨 backend 报历史 buggy rate，报的是日志详略，不是代码质量。**只在自己的 run 上报。**

### C.7.2 B 组 —— 得分

- **best `combined_score`（per-task）**。AlphaEvolve/OpenEvolve 约定越大越好；EvoTrace 的
  math 任务已归一化到基准（如 heilbronn_triangle 除以 0.036529889880030156），故 >1 有意义。
- **time-to-threshold**：达到某 `combined_score` 所需 iteration 数。**比终点值更能反映记忆的作用** ——
  记忆的价值主要体现在「更快到达」，而非「最终更高」。
- **best-so-far 曲线的 AUC**：整段搜索效率的单一数字。

**跨 task 不得直接平均 fitness**（不同 evaluator 契约不可通约）。聚合只能用 per-task
normalized improvement 或 rank。这与 `gem_eg/groundtruth.py` 拒绝跨 evaluator 比较 reward 是同一条规则。

### C.7.3 C 组 —— 成本

iteration 数、LLM 调用次数、prompt/completion token 总量。论文级 claim（如「达标速度约 10×、
token 成本降 52%」）落在这一组。注意 completion token 受 thinking 开关影响极大
（EvoTrace 实测 reasoning tokens p50 1,941 / p99 16,384），必须声明配置。

### C.7.4 D 组 —— 搜索健康度

- **edit acceptance rate**：被接受进 database 的比例；
- **dead-branch rate**：`gem_eg/groundtruth.py:is_dead_end()` 已实现，四 backend 均有定义，
  不依赖各家日志详略；
- **duplicate artifact rate**：OpenEvolve 的 `similarity_threshold = 0.99` 直接给出判据。
  历史语料里 10,672 个 node 只有 9,733 个不同 artifact，重复生成是可量化的浪费。

### C.7.5 明确不报的

- **lint 分 / 圈复杂度 / 代码整洁度**。在这类任务里方向可能与目标相反 —— 竞赛级启发式常常又长又丑
  却得分最高。报这类 proxy 只会削弱结论。
- **跨 task 平均 fitness**（见 C.7.2）。
- **我们的 fitness 与 EvoTrace/AlphaEvolve 论文数字的对比**（模型不同，是不同的搜索过程）。

## C.8 实验设计与前置条件

**Split**（沿用正文 §4.4）：
- `same-task cross-session`：同一 task 的若干 source session 建 memory，held-out session 只做 target；
- `cross-task transfer`：按 task family 留出整个 task，source 中不得出现其 exact code blob。

**前置闸门**：
1. source/target artifact SHA 去重审计（语料约 5% 跨 run 内容重复，不审计会直接泄漏答案）；
2. target session 的所有 Node 从 memory 图中排除；
3. 两 arm 的 cascade 行为一致性检查（见 C.2 的静默退化）；
4. 每个 arm 的最终注入 token 配平核对（C.4）。

**已知限制**：
- run-local `evaluate.py` 在 EvoTrace 里只有 **18/121** 覆盖，因此能做 agent-outcome 的 task 集合
  远小于 18 个 task 的全集，必须先枚举哪些 task 有可用 evaluator；
- R3（完整 session 确定性重放）在公开数据上做不到，所以 outcome 实验只能是**新跑 target run**，
  不能是「重放历史 run」；
- 本地 qwen3.8 部署的 context 需求已实测：`prompt+completion` p90 = 46,359、p99 = 73,702，
  当前 `--max-model-len 32768` 会让 **25.10%** 的调用放不下，需提到 98,304。详见会话记录。

## C.9 与第一类指标的关系

第一类（检索 quality/latency，`gem_eg/metrics.py` + `groundtruth.py`）是**主证据**：确定性、
可审计、不需要 GPU、121/121 run 全覆盖。第二类（本附录）是**因果补充**：贵、随机、需要多 seed，
但只有它能回答「检索得更准，是否真的让 agent 干得更好」。

**两者必须分表报告。** 数据库更快推不出 agent 更聪明；反之，agent 变好也不能倒过来证明检索算子的
系统主张。若要连接二者，只能增加「同一 operating point 下的配对实验」，而不是把两组数字放进一张表。

---

# 附录 D — EvoMemBench → GEM Experience Graph 实现计划（2026-08-25）

## D.0 结论

**可以用，而且值得用；但必须拆成两条互不替代的证据链。**

1. **Agent-effectiveness track**：用 EvoMemBench 原始规模回答“过去 episode 的经验是否让后续
   episode 答得更准、工具执行得更成功”。先接 `CrossEp-Know`，再接 `CrossEp-Tool`。
2. **Database-systems track**：在不改变 target 答案和有效历史的前提下加入可审计 hard negatives，回答
   “Task ANN seed → native graph traversal → relational hard filter”是否比单向量和分阶段执行更快、更省工作，
   同时保持相同 outcome/recall。

原始 EvoMemBench **不能单独证明 multi-modal database 的系统收益**：CrossEp-Know 每个 context 只有
5–12 个 episode，CrossEp-Tool 当前 release 每个 environment 只有 50 个 episode。在这么小的 history 上，
HNSW、图遍历和 predicate pushdown 的固定成本很可能高于收益；这不是 GEM 失败，而是数据规模没有触发数据库
优化问题。

首选落地顺序仍然是：

```text
CrossEp-Know 12-context pilot
    → CrossEp-Know 120-context full agent-effectiveness run
    → CrossEp-Tool 4 environments + 12 source→target pairs
    → 独立 systems-scale expansion
```

论文目前是 2026-05-18 提交的 arXiv preprint，不是同行评审后的最终版本；以下结论同时核对了
[论文 HTML](https://arxiv.org/html/2605.18421)、[论文 PDF](https://arxiv.org/pdf/2605.18421) 和
[官方代码固定版本](https://github.com/DSAIL-Memory/EvoMemBench/tree/aa4cea8fd936b76b2d3591d3ef897030617dc43a)。

## D.1 论文、数据与实现的精读结果

### D.1.1 论文真正测的是什么

论文把 agent memory 分成两个轴：scope 是 in-episode / cross-episode，content 是 knowledge / execution。
对 GEM 最相关的是：

- `CrossEp-Know`：同一 background context 下顺序执行多个任务，每个 episode 结束后更新 memory，后续
  episode 使用它。论文给出 120 contexts、884 samples；固定数据文件验证为 884 行，扣除每个 context
  的首 episode 后，真正有可复用历史的 decision points 是 **764**。
- `CrossEp-Tool`：在四个 BFCL tool environments 内积累执行经验，并做 12 个有向
  source-environment → target-environment 的 frozen-bank transfer；target 阶段 memory 只读。

论文原生指标是：knowledge track 的 answer accuracy；execution track 的 success rate、progress score、
average steps；两类都统计 agent 与 memory module 的全部 LLM input/output tokens。官方协议还规定
knowledge top-k=10、execution top-k=3；cross-environment transfer 先建 source bank，再冻结 target。
这些定义见论文的 [benchmark protocol 与 metrics](https://arxiv.org/html/2605.18421#S5) 以及
[CrossEp-Tool 官方运行说明](https://github.com/DSAIL-Memory/EvoMemBench/blob/aa4cea8fd936b76b2d3591d3ef897030617dc43a/Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL/README.md)。

### D.1.2 CrossEp-Know 的重要语义

固定文件：

```text
revision: aa4cea8fd936b76b2d3591d3ef897030617dc43a
path: Cross-Episode-Knowledge/CROSSEP-KNOW/CL-bench_context_ge5.jsonl
sha256: f4652ddcf954dd33653f91c9b40ed6138617b92e0f80d470cd1c92bb50890ce9
contexts / episodes / reuse decisions: 120 / 884 / 764
episodes per context: min 5 / median 7 / max 12
```

四类 episode 分布为 306 Procedural Task Execution、294 Domain Knowledge Reasoning、257 Rule
System Application、27 Empirical Discovery & Simulation；与论文 Table 2 一致。每个 context 内 system
background 完全相同，且每个 episode **仍然拿到完整的当前 background**。所以它测的是：过去的解题经验能否
帮助 agent 更好地应用当前背景，而不是“背景已丢失时能否从数据库恢复原文”。论文也明确说明 required
knowledge 已在 context 中提供，并按 context 串行更新 memory。

另一个重要细节是评分：官方 `eval.py` 用 rubric 驱动的 LLM judge 给每个 sample 一个严格二值分，要求所有
rubric 同时通过才记 1；`requirement_status` 还能导出 rubric pass rate。因此正式报告应写成：

- `strict sample accuracy`：与论文主表对齐；
- `rubric pass rate`：诊断指标，不替换主指标；
- `reuse-conditioned accuracy`：只统计 ordinal > 0 的 764 个 decision，专门回答 cross-session reuse。

### D.1.3 CrossEp-Tool 的 release drift

论文 Table 2 和仓库顶层 README 都写成四个 environment 各 200、合计 800；但在固定 commit 的实际发布物中：

```text
BFCL_v4_multi_turn_ours.json:                 200 JSONL rows
possible_answer/BFCL_v4_multi_turn_ours.json: 200 JSONL rows
gorilla_fs / vehicle / trading / travel ids:  50 / 50 / 50 / 50
prompt sha256: 08fd17fa87a2b915a45aa0787668f8196d034dcfea4e6bab55d41abd4b13ea2d
answer sha256: e205c7118c20c19aa0d8b1850a59f85b7d6f7a91d05e32322670e4a429f43473
```

Table 6 的分数也以 2 个百分点为粒度，与每类 50 条相符。实现时以 **实际文件和 checksum** 为准，在 manifest
记录 `paper_claimed_n=800`、`released_n=200`，不把论文数字静默当成实际样本量。

### D.1.4 证据与复现边界

- 官方仓库在固定 commit 没有顶层 LICENSE；在许可澄清前，只做本地拉取和 adapter，不把上游数据复制进
  TriDB release。
- CrossEp-Know 没有“哪个历史 episode 对当前答案相关”的标注；CrossEp-Tool 也没有 experience relevance
  qrels。因此 retrieval Recall/nDCG 不能从原数据直接计算。
- CrossEp-Know 的主分数依赖 LLM judge；必须固定 judge endpoint/version、temperature、prompt hash，并在
  抽样上做双 judge 或人工一致性检查。
- 论文没有给 repeated-seed CI。我们的统计单位必须是 context/environment，而不是把相互依赖的 episode
  当独立样本。

## D.2 GEM 的 Experience Graph 建模

### D.2.1 一个关键选择：Experience unit 同时承载 task seed 与可注入经验

不需要先把当前 query 持久化为图节点。对每个已经完成的历史 episode，写一个可检索的 `ExperienceUnit`：

```text
ExperienceUnit
├── embedding     = task_signature 的 embedding
├── memory_payload= 可注入的事实 / 约束 / 程序 / failure-aware lesson
├── provenance    = source episode、trajectory hash、extractor/prompt version
├── validity      = valid_from / valid_to / state
└── relational metadata
    ├── track, context_id, environment, episode_ordinal
    ├── tool_schema_hash, source_phase, readonly
    └── online_observed_outcome（不是 withheld evaluator label）
```

当前 task 只产生 query vector；ANN 返回 task 相似的历史 `ExperienceUnit` 作为 graph seeds。这样
`tjs_open` 的单一 row predicate 可以同时约束 seed 和最终结果为 experience，不需要“Task 行用一种 filter、
Experience 行用另一种 filter”的新算子语义。

### D.2.2 Task signature 必须只含 decision 时已可见的信息

`CrossEp-Know`：

```text
track + context_category + sub_category + current user task
```

部分 user message 很长，最大超过 16 万字符。不能让 embedding provider 隐式截断。adapter 应显式抽取末尾
`Final Task` 区段；抽不到时使用确定性的 head+tail token window，并把原文 hash、截断规则和 embedding input
hash 写进 receipt。

`CrossEp-Tool`：只用 agent 当时已经看到的 **首个 user turn + tool/function schema signature**。不能在 episode
开始时把后续 user turns、ground-truth action、`possible_answer` 或 evaluator-only `initial_config` 放进 seed；
否则是未来信息泄漏。若要每 turn 重新检索，可以使用“截至当前 turn 的可见 history”，但必须作为另一种预注册
protocol。

这里与官方实现有一个必须公开的差异：CrossEp-Tool 的 FC 路径当前用 function docs 字符串做 memory query，
不是当前 task。故结果分成：

- `upstream-parity`：完全复现官方 query 构造；
- `gem-task-seed`：使用上述 agent-visible task signature，是本研究预先声明的扩展。

二者不能混成一张“复现论文”表。

### D.2.3 图节点与边

所有 topology 只写 native adjacency-list AM；`gem_edge` 只保留关系名、权重与审计元数据，不把边实现成
relational join table。

建议节点：

```text
Episode          原始 attempt 的身份与顺序
ExperienceUnit   真正被 ANN 检索和注入的经验
Skill            可复用策略 / workflow
Tool/API         函数或 environment capability
Constraint       rule、precondition、safety/format constraint
FailureMode      可观察到的失败模式
Evidence         trajectory 中的来源片段，仅作 provenance
```

建议关系名：

```text
Episode       -DERIVED_INTO-> ExperienceUnit
ExperienceUnit-APPLIES_SKILL-> Skill
ExperienceUnit-USES_TOOL----> Tool/API
ExperienceUnit-HAS_CONSTRAINT> Constraint
ExperienceUnit-AVOIDS-------> FailureMode
ExperienceUnit-DERIVED_FROM-> Evidence
Episode(i)    -PRECEDES-----> Episode(i+1)
new unit      -SUPERSEDES/CONTRADICTS-> old unit
```

为支持 `Experience → Skill/Tool/Constraint → Experience` 的 2-hop expansion，association 边在 native graph
中双向插入；`SUPERSEDES/CONTRADICTS` 走 GEM extension 类型，只用于 revision/propagation。关系名仍在
`gem_edge.rel`，native type 只保留 `association` 与 `extension` 两类，符合 GEM 现有合同。

**修正正文 §9.4：Rubric 不得进入在线可检索图。** 当前
`bench/agent_memory/evomembench/experience_graph.py` 生成 `EVALUATED_BY` 边，可保留为 run 完成后的 evaluation
audit graph，但不能 load 到 agent 可见的 `gem_unit`/native graph，也不能参与 task embedding、遍历、过滤或
extractor prompt。否则 rubric 会直接告诉 agent 评测要求，构成 benchmark leakage。

### D.2.4 三种操作分别在哪里发生

```text
current task_signature
        │ embed
        ▼
Vector search
  在 gem_unit.embedding 上 ANN，找 top-m 个相似历史 ExperienceUnit
        │ seeds
        ▼
Graph traversal
  从 seeds 沿 association 边做 bounded 2-hop/PPR expansion，发现共享 Skill、Tool、
  Constraint、FailureMode 的 vector-far bridge experiences
        │ candidates
        ▼
Relational scan/filter
  在同一 gem_unit row 上强制 scope、cutoff、state、validity、source/target phase、
  environment/tool-schema compatibility、ACL 与 token budget；不合法候选永远不进入 top-k
        │
        ▼
bounded top-k memory payload → token-matched prompt injection
```

对应 `tjs_open` 的概念查询为：

```sql
SELECT t
FROM tjs_open(
  'gem_unit', :k, :term_cond, :m_seeds, :hops, 'id',
  :eligible_experience_predicate,
  :task_embedding,
  NULL,
  :association_edge_type
) AS t;
```

谓词至少包含：

```text
scope_id = :context_or_source_bank
state = 'active'
metadata.kind = 'experience'
metadata.episode_ordinal < :cutoff
metadata.source_phase ∈ :allowed_phases
metadata.target_episode_id != :current_target
metadata.tool_schema_hash compatible with :current_schema
valid_from <= :decision_time < valid_to（或 valid_to IS NULL）
```

不要默认加 `official_score=success`：CrossEp-Know 的 rubric score 是 answer 之后离线得到的 withheld evaluation
信号，若用于后续 episode 的 online memory，会改变官方协议并泄漏标签。可以使用 trajectory 中 agent 当时可见的
tool feedback 或独立、rubric-free 的 online extractor verdict，但必须记录其来源和误差。

### D.2.5 现有 TriDB/GEM 的两个实现缺口

1. `RetrieveOperator._tjs_open()` 当前未显式传第 10 个 `edge_type` 参数，默认遍历 ANY。Experience retrieval
   必须新增独立 `Query.edge_kind/edge_type`，固定传 `association`，避免把 revision propagation edge 混入检索。
2. `tjs_open_graph_reached()` 已由 extension 暴露，但 GEM probes 尚未读取；还缺 `filter_rejected`。前者直接补
   receipt，后者若不扩展 C probe 就报告 N/A，不能用 `candidates_examined - returned` 冒充。

保持 TR-1：正式 fused path 只能使用 `tjs_open` 的 Open/Next/Close、bounded graph work 与 early termination；
任何“先物化全部 ANN seeds 和全部 reachable set 再排序”的实现只能作为 correctness reference，不能作为 GEM
产品路径。

## D.3 实验设计：把 memory algorithm 与 database architecture 分开

### D.3.1 Agent-effectiveness arms

所有臂冻结 backbone、episode order、context isolation、extractor、可注入 payload、top-k、prompt token budget、
sampling seed 与 judge；memory-on 用“替换”而不是“追加”保持 token 配平。

| Arm | 用途 |
|---|---|
| `no_memory` | 官方 long-context 下界；不 retrieve、不 update |
| `recent_fifo` | 相同 token budget 的最近历史基线 |
| `vector_only` | 相同 ExperienceUnit，仅 ANN + 同一 relational eligibility filter |
| `graph_relational` | 显式 anchor/邻接扩展 + filter，不用 query similarity；诊断图本身 |
| `gem_fused` | Task ANN seeds + native graph association traversal + pushed relational filter |
| `upstream_graphrag` | 复现官方已有 graph baseline，区分“GEM graph”与任意 GraphRAG |
| `oracle` | 原始 release 为 N/A；只有独立 qrels 冻结后才启用 |

为证明“图带来额外信号”，`vector_only` 与 `gem_fused` 必须复用完全相同的 ExperienceUnit、embedding、extractor
和 injection formatter。官方 GraphRAG 的图边主要由 chunk cosine threshold 建立，容易重复 vector 信号；GEM
应报告 typed skill/tool/constraint graph 与 `similarity-edge-only` 的 ablation。

### D.3.2 Systems-architecture arms

在选定相同 quality operating point 后比较：

| Arm | 说明 |
|---|---|
| `gem_fused_streaming` | 单次 `tjs_open`，同进程、同事务、native graph AM |
| `same_pg_staged_reference` | ANN → bounded graph calls → relational filter 的分阶段 correctness reference；明确非产品路径 |
| `vector_only_pg` | 相同 HNSW、filter、k 与并发下的单模态下界 |
| `multi_system_baseline` | 仅在已有 DEV-1171 baseline 可做严格同数据、同质量、同硬件比较时加入 |

Agent outcome 与 systems latency 分表；只有同一 operating point 的配对点可以建立联系。

### D.3.3 两条 workload 不能混在一起

**Native-quality track** 使用原始 884 / 200 条数据，不加任何 noise，回答 agent quality。

**Scaled-systems track** 逐级构造 10³、10⁴、10⁵、10⁶ 个 historical ExperienceUnit：

- cross-context hard negatives：测 relational scope/policy pruning；
- same-domain semantic negatives：测 ANN 候选流与 term condition；
- graph decoy branches：测 graph work budget、PPR 与 censoring；
- stale/superseded versions：测 temporal validity 与 revision filter。

扩容数据必须有唯一 source id、独立 content hash，并与 target/positive experience 去重。不得复制 target task、
ground-truth answer 或同一 positive 来“放大”样本量。扩容 track 只声称 systems scalability；其 accuracy 只做
“相对 native track 不退化”的 guardrail。

## D.4 指标：哪些现在能返回，哪些不能

### D.4.1 原始 release 直接可返回

| 层 | CrossEp-Know | CrossEp-Tool |
|---|---|---|
| 主 outcome | strict sample accuracy | exact execution success rate |
| 次 outcome | rubric pass rate | progress score / longest passing prefix |
| 行为效率 | reuse-conditioned accuracy、ordinal curve | average steps、force-terminated/error type |
| transfer | memory-on − off，正/负迁移率 | 4×4 source→target frozen-bank transfer matrix |
| 成本 | agent + extract + embed + injection tokens | agent + memory tokens、LLM/embed call counts |
| wall time | retrieve / generation / extract / total | retrieve / per-turn execution / update / total |

额外报告：

- `paired_gain_i = outcome_i(gem) - outcome_i(no_memory)`；
- positive / neutral / negative transfer fraction；
- quality-vs-history-length 曲线，只在 ordinal > 0 上计算 reuse 结论；
- success per 1M total tokens，以及达到同 success 的 token/wall-clock cost；
- CrossEp-Know 按 category、difficulty、ordinal 分层；CrossEp-Tool 按 environment、turn count、source-target
  alignment 分层。

第一 episode 没有历史。论文 parity 表保留全部样本，但 cross-session reuse 表必须排除首 episode，避免 120 个
必然 memory-empty 样本稀释效果。

### D.4.2 GEM/TriDB 应新增并返回的系统指标

| 层 | 指标 |
|---|---|
| Streaming latency | TTFR、time-to-k、p50/p95/p99、timeout/censor fraction |
| Vector work | candidates examined、termination reason、possibly budget-capped |
| Graph work | graph reached、edge steps examined、bridges injected、graph censored、PPR/membership mode |
| Relational work | eligible selectivity、filter rejected（实现前 N/A）、future/target/cross-scope violations |
| Write path | episode update/commit latency、extract latency、WAL bytes/episode、aborts/retries |
| Footprint | active units/fields、native edges、heap/HNSW/graph bytes、growth over episode ordinal |
| Throughput | open-loop achieved QPS、p95/p99 under concurrency、successful episodes/s |
| Transaction correctness | snapshot/cutoff receipt、one-xid retrieve+reinforce、target readonly、receipt digest |

每个 query receipt 至少固定：dataset/revision/checksum、task/input hash、episode/cutoff、eligible-scope hash、
embedding model/input hash、extractor/prompt hash、returned ids+ranks+provenance、injection hash/tokens、全部 TJS probes、
transaction id 和 latency 分段。

### D.4.3 当前必须返回 N/A 的指标

没有独立 qrels 时：

```text
Evidence Recall@k       = N/A
Dependency Recall@k     = N/A
nDCG@k / MRR            = N/A
oracle gap              = N/A
graph path recall       = N/A
SUPPORTS/HELPED precision= N/A
```

“所有 prior episodes”只是 eligibility set，不是 relevance set；把它当 ground truth 会奖励 indiscriminate retrieval。

若需要这些指标，新增一个独立 annotation release：从 100–200 个 reuse decisions 抽样，两名标注者只看 target
任务与候选历史，标注 `necessary / helpful / irrelevant / harmful` 及最小 provenance path；裁决分歧、报告
Cohen's κ，再冻结 qrels SHA。标注过程不得把 target rubric/ground truth 暴露给 online extractor。

## D.5 统计分析

- CrossEp-Know 用 **context-clustered paired bootstrap 95% CI**；CrossEp-Tool 用 environment/task-level paired
  bootstrap，并单独展示 12 个 transfer cells。不能对 884/200 个 episode 做 iid 检验。
- 对 stochastic generation 至少运行 3 个 generation seeds；embedding/extractor 若非确定性也分别冻结 seed。
- 预注册主 comparison：`gem_fused - vector_only` 和 `gem_fused - no_memory`；其余为诊断，避免多重比较后挑最好点。
- 同时报告均值、median、CI 和样本数；失败、超时、judge parse failure 不静默丢弃，按预注册规则计 0 或单列。
- 因官方顺序只是文件顺序，增加一次 within-context order permutation sensitivity；它不替换官方固定顺序主表。

## D.6 代码实施计划

### Phase 0 — Source freeze 与 protocol gate

1. 在 `bench/agent_memory/evomembench/manifests/` 固定 repo revision、两个 track 的文件 checksum、实际 row
   count、paper-claimed count、license status、task-id sets。
2. 扩展 `dataset.py`：保留现有 CrossEp-Know loader；新增 CrossEp-Tool prompt/answer/四个 ids loader，校验
   200 条与 50×4 分区。
3. 加 protocol oracle：同 context 顺序、`ordinal < cutoff`、target exclusion、source-bank frozen、target readonly。
4. 将 rubric/possible_answer/evaluator state 标为 `evaluation_only`，任何 online serialization 命中即 fail-fast。

**Gate**：checksum/count/order 全通过；许可未澄清时 dataset 仍只允许外部路径引用。

### Phase 1 — Query contract 与 graph safety

1. 在 GEM `Query` 增加显式 `edge_type`，`RetrieveOperator._tjs_open()` 传 association type；不要复用
   `extra_filter` 同时表达 row predicate 和 edge semantics。
2. probes 增加 `tjs_open_graph_reached()`；`filter_rejected` 未实现前保持 `null`。
3. refactor `experience_graph.py`：在线图只含 Episode/Experience/Skill/Tool/Constraint/FailureMode/Evidence；rubric
   图移到 evaluation namespace，不 load 到 native graph。
4. predicate 使用 whitelisted builder 和 SQL literal，不把 dataset/user text拼成 raw SQL。

**Gate**：association-only traversal、future/target/rubric leakage 全为 0；TR-1 early-close test 通过 stock PG16/17。

### Phase 2 — Experience extraction 与 Task seed

1. 新增 `task_signature.py`：CrossEp-Know Final Task/head-tail 规则；CrossEp-Tool agent-visible first-turn + schema
   规则；输出原文/input hash。
2. 新增 `extractor.py`：把 completed trajectory 转成单个 bounded `memory_payload`，再抽取 typed concepts 和
   provenance；rubric-free、版本化、token/call 可计量。
3. `load.py` 将 ExperienceUnit、concept nodes 和双向 association edges 在一个 PostgreSQL transaction 中写入；
   不创建 relational edge table。
4. 为 metadata cutoff/kind/source-phase/schema predicates 建 expression/partial indexes，并记录 index bytes。

**Gate**：相同输入重复 extraction 的 schema 合法率 100%；所有经验可追溯到 source episode；无 evaluator-only
字段；一次失败不会留下半写 graph/row。

### Phase 3 — 两个官方 adapter

1. `crossep_know_backend.py` 实现官方 `retrieve(query)` / `extract(trajectory)` 接口。
2. `crossep_tool_backend.py` 实现 `utilize()` / `update()`、in-env writable bank 和 cross-env readonly clone/snapshot。
3. 同时提供 `upstream-parity` 与 `gem-task-seed` query mode；默认实验配置明确选择其一。
4. memory injection 做 item count 与 token 双限额；所有 arms 使用同一 formatter。

**Gate**：no-memory 输出与官方 runner parity；首 episode empty-memory；target readonly；同 token budget arms 的
injection token 差在预设容差内。

### Phase 4 — Receipts、metrics 与 pilots

1. 新增 `receipts.py`、`metrics.py`、`runner.py`，复用 MemoryArena 的内容寻址 receipt/leakage gate。
2. 先跑 12 个 CrossEp-Know contexts：四 category 各 3 个，覆盖 easy/medium/hard；跑
   no-memory/FIFO/vector/graph/fused。
3. smoke 只验证 wiring 时可用 deterministic local embedding；任何 agent-quality 表必须使用冻结的真实
   embedding 和相同 backbone。
4. pilot 后只根据预注册 gate 调整 bug/预算，不按 outcome 挑 context 或 operating point。

**Gate**：所有 result/receipt 数一致；leakage=0；TJS probes 非空；judge failure 可追踪；fused 与 exact bounded
reference 的 top-k parity 达预设阈值。

### Phase 5 — Full agent-effectiveness run

1. CrossEp-Know 120 contexts：论文 parity 表 + 764 decision reuse 表。
2. CrossEp-Tool 实际 release 200 episodes：四个 in-env banks + 12 frozen transfer pairs。
3. 3 generation seeds；context/environment-clustered CI；easy/hard、knowledge/procedure、aligned/misaligned transfer
   分层。

**Gate**：只有 `gem_fused - vector_only` 的 paired effect 与 CI 能回答 graph 增量价值；只有
`gem_fused - no_memory` 能回答整体 memory 价值。负结果同样保留。

### Phase 6 — Systems scale 与 matched-quality

1. 固定从 Phase 5 选出的 quality operating point；扫 history size、relational selectivity、graph fan-out、
   `m_seeds/hops/term_cond/graph_work_budget`。
2. cold-cache / warm-cache、QPS 1/5/10、并发与 p99；同时量 ingest/update/WAL。
3. fused vs same-PG staged reference 在相同 returned-set quality 下比较 TTFR、time-to-k 和 examined work。
4. GX10 上才能做 ARM64/CUDA/128GB 正式 sign-off；x86 stock-PG 结果只声明 operator/adapter correctness 与
   off-target performance。

## D.7 最小可交付物与停止条件

最小可交付物不是“跑出一个更高 accuracy”，而是：

```text
1 个 pinned manifest
2 个 dataset adapters
1 个 rubric-free task/experience extractor
1 个 association-only fused query path
5 个公平 arms + content-addressed receipts
1 张 agent outcome 表
1 张 systems matched-quality 表
1 张 leakage/censoring/coverage gate 表
```

必须停止并判 run 无效的条件：future/target/rubric/answer 任一泄漏；target bank 被写；两个 arm 的 injection token
未配平；model/judge/extractor 版本漂移；graph work 被 censor 却当 exact；CrossEp-Tool 仍按 800 报分母；把
retrieval qrels 缺失记成 recall=0；或 fused path 物化完整 reach 违反 TR-1。

## D.8 最强反例与最终判定

最强反例是：EvoMemBench 的提升可能来自“LLM 看到了更多历史文本”，而不是 vector+graph+relational 的数据库
能力；在原生 5–12/50 条 history 上，任何数据库延迟优势也缺乏外推性。这个反例不能靠只跑 `gem_fused` 消除。

因此最终 claim 必须满足两项：

1. **算法控制**：相同 ExperienceUnit、extractor、embedding、top-k 与 injection tokens 下，fused 相对
   vector-only/graph-only 的 outcome 或 harmful-transfer 更好；
2. **系统控制**：相同 returned-set quality 下，fused streaming 相对 staged execution 在足够大、独立扩容的
   history 上降低 TTFR/time-to-k、examined work 或 tail latency，同时 leakage=0、censoring 完整披露。

若只满足第一项，只能说 Experience Graph 改善了 agent memory；若只满足第二项，只能说 TriDB 更高效地执行
了三模态查询；**两项同时满足，才支持“multi-modal database 作为 agent memory 有可测收益”的完整结论。**

## D.9 实施状态（2026-08-25）

本轮已完成可在 x86 stock-PG 验证的 pilot 路径；GPU agent-quality 实验已挂到现有 Figure10 队列之后，尚未把
排队状态冒充为结果。

- `protocol_manifest.json` 固定上游 revision 与三个资产 SHA；本地固定版本实查通过：Know
  `120 / 884 / 764`，Tool `200`。数据仍从 repo 外部路径读取，不随 TriDB 分发。
- `tool_dataset.py` 将 Tool prompts 与 `evaluation.py` 的 possible answers 分开加载；`leakage.py` 对
  rubric、ground truth、expected tool path 等字段 fail closed。
- `task_signature.py` 的 Know seed 仅含 category/subcategory/current user task，超长输入采用带原文 SHA 的
  deterministic head-tail bound；Tool seed 只含 admission 时可见的首个 user turn 与允许的 schema。
- `modeling.py` 已实现 `ExperienceUnit`、Category/Skill/Environment/Tool/Function feature nodes、双向 native
  association arcs 与 cutoff-bearing canonical query。memory payload 只含已经完成的 task/response/trajectory。
- 在线 `experience_graph.py` 已移除 Rubric/EVALUATED_BY；它们只存在于显式 `evaluation_only` audit graph。
- GEM `Query.graph_edge_kind` 默认 association；`tjs_open` 显式传第十个 edge type。probes 增加
  `graph_reached` 能力检测：新扩展记录实值，当前已安装的 stock-PG `tjs_pg 0.2.0` 尚无该 SQL function，故诚实
  返回 `null` 和 `graph_reached_available=false`，不伪造 0。
- graph-only 路径把 scope/state/cutoff predicate 和 `LIMIT k` 推入 native traversal 的 bounded SQL，避免先取完
  reachable set 再切片，保持 TR-1。
- `run_pilot.py` 实现 memory-off/FIFO/vector/graph+relational/fused 五臂的 context-serial replay、post-answer
  admission、内容寻址 retrieval receipt、token budget、model/embedding call ledger、WAL/update/probe 记录；
  `summarize.py` 汇总 strict accuracy、reuse-conditioned delta、p50/p95、examined work 与 N/A 指标。
- `scripts/evomembench_gem_pilot_after_gpu_idle.sh` 是 one-shot launcher：等 Figure10 队列结束，再要求两张卡的
  request queue 为空且 compute utilization 连续三次低于 10%，之后才 clone 固定 revision、验 SHA、创建独立
  DB、运行 12-context 五臂 pilot 与官方 rubric judge。已有 output/DB 或正式实验失败时拒绝重跑并保留现场。

验证证据：Python 定向 suite `90 passed`，ruff 通过；真实 stock-PG live smoke 写入 2 个 historical episodes、
4 个 unit、8 条双向 association arcs，并由 fused query 返回 cutoff 前的 episode。该 smoke 的 probe 为
`candidates_examined=8`、`graph_examined=4`、`graph_censored=false`、association `edge_type=3`；这是 wiring 证据，
不是 agent outcome 或 GX10 performance sign-off。

## D.10 继续实施记录：Tool adapter、task-seeded traversal 与正式 pilot（2026-08-25）

在 D.9 的 Know pilot 基础上，现已补齐下列实现；这里区分“代码/接线已验证”和“agent outcome 尚在运行”，不把
后台进程存活当作实验结论。

1. `extractor.py` 只接收 agent-visible task/trajectory/tool feedback，任何 rubric、ground truth、possible answer、
   expected path/score 字段都会 fail closed。输出一个有字符上限和原文 SHA 的 Experience payload；Category、Skill、
   Environment、Tool、Function 来自 pinned prompt/schema，FailureMode 只从实际可见的 tool error 归入固定小词表。
   extractor 不调用 LLM，因此 `llm_calls=0`，不会把 judge label 变成下一 session 的 online 信号。
2. `load.py` 把 Experience row/vector、payload/provenance、concept vertices、双向 concept association arcs，以及相邻
   Experience 的 native `PRECEDES/FOLLOWS` association arcs 交给一次 GEM
   ingest transition；失败由同一个 PostgreSQL transaction 回滚。`gem_unit` 新增 experience cutoff 与
   source-phase/schema/validity expression indexes，但 edge topology 仍只写 native graph AM，`gem_edge` 不是 join table。
3. `crossep_know_backend.py` 已实现官方 `retrieve(query) / extract(content, **kwargs)` surface；
   `crossep_tool_backend.py` 已实现官方 `begin_sample / utilize / update / drain_usage / load_from_disk` surface。
   Tool 的 Phase 1 为每个 environment 一个 writable scope；Phase 2 不是复制出第二个图数据库，而是对 source scope
   生成内容寻址 logical snapshot，并以 `readonly=True` 查询。每个 target sample 后重算 Experience payload + incident
   edge digest，若 source bank 改变立即判 run 无效。
4. 两种 query mode 已分离。`upstream_parity` 原样使用官方 runner 交给 memory 的 system prompt/function docs；
   `gem_task_seed` 由 runner 在 sample 开始前注入首个可见 user turn + allowed function schema 的 signature。后者执行
   `task embedding → ANN top-m seeds → association-only native bounded traversal → scope/state/cutoff/source-phase filter`
   的单次 `tjs_open`，即本计划要求的 **task-seeded / vector-seeded graph traversal**。不得把未来 user turn、expected
   path 或 possible answer 放进 seed。
5. `run_tool.py` 复用 pinned BFCL handler/tool executor，串行建立四个 50-episode in-env banks，再覆盖全部 12 个
   ordered source→target readonly transfer cells；支持 memory-off/FIFO/vector/graph+relational/fused 五臂。
   `evaluate_tool.py` 只在 generation 完成后启动官方 evaluator；`summarize_tool.py` 返回 success rate、longest-prefix
   progress、steps、inference/memory/total latency、token split、4×4 非对角 transfer matrix 与 cell-clustered paired
   bootstrap。固定 release 分母始终是 200，不写 800。
6. `scale.py/run_systems.py` 已实现 systems-only 的独立扩容记录：cross-scope hard negatives、same-domain semantic
   negatives、graph decoy branches、stale versions；每条有唯一 source id/content hash，并拒绝 native target/positive
   hash。harness 对 fused streaming 与 bounded same-PG staged diagnostic 返回 returned-set Jaccard、examined/censor
   probes、WAL、relation/index/native-graph footprint，以及 QPS 1/5/10 的 warm open-loop p50/p95/p99。它明确不把 synthetic
   accuracy 当 agent quality；当前还没有真实 client-cursor TTFR/time-to-k，也没有做 destructive cold-cache eviction，
   所以这两项继续标 N/A/未实施，而不是用总 latency 冒充。

stock-PG live adapter preflight 使用独立数据库和 deterministic 1024-d stub embedding：第一个 Tool episode 在单事务
产生 Experience/concept/native edges；第二个任务的 fused receipt 返回 cutoff 前 unit，`edge_type=3 (association)`、
`candidates_examined=7`、`graph_examined=6`、`graph_censored=false`；再以同一 source snapshot 对 `travel_api` target
只读检索，snapshot digest 保持不变。这只证明 x86 PG16 的事务、filter、task-seed 与 native traversal wiring。

正式 GPU pilot 使用 run id `evomembench_gem_cross_session_pilot_b2_20260825`，独立 DB
`evomembench_gem_pilot_b2_20260825`。GPU 0 是 `Qwen/Qwen3.8-27B-FP8` answer/judge，GPU 1 是固定 revision
`Qwen/Qwen3-Embedding-0.6B`；12-context × 五臂 generation 已启动，完成后才运行官方 rubric judge 和 summary。
截至本记录写入时它仍在 memory-off generation，故 accuracy、paired delta、CI 和预计完成时间都还不是结果。

原始论文/仓库指标与 GEM 新增指标的边界保持不变：原始 Know 主指标是 rubric-driven strict solving accuracy，原始
Tool 主指标是 exact execution success 与 progress score，并有 steps、latency、agent/memory/embedding token 成本；
vector candidates、graph reached/edge steps、relational selectivity、TTFR、WAL、footprint 是 GEM 系统指标，不是
EvoMemBench 原始 leaderboard 指标。没有独立 qrels 时 Recall@n、nDCG、MRR、path recall 和 oracle gap 仍为 N/A。

### D.10.1 b2 outcome validity correction

正式启动后复核预注册 gate，发现 b2 runner 仍把 memory block **追加**到 system prompt；相同的
`injection_token_budget=4096` 只是相同上限，不等于 memory-off/vector/graph/fused 的总输入 token 配平。这违反 D.3.1
的 replacement policy 和 D.7 的停止条件。因此 b2 即使完成 generation/judge，也只能保留为 wiring、成本和候选工作量
pilot，**不得把其 accuracy delta/CI 作为 multi-modal memory 的因果结果**；实验合同不允许除 hard timeout 外自动杀死
已启动 formal attempt，所以不在后台中途终止，也不自动重跑。

后续代码已改为 `replace_equal_total_input_tokens`：先在每个原始 system prompt 中保留至少 128 tokens，再用同 tokenizer
计数的 memory block 等量替换尾部 token slot；memory-off 保持原 prompt，memory-on 的 system token count 必须与原值
完全相同。run receipt 固定 `injection_policy`/`token_counter`，每个 prediction/receipt 记录 original/final/replaced tokens；
`summarize.py` 对缺少该 policy 或 original≠final 的 run fail closed。b2 的进程在代码修正前已经 import 旧 runner，故不会
被事后升级成有效 outcome run。

CrossEp-Tool 的 FC surface 原本没有可替换的 system prompt，因此采用显式双协议：`upstream_parity` 保留官方
function-doc query 和追加行为，只用于论文 parity；`gem_task_seed` 的五个因果 arms 都预置相同长度的 neutral system
slot，memory-off 保留 neutral slot，memory arms 用“meaningful Experience + neutral padding”的同 token 长度字符串整体替换
该 slot。runner 使用 answer vLLM 的 `/tokenize` 与 `/detokenize`，不是用近似 tokenizer；receipt 记录 meaningful、padding
与 total slot tokens，Tool summary 对任一 slot 长度漂移 fail closed。写回 extractor 会去除这个控制用 system slot，避免把
padding 或已注入 memory 递归写成下一条 experience。

### D.10.2 systems baseline 语义修正与 off-target smoke

第一次 systems smoke 的 staged comparator 只返回 graph reach，遗漏了 fused operator 最终预算里的 vector winners；同时
stock `tjs_pg` 默认 graph scoring 是 PPR，而 comparator 实际实现的是 membership merge。因此当时的 returned-set
Jaccard=0.5 既不是 correctness failure，也不能作为两条路径的 matched-quality 证据。

`run_systems.py` 现把该路径明确改名为 **bounded staged membership baseline**，并固定、记录同一组
`k/m_seeds/hops/association edge/relational predicate/graph_work_budget`。它先有界物化 vector candidate window，用相同的
nearest seeds 调用 native `gph_traverse_bounded`，共享并记录 edge-step budget/censor flag，再按 `k/2` bridge cap（有 bridge
时至少 1）与 vector winners 合并。它仍不是 correctness oracle：分开的 SQL statements 无法逐项复现 fused operator 的
relaxed-order incremental HNSW stream；returned-set Jaccard 只作为 diagnostic overlap。product path 仍只有单次
`tjs_open`，没有引入 blocking operator。

在新独立数据库 `evomembench_systems_smoke_b3_20260825` 上，16 条 synthetic history、deterministic 1024-d embedding、
`k=3,m=2,hops=2,term_cond=8,graph_work_budget=65536,membership` 的 x86 stock-PG wiring smoke 得到：fused 与 staged 均返回
`[17,16,4]`，Jaccard=1.0，两边 `graph_examined=7` 且未 censor。1 秒短窗的 warm open-loop 在 offered 1/5/10 QPS 下分别
完成 1/5/10 请求，observed achieved QPS 为 1/5/10、error=0；样本太少，延迟分位数只验证计数/接线，不构成性能结论。
该轮还修正了 nearest-rank p95/p99 与 observation-window 计算。真实 embedding、10^3–10^6 history、足够长稳态窗口、
cold-cache/TTFR 以及 GX10 sign-off 仍待正式 systems run。

后续 b4 smoke 还把 relation filter 的 population denominator 显式化：16 条 Experience 中 4 条是 cross-scope、4 条是
stale，故当前 query 的 generator-known eligible=8、eligibility fraction=0.5；该值来自 deterministic generator label，
不额外执行 blocking database `count(*)`，也不冒充 TJS stream 内部的 candidate-pass selectivity。

Material Passport 也已 fail closed：若 summary 标记 `outcome_valid=false`，verification status 必须是
`executed_systems_only_outcome_invalid`，并在 limitations 中显式禁止 agent-outcome claim。这样 b2 即使跑完 judge，也不会
被包装成有效的 memory quality 实验。

b2 还有第二个 outcome gate failure：generation `max_tokens=1024`，而 pinned upstream result 记录使用 4096；已完成臂中
84%–100% 的回答恰好停在 1024 tokens，说明 ceiling 不是偶发。invalid summary 会报告 observed completion maximum 及其
hit fraction，但 accuracy 仍为 null。未排队的 b3 launcher 已固定为 4096，并把 configured cap 与 cap-hit fraction 写入
run receipt/summary。
Passport 还会对 EvoMemBench adapter、GEM Python/schema、native TJS/graph C、相关测试与 launchers 生成逐文件 SHA-256
和 canonical implementation-manifest SHA；因此即使当前工作树含未提交文件，也不只靠 `git status` 猜测实际运行代码。

Experience graph 的跨 session 时序边也做了独立 live smoke：连续写入两条 Experience 后，metadata edge 分别为
`0 --PRECEDES--> 2` 与 `2 --FOLLOWS--> 0`，两条 topology 都是 native `association` edge；从第一条 Experience 做一跳
native traversal 可到达共享 skill feature 和第二条 Experience。previous-experience lookup 使用 scope +
`experience_ordinal` cutoff index，并在同一 ingest transition 内写边，因此不是事后 relational edge join。
共享 feature hub 也按 `(scope_id,title)` unique index 在 transition 内复用：第三条相同 skill 的 live admission 只新增 1 个
Experience unit、0 unit update、1 个 embedding sequence，并写 2 条 concept arcs + 2 条 temporal arcs，避免每个 session
重复 re-embed/rewrite hub。

后续 GPU 队列已有 fail-closed 的 `scripts/evomembench_gem_tool_pilot_t1.sh`：先跑 memory-off 与 GEM-fused 两臂、每个
environment 1 条（phase 1 共 8 samples），再跑 fused 的全部 12 个 ordered transfer cells（phase 2 共 12 samples），合计
20 个 Tool samples。它会调用 pinned official evaluator、输出完整 inference/memory/total latency 与 input/output/embedding
token splits、success/progress/steps、12-cell paired effect，并生成 Material Passport；该 launcher 拒绝已有 DB/output，失败
后保留现场且不重试。只有这个 t1 gate 通过，才允许扩到 50×4 和三 seeds。

当前定向 EvoMemBench/GEM suite 为 104 passed；全量 `make test` 为 1121 passed、30 skipped、3 failed，三个 failure 都是
既有 E1 polyglot integration test 无法连接未启动的 `127.0.0.1:19530` Milvus，堆栈在 backend construction 处发生，与
本 track 无关。没有为了制造全绿结论而自动启动/修改该外部 baseline。

修正后的 Know launcher 为 `scripts/evomembench_gem_know_pilot_b3.sh`，它在 judge 前逐条检查 run-level replacement policy、
每臂样本数，以及所有实际注入样本的 original/final system token parity；失败时不支付 judge 成本。b2 是无效 causal
attempt，仍按 no-auto-retry 合同原样封存；用户随后明确授权继续实施并执行既定计划，因此 b3 作为**新的、预注册配置已
修正的首次因果有效 attempt**，通过 `scripts/evomembench_after_systems_queue_b3.sh` 单独排在 Tool t1 与 systems s1 之后。
该门控要求 systems receipt/summary 完整、`agent_outcome_measured=false` 且 Material Passport 为
`executed_systems_only`，并在已有 b3 output 时拒绝启动，所以不会把 b2 静默重试或覆盖。

Tool t1 成功后还排了一个独立 systems s1 gate：真实 Qwen3-Embedding-0.6B、1k/10k Experience histories、
membership-matched fused/staged、1/5/10 QPS×10 秒、共享 65,536 edge-step budget；它只在 Tool summary 的
`outcome_valid=true` 且 Passport=`executed_and_summarized` 后启动。systems summary/Passport 明确是
`executed_systems_only`，不含 agent outcome；本 gate 不自动扩到 100k/1M。

截至 2026-08-25 11:55 PDT，b2 主进程、b2→Tool、Tool→systems、systems→b3 四个 user-systemd unit 均为
`active`；新增 b3 门控脚本通过 `bash -n`，EvoMemBench 定向测试为 `22 passed`。这里的 `active` 只证明进程/队列
存活，不证明任何 outcome 或 performance 结论。

---

# 附录 E — EvoTrace × GEM × OpenEvolve：agent-outcome 实验方案（2026-08-24）

## E.0 Material Passport

- Verification Status：**方案已定，未执行**。本附录的 evaluator 可行性判定来自实际读取
  `data/evotrace/raw/shinkaevolve/*/evaluate.py`；运行时长与 GPU 预算为**算术估算**，未实测。
- 上游依据：[AlphaEvolve, arXiv:2506.13131](https://ar5iv.labs.arxiv.org/html/2506.13131)（含 §Ablations / Figure 8）；
  本地 `openevolve 0.3.2` 源码（审计见附录 C）。
- 前置：附录 C 的注入点、token 配平、cascade 陷阱结论。

## E.1 先纠正一件事：哪些指标是 AlphaEvolve 的，哪些是我加的

上一轮我提的三个指标，**只有一个真正是 AlphaEvolve 报的**：

| 指标 | AlphaEvolve 是否报告 | 依据 |
|---|---|---|
| **best-so-far vs 计算预算 曲线** | ✅ **是，就是 Figure 8** | 原文："Each curve shows the performance of an individual setting with increasing compute budget, averaged over all considered targets." x 轴 = compute budget，y 轴 = target metric |
| **per-task best score** | ✅ 报，但**刻意不做跨 task 平均** | 50+ 数学问题的聚合是分类率："In 75% of the cases *AlphaEvolve* rediscovered the best known constructions, and in 20% of the cases it discovered a new object that is better than a previously known best construction." |
| **time-to-threshold / 达标迭代数** | ⚠️ **不是显式指标**，只有定性表述 | 原文对比 FunSearch 的 "millions of LLM samples used" 与 AlphaEvolve 的 "thousands of LLM samples suffice"。这是 sample-efficiency 的**说法**，不是它报的**数字** |
| **best-so-far 曲线的 AUC** | ❌ **AlphaEvolve 没有报** | 曲线是它的，把曲线降维成 AUC 是我加的。属于我们的自定义指标，必须如此标注 |

**更重要的收获：AlphaEvolve 的 ablation 结构本身就是我们的模板。** 它的五个 arm 是：

```
No evolution                 —— 反复把同一初始程序喂给 LLM，不用 database
No context in the prompt     —— 去掉问题相关上下文     ← 与我们的 memory-off 结构同型
No meta prompt evolution
No full-file evolution
Small base LLM only
```

其中 **"No context in the prompt"** 与我们要做的 memory-on/off 是同一种消融形状。这给了方案两个直接约束：

1. **画图照 Figure 8 画**：x 轴 = 计算预算（我们用 iteration 数或累计 LLM token），
   y 轴 = 目标 metric，**跨 target 平均**，每个 arm 一条曲线；
2. **跨 task 聚合照 AlphaEvolve 的做法回避分数平均**：报「达到/超过历史最佳的任务比例」这类
   分类率，而不是把不可通约的 `combined_score` 平均。这与 `gem_eg/groundtruth.py`
   拒绝跨 evaluator 比较 reward 是同一条规则。

## E.2 EvoTrace + GEM 到底"加强"了什么

不是"提供数据"这么简单。EvoTrace 带来四样东西，其中第 2、3 项是方案能不能出结论的关键。

### E.2.1 提供 memory 的内容（必要条件）

10,672 个真实程序 / 121 个 session / 17 个有 evaluator 的 task，且每个程序都带真实 reward。
没有它，memory-on arm 无物可检索；自己造这个语料要跑 121×100 次演化。

### E.2.2 提供 **oracle 上界 arm** —— 把 A/B 变成可诊断的三点

因为 EvoTrace 记录了每个历史程序**实际拿到的分数**，我们有后见之明。于是可以加一个
**Arm D「oracle memory」**：不做检索，直接注入该 task 历史上最好的 M 个程序（排除 held-out session）。

这条 arm 的成本与 Arm B 相同，但它把二元对照变成三点诊断：

| 观察 | 结论 |
|---|---|
| B ≈ D | **检索已经足够好**，瓶颈不在检索 |
| B ≈ A 但 D ≫ A | **记忆有用，是我们的检索不行** ← 唯一可行动的诊断 |
| D ≈ A | **记忆在这个任务上根本没用**，整个方向需要重估，与检索质量无关 |

没有 D，A vs B 打平时你无法区分「记忆没用」和「检索没做好」——而这两个结论对项目的含义完全相反。

### E.2.3 提供 **negative control** —— 注入通道的有效性检查

`gem_eg/groundtruth.py:is_dead_end()` 已实现：某节点之后再无任何后代超过它。
据此构造 **Arm E「poisoned memory」**：注入已知死枝。

- outcome **下降** → 证明注入通道确实在影响 agent，其余 arm 的差异可以解读；
- outcome **不变** → **注入通道是惰性的**（prompt 被截断、inspiration 区块被去重吃掉、
  模型忽略该区块……），此时 A/B/C/D 的所有结论都不成立。

这是 manipulation check。成本只有一个 task × 3 seed，但**不可省** —— 它是唯一能证伪
"我们其实什么都没注入进去"的实验。

### E.2.4 同 schema 回灌（可选，为后续铺路）

OpenEvolve 的 `EvolutionTraceConfig` 吐出的就是 EvoTrace 的 schema（附录 C.6）。
新 run 可经 `tools/evotrace/normalize.py` 进 `gem_eg_*`，支持后续的累积记忆 / 多轮复用实验。
必须 `include_code: true`，否则 artifact 层为空、无法做泄漏审计。

## E.3 可行性：evaluator 实查结果

`find data/evotrace/raw -name evaluate.py` = **18 个，全部在 `shinkaevolve/` 下**，
覆盖 **17 / 18 个 task**（唯一缺失：`math:signal_processing`，该 task 的 backend 为 EGO，无 S）。

| 域 | task 数 | 可跑性 | 依赖 |
|---|---:|---|---|
| **math** | 7 | ✅ **可以直接跑** | 仅 `numpy` / `sympy` / stdlib。文件头的 `sys.path.insert('/home/user/anon/skydiscover/...')` 是匿名化残留，实际未从该路径 import 任何符号 |
| **ale** | 10 | ⚠️ 需补依赖 | `ale_bench` 包 + `ale-bench-lite-problems/<task>` 测试数据 + C++ 工具链，来自 [SkyDiscover](https://github.com/skydiscover-ai/skydiscover)（EvoTraceDoc 审计 rev `59643ef8`） |

入口签名正是 OpenEvolve 的标准 evaluator 接口 `evaluate(program_path)`；
`circle_packing` 还自带 `evaluate_stage1` / `evaluate_stage2`，cascade 开箱即用。

> **结论：Phase 1/2 只做 math 的 7 个 task。** ALE 的 10 个作为 Phase 4 的扩展，
> 需先独立验证 SkyDiscover + ale-bench 能在本机复现历史分数，否则评测契约不成立。

七个 math task：`circle_packing`、`heilbronn_triangle`、`heilbronn_convex_13`、
`first/second/third_autocorr_ineq`、`uncertainty_ineq`。

## E.4 实验 arm 定义

所有 arm 共享同一 `num_diverse_programs = N`，**只改 inspiration 的来源**（附录 C.4）：

| Arm | inspiration 来源 | 作用 |
|---|---|---|
| **A** memory-off | N 条全部来自本 run 的 database（stock OpenEvolve） | 基线。注意这**不是**「无上下文」 |
| **B** GEM | M 条替换为 GEM 跨 session 检索（vector-seeded + typed traversal + 谓词下推），N−M 条仍来自本 run | 主张 |
| **C** vector-only | 同 B，但检索退化为纯 ANN，无图遍历、无下推 | 隔离「图」的贡献 |
| **D** oracle | M 条替换为该 task 历史最优（后见之明，排除 held-out） | **上界**（E.2.2） |
| **E** poisoned | M 条替换为已知死枝（`is_dead_end`） | **negative control**（E.2.3） |
| **F** no-inspiration | N = 0 | 注入率扫描的下界 |

`N` 取 OpenEvolve 默认 5（`num_top_programs=3 + num_diverse_programs=2`）中的
`num_diverse_programs`，Phase 2 固定 `N = M = 5`（把 diverse 区块整块替换），Phase 3 再扫。

## E.5 分阶段方案

### Phase 0 —— 前置闸门（不通过不得进入 Phase 1）

1. **qwen3.8 部署实测**：`--max-model-len` 提到 98,304（实测 p99 = 73,702，当前 32,768
   会让 25.10% 的调用放不下）；用真实 prompt 长度分布回放 50 条，量实际 tok/s 与峰值显存。
   两卡 tp=1 × 2 副本（无 NVLink，不用 tp=2）。
2. **evaluator 复现闸**：取每个 math task 的历史最优程序，用其 run-local `evaluate.py` 重跑，
   **分数必须复现**。不复现的 task 直接出局 —— 评测契约不成立时任何 outcome 都无意义。
3. **cascade 一致性**：确认所有 arm 的 `cascade_evaluation` 与 `cascade_thresholds` 相同，
   且 evaluator 确实定义了 `evaluate_stage1`（否则会静默退化，附录 C.2）。
4. **泄漏审计**：source/target artifact SHA 去重；target session 的所有 Node 从 memory 图中排除；
   语料约 5% 跨 run 内容重复，不审计会直接泄漏答案。
5. **token 配平核对**：记录每个 arm 每轮的实际 prompt token，两两之差须在 ±5% 内。

### Phase 1 —— Pilot + manipulation check

- **1 个 task**：`circle_packing`（规格最短、AlphaEvolve 有公开基线 2.635、自带 cascade stage）
- **arm**：A / B / D / **E** / F
- **seed**：3
- **规模**：5 arm × 3 seed = **15 runs**

**门槛（全部满足才进 Phase 2）**：

- E 显著劣于 A → 注入通道有效；**E ≈ A 则整个实验作废，先修注入路径**；
- D 显著优于 A → 该任务上记忆有上升空间可测；
- F ≤ A → `num_diverse_programs` 这个旋钮确实在起作用。

### Phase 2 —— 主实验

- **task**：7 个 math task
- **arm**：A / B / C / D
- **seed**：3（演化搜索本身随机，单 run 说明不了任何事）
- **规模**：7 × 4 × 3 = **84 runs**，每 run 100 iterations

### Phase 3 —— 注入率扫描

- **arm**：仅 B
- **`num_diverse_programs` ∈ {0, 2, 5, 10}**（0 即 arm F）
- 7 task × 4 级 × 3 seed = 84 runs，其中 2 级与 Phase 2 重合 → 新增约 **63 runs**

> ⚠️ 注入率提高会同时提高 prompt token。此处**不可能**同时配平 token 与扫描注入率 ——
> 这正是本阶段要测的权衡。因此 Phase 3 必须**同时报告 token 成本曲线**，
> 并明确声明它与 Phase 2 的等 token 对照是两类结论，不得混用。

### Phase 4 —— ALE 扩展（可选）

先独立验证 SkyDiscover + ale-bench 能复现历史分数，再按 Phase 2 结构扩到 10 个 ale task。

### 预算估算（**算术推演，未实测**）

Phase 1+2+3 ≈ 162 runs。按附录 C 的单 run 估算 ~17–20 min（100 iterations），
两卡 data-parallel → **约 23–27 小时 wall-clock**。这个数字建立在未实测的吞吐推算上，
Phase 0 的实测结果可能使其显著变化。

## E.6 报告什么

### E.6.1 主图 —— 照 AlphaEvolve Figure 8

x 轴 = 计算预算（iteration 数，并附一版以累计 LLM token 为 x 轴），
y 轴 = 目标 metric，**跨 7 个 task 平均**，每个 arm 一条曲线，阴影为 3 seed 的 CI。

### E.6.2 跨 task 聚合 —— 用分类率，不用分数平均

照 AlphaEvolve 的做法（E.1）。定义三档并报每档的任务比例：

- **超过**该 task 的历史最佳（EvoTrace 中该 task 所有 session 的 best）
- **达到**历史最佳（在 evaluator 容差内）
- **未达到**

### E.6.3 per-task 明细表

| 列 | 说明 |
|---|---|
| best `combined_score` | per-task，不跨 task 平均 |
| iterations-to-threshold | 阈值取该 task 的历史最佳；**标注为我们的自定义指标**，AlphaEvolve 只有定性表述 |
| best-so-far AUC | **标注为我们的自定义指标**，AlphaEvolve 未报 |
| 失败率分类 | timeout / execution_error / wrong_answer / cascade_stageN_reject / **non_improving 单列** |
| 成本 | LLM 调用数、prompt/completion token |
| 搜索健康度 | edit acceptance rate、dead-branch rate、duplicate artifact rate |

### E.6.4 必须同时出现的声明

1. 我们的 fitness **不能与 EvoTrace / AlphaEvolve 论文数字对比**（模型不同：原 trace 76% 是
   deepseek-reasoner，无 qwen3.8）；
2. thinking 开关状态（EvoTrace 实测 reasoning tokens p50 1,941 / p99 16,384，影响极大）；
3. Phase 2 是**等 token 对照**，Phase 3 是**变 token 扫描**，两类结论不得混用；
4. 检索指标（第一类，`gem_eg/metrics.py`）与 agent outcome（第二类，本附录）**分表报告**。
   数据库更快推不出 agent 更聪明；agent 变好也不能倒过来证明检索算子的系统主张。

## E.7 已知风险

| 风险 | 表现 | 处置 |
|---|---|---|
| 注入通道惰性 | E ≈ A | Phase 1 门槛拦截，先修再跑 |
| 记忆天花板过低 | D ≈ A | 该 task 出局，或整个方向重估；**先于 B 的结论判定** |
| evaluator 不复现 | 历史最优程序重跑分数不符 | 该 task 出局（Phase 0 闸门 2） |
| cascade 静默退化 | evaluator 无 `evaluate_stage1` 却配了 `cascade_evaluation: true` | Phase 0 闸门 3 |
| 泄漏 | 检索到的代码与 target session 逐字节相同 | `is_trivial_hit()` 已实现，单独计数并从 headline 剔除 |
| task 数偏小 | 7 个 math task，跨 task 结论的统计效力有限 | 如实声明；Phase 4 扩到 17 个可提升，但需先解决 ale-bench 依赖 |
