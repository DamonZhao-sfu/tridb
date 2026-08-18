# E0 数据准备与复现说明 v0.1.0

## Material Passport

- 实验阶段：E0 数据准备（尚未执行计划延迟测量）
- 数据：OpenEvolve v0.3.2 小规模演化轨迹、STARK-PRIME pinned revision
- OpenEvolve 模型：本地 `Qwen/Qwen3-32B`，`http://127.0.0.1:8000/v1`
- 统一 embedding：本地 `Qwen/Qwen3-Embedding-0.6B`，1024 维，`http://127.0.0.1:8001/v1`
- 大文件位置：`data/e0/`（被 `.gitignore` 排除）
- 可提交内容：本文件、`tools/e0/`、`configs/e0/`、`experiments/e0/`、测试和 Makefile

本管线实现 [trevillisPlan.md](../haikaidocs/trevillisPlan.md) 中 E0 的数据准备部分。
它只生成/校验数据，不运行 TriDB 延迟 benchmark，也不声称完成 GX10 sign-off。

## 1. 环境

E0 使用独立 Python 3.11 环境，避免 `stark-qa` 的旧 RDKit/pytdc 依赖污染主环境。
`numpy==1.26.4` 与 `setuptools==80.9.0` 是兼容性硬 pin：NumPy 2.x 无法加载
STARK 依赖的 RDKit wheel，而 setuptools 81+ 已移除 pytdc 使用的 `pkg_resources`。
`requirements-e0.txt` 记录直接依赖，`requirements-e0.lock` 固定完整 transitive closure。

```bash
make e0-setup
```

本地模型服务必须分别返回且只返回以下 served model ID：

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8001/v1/models
```

不要并发运行 OpenEvolve 生成与全量 embedding：即使它们暴露为两个端口，也可能共享
同一组 GPU 调度资源。固定顺序为“OpenEvolve 轨迹 → STARK/OpenEvolve embedding”。
生成模型 artifact revision 固定为
`aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`，embedding revision 固定为
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`；preflight/manifest 会同时记录 served ID、
artifact 名与本地可见 snapshot revision。

## 2. STARK-PRIME

下载被固定到 Hugging Face 数据集 `snap-stanford/stark` revision
`88269e23e90587f99476c5dd74e235a0877e69be`；processed zip 的 SHA256 为
`f6f265f60b761784fb7c2f052359f9cd3bf9ea20c4a307865f62ce173b294dfc`。

```bash
make e0-stark-prime-prepare
make e0-stark-prime-embed
```

归一化结果：

```text
data/e0/stark_prime/
├── raw/prime/                    # pinned official files + source_receipt.json
└── normalized/
    ├── nodes.parquet             # 129,375 nodes; text and relational attributes
    ├── edges.parquet             # 8,100,498 typed adjacency arcs
    ├── queries_v0.1.jsonl        # 10 manually audited pilot queries
    ├── embeddings.parquet        # checkpointed 1024-d node vectors
    ├── embeddings.manifest.json
    └── manifest.json
```

官方 `stark-qa` 默认把每条 PrimeKG 关系转换成两个方向的 adjacency arc；Parquet 保留
这一语义，后续必须加载到 TriDB native graph AM，不能用 relational edge join table。

10 条 query annotation 冻结在
`configs/e0/stark_prime_pilot_annotations.jsonl`。每条记录保存原始 qid、人工核对的
anchor、typed path、hop limit、结构化谓词和官方 answer IDs。answer IDs 只用于 oracle/
quality，不得注入在线执行器。

## 3. OpenEvolve 小规模轨迹

配置为 seed 42、32 iterations、2 islands、population 256、archive 64、单 evaluator。
内置 evolution trace 使用 JSONL，保存 code 和 parent/child metrics，不保存 prompt。
LLM client 使用单次 600 秒 timeout 且 `retries=0`；OpenEvolve v0.3.2 的同步 OpenAI
调用在 asyncio timeout 后不会终止服务端请求，开启双层 retry 会制造重复生成。每次 run
另存一份 `raw/seed_42/run_config.yaml`，该文件而不是之后可能更新的 tracked config 才是
对应原始 trace 的配置证据。

本次已完成的 seed 42 run 实际结果为 32 次 iteration、30 条成功 transition 和 2 次
`No valid diffs`（失败只保留在日志，不伪造成 transition）。归一化后有 31 个节点、30 条边、
10 条 exact-oracle query；lineage 为单 root DAG，8 个 branching node，最大深度 6。31 个节点
均已生成 1024 维 embedding，最终 `ready_for_e0=True`。本次实际 raw config 的 SHA256 为
`9c1bb04a374a1bfb20d6767624c2060bef6acb714edde4f36b86d33a57cf3b39`；它采用启动时的
`timeout=180, retries=2, max_tokens=4096`。tracked config 随后修正为
`timeout=600, retries=0, max_tokens=2048`，只影响未来 run，不追溯改写本次结果。

```bash
make e0-openevolve-smoke
make e0-openevolve-embed
```

`e0-openevolve-smoke` 在 trace 已存在时会失败，避免 OpenEvolve 的 append 行为破坏一次
run 的边界。要重跑，应先把整个 `data/e0/openevolve/raw/seed_42/` 移到新的归档目录，
不要只删除 trace。

归一化结果：

```text
data/e0/openevolve/
├── raw/seed_42/
│   ├── evolution_trace.jsonl
│   ├── model_receipt.json
│   ├── run_config.yaml
│   ├── program_db/
│   ├── logs/
│   └── run/checkpoints/
└── normalized/seed_42/
    ├── nodes.parquet
    ├── edges.parquet
    ├── queries.jsonl
    ├── embeddings.parquet
    ├── embeddings.manifest.json
    └── manifest.json
```

完整性 gate 要求：父子引用闭合、lineage 无环、所有节点有 code/数值 metrics、至少一个
branch、最大深度至少 3、能够生成 10 条 exact-oracle 查询。如果 32 iterations 未通过
branch/depth gate，保留该 run 作为失败证据，再使用新目录运行 128 iterations；不能篡改
原 trace。

## 4. 最终验证

```bash
make e0-data-verify
```

该命令校验 normalization gates、embedding 行数、模型 ID、1024 维度和文件 SHA256，
并写出 `data/e0/preparation_manifest.json`。只有输出 `ready_for_e0=True` 才能进入数据加载
和 E0 plan-space measurement。

## 5. 测试

```bash
.venv-e0/bin/python -m pytest tests/test_e0_data_prepare.py -q
.venv-e0/bin/python -m ruff check tools/e0 experiments/e0 tests/test_e0_data_prepare.py
.venv-e0/bin/python -m ruff format --check tools/e0 experiments/e0 tests/test_e0_data_prepare.py
```

本阶段不包含 `traverse-first` 执行器，也不产生 E0 latency/plan-spread 结果。当前系统层
pilot 仍只应比较已经存在的 `vector-first` 和 `filter-first`；第三种 plan shape 要等原生
streaming traverse-first 落地后再加入。
