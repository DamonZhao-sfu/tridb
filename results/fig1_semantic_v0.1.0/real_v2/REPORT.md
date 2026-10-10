# Figure 1 experiment: `results/fig1_semantic_v0.1.0/real_v2`

LLM `google/gemma-4-26B-A4B-it`, 662 events, tag `realv2`.

## Ingest (per arm)

| arm | mode | ingest p50 ms | p95 ms | mean ms | LLM calls on ingest path | pending at end | final materialize calls | total LLM calls | tokens (p+c) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eager | eager | 309.8 | 466.1 | 306.0 | 1286 | 0 | 0 | 1286 | 358160 |
| eager_ctl | eager | 309.3 | 462.1 | 306.0 | 1286 | 0 | 0 | 1286 | 358242 |
| onread | deferred/on_read | 5.3 | 8.9 | 5.6 | 411 | 432 | 875 | 1286 | 358153 |
| onread_stale | deferred/on_read | 4.9 | 6.4 | 4.9 | 0 | 825 | 1286 | 1286 | 358201 |
| batch | deferred/batch | 5.3 | 8.9 | 35.4 | 1243 | 26 | 43 | 1286 | 358242 |

## Retrieval plans

| arm | plan | consistency | p50 ms | p95 ms | n_sem / retrieval | LLM calls | rows pulled / retrieval | early close rate | hints / retrieval | resolved on read | stale items seen |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eager | trimodal | strong | 4.2 | 5.3 | 0.00 | 0 | 2.2 | 0.00 | 2.21 | 0 | 0 |
| eager | naive | strong | 85.8 | 269.0 | 6.37 | 1877 | 6.4 | 0.00 | 1.66 | 0 | 0 |
| eager | planned_mat | strong | 7.0 | 151.1 | 3.20 | 898 | 3.2 | 0.00 | 1.65 | 0 | 0 |
| eager | planned | strong | 6.9 | 152.9 | 3.03 | 859 | 3.0 | 0.04 | 1.65 | 0 | 0 |
| eager_ctl | planned | strong | 7.1 | 199.5 | 3.04 | 863 | 3.0 | 0.04 | 1.62 | 0 | 0 |
| onread | planned | strong | 93.9 | 614.0 | 3.04 | 862 | 3.0 | 0.04 | 1.66 | 393 | 0 |
| onread_stale | planned | stale | 7.6 | 187.4 | 3.06 | 875 | 3.1 | 0.03 | 1.20 | 0 | 1518 |
| batch | planned | strong | 70.2 | 512.3 | 3.03 | 861 | 3.0 | 0.04 | 1.66 | 198 | 0 |

## Mat(S) equivalence vs eager (after final materialization)

| arm | lessons equal | lessons agree | fixed_by equal (mirror) | fixed_by equal (native AM) | AM == mirror | fixed_by Jaccard |
|---|---|---:|---|---|---|---:|
| eager | True | 1.0000 | True | True | True | 1.0000 |
| eager_ctl | False | 0.5242 | False | False | True | 0.8951 |
| onread | False | 0.7054 | False | False | True | 0.8885 |
| onread_stale | False | 0.5378 | False | False | True | 0.8947 |
| batch | False | 0.6208 | False | False | True | 0.9057 |

## Retrieved-set agreement vs eager/planned_mat (per retrieval)

| arm/plan | mean Jaccard | exact set | exact ordered | same text |
|---|---:|---:|---:|---:|
| eager/trimodal | 0.7030 | 0.5600 | 0.5600 | 0.5600 |
| eager/naive | 0.9781 | 0.9477 | 0.9477 | 0.9477 |
| eager/planned_mat | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| eager/planned | 0.9911 | 0.9754 | 0.9754 | 0.9754 |
| eager_ctl/planned | 0.8913 | 0.8046 | 0.7954 | 0.5200 |
| onread/planned | 0.9357 | 0.8738 | 0.8646 | 0.7185 |
| onread_stale/planned | 0.6536 | 0.5354 | 0.5323 | 0.4231 |
| batch/planned | 0.9196 | 0.8385 | 0.8323 | 0.6277 |

Offline replay of the eager arm from the cache with the LLM disabled: lessons equal **True**, fixed_by equal **True**, LLM calls **0**.

