# Figure 1 experiment: `results/fig1_semantic_v0.1.0/real_slice`

LLM `google/gemma-4-26B-A4B-it`, 662 events, tag `real1`.

## Ingest (per arm)

| arm | mode | ingest p50 ms | p95 ms | mean ms | LLM calls on ingest path | pending at end | final materialize calls | total LLM calls | tokens (p+c) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eager | eager | 323.8 | 496.5 | 323.7 | 1286 | 0 | 0 | 1286 | 358245 |
| eager_ctl | eager | 319.9 | 492.3 | 320.8 | 1286 | 0 | 0 | 1286 | 358223 |
| onread | deferred/on_read | 5.4 | 8.8 | 5.7 | 411 | 433 | 875 | 1286 | 358153 |
| onread_stale | deferred/on_read | 4.9 | 6.6 | 4.9 | 0 | 825 | 1286 | 1286 | 358247 |
| batch | deferred/batch | 5.5 | 9.1 | 35.8 | 1243 | 26 | 43 | 1286 | 358219 |

## Retrieval plans

| arm | plan | consistency | p50 ms | p95 ms | n_sem / retrieval | LLM calls | rows pulled / retrieval | early close rate | hints / retrieval | resolved on read | stale items seen |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eager | trimodal | strong | 4.0 | 5.1 | 0.00 | 0 | 2.2 | 0.00 | 2.21 | 0 | 0 |
| eager | naive | strong | 98.7 | 314.8 | 6.37 | 1879 | 6.4 | 0.00 | 1.69 | 0 | 0 |
| eager | planned_mat | strong | 7.1 | 157.5 | 3.20 | 900 | 3.2 | 0.00 | 1.68 | 0 | 0 |
| eager | planned | strong | 7.7 | 156.5 | 3.04 | 862 | 3.0 | 0.04 | 1.68 | 0 | 0 |
| eager_ctl | planned | strong | 7.9 | 194.4 | 3.05 | 863 | 3.0 | 0.04 | 1.65 | 0 | 0 |
| onread | planned | strong | 96.8 | 617.6 | 3.04 | 858 | 3.0 | 0.04 | 1.65 | 392 | 0 |
| onread_stale | planned | stale | 7.8 | 191.1 | 3.06 | 876 | 3.1 | 0.03 | 1.21 | 0 | 1525 |
| batch | planned | strong | 69.5 | 515.7 | 3.04 | 860 | 3.0 | 0.04 | 1.62 | 198 | 0 |

## Mat(S) equivalence vs eager (after final materialization)

| arm | lessons equal | lessons agree | fixed_by equal | fixed_by Jaccard |
|---|---|---:|---|---:|
| eager | True | 1.0000 | True | 1.0000 |
| eager_ctl | False | 0.7719 | False | 0.9094 |
| onread | False | 0.4894 | False | 0.8906 |
| onread_stale | False | 0.5000 | False | 0.8906 |
| batch | False | 0.4819 | False | 0.8910 |

## Retrieved-set agreement vs eager/planned_mat (per retrieval)

| arm/plan | mean Jaccard | exact-set rate |
|---|---:|---:|
| eager/trimodal | 0.7197 | 0.5646 |
| eager/naive | 0.9810 | 0.9569 |
| eager/planned_mat | 1.0000 | 1.0000 |
| eager/planned | 0.9934 | 0.9785 |
| eager_ctl/planned | 0.9326 | 0.8738 |
| onread/planned | 0.8740 | 0.7908 |
| onread_stale/planned | 0.6503 | 0.5323 |
| batch/planned | 0.8733 | 0.7862 |

Offline replay of the eager arm from the cache with the LLM disabled: lessons equal **True**, fixed_by equal **True**, LLM calls **0**.

