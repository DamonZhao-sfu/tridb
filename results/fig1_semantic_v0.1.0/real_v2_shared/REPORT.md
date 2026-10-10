# Figure 1 experiment: `results/fig1_semantic_v0.1.0/real_v2_shared`

LLM `google/gemma-4-26B-A4B-it`, 662 events, tag `realv2s`.

## Ingest (per arm)

| arm | mode | ingest p50 ms | p95 ms | mean ms | LLM calls on ingest path | pending at end | final materialize calls | total LLM calls | tokens (p+c) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eager | eager | 306.9 | 453.8 | 301.1 | 1286 | 0 | 0 | 1286 | 358163 |
| eager_ctl | eager | 8.0 | 10.8 | 7.8 | 0 | 0 | 0 | 0 | 0 |
| onread | deferred/on_read | 4.8 | 9.4 | 5.3 | 0 | 432 | 0 | 0 | 0 |
| onread_stale | deferred/on_read | 4.9 | 6.4 | 4.9 | 0 | 825 | 0 | 0 | 0 |
| batch | deferred/batch | 5.2 | 10.3 | 6.8 | 0 | 26 | 0 | 0 | 0 |

## Retrieval plans

| arm | plan | consistency | p50 ms | p95 ms | n_sem / retrieval | LLM calls | rows pulled / retrieval | early close rate | hints / retrieval | resolved on read | stale items seen |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eager | trimodal | strong | 4.2 | 5.3 | 0.00 | 0 | 2.2 | 0.00 | 2.21 | 0 | 0 |
| eager | naive | strong | 90.9 | 284.9 | 6.37 | 1882 | 6.4 | 0.00 | 1.64 | 0 | 0 |
| eager | planned_mat | strong | 7.0 | 153.7 | 3.20 | 899 | 3.2 | 0.00 | 1.63 | 0 | 0 |
| eager | planned | strong | 6.5 | 153.8 | 3.03 | 861 | 3.0 | 0.04 | 1.63 | 0 | 0 |
| eager_ctl | planned | strong | 6.1 | 7.8 | 3.04 | 3 | 3.0 | 0.04 | 1.63 | 0 | 0 |
| onread | planned | strong | 7.0 | 16.1 | 3.04 | 4 | 3.0 | 0.04 | 1.63 | 393 | 0 |
| onread_stale | planned | stale | 8.8 | 190.6 | 3.06 | 875 | 3.1 | 0.03 | 1.24 | 0 | 1541 |
| batch | planned | strong | 6.7 | 14.6 | 3.04 | 2 | 3.0 | 0.04 | 1.63 | 199 | 0 |

## Mat(S) equivalence vs eager (after final materialization)

| arm | lessons equal | lessons agree | fixed_by equal (mirror) | fixed_by equal (native AM) | AM == mirror | fixed_by Jaccard |
|---|---|---:|---|---|---|---:|
| eager | True | 1.0000 | True | True | True | 1.0000 |
| eager_ctl | True | 1.0000 | True | True | True | 1.0000 |
| onread | True | 1.0000 | True | True | True | 1.0000 |
| onread_stale | True | 1.0000 | True | True | True | 1.0000 |
| batch | True | 1.0000 | True | True | True | 1.0000 |

## Retrieved-set agreement vs eager/planned_mat (per retrieval)

| arm/plan | mean Jaccard | exact set | exact ordered | same text |
|---|---:|---:|---:|---:|
| eager/trimodal | 0.7009 | 0.5677 | 0.5677 | 0.5677 |
| eager/naive | 0.9779 | 0.9462 | 0.9462 | 0.9462 |
| eager/planned_mat | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| eager/planned | 0.9890 | 0.9723 | 0.9723 | 0.9723 |
| eager_ctl/planned | 0.9849 | 0.9677 | 0.9569 | 0.9569 |
| onread/planned | 0.9835 | 0.9631 | 0.9523 | 0.9523 |
| onread_stale/planned | 0.6553 | 0.5477 | 0.5446 | 0.4277 |
| batch/planned | 0.9852 | 0.9662 | 0.9569 | 0.9569 |

Offline replay of the eager arm from the cache with the LLM disabled: lessons equal **True**, fixed_by equal **True**, LLM calls **0**.

