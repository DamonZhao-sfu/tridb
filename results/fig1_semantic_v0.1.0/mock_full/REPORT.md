# Figure 1 experiment: `results/fig1_semantic_v0.1.0/mock_full`

LLM `mock-llm-v1`, 10672 events, tag `mockfull`.

## Ingest (per arm)

| arm | mode | ingest p50 ms | p95 ms | mean ms | LLM calls on ingest path | pending at end | final materialize calls | total LLM calls | tokens (p+c) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eager | eager | 7.2 | 16.8 | 8.6 | 19220 | 0 | 0 | 19220 | 4250850 |
| eager_ctl | eager | 7.0 | 18.2 | 8.9 | 19220 | 0 | 0 | 19220 | 4250850 |
| onread | deferred/on_read | 5.3 | 9.8 | 5.9 | 5459 | 8222 | 13761 | 19220 | 4250850 |
| onread_stale | deferred/on_read | 5.5 | 10.0 | 6.1 | 0 | 14226 | 19220 | 19220 | 4250850 |
| batch | deferred/batch | 4.6 | 8.5 | 7.5 | 19158 | 53 | 62 | 19220 | 4250850 |

## Retrieval plans

| arm | plan | consistency | p50 ms | p95 ms | n_sem / retrieval | LLM calls | rows pulled / retrieval | early close rate | hints / retrieval | resolved on read | stale items seen |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eager | trimodal | strong | 3.6 | 9.9 | 0.00 | 0 | 1.7 | 0.00 | 1.91 | 0 | 0 |
| eager | naive | strong | 6.0 | 14.2 | 8.63 | 30502 | 8.6 | 0.00 | 1.22 | 0 | 0 |
| eager | planned_mat | strong | 5.1 | 15.0 | 3.37 | 11917 | 3.4 | 0.00 | 1.22 | 0 | 0 |
| eager | planned | strong | 4.9 | 13.7 | 2.88 | 10813 | 2.9 | 0.06 | 1.22 | 0 | 0 |
| eager_ctl | planned | strong | 5.2 | 15.8 | 2.86 | 10886 | 2.9 | 0.06 | 1.21 | 0 | 0 |
| onread | planned | strong | 11.3 | 31.9 | 2.86 | 10830 | 2.9 | 0.06 | 1.22 | 6004 | 0 |
| onread_stale | planned | stale | 7.6 | 20.2 | 2.86 | 11059 | 2.9 | 0.06 | 1.04 | 0 | 23218 |
| batch | planned | strong | 4.4 | 15.9 | 2.99 | 11142 | 3.0 | 0.06 | 1.25 | 259 | 0 |

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
| eager/trimodal | 0.6922 | 0.5614 | 0.5614 | 0.5614 |
| eager/naive | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| eager/planned_mat | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| eager/planned | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| eager_ctl/planned | 0.9524 | 0.9422 | 0.9414 | 0.9415 |
| onread/planned | 0.9513 | 0.9360 | 0.9300 | 0.9301 |
| onread_stale/planned | 0.6281 | 0.5705 | 0.5705 | 0.5357 |
| batch/planned | 0.9337 | 0.9167 | 0.9159 | 0.9160 |

Offline replay of the eager arm from the cache with the LLM disabled: lessons equal **True**, fixed_by equal **True**, LLM calls **0**.

