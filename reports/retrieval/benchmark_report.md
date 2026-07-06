# Retrieval Benchmark

Retrieval-eligible golden cases: 75
Aligned cases used for metrics: 75
Golden unique documents: 75
Indexed unique documents: 4997
Missing golden documents from indexed corpus: 0

Embedding model: `BAAI/bge-m3`.

| Mode | Recall@5 | Recall@10 | MRR | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.3867 | 0.4267 | 0.3556 | 187.74 | 331.44 |
| sparse | 0.4133 | 0.4267 | 0.4067 | 113.48 | 236.4 |
| hybrid | 0.4133 | 0.4533 | 0.4005 | 282.0 | 509.69 |
