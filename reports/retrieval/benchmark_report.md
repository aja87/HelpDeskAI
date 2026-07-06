# Retrieval Benchmark

Retrieval-eligible golden cases: 75
Aligned cases used for metrics: 75
Golden unique documents: 75
Indexed unique documents: 4997
Missing golden documents from indexed corpus: 0

Embedding model: `BAAI/bge-m3`.

| Mode | Recall@5 | Recall@10 | MRR | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.64 | 0.68 | 0.6042 | 139.72 | 294.73 |
| sparse | 0.6 | 0.6533 | 0.5871 | 9.74 | 18.55 |
| hybrid | 0.6533 | 0.7067 | 0.6066 | 146.62 | 291.06 |
