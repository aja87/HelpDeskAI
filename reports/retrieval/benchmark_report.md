# Retrieval Benchmark

Golden cases: 75

Embedding model: `BAAI/bge-m3`.

Model choice: `BAAI/bge-m3` is multilingual, aligns with the ingestion tokenizer already selected for chunking, runs locally without an API key, and is a strong retrieval-oriented BGE model on MTEB-style benchmarks.

| Mode | Recall@5 | Recall@10 | MRR | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.0 | 0.0 | 0.0 | 314.03 | 782.45 |
| sparse | 0.0 | 0.0 | 0.0 | 334.23 | 714.67 |
| hybrid | 0.0 | 0.0 | 0.0 | 671.11 | 1424.67 |
