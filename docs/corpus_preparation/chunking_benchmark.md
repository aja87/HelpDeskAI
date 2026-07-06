# Chunking benchmark

Deterministic comparison on 50 TechQA documents


| Strategy | Chunks | Mean tokens | Median | Min | Max | Duplicates | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 184 | 321.7 | 384.0 | 12 | 384 | 0 | 1.2461 |
| recursive | 173 | 317.23 | 364 | 61 | 384 | 0 | 1.1796 |
| semantic | 206 | 247.54 | 211.0 | 3 | 512 | 7 | 752.6653 |