# HelpDeskAI FinOps summary

| Scenario | Baseline/month | Optimized/month | Savings | Optimized cost/query |
| --- | ---: | ---: | ---: | ---: |
| POC (1k req/month) | $58.50 | $74.92 | $-16.42 | $0.07492 |
| Small scale (10k) | $247.00 | $141.16 | $105.84 | $0.01412 |
| Medium scale (100k) | $1852.00 | $523.65 | $1328.35 | $0.00524 |
| Large scale (1M) | $17302.00 | $3748.49 | $13553.51 | $0.00375 |

## Recommendations

- [POC (1k req/month)] Save $-16.42/month (-28.1%) with prompt caching, Haiku routing, compression and semantic cache. Prioritize prompt caching and Haiku routing; semantic cache can wait.
- [Small scale (10k)] Save $105.84/month (42.9%) with prompt caching, Haiku routing, compression and semantic cache. Add semantic cache once repeated support questions are visible in traces.
- [Medium scale (100k)] Save $1328.35/month (71.7%) with prompt caching, Haiku routing, compression and semantic cache. Add batch/offline evaluation and stricter model routing governance.
- [Large scale (1M)] Save $13553.51/month (78.3%) with prompt caching, Haiku routing, compression and semantic cache. Add batch/offline evaluation and stricter model routing governance.
