# HelpDeskAI FinOps summary

| Scenario | Baseline/month | Current POC/month | Optimized/month | Current POC cost/query | Optimized cost/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| POC (1k req/month) | $58.50 | $53.70 | $74.92 | $0.05370 | $0.07492 |
| Small scale (10k) | $247.00 | $199.00 | $141.16 | $0.01990 | $0.01412 |
| Medium scale (100k) | $1852.00 | $1372.00 | $523.65 | $0.01372 | $0.00524 |
| Large scale (1M) | $17302.00 | $12502.00 | $3748.49 | $0.01250 | $0.00375 |

## Recommendations

- [POC (1k req/month)] Full optimization costs $16.42/month more at this volume (28.1%). Keep current POC compression first; add other levers once traces justify them.
- [Small scale (10k)] Save $105.84/month (42.9%) with prompt caching, Haiku routing, compression and semantic cache. Add semantic cache once repeated support questions are visible in traces.
- [Medium scale (100k)] Save $1328.35/month (71.7%) with prompt caching, Haiku routing, compression and semantic cache. Add batch/offline evaluation and stricter model routing governance.
- [Large scale (1M)] Save $13553.51/month (78.3%) with prompt caching, Haiku routing, compression and semantic cache. Add batch/offline evaluation and stricter model routing governance.
