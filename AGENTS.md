# HelpDeskAI Agent Instructions

## Mission
Build a defendable enterprise-grade N1 support assistant with:
- RAG over technical knowledge corpus
- Agent orchestration with tool use through MCP
- End-to-end observability and evaluation
- Cost and quality controls suitable for production constraints

## Project Priorities
1. Reproducibility first
2. Evaluation before optimization claims
3. Safe behavior before feature breadth
4. Clear architecture and traceability

## Working Mode
- Implement directly when the request is actionable.
- Keep changes minimal and focused on the user ask.
- Preserve existing style and public APIs unless change is requested.
- Prefer deterministic pipelines and explicit configuration over implicit defaults.

## Technical Baseline
- Python version: 3.11 to 3.12
- Package manager/runtime: project standard from pyproject
- Data exchange formats:
  - Raw: jsonl or parquet in data/raw
  - Normalized: parquet or jsonl with explicit schema and metadata
- Quality gates:
  - Type-safe code where practical
  - Lint-clean and testable modules
  - No hidden magic constants without explanation

## Architecture Alignment
Keep implementation aligned to repository modules:
- helpdeskai/ingestion: extraction, normalization, chunking, data quality reports
- helpdeskai/retrieval: indexing and dense sparse hybrid search
- helpdeskai/rag: query rewriting, rerank, compression, generation
- helpdeskai/agents: LangGraph orchestration with confidence-based branching
- helpdeskai/mcp_servers: CRM and knowledge tool servers with validation and auth
- helpdeskai/observability: MLflow and Langfuse instrumentation

## Data And Ingestion Rules
- Keep a strict document schema and include source metadata.
- Always preserve provenance fields such as source dataset and source ids.
- Deduplicate deterministically.
- Chunking must be benchmarked on a fixed sample before final choice.
- Record pipeline parameters and checksums for reproducibility.

## Retrieval And RAG Rules
- Retrieval mode must be explicit: dense, sparse, or hybrid.
- Support metadata filters by product, version, and date.
- Evaluate with recall at k, MRR, and latency p95 on golden dataset.
- RAG changes must be validated with faithfulness and relevancy metrics.

## Agent And MCP Rules
- Model the support workflow as explicit graph states.
- Add clarification path for ambiguous requests.
- Require human approval for sensitive actions.
- Enforce session budgets for steps and token usage.
- MCP tools must validate inputs and reject unsafe payloads.

## Observability And FinOps Rules
- Log evaluation runs with parameters, metrics, and artifacts.
- Trace conversation and tool calls for debuggability.
- Never claim cost or quality improvements without measured evidence.

## Testing Expectations
- Add or update tests for every behavior change.
- Prefer small deterministic fixtures.
- Include at least one failure-path test for new logic.
- If full test suite cannot run, explain what was executed and what remains.

## Communication Style For This Repository
- Be concise and concrete.
- Explain why a design decision is chosen.
- Surface trade-offs and risks early.
- If assumptions are needed, state them explicitly.

## Definition Of Done For Any Change
1. Code updated and consistent with module boundaries
2. Basic validation run and reported
3. Risks and limitations clearly documented
4. Next steps proposed only when they are truly useful
