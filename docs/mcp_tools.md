# Phase 6 MCP Tools

HelpDeskAI exposes two FastMCP servers for the Phase 6 mock SI integration.
Both servers use stdio by default, require a shared `token` argument, validate
tool inputs with Pydantic, apply a per-actor sliding-window rate limit, and
write JSON audit logs to `data/audit/mcp.jsonl` unless
`HELPDESKAI_MCP_AUDIT_PATH` is set.

Default POC token: `helpdeskai-dev-token`.

## CRM Server

Module: `helpdeskai.mcp_servers.crm`

Tools:

- `get_customer(customer_id, token, actor_id="agent_default")`
  Returns mock CRM identity data. `customer_id` must match `cust_xxx`.
- `get_subscription_status(customer_id, token, actor_id="agent_default")`
  Returns status, plan, seat usage and renewal date.
- `list_recent_tickets(customer_id, token, limit=5, actor_id="agent_default")`
  Returns recent tickets for a customer. `limit` is 1 to 20.
- `create_ticket(customer_id, subject, body, token, priority="medium", actor_id="agent_default")`
  Creates a mock support ticket. Priority must be `low`, `medium`, `high` or
  `urgent`; subject is 5 to 120 chars; body is 10 to 2000 chars.

## Knowledge Server

Module: `helpdeskai.mcp_servers.knowledge`

Tool:

- `search_knowledge(query, token, top_k=5, product=None, version=None, tenant=None, actor_id="agent_default")`
  Searches the NovaCloud knowledge base and returns source IDs, document IDs,
  snippets, scores and metadata. `query` is 3 to 500 chars and `top_k` is 1 to
  10.

## Local Commands

List stdio tools:

```powershell
python scripts/list_mcp_tools.py
```

Run the CRM server in stdio mode:

```powershell
python -m helpdeskai.mcp_servers.crm
```

Run the demo:

```powershell
python scripts/demo_agent_mcp.py
```

