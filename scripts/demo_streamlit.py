"""Streamlit demo UI for HelpDeskAI."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpdeskai.agents import AgentConfig, SupportAgent, open_sqlite_checkpointer  # noqa: E402
from helpdeskai.agents.support_agent import IntentClassificationError  # noqa: E402
from helpdeskai.mcp_servers.client import (  # noqa: E402
    McpServerScripts,
    McpServerUrls,
    StdioMcpClient,
)
from helpdeskai.rag.llm import MissingAnthropicKeyError  # noqa: E402

DEFAULT_TOKEN = "helpdeskai-dev-token"
DEFAULT_CHECKPOINT_DB = PROJECT_ROOT / "data" / "agent_checkpoints.sqlite"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _langfuse_internal_host() -> str | None:
    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
    return host.rstrip("/") if host else None


def _normalize_langfuse_env() -> None:
    host = _langfuse_internal_host()
    if not host:
        return
    if os.environ.get("LANGFUSE_HOST"):
        host = os.environ["LANGFUSE_HOST"].rstrip("/")
    os.environ["LANGFUSE_HOST"] = host
    os.environ["LANGFUSE_BASE_URL"] = host


def _env_status(name: str) -> str:
    return "configured" if os.environ.get(name) else "missing"


def _new_thread() -> str:
    return f"streamlit-{uuid.uuid4().hex[:10]}"


def _checkpoint_db_path() -> Path:
    return Path(os.environ.get("HELPDESKAI_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB))


def _langfuse_configured() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
        and _langfuse_internal_host()
    )


def _langfuse_handler() -> Any | None:
    if not _langfuse_configured():
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except ModuleNotFoundError:
        return None
    return CallbackHandler()


def _langfuse_metadata() -> dict[str, str]:
    state = st.session_state.demo_state
    return {
        "langfuse_session_id": state["langfuse_session_id"],
        "langfuse_user_id": "streamlit_demo",
    }


def _public_langfuse_host() -> str:
    return os.environ.get("HELPDESKAI_LANGFUSE_PUBLIC_URL", "http://localhost:3000").rstrip("/")


def _langfuse_trace_url(trace_id: str) -> str | None:
    try:
        from langfuse import get_client
    except ModuleNotFoundError:
        return None
    trace_url = get_client().get_trace_url(trace_id=trace_id)
    if not trace_url:
        return None
    internal_host = _langfuse_internal_host() or ""
    if internal_host and trace_url.startswith(internal_host):
        return _public_langfuse_host() + trace_url[len(internal_host) :]
    return trace_url


def _flush_langfuse(handler: Any | None) -> None:
    flush = getattr(handler, "flush", None)
    if callable(flush):
        flush()
    if handler is None:
        return
    try:
        from langfuse import get_client
    except ModuleNotFoundError:
        return
    get_client().flush()


def _remember_langfuse_trace(handler: Any | None) -> None:
    trace_id = getattr(handler, "last_trace_id", None)
    if trace_id:
        st.session_state.demo_state["last_langfuse_trace_id"] = str(trace_id)
        st.session_state.demo_state["last_langfuse_trace_url"] = _langfuse_trace_url(
            str(trace_id)
        )


def _mcp_server_urls() -> McpServerUrls | None:
    crm_url = os.environ.get("HELPDESKAI_MCP_CRM_URL")
    knowledge_url = os.environ.get("HELPDESKAI_MCP_KNOWLEDGE_URL")
    if crm_url and knowledge_url:
        return McpServerUrls(crm=crm_url, knowledge=knowledge_url)
    return None


def _initial_state() -> dict[str, Any]:
    return {
        "messages": [],
        "last_state": None,
        "thread_id": _new_thread(),
        "langfuse_session_id": f"streamlit-{uuid.uuid4().hex}",
        "last_langfuse_trace_id": None,
        "last_langfuse_trace_url": None,
        "mcp_token": os.environ.get("HELPDESKAI_MCP_TOKEN", DEFAULT_TOKEN),
    }


def _reset_conversation() -> None:
    st.session_state.demo_state = _initial_state()


def _build_agent(*, token: str) -> tuple[SupportAgent, Any]:
    checkpoint_db = _checkpoint_db_path()
    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    checkpointer_context = open_sqlite_checkpointer(checkpoint_db)
    checkpointer = checkpointer_context.__enter__()
    mcp_client = StdioMcpClient(
        urls=_mcp_server_urls(),
        scripts=McpServerScripts(
            crm=PROJECT_ROOT / "helpdeskai" / "mcp_servers" / "crm.py",
            knowledge=PROJECT_ROOT / "helpdeskai" / "mcp_servers" / "knowledge.py",
        ),
        token=token,
        actor_id="streamlit_demo",
    )
    agent = SupportAgent.create(
        config=AgentConfig(),
        checkpointer=checkpointer,
        mcp_client=mcp_client,
    )
    return agent, checkpointer_context


def _run_agent(question: str) -> dict[str, Any]:
    state = st.session_state.demo_state
    agent, context = _build_agent(token=state["mcp_token"])
    handler = _langfuse_handler()
    try:
        return agent.ask(
            question,
            thread_id=state["thread_id"],
            callbacks=[handler] if handler else None,
            metadata=_langfuse_metadata() if handler else None,
        )
    finally:
        _remember_langfuse_trace(handler)
        _flush_langfuse(handler)
        context.__exit__(None, None, None)


def _resume_agent(approval: str) -> dict[str, Any]:
    state = st.session_state.demo_state
    agent, context = _build_agent(token=state["mcp_token"])
    handler = _langfuse_handler()
    try:
        if approval == "approved":
            return agent.approve(
                thread_id=state["thread_id"],
                callbacks=[handler] if handler else None,
                metadata=_langfuse_metadata() if handler else None,
            )
        return agent.reject(
            thread_id=state["thread_id"],
            callbacks=[handler] if handler else None,
            metadata=_langfuse_metadata() if handler else None,
        )
    finally:
        _remember_langfuse_trace(handler)
        _flush_langfuse(handler)
        context.__exit__(None, None, None)


def _append_assistant_state(output: dict[str, Any]) -> None:
    answer = _display_answer(output)
    st.session_state.demo_state["messages"].append({"role": "assistant", "content": answer})
    st.session_state.demo_state["last_state"] = output


def _display_answer(output: dict[str, Any]) -> str:
    if output.get("answer"):
        return str(output["answer"])
    if output.get("pending_action") and not output.get("action_result"):
        return "Action sensible detectee. Validez ou rejetez l'action pour continuer."
    return "Aucune reponse produite."


def _render_sidebar() -> None:
    state = st.session_state.demo_state
    with st.sidebar:
        st.header("Parametres")
        state["mcp_token"] = st.text_input("MCP token", value=state["mcp_token"], type="password")
        st.text_input("Thread", value=state["thread_id"], disabled=True)
        if st.button("Nouvelle conversation", use_container_width=True):
            _reset_conversation()
            st.rerun()

        st.divider()
        st.subheader("Observabilite")
        st.link_button("Langfuse traces", "http://localhost:3000", use_container_width=True)
        st.link_button("MLflow tracking", "http://127.0.0.1:5000", use_container_width=True)
        st.caption(f"Langfuse tracing: {'actif' if _langfuse_configured() else 'inactif'}")
        trace_id = state.get("last_langfuse_trace_id")
        if trace_id:
            trace_url = state.get("last_langfuse_trace_url")
            if trace_url:
                st.link_button("Derniere trace", trace_url, use_container_width=True)
            st.caption("Derniere trace Langfuse")
            st.code(str(trace_id), language=None)

        st.divider()
        st.subheader("Environment")
        st.code(
            "\n".join(
                [
                    f"ANTHROPIC_API_KEY: {_env_status('ANTHROPIC_API_KEY')}",
                    f"LANGFUSE_PUBLIC_KEY: {_env_status('LANGFUSE_PUBLIC_KEY')}",
                    f"LANGFUSE_SECRET_KEY: {_env_status('LANGFUSE_SECRET_KEY')}",
                    f"LANGFUSE_HOST: {os.environ.get('LANGFUSE_HOST', 'not set')}",
                    f"MLFLOW_TRACKING_URI: {os.environ.get('MLFLOW_TRACKING_URI', 'not set')}",
                ]
            )
        )


def _render_state_details(output: dict[str, Any] | None) -> None:
    if not output:
        return
    with st.expander("Details agent", expanded=True):
        path = output.get("path_taken") or []
        st.write("Chemin:", " -> ".join(path) if path else "n/a")
        st.write("Intent:", output.get("intent", "n/a"))
        st.write("Confiance:", output.get("confidence", "n/a"))
        if output.get("sources"):
            st.write("Sources:", ", ".join(output["sources"]))
        if output.get("account_context"):
            st.json(output["account_context"])
        if output.get("pending_action"):
            st.warning("Action sensible en attente de validation.")
            st.json(output["pending_action"])


def _render_approval_controls() -> None:
    last_state = st.session_state.demo_state.get("last_state")
    if not last_state:
        return
    if not last_state.get("pending_action") or last_state.get("action_result"):
        return

    st.warning("Action sensible en attente d'approbation humaine.")
    cols = st.columns(2)
    if cols[0].button("Approuver l'action", type="primary", use_container_width=True):
        try:
            output = _resume_agent("approved")
            _append_assistant_state(output)
            st.rerun()
        except Exception as exc:  # pragma: no cover - Streamlit UI guard
            st.error(f"Erreur pendant l'approbation: {exc}")
    if cols[1].button("Rejeter l'action", use_container_width=True):
        try:
            output = _resume_agent("rejected")
            _append_assistant_state(output)
            st.rerun()
        except Exception as exc:  # pragma: no cover - Streamlit UI guard
            st.error(f"Erreur pendant le rejet: {exc}")


def main() -> None:
    st.set_page_config(page_title="HelpDeskAI POC", page_icon="H", layout="wide")
    _load_dotenv()
    _normalize_langfuse_env()
    if "demo_state" not in st.session_state:
        st.session_state.demo_state = _initial_state()
    elif "langfuse_session_id" not in st.session_state.demo_state:
        st.session_state.demo_state["langfuse_session_id"] = f"streamlit-{uuid.uuid4().hex}"
    if "last_langfuse_trace_id" not in st.session_state.demo_state:
        st.session_state.demo_state["last_langfuse_trace_id"] = None
    if "last_langfuse_trace_url" not in st.session_state.demo_state:
        st.session_state.demo_state["last_langfuse_trace_url"] = None

    _render_sidebar()

    st.title("HelpDeskAI")
    st.caption("Assistant support N1 avec RAG via MCP, agent LangGraph et observabilite.")

    for message in st.session_state.demo_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Posez une question support ou demandez une action CRM...")
    if question:
        st.session_state.demo_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Execution de l'agent..."):
                try:
                    output = _run_agent(question)
                    answer = _display_answer(output)
                    st.write(answer)
                    st.session_state.demo_state["last_state"] = output
                    st.session_state.demo_state["messages"].append(
                        {"role": "assistant", "content": answer}
                    )
                except MissingAnthropicKeyError as exc:
                    st.error(f"Cle Anthropic manquante: {exc}")
                except IntentClassificationError as exc:
                    st.error(f"Classification d'intention invalide: {exc}")
                except Exception as exc:  # pragma: no cover - Streamlit UI guard
                    st.error(f"Erreur demo: {exc}")

    _render_approval_controls()
    _render_state_details(st.session_state.demo_state.get("last_state"))


if __name__ == "__main__":
    main()
