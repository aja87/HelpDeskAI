"""Agentic workflow package for HelpDeskAI."""

from .config import AgentsConfig

__all__ = ["AgentsConfig", "build_graph", "export_graph_core", "run_agents_core"]


def build_graph(*args, **kwargs):
	from .workflow import build_graph as _build_graph

	return _build_graph(*args, **kwargs)


def export_graph_core(*args, **kwargs):
	from .workflow import export_graph_core as _export_graph_core

	return _export_graph_core(*args, **kwargs)


def run_agents_core(*args, **kwargs):
	from .workflow import run_agents_core as _run_agents_core

	return _run_agents_core(*args, **kwargs)
