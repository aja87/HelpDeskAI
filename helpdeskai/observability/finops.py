"""FinOps cost model for HelpDeskAI LLM traffic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PRICING_USD_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
}


@dataclass(frozen=True)
class Scenario:
    """Monthly LLM usage scenario with common optimization levers."""

    name: str
    requests_per_month: int
    input_tokens: int = 4_000
    output_tokens: int = 300
    model: str = "claude-sonnet-4-6"
    cache_ratio: float = 0.0
    haiku_ratio: float = 0.0
    compression_ratio: float = 0.0
    semantic_cache_hit: float = 0.0
    infra_usd_per_month: float = 80.0
    embedding_usd_per_month: float = 2.0

    def effective_cost(self) -> dict[str, float | int]:
        """Return the effective monthly cost after optimization levers."""
        if self.model not in PRICING_USD_PER_MILLION_TOKENS:
            allowed = ", ".join(sorted(PRICING_USD_PER_MILLION_TOKENS))
            raise ValueError(f"unknown pricing model '{self.model}'. Expected one of: {allowed}")
        for field_name in (
            "cache_ratio",
            "haiku_ratio",
            "compression_ratio",
            "semantic_cache_hit",
        ):
            value = getattr(self, field_name)
            if value < 0 or value > 1:
                raise ValueError(f"{field_name} must be between 0 and 1")

        effective_calls = self.requests_per_month * (1 - self.semantic_cache_hit)
        effective_input_tokens = self.input_tokens * (1 - self.compression_ratio)
        primary_pricing = PRICING_USD_PER_MILLION_TOKENS[self.model]
        haiku_pricing = PRICING_USD_PER_MILLION_TOKENS["claude-haiku-4-5"]

        if self.haiku_ratio > 0 and self.model not in {
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
        }:
            haiku_calls = int(effective_calls * self.haiku_ratio)
            primary_calls = int(effective_calls - haiku_calls)
        else:
            haiku_calls = 0
            primary_calls = int(effective_calls)

        cache_read_multiplier = 0.1

        def token_cost(calls: int, pricing: dict[str, float]) -> float:
            input_unit = effective_input_tokens / 1_000_000
            cached_input = input_unit * self.cache_ratio * pricing["input"] * cache_read_multiplier
            normal_input = input_unit * (1 - self.cache_ratio) * pricing["input"]
            output = self.output_tokens / 1_000_000 * pricing["output"]
            return calls * (cached_input + normal_input + output)

        llm_cost = token_cost(primary_calls, primary_pricing) + token_cost(
            haiku_calls,
            haiku_pricing,
        )
        infra_cost = self.infra_usd_per_month + self.embedding_usd_per_month
        total = llm_cost + infra_cost
        return {
            "n_effective_calls": int(effective_calls),
            "llm_cost_usd": round(llm_cost, 2),
            "infra_cost_usd": round(infra_cost, 2),
            "total_usd": round(total, 2),
            "cost_per_query_usd": round(total / max(self.requests_per_month, 1), 5),
        }

    def to_row(self, *, variant: str) -> dict[str, Any]:
        """Return a CSV/Markdown friendly row."""
        return {**asdict(self), "variant": variant, **self.effective_cost()}


def make_optimized(base: Scenario) -> Scenario:
    """Apply the default optimization stack from Module 9."""
    return Scenario(
        name=f"{base.name} (optimized)",
        requests_per_month=base.requests_per_month,
        input_tokens=base.input_tokens,
        output_tokens=base.output_tokens,
        model=base.model,
        cache_ratio=0.6,
        haiku_ratio=0.7,
        compression_ratio=0.4,
        semantic_cache_hit=0.3,
        infra_usd_per_month=base.infra_usd_per_month + 30.0,
        embedding_usd_per_month=base.embedding_usd_per_month,
    )


def default_scenarios() -> list[Scenario]:
    """Return the standard Phase 7 scenarios."""
    return [
        Scenario("POC (1k req/month)", 1_000, infra_usd_per_month=40.0),
        Scenario("Small scale (10k)", 10_000),
        Scenario("Medium scale (100k)", 100_000, infra_usd_per_month=200.0),
        Scenario("Large scale (1M)", 1_000_000, infra_usd_per_month=800.0),
    ]


def recommend(scenarios: list[Scenario]) -> list[str]:
    """Return pragmatic FinOps recommendations for each scenario."""
    recommendations: list[str] = []
    for base in scenarios:
        base_cost = base.effective_cost()
        optimized_cost = make_optimized(base).effective_cost()
        savings = float(base_cost["total_usd"]) - float(optimized_cost["total_usd"])
        savings_pct = savings / float(base_cost["total_usd"]) * 100 if base_cost["total_usd"] else 0
        prefix = (
            f"[{base.name}] Save ${savings:.2f}/month ({savings_pct:.1f}%) with "
            "prompt caching, Haiku routing, compression and semantic cache. "
        )
        if base.requests_per_month < 5_000:
            advice = "Prioritize prompt caching and Haiku routing; semantic cache can wait."
        elif base.requests_per_month < 50_000:
            advice = "Add semantic cache once repeated support questions are visible in traces."
        else:
            advice = "Add batch/offline evaluation and stricter model routing governance."
        recommendations.append(prefix + advice)
    return recommendations
