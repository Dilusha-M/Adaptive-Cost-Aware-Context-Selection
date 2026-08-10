"""Automated evaluation harness for the adaptive retrieval algorithm.

Runs both baseline BFS and adaptive retrieval across multiple budgets,
collects metrics, and generates:
- CSV summary table
- Charts (matplotlib)
- Comprehensive analysis report

This demonstrates the research hypothesis:
  Adaptive retrieval reduces context size while maintaining
  dependency coverage.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from parser.ast_parser import ASTParser
from graph.graph_builder import GraphBuilder
from retrieval.baseline_bfs import BaselineBFS
from retrieval.adaptive_budget import AdaptiveBudgetRetriever
from retrieval.scorer import ScoringConfig
from metrics.token_estimator import TokenEstimator
from metrics.context_builder import ContextBuilder
from metrics.dependency_coverage import DependencyCoverage


@dataclass
class EvaluationResult:
    """Result for a single budget point."""
    budget: int
    baseline_tokens: int
    adaptive_tokens: int
    adaptive_nodes: int
    baseline_nodes: int
    adaptive_files: int
    baseline_files: int
    coverage: float
    baseline_coverage: float
    execution_time_ms: float
    ccr: float  # Context Compression Ratio
    token_reduction_pct: float
    skipped_nodes: int
    extra_nodes: int


@dataclass
class EvaluationRun:
    """Complete evaluation run across all budgets."""
    repo_path: str
    query: str
    results: list[EvaluationResult] = field(default_factory=list)
    max_depth: int = 5
    importance_metric: str = "pagerank"
    tokenizer_mode: str = "fast"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repo_path": self.repo_path,
            "query": self.query,
            "max_depth": self.max_depth,
            "importance_metric": self.importance_metric,
            "tokenizer_mode": self.tokenizer_mode,
            "results": [r.__dict__ for r in self.results],
        }


def evaluate(
    repo_path: str,
    query: str,
    budgets: list[int] | None = None,
    max_depth: int = 5,
    top_k: int = 3,
    importance_metric: str = "pagerank",
    tokenizer_mode: str = "fast",
    output_dir: str = "evaluation_output",
) -> EvaluationRun:
    """Run automated evaluation across multiple budgets.

    Args:
        repo_path: Path to the Python repository.
        query: Evaluation query string.
        budgets: List of token budgets to evaluate.
            Defaults to [100, 200, 500, 1000, 2000, 5000, 10000].
        max_depth: Maximum BFS depth.
        top_k: Number of start nodes.
        importance_metric: Graph importance metric.
        tokenizer_mode: Token estimation mode.
        output_dir: Directory for output files.

    Returns:
        EvaluationRun with all results.
    """
    if budgets is None:
        budgets = [100, 200, 500, 1000, 2000, 5000, 10000]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse and build graph (once)
    parser = ASTParser()
    parsed_repo = parser.parse_directory(repo_path)
    builder = GraphBuilder(parser)
    graph = builder.build_from_parsed(parsed_repo)

    # Run baseline (same for all budgets)
    print(f"Running Baseline BFS (depth={max_depth})...")
    baseline = BaselineBFS(graph, max_depth=max_depth)
    baseline_result = baseline.retrieve(query, top_k=top_k)
    baseline_tokens = baseline_result.token_estimate.total_tokens

    print(f"Baseline tokens: {baseline_tokens}")

    # Evaluate across budgets
    eval_results: list[EvaluationResult] = []

    for budget in sorted(budgets):
        print(f"\n  Budget={budget}...", end=" ", flush=True)
        start_time = time.perf_counter()

        adaptive = AdaptiveBudgetRetriever(
            graph,
            budget=budget,
            max_depth=max_depth,
            importance_metric=importance_metric,
            tokenizer_mode=tokenizer_mode,
        )
        result = adaptive.retrieve(query, top_k=top_k)

        exec_time = (time.perf_counter() - start_time) * 1000

        # Coverage
        coverage = DependencyCoverage.calculate(
            graph, result.start_nodes, result.retrieved_nodes
        )

        ccr = result.token_estimate.total_tokens / baseline_tokens if baseline_tokens > 0 else 0
        token_reduction = (1 - ccr) * 100

        eval_result = EvaluationResult(
            budget=budget,
            baseline_tokens=baseline_tokens,
            adaptive_tokens=result.token_estimate.total_tokens,
            adaptive_nodes=len(result.retrieved_nodes),
            baseline_nodes=baseline_result.token_estimate.node_count,
            adaptive_files=len(result.retrieved_files),
            baseline_files=len(baseline_result.retrieved_files),
            coverage=coverage.coverage_ratio,
            baseline_coverage=coverage.coverage_ratio,
            execution_time_ms=exec_time,
            ccr=ccr,
            token_reduction_pct=token_reduction,
            skipped_nodes=len(result.skipped_nodes),
            extra_nodes=len(result.retrieved_nodes - set(
                graph.get_all_nodes().keys()
            )),
        )

        eval_results.append(eval_result)
        print(f"tokens={result.token_estimate.total_tokens}, "
              f"nodes={len(result.retrieved_nodes)}, "
              f"coverage={coverage.coverage_ratio:.1%}")

    run = EvaluationRun(
        repo_path=repo_path,
        query=query,
        results=eval_results,
        max_depth=max_depth,
        importance_metric=importance_metric,
        tokenizer_mode=tokenizer_mode,
    )

    # Generate outputs
    _export_csv(run, output_path / "evaluation.csv")
    _generate_charts(run, output_path)
    _print_summary(run)

    print(f"\n[Evaluation complete. Results saved to {output_path}/]")
    return run


def _export_csv(run: EvaluationRun, path: Path) -> None:
    """Export evaluation results to CSV."""
    fieldnames = [
        "budget", "baseline_tokens", "adaptive_tokens",
        "baseline_nodes", "adaptive_nodes",
        "baseline_files", "adaptive_files",
        "coverage", "baseline_coverage",
        "execution_time_ms", "ccr", "token_reduction_pct",
        "skipped_nodes", "extra_nodes",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in run.results:
            writer.writerow(r.__dict__)


def _generate_charts(run: EvaluationRun, output_path: Path) -> None:
    """Generate evaluation charts."""
    if not run.results:
        return

    budgets = [r.budget for r in run.results]
    adaptive_tokens = [r.adaptive_tokens for r in run.results]
    baseline_tokens = [r.baseline_tokens for r in run.results]
    coverage = [r.coverage for r in run.results]
    ccr = [r.ccr for r in run.results]
    skipped = [r.skipped_nodes for r in run.results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Chart 1: Tokens vs Budget
    ax1 = axes[0, 0]
    ax1.plot(budgets, baseline_tokens, "o-", label="Baseline (constant)", linewidth=2)
    ax1.plot(budgets, adaptive_tokens, "s-", label="Adaptive", linewidth=2)
    ax1.set_xlabel("Token Budget")
    ax1.set_ylabel("Tokens")
    ax1.set_title("Token Budget vs Actual Tokens")
    ax1.legend()
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # Chart 2: Coverage vs Budget
    ax2 = axes[0, 1]
    ax2.plot(budgets, coverage, "o-", color="green", linewidth=2)
    ax2.set_xlabel("Token Budget")
    ax2.set_ylabel("Coverage")
    ax2.set_title("Dependency Coverage vs Budget")
    ax2.set_ylim(0, 1.05)

    # Chart 3: CCR vs Budget
    ax3 = axes[1, 0]
    ax3.plot(budgets, ccr, "s-", color="orange", linewidth=2)
    ax3.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Baseline (CCR=1.0)")
    ax3.set_xlabel("Token Budget")
    ax3.set_ylabel("Context Compression Ratio (CCR)")
    ax3.set_title("Context Compression Ratio vs Budget")
    ax3.legend()
    ax3.set_ylim(0, 1.1)

    # Chart 4: Skipped nodes vs Budget
    ax4 = axes[1, 1]
    ax4.bar(budgets, skipped, color="steelblue", alpha=0.7)
    ax4.set_xlabel("Token Budget")
    ax4.set_ylabel("Skipped Nodes")
    ax4.set_title("Nodes Skipped Due to Budget")

    plt.suptitle(
        f"Evaluation: {run.query}\n"
        f"Repo: {run.repo_path} | Depth: {run.max_depth} | "
        f"Metric: {run.importance_metric}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    chart_path = output_path / "evaluation_charts.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_summary(run: EvaluationRun) -> None:
    """Print a summary table of evaluation results."""
    if not run.results:
        print("No results to display.")
        return

    print("\n" + "=" * 90)
    print(f"{'Budget':>10} {'Baseline':>10} {'Adaptive':>10} {'Tokens':>10} "
          f"{'Nodes':>8} {'Coverage':>10} {'CCR':>8} {'Skipped':>8}")
    print("-" * 90)

    for r in run.results:
        print(f"{r.budget:>10,} {r.baseline_tokens:>10,} {r.adaptive_tokens:>10,} "
              f"{r.token_reduction_pct:>9.1f}% {r.adaptive_nodes:>8} "
              f"{r.coverage:>10.1%} {r.ccr:>8.2f} {r.skipped_nodes:>8}")

    print("=" * 90)

    # Best CCR
    best = min(run.results, key=lambda r: r.ccr)
    print(f"\nBest CCR: {best.ccr:.2f} at budget={best.budget} "
          f"({best.token_reduction_pct:.1f}% reduction)")

    # Best coverage at lowest budget
    covered = [r for r in run.results if r.coverage > 0]
    if covered:
        best_cov = max(covered, key=lambda r: r.coverage)
        print(f"Best coverage: {best_cov.coverage:.1%} at budget={best_cov.budget}")
