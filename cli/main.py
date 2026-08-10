"""CLI entry point for the adaptive context retrieval system.

Provides a command-line interface to:
1. Parse a Python repository
2. Build a dependency graph
3. Run baseline BFS retrieval
4. Run adaptive budget-aware retrieval with configurable scoring
5. Compare results with comprehensive metrics
6. Generate visualizations and export context

Improvements implemented:
- Node-level context with accurate token estimation
- Strict token budget enforcement
- Configurable scoring via config.yaml
- Graph importance metrics (PageRank, Betweenness, Degree)
- Explainable retrieval with score breakdowns
- Context Compression Ratio (CCR)
- Comprehensive comparison tables
- Visualization with colored nodes
- Multiple export formats (TXT, Markdown, JSON)
- Automated evaluation
- Structured logging
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich import print as rprint

from parser.ast_parser import ASTParser
from graph.graph_builder import GraphBuilder, DependencyGraph
from graph.graph_search import GraphSearch
from retrieval.baseline_bfs import BaselineBFS
from retrieval.adaptive_budget import AdaptiveBudgetRetriever
from retrieval.scorer import ScoringConfig
from metrics.token_estimator import TokenEstimator
from metrics.context_builder import ContextBuilder
from metrics.dependency_coverage import DependencyCoverage

app = typer.Typer(
    name="adaptive-context",
    help="Adaptive Budget-Aware Context Retrieval for LLM Coding Agents",
    pretty_exceptions_enable=False,
)

console = Console()

# Logging setup
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _setup_logging(query: str, budget: int) -> None:
    """Set up structured logging for this run.

    Args:
        query: The user's query.
        budget: The token budget.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"run_{timestamp}_{budget}.log"

    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Query: %s, Budget: %d", query, budget)


@app.command()
def retrieve(
    repo_path: str = typer.Argument(..., help="Path to the Python repository"),
    query: str = typer.Argument(..., help="Developer request (e.g., 'Modify LoginService')"),
    budget: int = typer.Option(5000, "--budget", "-b", help="Token budget for adaptive retrieval"),
    max_depth: int = typer.Option(5, "--max-depth", "-d", help="Maximum BFS/traversal depth"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of top start nodes to use"),
    no_baseline: bool = typer.Option(False, "--no-baseline", help="Skip baseline BFS retrieval"),
    no_adaptive: bool = typer.Option(False, "--no-adaptive", help="Skip adaptive retrieval"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output results to JSON file"),
    export_format: str = typer.Option(
        "json", "--format", "-f",
        help="Export format for context output",
    ),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Show visualization"),
    importance: str = typer.Option(
        "pagerank", "--importance", "-i",
        help="Graph importance metric",
    ),
    tokenizer: str = typer.Option(
        "fast", "--tokenizer", "-t",
        help="Token estimation mode",
    ),
    force_entry: bool = typer.Option(
        False, "--force-entry-node",
        help="Allow entry node even if it exceeds budget",
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to scoring config YAML",
    ),
) -> None:
    """Run adaptive budget-aware context retrieval on a Python repository.

    Parses the repository, builds a dependency graph, runs both
    baseline BFS and adaptive budget-aware retrieval, then
    compares the results with metrics, visualizations, and export.
    """
    console.print(Panel.fit(
        "[bold cyan]Adaptive Budget-Aware Context Retrieval[/]\n"
        "[dim]Research Proof-of-Concept for MSc Dissertation[/]",
        border_style="cyan",
    ))

    # Setup logging
    _setup_logging(query, budget)
    console.print(f"\n[dim]Logging to: {LOG_DIR}/run_*.log[/]")

    # Step 1: Parse repository
    console.print("\n[bold]Step 1/7:[/] Parsing repository...")
    parser = ASTParser()
    parsed_repo = parser.parse_directory(repo_path)
    console.print(f"  [green]✓[/] Parsed {len(parsed_repo.files)} files")

    # Step 2: Build dependency graph
    console.print("\n[bold]Step 2/7:[/] Building dependency graph...")
    builder = GraphBuilder(parser)
    graph = builder.build_from_parsed(parsed_repo)
    console.print(f"  [green]✓[/] Graph: {graph.node_count} nodes, {graph.edge_count} edges")

    if visualize:
        _print_graph_summary(graph)

    # Step 3: Find start nodes
    console.print(f"\n[bold]Step 3/7:[/] Locating start nodes for: [yellow]{query}[/]")
    start_nodes = GraphSearch.find_start_nodes(graph, query, top_k=top_k)
    matches = GraphSearch.search(graph, query)
    for i, m in enumerate(matches[:top_k]):
        node = graph.get_node_data(m.node_id)
        node_type = node.node_type if node else "unknown"
        console.print(f"  [blue]→[/] [{node_type}] {m.node_id} (score: {m.score:.1f})")
    console.print(f"  [green]✓[/] Found {len(start_nodes)} start node(s)")

    results = {}
    baselines = {}

    # Step 4a: Baseline BFS
    if not no_baseline:
        console.print(f"\n[bold]Step 4a/7:[/] Running Baseline BFS (max_depth={max_depth})...")
        baseline = BaselineBFS(graph, max_depth=max_depth)
        baseline_result = baseline.retrieve(query, top_k=top_k)
        results["baseline"] = baseline_result

        _print_retrieval_result(baseline_result, title="Baseline BFS")
    else:
        console.print("\n[bold]Step 4a/7:[/] Baseline BFS skipped (--no-baseline)")

    # Step 4b: Adaptive budget-aware
    if not no_adaptive:
        console.print(f"\n[bold]Step 4b/7:[/] Running Adaptive (budget={budget}, importance={importance})...")
        adaptive = AdaptiveBudgetRetriever(
            graph,
            budget=budget,
            max_depth=max_depth,
            importance_metric=importance,
            force_entry_node=force_entry,
            tokenizer_mode=tokenizer,
            scoring_config=ScoringConfig(config),
        )
        adaptive_result = adaptive.retrieve(query, top_k=top_k)
        results["adaptive"] = adaptive_result

        _print_retrieval_result(adaptive_result, title="Adaptive Budget-Aware")

        # Show explainability (Improvement 7)
        _print_explainability(adaptive_result, graph)
    else:
        console.print(f"\n[bold]Step 4b/7:[/] Adaptive retrieval skipped (--no-adaptive)")

    # Step 5: Compare
    if "baseline" in results and "adaptive" in results:
        console.print(f"\n[bold cyan]══{'═' * 50}[/]")
        console.print("[bold]Step 5/7: Comparison[/]")
        _print_comparison(graph, results["baseline"], results["adaptive"], query)

        # Context Compression Ratio (Improvement 10)
        baseline_tokens = results["baseline"].token_estimate.total_tokens
        adaptive_tokens = results["adaptive"].token_estimate.total_tokens
        if baseline_tokens > 0:
            ccr = adaptive_tokens / baseline_tokens
            reduction = (1 - ccr) * 100
            console.print(f"\n[bold]Context Compression Ratio (CCR):[/] {ccr:.2f}")
            console.print(f"[bold]Token Reduction:[/]", end=" ")
            if reduction > 0:
                console.print(f"[bold green]{reduction:.1f}%[/] ({baseline_tokens} → {adaptive_tokens} tokens)")
            else:
                console.print(f"[yellow]{reduction:.1f}%[/] ({baseline_tokens} → {adaptive_tokens} tokens)")

    # Step 6: Generate visualization
    if visualize and "adaptive" in results and "baseline" in results:
        console.print(f"\n[bold cyan]══{'═' * 50}[/]")
        console.print("[bold]Step 6/7: Visualization[/]")
        _generate_visualization(
            graph,
            results["baseline"].retrieved_nodes,
            results["adaptive"].retrieved_nodes,
            start_nodes,
            query,
        )

    # Step 7: Export
    if "adaptive" in results:
        console.print(f"\n[bold cyan]══{'═' * 50}[/]")
        console.print("[bold]Step 7/7: Export[/]")
        _export_context(results["adaptive"], export_format, graph)

    # Save JSON output if requested
    if output:
        output_data = {
            "query": query,
            "repo_path": repo_path,
            "baseline": results["baseline"].to_dict() if "baseline" in results else None,
            "adaptive": results["adaptive"].to_dict() if "adaptive" in results else None,
        }
        Path(output).write_text(json.dumps(output_data, indent=2))
        console.print(f"\n[green]✓[/] Results saved to {output}")


def _print_graph_summary(graph: DependencyGraph) -> None:
    """Print a summary of the dependency graph structure."""
    tree = Tree("[bold]Dependency Graph[/]")

    for file_path in sorted(graph.get_all_files()):
        file_node = graph.get_node_data(file_path)
        if not file_node:
            continue

        file_branch = tree.add(f"[bold cyan]{file_path}[/] "
                               f"(~{file_node.estimated_token_cost} tokens, "
                               f"{file_node.char_count} chars)")

        # Find classes in this file
        classes_in_file = [
            nid for nid, nd in graph.get_all_nodes().items()
            if nd.file_path == file_path and nd.node_type == "class"
        ]
        for class_id in classes_in_file:
            class_node = graph.get_node_data(class_id)
            if class_node:
                base_info = ""
                parents = graph.get_inheritance_parents(class_id)
                if parents:
                    base_info = f" [dim]inherits {','.join(str(p) for p in parents[:3])}[/]"
                file_branch.add(f"[green]class {class_node.name}{base_info}[/]")

        # Find functions in this file
        funcs_in_file = [
            nid for nid, nd in graph.get_all_nodes().items()
            if nd.file_path == file_path and nd.node_type == "function"
        ]
        for func_id in funcs_in_file[:5]:
            func_node = graph.get_node_data(func_id)
            if func_node:
                file_branch.add(f"[blue]func {func_node.name}[/]")

    console.print(tree)


def _print_retrieval_result(result, title: str = "Retrieval Result") -> None:
    """Print a formatted retrieval result."""
    console.print(Panel.fit(
        f"[bold]{title}[/]",
        border_style="green",
    ))

    table = Table(title=f"{title} Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Retrieved Files", str(len(result.retrieved_files)))
    table.add_row("Retrieved Nodes", str(len(result.retrieved_nodes)))
    table.add_row("Traversed Order", str(len(result.traversal_order)))
    table.add_row("Total Characters", str(result.token_estimate.total_characters))
    table.add_row("Total Lines", str(result.token_estimate.line_count))

    if hasattr(result, "budget"):
        table.add_row("Token Budget", str(result.budget))
        table.add_row("Estimated Tokens", str(result.token_estimate.total_tokens))
        table.add_row("Budget Remaining", str(result.budget_remaining))

    table.add_row("Estimated Tokens", str(result.token_estimate.total_tokens))
    table.add_row("Execution Time", f"{result.execution_time_ms:.2f}ms")

    console.print(table)

    # Show skipped nodes summary
    if result.skipped_nodes:
        skipped_count = len(result.skipped_nodes)
        console.print(f"  [dim]⚠ {skipped_count} node(s) skipped[/]")


def _print_explainability(result, graph: DependencyGraph) -> None:
    """Print explainability details for adaptive retrieval.

    Shows each node's score breakdown and reason for being selected/skipped.
    """
    if not hasattr(result, "explanations") or not result.explanations:
        return

    # Retrieved nodes
    console.print("\n[bold green]── Retrieved Nodes ──[/]")
    for exp in result.explanations:
        if exp.status.startswith("retrieved"):
            reasons = ", ".join(exp.reasons) if exp.reasons else "N/A"
            console.print(f"  [green]✓[/] {exp.node_id}")
            console.print(f"      Score: {exp.total_score:.2f} | "
                         f"Keyword: {exp.keyword_relevance:.2f} | "
                         f"Dep: {exp.dependency_importance:.2f} | "
                         f"Cent: {exp.centrality_score:.2f} | "
                         f"Tok: {exp.token_score:.2f}")
            console.print(f"      Cost: {exp.token_cost} tokens | "
                         f"Budget remaining: {exp.budget_remaining}")
            console.print(f"      Reason: {reasons}")

    # Skipped nodes
    skipped_explanations = [e for e in result.explanations if e.status == "skipped"]
    if skipped_explanations:
        console.print("\n[bold red]── Skipped Nodes ──[/]")
        for exp in skipped_explanations[:10]:  # Limit display
            reason = exp.skip_reason or "N/A"
            console.print(f"  [red]✗[/] {exp.node_id} - {reason}")
            console.print(f"      Score: {exp.total_score:.2f} | "
                         f"Cost: {exp.token_cost} tokens")


def _print_comparison(graph, baseline, adaptive, query: str) -> None:
    """Print a comprehensive comparison table.

    Shows all metrics including CCR, coverage, and budget usage.
    """
    console.print(Panel.fit(
        "[bold cyan]Algorithm Comparison[/]",
        border_style="cyan",
    ))

    # Calculate coverage for both algorithms
    baseline_coverage = DependencyCoverage.calculate(
        graph, baseline.start_nodes, baseline.retrieved_nodes,
    )
    adaptive_coverage = DependencyCoverage.calculate(
        graph, adaptive.start_nodes, adaptive.retrieved_nodes,
    )

    baseline_tokens = baseline.token_estimate.total_tokens
    adaptive_tokens = adaptive.token_estimate.total_tokens

    # Compression Ratio
    ccr = adaptive_tokens / baseline_tokens if baseline_tokens > 0 else 0
    token_reduction = (1 - ccr) * 100

    table = Table("Metric", "Baseline", "Adaptive", "Delta")

    table.add_row("Retrieved Files", str(len(baseline.retrieved_files)),
                  str(len(adaptive.retrieved_files)),
                  f"{len(adaptive.retrieved_files) - len(baseline.retrieved_files):+d}")
    table.add_row("Retrieved Nodes", str(baseline.token_estimate.node_count),
                  str(adaptive.token_estimate.node_count),
                  f"{adaptive.token_estimate.node_count - baseline.token_estimate.node_count:+d}")
    table.add_row("Characters", f"{baseline.token_estimate.total_characters:,}",
                  f"{adaptive.token_estimate.total_characters:,}",
                  f"{adaptive.token_estimate.total_characters - baseline.token_estimate.total_characters:+,}")
    table.add_row("Estimated Tokens", f"{baseline_tokens:,}", f"{adaptive_tokens:,}",
                  f"{adaptive_tokens - baseline_tokens:+,}")
    table.add_row("Execution Time", f"{baseline.execution_time_ms:.2f}ms",
                  f"{adaptive.execution_time_ms:.2f}ms",
                  f"{adaptive.execution_time_ms - baseline.execution_time_ms:+.2f}ms")

    if baseline_coverage and adaptive_coverage:
        table.add_row("Dependency Coverage",
                      f"{baseline_coverage.coverage_ratio:.1%}",
                      f"{adaptive_coverage.coverage_ratio:.1%}",
                      f"{adaptive_coverage.coverage_ratio - baseline_coverage.coverage_ratio:+.1%}")
        table.add_row("Missing Nodes", str(len(baseline_coverage.missing_nodes)),
                      str(len(adaptive_coverage.missing_nodes)), "")
        table.add_row("Missing Classes", str(len(baseline_coverage.missing_classes)),
                      str(len(adaptive_coverage.missing_classes)), "")
        table.add_row("Missing Functions", str(len(baseline_coverage.missing_functions)),
                      str(len(adaptive_coverage.missing_functions)), "")

    if hasattr(adaptive, "budget") and adaptive.budget > 0:
        budget_pct = f"{adaptive.token_estimate.total_tokens / adaptive.budget * 100:.1f}%"
        table.add_row("Budget Usage", "N/A", budget_pct, "")

    # Skipped nodes count
    if hasattr(adaptive, "skipped_nodes"):
        table.add_row("Skipped Nodes", "N/A", str(len(adaptive.skipped_nodes)), "")

    # CCR
    table.add_row("CCR", "N/A", f"{ccr:.2f}", "")

    console.print(table)

    # Highlight key finding
    console.print("\n[bold]Key Finding:[/]", end=" ")
    if adaptive_tokens < baseline_tokens:
        reduction = (1 - adaptive_tokens / baseline_tokens) * 100
        console.print(
            f"[bold green]Adaptive retrieval reduced context by {reduction:.1f}%[/] "
            f"({baseline_tokens} → {adaptive_tokens} tokens)"
        )
    elif adaptive_tokens == baseline_tokens:
        console.print(
            f"[yellow]Both algorithms retrieved equivalent context[/]"
        )
    else:
        console.print(
            f"[yellow]Adaptive selected {adaptive_tokens - baseline_tokens} more tokens[/] "
            f"but with better relevance scoring"
        )


def _generate_visualization(graph, baseline_nodes, adaptive_nodes,
                            start_nodes, query: str) -> None:
    """Generate a visualization of the dependency graph.

    Colors:
    - Green: Adaptive retrieved nodes
    - Blue: Start nodes
    - Grey: Skipped/unrelated nodes
    - Red: Missing required nodes

    Saves to visualization.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        console.print("  [yellow]matplotlib not available, skipping visualization[/]")
        return

    # Build subgraph of relevant nodes
    relevant_nodes = set(adaptive_nodes) | set(start_nodes) | {start_nodes[0]} if start_nodes else set()

    # Find missing required nodes
    required = GraphSearch.find_required_deps(graph, start_nodes)
    missing = required - adaptive_nodes

    # Color each node
    node_colors = []
    for node_id in graph.graph.nodes():
        if node_id in start_nodes:
            node_colors.append("blue")
        elif node_id in adaptive_nodes:
            node_colors.append("green")
        elif node_id in missing:
            node_colors.append("red")
        else:
            node_colors.append("lightgrey")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    pos = nx.spring_layout(graph.graph, k=2, iterations=50, seed=42)

    # Draw nodes by type
    file_nodes = [n for n in graph.graph.nodes()
                  if graph.graph.nodes[n].get("node_type") == "file"]
    class_nodes = [n for n in graph.graph.nodes()
                   if graph.graph.nodes[n].get("node_type") == "class"]
    func_nodes = [n for n in graph.graph.nodes()
                  if graph.graph.nodes[n].get("node_type") == "function"]

    # Draw file nodes
    nx.draw_networkx_nodes(
        graph.graph, pos,
        nodelist=[n for n in file_nodes if n in relevant_nodes] or file_nodes,
        node_color=[node_colors[i] for i, _ in enumerate(file_nodes)],
        node_size=2000,
        node_shape="s",
        ax=ax,
        alpha=0.8,
    )

    # Draw class nodes
    nx.draw_networkx_nodes(
        graph.graph, pos,
        nodelist=class_nodes,
        node_color=[node_colors[i] for i, _ in enumerate(class_nodes)] if len(class_nodes) == len(node_colors) else ["lightgrey"] * len(class_nodes),
        node_size=1500,
        node_shape="o",
        ax=ax,
        alpha=0.8,
    )

    # Draw edges
    nx.draw_networkx_edges(
        graph.graph, pos,
        edgelist=list(graph.graph.edges()),
        width=0.8,
        alpha=0.4,
        arrows=True,
        arrowsize=15,
        ax=ax,
    )

    # Draw labels for file nodes
    file_labels = {n: n for n in file_nodes}
    nx.draw_networkx_labels(
        graph.graph, pos,
        labels=file_labels,
        font_size=8,
        font_family="monospace",
        ax=ax,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", label="Adaptive Retrieved"),
        Patch(facecolor="blue", label="Start Node"),
        Patch(facecolor="red", label="Missing Required"),
        Patch(facecolor="lightgrey", label="Skipped/Irrelevant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    ax.set_title(
        f"Adaptive Context Retrieval: '{query}'\n"
        f"Nodes: {graph.node_count} | Edges: {graph.edge_count} | "
        f"Adaptive: {len(adaptive_nodes)} retrieved, {len(missing)} missing",
        fontsize=14, fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()

    output_path = Path("visualization.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    console.print(f"  [green]✓[/] Visualization saved to {output_path}")

    # Also generate a mermaid diagram (Improvement 15)
    _generate_mermaid(graph, adaptive_nodes, start_nodes, missing)


def _generate_mermaid(graph, retrieved_nodes, start_nodes, missing_nodes) -> None:
    """Generate a Mermaid diagram of the graph structure."""
    lines = ["```mermaid", "graph TD"]

    for src, tgt, data in graph.graph.edges(data=True):
        edge_type = data.get("edge_type", "")
        # Determine styling
        src_cls = "classDef fileNode fill:#f9f9f9,stroke:#333,stroke-width:2px"
        tgt_cls = ""
        style = ""

        if tgt in retrieved_nodes:
            style = " style " + tgt + " fill:#90EE90,stroke:#2E8B57,stroke-width:2px"
        elif tgt in missing_nodes:
            style = " style " + tgt + " fill:#FFB6C6,stroke:#8B0000,stroke-width:2px"

        label = f" [{edge_type}]" if edge_type else ""
        lines.append(f"    {src} --> {tgt}{label}{style}")

    lines.append("```")

    mermaid_path = Path("architecture.mmd")
    mermaid_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [dim]Mermaid diagram saved to {mermaid_path}[/]")


def _export_context(result, format: str, graph: DependencyGraph) -> None:
    """Export retrieved context in the specified format.

    Formats:
    - json: Structured JSON with node details
    - txt: Plain text source code
    - markdown: Markdown formatted with headers
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"context_{timestamp}.{format}"

    if format == "json":
        # Include context builder output
        builder = ContextBuilder(tokenizer_mode="fast")
        context = builder.build(result.retrieved_nodes, graph)
        output = {
            "query": result.query,
            "algorithm": result.algorithm_name,
            "budget": result.budget,
            "tokens_used": result.token_estimate.total_tokens,
            "nodes_retrieved": len(result.retrieved_nodes),
            "context": context.to_dict(),
        }
        Path(filename).write_text(json.dumps(output, indent=2), encoding="utf-8")

    elif format == "txt":
        builder = ContextBuilder(tokenizer_mode="fast")
        context = builder.build(result.retrieved_nodes, graph)
        header = f"# Context Retrieved\n# Query: {result.query}\n" \
                 f"# Algorithm: {result.algorithm_name}\n" \
                 f"# Tokens: {result.token_estimate.total_tokens}\n" \
                 f"# Files: {result.token_estimate.file_count}\n\n"
        Path(filename).write_text(header + context.source, encoding="utf-8")

    elif format == "markdown":
        builder = ContextBuilder(tokenizer_mode="fast")
        context = builder.build(result.retrieved_nodes, graph)
        lines = [
            f"# Context Retrieved\n",
            f"**Query:** {result.query}\n",
            f"**Algorithm:** {result.algorithm_name}\n",
            f"**Tokens:** {result.token_estimate.total_tokens}\n",
            f"**Files:** {result.token_estimate.file_count}\n",
            f"**Nodes:** {result.token_estimate.node_count}\n\n",
            f"---\n\n",
        ]
        lines.append(context.source)
        Path(filename).write_text("".join(lines), encoding="utf-8")

    console.print(f"  [green]✓[/] Context exported to {filename} ({format})")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
