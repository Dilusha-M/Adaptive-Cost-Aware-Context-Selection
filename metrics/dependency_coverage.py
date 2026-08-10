"""Dependency coverage metrics.

Measures how well a retrieved set of nodes preserves the
required dependency chain from a query's starting points.

Improved to show:
- Required vs retrieved vs missing nodes
- Coverage percentage with breakdown by type
- Highlighted missing dependencies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graph.graph_builder import DependencyGraph, GraphNode
from graph.graph_search import GraphSearch


@dataclass
class CoverageResult:
    """Result of a dependency coverage calculation.

    Attributes:
        total_required: Total number of required dependency nodes.
        retrieved_required: Number of required dependencies retrieved.
        coverage_ratio: Fraction of dependencies covered (0.0 to 1.0).
        missing_nodes: Set of required nodes that were not retrieved.
        extra_nodes: Set of retrieved nodes that were not required.
        missing_classes: Missing nodes that are classes.
        missing_functions: Missing nodes that are functions.
        missing_files: Missing nodes that are files.
    """
    total_required: int
    retrieved_required: int
    coverage_ratio: float
    missing_nodes: list[str]
    extra_nodes: list[str]
    missing_classes: list[str] = field(default_factory=list)
    missing_functions: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_required": self.total_required,
            "retrieved_required": self.retrieved_required,
            "coverage_ratio": round(self.coverage_ratio, 4),
            "missing_nodes": sorted(self.missing_nodes),
            "extra_nodes": sorted(self.extra_nodes),
            "missing_classes": sorted(self.missing_classes),
            "missing_functions": sorted(self.missing_functions),
            "missing_files": sorted(self.missing_files),
        }


class DependencyCoverage:
    """Calculates dependency coverage for retrieval results.

    Measures what fraction of required dependencies (found via
    transitive import traversal from start nodes) are included
    in the retrieved set.

    Improved to provide detailed breakdown by node type.
    """

    @classmethod
    def calculate(
        cls,
        graph: DependencyGraph,
        start_nodes: list[str],
        retrieved_nodes: set[str],
    ) -> CoverageResult:
        """Calculate dependency coverage for a retrieval result.

        Args:
            graph: The DependencyGraph.
            start_nodes: Query start node IDs.
            retrieved_nodes: Set of retrieved node IDs.

        Returns:
            CoverageResult with detailed coverage metrics.
        """
        required = GraphSearch.find_required_deps(graph, start_nodes)
        missing = required - retrieved_nodes
        extra = retrieved_nodes - required

        total_required = len(required)
        retrieved_required = len(required & retrieved_nodes)

        if total_required == 0:
            coverage = 1.0
        else:
            coverage = retrieved_required / total_required

        # Classify missing nodes by type
        missing_classes = []
        missing_functions = []
        missing_files = []

        for node_id in missing:
            node = graph.get_node_data(node_id)
            if node:
                if node.node_type == "class":
                    missing_classes.append(node_id)
                elif node.node_type == "function":
                    missing_functions.append(node_id)
                elif node.node_type == "file":
                    missing_files.append(node_id)

        return CoverageResult(
            total_required=total_required,
            retrieved_required=retrieved_required,
            coverage_ratio=coverage,
            missing_nodes=list(missing),
            extra_nodes=list(extra),
            missing_classes=missing_classes,
            missing_functions=missing_functions,
            missing_files=missing_files,
        )

    @classmethod
    def calculate_with_neighbors(
        cls,
        graph: DependencyGraph,
        start_nodes: list[str],
        retrieved_nodes: set[str],
        neighbor_depth: int = 1,
    ) -> CoverageResult:
        """Calculate dependency coverage including neighbor context.

        Expands the required set to include direct neighbors
        of start nodes (e.g., directly imported modules),
        providing a more lenient coverage metric.

        Args:
            graph: The DependencyGraph.
            start_nodes: Query start node IDs.
            retrieved_nodes: Set of retrieved node IDs.
            neighbor_depth: How many hops of neighbors to include.

        Returns:
            CoverageResult with expanded coverage metrics.
        """
        required = set(start_nodes)

        # Expand to include neighbors up to neighbor_depth
        current = set(start_nodes)
        for _ in range(neighbor_depth):
            expanded = set()
            for node in current:
                for neighbor in graph.graph.successors(node):
                    edge_data = graph.graph.get_edge_data(node, neighbor)
                    if edge_data:
                        expanded.add(neighbor)
            required |= expanded
            current = expanded

        missing = required - retrieved_nodes
        extra = retrieved_nodes - required

        total_required = len(required)
        retrieved_required = len(required & retrieved_nodes)

        if total_required == 0:
            coverage = 1.0
        else:
            coverage = retrieved_required / total_required

        return CoverageResult(
            total_required=total_required,
            retrieved_required=retrieved_required,
            coverage_ratio=coverage,
            missing_nodes=list(missing),
            extra_nodes=list(extra),
        )
