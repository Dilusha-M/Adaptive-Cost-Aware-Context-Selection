"""Token estimation utilities for graph nodes.

Provides functions to estimate the LLM token cost of including
graph nodes in the context window.

Supports two modes (Improvement 3):
- Fast: characters / 4 (default)
- Accurate: tiktoken-based estimation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graph.graph_builder import DependencyGraph, GraphNode


@dataclass
class TokenEstimate:
    """Estimate of token cost for a set of retrieved nodes.

    Attributes:
        total_tokens: Total estimated token count.
        total_characters: Total character count across all nodes.
        file_count: Number of unique files represented.
        node_count: Number of graph nodes represented.
        line_count: Total number of lines in assembled context.
        per_node: Breakdown of token cost per node.
    """
    total_tokens: int
    total_characters: int
    file_count: int
    node_count: int
    line_count: int = 0
    per_node: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_tokens": self.total_tokens,
            "total_characters": self.total_characters,
            "file_count": self.file_count,
            "node_count": self.node_count,
            "line_count": self.line_count,
            "per_node": dict(sorted(self.per_node.items())),
        }


class TokenEstimator:
    """Estimates LLM token costs for graph node sets.

    Supports two modes:
    - Fast (default): tokens ~ characters / 4
    - Accurate: tiktoken-based estimation (if available)
    """

    # Approximate tokens per character for Python code
    TOKENS_PER_CHARACTER = 4.0

    @classmethod
    def estimate_node(cls, node: GraphNode) -> int:
        """Estimate token cost for a single node.

        Uses the node's pre-computed estimated_token_cost if available.
        For class/function nodes, this is based on the source snippet.
        For file nodes, this is the full file size.

        Args:
            node: The graph node to estimate.

        Returns:
            Estimated token count.
        """
        return node.estimated_token_cost

    @classmethod
    def estimate_nodes(
        cls,
        node_ids: set[str],
        graph: DependencyGraph,
        tokenizer_mode: str = "fast",
    ) -> TokenEstimate:
        """Estimate token cost for a set of nodes.

        Uses NODE-LEVEL token estimation: each class or function
        node contributes only its own source code size, not the
        entire file. If the file itself is retrieved, the full
        file size is used.

        Args:
            node_ids: Set of node IDs to estimate.
            graph: The DependencyGraph containing the nodes.
            tokenizer_mode: 'fast' (chars/4) or 'accurate' (tiktoken).

        Returns:
            TokenEstimate with cost breakdown.
        """
        per_node: dict[str, int] = {}
        files_retrieved: set[str] = set()
        sub_nodes_by_file: dict[str, list[GraphNode]] = {}
        total_tokens = 0
        total_chars = 0
        total_lines = 0

        # Separate file nodes from sub-nodes
        file_nodes: list[GraphNode] = []
        sub_nodes: list[GraphNode] = []

        for node_id in sorted(node_ids):
            node = graph.get_node_data(node_id)
            if not node:
                per_node[node_id] = 0
                continue

            if node.node_type == "file":
                file_nodes.append(node)
                files_retrieved.add(node.file_path)
            else:
                sub_nodes.append(node)
                fp = node.file_path
                if fp not in sub_nodes_by_file:
                    sub_nodes_by_file[fp] = []
                sub_nodes_by_file[fp].append(node)

        # Count file nodes
        for node in file_nodes:
            file_nodes_count = max(1, node.char_count // cls.TOKENS_PER_CHARACTER)
            per_node[node.node_id] = file_nodes_count
            total_tokens += file_nodes_count
            total_chars += node.char_count
            total_lines += node.char_count // cls.TOKENS_PER_CHARACTER

        # Count sub-nodes (classes/functions) - NOT double-counting
        # Sub-nodes from files that are fully retrieved are skipped
        for node in sub_nodes:
            fp = node.file_path
            # If the file itself is retrieved, skip sub-node
            if fp in files_retrieved:
                per_node[node.node_id] = 0
                continue

            # Estimate tokens from the node's source snippet
            node_tokens = max(1, node.char_count // cls.TOKENS_PER_CHARACTER)
            per_node[node.node_id] = node_tokens
            total_tokens += node_tokens
            total_chars += node.char_count
            total_lines += node.char_count // cls.TOKENS_PER_CHARACTER

        # Count unique files
        unique_files = set()
        for nid in node_ids:
            nd = graph.get_node_data(nid)
            if nd:
                unique_files.add(nd.file_path)

        return TokenEstimate(
            total_tokens=total_tokens,
            total_characters=total_chars,
            file_count=len(unique_files),
            node_count=len(node_ids),
            line_count=total_lines,
            per_node=per_node,
        )

    @classmethod
    def estimate_remaining(
        cls,
        current_estimate: TokenEstimate,
        budget: int,
    ) -> int:
        """Estimate remaining token budget.

        Args:
            current_estimate: Current cumulative estimate.
            budget: Total token budget.

        Returns:
            Remaining budget (0 if over budget).
        """
        return max(0, budget - current_estimate.total_tokens)

    @classmethod
    def would_exceed_budget(
        cls,
        current_estimate: TokenEstimate,
        node: GraphNode,
        budget: int,
        graph: DependencyGraph | None = None,
    ) -> bool:
        """Check if adding a node would exceed the token budget.

        Respects node-level granularity: for class/function nodes,
        checks if the containing file is already counted.

        Args:
            current_estimate: Current cumulative estimate.
            node: Node to potentially add.
            budget: Total token budget.
            graph: Optional graph for file-level deduplication.

        Returns:
            True if adding the node would exceed the budget.
        """
        if graph:
            # Check if the file is already fully counted
            for nid, nd in graph.get_all_nodes().items():
                if (nd.file_path == node.file_path
                        and nd.node_type == "file"
                        and nid in current_estimate.per_node
                        and current_estimate.per_node[nid] > 0):
                    # File is already counted, sub-node adds nothing
                    return False

        additional = node.estimated_token_cost
        return (current_estimate.total_tokens + additional) > budget
