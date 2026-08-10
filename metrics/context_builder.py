"""Context builder for assembling retrieved node context.

Receives retrieved nodes, assembles their source code snippets,
sorts by source order, merges snippets, and computes accurate
token estimates.

This module ensures that context is assembled at the node level,
not the file level, providing accurate token estimation for
partial retrievals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from graph.graph_builder import DependencyGraph, GraphNode


@dataclass
class Context:
    """Assembled context from retrieved nodes.

    Attributes:
        nodes: List of retrieved GraphNodes in source order.
        source: The assembled source code text.
        total_characters: Total character count.
        estimated_tokens: Estimated token count.
        line_count: Total number of lines in the assembled context.
    """
    nodes: list[GraphNode]
    source: str
    total_characters: int
    estimated_tokens: int
    line_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_characters": self.total_characters,
            "estimated_tokens": self.estimated_tokens,
            "line_count": self.line_count,
            "node_count": len(self.nodes),
            "files": sorted(set(n.file_path for n in self.nodes)),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "node_type": n.node_type,
                    "file_path": n.file_path,
                    "char_count": n.char_count,
                    "estimated_token_cost": n.estimated_token_cost,
                    "start_line": n.start_line,
                    "end_line": n.end_line,
                }
                for n in self.nodes
            ],
        }


class ContextBuilder:
    """Builds assembled context from retrieved graph nodes.

    Responsibilities:
    - Receive retrieved nodes from retrieval algorithms
    - Sort by source order (file path, then line number)
    - Merge snippets with file headers
    - Remove duplicates
    - Compute accurate token estimates

    Context is assembled at the node level: if only a class or
    function is retrieved, only that node's source is included,
    not the entire file.
    """

    # Token estimation modes
    MODE_FAST = "fast"
    MODE_ACCURATE = "accurate"

    def __init__(
        self,
        tokenizer_mode: str = MODE_FAST,
    ) -> None:
        """Initialize the context builder.

        Args:
            tokenizer_mode: Token estimation mode.
                'fast': chars / 4 (default)
                'accurate': tiktoken-based estimation
        """
        self.tokenizer_mode = tokenizer_mode

    def build(
        self,
        node_ids: set[str],
        graph: DependencyGraph,
    ) -> Context:
        """Build assembled context from retrieved nodes.

        Sorts nodes by source order, merges snippets, removes
        duplicates, and computes accurate token estimates.

        Args:
            node_ids: Set of retrieved node IDs.
            graph: The DependencyGraph containing the nodes.

        Returns:
            Context with assembled source and metadata.
        """
        # Collect and sort nodes by source order
        nodes = self._collect_nodes(node_ids, graph)
        nodes = self._sort_by_source_order(nodes)
        nodes = self._remove_duplicates(nodes)

        # Assemble source
        source_parts: list[str] = []
        current_file: str | None = None

        for node in nodes:
            if node.file_path != current_file:
                # File header
                if current_file is not None:
                    source_parts.append(
                        f"\n{'=' * 72}\n"
                    )
                current_file = node.file_path
                source_parts.append(f"--- {node.file_path} ---\n")

            # Node header
            source_parts.append(
                f"# {node.node_type}: {node.name} "
                f"(lines {node.start_line}-{node.end_line})\n"
            )
            source_parts.append(node.source_code)
            source_parts.append("\n")

        source = "".join(source_parts)

        # Compute metrics
        total_chars = len(source)

        if self.tokenizer_mode == self.MODE_ACCURATE:
            estimated_tokens = self._estimate_tokens_accurate(source)
        else:
            estimated_tokens = max(1, total_chars // 4)

        line_count = source.count("\n")

        return Context(
            nodes=nodes,
            source=source,
            total_characters=total_chars,
            estimated_tokens=estimated_tokens,
            line_count=line_count,
        )

    def _collect_nodes(
        self,
        node_ids: set[str],
        graph: DependencyGraph,
    ) -> list[GraphNode]:
        """Collect GraphNode objects from IDs, filtering out invalid ones."""
        nodes = []
        for nid in node_ids:
            node = graph.get_node_data(nid)
            if node and node.source_code:  # Only include nodes with source code
                nodes.append(node)
            elif node and node.node_type == "file":
                # For file nodes without source_code set, skip them
                # (files don't carry snippets in the graph)
                pass
        return nodes

    @staticmethod
    def _sort_by_source_order(nodes: list[GraphNode]) -> list[GraphNode]:
        """Sort nodes by file path, then by start line.

        Ensures the assembled context is in a logical source order.

        Args:
            nodes: List of GraphNodes.

        Returns:
            Sorted list of GraphNodes.
        """
        return sorted(nodes, key=lambda n: (n.file_path, n.start_line))

    @staticmethod
    def _remove_duplicates(nodes: list[GraphNode]) -> list[GraphNode]:
        """Remove duplicate nodes by node_id, keeping first occurrence.

        Args:
            nodes: List of GraphNodes (expected to be sorted).

        Returns:
            Deduplicated list of GraphNodes.
        """
        seen: set[str] = set()
        unique = []
        for node in nodes:
            if node.node_id not in seen:
                seen.add(node.node_id)
                unique.append(node)
        return unique

    @staticmethod
    def _estimate_tokens_accurate(text: str) -> int:
        """Estimate tokens using tiktoken (gpt-4o-mini encoding).

        Falls back to fast estimation if tiktoken is not available.

        Args:
            text: Source code text to estimate.

        Returns:
            Estimated token count.
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Fallback to fast estimation
            return max(1, len(text) // 4)
