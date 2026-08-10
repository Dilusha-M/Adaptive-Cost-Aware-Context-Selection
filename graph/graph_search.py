"""Graph search utilities for repository dependency graphs.

Provides helper functions for locating nodes by keyword matching
and finding relevant neighbors in a DependencyGraph.

Includes improved query matching with camelCase/split support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from graph.graph_builder import DependencyGraph, GraphNode


@dataclass
class SearchMatch:
    """Result of a keyword search against graph nodes.

    Attributes:
        node_id: The matching node identifier.
        score: Relevance score (higher = better match).
        matched_text: The text that matched the query.
    """
    node_id: str
    score: float
    matched_text: str


class GraphSearch:
    """Keyword-based search utility for DependencyGraph.

    Supports case-insensitive matching with improved token splitting
    for camelCase and snake_case identifiers.
    """

    @staticmethod
    def search(
        graph: DependencyGraph,
        query: str,
        node_types: list[str] | None = None,
    ) -> list[SearchMatch]:
        """Search for nodes matching a keyword query.

        Performs improved case-insensitive matching across node names
        and file paths, with support for camelCase and snake_case.

        Args:
            graph: The DependencyGraph to search.
            query: Search keywords (space-separated, all must match).
            node_types: Optional filter for node types ('file', 'class', 'function').

        Returns:
            List of SearchMatch sorted by relevance score descending.
        """
        query_terms = _split_query(query)
        matches: list[SearchMatch] = []

        for node_id, node_data in graph.get_all_nodes().items():
            if node_types and node_data.node_type not in node_types:
                continue

            match_score = 0.0
            matched_text = ""

            for term in query_terms:
                # Check node name
                if _match_token(term, node_data.name):
                    # Exact word match scores higher
                    if _exact_match(term, node_data.name):
                        match_score += 3.0
                        matched_text = node_data.name
                    else:
                        match_score += 1.0
                        if not matched_text:
                            matched_text = node_data.name

                # Check file path
                if _match_token(term, node_data.file_path):
                    match_score += 1.5
                    if not matched_text:
                        matched_text = node_data.file_path

            if match_score > 0:
                matches.append(SearchMatch(
                    node_id=node_id,
                    score=match_score,
                    matched_text=matched_text,
                ))

        return sorted(matches, key=lambda m: m.score, reverse=True)

    @staticmethod
    def find_start_nodes(
        graph: DependencyGraph,
        query: str,
        top_k: int = 3,
    ) -> list[str]:
        """Find the top-K starting nodes for a query.

        Args:
            graph: The DependencyGraph to search.
            query: Search query string.
            top_k: Number of top results to return.

        Returns:
            List of node IDs, ordered by relevance.
        """
        matches = GraphSearch.search(graph, query)
        return [m.node_id for m in matches[:top_k]]

    @staticmethod
    def get_import_chain(
        graph: DependencyGraph,
        start_node: str,
        direction: str = "outgoing",
    ) -> list[str]:
        """Get the chain of import dependencies from a node.

        Args:
            graph: The DependencyGraph.
            start_node: Node ID to start from.
            direction: 'outgoing' for imports made by node,
                       'incoming' for modules that import the node.

        Returns:
            List of connected node IDs.
        """
        if direction == "outgoing":
            neighbors = list(graph.graph.successors(start_node))
        else:
            neighbors = list(graph.graph.predecessors(start_node))

        # Filter to only import-type edges
        import_neighbors = []
        for neighbor in neighbors:
            edge_data = graph.graph.get_edge_data(start_node, neighbor)
            if edge_data and edge_data.get("edge_type") == "imports":
                import_neighbors.append(neighbor)
        return import_neighbors

    @staticmethod
    def get_all_dependencies(
        graph: DependencyGraph,
        start_node: str,
        max_depth: int = 5,
    ) -> set[str]:
        """Get all transitive dependencies of a node up to max_depth.

        Args:
            graph: The DependencyGraph.
            start_node: Node ID to start from.
            max_depth: Maximum traversal depth.

        Returns:
            Set of all reachable node IDs.
        """
        visited: set[str] = set()
        current_level: set[str] = {start_node}
        depth = 0

        while current_level and depth < max_depth:
            next_level: set[str] = set()
            for node in current_level:
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in graph.graph.successors(node):
                    if neighbor not in visited:
                        next_level.add(neighbor)
            current_level = next_level
            depth += 1

        return visited

    @staticmethod
    def find_required_deps(
        graph: DependencyGraph,
        start_nodes: list[str],
    ) -> set[str]:
        """Find all required dependency nodes for the given start nodes.

        Traverses outgoing import edges to find all files that
        must be included for complete context.

        Args:
            graph: The DependencyGraph.
            start_nodes: List of starting node IDs.

        Returns:
            Set of all required dependency node IDs.
        """
        required: set[str] = set()
        queue: list[str] = list(start_nodes)

        while queue:
            current = queue.pop(0)
            if current in required:
                continue
            required.add(current)

            for neighbor in graph.graph.successors(current):
                edge_data = graph.graph.get_edge_data(current, neighbor)
                if edge_data and edge_data.get("edge_type") == "imports":
                    if neighbor not in required:
                        queue.append(neighbor)

        return required


def _split_query(query: str) -> list[str]:
    """Split a query string into meaningful tokens.

    Supports camelCase, snake_case, and whitespace splitting.

    Args:
        query: The query string.

    Returns:
        List of lowercased tokens (each at least 2 chars).
    """
    tokens: list[str] = []

    # Split camelCase / PascalCase
    spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', query)
    # Split snake_case
    spaced = spaced.replace("_", " ")
    # Split by whitespace
    for token in spaced.split():
        token = token.strip().lower()
        if token and len(token) >= 2:
            tokens.append(token)

    if not tokens:
        tokens = [t.lower() for t in query.split() if len(t) >= 2]

    return tokens


def _match_token(token: str, text: str) -> bool:
    """Check if a token matches part of the text.

    Case-insensitive substring match with word-boundary support.

    Args:
        token: The search token (lowercased).
        text: The text to search in.

    Returns:
        True if the token matches.
    """
    return token.lower() in text.lower()


def _split_camelcase(text: str) -> list[str]:
    """Split camelCase/snake_case identifiers into tokens."""
    parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    parts = parts.replace('_', ' ').replace('-', ' ')
    return parts.split()


def _exact_match(token: str, text: str) -> bool:
    """Check if token is an exact match with word boundaries.
    
    Args:
        token: The search token (lowercased).
        text: The text to search in.

    Returns:
        True if the token is a whole-word match.
    """
    token_lower = token.lower()
    # Check word boundaries first
    pattern = r'\b' + re.escape(token_lower) + r'\b'
    if re.search(pattern, text.lower()):
        return True
    # Check camelCase/snake_case boundaries
    parts = _split_camelcase(text)
    return any(part_lower == token_lower for part in parts for part_lower in [part.lower()])
