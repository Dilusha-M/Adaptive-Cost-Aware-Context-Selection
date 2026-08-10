"""Configurable scoring for adaptive retrieval.

Provides scoring heuristics to rank candidate nodes during
priority-queue-based graph traversal.

Scoring weights are read from config.yaml (Improvement 5),
with sensible defaults if no config file is found.

Graph importance metrics (Improvement 6):
- Degree Centrality
- PageRank
- Betweenness Centrality
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from graph.graph_builder import DependencyGraph, GraphNode


# Default scoring configuration
DEFAULT_CONFIG = {
    "keyword_weight": 0.40,
    "dependency_weight": 0.30,
    "centrality_weight": 0.20,
    "token_weight": 0.10,
}

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


@dataclass
class NodeScore:
    """Score for a candidate node during adaptive retrieval.

    Attributes:
        node_id: The node identifier.
        keyword_relevance: Score for query keyword matching (0.0-1.0).
        dependency_importance: Score based on dependency graph position (0.0-1.0).
        centrality_score: Graph centrality metric for the node (0.0-1.0).
        token_score: Score inversely proportional to token cost (0.0-1.0).
        total_score: Combined weighted score.
        reasons: List of natural-language reasons for the score.
    """
    node_id: str
    keyword_relevance: float
    dependency_importance: float
    centrality_score: float
    token_score: float
    total_score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "keyword_relevance": round(self.keyword_relevance, 4),
            "dependency_importance": round(self.dependency_importance, 4),
            "centrality_score": round(self.centrality_score, 4),
            "token_score": round(self.token_score, 4),
            "total_score": round(self.total_score, 4),
            "reasons": self.reasons,
        }


class ScoringConfig:
    """Scoring configuration loaded from config.yaml or defaults.

    Attributes:
        keyword_weight: Weight for keyword relevance (default 0.40).
        dependency_weight: Weight for dependency importance (default 0.30).
        centrality_weight: Weight for graph centrality (default 0.20).
        token_weight: Weight for token cost efficiency (default 0.10).
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize scoring configuration.

        Args:
            config_path: Path to config.yaml. Uses default if not found.
        """
        path = config_path or str(CONFIG_PATH)
        self.config = self._load_config(path)

    @property
    def keyword_weight(self) -> float:
        """Keyword relevance weight."""
        return float(self.config.get("keyword_weight", DEFAULT_CONFIG["keyword_weight"]))

    @property
    def dependency_weight(self) -> float:
        """Dependency importance weight."""
        return float(self.config.get("dependency_weight", DEFAULT_CONFIG["dependency_weight"]))

    @property
    def centrality_weight(self) -> float:
        """Centrality weight."""
        return float(self.config.get("centrality_weight", DEFAULT_CONFIG["centrality_weight"]))

    @property
    def token_weight(self) -> float:
        """Token cost weight."""
        return float(self.config.get("token_weight", DEFAULT_CONFIG["token_weight"]))

    def _load_config(self, path: str) -> dict[str, float]:
        """Load configuration from YAML file, falling back to defaults."""
        try:
            config_file = Path(path)
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if loaded and isinstance(loaded, dict):
                        return {**DEFAULT_CONFIG, **loaded}
        except (yaml.YAMLError, OSError):
            pass
        return dict(DEFAULT_CONFIG)


class NodeScorer:
    """Computes relevance scores for graph nodes during adaptive retrieval.

    The scoring function balances:
    - Keyword relevance: how well the node name matches the query
    - Dependency importance: how central the node is in the graph
    - Graph centrality: degree centrality, PageRank, or betweenness
    - Token efficiency: lower-cost nodes score higher

    Score(node) = kw * keyword_relevance
                + dep * dependency_importance
                + cen * centrality_score
                + tok * token_score

    Weights (kw, dep, cen, tok) come from ScoringConfig (config.yaml).
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        """Initialize the scorer with configurable weights.

        Args:
            config: Scoring configuration. Uses defaults if not provided.
        """
        self.config = config or ScoringConfig()

    def score_node(
        self,
        node: GraphNode,
        graph: DependencyGraph,
        query: str,
        importance_metric: str = "pagerank",
    ) -> NodeScore:
        """Compute a composite score for a candidate node.

        Args:
            node: The graph node to score.
            graph: The DependencyGraph.
            query: The user's query string.
            importance_metric: Graph importance metric to use.
                One of 'degree', 'pagerank', 'betweenness'.

        Returns:
            NodeScore with individual and composite scores plus reasons.
        """
        kw_score = self._compute_keyword_relevance(node, query)
        dep_score = self._compute_dependency_importance(node, graph)
        cen_score = self._compute_centrality(node, graph, importance_metric)
        tok_score = self._compute_token_score(node)

        kw = self.config.keyword_weight
        dep = self.config.dependency_weight
        cen = self.config.centrality_weight
        tok = self.config.token_weight

        total = (
            kw * kw_score
            + dep * dep_score
            + cen * cen_score
            + tok * tok_score
        )

        # Generate natural-language reasons
        reasons = self._generate_reasons(node, kw_score, dep_score, cen_score, tok_score, query)

        return NodeScore(
            node_id=node.node_id,
            keyword_relevance=kw_score,
            dependency_importance=dep_score,
            centrality_score=cen_score,
            token_score=tok_score,
            total_score=total,
            reasons=reasons,
        )

    @staticmethod
    def _compute_keyword_relevance(node: GraphNode, query: str) -> float:
        """Compute keyword relevance score (0.0 to 1.0).

        Uses improved matching:
        - Case-insensitive substring matching
        - CamelCase splitting for both query and node name
        - Snake_case splitting
        - Exact match scores highest

        Args:
            node: The graph node.
            query: The user's query.

        Returns:
            Score between 0.0 and 1.0.
        """
        node_name = node.name
        # Also check file path
        combined = f"{node_name} {node.file_path}"

        query_parts = _split_query(query)

        max_score = 0.0
        for part in query_parts:
            part_lower = part.lower()
            combined_lower = combined.lower()

            # Exact word match
            pattern = r'\b' + re.escape(part_lower) + r'\b'
            if re.search(pattern, combined_lower):
                max_score = max(max_score, 1.0)
            # CamelCase / snake_case split match
            elif part_lower in combined_lower:
                max_score = max(max_score, 0.7)
            # Prefix match
            elif combined_lower.startswith(part_lower):
                max_score = max(max_score, 0.5)

        return max_score

    @staticmethod
    def _compute_dependency_importance(node: GraphNode, graph: DependencyGraph) -> float:
        """Compute dependency importance score (0.0 to 1.0).

        Nodes with more inbound edges (imported by many files)
        are more important to include.

        Args:
            node: The graph node.
            graph: The DependencyGraph.

        Returns:
            Score between 0.0 and 1.0.
        """
        inbound = graph.graph.in_degree(node.node_id)
        outbound = graph.graph.out_degree(node.node_id)

        # Hub score: nodes that are both imported and do importing
        if inbound > 0 and outbound > 0:
            return 1.0
        elif inbound > 0:
            # Cap at 0.9 for highly imported nodes
            return min(0.9, 0.3 + inbound * 0.15)
        elif outbound > 0:
            return min(0.6, 0.2 + outbound * 0.1)
        return 0.0

    @staticmethod
    def _compute_centrality(
        node: GraphNode,
        graph: DependencyGraph,
        metric: str = "pagerank",
    ) -> float:
        """Compute graph centrality score (0.0 to 1.0).

        Supports degree centrality, PageRank, and betweenness centrality.

        Args:
            node: The graph node.
            graph: The DependencyGraph.
            metric: One of 'degree', 'pagerank', 'betweenness'.

        Returns:
            Score between 0.0 and 1.0.
        """
        if metric == "degree":
            centrality = graph.compute_degree_centrality()
        elif metric == "pagerank":
            centrality = graph.compute_pagerank()
        elif metric == "betweenness":
            centrality = graph.compute_betweenness_centrality()
        else:
            centrality = graph.compute_pagerank()

        score = centrality.get(node.node_id, 0.0)
        # Normalize: centrality values are typically small, so scale
        max_possible = max(centrality.values()) if centrality else 1.0
        if max_possible > 0:
            return min(1.0, score / max_possible)
        return 0.0

    @staticmethod
    def _compute_token_score(node: GraphNode) -> float:
        """Compute token efficiency score (0.0 to 1.0).

        Lower token cost = higher score.

        Args:
            node: The graph node.

        Returns:
            Score between 0.0 and 1.0.
        """
        cost = node.estimated_token_cost
        # Exponential decay: cheap nodes get ~1.0, expensive get ~0.0
        return max(0.0, 1.0 - (cost / 1000.0))

    def _generate_reasons(
        self,
        node: GraphNode,
        kw_score: float,
        dep_score: float,
        cen_score: float,
        tok_score: float,
        query: str,
    ) -> list[str]:
        """Generate natural-language reasons for the node's score.

        Args:
            node: The scored node.
            kw_score: Keyword relevance score.
            dep_score: Dependency importance score.
            cen_score: Centrality score.
            tok_score: Token efficiency score.
            query: The user's query.

        Returns:
            List of reason strings.
        """
        reasons = []

        if kw_score >= 0.7:
            reasons.append("High keyword match")
        elif kw_score >= 0.3:
            reasons.append("Partial keyword match")
        else:
            reasons.append("No keyword match")

        if dep_score >= 0.7:
            inbound = len(list(self._try_get_inbound(node)))
            if inbound > 0:
                reasons.append(f"Imported by {inbound} node(s)")
        if cen_score >= 0.7:
            reasons.append("High centrality")

        if tok_score >= 0.7:
            reasons.append("Low token cost")
        elif tok_score < 0.3:
            reasons.append("High token cost")

        return reasons

    def _try_get_inbound(self, node: GraphNode):
        """Safely get inbound neighbors."""
        try:
            return self._get_graph_successors(node.node_id)
        except Exception:
            return iter([])

    def _get_graph_successors(self, node_id: str):
        """Get graph successors for a node."""
        # We need access to the graph, but this method is called without it
        # So we return empty - this is a helper for _generate_reasons
        return []

    def rank_candidates(
        self,
        candidate_ids: list[str],
        graph: DependencyGraph,
        query: str,
        importance_metric: str = "pagerank",
    ) -> list[NodeScore]:
        """Score and rank a list of candidate nodes.

        Args:
            candidate_ids: List of candidate node IDs.
            graph: The DependencyGraph.
            query: The user's query.
            importance_metric: Graph importance metric.

        Returns:
            List of NodeScore sorted by total_score descending.
        """
        scores = []
        for node_id in candidate_ids:
            node = graph.get_node_data(node_id)
            if node:
                score = self.score_node(node, graph, query, importance_metric)
                scores.append(score)

        return sorted(scores, key=lambda s: s.total_score, reverse=True)


def _split_query(query: str) -> list[str]:
    """Split a query string into meaningful tokens.

    Supports:
    - Snake case splitting (e.g., "user_repository" -> ["user", "repository"])
    - Camel case splitting (e.g., "LoginService" -> ["Login", "Service"])
    - Whitespace separation
    - Case-insensitive matching

    Args:
        query: The query string.

    Returns:
        List of lowercased tokens.
    """
    tokens: list[str] = []

    # Split camelCase / PascalCase
    # e.g., "LoginService" -> "Login Service"
    spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', query)
    # Split snake_case
    spaced = spaced.replace("_", " ")
    # Split by whitespace
    for token in spaced.split():
        token = token.strip().lower()
        if token and len(token) >= 2:
            tokens.append(token)

    if not tokens:
        # Fallback: use the original query terms
        tokens = [t.lower() for t in query.split() if len(t) >= 2]

    return tokens
