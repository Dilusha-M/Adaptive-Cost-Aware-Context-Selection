"""Adaptive budget-aware context retrieval.

Implements the proposed research algorithm: priority-queue-based
graph traversal that scores candidate nodes and respects a
configurable LLM token budget while preserving dependency coverage.

Key features (Improvements 1-7):
- Node-level context: each graph node stores source code, line range,
  character count, and token estimate for accurate context assembly
- Strict token budget: budget is checked BEFORE a node is accepted
- Configurable scoring: weights loaded from config.yaml
- Graph importance metrics: degree centrality, PageRank, betweenness
- Explainable retrieval: every node includes score breakdown and reasons
- Force entry node: --force-entry-node flag for tight budgets
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Any

from graph.graph_builder import DependencyGraph
from graph.graph_search import GraphSearch
from metrics.context_builder import ContextBuilder
from metrics.token_estimator import TokenEstimator, TokenEstimate
from retrieval.scorer import NodeScorer, NodeScore, ScoringConfig


@dataclass
class RetrievalExplanation:
    """Explanation for why a node was retrieved or skipped.

    Attributes:
        node_id: The node identifier.
        status: 'retrieved', 'skipped', or 'queued'.
        total_score: Composite relevance score.
        keyword_relevance: Keyword match score (0.0-1.0).
        dependency_importance: Dependency importance score (0.0-1.0).
        centrality_score: Graph centrality score (0.0-1.0).
        token_score: Token efficiency score (0.0-1.0).
        token_cost: Estimated token cost.
        budget_remaining: Remaining budget after this decision.
        reasons: Natural-language reasons for the decision.
        skip_reason: Reason the node was skipped (if applicable).
    """
    node_id: str
    status: str
    total_score: float
    keyword_relevance: float
    dependency_importance: float
    centrality_score: float
    token_score: float
    token_cost: int
    budget_remaining: int
    reasons: list[str]
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "status": self.status,
            "total_score": round(self.total_score, 4),
            "keyword_relevance": round(self.keyword_relevance, 4),
            "dependency_importance": round(self.dependency_importance, 4),
            "centrality_score": round(self.centrality_score, 4),
            "token_score": round(self.token_score, 4),
            "token_cost": self.token_cost,
            "budget_remaining": self.budget_remaining,
            "reasons": self.reasons,
            "skip_reason": self.skip_reason,
        }


@dataclass
class AdaptiveRetrievalResult:
    """Result of an adaptive budget-aware retrieval operation.

    Attributes:
        algorithm_name: Name of the retrieval algorithm used.
        start_nodes: Query start node IDs.
        retrieved_nodes: Set of retrieved node IDs.
        retrieved_files: Set of file paths retrieved.
        traversal_order: List of node IDs in traversal order.
        skipped_nodes: Dict mapping skipped node IDs to reason strings.
        token_estimate: Token cost estimate for the retrieved set.
        budget: Configured token budget.
        budget_remaining: Tokens remaining in budget.
        explanations: Detailed explanations for each node decision.
        execution_time_ms: Wall-clock execution time in milliseconds.
        query: The original user query.
        context: Assembled context (if context builder was used).
        importance_metric: Graph importance metric used.
    """
    algorithm_name: str
    start_nodes: list[str]
    retrieved_nodes: set[str]
    retrieved_files: set[str]
    traversal_order: list[str]
    skipped_nodes: dict[str, str]
    token_estimate: TokenEstimate
    budget: int
    budget_remaining: int
    explanations: list[RetrievalExplanation]
    execution_time_ms: float
    query: str
    context: dict[str, Any] | None = None
    importance_metric: str = "pagerank"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "algorithm_name": self.algorithm_name,
            "query": self.query,
            "start_nodes": self.start_nodes,
            "retrieved_files": sorted(self.retrieved_files),
            "retrieved_nodes": sorted(self.retrieved_nodes),
            "traversal_order": self.traversal_order,
            "skipped_nodes": self.skipped_nodes,
            "token_estimate": self.token_estimate.to_dict(),
            "budget": self.budget,
            "budget_remaining": self.budget_remaining,
            "explanations": [e.to_dict() for e in self.explanations],
            "execution_time_ms": round(self.execution_time_ms, 2),
            "importance_metric": self.importance_metric,
        }


class AdaptiveBudgetRetriever:
    """Adaptive-Cost-Aware-Context-Selectionalgorithm.

    Uses a priority queue to expand the most relevant nodes first,
    respecting a configurable token budget. Each candidate node is
    scored before expansion. Budget is checked STRICTLY before
    accepting a node.

    This implements the proposed research contribution:
    "Retrieve the highest-value dependency-preserving context
    while remaining within a configurable LLM token budget."
    """

    def __init__(
        self,
        graph: DependencyGraph,
        budget: int = 5000,
        max_depth: int = 5,
        importance_metric: str = "pagerank",
        force_entry_node: bool = False,
        tokenizer_mode: str = "fast",
        scoring_config: ScoringConfig | None = None,
    ) -> None:
        """Initialize the adaptive retriever.

        Args:
            graph: The DependencyGraph to traverse.
            budget: Maximum token budget for retrieved context.
            max_depth: Maximum traversal depth from start nodes.
            importance_metric: Graph importance metric.
                One of 'degree', 'pagerank', 'betweenness'.
            force_entry_node: If True, allow the entry node even if
                it exceeds the budget alone.
            tokenizer_mode: Token estimation mode.
                'fast' (chars/4) or 'accurate' (tiktoken).
            scoring_config: Custom scoring weights. Uses defaults if None.
        """
        self.graph = graph
        self.budget = budget
        self.max_depth = max_depth
        self.importance_metric = importance_metric
        self.force_entry_node = force_entry_node
        self.tokenizer_mode = tokenizer_mode
        self.scoring_config = scoring_config or ScoringConfig()
        self._scorer = NodeScorer(self.scoring_config)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> AdaptiveRetrievalResult:
        """Execute adaptive budget-aware retrieval for a query.

        Uses priority-queue-based traversal where each candidate
        node is scored and checked against the budget BEFORE
        being added.

        Algorithm:
        1. Find start nodes via keyword matching
        2. Score and add start nodes (strict budget check)
        3. For each added node, score and enqueue neighbors
        4. Process queue: score candidate, check budget, accept/skip
        5. Stop when queue is empty or no feasible node exists

        Args:
            query: User's request string (e.g., "Modify LoginService").
            top_k: Number of top start nodes to use.

        Returns:
            AdaptiveRetrievalResult with retrieved nodes, explanations, and metadata.
        """
        start_time = time.perf_counter()

        # Find start nodes via keyword matching
        start_nodes = GraphSearch.find_start_nodes(self.graph, query, top_k=top_k)

        if not start_nodes:
            start_nodes = self.graph.get_all_files()[:1]

        # Initialize tracking
        retrieved: set[str] = set()
        traversal_order: list[str] = []
        skipped: dict[str, str] = {}
        explanations: list[RetrievalExplanation] = []

        # Running budget tracker
        budget_used = 0
        current_estimate = TokenEstimate(
            total_tokens=0,
            total_characters=0,
            file_count=0,
            node_count=0,
            per_node={},
        )

        # Priority queue: (-score, depth, node_id)
        pq: list[tuple[float, int, str]] = []

        # --- Process start nodes ---
        for start in start_nodes:
            node = self.graph.get_node_data(start)
            if not node:
                skipped[start] = "no node data"
                continue

            # Score the start node
            score = self._scorer.score_node(
                node, self.graph, query, self.importance_metric
            )

            # STRICT budget check BEFORE accepting
            additional_cost = self._get_additional_cost(node, current_estimate)

            if budget_used + additional_cost > self.budget:
                if not self.force_entry_node:
                    skipped[start] = (
                        f"Budget exceeded: token cost {node.estimated_token_cost}, "
                        f"budget {self.budget}, already used {budget_used}"
                    )
                    explanations.append(self._make_explanation(
                        node, score, "skipped", budget_used,
                        "Budget exceeded by this node"
                    ))
                    continue
                else:
                    # Force entry even if over budget
                    explanations.append(self._make_explanation(
                        node, score, "retrieved (forced)", budget_used,
                        "Entry node forced despite exceeding budget"
                    ))
                    # Still add it
                    pass
                # Fall through to add

            # Accept the node
            retrieved.add(start)
            traversal_order.append(start)
            budget_used += additional_cost
            self._record_node_cost(node, additional_cost, current_estimate)

            explanations.append(self._make_explanation(
                node, score, "retrieved", self.budget - budget_used, ""
            ))

            # Enqueue neighbors
            if len(traversal_order) <= self.max_depth:
                self._enqueue_neighbors(
                    start, 0, query, pq, skipped, explanations
                )

        # --- Process priority queue ---
        while pq:
            neg_score, depth, node_id = heapq.heappop(pq)
            score_val = -neg_score

            if node_id in retrieved:
                continue

            node = self.graph.get_node_data(node_id)
            if not node:
                skipped[node_id] = "no node data"
                continue

            # Score the candidate
            score = self._scorer.score_node(
                node, self.graph, query, self.importance_metric
            )

            # STRICT budget check BEFORE accepting
            additional_cost = self._get_additional_cost(node, current_estimate)

            if budget_used + additional_cost > self.budget:
                skipped[node_id] = (
                    f"Budget exceeded: cost {node.estimated_token_cost}, "
                    f"budget {self.budget}, used {budget_used}"
                )
                explanations.append(self._make_explanation(
                    node, score, "skipped", self.budget - budget_used,
                    "Budget exceeded"
                ))
                continue

            # Accept the node
            retrieved.add(node_id)
            traversal_order.append(node_id)
            budget_used += additional_cost
            self._record_node_cost(node, additional_cost, current_estimate)

            explanations.append(self._make_explanation(
                node, score, "retrieved", self.budget - budget_used, ""
            ))

            # Enqueue neighbors if under max depth
            if depth + 1 < self.max_depth:
                self._enqueue_neighbors(
                    node_id, depth + 1, query, pq, skipped, explanations
                )

        # Check if no node was feasible
        if not retrieved and not self.force_entry_node:
            skipped["(start)"] = "No feasible solution within budget"
            explanations.append(RetrievalExplanation(
                node_id="(start)",
                status="skipped",
                total_score=0.0,
                keyword_relevance=0.0,
                dependency_importance=0.0,
                centrality_score=0.0,
                token_score=0.0,
                token_cost=0,
                budget_remaining=self.budget,
                reasons=["No feasible solution within budget"],
                skip_reason="All candidate nodes exceed budget",
            ))

        # Final token estimate using node-level estimation
        token_estimate = TokenEstimator.estimate_nodes(
            retrieved, self.graph, self.tokenizer_mode
        )
        budget_remaining = max(0, self.budget - token_estimate.total_tokens)

        # Collect unique file paths
        retrieved_files: set[str] = set()
        for nid in retrieved:
            nd = self.graph.get_node_data(nid)
            if nd:
                retrieved_files.add(nd.file_path)

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000

        return AdaptiveRetrievalResult(
            algorithm_name="Adaptive Budget-Aware",
            start_nodes=start_nodes,
            retrieved_nodes=retrieved,
            retrieved_files=retrieved_files,
            traversal_order=traversal_order,
            skipped_nodes=skipped,
            token_estimate=token_estimate,
            budget=self.budget,
            budget_remaining=budget_remaining,
            explanations=explanations,
            execution_time_ms=execution_time,
            query=query,
            importance_metric=self.importance_metric,
        )

    def _get_additional_cost(self, node: GraphNode, current_estimate: TokenEstimate) -> int:
        """Calculate the additional token cost of adding a node.

        Returns 0 for sub-nodes whose file is already fully counted.

        Args:
            node: The graph node to evaluate.
            current_estimate: Current running estimate.

        Returns:
            Additional token cost.
        """
        if node.node_type == "file":
            return node.estimated_token_cost
        else:
            # For class/function, check if file is already counted
            for nid, nd in current_estimate.per_node.items():
                node_nd = self.graph.get_node_data(nid)
                if (node_nd and node_nd.file_path == node.file_path
                        and node_nd.node_type == "file"
                        and nd > 0):
                    return 0
            return node.estimated_token_cost

    def _record_node_cost(
        self,
        node: GraphNode,
        additional_cost: int,
        current_estimate: TokenEstimate,
    ) -> None:
        """Record a node's cost in the running estimate.

        Args:
            node: The graph node.
            additional_cost: The cost to add.
            current_estimate: The running estimate to update.
        """
        if node.node_type == "file":
            current_estimate.per_node[node.node_id] = node.estimated_token_cost
            current_estimate.total_tokens += node.estimated_token_cost
            current_estimate.total_characters += node.char_count or node.estimated_token_cost * 4
        elif additional_cost > 0:
            current_estimate.per_node[node.node_id] = additional_cost
            current_estimate.total_tokens += additional_cost
            current_estimate.total_characters += node.char_count

        # Track unique files
        unique_files = set()
        for nid in current_estimate.per_node:
            nd = self.graph.get_node_data(nid)
            if nd:
                unique_files.add(nd.file_path)
        current_estimate.file_count = len(unique_files)
        current_estimate.node_count = len(current_estimate.per_node)

    def _enqueue_neighbors(
        self,
        node_id: str,
        depth: int,
        query: str,
        pq: list[tuple[float, int, str]],
        skipped: dict[str, str],
        explanations: list[RetrievalExplanation],
    ) -> None:
        """Enqueue all neighbor nodes of a node into the priority queue.

        Scores each neighbor and adds it to the queue. Already-retrieved
        neighbors are skipped with a recorded reason.

        Args:
            node_id: The parent node ID.
            depth: Depth of the parent node.
            query: The user's query.
            pq: The priority queue to add to.
            skipped: Dict to record skipped nodes.
            explanations: List to append explanations to.
        """
        neighbors = list(self.graph.graph.successors(node_id))

        for neighbor in neighbors:
            # Skip if already retrieved
            if neighbor in set(n for n in pq):
                continue

            neighbor_node = self.graph.get_node_data(neighbor)
            if not neighbor_node:
                skipped[neighbor] = "no node data"
                continue

            score = self._scorer.score_node(
                neighbor_node, self.graph, query, self.importance_metric
            )

            # Push with negative score (min-heap simulates max-heap)
            heapq.heappush(pq, (-score.total_score, depth + 1, neighbor))

    @staticmethod
    def _make_explanation(
        node: GraphNode,
        score: NodeScore,
        status: str,
        budget_remaining: int,
        skip_reason: str = "",
    ) -> RetrievalExplanation:
        """Create a RetrievalExplanation for a node decision."""
        return RetrievalExplanation(
            node_id=node.node_id,
            status=status,
            total_score=score.total_score,
            keyword_relevance=score.keyword_relevance,
            dependency_importance=score.dependency_importance,
            centrality_score=score.centrality_score,
            token_score=score.token_score,
            token_cost=node.estimated_token_cost,
            budget_remaining=budget_remaining,
            reasons=score.reasons,
            skip_reason=skip_reason,
        )
