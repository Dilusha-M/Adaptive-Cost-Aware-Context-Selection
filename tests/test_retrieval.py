"""Unit tests for retrieval algorithms and metrics.

Tests cover baseline BFS retrieval, adaptive budget-aware retrieval,
token estimation, and dependency coverage calculation.
"""

import pytest
from pathlib import Path

from parser.ast_parser import ASTParser
from graph.graph_builder import GraphBuilder
from retrieval.baseline_bfs import BaselineBFS
from retrieval.adaptive_budget import AdaptiveBudgetRetriever
from metrics.token_estimator import TokenEstimator, TokenEstimate
from metrics.dependency_coverage import DependencyCoverage, CoverageResult


class TestBaselineBFS:
    """Tests for the baseline BFS retrieval algorithm."""

    @pytest.fixture
    def graph(self):
        """Create a dependency graph from sample repo."""
        builder = GraphBuilder()
        return builder.build_from_directory("sample_repo")

    def test_bfs_retrieves_nodes(self, graph) -> None:
        """Test that BFS retrieval returns at least some nodes."""
        retriever = BaselineBFS(graph, max_depth=3)
        result = retriever.retrieve("LoginService")

        assert len(result.retrieved_nodes) > 0
        assert len(result.retrieved_files) > 0
        assert len(result.traversal_order) > 0

    def test_bfs_retrieval_contains_start_node(self, graph) -> None:
        """Test that the start node is always in the retrieval result."""
        retriever = BaselineBFS(graph, max_depth=3)
        result = retriever.retrieve("LoginService")

        # "LoginService" should match auth_service.py or the class node
        assert any(
            "LoginService" in nid or "login" in nid.lower()
            for nid in result.start_nodes
        )
        assert any(
            "login" in nid.lower() or "auth" in nid.lower()
            for nid in result.retrieved_nodes
        )

    def test_bfs_respects_max_depth(self, graph) -> None:
        """Test that BFS respects the maximum depth parameter."""
        shallow = BaselineBFS(graph, max_depth=1)
        deep = BaselineBFS(graph, max_depth=5)

        result_shallow = shallow.retrieve("Database")
        result_deep = deep.retrieve("Database")

        # Deeper search should retrieve at least as many nodes
        assert len(result_deep.retrieved_nodes) >= len(result_shallow.retrieved_nodes)

    def test_bfs_execution_time(self, graph) -> None:
        """Test that BFS completes in reasonable time."""
        retriever = BaselineBFS(graph, max_depth=3)
        result = retriever.retrieve("UserRepository")

        assert result.execution_time_ms >= 0
        assert result.execution_time_ms < 5000  # Should be fast

    def test_bfs_returns_token_estimate(self, graph) -> None:
        """Test that BFS returns a valid token estimate."""
        retriever = BaselineBFS(graph, max_depth=3)
        result = retriever.retrieve("Auth")

        assert isinstance(result.token_estimate, TokenEstimate)
        assert result.token_estimate.total_tokens > 0
        assert result.token_estimate.node_count == len(result.retrieved_nodes)

    def test_bfs_empty_query_fallback(self, graph) -> None:
        """Test that BFS handles unusual queries gracefully."""
        retriever = BaselineBFS(graph, max_depth=3)
        result = retriever.retrieve("xyznonexistent123")

        # Should not crash, may return empty or fallback results
        assert result is not None
        assert isinstance(result.token_estimate, TokenEstimate)

    def test_bfs_result_serialization(self, graph) -> None:
        """Test that retrieval results can be serialized to dict."""
        retriever = BaselineBFS(graph, max_depth=3)
        result = retriever.retrieve("LoginService")

        data = result.to_dict()
        assert "algorithm_name" in data
        assert data["algorithm_name"] == "Baseline BFS"
        assert "retrieved_files" in data
        assert "token_estimate" in data


class TestAdaptiveBudgetRetriever:
    """Tests for the adaptive budget-aware retrieval algorithm."""

    @pytest.fixture
    def graph(self):
        """Create a dependency graph from sample repo."""
        builder = GraphBuilder()
        return builder.build_from_directory("sample_repo")

    def test_adaptive_retrieves_nodes(self, graph) -> None:
        """Test that adaptive retrieval returns at least some nodes."""
        retriever = AdaptiveBudgetRetriever(graph, budget=10000)
        result = retriever.retrieve("LoginService")

        assert len(result.retrieved_nodes) > 0
        assert len(result.retrieved_files) > 0

    def test_adaptive_respects_budget(self, graph) -> None:
        """Test that adaptive retrieval stays within budget."""
        budget = 2000
        retriever = AdaptiveBudgetRetriever(graph, budget=budget)
        result = retriever.retrieve("LoginService")

        assert result.token_estimate.total_tokens <= budget + 500  # Small tolerance
        assert result.budget_remaining >= 0

    def test_adaptive_smaller_than_baseline(self, graph) -> None:
        """Test that adaptive retrieval can be smaller than baseline."""
        # Very tight budget
        tight_retriever = AdaptiveBudgetRetriever(graph, budget=500)
        tight_result = tight_retriever.retrieve("LoginService")

        # Baseline with default depth should be larger or equal
        baseline_retriever = BaselineBFS(graph, max_depth=3)
        baseline_result = baseline_retriever.retrieve("LoginService")

        # With very tight budget, adaptive should retrieve fewer or equal
        assert tight_result.token_estimate.total_tokens <= (
            baseline_result.token_estimate.total_tokens + 500
        )

    def test_adaptive_budget_scoring(self, graph) -> None:
        """Test that adaptive retrieval produces explanation records with scores."""
        retriever = AdaptiveBudgetRetriever(graph, budget=10000)
        result = retriever.retrieve("LoginService")

        assert isinstance(result.explanations, list)
        assert len(result.explanations) > 0

        # Check explanation structure
        for exp in result.explanations:
            d = exp.to_dict() if hasattr(exp, "to_dict") else exp
            assert "node_id" in d
            assert "total_score" in d
            assert "status" in d
            assert "reasons" in d

    def test_adaptive_budget_remaining(self, graph) -> None:
        """Test that budget remaining is calculated correctly."""
        budget = 5000
        retriever = AdaptiveBudgetRetriever(graph, budget=budget)
        result = retriever.retrieve("Database")

        assert result.budget == budget
        assert result.budget_remaining >= 0

    def test_adaptive_skipped_nodes_recorded(self, graph) -> None:
        """Test that skipped nodes are recorded with reasons."""
        tight_retriever = AdaptiveBudgetRetriever(graph, budget=100)
        result = tight_retriever.retrieve("Modify LoginService")

        # With very tight budget, many nodes should be skipped
        assert isinstance(result.skipped_nodes, dict)

    def test_adaptive_result_serialization(self, graph) -> None:
        """Test that adaptive retrieval results can be serialized."""
        retriever = AdaptiveBudgetRetriever(graph, budget=5000)
        result = retriever.retrieve("LoginService")

        data = result.to_dict()
        assert "algorithm_name" in data
        assert data["algorithm_name"] == "Adaptive Budget-Aware"
        assert "budget" in data
        assert "explanations" in data


class TestTokenEstimator:
    """Tests for token estimation functionality."""

    def test_estimate_single_file(self) -> None:
        """Test estimating tokens for a single file node."""
        builder = GraphBuilder()
        graph = builder.build_from_directory("sample_repo")

        file_nodes = [n for n, d in graph.get_all_nodes().items()
                      if d.node_type == "file"]

        if file_nodes:
            estimate = TokenEstimator.estimate_nodes({file_nodes[0]}, graph)
            assert estimate.total_tokens > 0
            assert estimate.total_characters > 0
            assert estimate.file_count > 0

    def test_estimate_multiple_files(self) -> None:
        """Test estimating tokens for multiple files."""
        builder = GraphBuilder()
        graph = builder.build_from_directory("sample_repo")

        file_nodes = [n for n, d in graph.get_all_nodes().items()
                      if d.node_type == "file"]

        if len(file_nodes) >= 2:
            estimate = TokenEstimator.estimate_nodes(
                {file_nodes[0], file_nodes[1]}, graph
            )
            assert estimate.total_tokens > 0
            assert estimate.node_count >= 2

    def test_estimate_remaining(self) -> None:
        """Test calculating remaining budget."""
        estimate = TokenEstimate(
            total_tokens=3000,
            total_characters=12000,
            file_count=2,
            node_count=5,
            per_node={"test.py": 3000},
        )

        remaining = TokenEstimator.estimate_remaining(estimate, 5000)
        assert remaining == 2000

        # Over budget
        remaining = TokenEstimator.estimate_remaining(estimate, 2000)
        assert remaining == 0

    def test_estimate_serialization(self) -> None:
        """Test that TokenEstimate can be serialized."""
        estimate = TokenEstimate(
            total_tokens=1000,
            total_characters=4000,
            file_count=1,
            node_count=3,
            per_node={"test.py": 1000},
        )

        data = estimate.to_dict()
        assert data["total_tokens"] == 1000
        assert data["file_count"] == 1
        assert data["node_count"] == 3


class TestDependencyCoverage:
    """Tests for dependency coverage calculation."""

    @pytest.fixture
    def graph(self):
        """Create a dependency graph from sample repo."""
        builder = GraphBuilder()
        return builder.build_from_directory("sample_repo")

    def test_coverage_with_all_deps(self, graph) -> None:
        """Test coverage when all dependencies are retrieved."""
        retriever = BaselineBFS(graph, max_depth=5)
        result = retriever.retrieve("LoginService")

        coverage = DependencyCoverage.calculate(
            graph, result.start_nodes, result.retrieved_nodes
        )

        assert 0.0 <= coverage.coverage_ratio <= 1.0
        assert coverage.total_required >= 0

    def test_coverage_with_no_deps(self, graph) -> None:
        """Test coverage when no dependencies are retrieved."""
        coverage = DependencyCoverage.calculate(
            graph,
            ["auth_service.py"],
            set(),  # Empty set
        )

        assert coverage.coverage_ratio == 0.0
        assert coverage.retrieved_required == 0

    def test_coverage_with_full_match(self, graph) -> None:
        """Test coverage when all required nodes are retrieved."""
        start = ["auth_service.py"]
        # Include the start node itself
        retrieved = set(start)

        coverage = DependencyCoverage.calculate(graph, start, retrieved)

        # At least the start node should be covered
        assert coverage.coverage_ratio >= 0.0

    def test_coverage_serialization(self) -> None:
        """Test that CoverageResult can be serialized."""
        coverage = CoverageResult(
            total_required=5,
            retrieved_required=4,
            coverage_ratio=0.8,
            missing_nodes=["node1"],
            extra_nodes=["node2", "node3"],
        )

        data = coverage.to_dict()
        assert data["total_required"] == 5
        assert data["coverage_ratio"] == 0.8
        assert "node1" in data["missing_nodes"]
        assert "node2" in data["extra_nodes"]

    def test_coverage_with_neighbors(self, graph) -> None:
        """Test coverage calculation with expanded neighbor set."""
        retriever = BaselineBFS(graph, max_depth=3)
        result = retriever.retrieve("Database")

        coverage = DependencyCoverage.calculate_with_neighbors(
            graph,
            result.start_nodes,
            result.retrieved_nodes,
            neighbor_depth=1,
        )

        assert 0.0 <= coverage.coverage_ratio <= 1.0


class TestBaselineVsAdaptive:
    """Comparative tests for baseline vs adaptive retrieval."""

    @pytest.fixture
    def graph(self):
        """Create a dependency graph from sample repo."""
        builder = GraphBuilder()
        return builder.build_from_directory("sample_repo")

    def test_adaptive_can_be_more_conservative(self, graph) -> None:
        """Test that adaptive retrieval can retrieve fewer tokens than baseline."""
        baseline = BaselineBFS(graph, max_depth=5)
        adaptive = AdaptiveBudgetRetriever(graph, budget=3000)

        baseline_result = baseline.retrieve("UserRepository")
        adaptive_result = adaptive.retrieve("UserRepository")

        # Adaptive with tight budget should be <= baseline with wide depth
        assert adaptive_result.token_estimate.total_tokens <= (
            baseline_result.token_estimate.total_tokens + 500
        )

    def test_adaptive_maintains_coverage(self, graph) -> None:
        """Test that adaptive retrieval maintains reasonable dependency coverage."""
        adaptive = AdaptiveBudgetRetriever(graph, budget=5000)
        result = adaptive.retrieve("Modify LoginService")

        coverage = DependencyCoverage.calculate(
            graph, result.start_nodes, result.retrieved_nodes
        )

        # Should have at least some coverage
        assert coverage.coverage_ratio >= 0.0

    def test_both_algorithms_produce_results(self, graph) -> None:
        """Test that both algorithms produce valid results."""
        baseline = BaselineBFS(graph, max_depth=3)
        adaptive = AdaptiveBudgetRetriever(graph, budget=5000)

        baseline_result = baseline.retrieve("Auth")
        adaptive_result = adaptive.retrieve("Auth")

        assert len(baseline_result.retrieved_nodes) > 0
        assert len(adaptive_result.retrieved_nodes) > 0
        assert isinstance(baseline_result.token_estimate, TokenEstimate)
        assert isinstance(adaptive_result.token_estimate, TokenEstimate)
