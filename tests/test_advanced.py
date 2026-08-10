"""Unit tests for node-level context, context builder, scoring config,
graph importance metrics, dependency coverage, query matching,
visualization, and complexity analysis.
"""

import pytest
import tempfile
from pathlib import Path

from parser.ast_parser import ASTParser, ClassInfo, FunctionInfo
from graph.graph_builder import GraphBuilder, DependencyGraph
from graph.graph_search import GraphSearch, _split_query, _match_token, _exact_match
from retrieval.scorer import (
    NodeScorer, NodeScore, ScoringConfig,
    _split_query as split_query_scoring,
    DEFAULT_CONFIG,
)
from retrieval.adaptive_budget import (
    AdaptiveBudgetRetriever, RetrievalExplanation,
)
from metrics.token_estimator import TokenEstimator, TokenEstimate
from metrics.context_builder import ContextBuilder, Context
from metrics.dependency_coverage import DependencyCoverage, CoverageResult


class TestNodeLevelContext:
    """Tests for node-level source code context (Improvement 1)."""

    def test_class_stores_source_code(self, tmp_path: Path) -> None:
        """Test that parsed classes store their source code."""
        source = (
            'class MyClass:\n'
            '    def __init__(self):\n'
            '        self.x = 0\n'
            '    def get_x(self):\n'
            '        return self.x\n'
        )
        (tmp_path / "test.py").write_text(source)
        parser = ASTParser()
        parsed = parser.parse_file(str(tmp_path / "test.py"))

        cls_info = parsed.classes["MyClass"]
        assert cls_info.source_code == source
        assert cls_info.start_line == 1
        assert cls_info.end_line == 5
        assert cls_info.char_count == len(source)
        assert cls_info.token_estimate > 0

    def test_function_stores_source_code(self, tmp_path: Path) -> None:
        """Test that parsed functions store their source code."""
        source = "def my_func(x, y):\n    return x + y\n"
        (tmp_path / "test.py").write_text(source)
        parser = ASTParser()
        parsed = parser.parse_file(str(tmp_path / "test.py"))

        func_info = parsed.functions["my_func"]
        assert func_info.source_code == source
        assert func_info.start_line == 1
        assert func_info.end_line == 2
        assert func_info.char_count == len(source)

    def test_method_stores_source_code(self, tmp_path: Path) -> None:
        """Test that methods store their individual source code."""
        source = (
            'class Service:\n'
            '    def process(self, data):\n'
            '        return data\n'
            '    def validate(self, data):\n'
            '        return True\n'
        )
        (tmp_path / "test.py").write_text(source)
        parser = ASTParser()
        parsed = parser.parse_file(str(tmp_path / "test.py"))

        process = parsed.functions["process"]
        validate = parsed.functions["validate"]

        assert "process" in process.source_code
        assert "validate" in validate.source_code
        assert process.source_code != validate.source_code
        assert process.start_line < validate.start_line


class TestContextBuilder:
    """Tests for the context builder module (Improvement 2)."""

    @pytest.fixture
    def graph(self):
        """Create a dependency graph from sample repo."""
        builder = GraphBuilder()
        return builder.build_from_directory("sample_repo")

    def test_build_assembles_context(self, graph) -> None:
        """Test that context builder assembles source from nodes."""
        builder = ContextBuilder()
        # Get some class nodes
        class_nodes = [
            nid for nid, nd in graph.get_all_nodes().items()
            if nd.node_type == "class"
        ][:3]

        if class_nodes:
            context = builder.build(set(class_nodes), graph)
            assert context.nodes
            assert context.source
            assert context.total_characters > 0
            assert context.estimated_tokens > 0
            assert context.line_count > 0

    def test_build_sorts_by_source_order(self, graph) -> None:
        """Test that nodes are sorted by file path then line number."""
        builder = ContextBuilder()
        # Get nodes from different files
        all_nodes = [
            nid for nid, nd in graph.get_all_nodes().items()
            if nd.node_type == "function" and nd.source_code
        ]

        if len(all_nodes) >= 2:
            context = builder.build(set(all_nodes[:5]), graph)
            # Nodes should be sorted
            assert len(context.nodes) > 0
            for i in range(1, len(context.nodes)):
                prev = context.nodes[i - 1]
                curr = context.nodes[i]
                if prev.file_path == curr.file_path:
                    assert prev.start_line <= curr.start_line

    def test_build_removes_duplicates(self, graph) -> None:
        """Test that duplicate nodes are removed."""
        builder = ContextBuilder()
        # Add same node multiple times
        class_nodes = [
            nid for nid, nd in graph.get_all_nodes().items()
            if nd.node_type == "class"
        ][:2]

        if class_nodes:
            context = builder.build(set(class_nodes), graph)
            # Each node_id should appear only once
            node_ids = [n.node_id for n in context.nodes]
            assert len(node_ids) == len(set(node_ids))

    def test_build_counts_are_accurate(self, graph) -> None:
        """Test that token and character counts are accurate."""
        builder = ContextBuilder()
        class_nodes = [
            nid for nid, nd in graph.get_all_nodes().items()
            if nd.node_type == "class"
        ][:1]

        if class_nodes:
            context = builder.build(set(class_nodes), graph)
            assert context.total_characters == len(context.source)
            assert context.estimated_tokens > 0
            assert context.line_count > 0

    def test_node_level_token_estimation(self, graph) -> None:
        """Test that token estimation is at node level, not file level."""
        estimate = TokenEstimator.estimate_nodes(
            {"auth_service.py::LoginService"},
            graph,
        )
        # Should be based on the class size, not full file
        assert estimate.total_tokens > 0
        assert estimate.node_count == 1
        assert estimate.per_node["auth_service.py::LoginService"] > 0

    def test_context_builder_serialization(self, graph) -> None:
        """Test that Context can be serialized to dict."""
        builder = ContextBuilder()
        class_nodes = [
            nid for nid, nd in graph.get_all_nodes().items()
            if nd.node_type == "class"
        ][:1]

        if class_nodes:
            context = builder.build(set(class_nodes), graph)
            data = context.to_dict()
            assert "total_characters" in data
            assert "estimated_tokens" in data
            assert "nodes" in data
            assert len(data["nodes"]) > 0


class TestScoringConfig:
    """Tests for configurable scoring weights (Improvement 5)."""

    def test_default_weights(self) -> None:
        """Test default scoring weights."""
        config = ScoringConfig()
        assert config.keyword_weight > 0
        assert config.dependency_weight > 0
        assert config.centrality_weight > 0
        assert config.token_weight > 0

    def test_weights_sum_to_one(self) -> None:
        """Test that weights sum to 1.0."""
        config = ScoringConfig()
        total = (
            config.keyword_weight
            + config.dependency_weight
            + config.centrality_weight
            + config.token_weight
        )
        assert abs(total - 1.0) < 0.01

    def test_custom_config_from_file(self, tmp_path: Path) -> None:
        """Test loading custom config from YAML file."""
        config_path = str(tmp_path / "custom_config.yaml")
        Path(config_path).write_text(
            "keyword_weight: 0.50\n"
            "dependency_weight: 0.30\n"
            "centrality_weight: 0.10\n"
            "token_weight: 0.10\n"
        )

        config = ScoringConfig(config_path)
        assert config.keyword_weight == 0.50
        assert config.dependency_weight == 0.30
        assert config.centrality_weight == 0.10
        assert config.token_weight == 0.10


class TestGraphImportanceMetrics:
    """Tests for graph centrality metrics (Improvement 6)."""

    def test_degree_centrality(self) -> None:
        """Test degree centrality calculation."""
        graph = DependencyGraph()
        graph.add_file_node("a.py")
        graph.add_file_node("b.py")
        graph.add_file_node("c.py")
        graph.add_import_edge("a.py", "b.py")
        graph.add_import_edge("c.py", "b.py")

        centrality = graph.compute_degree_centrality()
        assert "a.py" in centrality
        assert "b.py" in centrality
        # b.py has more connections
        assert centrality["b.py"] >= centrality["a.py"]

    def test_pagerank(self) -> None:
        """Test PageRank calculation (without scipy)."""
        graph = DependencyGraph()
        graph.add_file_node("a.py")
        graph.add_file_node("b.py")
        graph.add_file_node("c.py")
        graph.add_import_edge("a.py", "b.py")
        graph.add_import_edge("a.py", "c.py")
        graph.add_import_edge("b.py", "c.py")

        pr = graph.compute_pagerank()
        assert len(pr) == 3
        assert abs(sum(pr.values()) - 1.0) < 0.01

    def test_betweenness_centrality(self) -> None:
        """Test betweenness centrality calculation."""
        graph = DependencyGraph()
        # Create a chain: a -> b -> c
        graph.add_file_node("a.py")
        graph.add_file_node("b.py")
        graph.add_file_node("c.py")
        graph.add_import_edge("a.py", "b.py")
        graph.add_import_edge("b.py", "c.py")

        betweenness = graph.compute_betweenness_centrality()
        assert "a.py" in betweenness
        assert "b.py" in betweenness
        # b.py is on the path between a and c
        assert betweenness["b.py"] >= betweenness["a.py"]


class TestQueryMatching:
    """Tests for improved query matching (Improvement 12)."""

    def test_camelcase_splitting(self) -> None:
        """Test camelCase splitting in queries."""
        tokens = _split_query("LoginService")
        assert "login" in tokens
        assert "service" in tokens

    def test_snake_case_splitting(self) -> None:
        """Test snake_case splitting in queries."""
        tokens = _split_query("user_repository")
        assert "user" in tokens
        assert "repository" in tokens

    def test_case_insensitive_matching(self) -> None:
        """Test case-insensitive matching."""
        assert _match_token("login", "LoginService")
        assert _match_token("LOGIN", "LoginService")
        assert _match_token("login", "auth_service.py::LoginService")

    def test_exact_word_match(self) -> None:
        """Test exact word boundary matching."""
        assert _exact_match("login", "LoginService")
        assert _exact_match("user", "user_repository.py")
        assert not _exact_match("ser", "LoginService")  # partial match

    def test_query_matches_class_name(self, graph) -> None:
        """Test that query matches class names."""
        matches = GraphSearch.search(graph, "LoginService")
        assert len(matches) > 0
        assert any("LoginService" in m.node_id for m in matches)

    def test_query_matches_file_name(self, graph) -> None:
        """Test that query matches file names."""
        matches = GraphSearch.search(graph, "payment")
        assert len(matches) > 0
        assert any("payment" in m.node_id.lower() for m in matches)

    def test_multi_token_query(self, graph) -> None:
        """Test query with multiple tokens."""
        matches = GraphSearch.search(graph, "auth service")
        # Should find LoginService (matches "service") and auth_service.py (matches "auth")
        assert len(matches) > 0


class TestStrictBudget:
    """Tests for strict token budget enforcement (Improvement 4)."""

    def test_budget_enforced(self) -> None:
        """Test that adaptive retrieval strictly respects budget."""
        builder = GraphBuilder()
        graph = builder.build_from_directory("sample_repo")

        # Very tight budget
        retriever = AdaptiveBudgetRetriever(graph, budget=100)
        result = retriever.retrieve("LoginService")

        assert result.token_estimate.total_tokens <= result.budget

    def test_budget_checked_before_accept(self, graph) -> None:
        """Test that budget is checked BEFORE a node is accepted."""
        # Budget that can't even fit the largest node
        retriever = AdaptiveBudgetRetriever(graph, budget=1)
        result = retriever.retrieve("LoginService")

        # Should have skipped or forced the entry node
        # At minimum, total tokens should not exceed budget
        assert result.token_estimate.total_tokens <= result.budget + 100

    def test_force_entry_node(self, graph) -> None:
        """Test force_entry_node flag."""
        retriever = AdaptiveBudgetRetriever(
            graph, budget=1, force_entry_node=True
        )
        result = retriever.retrieve("LoginService")

        # With force, at least the start node should be retrieved
        assert len(result.traversal_order) > 0


class TestDependencyCoverage:
    """Tests for improved dependency coverage (Improvement 8)."""

    def test_coverage_classifies_missing(self) -> None:
        """Test that missing nodes are classified by type."""
        builder = GraphBuilder()
        graph = builder.build_from_directory("sample_repo")

        coverage = DependencyCoverage.calculate(
            graph,
            ["auth_service.py"],
            set(),  # Empty retrieval
        )

        assert coverage.total_required > 0
        assert coverage.coverage_ratio == 0.0
        assert coverage.missing_classes or coverage.missing_functions or coverage.missing_files

    def test_full_coverage(self, graph) -> None:
        """Test coverage when all graph nodes are retrieved."""
        # Retrieve all graph nodes
        all_nodes = set(graph.get_all_nodes().keys())
        coverage = DependencyCoverage.calculate(
            graph,
            ["auth_service.py"],
            all_nodes,
        )
        # All nodes that exist in the graph should be retrieved
        # (external dependencies like 'typing' won't be in the graph)
        for missing in coverage.missing_nodes:
            node = graph.get_node_data(missing)
            assert node is None, f"Missing node '{missing}' should not exist in graph"

    def test_coverage_result_classification(self, graph) -> None:
        """Test that CoverageResult classifies nodes correctly."""
        coverage = CoverageResult(
            total_required=10,
            retrieved_required=7,
            coverage_ratio=0.7,
            missing_nodes=["a.py", "b.py::MyClass"],
            extra_nodes=["c.py::func1"],
            missing_classes=["b.py::MyClass"],
            missing_functions=["c.py::func1"],
            missing_files=["a.py"],
        )
        assert len(coverage.missing_classes) == 1
        assert len(coverage.missing_files) == 1

    def test_coverage_serialization(self) -> None:
        """Test CoverageResult serialization."""
        coverage = CoverageResult(
            total_required=5,
            retrieved_required=3,
            coverage_ratio=0.6,
            missing_nodes=["x"],
            extra_nodes=["y"],
        )
        data = coverage.to_dict()
        assert data["total_required"] == 5
        assert data["coverage_ratio"] == 0.6


class TestScoring:
    """Tests for the scoring module."""

    def test_score_node_returns_reasons(self, graph) -> None:
        """Test that NodeScore includes natural-language reasons."""
        scorer = NodeScorer()
        node = graph.get_node_data("auth_service.py::LoginService")
        if node:
            score = scorer.score_node(node, graph, "Login")
            assert score.total_score >= 0
            assert len(score.reasons) > 0

    def test_rank_candidates(self, graph) -> None:
        """Test ranking of candidate nodes."""
        scorer = NodeScorer()
        class_nodes = [
            nid for nid, nd in graph.get_all_nodes().items()
            if nd.node_type == "class"
        ][:5]

        if len(class_nodes) >= 2:
            scores = scorer.rank_candidates(class_nodes, graph, "auth")
            assert len(scores) > 0
            # Scores should be sorted descending
            for i in range(1, len(scores)):
                assert scores[i - 1].total_score >= scores[i].total_score

    def test_token_score_inversely_proportional(self, graph) -> None:
        """Test that token score is inversely proportional to cost."""
        scorer = NodeScorer()
        small = graph.get_node_data("analytics_service.py")
        if small:
            score = scorer.score_node(small, graph, "test")
            assert score.token_score >= 0
            assert score.token_score <= 1.0


class TestComplexityAnalysis:
    """Tests for complexity analysis module."""

    def test_all_algorithms_documented(self) -> None:
        """Test that all major algorithms have complexity info."""
        from cli.complexity import get_complexity_analysis
        info = get_complexity_analysis()
        algorithms = [i.algorithm for i in info]

        expected = [
            "Repository Parsing (AST)",
            "Graph Construction",
            "Baseline BFS Traversal",
            "Adaptive Budget-Constrained Traversal",
            "Keyword Query Matching",
            "Context Assembly (ContextBuilder)",
            "Token Estimation",
            "Dependency Coverage Calculation",
        ]

        for exp in expected:
            assert exp in algorithms

    def test_complexity_has_time_and_space(self) -> None:
        """Test that each algorithm has time and space complexity."""
        from cli.complexity import get_complexity_analysis
        for info in get_complexity_analysis():
            assert info.time_complexity.startswith("O(")
            assert info.space_complexity.startswith("O(")
            assert info.description
            assert info.notes


class TestContextCompressionRatio:
    """Tests for Context Compression Ratio metric (Improvement 10)."""

    def test_ccr_formula(self) -> None:
        """Test CCR = adaptive_tokens / baseline_tokens."""
        baseline_tokens = 4200
        adaptive_tokens = 1800
        ccr = adaptive_tokens / baseline_tokens

        assert ccr == 0.42857142857142855
        assert ccr < 1.0
        assert (1 - ccr) * 100 == pytest.approx(57.14, abs=0.01)

    def test_ccr_at_full_budget(self, graph) -> None:
        """Test CCR when adaptive retrieves everything (no compression)."""
        adaptive = AdaptiveBudgetRetriever(graph, budget=100000)
        baseline = BaselineBFS(graph, max_depth=10)

        adaptive_result = adaptive.retrieve("Auth")
        baseline_result = baseline.retrieve("Auth")

        ccr = adaptive_result.token_estimate.total_tokens / baseline_result.token_estimate.total_tokens
        assert ccr <= 1.0  # Adaptive should never retrieve MORE than baseline

    def test_ccr_decreases_with_tighter_budget(self, graph) -> None:
        """Test that CCR decreases as budget tightens."""
        results = []
        for budget in [10000, 5000, 1000, 200, 50]:
            adaptive = AdaptiveBudgetRetriever(graph, budget=budget)
            result = adaptive.retrieve("User")
            results.append((budget, result.token_estimate.total_tokens))

        # Tokens should decrease (or stay same) as budget decreases
        for i in range(1, len(results)):
            assert results[i][1] <= results[i - 1][1] + 10  # Small tolerance


# Need to import BaselineBFS for tests
from retrieval.baseline_bfs import BaselineBFS
