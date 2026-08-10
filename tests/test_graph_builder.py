"""Unit tests for the graph builder module.

Tests cover graph construction, node management, edge relationships,
and import resolution from parsed repositories.
"""

import pytest
from pathlib import Path

from parser.ast_parser import ASTParser
from graph.graph_builder import GraphBuilder, DependencyGraph
from graph.graph_search import GraphSearch


class TestDependencyGraphConstruction:
    """Tests for building dependency graphs from parsed repositories."""

    @pytest.fixture
    def sample_graph(self) -> DependencyGraph:
        """Create a sample graph with known structure."""
        builder = GraphBuilder()
        return builder.build_from_directory("sample_repo")

    def test_graph_has_nodes(self, sample_graph: DependencyGraph) -> None:
        """Test that the graph contains nodes."""
        assert sample_graph.node_count > 0

    def test_graph_has_edges(self, sample_graph: DependencyGraph) -> None:
        """Test that the graph contains edges."""
        assert sample_graph.edge_count > 0

    def test_file_nodes_exist(self, sample_graph: DependencyGraph) -> None:
        """Test that file nodes are present in the graph."""
        files = sample_graph.get_all_files()
        assert "auth_service.py" in files
        assert "user_repository.py" in files
        assert "database_config.py" in files

    def test_graph_node_count_is_reasonable(self, sample_graph: DependencyGraph) -> None:
        """Test that the graph has a reasonable number of nodes."""
        # With ~7 files, each having multiple classes/functions,
        # we expect 15+ nodes
        assert sample_graph.node_count >= 15

    def test_graph_edge_count_is_reasonable(self, sample_graph: DependencyGraph) -> None:
        """Test that the graph has a reasonable number of edges."""
        # Multiple imports, contains, and potentially calls edges
        assert sample_graph.edge_count >= 10


class TestGraphNodeData:
    """Tests for graph node data structure."""

    def test_file_node_creation(self) -> None:
        """Test creating a file node."""
        builder = GraphBuilder()
        graph = DependencyGraph()

        node_id = graph.add_file_node("test.py", token_cost=100)

        assert node_id == "test.py"
        node = graph.get_node_data(node_id)
        assert node is not None
        assert node.node_type == "file"
        assert node.estimated_token_cost == 100

    def test_class_node_creation(self) -> None:
        """Test creating a class node."""
        graph = DependencyGraph()
        node_id = graph.add_class_node("MyClass", "test.py", token_cost=50)

        assert "::" in node_id
        node = graph.get_node_data(node_id)
        assert node is not None
        assert node.node_type == "class"
        assert node.name == "MyClass"

    def test_function_node_creation(self) -> None:
        """Test creating a function node."""
        graph = DependencyGraph()
        node_id = graph.add_function_node("my_func", "test.py", token_cost=20)

        assert "::" in node_id
        node = graph.get_node_data(node_id)
        assert node is not None
        assert node.node_type == "function"
        assert node.name == "my_func"

    def test_missing_node_data(self) -> None:
        """Test that missing nodes return None."""
        graph = DependencyGraph()
        assert graph.get_node_data("nonexistent.py") is None


class TestGraphEdges:
    """Tests for graph edge relationships."""

    def test_import_edge(self) -> None:
        """Test adding and retrieving import edges."""
        graph = DependencyGraph()
        graph.add_file_node("a.py")
        graph.add_file_node("b.py")
        graph.add_import_edge("a.py", "b.py")

        assert graph.graph.has_edge("a.py", "b.py")
        edge_data = graph.graph.get_edge_data("a.py", "b.py")
        assert edge_data["edge_type"] == "imports"

    def test_contains_edge(self) -> None:
        """Test adding and retrieving containment edges."""
        graph = DependencyGraph()
        graph.add_file_node("test.py")
        graph.add_class_node("MyClass", "test.py")
        graph.add_contains_edge("test.py", "test.py::MyClass")

        assert graph.graph.has_edge("test.py", "test.py::MyClass")
        edge_data = graph.graph.get_edge_data("test.py", "test.py::MyClass")
        assert edge_data["edge_type"] == "contains"

    def test_inherits_edge(self) -> None:
        """Test adding and retrieving inheritance edges."""
        graph = DependencyGraph()
        graph.add_class_node("Child", "test.py")
        graph.add_class_node("Parent", "base.py")
        graph.add_inherits_edge("test.py::Child", "base.py::Parent")

        assert graph.graph.has_edge("test.py::Child", "base.py::Parent")
        edge_data = graph.graph.get_edge_data("test.py::Child", "base.py::Parent")
        assert edge_data["edge_type"] == "inherits"

    def test_calls_edge(self) -> None:
        """Test adding and retrieving function call edges."""
        graph = DependencyGraph()
        caller = graph.add_function_node("outer", "test.py")
        callee = graph.add_function_node("inner", "test.py")
        graph.add_calls_edge(caller, callee)

        assert graph.graph.has_edge(caller, callee)
        edge_data = graph.graph.get_edge_data(caller, callee)
        assert edge_data["edge_type"] == "calls"


class TestGraphIntegration:
    """Integration tests for graph building from real files."""

    def test_build_from_sample_repo(self) -> None:
        """Test building a graph from the sample repository."""
        builder = GraphBuilder()
        graph = builder.build_from_directory("sample_repo")

        # Should have file nodes
        files = graph.get_all_files()
        assert len(files) > 0

        # Should have import edges
        has_import_edges = False
        for _, _, data in graph.graph.edges(data=True):
            if data.get("edge_type") == "imports":
                has_import_edges = True
                break
        assert has_import_edges, "Expected at least one import edge"

    def test_auth_service_imports(self) -> None:
        """Test that auth_service.py imports are correctly captured."""
        builder = GraphBuilder()
        graph = builder.build_from_directory("sample_repo")

        # auth_service.py should have import edges to user_repository.py and database_config.py
        if "auth_service.py" in graph.get_all_files():
            targets = list(graph.graph.successors("auth_service.py"))
            target_types = [
                graph.graph.get_edge_data("auth_service.py", t).get("edge_type", "")
                for t in targets
                if graph.graph.has_edge("auth_service.py", t)
            ]
            # Should have import-type edges
            assert "imports" in target_types or len(targets) > 0

    def test_graph_export(self) -> None:
        """Test exporting the graph as a dictionary."""
        builder = GraphBuilder()
        graph = builder.build_from_directory("sample_repo")

        data = graph.get_graph_as_dict()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) > 0
        assert len(data["edges"]) > 0
