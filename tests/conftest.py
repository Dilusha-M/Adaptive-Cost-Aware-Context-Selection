import pytest
from graph.graph_builder import GraphBuilder

@pytest.fixture
def graph():
    """Create a dependency graph from sample repo."""
    builder = GraphBuilder()
    return builder.build_from_directory("sample_repo")
