"""Dependency graph builder for Python repositories.

Constructs a directed graph from parsed Python files, capturing
import dependencies, class inheritance, file-to-class containment,
and function call relationships.

Each graph node stores source-level metadata:
- source_code snippet
- start_line / end_line
- character count
- estimated token count

This enables precise, node-level token estimation instead of
estimating entire file sizes for partial retrievals.
"""

from __future__ import annotations

import networkx as nx
from dataclasses import dataclass, field
from typing import Any

from parser.ast_parser import ASTParser, ParsedRepository, ClassInfo, FunctionInfo


@dataclass
class GraphNode:
    """A node in the repository dependency graph.

    Attributes:
        node_id: Unique identifier (file path or qualified class/function name).
        name: Display name of the node.
        node_type: One of 'file', 'class', or 'function'.
        file_path: Source file path.
        source_code: Source code snippet for this node.
        start_line: Line number where this node's source starts (1-indexed).
        end_line: Line number where this node's source ends (1-indexed).
        char_count: Character count of the source snippet.
        estimated_token_cost: Approximate token count for LLM context.
        dependencies: Set of node IDs this node depends on.
    """
    node_id: str
    name: str
    node_type: str  # 'file', 'class', 'function'
    file_path: str
    source_code: str = ""
    start_line: int = 0
    end_line: int = 0
    char_count: int = 0
    estimated_token_cost: int = 0
    dependencies: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type,
            "file_path": self.file_path,
            "source_code": self.source_code[:200] + "..." if len(self.source_code) > 200 else self.source_code,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "char_count": self.char_count,
            "estimated_token_cost": self.estimated_token_cost,
            "dependencies": sorted(self.dependencies),
        }


class DependencyGraph:
    """Repository dependency graph built using NetworkX.

    Provides a directed graph where nodes represent files, classes,
    and functions, and edges represent imports, containment, inheritance,
    and call relationships.

    Each node carries source-level context (source_code, line range,
    char_count, token_estimate) for accurate, node-level token estimation.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._node_data: dict[str, GraphNode] = {}

    @property
    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return self.graph.number_of_edges()

    def add_file_node(self, file_path: str, token_cost: int = 0) -> str:
        """Add a file node to the graph.

        Args:
            file_path: Path to the Python file.
            token_cost: Estimated token cost for this file.

        Returns:
            The node ID of the added file.
        """
        node_id = file_path
        self.graph.add_node(node_id, node_type="file", file_path=file_path)
        self._node_data[node_id] = GraphNode(
            node_id=node_id,
            name=file_path,
            node_type="file",
            file_path=file_path,
            estimated_token_cost=token_cost,
        )
        return node_id

    def add_class_node(
        self,
        class_name: str,
        file_path: str,
        token_cost: int = 0,
        source_code: str = "",
        start_line: int = 0,
        end_line: int = 0,
        char_count: int = 0,
    ) -> str:
        """Add a class node to the graph.

        Args:
            class_name: Name of the class.
            file_path: Source file path.
            token_cost: Estimated token cost.
            source_code: Source code snippet of the class.
            start_line: Start line of the class definition.
            end_line: End line of the class definition.
            char_count: Character count of the source code.

        Returns:
            The node ID of the added class.
        """
        node_id = f"{file_path}::{class_name}"
        self.graph.add_node(node_id, node_type="class", file_path=file_path)
        self._node_data[node_id] = GraphNode(
            node_id=node_id,
            name=class_name,
            node_type="class",
            file_path=file_path,
            estimated_token_cost=token_cost,
            source_code=source_code,
            start_line=start_line,
            end_line=end_line,
            char_count=char_count,
        )
        return node_id

    def add_function_node(
        self,
        func_name: str,
        file_path: str,
        token_cost: int = 0,
        source_code: str = "",
        start_line: int = 0,
        end_line: int = 0,
        char_count: int = 0,
    ) -> str:
        """Add a function node to the graph.

        Args:
            func_name: Name of the function.
            file_path: Source file path.
            token_cost: Estimated token cost.
            source_code: Source code snippet of the function.
            start_line: Start line of the function definition.
            end_line: End line of the function definition.
            char_count: Character count of the source code.

        Returns:
            The node ID of the added function.
        """
        node_id = f"{file_path}::{func_name}"
        self.graph.add_node(node_id, node_type="function", file_path=file_path)
        self._node_data[node_id] = GraphNode(
            node_id=node_id,
            name=func_name,
            node_type="function",
            file_path=file_path,
            estimated_token_cost=token_cost,
            source_code=source_code,
            start_line=start_line,
            end_line=end_line,
            char_count=char_count,
        )
        return node_id

    def add_import_edge(self, source: str, target: str) -> None:
        """Add an import dependency edge.

        Args:
            source: File that imports (e.g., 'auth_service.py').
            target: Module being imported (e.g., 'user_repository').
        """
        self.graph.add_edge(source, target, edge_type="imports")

    def add_contains_edge(self, file_path: str, class_name_or_func: str) -> None:
        """Add a containment edge (file contains class/function).

        Args:
            file_path: The containing file.
            class_name_or_func: Qualified name of the class or function.
        """
        self.graph.add_edge(file_path, class_name_or_func, edge_type="contains")

    def add_inherits_edge(self, child: str, parent: str) -> None:
        """Add an inheritance edge.

        Args:
            child: Node ID of the child class.
            parent: Node ID or base class name of the parent.
        """
        self.graph.add_edge(child, parent, edge_type="inherits")

    def add_calls_edge(self, caller: str, callee: str) -> None:
        """Add a function call edge.

        Args:
            caller: Node ID of the calling function.
            callee: Node ID of the called function.
        """
        self.graph.add_edge(caller, callee, edge_type="calls")

    def get_node_data(self, node_id: str) -> GraphNode | None:
        """Retrieve the GraphNode data for a given node ID.

        Args:
            node_id: The node identifier.

        Returns:
            GraphNode if found, None otherwise.
        """
        return self._node_data.get(node_id)

    def get_all_nodes(self) -> dict[str, GraphNode]:
        """Return all graph nodes."""
        return dict(self._node_data)

    def get_import_targets(self, file_path: str) -> list[str]:
        """Get file-level import targets for a given file.

        Args:
            file_path: The source file path.

        Returns:
            List of imported module name strings.
        """
        return self.graph.successors(file_path)

    def get_class_import_deps(self, file_path: str) -> list[str]:
        """Get the file-level import dependencies resolved as full node IDs.

        Args:
            file_path: The source file path.

        Returns:
            List of node IDs that represent imported modules' classes/functions.
        """
        targets = []
        for succ in self.graph.successors(file_path):
            edge = self.graph.get_edge_data(file_path, succ)
            if edge and edge.get("edge_type") == "imports":
                targets.append(succ)
        return targets

    def get_inheritance_parents(self, class_node_id: str) -> list[str]:
        """Get parent class node IDs for inheritance edges.

        Args:
            class_node_id: Node ID of the class.

        Returns:
            List of parent class node IDs.
        """
        return list(self.graph.predecessors(class_node_id))

    def get_all_files(self) -> list[str]:
        """Return all file node IDs."""
        return [
            n for n, data in self.graph.nodes(data="node_type")
            if data == "file"
        ]

    def get_graph_as_dict(self) -> dict[str, Any]:
        """Export the full graph structure as a dictionary."""
        nodes = {}
        for node_id in self.graph.nodes():
            node_data = self._node_data.get(node_id)
            if node_data:
                nodes[node_id] = node_data.to_dict()
            else:
                nodes[node_id] = {
                    "node_id": node_id,
                    "node_type": self.graph.nodes[node_id].get("node_type", "unknown"),
                }

        edges = []
        for src, tgt, data in self.graph.edges(data=True):
            edges.append({
                "source": src,
                "target": tgt,
                "edge_type": data.get("edge_type", "unknown"),
            })

        return {"nodes": nodes, "edges": edges}

    # --- Graph importance metrics (Improvement 6) ---

    def compute_degree_centrality(self) -> dict[str, float]:
        """Compute degree centrality for all nodes.

        Degree centrality = (in_degree + out_degree) / (n - 1)

        Returns:
            Dict mapping node_id to centrality score (0.0 to 1.0).
        """
        n = self.node_count
        if n <= 1:
            return {nid: 0.0 for nid in self._node_data}
        centrality = nx.degree_centrality(self.graph)
        return centrality

    def compute_pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        """Compute PageRank for all nodes.

        PageRank measures the importance of a node based on the
        structure of incoming links.

        Uses a lightweight iterative implementation that does NOT
        require scipy, keeping the project lightweight.

        Args:
            alpha: Damping factor (default 0.85).

        Returns:
            Dict mapping node_id to PageRank score.
        """
        if self.node_count == 0:
            return {}
        try:
            return nx.pagerank(self.graph, alpha=alpha)
        except ImportError:
            # scipy not available — use lightweight iterative PageRank
            return self._pagerank_iterative(self.graph, alpha)
        except nx.PowerIterationFailedConvergence:
            return {nid: 1.0 / self.node_count for nid in self._node_data}

    @staticmethod
    def _pagerank_iterative(graph: Any, alpha: float = 0.85,
                            max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
        """Lightweight iterative PageRank without scipy.

        Args:
            graph: NetworkX DiGraph.
            alpha: Damping factor.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            Dict mapping node_id to PageRank score.
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        if n == 0:
            return {}

        pr = {node: 1.0 / n for node in nodes}

        for _ in range(max_iter):
            new_pr = {}
            for node in nodes:
                incoming_sum = 0.0
                for pred in graph.predecessors(node):
                    out_degree = max(graph.out_degree(pred), 1)
                    incoming_sum += pr[pred] / out_degree
                new_pr[node] = (1 - alpha) / n + alpha * incoming_sum

            diff = sum(abs(new_pr[n] - pr[n]) for n in nodes)
            pr = new_pr
            if diff < tol:
                break

        total = sum(pr.values())
        if total > 0:
            pr = {node: val / total for node, val in pr.items()}

        return pr

    def compute_betweenness_centrality(self) -> dict[str, float]:
        """Compute betweenness centrality for all nodes.

        Betweenness centrality measures how often a node appears
        on shortest paths between other nodes.

        Returns:
            Dict mapping node_id to betweenness score (0.0 to 1.0).
        """
        if self.node_count < 2:
            return {nid: 0.0 for nid in self._node_data}
        try:
            return nx.betweenness_centrality(self.graph)
        except nx.NetworkXError:
            return {nid: 0.0 for nid in self._node_data}


class GraphBuilder:
    """Builds a DependencyGraph from a parsed repository.

    Processes parsed files to extract:
    - File-level import relationships
    - Class definitions with inheritance
    - Function definitions with call targets
    - Containment edges (file -> class/function)

    Each class and function node stores its source code snippet,
    line range, character count, and token estimate.
    """

    def __init__(self, parser: ASTParser | None = None) -> None:
        """Initialize the graph builder.

        Args:
            parser: ASTParser instance. Creates a new one if not provided.
        """
        self.parser = parser or ASTParser()
        self.graph = DependencyGraph()

    def build_from_parsed(self, parsed_repo: ParsedRepository) -> DependencyGraph:
        """Build the dependency graph from a parsed repository.

        Args:
            parsed_repo: ParsedRepository from ASTParser.parse_directory().

        Returns:
            Fully constructed DependencyGraph with node-level context.
        """
        self.graph = DependencyGraph()

        # Phase 1: Add all file nodes with char_count from source
        for file_path, parsed_file in parsed_repo.files.items():
            self.graph.add_file_node(
                file_path,
                token_cost=parsed_file.source_length // 4,
            )
            # Set char_count on the file node
            file_node = self.graph.get_node_data(file_path)
            if file_node:
                file_node.char_count = parsed_file.source_length

        # Phase 2: Add class and function nodes with full context
        for file_path, parsed_file in parsed_repo.files.items():
            file_node = self.graph.get_node_data(file_path)
            if not file_node:
                continue

            for class_name, class_info in parsed_file.classes.items():
                class_node_id = self.graph.add_class_node(
                    class_name,
                    file_path,
                    token_cost=class_info.token_estimate,
                    source_code=class_info.source_code,
                    start_line=class_info.start_line,
                    end_line=class_info.end_line,
                    char_count=class_info.char_count,
                )
                self.graph.add_contains_edge(file_path, class_node_id)

            for func_name, func_info in parsed_file.functions.items():
                func_node_id = self.graph.add_function_node(
                    func_name,
                    file_path,
                    token_cost=func_info.token_estimate,
                    source_code=func_info.source_code,
                    start_line=func_info.start_line,
                    end_line=func_info.end_line,
                    char_count=func_info.char_count,
                )
                self.graph.add_contains_edge(file_path, func_node_id)

        # Phase 3: Add import edges and resolve cross-file dependencies
        for file_path, parsed_file in parsed_repo.files.items():
            for imported_module in parsed_file.module.imports:
                self.graph.add_import_edge(file_path, imported_module)

        # Phase 4: Resolve imports to actual nodes in other files
        self._resolve_imports(parsed_repo)

        # Phase 5: Add inheritance edges
        for file_path, parsed_file in parsed_repo.files.items():
            for class_name, class_info in parsed_file.classes.items():
                class_node_id = f"{file_path}::{class_name}"
                for base_name in class_info.base_classes:
                    self.graph.add_inherits_edge(class_node_id, base_name)

        # Phase 6: Add call edges between function nodes
        self._resolve_calls(parsed_repo)

        return self.graph

    def build_from_directory(self, directory_path: str) -> DependencyGraph:
        """Parse a directory and build the dependency graph.

        Args:
            directory_path: Path to the repository root directory.

        Returns:
            Fully constructed DependencyGraph.
        """
        parsed_repo = self.parser.parse_directory(directory_path)
        return self.build_from_parsed(parsed_repo)

    def _resolve_imports(self, parsed_repo: ParsedRepository) -> None:
        """Resolve import module names to actual class/function nodes.

        Maps imported module names to the corresponding file nodes
        and their classes/functions.
        """
        # Build a mapping from module name to file path
        module_to_file: dict[str, str] = {}
        for file_path in parsed_repo.files:
            module_name = file_path.replace("/", ".").replace(".py", "")
            module_to_file[module_name] = file_path
            # Also map by stem
            stem = file_path.split("/")[-1].replace(".py", "")
            module_to_file[stem] = file_path

        # For each file, resolve its imports to actual nodes
        for file_path, parsed_file in parsed_repo.files.items():
            for imported_module in parsed_file.module.imports:
                target_file = module_to_file.get(imported_module)
                if target_file and target_file in self.graph._node_data:
                    # Update the import target to be the file node
                    if self.graph.graph.has_edge(file_path, imported_module):
                        self.graph.graph.remove_edge(file_path, imported_module)
                    self.graph.add_import_edge(file_path, target_file)

    def _resolve_calls(self, parsed_repo: ParsedRepository) -> None:
        """Resolve function calls to actual function nodes in the graph."""
        # Build a mapping of function names to their node IDs
        func_name_to_node: dict[str, str] = {}
        for node_id, node_data in self.graph._node_data.items():
            if node_data.node_type == "function":
                func_name_to_node[node_data.name] = node_id

        # Map class names to files
        class_name_to_file: dict[str, str] = {}
        for file_path, parsed_file in parsed_repo.files.items():
            for class_name in parsed_file.classes:
                class_name_to_file[class_name] = file_path

        # For each function, resolve its calls
        for file_path, parsed_file in parsed_repo.files.items():
            for func_name, func_info in parsed_file.functions.items():
                caller_node_id = f"{file_path}::{func_name}"
                for called_name in func_info.calls:
                    callee_node_id = func_name_to_node.get(called_name)
                    if callee_node_id:
                        self.graph.add_calls_edge(caller_node_id, callee_node_id)
