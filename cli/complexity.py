"""Complexity analysis for retrieval algorithms.

Reports time and space complexity of all major operations.
This is used for the complexity analysis section of the research paper.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ComplexityInfo:
    """Time and space complexity information for an algorithm.

    Attributes:
        algorithm: Name of the algorithm.
        time_complexity: Asymptotic time complexity (Big-O).
        space_complexity: Asymptotic space complexity (Big-O).
        description: Human-readable explanation.
        notes: Additional implementation notes.
    """
    algorithm: str
    time_complexity: str
    space_complexity: str
    description: str
    notes: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm,
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
            "description": self.description,
            "notes": self.notes,
        }


def get_complexity_analysis() -> list[ComplexityInfo]:
    """Return complexity analysis for all major algorithms.

    Returns:
        List of ComplexityInfo objects.
    """
    return [
        ComplexityInfo(
            algorithm="Repository Parsing (AST)",
            time_complexity="O(V + E)",
            space_complexity="O(V)",
            description=(
                "Parsing a repository visits each AST node once. V is the number "
                "of AST nodes (proportional to total source lines), and E is the "
                "number of edges (imports, calls, etc.) extracted."
            ),
            notes=[
                "Uses Python's ast.parse() which is O(source_length)",
                "Class/function extraction walks each function body once",
                "Per-file parsing: O(file_size), total: O(sum of file_size)",
            ],
        ),
        ComplexityInfo(
            algorithm="Graph Construction",
            time_complexity="O(V + E)",
            space_complexity="O(V + E)",
            description=(
                "Building the dependency graph adds each file, class, and function "
                "as a node (V), and each import, containment, inheritance, and call "
                "relationship as an edge (E)."
            ),
            notes=[
                "Phase 1: Add file nodes - O(F) where F = file count",
                "Phase 2: Add class/function nodes - O(C + N) where C = classes, N = functions",
                "Phase 3-4: Resolve imports - O(F * I) where I = imports per file",
                "Phase 5: Add inheritance edges - O(C)",
                "Phase 6: Resolve call edges - O(N * calls_per_function)",
            ],
        ),
        ComplexityInfo(
            algorithm="Baseline BFS Traversal",
            time_complexity="O(V + E)",
            space_complexity="O(V)",
            description=(
                "BFS visits each node at most once and examines each edge at most "
                "twice (once from each endpoint). The queue and visited set use "
                "O(V) space."
            ),
            notes=[
                "Level-by-level traversal up to max_depth",
                "Each node added to queue once",
                "Edge examination is bounded by total edges",
            ],
        ),
        ComplexityInfo(
            algorithm="Adaptive Budget-Constrained Traversal",
            time_complexity="O((V + E) log V)",
            space_complexity="O(V + E)",
            description=(
                "Worst case: all nodes are examined. Each candidate is scored using "
                "a priority queue with O(log V) operations. Graph centrality metrics "
                "(PageRank, betweenness) are precomputed in O(V + E). The scoring "
                "itself is O(1) per node."
            ),
            notes=[
                "Priority queue operations: O(log V) per enqueue/dequeue",
                "Each node scored once: O(V) total scoring",
                "Budget check per node: O(1)",
                "Graph metrics precomputation: O(V + E)",
                "In practice, much fewer than V nodes are processed due to budget",
                "Actual complexity is O(k * log V) where k <= V is nodes processed",
            ],
        ),
        ComplexityInfo(
            algorithm="Keyword Query Matching",
            time_complexity="O(V * L * T)",
            space_complexity="O(V)",
            description=(
                "For each node (V), splits the query into tokens (T) and checks "
                "each token against the node name and file path (L = average length)."
            ),
            notes=[
                "Supports camelCase and snake_case splitting",
                "Case-insensitive matching",
                "Exact word matches score higher than substring matches",
            ],
        ),
        ComplexityInfo(
            algorithm="Context Assembly (ContextBuilder)",
            time_complexity="O(N log N + S)",
            space_complexity="O(S)",
            description=(
                "Sorting N retrieved nodes by source order takes O(N log N). "
                "Merging source snippets takes O(S) where S is total source size."
            ),
            notes=[
                "Sort by (file_path, start_line)",
                "Remove duplicates: O(N) with hash set",
                "Token estimation: O(S) for fast mode, O(S * log S) for tiktoken",
            ],
        ),
        ComplexityInfo(
            algorithm="Token Estimation",
            time_complexity="O(N)",
            space_complexity="O(N)",
            description=(
                "Each node is processed once to compute its token cost. "
                "File-level deduplication uses a hash set for O(1) lookups."
            ),
            notes=[
                "Fast mode: chars / 4, O(1) per node",
                "Accurate mode: tiktoken encoding, O(source_size) total",
                "File deduplication: O(N) with hash set",
            ],
        ),
        ComplexityInfo(
            algorithm="Dependency Coverage Calculation",
            time_complexity="O(V + E)",
            space_complexity="O(V)",
            description=(
                "Finding required dependencies uses BFS from start nodes, "
                "visiting each node and edge at most once."
            ),
            notes=[
                "BFS over import edges only",
                "Set operations for missing/extra: O(V)",
            ],
        ),
    ]


def print_complexity_report() -> None:
    """Print a formatted complexity analysis report."""
    print("=" * 80)
    print("COMPLEXITY ANALYSIS")
    print("=" * 80)

    for info in get_complexity_analysis():
        print(f"\n[{info.algorithm}]")
        print(f"  Time:    {info.time_complexity}")
        print(f"  Space:   {info.space_complexity}")
        print(f"  {info.description}")
        for note in info.notes:
            print(f"    - {note}")

    print("\n" + "=" * 80)
