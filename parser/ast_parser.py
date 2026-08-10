"""AST-based parser for Python source files.

Parses Python repositories to extract modules, classes, functions,
imports, and inheritance relationships using Python's built-in AST module.
Each class and function node stores its source code snippet, line range,
character count, and estimated token count for accurate context assembly.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FunctionInfo:
    """Information about a parsed Python function or method.

    Attributes:
        name: Function or method name.
        file_path: Source file path.
        class_name: Parent class name (None for top-level functions).
        parameters: List of parameter names.
        return_annotation: Return type annotation string.
        child_classes: Nested class names defined inside this function.
        child_functions: Nested function names defined inside this function.
        calls: List of function call names found in the function body.
        start_line: Line number where the function definition starts.
        end_line: Line number where the function definition ends.
        source_code: The raw source code of this function.
        char_count: Character count of the source code.
        token_estimate: Estimated token count (chars / 4).
    """
    name: str
    file_path: str
    class_name: str | None = None
    parameters: list[str] = field(default_factory=list)
    return_annotation: str = ""
    child_classes: list[str] = field(default_factory=list)
    child_functions: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    source_code: str = ""
    char_count: int = 0
    token_estimate: int = 0

    @property
    def full_name(self) -> str:
        """Return the fully qualified name of this function."""
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name


@dataclass
class ClassInfo:
    """Information about a parsed Python class.

    Attributes:
        name: Class name.
        file_path: Source file path.
        base_classes: List of base class names.
        methods: Method names defined in this class.
        child_functions: Top-level function names inside the class body.
        start_line: Line number where the class definition starts.
        end_line: Line number where the class definition ends.
        source_code: The raw source code of this class (including all methods).
        char_count: Character count of the source code.
        token_estimate: Estimated token count (chars / 4).
    """
    name: str
    file_path: str
    base_classes: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    child_functions: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    source_code: str = ""
    char_count: int = 0
    token_estimate: int = 0

    @property
    def full_name(self) -> str:
        """Return the fully qualified name of this class."""
        return f"{self.file_path}::{self.name}"


@dataclass
class ModuleInfo:
    """Information about a parsed Python module (file).

    Attributes:
        name: Module name (file stem).
        file_path: Full file path.
        imports: List of imported module names (import X).
        from_imports: Dict mapping module name to list of imported names (from X import Y).
        classes: List of class names defined at module level.
        functions: List of function names defined at module level.
    """
    name: str
    file_path: str
    imports: list[str] = field(default_factory=list)
    from_imports: dict[str, list[str]] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)

    @property
    def full_path(self) -> str:
        """Return the full file path."""
        return self.file_path


@dataclass
class ParsedFile:
    """Complete parsed information for a single Python file.

    Attributes:
        module: Module-level metadata.
        classes: Dict mapping class name to ClassInfo.
        functions: Dict mapping function name to FunctionInfo.
        source_code: The full raw source code of the file.
        source_length: Character count of the full source.
    """
    module: ModuleInfo
    classes: dict[str, ClassInfo] = field(default_factory=dict)
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    source_code: str = ""
    source_length: int = 0


@dataclass
class ParsedRepository:
    """Complete parsed information for a Python repository.

    Attributes:
        root_path: Absolute path to the repository root.
        files: Dict mapping relative file path to ParsedFile.
    """
    root_path: str
    files: dict[str, ParsedFile] = field(default_factory=dict)

    @property
    def modules(self) -> dict[str, ModuleInfo]:
        """Return all module info dictionaries."""
        return {fp: f.module for fp, f in self.files.items()}

    def get_all_classes(self) -> dict[str, ClassInfo]:
        """Return all classes from all files."""
        result: dict[str, ClassInfo] = {}
        for parsed_file in self.files.values():
            result.update(parsed_file.classes)
        return result

    def get_all_functions(self) -> dict[str, FunctionInfo]:
        """Return all functions from all files."""
        result: dict[str, FunctionInfo] = {}
        for parsed_file in self.files.values():
            result.update(parsed_file.functions)
        return result


class ASTParser:
    """Parses Python source files using the AST module.

    Extracts modules, classes, functions, imports, inheritance
    relationships, and function call information from Python repositories.
    Each class and function stores its source code snippet, line range,
    character count, and estimated token count for context assembly.
    """

    def parse_file(self, file_path: str) -> ParsedFile:
        """Parse a single Python file.

        Args:
            file_path: Absolute or relative path to the Python file.

        Returns:
            ParsedFile containing all extracted information including
            source code snippets for each class and function.

        Raises:
            FileNotFoundError: If the file does not exist.
            SyntaxError: If the file contains invalid Python syntax.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=file_path)
        lines = source.splitlines(keepends=True)

        parsed = ParsedFile(
            module=ModuleInfo(
                name=path.stem,
                file_path=file_path,
            ),
            source_code=source,
            source_length=len(source),
        )

        self._process_top_level(tree, parsed, file_path, lines)
        return parsed

    def parse_directory(self, directory_path: str) -> ParsedRepository:
        """Parse all Python files in a directory recursively.

        Args:
            directory_path: Path to the root directory of the repository.

        Returns:
            ParsedRepository containing all parsed files.
        """
        root = Path(directory_path)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        repo = ParsedRepository(root_path=str(root.resolve()))

        for py_file in sorted(root.rglob("*.py")):
            rel_path = str(py_file.relative_to(root))
            try:
                parsed = self.parse_file(str(py_file))
                repo.files[rel_path] = parsed
            except (SyntaxError, UnicodeDecodeError):
                continue

        return repo

    def _process_top_level(
        self,
        tree: ast.Module,
        parsed: ParsedFile,
        file_path: str,
        lines: list[str],
    ) -> None:
        """Process top-level AST nodes in a module.

        Handles imports, class definitions, and function definitions
        at the module level.
        """
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._process_imports(node, parsed)
            elif isinstance(node, ast.ClassDef):
                self._process_class(node, parsed, file_path, lines)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._process_function(node, parsed, file_path, class_name=None, lines=lines)

    def _process_imports(self, node: ast.Import | ast.ImportFrom, parsed: ParsedFile) -> None:
        """Process import statements and record module dependencies."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name not in parsed.module.imports:
                    parsed.module.imports.append(module_name)

        elif isinstance(node, ast.ImportFrom):
            module_name: str | None = getattr(node, "module", None)
            if module_name:
                first_part = module_name.split(".")[0]
                names = [alias.name for alias in (node.names or [])]
                parsed.module.from_imports[first_part] = names
                if first_part not in parsed.module.imports:
                    parsed.module.imports.append(first_part)

    def _extract_source_snippet(
        self,
        node: ast.AST,
        lines: list[str],
    ) -> tuple[str, int, int]:
        """Extract the source code snippet for an AST node.

        Uses the node's lineno and end_lineno attributes (available
        in Python 3.8+) to extract the exact source text.

        Args:
            node: The AST node.
            lines: List of source lines.

        Returns:
            Tuple of (source_code, start_line, end_line).
        """
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)

        # Lines are 1-indexed in ast, 0-indexed in the list
        snippet_lines = lines[start - 1 : end]
        source = "".join(snippet_lines)
        char_count = len(source)
        # Fast token estimate: chars / 4
        token_est = max(1, char_count // 4)

        return source, start, end

    def _process_class(
        self,
        node: ast.ClassDef,
        parsed: ParsedFile,
        file_path: str,
        lines: list[str],
    ) -> None:
        """Process a class definition node.

        Extracts class name, base classes, methods, nested functions,
        and the full source code snippet.
        """
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)

        source, start_line, end_line = self._extract_source_snippet(node, lines)

        class_info = ClassInfo(
            name=node.name,
            file_path=file_path,
            base_classes=base_names,
            start_line=start_line,
            end_line=end_line,
            source_code=source,
            char_count=len(source),
            token_estimate=max(1, len(source) // 4),
        )

        parsed.classes[node.name] = class_info
        parsed.module.classes.append(node.name)

        # Process methods and nested functions
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._make_function_info(
                    item, parsed, file_path, class_name=node.name, lines=lines
                )
                class_info.methods.append(item.name)
                parsed.functions[item.name] = func_info

    def _process_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parsed: ParsedFile,
        file_path: str,
        class_name: str | None,
        lines: list[str],
    ) -> None:
        """Process a function definition node.

        Extracts function name, parameters, return type, function calls,
        and the full source code snippet.
        """
        func_info = self._make_function_info(
            node, parsed, file_path, class_name=class_name, lines=lines
        )
        parsed.functions[node.name] = func_info
        parsed.module.functions.append(node.name)

    def _make_function_info(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parsed: ParsedFile,
        file_path: str,
        class_name: str | None,
        lines: list[str],
    ) -> FunctionInfo:
        """Create a FunctionInfo from a function AST node.

        Args:
            node: The function AST node.
            parsed: The parsed file being processed.
            file_path: Source file path.
            class_name: Parent class name or None.
            lines: Source lines.

        Returns:
            Fully populated FunctionInfo.
        """
        source, start_line, end_line = self._extract_source_snippet(node, lines)

        func_info = FunctionInfo(
            name=node.name,
            file_path=file_path,
            class_name=class_name,
            start_line=start_line,
            end_line=end_line,
            source_code=source,
            char_count=len(source),
            token_estimate=max(1, len(source) // 4),
        )
        func_info.parameters = [
            self._extract_param_name(arg)
            for arg in node.args.args
        ]
        if isinstance(node.returns, ast.Name):
            func_info.return_annotation = node.returns.id

        # Extract function calls within the function body
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._resolve_call_name(child)
                if call_name:
                    func_info.calls.append(call_name)

        return func_info

    @staticmethod
    def _extract_param_name(arg: ast.arg) -> str:
        """Extract parameter name including 'self' and 'cls'."""
        return arg.arg

    @staticmethod
    def _resolve_call_name(call_node: ast.Call) -> str | None:
        """Resolve the name of a function call node."""
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            return func.attr
        return None
