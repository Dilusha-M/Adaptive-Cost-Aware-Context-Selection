"""Unit tests for the AST parser module.

Tests cover file parsing, directory parsing, import extraction,
class extraction, function extraction, and inheritance detection.
"""

import pytest
import tempfile
import os
from pathlib import Path

from parser.ast_parser import ASTParser, ParsedFile, ParsedRepository


class TestASTParserFileParsing:
    """Tests for parsing individual Python files."""

    def test_parse_simple_module(self, tmp_path: Path) -> None:
        """Test parsing a module with a simple function."""
        source = 'def hello():\n    return "hello"\n'
        file_path = tmp_path / "test_module.py"
        file_path.write_text(source)

        parser = ASTParser()
        result = parser.parse_file(str(file_path))

        assert result.module.name == "test_module"
        assert result.module.file_path == str(file_path)
        assert result.module.functions == ["hello"]
        assert result.source_length > 0

    def test_parse_module_with_import(self, tmp_path: Path) -> None:
        """Test parsing a module that imports another module."""
        source = 'import os\nimport sys\nfrom pathlib import Path\n'
        file_path = tmp_path / "test_imports.py"
        file_path.write_text(source)

        parser = ASTParser()
        result = parser.parse_file(str(file_path))

        assert "os" in result.module.imports
        assert "sys" in result.module.imports
        assert "pathlib" in result.module.from_imports

    def test_parse_module_with_class(self, tmp_path: Path) -> None:
        """Test parsing a module with a class definition."""
        source = (
            'class MyClass:\n'
            '    def __init__(self):\n'
            '        self.value = 0\n'
            '\n'
            '    def get_value(self):\n'
            '        return self.value\n'
        )
        file_path = tmp_path / "test_class.py"
        file_path.write_text(source)

        parser = ASTParser()
        result = parser.parse_file(str(file_path))

        assert "MyClass" in result.classes
        cls_info = result.classes["MyClass"]
        assert cls_info.name == "MyClass"
        assert cls_info.file_path == str(file_path)
        assert "get_value" in cls_info.methods

    def test_parse_nonexistent_file(self, tmp_path: Path) -> None:
        """Test that parsing a non-existent file raises FileNotFoundError."""
        parser = ASTParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file(str(tmp_path / "nonexistent.py"))

    def test_parse_syntax_error(self, tmp_path: Path) -> None:
        """Test that parsing a file with syntax errors raises SyntaxError."""
        source = 'def broken(\n'  # Incomplete function definition
        file_path = tmp_path / "broken.py"
        file_path.write_text(source)

        parser = ASTParser()
        with pytest.raises(SyntaxError):
            parser.parse_file(str(file_path))


class TestASTParserDirectoryParsing:
    """Tests for parsing entire directories."""

    def test_parse_directory(self, tmp_path: Path) -> None:
        """Test parsing a directory with multiple Python files."""
        # Create test files
        (tmp_path / "module_a.py").write_text('import os\nx = 1\n')
        (tmp_path / "module_b.py").write_text('from module_a import x\ny = 2\n')
        (tmp_path / "module_c.py").write_text(
            'from module_a import x\n'
            'from module_b import y\n'
            'class C:\n'
            '    pass\n'
        )

        parser = ASTParser()
        result = parser.parse_directory(str(tmp_path))

        assert isinstance(result, ParsedRepository)
        assert result.root_path == str(tmp_path.resolve())
        assert len(result.files) == 3

    def test_parse_directory_empty(self, tmp_path: Path) -> None:
        """Test parsing an empty directory."""
        parser = ASTParser()
        result = parser.parse_directory(str(tmp_path))

        assert isinstance(result, ParsedRepository)
        assert len(result.files) == 0

    def test_parse_directory_subdirectories(self, tmp_path: Path) -> None:
        """Test parsing a directory with subdirectories."""
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "nested.py").write_text('z = 3\n')
        (tmp_path / "top.py").write_text('t = 1\n')

        parser = ASTParser()
        result = parser.parse_directory(str(tmp_path))

        assert len(result.files) == 2
        assert "subdir/nested.py" in result.files

    def test_parse_directory_ignores_non_python(self, tmp_path: Path) -> None:
        """Test that non-Python files are ignored during directory parsing."""
        (tmp_path / "test.py").write_text('x = 1\n')
        (tmp_path / "test.txt").write_text('not python\n')
        (tmp_path / "test.md").write_text('# README\n')

        parser = ASTParser()
        result = parser.parse_directory(str(tmp_path))

        assert len(result.files) == 1
        assert "test.py" in result.files


class TestASTParserClassAndFunctionExtraction:
    """Tests for detailed class and function extraction."""

    def test_class_with_inheritance(self, tmp_path: Path) -> None:
        """Test parsing a class with base class inheritance."""
        source = (
            'class BaseModel:\n'
            '    pass\n'
            '\n'
            'class ChildModel(BaseModel):\n'
            '    pass\n'
        )
        file_path = tmp_path / "inheritance.py"
        file_path.write_text(source)

        parser = ASTParser()
        result = parser.parse_file(str(file_path))

        assert "BaseModel" in result.classes
        assert "ChildModel" in result.classes
        assert "BaseModel" in result.classes["ChildModel"].base_classes

    def test_function_parameter_extraction(self, tmp_path: Path) -> None:
        """Test that function parameters are correctly extracted."""
        source = (
            'def process(self, name: str, count: int = 0) -> bool:\n'
            '    return True\n'
        )
        file_path = tmp_path / "params.py"
        file_path.write_text(source)

        parser = ASTParser()
        result = parser.parse_file(str(file_path))

        func = result.functions["process"]
        assert "self" in func.parameters
        assert "name" in func.parameters
        assert "count" in func.parameters

    def test_function_calls_extraction(self, tmp_path: Path) -> None:
        """Test that function call targets are extracted."""
        source = (
            'def outer():\n'
            '    helper()\n'
            '    obj.method()\n'
            '\n'
            'def helper():\n'
            '    pass\n'
        )
        file_path = tmp_path / "calls.py"
        file_path.write_text(source)

        parser = ASTParser()
        result = parser.parse_file(str(file_path))

        outer = result.functions["outer"]
        assert "helper" in outer.calls
        assert "method" in outer.calls

    def test_parsed_repository_accessors(self, tmp_path: Path) -> None:
        """Test repository-level accessors for classes and functions."""
        source = (
            'class A:\n'
            '    pass\n'
            '\n'
            'class B:\n'
            '    pass\n'
            '\n'
            'def func1():\n'
            '    pass\n'
        )
        file_path = tmp_path / "multi.py"
        file_path.write_text(source)

        parser = ASTParser()
        result = parser.parse_file(str(file_path))

        repo = ParsedRepository(root_path=str(tmp_path))
        repo.files[str(file_path)] = result

        all_classes = repo.get_all_classes()
        all_functions = repo.get_all_functions()

        assert "A" in all_classes
        assert "B" in all_classes
        assert "func1" in all_functions
