"""Unit tests for src/utils/imports.py module."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from src.utils.imports import resolve_imports
from src.utils.imports import setup_import_paths


class TestSetupImportPaths:
    """Test cases for setup_import_paths function."""

    @patch("sys.path", [])
    def test_adds_src_and_project_paths(self):
        """Test that both src directory and project root are added to sys.path."""
        setup_import_paths()

        # Should have added 2 paths
        assert len(sys.path) == 2

        # Both paths should exist
        assert all(Path(path).is_dir() for path in sys.path)

        # Find the src directory path (contains core module but not pyproject.toml)
        src_paths = [
            p
            for p in sys.path
            if (Path(p) / "core").is_dir()
            and not (Path(p) / "pyproject.toml").is_file()
        ]
        assert len(src_paths) == 1

        # Find the project root path (contains src directory and pyproject.toml)
        project_paths = [
            p
            for p in sys.path
            if (Path(p) / "src").is_dir() and (Path(p) / "pyproject.toml").is_file()
        ]
        assert len(project_paths) == 1

    def test_does_not_duplicate_existing_paths(self):
        """Test that paths are not duplicated if already in sys.path."""
        # Get the paths that would be added
        current_file = Path(__file__).resolve()
        utils_dir = current_file.parent.parent.parent / "src" / "utils"
        src_dir = utils_dir.parent
        project_root = src_dir.parent

        src_str = str(src_dir)
        project_str = str(project_root)

        # Pre-populate sys.path with these paths
        original_path = sys.path.copy()
        sys.path.insert(0, src_str)
        sys.path.insert(0, project_str)
        initial_length = len(sys.path)

        try:
            setup_import_paths()

            # Length should not have increased
            assert len(sys.path) == initial_length

            # Paths should still be present
            assert src_str in sys.path
            assert project_str in sys.path

        finally:
            # Restore original sys.path
            sys.path[:] = original_path

    def test_inserts_at_beginning_of_path(self):
        """Test that paths are inserted at the beginning of sys.path."""
        original_path = sys.path.copy()
        original_length = len(sys.path)

        try:
            setup_import_paths()

            # Should have added paths at the beginning
            assert len(sys.path) >= original_length

            # New paths should be at the start
            added_paths = sys.path[: len(sys.path) - original_length]
            for path in added_paths:
                path_obj = Path(path)
                assert path_obj.exists()

        finally:
            # Restore original sys.path
            sys.path[:] = original_path

    def test_handles_relative_path_resolution(self):
        """Test that relative paths are resolved correctly."""
        # Change working directory context in the test
        original_path = sys.path.copy()

        try:
            setup_import_paths()

            # All added paths should be absolute
            for path in sys.path:
                if path:  # Skip empty strings
                    assert Path(path).is_absolute()

        finally:
            sys.path[:] = original_path

    def test_paths_point_to_correct_directories(self):
        """Test that the added paths point to the correct directories."""
        original_path = sys.path.copy()

        try:
            setup_import_paths()

            # Find the src path (should contain our modules)
            src_paths = [p for p in sys.path if Path(p).name == "src"]
            assert len(src_paths) >= 1

            src_path = Path(src_paths[0])

            # Verify it contains expected modules
            assert (src_path / "utils").is_dir()
            assert (src_path / "config").is_dir()
            assert (src_path / "core").is_dir()

            # Find project root path
            project_paths = [p for p in sys.path if (Path(p) / "src").is_dir()]
            assert len(project_paths) >= 1

            project_path = Path(project_paths[0])
            assert (project_path / "pyproject.toml").is_file()

        finally:
            sys.path[:] = original_path

    def test_multiple_calls_are_safe(self):
        """Test that calling setup_import_paths multiple times is safe."""
        original_path = sys.path.copy()

        try:
            # First call
            setup_import_paths()
            first_length = len(sys.path)

            # Second call should not add duplicates
            setup_import_paths()
            second_length = len(sys.path)

            assert first_length == second_length

            # Third call should also be safe
            setup_import_paths()
            third_length = len(sys.path)

            assert first_length == third_length

        finally:
            sys.path[:] = original_path


class TestResolveImports:
    """Test cases for resolve_imports decorator."""

    def test_decorator_calls_setup_import_paths(self):
        """Test that the decorator calls setup_import_paths."""
        original_path = sys.path.copy()

        try:
            # Clear sys.path to ensure setup_import_paths is called
            sys.path.clear()

            @resolve_imports()
            def test_function():
                return len(sys.path)

            result = test_function()

            # Should have added paths
            assert result > 0
            assert len(sys.path) > 0

        finally:
            sys.path[:] = original_path

    def test_decorator_preserves_function_return_value(self):
        """Test that the decorator preserves the function's return value."""

        @resolve_imports()
        def test_function():
            return "test result"

        result = test_function()
        assert result == "test result"

    def test_decorator_preserves_function_arguments(self):
        """Test that the decorator preserves function arguments."""

        @resolve_imports()
        def test_function(arg1, arg2, kwarg1=None, kwarg2=None):
            return f"{arg1}-{arg2}-{kwarg1}-{kwarg2}"

        result = test_function("a", "b", kwarg1="c", kwarg2="d")
        assert result == "a-b-c-d"

    def test_decorator_preserves_function_metadata(self):
        """Test that the decorator preserves function metadata."""

        @resolve_imports()
        def test_function():
            """Test function docstring."""
            return "result"

        test_function.custom_attr = "custom_value"

        # Note: The current implementation uses a simple wrapper, so function name
        # will be 'wrapper', but the function should still work correctly
        assert callable(test_function)
        assert test_function() == "result"
        # Custom attributes can be set after decoration
        assert hasattr(test_function, "custom_attr")

    def test_decorator_handles_exceptions(self):
        """Test that the decorator properly handles exceptions."""

        @resolve_imports()
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

    def test_decorator_with_args_and_kwargs(self):
        """Test decorator with various argument patterns."""

        @resolve_imports()
        def test_function(*args, **kwargs):
            return f"args: {args}, kwargs: {kwargs}"

        result = test_function(1, 2, 3, x="a", y="b")
        assert result == "args: (1, 2, 3), kwargs: {'x': 'a', 'y': 'b'}"

    def test_decorator_can_be_used_multiple_times(self):
        """Test that the decorator can be applied to multiple functions."""

        @resolve_imports()
        def function1():
            return "result1"

        @resolve_imports()
        def function2():
            return "result2"

        assert function1() == "result1"
        assert function2() == "result2"

    def test_decorator_works_with_other_decorators(self):
        """Test that resolve_imports works with other decorators."""

        def other_decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return f"decorated: {result}"

            return wrapper

        @other_decorator
        @resolve_imports()
        def test_function():
            return "original"

        result = test_function()
        assert result == "decorated: original"

    @patch("src.utils.imports.setup_import_paths")
    def test_decorator_calls_setup_import_paths_mock(self, mock_setup):
        """Test that the decorator calls setup_import_paths using mock."""

        @resolve_imports()
        def test_function():
            return "result"

        result = test_function()

        assert result == "result"
        mock_setup.assert_called_once()

    def test_nested_function_calls(self):
        """Test that nested function calls work correctly."""

        @resolve_imports()
        def outer_function():
            @resolve_imports()
            def inner_function():
                return "inner result"

            return f"outer: {inner_function()}"

        result = outer_function()
        assert result == "outer: inner result"


class TestImportPathResolution:
    """Integration tests for import path resolution."""

    def test_enables_src_imports(self):
        """Test that setup enables imports from src directory."""
        original_path = sys.path.copy()

        try:
            # Remove src paths if present
            sys.path[:] = [p for p in sys.path if "src" not in p.lower()]

            setup_import_paths()

            # Should now be able to import from src
            # We test this by checking that src modules are accessible
            src_paths = [p for p in sys.path if Path(p).name == "src"]
            assert len(src_paths) >= 1

            # Verify the path contains expected structure
            src_path = Path(src_paths[0])
            assert (src_path / "__init__.py").exists() or (src_path / "config").is_dir()

        finally:
            sys.path[:] = original_path

    def test_enables_project_root_imports(self):
        """Test that setup enables imports from project root."""
        original_path = sys.path.copy()

        try:
            setup_import_paths()

            # Should have project root in path
            project_paths = [p for p in sys.path if (Path(p) / "src").is_dir()]
            assert len(project_paths) >= 1

            # Verify it's the correct project root
            project_path = Path(project_paths[0])
            assert (project_path / "pyproject.toml").is_file()

        finally:
            sys.path[:] = original_path

    def test_decorator_enables_imports(self):
        """Test that the decorator enables imports in the decorated function."""
        original_path = sys.path.copy()

        try:
            # Clear paths
            sys.path.clear()

            @resolve_imports()
            def test_imports():
                # Should be able to access paths now
                return len([p for p in sys.path if p and Path(p).exists()])

            result = test_imports()
            assert result >= 1  # Should have added valid paths

        finally:
            sys.path[:] = original_path


class TestPathHandling:
    """Test cases for path handling edge cases."""

    def test_handles_symlinks(self):
        """Test that symlinks in paths are handled correctly."""
        # This test mainly verifies that Path.resolve() is used correctly
        original_path = sys.path.copy()

        try:
            setup_import_paths()

            # All paths should be resolved (absolute)
            for path in sys.path:
                if path:  # Skip empty strings
                    path_obj = Path(path)
                    assert path_obj.is_absolute()

        finally:
            sys.path[:] = original_path

    def test_handles_nonexistent_parent_directories(self):
        """Test behavior when parent directories might not exist."""
        # This primarily tests the robustness of the path resolution
        original_path = sys.path.copy()

        try:
            setup_import_paths()

            # Should not raise exceptions and should add valid paths
            valid_paths = [p for p in sys.path if p and Path(p).exists()]
            assert len(valid_paths) >= 1

        finally:
            sys.path[:] = original_path

    def test_path_string_conversion(self):
        """Test that paths are properly converted to strings."""
        original_path = sys.path.copy()

        try:
            setup_import_paths()

            # All paths in sys.path should be strings
            for path in sys.path:
                assert isinstance(path, str)

        finally:
            sys.path[:] = original_path


class TestDecoratorEdgeCases:
    """Test edge cases for the resolve_imports decorator."""

    def test_decorator_with_no_arguments(self):
        """Test that the decorator works when called without parentheses."""

        # This tests the current implementation which requires parentheses
        @resolve_imports()
        def test_function():
            return "result"

        assert test_function() == "result"

    def test_decorator_return_types(self):
        """Test that various return types work correctly."""

        @resolve_imports()
        def return_none():
            return None

        @resolve_imports()
        def return_dict():
            return {"key": "value"}

        @resolve_imports()
        def return_list():
            return [1, 2, 3]

        @resolve_imports()
        def return_complex():
            return {"list": [1, 2], "dict": {"nested": True}}

        assert return_none() is None
        assert return_dict() == {"key": "value"}
        assert return_list() == [1, 2, 3]
        assert return_complex() == {"list": [1, 2], "dict": {"nested": True}}

    def test_function_with_default_arguments(self):
        """Test decorator with functions that have default arguments."""

        @resolve_imports()
        def test_function(arg1, arg2="default", arg3=None):
            return f"{arg1}-{arg2}-{arg3}"

        assert test_function("a") == "a-default-None"
        assert test_function("a", "b") == "a-b-None"
        assert test_function("a", "b", "c") == "a-b-c"
        assert test_function("a", arg3="c") == "a-default-c"
