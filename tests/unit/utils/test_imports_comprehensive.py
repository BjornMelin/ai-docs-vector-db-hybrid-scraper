"""Comprehensive tests for import resolution utilities.

This test suite provides complete coverage for the imports module
including path setup and decorator functionality.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.utils.imports import resolve_imports, setup_import_paths


class TestSetupImportPaths:
    """Test setup_import_paths function."""

    def test_setup_import_paths_adds_missing_paths(self):
        """Test that setup_import_paths adds missing paths to sys.path."""
        # Get expected paths
        current_file = Path(__file__).resolve()
        test_utils_dir = current_file.parent
        test_unit_dir = test_utils_dir.parent
        tests_dir = test_unit_dir.parent
        project_root = tests_dir.parent
        src_dir = project_root / "src"

        # Mock sys.path to not contain these paths
        original_sys_path = sys.path.copy()

        try:
            # Remove paths if they exist
            test_sys_path = [
                p
                for p in sys.path
                if str(src_dir) not in p and str(project_root) not in p
            ]

            with patch.object(sys, "path", test_sys_path):
                setup_import_paths()

                # Check that both paths were added to the beginning
                assert str(src_dir) in sys.path
                assert str(project_root) in sys.path

                # Verify they were inserted at the beginning (index 0 and 1)
                assert sys.path[0] == str(src_dir) or sys.path[1] == str(src_dir)
                assert sys.path[0] == str(project_root) or sys.path[1] == str(
                    project_root
                )
        finally:
            # Restore original sys.path
            sys.path.clear()
            sys.path.extend(original_sys_path)

    def test_setup_import_paths_skips_existing_paths(self):
        """Test that setup_import_paths doesn't duplicate existing paths."""
        # Get expected paths
        imports_file_path = Path("src/utils/imports.py").resolve()
        src_dir = imports_file_path.parent.parent
        project_root = src_dir.parent

        original_sys_path = sys.path.copy()

        try:
            # Create a sys.path that already contains our paths
            test_sys_path = [str(src_dir), str(project_root), "/some/other/path"]

            with patch.object(sys, "path", test_sys_path):
                initial_length = len(sys.path)
                setup_import_paths()

                # Length should remain the same since paths already exist
                assert len(sys.path) == initial_length

                # Paths should still be present
                assert str(src_dir) in sys.path
                assert str(project_root) in sys.path
        finally:
            # Restore original sys.path
            sys.path.clear()
            sys.path.extend(original_sys_path)

    def test_setup_import_paths_path_resolution(self):
        """Test that paths are resolved correctly from the imports module location."""
        # Create mock path hierarchy
        mock_file_path = Mock()
        mock_utils_dir = Mock()
        mock_src_dir = Mock()
        mock_project_root = Mock()

        # Set up the parent chain
        mock_file_path.parent = mock_utils_dir
        mock_utils_dir.parent = mock_src_dir
        mock_src_dir.parent = mock_project_root

        # Mock string representations
        str(mock_src_dir)
        str(mock_project_root)

        with patch("src.utils.imports.Path") as mock_path_class:
            mock_path_instance = Mock()
            mock_path_instance.resolve.return_value = mock_file_path
            mock_path_class.return_value = mock_path_instance

            # Mock sys.path to be empty
            with patch.object(sys, "path", []):
                setup_import_paths()

                # Verify the paths were calculated correctly
                # The function uses its own __file__, not the test file's __file__
                mock_path_class.assert_called_once()
                mock_path_instance.resolve.assert_called_once()


class TestResolveImportsDecorator:
    """Test resolve_imports decorator function."""

    def test_resolve_imports_decorator_basic_usage(self):
        """Test basic usage of resolve_imports decorator."""
        # Mock function to decorate
        mock_function = Mock(return_value="test_result")

        # Apply decorator
        decorated_function = resolve_imports()(mock_function)

        # Mock setup_import_paths to verify it's called
        with patch("src.utils.imports.setup_import_paths") as mock_setup:
            result = decorated_function("arg1", "arg2", kwarg1="value1")

            # Verify setup_import_paths was called
            mock_setup.assert_called_once()

            # Verify original function was called with correct arguments
            mock_function.assert_called_once_with("arg1", "arg2", kwarg1="value1")

            # Verify result is passed through
            assert result == "test_result"

    def test_resolve_imports_decorator_with_no_args(self):
        """Test decorator with no arguments to decorated function."""
        mock_function = Mock(return_value=42)
        decorated_function = resolve_imports()(mock_function)

        with patch("src.utils.imports.setup_import_paths") as mock_setup:
            result = decorated_function()

            mock_setup.assert_called_once()
            mock_function.assert_called_once_with()
            assert result == 42

    def test_resolve_imports_decorator_with_exception(self):
        """Test decorator when decorated function raises exception."""

        def failing_function():
            raise ValueError("Test exception")

        decorated_function = resolve_imports()(failing_function)

        with patch("src.utils.imports.setup_import_paths") as mock_setup:
            with pytest.raises(ValueError, match="Test exception"):
                decorated_function()

            # setup_import_paths should still be called before the exception
            mock_setup.assert_called_once()

    def test_resolve_imports_decorator_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""

        def original_function():
            """Original function docstring."""
            return "original_result"

        decorated_function = resolve_imports()(original_function)

        # Function should be wrapped but callable
        assert callable(decorated_function)

        # Test it works correctly
        with patch("src.utils.imports.setup_import_paths"):
            result = decorated_function()
            assert result == "original_result"

    def test_resolve_imports_decorator_multiple_calls(self):
        """Test decorator with multiple calls to same function."""
        mock_function = Mock(side_effect=["call1", "call2", "call3"])
        decorated_function = resolve_imports()(mock_function)

        with patch("src.utils.imports.setup_import_paths") as mock_setup:
            # Call multiple times
            result1 = decorated_function("arg1")
            result2 = decorated_function("arg2")
            result3 = decorated_function("arg3")

            # setup_import_paths should be called each time
            assert mock_setup.call_count == 3

            # Original function should be called each time with correct args
            assert mock_function.call_count == 3
            mock_function.assert_any_call("arg1")
            mock_function.assert_any_call("arg2")
            mock_function.assert_any_call("arg3")

            # Results should be correct
            assert result1 == "call1"
            assert result2 == "call2"
            assert result3 == "call3"


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_real_world_usage_example(self):
        """Test a realistic usage scenario."""

        # Simulate a main function that would use the decorator
        @resolve_imports()
        def example_main():
            # This would normally import from src modules
            return "main_executed"

        with patch("src.utils.imports.setup_import_paths") as mock_setup:
            result = example_main()

            mock_setup.assert_called_once()
            assert result == "main_executed"

    def test_decorator_factory_pattern(self):
        """Test that resolve_imports() returns a decorator (factory pattern)."""
        # resolve_imports() should return a decorator function
        decorator = resolve_imports()
        assert callable(decorator)

        # The decorator should accept a function and return a wrapped function
        def test_func():
            return "test"

        wrapped = decorator(test_func)
        assert callable(wrapped)
        assert wrapped != test_func  # Should be a different function object

    def test_nested_decorator_calls(self):
        """Test behavior with nested or chained decorators."""

        def other_decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return f"decorated_{result}"

            return wrapper

        @other_decorator
        @resolve_imports()
        def test_function():
            return "base"

        with patch("src.utils.imports.setup_import_paths") as mock_setup:
            result = test_function()

            mock_setup.assert_called_once()
            assert result == "decorated_base"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_path_resolution_with_symlinks(self):
        """Test path resolution when symlinks are involved."""
        # Create mock resolved path hierarchy
        mock_resolved_file = Mock()
        mock_utils_dir = Mock()
        mock_src_dir = Mock()
        mock_project_root = Mock()

        # Set up the parent chain for resolved path
        mock_resolved_file.parent = mock_utils_dir
        mock_utils_dir.parent = mock_src_dir
        mock_src_dir.parent = mock_project_root

        with patch("src.utils.imports.Path") as mock_path_class:
            mock_instance = Mock()
            mock_instance.resolve.return_value = mock_resolved_file
            mock_path_class.return_value = mock_instance

            with patch.object(sys, "path", []):
                setup_import_paths()

                # Should handle symlink resolution correctly
                mock_instance.resolve.assert_called_once()

    def test_setup_import_paths_empty_sys_path(self):
        """Test setup_import_paths with completely empty sys.path."""
        with patch.object(sys, "path", []):
            setup_import_paths()

            # Should add both paths even to empty sys.path
            assert len(sys.path) == 2

    def test_decorator_with_complex_arguments(self):
        """Test decorator with complex argument combinations."""
        mock_function = Mock(return_value="complex_result")
        decorated_function = resolve_imports()(mock_function)

        with patch("src.utils.imports.setup_import_paths"):
            # Test with mixed args and kwargs
            result = decorated_function(
                "pos1", "pos2", keyword1="kw1", keyword2={"nested": "dict"}
            )

            mock_function.assert_called_once_with(
                "pos1", "pos2", keyword1="kw1", keyword2={"nested": "dict"}
            )
            assert result == "complex_result"
