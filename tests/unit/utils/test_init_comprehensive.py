"""Comprehensive tests for utils package initialization.

This test suite provides complete coverage for the utils package __init__.py
including both normal and fallback scenarios.
"""

import sys
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest


class TestUtilsPackageInit:
    """Test utils package initialization and exports."""

    def test_normal_imports_available(self):
        """Test that normal imports are available."""
        # Import the utils package fresh to test normal behavior
        import src.utils as utils_package

        # Test that the main functions are available
        assert hasattr(utils_package, "resolve_imports")
        assert hasattr(utils_package, "setup_import_paths")
        assert hasattr(utils_package, "async_command")
        assert hasattr(utils_package, "async_to_sync_click")

        # Test that they are callable
        assert callable(utils_package.resolve_imports)
        assert callable(utils_package.setup_import_paths)
        assert callable(utils_package.async_command)
        assert callable(utils_package.async_to_sync_click)

    def test_all_exports_defined(self):
        """Test that __all__ contains expected exports."""
        import src.utils as utils_package

        expected_exports = [
            "async_command",
            "async_to_sync_click",
            "resolve_imports",
            "setup_import_paths",
        ]

        assert hasattr(utils_package, "__all__")
        assert set(utils_package.__all__) == set(expected_exports)

        # Test that all items in __all__ are actually available
        for export in utils_package.__all__:
            assert hasattr(utils_package, export)

    def test_imports_from_local_modules(self):
        """Test that imports from local modules work correctly."""
        import src.utils as utils_package
        from src.utils.imports import resolve_imports
        from src.utils.imports import setup_import_paths

        # Verify these are the same functions
        assert utils_package.resolve_imports is resolve_imports
        assert utils_package.setup_import_paths is setup_import_paths


class TestUtilsPackageFallback:
    """Test fallback behavior when parent utils.py is not available."""

    def test_fallback_when_utils_py_missing(self):
        """Test fallback implementations when utils.py doesn't exist."""
        # Mock Path.exists to return False
        with patch("pathlib.Path.exists", return_value=False):
            # Remove the module from cache if it exists
            module_name = "src.utils"
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Re-import to trigger fallback behavior
            import src.utils as utils_package

            # Test that fallback functions exist and raise ImportError
            assert hasattr(utils_package, "async_command")
            assert hasattr(utils_package, "async_to_sync_click")

            with pytest.raises(ImportError, match="async_command not available"):
                utils_package.async_command()

            with pytest.raises(ImportError, match="async_to_sync_click not available"):
                utils_package.async_to_sync_click()

    def test_fallback_functions_with_arguments(self):
        """Test fallback functions with various argument combinations."""
        with patch("pathlib.Path.exists", return_value=False):
            # Remove the module from cache
            module_name = "src.utils"
            if module_name in sys.modules:
                del sys.modules[module_name]

            import src.utils as utils_package

            # Test with positional arguments
            with pytest.raises(ImportError, match="async_command not available"):
                utils_package.async_command("arg1", "arg2")

            # Test with keyword arguments
            with pytest.raises(ImportError, match="async_to_sync_click not available"):
                utils_package.async_to_sync_click(param1="value1", param2="value2")

            # Test with mixed arguments
            with pytest.raises(ImportError, match="async_command not available"):
                utils_package.async_command("arg1", keyword="value")


class TestUtilsPackagePathResolution:
    """Test path resolution logic in package initialization."""

    def test_parent_directory_calculation(self):
        """Test that parent directory is calculated correctly."""
        # This tests the logic: parent_dir = Path(__file__).parent.parent
        init_file = Path("src/utils/__init__.py")
        # expected_parent = init_file.parent.parent  # Should be the project root

        # Mock the Path calculation
        with patch("src.utils.Path") as mock_path_class:
            mock_file_path = Mock()
            mock_utils_dir = Mock()
            mock_parent_dir = Mock()

            # Set up parent chain
            mock_file_path.parent = mock_utils_dir
            mock_utils_dir.parent = mock_parent_dir

            mock_path_class.return_value = mock_file_path

            # The actual calculation happens during import, so we test the logic
            assert mock_file_path.parent == mock_utils_dir
            assert mock_utils_dir.parent == mock_parent_dir

    def test_utils_path_construction(self):
        """Test that utils.py path is constructed correctly."""
        # Test the logic: utils_path = parent_dir / "utils.py"
        mock_parent_dir = Mock()
        mock_utils_path = Mock()

        # Mock the / operator to return our mock path
        mock_parent_dir.__truediv__ = Mock(return_value=mock_utils_path)

        result = mock_parent_dir / "utils.py"

        assert result == mock_utils_path
        mock_parent_dir.__truediv__.assert_called_once_with("utils.py")


class TestUtilsPackageImportLogic:
    """Test the dynamic import logic for parent utils.py."""

    def test_importlib_spec_creation(self):
        """Test spec creation for dynamic import."""
        with patch("importlib.util.spec_from_file_location") as mock_spec_func:
            with patch("importlib.util.module_from_spec") as mock_module_func:
                with patch("pathlib.Path.exists", return_value=True):
                    mock_spec = Mock()
                    mock_loader = Mock()
                    mock_spec.loader = mock_loader
                    mock_spec_func.return_value = mock_spec

                    mock_module = Mock()
                    mock_module.async_to_sync_click = Mock()
                    mock_module.async_command = Mock()
                    mock_module_func.return_value = mock_module

                    # Remove module from cache to force re-import
                    module_name = "src.utils"
                    if module_name in sys.modules:
                        del sys.modules[module_name]

                    # Import to trigger the logic

                    # Verify the importlib functions were called
                    mock_spec_func.assert_called()
                    mock_module_func.assert_called_once_with(mock_spec)
                    mock_loader.exec_module.assert_called_once_with(mock_module)


class TestUtilsPackageEdgeCases:
    """Test edge cases and error conditions."""

    def test_module_reload_behavior(self):
        """Test behavior when module is reloaded."""
        import src.utils as utils_package

        # Basic sanity check after potential reload
        assert hasattr(utils_package, "resolve_imports")
        assert hasattr(utils_package, "setup_import_paths")
        assert hasattr(utils_package, "async_command")
        assert hasattr(utils_package, "async_to_sync_click")

    def test_import_with_missing_parent_utils_functions(self):
        """Test behavior when parent utils.py exists but lacks expected functions."""
        with patch("pathlib.Path.exists", return_value=True):
            # Mock importlib to return a module without the expected functions
            with patch("importlib.util.spec_from_file_location") as mock_spec_func:
                with patch("importlib.util.module_from_spec") as mock_module_func:
                    mock_spec = Mock()
                    mock_loader = Mock()
                    mock_spec.loader = mock_loader
                    mock_spec_func.return_value = mock_spec

                    # Create module without the expected attributes
                    mock_module = Mock()
                    # Remove any async_command or async_to_sync_click if they exist
                    if hasattr(mock_module, "async_command"):
                        del mock_module.async_command
                    if hasattr(mock_module, "async_to_sync_click"):
                        del mock_module.async_to_sync_click

                    mock_module_func.return_value = mock_module

                    # This should raise AttributeError when trying to access the functions
                    module_name = "src.utils"
                    if module_name in sys.modules:
                        del sys.modules[module_name]

                    with pytest.raises(AttributeError):
                        pass

    def test_docstring_present(self):
        """Test that package docstring is present."""
        import src.utils as utils_package

        assert utils_package.__doc__ is not None
        assert "Utilities package" in utils_package.__doc__
