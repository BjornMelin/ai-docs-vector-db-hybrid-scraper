#!/usr/bin/env python3
"""Script to automatically add appropriate pytest markers to test files.

This script analyzes test files and adds markers based on:
- File path (unit/integration/performance)
- Test content (async, browser, network dependencies)
- Execution time characteristics
"""

import ast
import re
from pathlib import Path


class TestMarkerAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze test functions and determine appropriate markers."""

    def __init__(self):
        self.markers_needed: set[str] = set()
        self.has_async_tests = False
        self.has_browser_imports = False
        self.has_network_imports = False
        self.has_database_imports = False
        self.has_hypothesis_imports = False

    def visit_Import(self, node):
        """Analyze imports to determine test characteristics."""
        for alias in node.names:
            name = alias.name
            self._analyze_import(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Analyze from imports."""
        if node.module:
            self._analyze_import(node.module)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Detect async test functions."""
        if node.name.startswith("test_"):
            self.has_async_tests = True
            self.markers_needed.add("asyncio")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Analyze test function characteristics."""
        if node.name.startswith("test_"):
            # Check for slow operations in function body
            source = ast.get_source_segment(self.source, node) or ""
            if any(
                keyword in source.lower()
                for keyword in ["sleep", "time.sleep", "asyncio.sleep"]
            ):
                self.markers_needed.add("slow")
        self.generic_visit(node)

    def _analyze_import(self, import_name: str):
        """Analyze import to determine test characteristics."""
        import_name = import_name.lower()

        # Browser automation
        if any(
            browser in import_name
            for browser in ["playwright", "browser_use", "crawl4ai", "selenium"]
        ):
            self.has_browser_imports = True
            self.markers_needed.add("browser")

        # Network dependencies
        if any(
            net in import_name
            for net in ["requests", "httpx", "aiohttp", "urllib", "firecrawl"]
        ):
            self.has_network_imports = True
            self.markers_needed.add("network")

        # Database dependencies
        if any(
            db in import_name
            for db in ["qdrant", "redis", "sqlalchemy", "sqlite", "database"]
        ):
            self.has_database_imports = True
            self.markers_needed.add("database")

        # Property-based testing
        if "hypothesis" in import_name:
            self.has_hypothesis_imports = True
            self.markers_needed.add("hypothesis")

    def analyze_file(self, file_path: Path) -> set[str]:
        """Analyze a test file and return recommended markers."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.source = content
            tree = ast.parse(content)
            self.visit(tree)

            # Add path-based markers
            path_str = str(file_path)

            if "/unit/" in path_str:
                self.markers_needed.add("unit")
                self.markers_needed.add("fast")  # Most unit tests should be fast
            elif "/integration/" in path_str:
                self.markers_needed.add("integration")
            elif "/performance/" in path_str or "/benchmarks/" in path_str:
                self.markers_needed.add("performance")
                self.markers_needed.add("benchmark")
                self.markers_needed.add("slow")

            # Content-based analysis
            if "benchmark" in content.lower() or "@pytest.mark.benchmark" in content:
                self.markers_needed.add("benchmark")
                self.markers_needed.add("performance")

            if "localhost" in content or "http://" in content or "https://" in content:
                self.markers_needed.add("network")

            return self.markers_needed

        except Exception:
            print(f"Error analyzing {file_path}: {e}")
            return set()


def add_markers_to_test_file(file_path: Path, markers: set[str]) -> bool:
    """Add markers to a test file if they're not already present."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if file already has pytest imports
        has_pytest_import = "import pytest" in content

        # Find existing markers
        existing_markers = set()
        for line in content.split("\n"):
            if "@pytest.mark." in line:
                marker_match = re.search(r"@pytest\.mark\.(\w+)", line)
                if marker_match:
                    existing_markers.add(marker_match.group(1))

        # Only add new markers
        new_markers = markers - existing_markers
        if not new_markers:
            return False  # No new markers to add

        # Add pytest import if needed
        if not has_pytest_import and new_markers:
            content = content.replace('"""', '"""\n\nimport pytest', 1)

        # Add markers to test classes
        lines = content.split("\n")
        modified = False

        for i, line in enumerate(lines):
            # Look for test class definitions
            if line.strip().startswith("class Test") and "Test" in line:
                # Check if this class already has markers
                if i > 0 and "@pytest.mark." in lines[i - 1]:
                    continue  # Skip if already has markers

                # Add markers before the class
                marker_lines = [
                    f"@pytest.mark.{marker}" for marker in sorted(new_markers)
                ]
                lines[i:i] = marker_lines
                modified = True
                break

        if modified:
            new_content = "\n".join(lines)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(f"✅ Added markers to {file_path}: {', '.join(sorted(new_markers))}")
            return True

        return False

    except Exception:
        print(f"❌ Error adding markers to {file_path}: {e}")
        return False


def main():
    """Main entry point for marker addition script."""
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "tests"

    if not test_dir.exists():
        print("❌ Tests directory not found")
        return 1

    print("🏷️  Analyzing test files and adding appropriate markers...")

    analyzer = TestMarkerAnalyzer()
    files_modified = 0
    total_files = 0

    # Process all test files
    for test_file in test_dir.rglob("test_*.py"):
        total_files += 1

        # Skip __pycache__ and other non-test directories
        if "__pycache__" in str(test_file):
            continue

        analyzer = TestMarkerAnalyzer()  # Reset for each file
        markers = analyzer.analyze_file(test_file)

        if markers:
            if add_markers_to_test_file(test_file, markers):
                files_modified += 1

    print(f"\n📊 Summary:")
    print(f"   Total test files analyzed: {total_files}")
    print(f"   Files modified: {files_modified}")
    print(f"   Files up to date: {total_files - files_modified}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
