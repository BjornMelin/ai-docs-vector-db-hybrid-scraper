#!/usr/bin/env python3
"""Convert coverage-driven tests to behavior-driven tests.

Analyzes test files and suggests conversions from implementation-focused
tests to behavior-focused tests.
"""

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class TestConverter:
    """Converts coverage-driven tests to behavior-driven tests."""
    
    # Patterns that indicate coverage-driven testing
    COVERAGE_PATTERNS = {
        "private_method": re.compile(r"test_.*_private|test_.*_internal|test_private_"),
        "implementation_detail": re.compile(r"test_.*_helper|test_.*_util"),
        "getter_setter": re.compile(r"test_get_|test_set_|test_.*_property"),
        "simple_init": re.compile(r"test_.*_init|test_constructor|test_initialization"),
        "internal_state": re.compile(r"test_.*_state|test_.*_attribute"),
    }
    
    # Behavior-focused test name patterns
    BEHAVIOR_PATTERNS = {
        "when": re.compile(r"test_.*_when_"),
        "should": re.compile(r"test_.*_should_"),
        "returns": re.compile(r"test_.*_returns_"),
        "raises": re.compile(r"test_.*_raises_"),
        "handles": re.compile(r"test_.*_handles_"),
    }
    
    def __init__(self):
        self.conversions: List[Dict] = []
        self.analysis_results: Dict = {
            "coverage_driven_tests": 0,
            "behavior_driven_tests": 0,
            "mixed_tests": 0,
            "conversion_suggestions": [],
        }
    
    def analyze_test_file(self, file_path: Path) -> Dict:
        """Analyze a test file for coverage vs behavior patterns."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            file_analysis = {
                "file": str(file_path),
                "coverage_tests": [],
                "behavior_tests": [],
                "suggestions": [],
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    test_type = self._classify_test(node, content)
                    
                    if test_type == "coverage":
                        file_analysis["coverage_tests"].append(node.name)
                        suggestion = self._suggest_conversion(node, content)
                        if suggestion:
                            file_analysis["suggestions"].append(suggestion)
                    elif test_type == "behavior":
                        file_analysis["behavior_tests"].append(node.name)
            
            return file_analysis
            
        except Exception as e:
            return {
                "file": str(file_path),
                "error": str(e),
                "coverage_tests": [],
                "behavior_tests": [],
                "suggestions": [],
            }
    
    def _classify_test(self, node: ast.FunctionDef, content: str) -> str:
        """Classify a test as coverage-driven or behavior-driven."""
        test_name = node.name
        
        # Check for coverage patterns
        for pattern_type, pattern in self.COVERAGE_PATTERNS.items():
            if pattern.match(test_name):
                return "coverage"
        
        # Check for behavior patterns
        for pattern_type, pattern in self.BEHAVIOR_PATTERNS.items():
            if pattern.search(test_name):
                return "behavior"
        
        # Analyze test body for additional clues
        test_body = ast.get_source_segment(content, node)
        if test_body:
            # Look for private method calls
            if re.search(r"\._[a-zA-Z_]+\(", test_body):
                return "coverage"
            
            # Look for behavior assertions
            if re.search(r"assert.*raises|assert.*returns|assert.*should", test_body):
                return "behavior"
        
        return "mixed"
    
    def _suggest_conversion(self, node: ast.FunctionDef, content: str) -> Optional[Dict]:
        """Suggest how to convert a coverage-driven test to behavior-driven."""
        test_name = node.name
        test_body = ast.get_source_segment(content, node)
        
        suggestion = {
            "original_name": test_name,
            "suggested_name": None,
            "conversion_notes": [],
            "example": None,
        }
        
        # Suggest new name based on what the test does
        if "init" in test_name or "constructor" in test_name:
            suggestion["suggested_name"] = test_name.replace("test_", "test_creates_instance_with_")
            suggestion["conversion_notes"].append(
                "Focus on the observable behavior after creation, not just that __init__ runs"
            )
            suggestion["example"] = """
# Instead of:
def test_user_init():
    user = User("John", "john@example.com")
    assert user._name == "John"  # Testing private attribute
    
# Use:
def test_creates_user_with_valid_email():
    user = User("John", "john@example.com")
    assert user.get_display_name() == "John"  # Test public behavior
    assert user.can_login() is True  # Test capability
"""
        
        elif "private" in test_name or "_internal" in test_name:
            method_name = test_name.replace("test_", "").replace("_private", "").replace("_internal", "")
            suggestion["suggested_name"] = f"test_{method_name}_behavior_when_called"
            suggestion["conversion_notes"].append(
                "Test the public API that uses this private method instead"
            )
            suggestion["example"] = """
# Instead of:
def test_calculate_total_internal():
    order = Order()
    assert order._calculate_tax() == 0.1  # Testing private method
    
# Use:
def test_order_total_includes_tax():
    order = Order()
    order.add_item(Item("Widget", 100))
    assert order.get_total() == 110  # Test public behavior
"""
        
        elif "get_" in test_name or "set_" in test_name:
            suggestion["suggested_name"] = test_name.replace("test_get_", "test_provides_").replace("test_set_", "test_updates_")
            suggestion["conversion_notes"].append(
                "Test the behavior enabled by the getter/setter, not just data access"
            )
        
        elif "_property" in test_name:
            suggestion["suggested_name"] = test_name.replace("_property", "_behavior")
            suggestion["conversion_notes"].append(
                "Properties should be tested through their usage in actual behavior"
            )
        
        else:
            # Generic suggestion
            suggestion["suggested_name"] = f"test_{test_name.replace('test_', '')}_achieves_expected_outcome"
            suggestion["conversion_notes"].append(
                "Focus on what the user/caller observes, not how it's implemented"
            )
        
        # Add general conversion notes
        suggestion["conversion_notes"].extend([
            "Avoid testing private methods directly",
            "Test through public API",
            "Focus on observable outcomes",
            "Use descriptive names that explain the behavior",
        ])
        
        return suggestion
    
    def convert_file(self, file_path: Path, dry_run: bool = True) -> bool:
        """Convert tests in a file from coverage to behavior driven."""
        analysis = self.analyze_test_file(file_path)
        
        if not analysis["suggestions"]:
            return False
        
        if dry_run:
            print(f"\nWould convert {file_path}:")
            for suggestion in analysis["suggestions"]:
                print(f"  - {suggestion['original_name']} -> {suggestion['suggested_name']}")
            return True
        
        try:
            content = file_path.read_text()
            original_content = content
            
            # Apply conversions
            for suggestion in analysis["suggestions"]:
                if suggestion["suggested_name"]:
                    # Replace test name
                    content = re.sub(
                        rf"\bdef {suggestion['original_name']}\b",
                        f"def {suggestion['suggested_name']}",
                        content
                    )
                    
                    # Add comment with conversion notes
                    comment = "\n".join(f"    # {note}" for note in suggestion["conversion_notes"])
                    content = content.replace(
                        f"def {suggestion['suggested_name']}",
                        f"def {suggestion['suggested_name']}\n{comment}\n"
                    )
            
            if content != original_content:
                file_path.write_text(content)
                print(f"Converted {file_path}")
                return True
                
        except Exception as e:
            print(f"Error converting {file_path}: {e}")
            
        return False
    
    def generate_report(self, directory: Path) -> Dict:
        """Generate a conversion report for a directory."""
        test_files = list(directory.rglob("test_*.py"))
        
        for file_path in test_files:
            analysis = self.analyze_test_file(file_path)
            
            if "error" not in analysis:
                self.analysis_results["coverage_driven_tests"] += len(analysis["coverage_tests"])
                self.analysis_results["behavior_driven_tests"] += len(analysis["behavior_tests"])
                
                if analysis["suggestions"]:
                    self.analysis_results["conversion_suggestions"].append(analysis)
        
        return self.analysis_results
    
    def print_report(self) -> None:
        """Print a human-readable conversion report."""
        print("\n" + "=" * 60)
        print("TEST CONVERSION ANALYSIS")
        print("=" * 60 + "\n")
        
        total_tests = (
            self.analysis_results["coverage_driven_tests"] + 
            self.analysis_results["behavior_driven_tests"]
        )
        
        if total_tests == 0:
            print("No tests found.")
            return
        
        coverage_percent = (self.analysis_results["coverage_driven_tests"] / total_tests) * 100
        behavior_percent = (self.analysis_results["behavior_driven_tests"] / total_tests) * 100
        
        print(f"Total tests analyzed: {total_tests}")
        print(f"Coverage-driven tests: {self.analysis_results['coverage_driven_tests']} ({coverage_percent:.1f}%)")
        print(f"Behavior-driven tests: {self.analysis_results['behavior_driven_tests']} ({behavior_percent:.1f}%)")
        print(f"\nFiles needing conversion: {len(self.analysis_results['conversion_suggestions'])}")
        
        if self.analysis_results["conversion_suggestions"]:
            print("\nTop files to convert:")
            
            # Sort by number of coverage tests
            sorted_files = sorted(
                self.analysis_results["conversion_suggestions"],
                key=lambda x: len(x["coverage_tests"]),
                reverse=True
            )
            
            for file_info in sorted_files[:10]:
                file_path = Path(file_info["file"]).name
                coverage_count = len(file_info["coverage_tests"])
                print(f"  - {file_path}: {coverage_count} tests need conversion")
                
                # Show first few test names
                for test_name in file_info["coverage_tests"][:3]:
                    print(f"    • {test_name}")
                
                if len(file_info["coverage_tests"]) > 3:
                    print(f"    ... and {len(file_info['coverage_tests']) - 3} more")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert coverage-driven tests to behavior-driven tests"
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Path to test directory or file",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Actually convert files (default is analysis only)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Convert a specific file",
    )
    
    args = parser.parse_args()
    
    converter = TestConverter()
    
    if args.file:
        if args.convert:
            converter.convert_file(args.file, dry_run=False)
        else:
            analysis = converter.analyze_test_file(args.file)
            print(f"\nAnalysis for {args.file}:")
            print(f"Coverage-driven tests: {len(analysis['coverage_tests'])}")
            print(f"Behavior-driven tests: {len(analysis['behavior_tests'])}")
            
            if analysis['suggestions']:
                print("\nConversion suggestions:")
                for suggestion in analysis['suggestions']:
                    print(f"\n{suggestion['original_name']} -> {suggestion['suggested_name']}")
                    for note in suggestion['conversion_notes']:
                        print(f"  - {note}")
    else:
        report = converter.generate_report(args.path)
        converter.print_report()
        
        if args.convert and converter.analysis_results["conversion_suggestions"]:
            print("\nConverting files...")
            for file_info in converter.analysis_results["conversion_suggestions"]:
                converter.convert_file(Path(file_info["file"]), dry_run=False)


if __name__ == "__main__":
    main()