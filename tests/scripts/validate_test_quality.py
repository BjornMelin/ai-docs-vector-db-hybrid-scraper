#!/usr/bin/env python3
"""Automated test quality validation script.

Checks for anti-patterns and quality issues in test files:
- Forbidden naming patterns (, , )
- Excessive mocking complexity
- File size violations
- Directory depth violations
- Coverage-driven testing patterns
"""

import argparse
import json
import re
import sys
from pathlib import Path


# Anti-pattern definitions
FORBIDDEN_PATTERNS = {
    "enhanced": re.compile(r"\benhanced\b", re.IGNORECASE),
    "modern": re.compile(r"\bmodern\b", re.IGNORECASE),
    "advanced": re.compile(r"\badvanced\b", re.IGNORECASE),
    "coverage_driven": re.compile(r"# coverage:|# pragma: no cover|test_for_coverage"),
    "implementation_detail": re.compile(r"test_.*_internal|test_private_|_test_helper"),
    "shared_state": re.compile(
        r"@pytest\.fixture.*scope=[\"']module[\"']|@pytest\.fixture.*scope=[\"']session[\"']"
    ),
    "magic_values": re.compile(
        r"assert.*==\s*\d{4,}|assert.*==\s*[\"'][a-f0-9]{32,}[\"']"
    ),
    "timing_dependent": re.compile(r"time\.sleep|sleep\(|asyncio\.sleep\([^0]\)"),
}

# Quality thresholds
MAX_FILE_LINES = 500
MAX_MOCK_DEPTH = 3
MAX_DIRECTORY_DEPTH = 4
MAX_TEST_FUNCTION_LINES = 50


class TestQualityValidator:
    """Validates test quality and identifies anti-patterns."""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.issues: dict[str, list[dict[str, any]]] = {
            "naming": [],
            "structure": [],
            "mocking": [],
            "patterns": [],
        }
        self.stats = {
            "total_files": 0,
            "files_with_issues": 0,
            "total_issues": 0,
        }

    def validate_file(self, file_path: Path) -> list[dict[str, any]]:
        """Validate a single test file."""
        file_issues = []

        try:
            content = file_path.read_text()
            lines = content.splitlines()

            # Check file size
            if len(lines) > MAX_FILE_LINES:
                file_issues.append(
                    {
                        "type": "structure",
                        "severity": "warning",
                        "message": f"File exceeds {MAX_FILE_LINES} lines ({len(lines)} lines)",
                        "file": str(file_path),
                    }
                )

            # Check directory depth
            depth = len(file_path.relative_to(self.root_path).parts) - 1
            if depth > MAX_DIRECTORY_DEPTH:
                file_issues.append(
                    {
                        "type": "structure",
                        "severity": "error",
                        "message": f"Directory depth exceeds {MAX_DIRECTORY_DEPTH} ({depth} levels)",
                        "file": str(file_path),
                    }
                )

            # Check for forbidden patterns
            for pattern_name, pattern in FORBIDDEN_PATTERNS.items():
                matches = list(pattern.finditer(content))
                if matches:
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        file_issues.append(
                            {
                                "type": "naming"
                                if pattern_name in ["enhanced", "modern", "advanced"]
                                else "patterns",
                                "severity": "error",
                                "pattern": pattern_name,
                                "message": f"Found '{match.group()}' at line {line_num}",
                                "file": str(file_path),
                                "line": line_num,
                            }
                        )

            # Check mock complexity
            mock_issues = self._check_mock_complexity(content, file_path)
            file_issues.extend(mock_issues)

            # Check test function size
            function_issues = self._check_function_size(content, file_path)
            file_issues.extend(function_issues)

        except Exception as e:
            file_issues.append(
                {
                    "type": "error",
                    "severity": "error",
                    "message": f"Failed to analyze file: {str(e)}",
                    "file": str(file_path),
                }
            )

        return file_issues

    def _check_mock_complexity(
        self, content: str, file_path: Path
    ) -> list[dict[str, any]]:
        """Check for excessive mocking complexity."""
        issues = []

        # Count mock depth (nested mocks)
        mock_pattern = re.compile(r"Mock\(|MagicMock\(|patch\(|mock\.")
        mock_lines = [
            i
            for i, line in enumerate(content.splitlines(), 1)
            if mock_pattern.search(line)
        ]

        # Simple heuristic: if too many mocks in a small area, likely complex
        if len(mock_lines) > 10:
            for i in range(len(mock_lines) - 5):
                if mock_lines[i + 5] - mock_lines[i] < 20:  # 6 mocks in 20 lines
                    issues.append(
                        {
                            "type": "mocking",
                            "severity": "warning",
                            "message": f"High mock density detected around line {mock_lines[i]}",
                            "file": str(file_path),
                            "line": mock_lines[i],
                        }
                    )
                    break

        return issues

    def _check_function_size(
        self, content: str, file_path: Path
    ) -> list[dict[str, any]]:
        """Check for oversized test functions."""
        issues = []

        function_pattern = re.compile(r"^\s*def\s+test_\w+", re.MULTILINE)
        lines = content.splitlines()

        for match in function_pattern.finditer(content):
            start_line = content[: match.start()].count("\n") + 1
            indent = len(match.group()) - len(match.group().lstrip())

            # Find function end
            end_line = start_line
            for i in range(start_line, len(lines)):
                line = lines[i]
                if (
                    line.strip()
                    and not line.startswith(" " * (indent + 1))
                    and i > start_line
                ):
                    end_line = i
                    break
            else:
                end_line = len(lines)

            function_lines = end_line - start_line
            if function_lines > MAX_TEST_FUNCTION_LINES:
                issues.append(
                    {
                        "type": "structure",
                        "severity": "warning",
                        "message": f"Test function exceeds {MAX_TEST_FUNCTION_LINES} lines ({function_lines} lines)",
                        "file": str(file_path),
                        "line": start_line,
                        "function": match.group().strip(),
                    }
                )

        return issues

    def validate_directory(self, directory: Path) -> None:
        """Validate all test files in a directory."""
        test_files = list(directory.rglob("test_*.py"))
        self.stats["total_files"] = len(test_files)

        for file_path in test_files:
            issues = self.validate_file(file_path)
            if issues:
                self.stats["files_with_issues"] += 1
                self.stats["total_issues"] += len(issues)

                for issue in issues:
                    issue_type = issue["type"]
                    if issue_type in self.issues:
                        self.issues[issue_type].append(issue)

    def generate_report(self) -> dict[str, any]:
        """Generate validation report."""
        return {
            "summary": {
                "total_files": self.stats["total_files"],
                "files_with_issues": self.stats["files_with_issues"],
                "total_issues": self.stats["total_issues"],
                "issue_breakdown": {
                    category: len(issues) for category, issues in self.issues.items()
                },
            },
            "issues": self.issues,
        }

    def print_report(self) -> None:
        """Print human-readable report."""
        print("\n" + "=" * 60)
        print("TEST QUALITY VALIDATION REPORT")
        print("=" * 60 + "\n")

        print(f"Total files analyzed: {self.stats['total_files']}")
        print(f"Files with issues: {self.stats['files_with_issues']}")
        print(f"Total issues found: {self.stats['total_issues']}\n")

        if self.stats["total_issues"] == 0:
            print("✅ No quality issues found!")
            return

        # Print issues by category
        for category, issues in self.issues.items():
            if not issues:
                continue

            print(f"\n{category.upper()} ISSUES ({len(issues)}):")
            print("-" * 40)

            # Group by file
            by_file: dict[str, list[dict]] = {}
            for issue in issues:
                file_path = issue["file"]
                if file_path not in by_file:
                    by_file[file_path] = []
                by_file[file_path].append(issue)

            for file_path, file_issues in sorted(by_file.items()):
                print(f"\n  {file_path}:")
                for issue in file_issues[:5]:  # Limit to first 5 per file
                    severity_icon = "❌" if issue["severity"] == "error" else "⚠️"
                    line_info = f" (line {issue['line']})" if "line" in issue else ""
                    print(f"    {severity_icon} {issue['message']}{line_info}")

                if len(file_issues) > 5:
                    print(f"    ... and {len(file_issues) - 5} more issues")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate test quality and check for anti-patterns"
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Path to test directory or file (default: current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if issues found",
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path '{args.path}' does not exist", file=sys.stderr)
        sys.exit(1)

    validator = TestQualityValidator(
        args.path if args.path.is_dir() else args.path.parent
    )

    if args.path.is_file():
        issues = validator.validate_file(args.path)
        if issues:
            validator.stats["files_with_issues"] = 1
            validator.stats["total_issues"] = len(issues)
            for issue in issues:
                validator.issues[issue["type"]].append(issue)
    else:
        validator.validate_directory(args.path)

    if args.json:
        print(json.dumps(validator.generate_report(), indent=2))
    else:
        validator.print_report()

    if args.strict and validator.stats["total_issues"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
