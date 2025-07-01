#!/usr/bin/env python3
"""Targeted syntax fixer for specific patterns found in the codebase."""

import ast
import re
import subprocess
from pathlib import Path
from typing import Any


def fix_docstring_patterns(content: str) -> str:
    """Fix common docstring syntax errors."""
    # Fix "", "" pattern to proper """
    content = re.sub(
        r'^(\s*)"", ""([^"]*)"", ""', r'\1"""\2"""', content, flags=re.MULTILINE
    )

    # Fix single line docstrings with "", ""
    content = re.sub(r'(\s*)"", "([^"]*?)""', r'\1"""\2"""', content)

    # Fix unterminated triple quotes
    lines = content.split("\n")
    fixed_lines = []
    in_docstring = False
    docstring_indent = 0

    for i, line in enumerate(lines):
        # Check for start of docstring
        if '"""' in line and line.count('"""') == 1:
            if not in_docstring:
                in_docstring = True
                docstring_indent = len(line) - len(line.lstrip())
                fixed_lines.append(line)
            else:
                # End of docstring
                in_docstring = False
                fixed_lines.append(line)
        elif in_docstring:
            # Inside docstring - check if we need to close it
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if (
                next_line.strip()
                and not next_line.startswith(" " * (docstring_indent + 4))
                and not next_line.strip().startswith("#")
            ):
                # Need to close docstring
                fixed_lines.append(line)
                fixed_lines.append(" " * docstring_indent + '"""')
                in_docstring = False
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_function_definitions(content: str) -> str:
    """Fix malformed function definitions."""
    # Fix function definitions with malformed parameters
    # Pattern: def func(self, param: "type" -> return_type:):
    content = re.sub(
        r"def\s+(\w+)\(([^)]*)\s+->\s*([^:)]+)\)?\s*:?\s*\):",
        r"def \1(\2) -> \3:",
        content,
    )

    # Fix function definitions with missing colons
    content = re.sub(
        r"def\s+(\w+)\(([^)]*)\)\s*->\s*([^:)]+)\s*$",
        r"def \1(\2) -> \3:",
        content,
        flags=re.MULTILINE,
    )

    # Fix async function definitions
    content = re.sub(
        r"async\s+def\s+(\w+)\(([^)]*)\s+->\s*([^:)]+)\)?\s*:?\s*\):",
        r"async def \1(\2) -> \3:",
        content,
    )

    return content


def fix_parentheses_and_brackets(content: str) -> str:
    """Fix unmatched parentheses and brackets."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix unmatched opening parentheses at end of line
        if line.strip().endswith("(") and line.count("(") > line.count(")"):
            # Look for the next line to see if it should be combined
            fixed_lines.append(line.rstrip("("))

        # Fix dangling closing parentheses
        elif line.strip() == ")" or line.strip() == "))":
            # Skip standalone closing parentheses
            continue

        # Fix list/dict literals with syntax errors
        elif line.strip().endswith("[") or line.strip().endswith("{"):
            # Handle incomplete collections
            if "[" in line and "]" not in line:
                line = line.rstrip() + "]"
            elif "{" in line and "}" not in line:
                line = line.rstrip() + "}"
            fixed_lines.append(line)

        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_import_statements(content: str) -> str:
    """Fix malformed import statements."""
    # Fix incomplete imports
    content = re.sub(
        r"from\s+(\S+)\s+import\s+\(\s*$",
        r"from \1 import (",
        content,
        flags=re.MULTILINE,
    )

    # Fix imports with syntax errors
    content = re.sub(r"import\s+\(\s*$", "", content, flags=re.MULTILINE)

    return content


def fix_string_literals(content: str) -> str:
    """Fix malformed string literals."""
    # Fix string literals with quotes issues
    content = re.sub(r'"([^"]*)"([^"]*)"', r'"\1\2"', content)

    # Fix f-string patterns
    content = re.sub(r'f"([^"]*)\s*"([^"]*)"', r'f"\1\2"', content)

    return content


def fix_field_definitions(content: str) -> str:
    """Fix Pydantic field definitions with syntax errors."""
    # Fix Field() definitions with syntax errors
    # Pattern: field: "type = Field(..." -> field: type = Field(...)
    content = re.sub(r'(\w+):\s*"([^"]+)\s*=\s*Field\(', r"\1: \2 = Field(", content)

    # Fix Field definitions with missing quotes
    content = re.sub(r"Field\(\s*\.\.\.\s*,\s*([^)]+)\s*\)", r"Field(..., \1)", content)

    return content


def fix_logical_operators(content: str) -> str:
    """Fix malformed logical operators."""
    # Fix list definitions with syntax errors
    content = re.sub(r"list\[\s*\]\s*", "list[", content)

    # Fix dict definitions
    content = re.sub(r"dict\[\s*\]\s*", "dict[", content)

    # Fix Literal definitions
    content = re.sub(r"Literal\[\s*\]\s*", "Literal[", content)

    return content


def fix_class_syntax(content: str) -> str:
    """Fix class definition syntax errors."""
    # Fix class definitions with syntax errors in inheritance
    content = re.sub(
        r"class\s+(\w+)\([^)]*\)?\s*:\s*$", r"class \1:", content, flags=re.MULTILINE
    )

    return content


def fix_syntax_file(file_path: Path) -> bool:
    """Fix syntax errors in a single file."""
    try:
        with Path(file_path).open(encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply all fixes
        content = fix_docstring_patterns(content)
        content = fix_function_definitions(content)
        content = fix_parentheses_and_brackets(content)
        content = fix_import_statements(content)
        content = fix_string_literals(content)
        content = fix_field_definitions(content)
        content = fix_logical_operators(content)
        content = fix_class_syntax(content)

        # Only write if content changed
        if content != original_content:
            # Try to parse with AST to validate
            try:
                ast.parse(content)
                with Path(file_path).Path("w").open(encoding="utf-8") as f:
                    f.write(content)
                return True
            except SyntaxError:
                # If AST parsing fails, don't write the changes
                print(f"AST validation failed for {file_path}, skipping")
                return False

        return False

    except (ValueError, TypeError, RuntimeError) as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Run targeted syntax fixes."""
    print("Running targeted syntax fixes...")

    # Find Python files with syntax errors
    try:
        result = subprocess.run(
            [
                "python",
                "-m",
                "ruff",
                "check",
                "--select=E999",
                "--output-format=json",
                ".",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        if result.returncode == 0:
            print("No syntax errors found!")
            return

        # Parse the output to get files with syntax errors
        import json

        errors = json.loads(result.stdout)

        error_files = set()
        for error in errors:
            if error.get("code") == "E999":  # Syntax error
                error_files.add(Path(error["filename"]))

        print(f"Found {len(error_files)} files with syntax errors")

        # Fix each file
        fixed_count = 0
        for file_path in error_files:
            if fix_syntax_file(file_path):
                fixed_count += 1
                print(f"Fixed: {file_path}")

        print(f"Fixed {fixed_count} files")

    except (ValueError, TypeError, RuntimeError) as e:
        print(f"Error running targeted fixes: {e}")
        # Fallback to manual file list
        files_to_fix = [
            "src/models/vector_search.py",
            "src/infrastructure/container.py",
            "src/cli_worker.py",
            "src/infrastructure/database/monitoring.py",
            "src/infrastructure/clients/firecrawl_client.py",
        ]

        fixed_count = 0
        for file_name in files_to_fix:
            file_path = Path(file_name)
            if file_path.exists() and fix_syntax_file(file_path):
                fixed_count += 1
                print(f"Fixed: {file_path}")

        print(f"Fixed {fixed_count} files using manual list")


if __name__ == "__main__":
    main()
