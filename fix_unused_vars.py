#!/usr/bin/env python3
"""
Simple script to fix only F841 violations where exception variables are truly unused
"""

import ast
import re
from pathlib import Path


def find_unused_exception_vars(file_path):
    """Find exception variables that are captured but never used."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
    except Exception:
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        # Skip files with syntax errors
        return []

    lines = content.split("\n")
    unused_vars = []

    class ExceptionVisitor(ast.NodeVisitor):
        def visit_ExceptHandler(self, node):
            if node.name:
                var_name = node.name

                # Check if this variable is used anywhere in the except block
                except_block_code = ast.unparse(node)

                # Look for usage of the variable name in the block (excluding the declaration)
                # Simple heuristic: if variable appears only once, it's likely unused
                usage_count = except_block_code.count(var_name)

                # The variable appears once in the "as var_name:" part
                # If it appears only once total, it's unused
                if usage_count == 1:
                    unused_vars.append((node.lineno, var_name))

            self.generic_visit(node)

    visitor = ExceptionVisitor()
    visitor.visit(tree)

    return unused_vars


def fix_file(file_path):
    """Fix unused exception variables in a file."""
    unused_vars = find_unused_exception_vars(file_path)
    if not unused_vars:
        return 0

    with open(file_path, "r") as f:
        lines = f.readlines()

    fixed_count = 0

    # Process from bottom to top to avoid line number changes
    for line_num, var_name in sorted(unused_vars, reverse=True):
        # Get the line (1-indexed to 0-indexed)
        line_idx = line_num - 1
        line = lines[line_idx]

        # Replace "as var_name:" with ":"
        pattern = rf"\bas\s+{re.escape(var_name)}\s*:"
        new_line = re.sub(pattern, ":", line)

        if new_line != line:
            lines[line_idx] = new_line
            fixed_count += 1
            print(f"Fixed {file_path}:{line_num} - removed unused '{var_name}'")

    if fixed_count > 0:
        with open(file_path, "w") as f:
            f.writelines(lines)

    return fixed_count


def main():
    """Main function to fix F841 violations."""
    total_fixed = 0

    # Find all Python files in src and tests
    python_files = list(Path("src").rglob("*.py")) + list(Path("tests").rglob("*.py"))

    for file_path in python_files:
        if file_path.name.startswith("."):
            continue

        try:
            fixed = fix_file(file_path)
            total_fixed += fixed
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"\nTotal unused exception variables fixed: {total_fixed}")


if __name__ == "__main__":
    main()
