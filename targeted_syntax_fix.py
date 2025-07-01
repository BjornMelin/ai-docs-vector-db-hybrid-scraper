#!/usr/bin/env python3
"""
Targeted syntax fix script for specific patterns found in the codebase.

This script addresses the most common syntax errors:
1. Exception class definitions missing colon
2. Function definitions with misplaced parameters
3. Missing closing parentheses/brackets
4. Malformed function calls
5. String literal issues
"""

import ast
import re
from pathlib import Path
from typing import list, tuple


def validate_python_syntax(file_path: Path) -> tuple[bool, str]:
    """Check if a Python file has valid syntax."""
    try:
        with Path(file_path).open(encoding="utf-8") as f:
            content = f.read()
        ast.parse(content)
        return True, ""
    except SyntaxError as e:
        return False, str(e)
    except (ValueError, TypeError, RuntimeError) as e:
        return False, f"Error reading file: {e}"


def fix_specific_patterns(content: str) -> str:
    """Fix specific syntax patterns found in the codebase."""
    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        # Pattern 1: Exception class definitions missing colon
        if re.match(r"^\s*class\s+\w+\(Exception\s*$", line):
            line = line + "):"

        # Pattern 2: Exception class definitions with colon after parenthesis
        line = re.sub(r"class\s+(\w+)\(Exception:\)", r"class \1(Exception):", line)

        # Pattern 3: Function definitions with self in wrong place
        line = re.sub(
            r"def\s+(\w+)\(\s*self:\s*(\w+[^)]*)\)", r"def \1(self, \2)", line
        )

        # Pattern 4: Method signatures with misplaced self
        line = re.sub(
            r"def\s+(\w+)\(\s*(\w+:\s*[^)]+)\s*\)\s*->\s*([^:]+):",
            r"def \1(self, \2) -> \3:",
            line,
        )

        # Pattern 5: Fix incomplete function definitions
        if re.match(r"^\s*def\s+\w+\([^)]*\s*$", line) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.startswith("self:") or "-> " in next_line:
                # Merge lines
                line = line.rstrip() + " " + next_line
                lines[i + 1] = ""  # Remove the next line

        # Pattern 6: Fix string dictionary values with missing commas
        line = re.sub(r'(["\'])([^"\']*)\1\s*(["\'][^"\']*["\'])', r"\1\2\1, \3", line)

        # Pattern 7: Fix missing closing parentheses in lists/dicts
        if line.strip().endswith("[") or line.strip().endswith("{"):
            # Look ahead for content without proper closing
            j = i + 1
            open_brackets = 1 if line.strip().endswith("[") else 0
            open_braces = 1 if line.strip().endswith("{") else 0

            while j < len(lines) and j < i + 5:  # Look max 5 lines ahead
                next_line = lines[j]
                open_brackets += next_line.count("[") - next_line.count("]")
                open_braces += next_line.count("{") - next_line.count("}")

                if open_brackets == 0 and open_braces == 0:
                    break

                # If we find a line that should close but doesn't
                if next_line.strip() and not next_line.strip().endswith(
                    ("]", "}", ",", ")")
                ):
                    if open_brackets > 0:
                        lines[j] = next_line.rstrip() + "]"
                    elif open_braces > 0:
                        lines[j] = next_line.rstrip() + "}"
                    break
                j += 1

        # Pattern 8: Fix incomplete async with statements
        if "async with" in line and "as" in line and line.strip().endswith(":"):
            # Check if there's a missing parenthesis
            if "(" in line and ")" not in line:
                line = line.rstrip(":") + "):"

        # Pattern 9: Fix function calls with missing closing parentheses
        if re.search(r"\w+\([^)]*$", line.strip()) and not line.strip().endswith(
            ("\\", ",")
        ):
            # Simple case - just add closing parenthesis
            line = line.rstrip() + ")"

        # Pattern 10: Fix logger calls with missing parentheses
        if re.search(r"logger\.(info|debug|warning|error|exception)\(\s*$", line):
            line = re.sub(r"(logger\.\w+)\(\s*$", r'\1("")', line)

        fixed_lines.append(line)

    # Post-processing: Remove empty lines that were created during merging
    fixed_lines = [line for line in fixed_lines if line is not None]

    return "\n".join(fixed_lines)


def fix_function_parameters(content: str) -> str:
    """Fix function parameter issues."""
    # Fix method definitions missing 'self'
    content = re.sub(
        r"def\s+(\w+)\(\s*(\w+:\s*\w+[^)]*)\)\s*:", r"def \1(self, \2):", content
    )

    # Fix async method definitions missing 'self'
    content = re.sub(
        r"async def\s+(\w+)\(\s*(\w+:\s*\w+[^)]*)\)\s*:",
        r"async def \1(self, \2):",
        content,
    )

    # Fix incomplete return type annotations
    content = re.sub(r"(\) -> [^:]+):\s*$", r"\1:", content, flags=re.MULTILINE)

    return content


def fix_string_and_bracket_issues(content: str) -> str:
    """Fix string literal and bracket matching issues."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix unterminated string literals at end of line
        if re.search(r'r?"[^"]*,\s*$', line):
            line = re.sub(r'(r?"[^"]*),\s*$', r'\1",', line)

        # Fix missing quotes around dictionary values
        line = re.sub(r': ([^"\'\[\{][^,\}]+)([,\}])', r': "\1"\2', line)

        # Fix incomplete function calls in parentheses
        if "(" in line and ")" not in line and not line.strip().endswith("\\"):
            # Count unclosed parentheses
            open_parens = line.count("(") - line.count(")")
            if open_parens > 0:
                line = line + ")" * open_parens

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def apply_targeted_fixes(file_path: Path) -> bool:
    """Apply targeted fixes to a file."""
    try:
        with Path(file_path).open(encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply fixes in sequence
        content = fix_specific_patterns(content)
        content = fix_function_parameters(content)
        content = fix_string_and_bracket_issues(content)

        if content != original_content:
            with Path(file_path).Path("w").open(encoding="utf-8") as f:
                f.write(content)
            return True

        return False

    except (ValueError, TypeError, RuntimeError) as e:
        print(f"Error processing {file_path}: {e}")
        return False


def get_files_with_syntax_errors() -> list[Path]:
    """Get list of Python files with syntax errors."""
    python_files = []

    # Get all Python files
    patterns = ["src/**/*.py", "examples/**/*.py"]
    for pattern in patterns:
        python_files.extend(Path().glob(pattern))

    # Filter to only files with syntax errors
    error_files = []
    for file_path in python_files:
        is_valid, _ = validate_python_syntax(file_path)
        if not is_valid:
            error_files.append(file_path)

    return error_files


def main():
    """Main function to apply targeted fixes."""
    print("🎯 Targeted Syntax Error Fix")
    print("=" * 40)

    error_files = get_files_with_syntax_errors()
    print(f"Found {len(error_files)} files with syntax errors")

    if len(error_files) == 0:
        print("✅ No syntax errors found!")
        return

    fixed_count = 0

    # Process first 20 files to avoid overwhelming output
    for file_path in error_files[:20]:
        print(f"Fixing: {file_path}")

        if apply_targeted_fixes(file_path):
            # Verify the fix
            is_valid_after, error_after = validate_python_syntax(file_path)
            if is_valid_after:
                print("  ✅ Fixed successfully")
                fixed_count += 1
            else:
                print(f"  ❌ Still has errors: {error_after[:60]}...")
        else:
            print("  ⚠️  No changes made")

    print("\n📊 Batch Results:")
    print(f"  Files processed: {min(20, len(error_files))}")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Remaining files with errors: {len(error_files) - fixed_count}")

    if len(error_files) > 20:
        print(f"\n⚠️  {len(error_files) - 20} more files need fixing")
        print("Run the script again to process more files")


if __name__ == "__main__":
    main()
