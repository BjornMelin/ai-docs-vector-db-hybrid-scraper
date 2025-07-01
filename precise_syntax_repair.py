#!/usr/bin/env python3
"""
Precise syntax repair tool for the specific corruption patterns found in the codebase.

This tool addresses the exact syntax corruption patterns identified:
1. Docstring corruption: comma-comma-quote to triple quotes
2. Function signature corruption: broken parameter lists and return types
3. Import statement corruption: broken multi-line imports
4. Function call corruption: arguments split improperly across lines
5. String literal corruption: malformed quotes and escaping
"""

import ast
import logging
import re
from pathlib import Path
from typing import Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_docstring_corruption(content: str) -> str:
    """Fix the systematic docstring corruption pattern."""
    # Fix the most common pattern: "", "" -> """
    content = re.sub(r"\"\",\s*\"\"", '"""', content)

    # Fix variations with newlines
    content = re.sub(r"\"\",\s*\n\s*\"\"", '"""', content)

    # Fix single-line docstring corruption
    content = re.sub(
        r"^(\s*)\"\",\s*\"(.+?)\"\s*\"\"$", r'\1"""\2"""', content, flags=re.MULTILINE
    )

    return content


def fix_function_signature_corruption(content: str) -> str:
    """Fix corrupted function signatures."""

    # Fix return type corruption: func(param -> Type:) -> func(param) -> Type:
    content = re.sub(
        r"def\s+(\w+)\s*\([^)]*\s+\->\s*([^:]+?):\s*\)", r"def \1(\2)", content
    )

    # Fix parameter corruption: def func(self, param: "type", other: "type) -> def func(self, param: type, other: type):
    content = re.sub(
        r"def\s+(\w+)\s*\(\s*([^)]+?)\s*\)\s*\->\s*([^:]+?)\s*:",
        r"def \1(\2) -> \3:",
        content,
    )

    # Fix missing colons in function definitions
    content = re.sub(
        r"def\s+(\w+)\s*\([^)]*\)\s*\->\s*([^:\n]+?)\s*$",
        r"def \1() -> \2:",
        content,
        flags=re.MULTILINE,
    )

    return content


def fix_import_corruption(content: str) -> str:
    """Fix corrupted import statements."""

    # Fix broken from imports: from module import () -> from module import
    content = re.sub(
        r"from\s+([^\s]+)\s+import\s+\(\)\s*\n\s*([^)]+?)\n",
        r"from \1 import (\n    \2\n)",
        content,
    )

    # Fix malformed parentheses in imports
    content = re.sub(
        r"from\s+([^\s]+)\s+import\s+\(\s*\n\s*([^)]+?)\s*\n\s*\)",
        r"from \1 import (\n    \2\n)",
        content,
    )

    return content


def fix_function_call_corruption(content: str) -> str:
    """Fix corrupted function calls."""

    # Fix logging.basicConfig() corruption
    content = re.sub(
        r"logging\.basicConfig\(\)\s*\n\s*(.*?)\n\s*\)",
        r"logging.basicConfig(\n    \1\n)",
        content,
        flags=re.DOTALL,
    )

    # Fix general function call corruption: func() \n args \n )
    content = re.sub(
        r"(\w+)\(\)\s*\n\s*([^)]+?)\n\s*\)", r"\1(\n    \2\n)", content, flags=re.DOTALL
    )

    # Fix asyncio.run() corruption
    content = re.sub(
        r"asyncio\.run\(\)\s*\n\s*([^)]+?)\n\s*\)",
        r"asyncio.run(\n    \1\n)",
        content,
        flags=re.DOTALL,
    )

    return content


def fix_field_definition_corruption(content: str) -> str:
    """Fix corrupted Pydantic Field definitions."""

    # Fix Field() corruption: Field() ... -> Field(...)
    content = re.sub(
        r"Field\(\)\s*\n\s*([^)]+?)\n\s*", r"Field(\1)", content, flags=re.DOTALL
    )

    # Fix ConfigDict() corruption
    content = re.sub(
        r"ConfigDict\(\)\s*\n\s*([^)]+?)\n\s*",
        r"ConfigDict(\1)",
        content,
        flags=re.DOTALL,
    )

    return content


def fix_string_literal_corruption(content: str) -> str:
    """Fix corrupted string literals."""

    # Fix quote corruption in annotations
    content = re.sub(r'\"([^"]+?)\"', r'"\1"', content)

    # Fix malformed f-strings and format strings
    content = re.sub(r'\"([^"]*?)\"\"([^"]*?)\"\"', r'"\1\2"', content)

    return content


def fix_class_definition_corruption(content: str) -> str:
    """Fix corrupted class definitions."""

    # Fix missing pass statements in empty class bodies
    content = re.sub(
        r"(class\s+\w+[^:]*:)\s*\n(\s*)([^#\n])", r"\1\n\2    pass\n\2\3", content
    )

    return content


def fix_bracket_and_paren_corruption(content: str) -> str:
    """Fix mismatched brackets and parentheses."""

    # Fix common bracket mismatches
    content = re.sub(r"\[\s*\]\s*([^,\n])", r"[\1]", content)
    content = re.sub(r"\{\s*\}\s*([^,\n])", r"{\1}", content)

    return content


def validate_syntax(content: str, filename: str) -> bool:
    """Validate Python syntax using AST."""
    try:
        ast.parse(content)
        return True
    except SyntaxError as e:
        logger.warning(f"Syntax error in {filename}: {e}")
        return False


def repair_file(file_path: Path) -> tuple[bool, str]:
    """Repair a single Python file."""
    try:
        # Read the file
        with Path(file_path).open(encoding="utf-8") as f:
            original_content = f.read()

        # Skip empty files
        if not original_content.strip():
            return True, "Empty file, skipped"

        # Apply all fixes in sequence
        content = original_content
        content = fix_docstring_corruption(content)
        content = fix_function_signature_corruption(content)
        content = fix_import_corruption(content)
        content = fix_function_call_corruption(content)
        content = fix_field_definition_corruption(content)
        content = fix_string_literal_corruption(content)
        content = fix_class_definition_corruption(content)
        content = fix_bracket_and_paren_corruption(content)

        # Check if we made any changes
        if content == original_content:
            return True, "No changes needed"

        # Validate syntax before writing
        if not validate_syntax(content, str(file_path)):
            return False, "Syntax validation failed after repair"

        # Write the repaired content
        with Path(file_path).Path("w").open(encoding="utf-8") as f:
            f.write(content)

        return True, "Successfully repaired"

    except (ValueError, TypeError, RuntimeError) as e:
        return False, f"Error repairing file: {e}"


def main():
    """Main execution function."""
    project_root = Path("/workspace/repos/ai-docs-vector-db-hybrid-scraper")

    # Find all Python files
    python_files = list(project_root.rglob("*.py"))

    logger.info(f"Found {len(python_files)} Python files to process")

    success_count = 0
    error_count = 0

    for file_path in python_files:
        # Skip certain directories
        if any(
            skip_dir in str(file_path)
            for skip_dir in [".git", "__pycache__", ".pytest_cache", "venv"]
        ):
            continue

        success, message = repair_file(file_path)

        if success:
            success_count += 1
            if "Successfully repaired" in message:
                logger.info(f"REPAIRED: {file_path.relative_to(project_root)}")
        else:
            error_count += 1
            logger.error(f"FAILED: {file_path.relative_to(project_root)} - {message}")

    logger.info("\nSummary:")
    logger.info(f"Total files processed: {len(python_files)}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Errors: {error_count}")


if __name__ == "__main__":
    main()
