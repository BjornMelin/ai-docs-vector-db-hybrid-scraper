#!/usr/bin/env python3
"""
Manual docstring corruption fix for the specific pattern found in files.

This fixes the pattern where docstrings start with "", "" and end with "", ""
instead of proper triple quotes.
"""

import logging
import re
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_file_docstrings(content: str) -> str:
    """Fix docstring corruption in a file."""

    # Pattern 1: Fix module-level docstrings that start with "", ""
    # Pattern: "", "" at start of file followed by content and "", "" later
    pattern1 = r"^(\s*)\"\",\s*\"\"(.*?)\"\",\s*\"\"\s*$"
    replacement1 = r'\1"""\2"""'
    content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE | re.DOTALL)

    # Pattern 2: Fix inline docstring patterns like "", "text" ""
    pattern2 = r'\"\",\s*\"([^"]*?)\"\s*\"\"'
    replacement2 = r'"""\1"""'
    content = re.sub(pattern2, replacement2, content)

    # Pattern 3: Fix class/function docstrings
    pattern3 = r'(\s+)\"\",\s*\"([^"]*?)\"\s*\"\"'
    replacement3 = r'\1"""\2"""'
    content = re.sub(pattern3, replacement3, content)

    return content


def process_file(file_path: Path) -> bool:
    """Process a single file to fix docstring corruption."""
    try:
        with Path(file_path).open(encoding="utf-8") as f:
            original_content = f.read()

        if not original_content.strip():
            return True

        # Apply the fix
        fixed_content = fix_file_docstrings(original_content)

        # Only write if content changed
        if fixed_content != original_content:
            with Path(file_path).Path("w").open(encoding="utf-8") as f:
                f.write(fixed_content)
            logger.info(f"Fixed: {file_path}")
            return True

        return True

    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main execution function."""
    project_root = Path("/workspace/repos/ai-docs-vector-db-hybrid-scraper")

    # Start with a small sample to test
    test_files = [
        project_root / "src" / "models" / "vector_search.py",
        project_root / "src" / "infrastructure" / "container.py",
        project_root / "src" / "cli_worker.py",
    ]

    for file_path in test_files:
        if file_path.exists():
            logger.info(f"Processing: {file_path}")
            process_file(file_path)
        else:
            logger.warning(f"File not found: {file_path}")


if __name__ == "__main__":
    main()
