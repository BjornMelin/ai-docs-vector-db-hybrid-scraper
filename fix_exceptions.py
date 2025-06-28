#!/usr/bin/env python3
"""Fix exception handling patterns intelligently."""

import re
import subprocess
import sys
from pathlib import Path


def fix_exception_handling(file_path):
    """Fix exception handling in a single file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except Exception:
        print(f"Error reading {file_path}: {e}")
        return False

    # Find all except Exception: blocks and check if they need 'as e'
    original_content = content

    # Pattern to find except Exception: followed by the block
    pattern = r"(except Exception):(\s*\n(?:\s+[^\n]*\n)*?)"

    def replacer(match):
        except_line = match.group(1)
        block_content = match.group(2)

        # Check if the block uses 'e' variable
        if re.search(r"\be\b", block_content):
            # Block uses 'e', so we need 'as e'
            return f"{except_line} as e:{block_content}"
        else:
            # Block doesn't use 'e', keep as is
            return f"{except_line}:{block_content}"

    content = re.sub(pattern, replacer, content)

    # Now handle the reverse - remove 'as e' where 'e' is not used
    pattern_with_e = r"(except Exception) as e:(\s*\n(?:\s+[^\n]*\n)*?)"

    def reverse_replacer(match):
        except_line = match.group(1)
        block_content = match.group(2)

        # Check if the block actually uses 'e' variable
        if not re.search(r"\be\b", block_content):
            # Block doesn't use 'e', remove 'as e'
            return f"{except_line}:{block_content}"
        else:
            # Block uses 'e', keep 'as e'
            return f"{except_line} as e:{block_content}"

    content = re.sub(pattern_with_e, reverse_replacer, content)

    if content != original_content:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception:
            print(f"Error writing {file_path}: {e}")
            return False

    return False


def main():
    """Fix exception handling in all Python files."""
    changed_files = 0

    # Find all Python files
    python_files = list(Path().rglob("*.py"))

    for file_path in python_files:
        if fix_exception_handling(file_path):
            changed_files += 1
            print(f"Fixed: {file_path}")

    print(f"Fixed exception handling in {changed_files} files")


if __name__ == "__main__":
    main()
