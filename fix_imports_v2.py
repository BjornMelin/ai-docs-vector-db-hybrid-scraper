#!/usr/bin/env python3
"""Script to help fix PLC0415 import violations."""

import json
import subprocess
from typing import Dict, List


def get_violations() -> List[Dict]:
    """Get PLC0415 violations from ruff."""
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "ruff",
                "check",
                ".",
                "--select=PLC0415",
                "--output-format=json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 and not result.stdout:
            return []
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Error getting violations: {e}")
        return []


def fix_function_level_imports(file_path: str, violations: List[Dict]) -> bool:
    """Fix imports that are inside functions but can be moved to top level."""
    try:
        with open(file_path) as f:
            content = f.read()
            lines = content.splitlines()

        imports_to_add = set()
        lines_to_remove = set()

        for violation in violations:
            line_num = violation["location"]["row"] - 1  # Convert to 0-based
            if line_num < len(lines):
                line = lines[line_num].strip()

                # Check if this is a simple import that can be moved
                if (
                    line.startswith("from ") or line.startswith("import ")
                ) and not line.startswith("#"):
                    # Check if it's not a conditional import
                    context_start = max(0, line_num - 3)
                    context_lines = lines[context_start:line_num]
                    context = " ".join(l.strip() for l in context_lines)

                    # Skip if in try/except or if statement context
                    if not any(
                        keyword in context.lower()
                        for keyword in ["try:", "except:", "if ", "elif "]
                    ):
                        imports_to_add.add(line)
                        lines_to_remove.add(line_num)

        if not imports_to_add:
            return False

        # Find where to insert imports (after existing imports)
        insert_position = 0
        existing_imports = set()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and not stripped.startswith(
                "#"
            ):
                existing_imports.add(stripped)
                insert_position = i + 1
            elif (
                stripped
                and not stripped.startswith(("#", '"""', "'''"))
                and insert_position > 0
            ):
                break

        # Skip if imports already exist
        new_imports = imports_to_add - existing_imports
        if not new_imports:
            return False

        # Build new content
        new_lines = []

        # Add lines up to insert position
        new_lines.extend(lines[:insert_position])

        # Add new imports
        for import_stmt in sorted(new_imports):
            new_lines.append(import_stmt)

        # Add remaining lines, skipping the ones we're removing
        for i, line in enumerate(lines[insert_position:], start=insert_position):
            if i not in lines_to_remove:
                new_lines.append(line)

        # Write back
        with open(file_path, "w") as f:
            f.write("\n".join(new_lines))

        return len(lines_to_remove) > 0

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    """Main function."""
    print("Getting PLC0415 violations...")
    violations = get_violations()

    if not violations:
        print("No PLC0415 violations found!")
        return

    print(f"Found {len(violations)} violations")

    # Group by file
    files_with_violations = {}
    for violation in violations:
        filename = violation["filename"]
        if filename not in files_with_violations:
            files_with_violations[filename] = []
        files_with_violations[filename].append(violation)

    print(f"Files with violations: {len(files_with_violations)}")

    # Focus on source files first
    source_files = [
        f
        for f in files_with_violations
        if not f.endswith((".py.bak", ".py~")) and "/src/" in f and "/tests/" not in f
    ]

    print(f"Source files to fix: {len(source_files)}")

    fixed_count = 0
    for file_path in sorted(source_files):
        print(f"Trying to fix: {file_path}")
        if fix_function_level_imports(file_path, files_with_violations[file_path]):
            fixed_count += 1
            print(f"  ✓ Fixed {file_path}")
        else:
            print(f"  - No fixes applied to {file_path}")

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
