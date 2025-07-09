#!/usr/bin/env python3
"""Script to help migrate from asyncio.gather to TaskGroup."""

import re
import sys
from pathlib import Path


def check_imports(content: str) -> tuple[bool, str]:
    """Check if async_utils import is present and add if needed."""
    has_import = "from src.utils.async_utils import gather_with_taskgroup" in content

    if has_import:
        return True, content

    # Find a good place to add the import
    import_lines = []
    other_lines = []
    in_imports = True

    for line in content.split("\n"):
        if in_imports and (line.startswith(("import ", "from ")) or not line.strip()):
            import_lines.append(line)
        else:
            in_imports = False
            other_lines.append(line)

    # Add our import after other imports
    if import_lines:
        # Find the last non-empty import line
        last_import_idx = len(import_lines) - 1
        while last_import_idx > 0 and not import_lines[last_import_idx].strip():
            last_import_idx -= 1

        import_lines.insert(last_import_idx + 1, "\nfrom src.utils.async_utils import gather_with_taskgroup")

    return False, "\n".join(import_lines + other_lines)


def migrate_file(file_path: Path) -> bool:
    """Migrate a single file from asyncio.gather to gather_with_taskgroup."""
    try:
        content = file_path.read_text()
        original_content = content

        # Check if file uses asyncio.gather
        if "asyncio.gather" not in content:
            return False

        # Add import if needed
        had_import, content = check_imports(content)

        # Replace asyncio.gather with gather_with_taskgroup
        content = re.sub(
            r'\basyncio\.gather\b',
            'gather_with_taskgroup',
            content
        )

        # Write back if changed
        if content != original_content:
            file_path.write_text(content)
            print(f"✓ Migrated: {file_path}")
            return True

        return False

    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False


def main():
    """Main migration script."""
    src_dir = Path("/workspace/repos/ai-docs-vector-db-hybrid-scraper/src")

    if not src_dir.exists():
        print("Error: src directory not found!")
        sys.exit(1)

    # Find all Python files
    py_files = list(src_dir.rglob("*.py"))

    migrated_count = 0
    skipped_count = 0

    print(f"Scanning {len(py_files)} Python files...")

    for file_path in py_files:
        # Skip the async_utils.py file itself
        if file_path.name == "async_utils.py":
            continue

        if migrate_file(file_path):
            migrated_count += 1
        else:
            skipped_count += 1

    print("\nMigration complete!")
    print(f"  Migrated: {migrated_count} files")
    print(f"  Skipped: {skipped_count} files")

    # Show remaining instances
    remaining = []
    for file_path in py_files:
        content = file_path.read_text()
        if "asyncio.gather" in content:
            remaining.append(file_path)

    if remaining:
        print(f"\nWarning: {len(remaining)} files still contain asyncio.gather:")
        for path in remaining[:5]:
            print(f"  - {path}")
        if len(remaining) > 5:
            print(f"  ... and {len(remaining) - 5} more")


if __name__ == "__main__":
    main()
