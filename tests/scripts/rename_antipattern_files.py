#!/usr/bin/env python3
"""Automated file renaming script for removing anti-pattern names.

Renames files containing forbidden patterns (, , )
to more descriptive names based on their actual functionality.
"""

import argparse
import re
import sys
from pathlib import Path


# Anti-pattern words to remove
FORBIDDEN_WORDS = [
    "enhanced",
    "modern",
    "advanced",
    "ultimate",
    "next_gen",
    "sophisticated",
]

# Replacement mapping for common patterns
REPLACEMENTS = {
    "enhanced": "improved",
    "modern": "current",
    "advanced": "comprehensive",
    "ultimate": "complete",
    "next_gen": "new",
    "sophisticated": "detailed",
}


class FileRenamer:
    """Handles renaming of files with anti-pattern names."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.renamed_files: list[tuple[Path, Path]] = []
        self.skipped_files: list[tuple[Path, str]] = []

    def suggest_new_name(self, old_path: Path) -> Path:
        """Suggest a  name for a file by removing anti-patterns."""
        old_name = old_path.stem
        new_name = old_name

        # Replace forbidden words
        for word in FORBIDDEN_WORDS:
            # Case-insensitive replacement
            pattern = re.compile(r"\b" + word + r"\b", re.IGNORECASE)

            def replacer(match, current_word=word):
                original = match.group()
                replacement = REPLACEMENTS.get(current_word.lower(), "")

                # Preserve case
                if original.isupper():
                    return replacement.upper()
                elif original[0].isupper():
                    return replacement.capitalize()
                else:
                    return replacement

            new_name = pattern.sub(replacer, new_name)

        # Clean up multiple underscores and spaces
        new_name = re.sub(r"_+", "_", new_name)
        new_name = re.sub(r"\s+", " ", new_name)
        new_name = new_name.strip("_").strip()

        # If name becomes empty or too short, use a descriptive name
        if not new_name or len(new_name) < 3:
            new_name = f"test_{old_path.parent.name}"

        return old_path.parent / f"{new_name}{old_path.suffix}"

    def rename_file(self, file_path: Path) -> bool:
        """Rename a single file if it contains anti-patterns."""
        if not file_path.exists():
            self.skipped_files.append((file_path, "File not found"))
            return False

        # Check if file name contains forbidden patterns
        file_name = file_path.stem.lower()
        contains_pattern = any(word in file_name for word in FORBIDDEN_WORDS)

        if not contains_pattern:
            return False

        new_path = self.suggest_new_name(file_path)

        # Check if new name already exists
        if new_path.exists() and new_path != file_path:
            # Try adding a number suffix
            counter = 1
            while True:
                numbered_path = (
                    new_path.parent / f"{new_path.stem}_{counter}{new_path.suffix}"
                )
                if not numbered_path.exists():
                    new_path = numbered_path
                    break
                counter += 1
                if counter > 10:
                    self.skipped_files.append((file_path, "Could not find unique name"))
                    return False

        if new_path == file_path:
            return False

        if self.dry_run:
            print(f"Would rename: {file_path} -> {new_path}")
        else:
            try:
                file_path.rename(new_path)
                print(f"Renamed: {file_path} -> {new_path}")

                # Update imports in other files
                self._update_imports(file_path, new_path)

            except Exception as e:
                self.skipped_files.append((file_path, str(e)))
                return False

        self.renamed_files.append((file_path, new_path))
        return True

    def _update_imports(self, old_path: Path, new_path: Path):
        """Update imports in other Python files after renaming."""
        if self.dry_run:
            return

        # Convert paths to module names
        old_module = old_path.stem
        new_module = new_path.stem

        # Find all Python files in the project
        project_root = old_path.parent
        while (
            project_root.parent.name != "repos" and project_root.parent != project_root
        ):
            project_root = project_root.parent

        for py_file in project_root.rglob("*.py"):
            if py_file == new_path:
                continue

            try:
                content = py_file.read_text()
                updated_content = content

                # Update various import patterns
                patterns = [
                    (f"from .{old_module} import", f"from .{new_module} import"),
                    (f"from {old_module} import", f"from {new_module} import"),
                    (f"import {old_module}", f"import {new_module}"),
                    (f'"{old_module}"', f'"{new_module}"'),
                    (f"'{old_module}'", f"'{new_module}'"),
                ]

                for old_pattern, new_pattern in patterns:
                    updated_content = updated_content.replace(old_pattern, new_pattern)

                if updated_content != content:
                    py_file.write_text(updated_content)
                    print(f"  Updated imports in: {py_file}")

            except Exception as e:
                print(f"  Warning: Could not update imports in {py_file}: {e}")

    def rename_directory(self, directory: Path) -> None:
        """Rename all files with anti-patterns in a directory."""
        test_files = list(directory.rglob("test_*.py"))

        print(f"\nProcessing {len(test_files)} test files...\n")

        for file_path in test_files:
            self.rename_file(file_path)

        self.print_summary()

    def print_summary(self) -> None:
        """Print summary of renaming operations."""
        print("\n" + "=" * 60)
        print("RENAMING SUMMARY")
        print("=" * 60 + "\n")

        if self.dry_run:
            print("DRY RUN MODE - No files were actually renamed\n")

        print(f"Files to rename: {len(self.renamed_files)}")
        print(f"Files skipped: {len(self.skipped_files)}")

        if self.renamed_files:
            print("\nFiles renamed:")
            for old, new in self.renamed_files[:10]:
                print(f"  {old.name} -> {new.name}")

            if len(self.renamed_files) > 10:
                print(f"  ... and {len(self.renamed_files) - 10} more")

        if self.skipped_files:
            print("\nFiles skipped:")
            for path, reason in self.skipped_files[:5]:
                print(f"  {path.name}: {reason}")

            if len(self.skipped_files) > 5:
                print(f"  ... and {len(self.skipped_files) - 5} more")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rename test files containing anti-pattern names"
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Path to test directory (default: current directory)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually rename files (default is dry run)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Specific pattern to target (e.g., 'enhanced')",
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path '{args.path}' does not exist", file=sys.stderr)
        sys.exit(1)

    renamer = FileRenamer(dry_run=not args.execute)

    if args.pattern:
        # Filter to only target specific pattern
        global FORBIDDEN_WORDS
        FORBIDDEN_WORDS = [
            word for word in FORBIDDEN_WORDS if args.pattern.lower() in word.lower()
        ]

    renamer.rename_directory(args.path)


if __name__ == "__main__":
    main()
