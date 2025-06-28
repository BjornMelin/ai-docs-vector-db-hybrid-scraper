#!/usr/bin/env python3
"""
Script to add status indicators to documentation files that don't have them.
Follows the standardized document template format.
"""

import re
from datetime import date
from pathlib import Path


class StatusIndicatorAdder:
    """Add status indicators to markdown files that don't have them."""

    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.today = date.today().strftime("%Y-%m-%d")

        # Files to skip (already have proper headers or are special)
        self.skip_files = {
            "README.md",
            "document-template.md",
            "file-naming-guidelines.md",
            "documentation-restructure-plan.md",
            "link_updates.json",
        }

        # Determine document type and audience based on directory
        self.directory_mappings = {
            "getting-started": (
                "Current",
                "New users and developers",
                "getting-started guide",
            ),
            "tutorials": ("Current", "Users who learn by doing", "tutorial"),
            "how-to-guides": (
                "Current",
                "Developers with specific goals",
                "how-to guide",
            ),
            "reference": (
                "Current",
                "Developers needing technical details",
                "reference documentation",
            ),
            "concepts": (
                "Current",
                "Developers wanting to understand design",
                "concept explanation",
            ),
            "operations": (
                "Current",
                "System administrators and DevOps teams",
                "operations guide",
            ),
            "contributing": (
                "Current",
                "Contributors and team members",
                "contributing guide",
            ),
            "archive": ("Deprecated", "Historical reference", "archived documentation"),
            "examples": (
                "Current",
                "Developers needing examples",
                "example documentation",
            ),
            "features": (
                "Current",
                "Developers and product managers",
                "feature documentation",
            ),
        }

    def has_status_header(self, content: str) -> bool:
        """Check if file already has proper status header."""
        # Look for the pattern: > **Status**:
        pattern = r">\s*\*\*Status\*\*:\s*\w+"
        return bool(re.search(pattern, content))

    def get_document_metadata(self, file_path: Path) -> tuple[str, str, str]:
        """Determine status, audience, and purpose based on file path."""
        # Get directory name for context
        relative_path = file_path.relative_to(self.docs_dir)
        parts = relative_path.parts

        # Determine primary directory
        primary_dir = parts[0] if len(parts) > 1 else "docs"

        # Get metadata from directory mapping
        if primary_dir in self.directory_mappings:
            status, audience, doc_type = self.directory_mappings[primary_dir]
        else:
            status, audience, doc_type = "Current", "Developers", "documentation"

        # Determine purpose from filename
        filename = file_path.stem.replace("-", " ").title()
        purpose = f"{filename} {doc_type}"

        return status, audience, purpose

    def create_header(self, title: str, file_path: Path) -> str:
        """Create standardized header for document."""
        status, audience, purpose = self.get_document_metadata(file_path)

        return f"""# {title}

> **Status**: {status}
> **Last Updated**: {self.today}
> **Purpose**: {purpose}
> **Audience**: {audience}

"""

    def extract_title(self, content: str) -> str:
        """Extract title from first H1 header, or use filename."""
        lines = content.split("\n")
        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()
        return "Document"

    def process_file(self, file_path: Path, dry_run: bool = True) -> bool:
        """Process a single markdown file to add status indicators."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Skip if already has status header
            if self.has_status_header(content):
                print(f"SKIP: {file_path} (already has status header)")
                return False

            # Extract existing title
            title = self.extract_title(content)

            # Remove existing title if it exists
            lines = content.split("\n")
            if lines and lines[0].startswith("# "):
                content = "\n".join(lines[1:]).lstrip("\n")

            # Create new content with header
            new_header = self.create_header(title, file_path)
            new_content = new_header + content

            if dry_run:
                print(f"WOULD UPDATE: {file_path}")
                print(f"  New header: {new_header.strip()}")
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"UPDATED: {file_path}")

            return True

        except Exception:
            print(f"ERROR processing {file_path}: {e}")
            return False

    def find_markdown_files(self) -> list[Path]:
        """Find all markdown files that need processing."""
        files = []

        for md_file in self.docs_dir.rglob("*.md"):
            # Skip if in skip list
            if md_file.name in self.skip_files:
                continue

            # Skip if not a regular documentation file
            if md_file.is_file():
                files.append(md_file)

        return sorted(files)

    def process_all_files(self, dry_run: bool = True) -> dict[str, int]:
        """Process all markdown files to add status indicators."""
        files = self.find_markdown_files()

        stats = {"total": len(files), "updated": 0, "skipped": 0, "errors": 0}

        print(f"Found {len(files)} markdown files to process")
        print(f"{'DRY RUN - ' if dry_run else ''}Processing files...\n")

        for file_path in files:
            try:
                if self.process_file(file_path, dry_run):
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1
            except Exception:
                print(f"ERROR: {file_path}: {e}")
                stats["errors"] += 1

        return stats

    def print_summary(self, stats: dict[str, int], dry_run: bool):
        """Print summary of processing results."""
        print(f"\n{'DRY RUN ' if dry_run else ''}SUMMARY:")
        print(f"  Total files: {stats['total']}")
        print(f"  {'Would update' if dry_run else 'Updated'}: {stats['updated']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Errors: {stats['errors']}")


def main():
    """Main function to run the status indicator addition."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Add status indicators to documentation files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Actually modify the files"
    )
    parser.add_argument(
        "--docs-dir", default="docs", help="Documentation directory (default: docs)"
    )

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("Please specify either --dry-run or --execute")
        return

    adder = StatusIndicatorAdder(args.docs_dir)
    stats = adder.process_all_files(dry_run=args.dry_run)
    adder.print_summary(stats, args.dry_run)


if __name__ == "__main__":
    main()
