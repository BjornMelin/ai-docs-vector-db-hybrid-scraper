#!/usr/bin/env python3
"""
Documentation Restructuring Script

Handles mass renaming and moving of documentation files while tracking all changes
for link updates.
"""

import json
import re
import shutil
from pathlib import Path


class DocumentationRestructurer:
    """Handles documentation restructuring operations."""

    def __init__(self, docs_root: Path):
        self.docs_root = docs_root
        self.rename_map: dict[str, str] = {}
        self.move_map: dict[str, str] = {}
        self.changes_log: list[dict] = []

    def to_kebab_case(self, filename: str) -> str:
        """Convert filename to kebab-case."""
        # Skip README files
        if filename == "README.md":
            return filename

        # Remove .md extension for processing
        name = filename[:-3] if filename.endswith(".md") else filename

        # Handle UPPERCASE_UNDERSCORE pattern
        if name.isupper() or "_" in name:
            name = name.lower().replace("_", "-")

        # Handle camelCase
        name = re.sub("([a-z0-9])([A-Z])", r"\1-\2", name).lower()

        # Handle numbers at start (like 01_GUIDE.md)
        name = re.sub(r"^(\d+)_", r"\1-", name)

        return f"{name}.md"

    def generate_rename_map(self) -> dict[str, str]:
        """Generate mapping of files to rename."""
        rename_map = {
            # Core files
            "QUICK_START.md": "quick-start.md",
            "SYSTEM_OVERVIEW.md": "system-overview.md",
            "ADVANCED_SEARCH_IMPLEMENTATION.md": "advanced-search.md",
            "HYDE_QUERY_ENHANCEMENT.md": "hyde-enhancement.md",
            "RERANKING_GUIDE.md": "add-reranking.md",
            "VECTOR_DB_BEST_PRACTICES.md": "vector-db-tuning.md",
            "ENHANCED_CHUNKING_GUIDE.md": "chunking-guide.md",
            "EMBEDDING_MODEL_INTEGRATION.md": "embedding-models.md",
            "BROWSER_AUTOMATION_ARCHITECTURE.md": "browser-architecture.md",
            "UNIFIED_CONFIGURATION.md": "config-schema.md",
            "API_REFERENCE.md": "rest-api.md",
            "browser_automation_api.md": "browser-api.md",
            "MONITORING.md": "metrics-guide.md",
            "TROUBLESHOOTING.md": "troubleshooting.md",
            "CANARY_DEPLOYMENT_GUIDE.md": "canary-deployment.md",
            "DEVELOPMENT_WORKFLOW.md": "development-setup.md",
            "TESTING_DOCUMENTATION.md": "testing-guide.md",
            "ARCHITECTURE_IMPROVEMENTS.md": "architecture-guide.md",
            "INTEGRATED_V1_ARCHITECTURE.md": "v1-architecture.md",
            "UNIFIED_SCRAPING_ARCHITECTURE.md": "scraping-architecture.md",
            "CENTRALIZED_CLIENT_MANAGEMENT.md": "client-management.md",
            "V1_IMPLEMENTATION_PLAN.md": "v1-implementation-plan.md",
            "TESTING_QUALITY_ENHANCEMENTS.md": "testing-quality.md",
            "UNIFIED_BROWSER_MANAGER_REFACTOR.md": "browser-manager-refactor.md",
            "TASK_QUEUE.md": "task-queue.md",
            "PERFORMANCE_GUIDE.md": "performance-guide.md",
            "SETUP.md": "setup.md",
            "MIGRATION_GUIDE.md": "migration-guide.md",
            "MINIMAL_COST_TO_PERSONAL_USE_MIGRATION.md": "deployment-options.md",
            "CHUNKING_RESEARCH.md": "chunking-theory.md",
            "PAYLOAD_INDEXING_PERFORMANCE.md": "payload-indexing-performance.md",
            "COLLECTION_ALIASES.md": "collection-aliases.md",
            "BROWSER_AUTOMATION_INTEGRATION_ROADMAP.md": "browser-automation-roadmap.md",
        }

        # Auto-generate for any files not explicitly mapped
        for md_file in self.docs_root.rglob("*.md"):
            if "archive" in str(md_file):
                continue

            filename = md_file.name
            if filename not in rename_map and filename != "README.md":
                new_name = self.to_kebab_case(filename)
                if new_name != filename:
                    rename_map[filename] = new_name

        return rename_map

    def generate_move_map(self) -> dict[str, str]:
        """Generate mapping of where files should be moved."""
        return {
            # Getting Started
            "quick-start.md": "getting-started/quick-start.md",
            # Tutorials
            "user-guides/browser-automation.md": "tutorials/browser-automation.md",
            "user-guides/crawl4ai.md": "tutorials/crawl4ai-setup.md",
            # How-to Guides - Search
            "features/advanced-search.md": "how-to-guides/implement-search/advanced-search.md",
            "features/hyde-enhancement.md": "how-to-guides/implement-search/hyde-enhancement.md",
            "features/add-reranking.md": "how-to-guides/implement-search/add-reranking.md",
            # How-to Guides - Documents
            "features/chunking-guide.md": "how-to-guides/process-documents/chunking-guide.md",
            "features/embedding-models.md": "how-to-guides/process-documents/embedding-models.md",
            # How-to Guides - Performance
            "features/vector-db-tuning.md": "how-to-guides/optimize-performance/vector-db-tuning.md",
            "operations/metrics-guide.md": "how-to-guides/optimize-performance/monitoring.md",
            "operations/performance-guide.md": "how-to-guides/optimize-performance/performance-guide.md",
            # How-to Guides - Deploy
            "deployment/canary-deployment.md": "how-to-guides/deploy/canary-deployment.md",
            "operations/deployment-options.md": "how-to-guides/deploy/deployment-options.md",
            # Reference - API
            "api/rest-api.md": "reference/api/rest-api.md",
            "api/browser-api.md": "reference/api/browser-api.md",
            # Reference - Configuration
            "architecture/config-schema.md": "reference/configuration/config-schema.md",
            # Reference - MCP Tools
            "mcp/README.md": "reference/mcp-tools/README.md",
            "mcp/setup.md": "reference/mcp-tools/setup.md",
            "mcp/migration-guide.md": "reference/mcp-tools/migration-guide.md",
            # Concepts - Architecture
            "architecture/system-overview.md": "concepts/architecture/system-overview.md",
            "architecture/v1-architecture.md": "concepts/architecture/v1-architecture.md",
            "architecture/browser-architecture.md": "concepts/architecture/browser-architecture.md",
            "architecture/scraping-architecture.md": "concepts/architecture/scraping-architecture.md",
            "architecture/client-management.md": "concepts/architecture/client-management.md",
            # Concepts - Features
            "features/chunking/chunking-theory.md": "concepts/features/chunking-theory.md",
            # Operations
            "operations/troubleshooting.md": "operations/monitoring/troubleshooting.md",
            "operations/task-queue.md": "operations/maintenance/task-queue.md",
            # Contributing
            "development/development-setup.md": "contributing/development-setup.md",
            "development/testing-guide.md": "contributing/testing-guide.md",
            "development/architecture-guide.md": "contributing/architecture-guide.md",
            "development/v1-implementation-plan.md": "contributing/v1-implementation-plan.md",
            "development/testing-quality.md": "contributing/testing-quality.md",
            "development/browser-manager-refactor.md": "contributing/browser-manager-refactor.md",
        }

    def rename_files(self, dry_run: bool = True) -> list[tuple[Path, Path]]:
        """Rename files to kebab-case."""
        rename_map = self.generate_rename_map()
        renames = []

        for old_name, new_name in rename_map.items():
            # Find all instances of this file
            for md_file in self.docs_root.rglob(old_name):
                if "archive" in str(md_file):
                    continue

                new_path = md_file.parent / new_name

                if not dry_run:
                    md_file.rename(new_path)
                    self.changes_log.append(
                        {
                            "type": "rename",
                            "from": str(md_file.relative_to(self.docs_root)),
                            "to": str(new_path.relative_to(self.docs_root)),
                        }
                    )

                renames.append((md_file, new_path))
                print(
                    f"{'Would rename' if dry_run else 'Renamed'}: {md_file.name} → {new_name}"
                )

        return renames

    def move_files(self, dry_run: bool = True) -> list[tuple[Path, Path]]:
        """Move files to new directory structure."""
        move_map = self.generate_move_map()
        moves = []

        for source_path, dest_path in move_map.items():
            source = self.docs_root / source_path
            dest = self.docs_root / dest_path

            if source.exists():
                if not dry_run:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(dest))
                    self.changes_log.append(
                        {"type": "move", "from": source_path, "to": dest_path}
                    )

                moves.append((source, dest))
                print(
                    f"{'Would move' if dry_run else 'Moved'}: {source_path} → {dest_path}"
                )

        return moves

    def save_changes_log(self):
        """Save all changes for link update phase."""
        log_file = self.docs_root / "restructure_changes.json"
        with open(log_file, "w") as f:
            json.dump(self.changes_log, f, indent=2)
        print(f"\nChanges log saved to: {log_file}")

    def cleanup_empty_dirs(self):
        """Remove empty directories after moves."""
        for dir_path in self.docs_root.rglob("*"):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                print(f"Removing empty directory: {dir_path}")
                dir_path.rmdir()


def main():
    """Execute the restructuring."""
    import sys

    docs_root = Path("docs")
    restructurer = DocumentationRestructurer(docs_root)

    # Check for command line arguments
    dry_run = "--dry-run" in sys.argv
    execute = "--execute" in sys.argv

    if not dry_run and not execute:
        print("Usage: python restructure_docs.py [--dry-run | --execute]")
        sys.exit(1)

    print("=== Documentation Restructuring ===\n")

    if dry_run:
        # Phase 1: Rename files (dry run)
        print("Phase 1: File Renames (Dry Run)")
        print("-" * 40)
        restructurer.rename_files(dry_run=True)

        # Phase 2: Move files (dry run)
        print("\n\nPhase 2: File Moves (Dry Run)")
        print("-" * 40)
        restructurer.move_files(dry_run=True)

        print("\n\n✅ Dry run complete. Use --execute to apply changes.")
    else:
        # Execute actual changes
        print("Phase 1: Renaming files...")
        print("-" * 40)
        restructurer.rename_files(dry_run=False)

        print("\n\nPhase 2: Moving files...")
        print("-" * 40)
        restructurer.move_files(dry_run=False)
        restructurer.cleanup_empty_dirs()

        # Save changes log
        restructurer.save_changes_log()
        print("\n✅ Restructuring complete!")


if __name__ == "__main__":
    main()
