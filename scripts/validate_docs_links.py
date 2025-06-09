#!/usr/bin/env python3
"""
Documentation Link Validation Script

Validates all internal links in documentation to ensure they point to valid files.
Reports broken links, suggests fixes, and can be integrated into CI/CD.

Usage:
    python scripts/validate_docs_links.py
    python scripts/validate_docs_links.py --fix  # Auto-fix obvious issues
    python scripts/validate_docs_links.py --format json  # JSON output for CI
"""

import argparse
import json
import re
import sys
from pathlib import Path


class LinkValidator:
    """Validates internal documentation links."""

    def __init__(self, docs_root: Path):
        self.docs_root = docs_root
        self.markdown_files = list(docs_root.rglob("*.md"))
        self.broken_links: list[dict] = []
        self.valid_files: set[Path] = set()
        self._build_file_index()

    def _build_file_index(self):
        """Build index of all valid documentation files."""
        for md_file in self.markdown_files:
            # Store relative path from docs root
            rel_path = md_file.relative_to(self.docs_root)
            self.valid_files.add(rel_path)

    def validate_all_links(self) -> dict:
        """Validate all links in all markdown files."""
        total_links = 0
        broken_count = 0

        for md_file in self.markdown_files:
            if "archive" in str(md_file):
                continue  # Skip archived docs for now

            links = self._extract_links(md_file)
            for link_text, link_path, line_num in links:
                total_links += 1
                if not self._is_valid_link(md_file, link_path):
                    broken_count += 1
                    self.broken_links.append(
                        {
                            "file": str(md_file.relative_to(self.docs_root)),
                            "line": line_num,
                            "link_text": link_text,
                            "link_path": link_path,
                            "suggested_fix": self._suggest_fix(md_file, link_path),
                        }
                    )

        return {
            "total_links": total_links,
            "broken_links": broken_count,
            "broken_details": self.broken_links,
            "success_rate": (
                (total_links - broken_count) / total_links if total_links > 0 else 1.0
            ),
        }

    def _extract_links(self, file_path: Path) -> list[tuple[str, str, int]]:
        """Extract all markdown links from a file."""
        links = []
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Regex for markdown links: [text](link)
            link_pattern = r"\[([^\]]*)\]\(([^)]+)\)"

            for line_num, line in enumerate(lines, 1):
                matches = re.findall(link_pattern, line)
                for link_text, link_path in matches:
                    # Only validate internal links (not URLs)
                    if not self._is_external_link(link_path):
                        links.append((link_text, link_path, line_num))

        except UnicodeDecodeError:
            print(f"Warning: Could not read {file_path} (encoding issue)")
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}")

        return links

    def _is_external_link(self, link: str) -> bool:
        """Check if link is external (http/https/mailto/etc)."""
        external_prefixes = ("http://", "https://", "mailto:", "ftp://", "tel:")
        return link.startswith(external_prefixes) or link.startswith("#")

    def _is_valid_link(self, source_file: Path, link_path: str) -> bool:
        """Check if an internal link points to a valid file."""
        if self._is_external_link(link_path):
            return True

        # Remove anchor fragments (#section)
        clean_link = link_path.split("#")[0]
        if not clean_link:  # Pure anchor link
            return True

        # Handle relative paths
        if clean_link.startswith("./") or clean_link.startswith("../"):
            # Resolve relative path from source file's directory
            source_dir = source_file.parent
            target_path = (source_dir / clean_link).resolve()
            try:
                target_rel = target_path.relative_to(self.docs_root.resolve())
                return target_rel in self.valid_files
            except ValueError:
                return False

        # Handle absolute paths from docs root
        if clean_link.startswith("/"):
            clean_link = clean_link[1:]

        target_path = Path(clean_link)
        return target_path in self.valid_files

    def _suggest_fix(self, source_file: Path, broken_link: str) -> str:
        """Suggest a fix for a broken link."""
        clean_link = broken_link.split("#")[0]
        if not clean_link:
            return "Anchor-only link - check if target section exists"

        # Try to find similar files
        target_name = Path(clean_link).name
        similar_files = []

        for valid_file in self.valid_files:
            if valid_file.name == target_name:
                # Calculate relative path from source to target
                source_dir = source_file.parent.relative_to(self.docs_root)
                try:
                    rel_path = (
                        Path("..") / valid_file
                        if source_dir != Path(".")
                        else valid_file
                    )
                    similar_files.append(str(rel_path))
                except Exception:
                    similar_files.append(str(valid_file))

        if similar_files:
            return f"Try: {similar_files[0]}" + (
                f" (or {len(similar_files) - 1} others)"
                if len(similar_files) > 1
                else ""
            )

        return "File not found - check if it exists or was moved"

    def fix_obvious_issues(self) -> int:
        """Auto-fix obvious link issues where possible."""
        fixes_applied = 0

        for broken_link in self.broken_links:
            # Only auto-fix if we have a clear single suggestion
            if (
                broken_link["suggested_fix"].startswith("Try: ")
                and "others" not in broken_link["suggested_fix"]
            ):
                fix = broken_link["suggested_fix"][5:]  # Remove "Try: "
                file_path = self.docs_root / broken_link["file"]

                try:
                    content = file_path.read_text(encoding="utf-8")
                    # Simple replacement - be careful with this
                    old_link = f"]({broken_link['link_path']})"
                    new_link = f"]({fix})"

                    if old_link in content and content.count(old_link) == 1:
                        content = content.replace(old_link, new_link)
                        file_path.write_text(content, encoding="utf-8")
                        fixes_applied += 1
                        print(
                            f"Fixed: {broken_link['file']}:{broken_link['line']} - {broken_link['link_path']} -> {fix}"
                        )

                except Exception as e:
                    print(f"Could not fix {broken_link['file']}: {e}")

        return fixes_applied


def main():
    parser = argparse.ArgumentParser(description="Validate documentation links")
    parser.add_argument("--fix", action="store_true", help="Auto-fix obvious issues")
    parser.add_argument(
        "--check-only", action="store_true", help="Only validate, don't fix (for CI)"
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=Path("docs"),
        help="Documentation root directory",
    )

    args = parser.parse_args()

    if not args.docs_root.exists():
        print(f"Error: Documentation directory {args.docs_root} not found")
        sys.exit(1)

    validator = LinkValidator(args.docs_root)
    results = validator.validate_all_links()

    if args.fix and not args.check_only:
        fixes = validator.fix_obvious_issues()
        results["fixes_applied"] = fixes
        # Re-validate after fixes
        validator.broken_links = []
        results = validator.validate_all_links()
        results["fixes_applied"] = fixes

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        # Text output
        print("Documentation Link Validation Results")
        print("=====================================")
        print(f"Total links checked: {results['total_links']}")
        print(f"Broken links found: {results['broken_links']}")
        print(f"Success rate: {results['success_rate']:.1%}")

        if results["broken_links"] > 0:
            print("\nBroken Links Details:")
            print("---------------------")
            for broken in results["broken_details"]:
                print(f"📁 {broken['file']}:{broken['line']}")
                print(f"   Link: [{broken['link_text']}]({broken['link_path']})")
                print(f"   Fix:  {broken['suggested_fix']}")
                print()

        if args.fix and "fixes_applied" in results:
            print(f"Auto-fixes applied: {results['fixes_applied']}")

    # Exit with error code if there are broken links
    sys.exit(1 if results["broken_links"] > 0 else 0)


if __name__ == "__main__":
    main()
