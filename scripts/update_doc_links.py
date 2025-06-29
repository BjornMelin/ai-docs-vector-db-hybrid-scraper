#!/usr/bin/env python3
"""
Update Documentation Links After Restructuring

Updates all internal documentation links based on the changes log from restructuring.
"""

import json  # noqa: PLC0415
import re
from pathlib import Path


class LinkUpdater:
    """Updates documentation links after restructuring."""

    def __init__(self, docs_root: Path):
        self.docs_root = docs_root
        self.changes_log = self._load_changes_log()
        self.link_map = self._build_link_map()
        self.updated_files = []

    def _load_changes_log(self) -> list[dict]:
        """Load the restructuring changes log."""
        log_file = self.docs_root / "restructure_changes.json"
        with open(log_file) as f:
            return json.load(f)

    def _build_link_map(self) -> dict[str, str]:
        """Build a comprehensive map of old paths to new paths."""
        link_map = {}

        # First, track all renames
        renames = {}
        for change in self.changes_log:
            if change["type"] == "rename":
                old_name = Path(change["from"]).name
                new_name = Path(change["to"]).name
                renames[old_name] = new_name

        # Now build the full mapping including moves
        for change in self.changes_log:
            if change["type"] == "rename":
                # Map both the full path and just the filename
                link_map[change["from"]] = change["to"]
                # Also map with ./ prefix
                link_map[f"./{change['from']}"] = f"./{change['to']}"
                # And without extension for anchors
                old_no_ext = change["from"].replace(".md", "")
                new_no_ext = change["to"].replace(".md", "")
                link_map[old_no_ext] = new_no_ext

            elif change["type"] == "move":
                # For moves, we need to track the final destination
                old_path = change["from"]
                new_path = change["to"]

                # Check if the filename was also renamed
                old_name = Path(old_path).name
                if old_name in renames:
                    # Update old_path to use the original name
                    old_dir = str(Path(old_path).parent)
                    original_name = next(k for k, v in renames.items() if v == old_name)
                    if old_dir == ".":
                        old_with_original = original_name
                    else:
                        old_with_original = f"{old_dir}/{original_name}"
                    link_map[old_with_original] = new_path

                link_map[old_path] = new_path
                # Also map with ./ prefix
                link_map[f"./{old_path}"] = new_path
                # And just the filename
                link_map[Path(old_path).name] = new_path

        # Add special mappings for common patterns
        special_mappings = {
            "QUICK_START.md": "getting-started/quick-start.md",
            "quick-start.md": "getting-started/quick-start.md",
            "./QUICK_START.md": "../getting-started/quick-start.md",
            "./quick-start.md": "../getting-started/quick-start.md",
        }
        link_map.update(special_mappings)

        return link_map

    def _calculate_relative_path(self, from_file: Path, to_file: Path) -> str:
        """Calculate the relative path from one file to another."""
        # Get the directories
        from_dir = from_file.parent
        to_path = to_file

        # Calculate relative path
        try:
            rel_path = Path(to_path).relative_to(from_dir)
            return str(rel_path)
        except ValueError:
            # Files are in different branches, need to go up
            # Count how many levels up we need to go
            up_levels = len(from_dir.parts)
            prefix = "../" * up_levels
            return prefix + str(to_path)

    def _update_link(self, link: str, source_file: Path) -> str:
        """Update a single link based on the mapping."""
        # Remove any anchor
        anchor = ""
        if "#" in link:
            link, anchor = link.split("#", 1)
            anchor = f"#{anchor}"

        # Check if this link needs updating
        if link in self.link_map:
            new_link = self.link_map[link]
            # Calculate relative path from source to target
            new_path = Path(new_link)
            if not new_path.is_absolute():
                relative_link = self._calculate_relative_path(source_file, new_path)
                return relative_link + anchor
            return new_link + anchor

        # Check for partial matches (e.g., just filename)
        link_path = Path(link)
        if link_path.name in self.link_map:
            new_path = Path(self.link_map[link_path.name])
            relative_link = self._calculate_relative_path(source_file, new_path)
            return relative_link + anchor

        # No update needed
        return link + anchor

    def update_file_links(self, file_path: Path) -> int:
        """Update all links in a single file."""
        updates_made = 0

        try:
            content = file_path.read_text(encoding="utf-8")
            original_content = content

            # Find all markdown links
            link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

            def replace_link(match):
                nonlocal updates_made
                link_text = match.group(1)
                link_url = match.group(2)

                # Skip external links
                if link_url.startswith(("http://", "https://", "mailto:", "#")):
                    return match.group(0)

                # Update the link
                new_url = self._update_link(link_url, file_path)
                if new_url != link_url:
                    updates_made += 1
                    print(f"  {link_url} → {new_url}")
                    return f"[{link_text}]({new_url})"

                return match.group(0)

            # Replace all links
            content = re.sub(link_pattern, replace_link, content)

            # Write back if changes were made
            if content != original_content:
                file_path.write_text(content, encoding="utf-8")
                self.updated_files.append(file_path)
                print(
                    f"Updated {updates_made} links in {file_path.relative_to(self.docs_root)}"
                )

        except Exception:
            print(f"Error updating {file_path}: {e}")

        return updates_made

    def update_all_links(self):
        """Update links in all markdown files."""
        total_updates = 0

        print("Updating documentation links...")
        print("-" * 60)

        # Get all markdown files
        md_files = list(self.docs_root.rglob("*.md"))

        for md_file in sorted(md_files):
            # Skip archive files
            if "archive" in str(md_file):
                continue

            updates = self.update_file_links(md_file)
            total_updates += updates

        print("-" * 60)
        print(
            f"Total updates: {total_updates} links in {len(self.updated_files)} files"
        )

        # Save update log
        self._save_update_log()

    def _save_update_log(self):
        """Save a log of all updated files."""
        log_data = {
            "updated_files": [
                str(f.relative_to(self.docs_root)) for f in self.updated_files
            ],
            "link_mappings": self.link_map,
        }

        log_file = self.docs_root / "link_updates.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"\nUpdate log saved to: {log_file}")


def main():
    """Execute link updates."""
    docs_root = Path("docs")
    updater = LinkUpdater(docs_root)
    updater.update_all_links()

    # Clean up temporary files
    (docs_root / "restructure_changes.json").unlink()
    print("\n✅ Link updates complete!")


if __name__ == "__main__":
    main()
