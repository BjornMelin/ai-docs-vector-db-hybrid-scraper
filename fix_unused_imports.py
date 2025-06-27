#!/usr/bin/env python3
"""Script to fix F401 unused import violations systematically."""

import json
import subprocess
from pathlib import Path
from typing import Dict, List


def get_violations() -> List[Dict]:
    """Get all F401 violations from ruff."""
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "ruff",
                "check",
                ".",
                "--select=F401",
                "--output-format=json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        violations = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                try:
                    violation = json.loads(line)
                    if violation.get("code") == "F401":
                        violations.append(violation)
                except json.JSONDecodeError:
                    continue

        return violations
    except Exception as e:
        print(f"Error getting violations: {e}")
        return []


def parse_import_from_message(message: str) -> str:
    """Parse the import name from the ruff message."""
    if "imported but unused" in message:
        # Extract import name between backticks
        parts = message.split("`")
        if len(parts) >= 2:
            return parts[1]
    return ""


def fix_violation(violation: Dict) -> bool:
    """Fix a single violation by removing the unused import."""
    filename = violation["filename"]
    line_number = violation["location"]["row"]
    message = violation["message"]

    # Parse the import to remove
    import_to_remove = parse_import_from_message(message)
    if not import_to_remove:
        print(f"Could not parse import from message: {message}")
        return False

    try:
        file_path = Path(filename)
        if not file_path.exists():
            print(f"File not found: {filename}")
            return False

        # Read the file
        with open(file_path, "r") as f:
            lines = f.readlines()

        if line_number > len(lines):
            print(f"Line number {line_number} out of range for {filename}")
            return False

        # Get the line with the import
        import_line = lines[line_number - 1]  # ruff uses 1-based line numbers

        # Handle different import patterns
        if import_to_remove.startswith("typing."):
            # Handle typing imports like typing.Optional
            base_import = import_to_remove.split(".", 1)[1]
            new_line = handle_typing_import(import_line, base_import)
        elif "." in import_to_remove and not import_to_remove.startswith("typing"):
            # Handle qualified imports like src.config.core.Config
            parts = import_to_remove.split(".")
            import_name = parts[-1]
            new_line = handle_from_import(import_line, import_name)
        else:
            # Handle simple imports
            new_line = handle_simple_import(import_line, import_to_remove)

        # If the line should be removed entirely
        if new_line is None:
            lines.pop(line_number - 1)
        else:
            lines[line_number - 1] = new_line

        # Write back to file
        with open(file_path, "w") as f:
            f.writelines(lines)

        print(f"Fixed: {filename}:{line_number} - removed {import_to_remove}")
        return True

    except Exception as e:
        print(f"Error fixing {filename}:{line_number}: {e}")
        return False


def handle_typing_import(line: str, import_name: str) -> str:
    """Handle typing imports like Optional, Union, etc."""
    # Check if it's a multi-import line
    if "from typing import" in line and "," in line:
        # Remove the specific import from the list
        parts = line.split("from typing import", 1)
        if len(parts) == 2:
            imports_part = parts[1].strip()
            imports = [imp.strip() for imp in imports_part.split(",")]
            # Remove the target import
            imports = [imp for imp in imports if imp != import_name]

            if imports:
                # Reconstruct the line
                return f"{parts[0]}from typing import {', '.join(imports)}\n"
            else:
                # Remove the entire line if no imports remain
                return None
    elif f"import {import_name}" in line:
        # Single import line - remove entirely
        return None

    return line


def handle_from_import(line: str, import_name: str) -> str:
    """Handle 'from module import name' patterns."""
    if f"import {import_name}" in line and "," in line:
        # Multi-import line
        if "from " in line and " import " in line:
            parts = line.split(" import ", 1)
            if len(parts) == 2:
                imports_part = parts[1].strip()
                imports = [imp.strip() for imp in imports_part.split(",")]
                imports = [imp for imp in imports if imp != import_name]

                if imports:
                    return f"{parts[0]} import {', '.join(imports)}\n"
                else:
                    return None
    elif f"import {import_name}" in line:
        # Single import line
        return None

    return line


def handle_simple_import(line: str, import_name: str) -> str:
    """Handle simple imports like 'import module'."""
    if f"import {import_name}" in line:
        if "," in line:
            # Multi-import line like "import os, sys, typing"
            parts = line.split("import", 1)
            if len(parts) == 2:
                imports_part = parts[1].strip()
                imports = [imp.strip() for imp in imports_part.split(",")]
                imports = [imp for imp in imports if imp != import_name]

                if imports:
                    return f"{parts[0]}import {', '.join(imports)}\n"
                else:
                    return None
        else:
            # Single import line
            return None

    return line


def main():
    """Main function to fix all violations."""
    print("Getting F401 violations...")
    violations = get_violations()

    if not violations:
        print("No F401 violations found!")
        return

    print(f"Found {len(violations)} violations to fix")

    # Group violations by file for better output
    files_violations = {}
    for v in violations:
        filename = v["filename"]
        if filename not in files_violations:
            files_violations[filename] = []
        files_violations[filename].append(v)

    fixed_count = 0
    failed_count = 0

    # Process each file
    for filename, file_violations in files_violations.items():
        print(f"\nProcessing {filename} ({len(file_violations)} violations)")

        # Sort violations by line number in reverse order to avoid line number shifts
        file_violations.sort(key=lambda x: x["location"]["row"], reverse=True)

        for violation in file_violations:
            if fix_violation(violation):
                fixed_count += 1
            else:
                failed_count += 1

    print(f"\nSummary:")
    print(f"Fixed: {fixed_count}")
    print(f"Failed: {failed_count}")
    print(f"Total: {len(violations)}")

    # Check if there are still violations
    print("\nChecking for remaining violations...")
    remaining_violations = get_violations()
    print(f"Remaining violations: {len(remaining_violations)}")


if __name__ == "__main__":
    main()
