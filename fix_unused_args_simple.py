#!/usr/bin/env python3
"""Simple script to fix unused function arguments."""

import re
import subprocess
from pathlib import Path


def get_violations():
    """Get all ARG violations."""
    result = subprocess.run(
        ["uv", "run", "ruff", "check", ".", "--select=ARG001,ARG002"],
        capture_output=True,
        text=True,
    )
    return result.stderr


def parse_and_fix_violations(violations_text):
    """Parse violations and fix them."""
    lines = violations_text.split('\n')
    fixes = {}  # file_path -> [(line_num, old_arg, new_arg)]
    
    for line in lines:
        if '.py:' in line and ('ARG001' in line or 'ARG002' in line):
            # Extract file, line, and argument name
            match = re.match(r'([^:]+):(\d+):\d+: ARG\d+ Unused \w+ argument: `(\w+)`', line)
            if match:
                file_path, line_num, arg_name = match.groups()
                line_num = int(line_num)
                
                # Skip if already has underscore prefix
                if arg_name.startswith('_'):
                    continue
                
                if file_path not in fixes:
                    fixes[file_path] = []
                fixes[file_path].append((line_num, arg_name, f'_{arg_name}'))
    
    # Apply fixes
    for file_path, file_fixes in fixes.items():
        path = Path(file_path)
        if not path.exists():
            continue
        
        content = path.read_text()
        lines = content.split('\n')
        
        # Sort by line number in reverse to avoid offset issues
        file_fixes.sort(key=lambda x: x[0], reverse=True)
        
        for line_num, old_arg, new_arg in file_fixes:
            if line_num <= len(lines):
                line = lines[line_num - 1]  # Convert to 0-based
                
                # Replace the argument name with underscore version
                # Handle common patterns
                patterns = [
                    rf'\b{re.escape(old_arg)}\b(?=\s*:)',  # arg: type
                    rf'\b{re.escape(old_arg)}\b(?=\s*,)',  # arg,
                    rf'\b{re.escape(old_arg)}\b(?=\s*=)',  # arg=
                    rf'\b{re.escape(old_arg)}\b(?=\s*\))', # arg)
                ]
                
                for pattern in patterns:
                    if re.search(pattern, line):
                        lines[line_num - 1] = re.sub(pattern, new_arg, line)
                        break
        
        path.write_text('\n'.join(lines))
        print(f"Fixed {len(file_fixes)} violations in {file_path}")


def main():
    """Main function."""
    print("Getting violations...")
    violations = get_violations()
    
    if not violations:
        print("No violations found!")
        return
    
    print("Parsing and fixing violations...")
    parse_and_fix_violations(violations)
    
    print("Checking remaining violations...")
    remaining = get_violations()
    remaining_count = len([l for l in remaining.split('\n') if 'ARG001' in l or 'ARG002' in l])
    print(f"Remaining violations: {remaining_count}")


if __name__ == "__main__":
    main()