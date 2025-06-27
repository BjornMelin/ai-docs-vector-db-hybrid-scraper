#!/usr/bin/env python3
"""Final script to fix all remaining unused argument violations."""

import re
import subprocess
from pathlib import Path


def main():
    """Main function to fix all violations in one go."""
    
    # Get ruff output
    result = subprocess.run(
        ["uv", "run", "ruff", "check", ".", "--select=ARG001,ARG002"],
        capture_output=True,
        text=True,
    )
    
    # Process each violation line
    violations_by_file = {}
    
    for line in result.stderr.split('\n'):
        if '.py:' in line and ('ARG001' in line or 'ARG002' in line):
            # Extract file, line number, and argument name
            match = re.match(r'([^:]+):(\d+):\d+: ARG\d+ Unused \w+ argument: `(\w+)`', line)
            if match:
                file_path, line_num, arg_name = match.groups()
                
                if file_path not in violations_by_file:
                    violations_by_file[file_path] = []
                violations_by_file[file_path].append((int(line_num), arg_name))
    
    print(f"Found {sum(len(v) for v in violations_by_file.values())} violations in {len(violations_by_file)} files")
    
    # Fix each file
    for file_path, file_violations in violations_by_file.items():
        path = Path(file_path)
        if not path.exists():
            continue
        
        try:
            content = path.read_text()
            lines = content.split('\n')
            
            # Sort by line number in reverse order to avoid offset issues
            file_violations.sort(key=lambda x: x[0], reverse=True)
            
            for line_num, arg_name in file_violations:
                if line_num <= len(lines):
                    line = lines[line_num - 1]  # Convert to 0-based indexing
                    
                    # Skip if already has underscore prefix
                    if f'_{arg_name}' in line:
                        continue
                    
                    # Replace the argument name
                    patterns = [
                        rf'\b{re.escape(arg_name)}\b(?=\s*:)',   # arg: type
                        rf'\b{re.escape(arg_name)}\b(?=\s*,)',   # arg,
                        rf'\b{re.escape(arg_name)}\b(?=\s*=)',   # arg=
                        rf'\b{re.escape(arg_name)}\b(?=\s*\))',  # arg)
                        rf'\b{re.escape(arg_name)}\b(?=\s|$)',   # arg at end
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, line):
                            lines[line_num - 1] = re.sub(pattern, f'_{arg_name}', line)
                            break
            
            path.write_text('\n'.join(lines))
            print(f"Fixed {len(file_violations)} violations in {file_path}")
            
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
    
    # Check final result
    print("\nChecking final result...")
    final_result = subprocess.run(
        ["uv", "run", "ruff", "check", ".", "--select=ARG001,ARG002"],
        capture_output=True,
        text=True,
    )
    
    final_violations = len([l for l in final_result.stderr.split('\n') if 'ARG001' in l or 'ARG002' in l])
    print(f"Remaining violations: {final_violations}")


if __name__ == "__main__":
    main()