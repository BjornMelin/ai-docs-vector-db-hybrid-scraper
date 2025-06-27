#!/usr/bin/env python3
"""Script to fix TRY401 violations by removing redundant exception objects from logging.exception calls."""

import re
import subprocess
import sys
from pathlib import Path

def get_try401_violations():
    """Get all TRY401 violations from ruff."""
    result = subprocess.run([
        "uv", "run", "ruff", "check", ".", "--select=TRY401", "--output-format=json"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running ruff:", result.stderr)
        return []
    
    import json
    violations = []
    for line in result.stdout.strip().split('\n'):
        if line.strip():
            try:
                violations.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return violations

def fix_file(filepath, violations):
    """Fix TRY401 violations in a single file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        modified = False
        
        # Sort violations by line number in reverse order to avoid offset issues
        file_violations = [v for v in violations if v['filename'] == str(filepath)]
        file_violations.sort(key=lambda x: x['location']['row'], reverse=True)
        
        for violation in file_violations:
            line_num = violation['location']['row'] - 1  # Convert to 0-based index
            if line_num < len(lines):
                line = lines[line_num]
                
                # Pattern to match logger.exception(f"message: {e}")
                # Remove the : {e} part
                patterns = [
                    (r'logger\.exception\(f"([^"]*): \{e\}"\)', r'logger.exception("\1")'),
                    (r'logger\.exception\(f"([^"]*): \{.*?\}"\)', r'logger.exception("\1")'),
                    (r'logger\.exception\(f"([^"]*)\{e\}"\)', r'logger.exception("\1")'),
                    (r'logger\.exception\(f"([^"]*)\{.*?\}"\)', r'logger.exception("\1")'),
                    (r'self\.logger\.exception\(f"([^"]*): \{e\}"\)', r'self.logger.exception("\1")'),
                    (r'self\.logger\.exception\(f"([^"]*): \{.*?\}"\)', r'self.logger.exception("\1")'),
                    (r'self\.logger\.exception\(f"([^"]*)\{e\}"\)', r'self.logger.exception("\1")'),
                    (r'self\.logger\.exception\(f"([^"]*)\{.*?\}"\)', r'self.logger.exception("\1")'),
                ]
                
                for pattern, replacement in patterns:
                    new_line = re.sub(pattern, replacement, line)
                    if new_line != line:
                        lines[line_num] = new_line
                        modified = True
                        break
        
        if modified:
            with open(filepath, 'w') as f:
                f.write('\n'.join(lines))
            print(f"Fixed {filepath}")
            return True
    
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False
    
    return False

def main():
    """Main function to fix TRY401 violations."""
    print("Getting TRY401 violations...")
    violations = get_try401_violations()
    
    if not violations:
        print("No TRY401 violations found.")
        return
    
    print(f"Found {len(violations)} TRY401 violations.")
    
    # Group violations by file
    files_to_fix = {}
    for violation in violations:
        filepath = Path(violation['filename'])
        if filepath not in files_to_fix:
            files_to_fix[filepath] = []
        files_to_fix[filepath].append(violation)
    
    print(f"Files to fix: {len(files_to_fix)}")
    
    fixed_count = 0
    for filepath, file_violations in files_to_fix.items():
        if fix_file(filepath, file_violations):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} files.")

if __name__ == "__main__":
    main()