#!/usr/bin/env python3
"""
Batch fix TRY violations across the codebase.
"""

import json  # noqa: PLC0415
import re
import subprocess
from pathlib import Path
from typing import Dict, List

def get_violations() -> List[Dict]:
    """Get all TRY violations using ruff."""
    result = subprocess.run([
        "uv", "run", "ruff", "check", ".", 
        "--select=TRY300,TRY401,TRY002,TRY301",
        "--output-format=json"
    ], capture_output=True, text=True, cwd=Path.cwd())
    
    if result.returncode == 0:
        return []
        
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

def fix_try401_violations():
    """Fix TRY401: Remove redundant exception objects from logging."""
    violations = get_violations()
    
    # Group violations by file
    by_file = {}
    for v in violations:
        if v['code'] == 'TRY401':
            filepath = v['filename']
            if filepath not in by_file:
                by_file[filepath] = []
            by_file[filepath].append(v)
    
    fixed_count = 0
    
    for filepath, file_violations in by_file.items():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix common TRY401 patterns
            patterns = [
                # logger.exception(f"message: {e}")
                (r'logger\.exception\(f"([^"]*?): \{e\}"\)', r'logger.exception("\1")'),
                (r'logger\.exception\(f"([^"]*?) \{e\}"\)', r'logger.exception("\1")'),
                (r'logger\.exception\(f"([^"]*?)\{e\}"\)', r'logger.exception("\1")'),
                
                # logger.error(..., exc_info=True, exception=e)
                (r'(logger\.(?:error|warning|info|debug|critical)\([^)]+),\s*exception=\w+\)', r'\1)'),
                (r'(logger\.(?:error|warning|info|debug|critical)\([^)]+),\s*exc=\w+\)', r'\1)'),
                
                # Generic string formatting with exception
                (r'f"([^"]*?): \{[^}]+\}"', r'"\1"'),
            ]
            
            for pattern, replacement in patterns:
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    content = new_content
                    fixed_count += 1
            
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Fixed TRY401 violations in {filepath}")
                
        except Exception:
            print(f"Error processing {filepath}: {e}")
    
    return fixed_count

def fix_try002_violations():
    """Fix TRY002: Create custom exception classes."""
    violations = get_violations()
    
    by_file = {}
    for v in violations:
        if v['code'] == 'TRY002':
            filepath = v['filename']
            if filepath not in by_file:
                by_file[filepath] = []
            by_file[filepath].append(v)
    
    fixed_count = 0
    
    for filepath, file_violations in by_file.items():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Add custom exceptions at the top
            custom_exceptions = set()
            
            for i, line in enumerate(lines):
                if 'raise Exception(' in line:
                    exception_name = _get_custom_exception_name(filepath, 'Exception')
                    if exception_name not in custom_exceptions:
                        _add_custom_exception(lines, exception_name, 'Exception')
                        custom_exceptions.add(exception_name)
                    
                    lines[i] = line.replace('raise Exception(', f'raise {exception_name}(')
                    fixed_count += 1
                
                elif 'raise ValueError(' in line and 'test' not in filepath.lower():
                    exception_name = _get_custom_exception_name(filepath, 'ValueError')
                    if exception_name not in custom_exceptions:
                        _add_custom_exception(lines, exception_name, 'ValueError')
                        custom_exceptions.add(exception_name)
                    
                    lines[i] = line.replace('raise ValueError(', f'raise {exception_name}(')
                    fixed_count += 1
            
            if custom_exceptions:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                print(f"Fixed TRY002 violations in {filepath}")
                
        except Exception:
            print(f"Error processing {filepath}: {e}")
    
    return fixed_count

def _get_custom_exception_name(filepath: str, base_exception: str) -> str:
    """Generate a custom exception name based on file context."""
    path_parts = Path(filepath).stem.split('_')
    if 'test' in path_parts:
        return f"Test{base_exception.replace('Exception', 'Error')}"
    elif 'config' in path_parts:
        return f"Config{base_exception.replace('Exception', 'Error')}"
    elif 'service' in path_parts:
        return f"Service{base_exception.replace('Exception', 'Error')}"
    elif 'cache' in path_parts:
        return f"Cache{base_exception.replace('Exception', 'Error')}"
    elif 'embedding' in path_parts:
        return f"Embedding{base_exception.replace('Exception', 'Error')}"
    else:
        return f"Custom{base_exception.replace('Exception', 'Error')}"

def _add_custom_exception(lines: List[str], exception_name: str, base_name: str):
    """Add custom exception class to the file."""
    # Find a good place to add the exception (after imports)
    insert_row = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_row = i + 1
        elif line.strip() == '':
            continue
        else:
            break
    
    exception_def = f"\nclass {exception_name}({base_name}):\n    \"\"\"Custom exception for this module.\"\"\"\n    pass\n\n"
    lines.insert(insert_row, exception_def)

def main():
    """Main entry point."""
    print("🔧 Starting batch TRY violation fixes...")
    
    # Get initial count
    initial_violations = get_violations()
    initial_count = len(initial_violations)
    print(f"Initial violations: {initial_count}")
    
    # Fix TRY401 violations first (easiest)
    print("\n📝 Fixing TRY401 violations (redundant exception objects)...")
    fixed_401 = fix_try401_violations()
    print(f"Fixed {fixed_401} TRY401 violations")
    
    # Fix TRY002 violations (custom exceptions)
    print("\n🔧 Fixing TRY002 violations (custom exceptions)...")
    fixed_002 = fix_try002_violations()
    print(f"Fixed {fixed_002} TRY002 violations")
    
    # Check final count
    final_violations = get_violations()
    final_count = len(final_violations)
    
    print(f"\n📊 Results:")
    print(f"  Initial violations: {initial_count}")
    print(f"  Final violations: {final_count}")
    print(f"  Reduction: {initial_count - final_count}")
    print(f"  Fixed: {fixed_401 + fixed_002}")
    
    if final_count < 200:
        print("✅ Target achieved! Less than 200 violations remaining.")
    else:
        print(f"⚠️  Target not yet achieved. {final_count - 200} more violations to fix.")

if __name__ == "__main__":
    main()