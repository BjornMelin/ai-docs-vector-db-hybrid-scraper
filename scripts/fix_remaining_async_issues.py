#!/usr/bin/env python3
"""Fix remaining async test issues based on the modernization report."""

import re
from pathlib import Path
from typing import List, Tuple

# List of files and line numbers that need @pytest.mark.asyncio
ASYNCIO_FIXES = [
    ("tests/security/test_api_security.py", [108, 187, 236]),
    ("tests/examples/test_pattern_examples.py", [88, 149, 203, 253, 312]),
    ("tests/load/stress_testing/test_stress_scenarios.py", [131, 237, 338, 458]),
    ("tests/load/stress_testing/test_breaking_point.py", [25, 139, 273, 452]),
    ("tests/load/load_testing/test_concurrent_users.py", [20, 73, 100, 177, 258, 326]),
]


def fix_missing_asyncio_decorator(filepath: Path, line_numbers: List[int]) -> int:
    """Add @pytest.mark.asyncio decorators at specified line numbers."""
    content = filepath.read_text()
    lines = content.split('\n')
    
    # Sort line numbers in reverse order to avoid index shifting
    line_numbers_sorted = sorted(line_numbers, reverse=True)
    
    fixes_applied = 0
    
    for line_num in line_numbers_sorted:
        # Convert to 0-based index
        idx = line_num - 1
        
        if idx < len(lines):
            line = lines[idx]
            
            # Verify this is an async function definition
            if 'async def' in line:
                # Get the indentation
                indent = len(line) - len(line.lstrip())
                indent_str = ' ' * indent
                
                # Check if the previous line already has the decorator
                if idx > 0 and '@pytest.mark.asyncio' not in lines[idx - 1]:
                    # Insert the decorator
                    lines.insert(idx, f"{indent_str}@pytest.mark.asyncio")
                    fixes_applied += 1
    
    if fixes_applied > 0:
        filepath.write_text('\n'.join(lines))
    
    return fixes_applied


def main():
    """Apply fixes to all files."""
    total_fixes = 0
    
    print("Fixing missing @pytest.mark.asyncio decorators...")
    
    for filepath_str, line_numbers in ASYNCIO_FIXES:
        filepath = Path(filepath_str)
        
        if not filepath.exists():
            print(f"  ⚠️  {filepath} not found")
            continue
        
        fixes = fix_missing_asyncio_decorator(filepath, line_numbers)
        
        if fixes > 0:
            total_fixes += fixes
            print(f"  ✓ {filepath}: Fixed {fixes} missing decorators")
        else:
            print(f"  - {filepath}: No changes needed")
    
    print(f"\nTotal fixes applied: {total_fixes}")
    
    if total_fixes > 0:
        print("\nRunning formatters...")
        import subprocess
        subprocess.run(['ruff', 'check', 'tests', '--fix'], check=False)
        subprocess.run(['ruff', 'format', 'tests'], check=False)


if __name__ == '__main__':
    main()