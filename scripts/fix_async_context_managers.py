#!/usr/bin/env python3
"""Fix async context manager issues in test files.

This script automatically converts httpx.AsyncClient() usage
to proper async context manager patterns.
"""

import ast
import re
from pathlib import Path
from typing import List, Tuple


def extract_unique_issues_from_report(report_path: Path) -> dict[str, List[int]]:
    """Extract unique file paths and line numbers from the report."""
    files_and_lines = {}
    
    content = report_path.read_text()
    lines = content.split('\n')
    
    # Find the section with files needing async context managers
    in_context_manager_section = False
    
    for line in lines:
        if "Files needing async context managers:" in line:
            in_context_manager_section = True
            continue
        
        if in_context_manager_section and line.startswith("## "):
            # End of section
            break
            
        if in_context_manager_section and line.startswith("- tests/"):
            # Extract file path
            file_path = line[2:].strip()
            
            if file_path not in files_and_lines:
                files_and_lines[file_path] = []
                
    return files_and_lines


def find_async_client_issues(filepath: Path) -> List[Tuple[int, str]]:
    """Find lines with httpx.AsyncClient() that need context managers."""
    content = filepath.read_text()
    lines = content.split('\n')
    
    issues = []
    
    for i, line in enumerate(lines):
        # Pattern 1: Direct assignment
        if re.search(r'\b(\w+)\s*=\s*httpx\.AsyncClient\s*\(', line):
            # Check if it's already in an async with statement
            stripped = line.strip()
            if not stripped.startswith('async with'):
                issues.append((i + 1, line))
        
        # Pattern 2: Creating client without assignment (rare but possible)
        elif re.search(r'httpx\.AsyncClient\s*\(\)', line) and 'async with' not in line:
            issues.append((i + 1, line))
    
    return issues


def fix_async_client_usage(filepath: Path) -> int:
    """Fix httpx.AsyncClient usage to use async context managers."""
    content = filepath.read_text()
    lines = content.split('\n')
    
    fixes_applied = 0
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Pattern: variable = httpx.AsyncClient(...)
        match = re.match(r'^(\s*)(\w+)\s*=\s*httpx\.AsyncClient\s*\((.*?)\)(.*)$', line)
        
        if match and 'async with' not in line:
            indent = match.group(1)
            var_name = match.group(2)
            args = match.group(3)
            rest_of_line = match.group(4)
            
            # Check if this is inside a class __init__ or setup method
            # These often need special handling
            is_in_init = False
            for j in range(max(0, i - 10), i):
                if 'def __init__' in lines[j] or 'def setup' in lines[j]:
                    is_in_init = True
                    break
            
            if is_in_init:
                # Skip these for now - they need manual review
                i += 1
                continue
            
            # Replace with async context manager
            if args:
                lines[i] = f"{indent}async with httpx.AsyncClient({args}) as {var_name}:{rest_of_line}"
            else:
                lines[i] = f"{indent}async with httpx.AsyncClient() as {var_name}:{rest_of_line}"
            
            # Indent the following lines that use this client
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if not next_line.strip():
                    # Empty line, continue but don't indent
                    j += 1
                    continue
                
                # Check if this line is at the same or less indentation
                next_indent = len(next_line) - len(next_line.lstrip())
                current_indent = len(indent)
                
                if next_indent <= current_indent:
                    # End of the block that uses this client
                    break
                
                # This line is part of the block, add indentation
                lines[j] = "    " + next_line
                j += 1
            
            fixes_applied += 1
        
        i += 1
    
    if fixes_applied > 0:
        filepath.write_text('\n'.join(lines))
    
    return fixes_applied


def fix_client_in_async_functions(filepath: Path) -> int:
    """Fix AsyncClient usage in async functions specifically."""
    content = filepath.read_text()
    
    # More sophisticated pattern to handle various cases
    patterns = [
        # Pattern 1: client = httpx.AsyncClient()
        (
            r'(\s*)(client|http_client|async_client)\s*=\s*httpx\.AsyncClient\s*\((.*?)\)',
            r'\1async with httpx.AsyncClient(\3) as \2:'
        ),
        # Pattern 2: self.client = httpx.AsyncClient()
        (
            r'(\s*)(self\.\w+)\s*=\s*httpx\.AsyncClient\s*\((.*?)\)',
            r'\1async with httpx.AsyncClient(\3) as client:\n\1    \2 = client'
        ),
    ]
    
    fixes = 0
    for pattern, replacement in patterns:
        new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
        if count > 0:
            content = new_content
            fixes += count
    
    if fixes > 0:
        filepath.write_text(content)
    
    return fixes


def main():
    """Apply async context manager fixes."""
    report_path = Path('async_test_modernization_report.md')
    
    if not report_path.exists():
        print("Error: async_test_modernization_report.md not found")
        return
    
    # Get unique files from report
    files_to_fix = extract_unique_issues_from_report(report_path)
    
    # For more accurate detection, scan the specific files mentioned
    specific_files = {
        "tests/security/test_api_security.py": 14,
        "tests/security/vulnerability/test_dependency_scanning.py": 1,
        "tests/integration/end_to_end/api_flows/test_api_workflow_validation.py": 6,
        "tests/unit/services/observability/test_observability_dependencies.py": 10,
        "tests/unit/services/cache/test_browser_cache.py": 16,
        "tests/unit/services/crawling/test_lightweight_scraper.py": 21,
        "tests/unit/services/embeddings/test_crawl4ai_bulk_embedder_extended.py": 11,
        "tests/unit/mcp_tools/tools/test_search_utils.py": 19,
        "tests/unit/mcp_tools/tools/test_documents.py": 12,
    }
    
    total_fixes = 0
    
    print("Fixing async context manager issues...")
    
    for filepath_str, expected_count in specific_files.items():
        filepath = Path(filepath_str)
        
        if not filepath.exists():
            print(f"  ⚠️  {filepath} not found")
            continue
        
        # Find issues first
        issues = find_async_client_issues(filepath)
        
        if issues:
            print(f"\n  Analyzing {filepath}:")
            print(f"    Found {len(issues)} async client issues")
            
            # Apply fixes
            fixes = fix_async_client_usage(filepath)
            
            if fixes == 0:
                # Try alternative fixing method
                fixes = fix_client_in_async_functions(filepath)
            
            if fixes > 0:
                total_fixes += fixes
                print(f"    ✓ Fixed {fixes} issues")
            else:
                print(f"    ⚠️  Could not automatically fix - manual review needed")
                # Show the issues for manual review
                for line_num, line_content in issues[:3]:  # Show first 3
                    print(f"      Line {line_num}: {line_content.strip()}")
    
    print(f"\nTotal fixes applied: {total_fixes}")
    
    if total_fixes > 0:
        print("\nRunning formatters...")
        import subprocess
        subprocess.run(['ruff', 'check', 'tests', '--fix'], check=False)
        subprocess.run(['ruff', 'format', 'tests'], check=False)


if __name__ == '__main__':
    main()