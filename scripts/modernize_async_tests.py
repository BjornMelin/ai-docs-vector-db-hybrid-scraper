#!/usr/bin/env python3
"""Script to modernize async tests in the codebase."""

import os
import re
from pathlib import Path
from typing import List, Tuple


def find_async_test_files(root_dir: Path) -> List[Path]:
    """Find all test files containing async tests."""
    test_files = []
    for file_path in root_dir.rglob("test_*.py"):
        if ".venv" in str(file_path) or "__pycache__" in str(file_path):
            continue
        
        try:
            content = file_path.read_text()
            if "async def test_" in content:
                test_files.append(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return test_files


def needs_asyncio_import(content: str) -> bool:
    """Check if file needs pytest.mark.asyncio import."""
    has_async_tests = bool(re.search(r'async\s+def\s+test_', content))
    has_asyncio_mark = "@pytest.mark.asyncio" in content
    return has_async_tests and not has_asyncio_mark


def add_asyncio_decorators(content: str) -> Tuple[str, int]:
    """Add @pytest.mark.asyncio decorators to async tests."""
    lines = content.split('\n')
    modified_lines = []
    decorators_added = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is an async test function
        if re.match(r'^(\s*)async\s+def\s+test_', line):
            indent = re.match(r'^(\s*)', line).group(1)
            
            # Check if previous line is already a decorator
            prev_line_idx = i - 1
            while prev_line_idx >= 0 and lines[prev_line_idx].strip() == '':
                prev_line_idx -= 1
            
            if prev_line_idx >= 0:
                prev_line = lines[prev_line_idx].strip()
                # Skip if already has asyncio decorator
                if "@pytest.mark.asyncio" in prev_line:
                    modified_lines.append(line)
                    i += 1
                    continue
            
            # Add the decorator
            modified_lines.append(f"{indent}@pytest.mark.asyncio")
            decorators_added += 1
            
        modified_lines.append(line)
        i += 1
    
    return '\n'.join(modified_lines), decorators_added


def fix_asyncio_run_patterns(content: str) -> Tuple[str, List[str]]:
    """Replace asyncio.run() patterns with proper pytest-asyncio patterns."""
    issues_found = []
    
    # Find asyncio.run patterns
    asyncio_run_pattern = re.compile(r'asyncio\.run\((.*?)\)')
    if asyncio_run_pattern.search(content):
        issues_found.append("Found asyncio.run() - needs conversion to async test")
    
    # Find event loop patterns
    loop_patterns = [
        r'loop\.run_until_complete\(',
        r'get_event_loop\(\)',
        r'new_event_loop\(\)',
        r'set_event_loop\(',
    ]
    
    for pattern in loop_patterns:
        if re.search(pattern, content):
            issues_found.append(f"Found event loop pattern: {pattern}")
    
    return content, issues_found


def update_async_fixtures(content: str) -> Tuple[str, int]:
    """Update async fixtures to use proper patterns."""
    lines = content.split('\n')
    modified_lines = []
    fixtures_updated = 0
    
    for i, line in enumerate(lines):
        # Check for async fixtures
        if re.match(r'^(\s*)@pytest\.fixture.*async\s*def', line):
            # This is already an async fixture with decorator on same line
            modified_lines.append(line)
        elif i > 0 and "@pytest.fixture" in lines[i-1] and re.match(r'^(\s*)async\s+def\s+', line):
            # Async fixture with decorator on previous line
            modified_lines.append(line)
        elif re.match(r'^(\s*)async\s+def\s+.*\(.*\)\s*->\s*.*:.*fixture', line.lower()):
            # Likely an async fixture without decorator
            indent = re.match(r'^(\s*)', line).group(1)
            modified_lines.append(f"{indent}@pytest.fixture")
            modified_lines.append(line)
            fixtures_updated += 1
        else:
            modified_lines.append(line)
    
    return '\n'.join(modified_lines), fixtures_updated


def check_respx_usage(content: str) -> List[str]:
    """Check for proper respx usage patterns."""
    issues = []
    
    # Check if using respx
    if "respx" in content:
        # Check for proper import
        if "import respx" not in content and "from respx import" not in content:
            issues.append("Uses respx but missing import")
        
        # Check for respx decorators or context managers
        if "@respx.mock" not in content and "with respx.mock" not in content:
            issues.append("Uses respx but missing proper mock decorator/context")
    
    # Check for manual HTTP mocking that could use respx
    manual_mock_patterns = [
        r'mock.*\.get\s*=\s*AsyncMock',
        r'mock.*\.post\s*=\s*AsyncMock',
        r'mock.*\.request\s*=\s*AsyncMock',
    ]
    
    for pattern in manual_mock_patterns:
        if re.search(pattern, content):
            issues.append("Manual HTTP mocking detected - consider using respx")
    
    return issues


def process_file(file_path: Path) -> dict:
    """Process a single file and return results."""
    try:
        original_content = file_path.read_text()
        content = original_content
        
        results = {
            'file': str(file_path),
            'changes': [],
            'issues': [],
            'modified': False
        }
        
        # Add asyncio decorators
        content, decorators_added = add_asyncio_decorators(content)
        if decorators_added > 0:
            results['changes'].append(f"Added {decorators_added} @pytest.mark.asyncio decorators")
            results['modified'] = True
        
        # Check for asyncio.run patterns
        _, asyncio_issues = fix_asyncio_run_patterns(content)
        results['issues'].extend(asyncio_issues)
        
        # Update async fixtures
        content, fixtures_updated = update_async_fixtures(content)
        if fixtures_updated > 0:
            results['changes'].append(f"Updated {fixtures_updated} async fixtures")
            results['modified'] = True
        
        # Check respx usage
        respx_issues = check_respx_usage(content)
        results['issues'].extend(respx_issues)
        
        # Write back if modified
        if results['modified']:
            file_path.write_text(content)
        
        return results
        
    except Exception as e:
        return {
            'file': str(file_path),
            'error': str(e),
            'modified': False
        }


def main():
    """Main entry point."""
    root_dir = Path("/workspace/repos/ai-docs-vector-db-hybrid-scraper")
    test_dirs = [root_dir / "tests" / "unit", root_dir / "tests" / "integration"]
    
    print("🔍 Finding async test files...")
    all_files = []
    for test_dir in test_dirs:
        if test_dir.exists():
            files = find_async_test_files(test_dir)
            all_files.extend(files)
    
    print(f"Found {len(all_files)} files with async tests\n")
    
    total_changes = 0
    files_with_issues = []
    
    for file_path in all_files:
        result = process_file(file_path)
        
        if result.get('error'):
            print(f"❌ Error processing {result['file']}: {result['error']}")
        elif result['modified'] or result['issues']:
            if result['modified']:
                total_changes += 1
                print(f"✅ Modified {result['file']}:")
                for change in result['changes']:
                    print(f"   - {change}")
            
            if result['issues']:
                files_with_issues.append(result)
                print(f"⚠️  Issues in {result['file']}:")
                for issue in result['issues']:
                    print(f"   - {issue}")
            print()
    
    print(f"\n📊 Summary:")
    print(f"   - Files processed: {len(all_files)}")
    print(f"   - Files modified: {total_changes}")
    print(f"   - Files with issues: {len(files_with_issues)}")
    
    if files_with_issues:
        print(f"\n⚠️  Files requiring manual attention:")
        for result in files_with_issues:
            print(f"   - {result['file']}")


if __name__ == "__main__":
    main()