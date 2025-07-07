#!/usr/bin/env python3
"""Convert httpx mocks to respx patterns in test files.

This script converts traditional httpx mocking patterns to respx,
which is the modern boundary-level HTTP mocking approach.
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict


def find_httpx_mock_patterns(filepath: Path) -> List[Dict]:
    """Find httpx mocking patterns that should use respx."""
    patterns = []
    content = filepath.read_text()
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        # Pattern 1: patch("httpx.AsyncClient")
        if re.search(r'patch\s*\(\s*["\']httpx\.AsyncClient["\']', line):
            patterns.append({
                'line': i,
                'type': 'patch_async_client',
                'content': line.strip()
            })
        
        # Pattern 2: Mock httpx responses
        if re.search(r'mock.*httpx|httpx.*mock', line, re.IGNORECASE):
            patterns.append({
                'line': i, 
                'type': 'mock_httpx',
                'content': line.strip()
            })
            
        # Pattern 3: MagicMock for httpx
        if 'MagicMock' in line and 'httpx' in lines[max(0, i-3):i+3]:
            patterns.append({
                'line': i,
                'type': 'magic_mock_httpx',
                'content': line.strip()
            })
    
    return patterns


def convert_file_to_respx(filepath: Path) -> int:
    """Convert a test file to use respx instead of httpx mocks."""
    content = filepath.read_text()
    original_content = content
    changes = 0
    
    # Add respx import if needed
    if 'import respx' not in content and ('httpx' in content):
        # Find the right place to add import
        lines = content.split('\n')
        import_added = False
        
        for i, line in enumerate(lines):
            if 'import httpx' in line:
                lines.insert(i + 1, 'import respx')
                import_added = True
                changes += 1
                break
            elif line.startswith('import ') and not import_added:
                if i + 1 < len(lines) and not lines[i + 1].startswith('import'):
                    lines.insert(i + 1, 'import respx')
                    import_added = True
                    changes += 1
                    break
        
        content = '\n'.join(lines)
    
    # Convert patch patterns to respx
    # Pattern: with patch("httpx.AsyncClient") as mock_client:
    content = re.sub(
        r'with\s+patch\s*\(\s*["\']httpx\.AsyncClient["\']\s*\)\s*as\s*(\w+)\s*:',
        r'with respx.mock:',
        content
    )
    
    # Convert mock client response setups
    # Pattern: mock_client.return_value.post.return_value = ...
    content = re.sub(
        r'(\w+)\.return_value\.(get|post|put|delete|patch)\.return_value\s*=\s*',
        r'respx.\2().mock(return_value=',
        content
    )
    
    # Count actual changes
    if content != original_content:
        changes = content.count('respx') - original_content.count('respx')
        filepath.write_text(content)
    
    return changes


def add_respx_fixtures(test_dir: Path):
    """Add respx fixtures to conftest.py if not present."""
    conftest_path = test_dir / 'conftest.py'
    
    if not conftest_path.exists():
        conftest_path.write_text('''"""Test configuration and fixtures."""

import pytest
import respx


@pytest.fixture
async def respx_mock():
    """Provide respx mock for HTTP testing."""
    with respx.mock:
        yield respx


@pytest.fixture
def mock_http_client(respx_mock):
    """Mock HTTP client for testing."""
    return respx_mock
''')
        return True
    
    # Check if respx fixtures already exist
    content = conftest_path.read_text()
    if 'respx' not in content:
        # Add respx import and fixture
        lines = content.split('\n')
        
        # Add import
        import_added = False
        for i, line in enumerate(lines):
            if line.startswith('import pytest'):
                lines.insert(i + 1, 'import respx')
                import_added = True
                break
        
        if not import_added:
            lines.insert(0, 'import respx')
        
        # Add fixture
        lines.extend([
            '',
            '',
            '@pytest.fixture',
            'async def respx_mock():',
            '    """Provide respx mock for HTTP testing."""',
            '    with respx.mock:',
            '        yield respx',
            ''
        ])
        
        conftest_path.write_text('\n'.join(lines))
        return True
    
    return False


def main():
    """Convert httpx mocks to respx patterns."""
    # Files identified in the report
    target_files = [
        "tests/security/test_api_security.py",
        "tests/security/vulnerability/test_dependency_scanning.py", 
        "tests/integration/end_to_end/api_flows/test_api_workflow_validation.py",
        "tests/unit/services/observability/test_observability_dependencies.py",
        "tests/unit/services/cache/test_browser_cache.py",
        "tests/unit/services/crawling/test_lightweight_scraper.py",
        "tests/unit/services/embeddings/test_crawl4ai_bulk_embedder_extended.py",
        "tests/unit/mcp_tools/tools/test_search_utils.py",
        "tests/unit/mcp_tools/tools/test_documents.py",
    ]
    
    print("Converting httpx mocks to respx patterns...\n")
    
    total_changes = 0
    files_with_patterns = []
    
    # First, analyze what needs to be converted
    for filepath_str in target_files:
        filepath = Path(filepath_str)
        if not filepath.exists():
            print(f"  ⚠️  {filepath} not found")
            continue
            
        patterns = find_httpx_mock_patterns(filepath)
        if patterns:
            files_with_patterns.append((filepath, patterns))
            print(f"\n{filepath}:")
            for pattern in patterns[:5]:  # Show first 5
                print(f"  Line {pattern['line']} ({pattern['type']}): {pattern['content']}")
            if len(patterns) > 5:
                print(f"  ... and {len(patterns) - 5} more patterns")
    
    if not files_with_patterns:
        print("\nNo httpx mock patterns found to convert.")
        print("\nThe async context manager issues might be related to:")
        print("1. Missing async with statements for httpx.AsyncClient")
        print("2. Not using respx for HTTP mocking in async tests")
        print("3. General async resource management issues")
        return
    
    # Convert files
    print(f"\n\nConverting {len(files_with_patterns)} files to use respx...")
    
    for filepath, patterns in files_with_patterns:
        changes = convert_file_to_respx(filepath)
        if changes > 0:
            total_changes += changes
            print(f"  ✓ Converted {filepath} ({changes} changes)")
        else:
            print(f"  ⚠️  No automatic conversions for {filepath}")
    
    # Add respx fixtures
    print("\nChecking for respx fixtures...")
    if add_respx_fixtures(Path('tests')):
        print("  ✓ Added respx fixtures to conftest.py")
    
    print(f"\nTotal changes: {total_changes}")
    
    if total_changes > 0:
        print("\nRunning formatters...")
        import subprocess
        subprocess.run(['ruff', 'check', 'tests', '--fix'], check=False)
        subprocess.run(['ruff', 'format', 'tests'], check=False)


if __name__ == '__main__':
    main()