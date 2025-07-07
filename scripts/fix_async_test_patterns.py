#!/usr/bin/env python3
"""Automatically fix common async test patterns.

This script applies automated fixes for:
1. Missing @pytest.mark.asyncio decorators
2. Converting httpx mocks to respx patterns
3. Adding proper async context managers
"""

import re
import subprocess
from pathlib import Path
from typing import List, Set, Tuple

import click


def fix_missing_asyncio_marks(filepath: Path) -> bool:
    """Add missing @pytest.mark.asyncio decorators."""
    content = filepath.read_text()
    original = content
    
    # Pattern to find async test functions without the decorator
    pattern = r'^(\s*)(async def test_[^(]+\([^)]*\):)'
    
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is an async test function
        match = re.match(pattern, line)
        if match:
            indent = match.group(1)
            
            # Check if previous lines have @pytest.mark.asyncio
            has_asyncio_mark = False
            j = i - 1
            while j >= 0 and lines[j].strip():
                if '@pytest.mark.asyncio' in lines[j]:
                    has_asyncio_mark = True
                    break
                if not lines[j].strip().startswith('@'):
                    break
                j -= 1
            
            if not has_asyncio_mark:
                # Add the decorator
                fixed_lines.append(f"{indent}@pytest.mark.asyncio")
                
        fixed_lines.append(line)
        i += 1
    
    new_content = '\n'.join(fixed_lines)
    
    if new_content != original:
        filepath.write_text(new_content)
        return True
    return False


def convert_httpx_mock_to_respx(filepath: Path) -> bool:
    """Convert httpx mocks to respx patterns."""
    content = filepath.read_text()
    original = content
    
    # Check if respx is imported
    has_respx_import = 'import respx' in content or 'from respx' in content
    
    # Add respx import if needed
    if not has_respx_import and ('httpx' in content and ('mock' in content or 'Mock' in content)):
        # Find the import section
        lines = content.split('\n')
        import_section_end = 0
        
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_section_end = i + 1
            elif import_section_end > 0 and line.strip() and not line.startswith(' '):
                break
        
        lines.insert(import_section_end, 'import respx')
        content = '\n'.join(lines)
    
    # Convert common patterns
    replacements = [
        # Replace patch of httpx.AsyncClient with respx.mock decorator
        (r'@patch\(["\']httpx\.AsyncClient["\']\)', '@respx.mock'),
        (r'with patch\(["\']httpx\.AsyncClient["\']\) as mock.*:', 'with respx.mock() as respx_mock:'),
        
        # Replace AsyncMock responses with respx patterns
        (r'mock_client\.get\.return_value = AsyncMock\(.*\)', 'respx_mock.get(url).mock(return_value=httpx.Response(200, text=content))'),
        (r'mock_response = AsyncMock\(\)', '# Use respx pattern instead'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    if content != original:
        filepath.write_text(content)
        return True
    return False


def add_example_respx_test(filepath: Path) -> str:
    """Generate an example respx test pattern."""
    return '''
# Example respx pattern for HTTP mocking:
@respx.mock
@pytest.mark.asyncio
async def test_example_with_respx():
    """Example test using respx for HTTP mocking."""
    # Mock the HTTP response
    respx.get("https://example.com/api/data").mock(
        return_value=httpx.Response(
            200, 
            json={"status": "success", "data": [1, 2, 3]}
        )
    )
    
    # Your test code here
    async with httpx.AsyncClient() as client:
        response = await client.get("https://example.com/api/data")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
'''


def fix_async_context_managers(filepath: Path) -> bool:
    """Add async context managers where needed."""
    content = filepath.read_text()
    original = content
    
    # Pattern to find httpx.AsyncClient usage without context manager
    pattern = r'client = httpx\.AsyncClient\(\)'
    replacement = 'async with httpx.AsyncClient() as client:'
    
    content = re.sub(pattern, replacement, content)
    
    # Fix client initialization in fixtures
    pattern2 = r'self\._http_client = httpx\.AsyncClient\('
    if pattern2 in content:
        # This is likely in an initialize method, leave it as is
        pass
    
    if content != original:
        filepath.write_text(content)
        return True
    return False


def process_file(filepath: Path, fix_types: Set[str]) -> List[str]:
    """Process a single file and apply fixes."""
    fixes_applied = []
    
    if 'asyncio_marks' in fix_types:
        if fix_missing_asyncio_marks(filepath):
            fixes_applied.append('Added @pytest.mark.asyncio')
    
    if 'httpx_mocks' in fix_types:
        if convert_httpx_mock_to_respx(filepath):
            fixes_applied.append('Converted httpx mocks to respx')
    
    if 'context_managers' in fix_types:
        if fix_async_context_managers(filepath):
            fixes_applied.append('Added async context managers')
    
    return fixes_applied


@click.command()
@click.option('--fix-asyncio-marks', is_flag=True, help='Fix missing @pytest.mark.asyncio')
@click.option('--fix-httpx-mocks', is_flag=True, help='Convert httpx mocks to respx')
@click.option('--fix-context-managers', is_flag=True, help='Add async context managers')
@click.option('--test-file', help='Fix a specific test file')
@click.option('--dry-run', is_flag=True, help='Show what would be fixed without making changes')
def main(fix_asyncio_marks, fix_httpx_mocks, fix_context_managers, test_file, dry_run):
    """Fix common async test patterns."""
    fix_types = set()
    
    if fix_asyncio_marks:
        fix_types.add('asyncio_marks')
    if fix_httpx_mocks:
        fix_types.add('httpx_mocks')
    if fix_context_managers:
        fix_types.add('context_managers')
    
    if not fix_types:
        click.echo("No fix types specified. Use --help to see available options.")
        return
    
    if test_file:
        files = [Path(test_file)]
    else:
        # Get files from the modernization report
        report_path = Path('async_test_modernization_report.md')
        if not report_path.exists():
            click.echo("Run modernize_async_tests.py first to generate the report.")
            return
        
        report = report_path.read_text()
        files = set()
        
        # Extract file paths from the report
        for line in report.split('\n'):
            if line.startswith('- tests/'):
                # Handle lines with line numbers (e.g., "- tests/file.py:123")
                file_path = line[2:].split(':')[0]
                files.add(Path(file_path))
    
    if dry_run:
        click.echo("DRY RUN - No changes will be made")
    
    total_fixes = 0
    for filepath in files:
        if not filepath.exists():
            continue
        
        if dry_run:
            click.echo(f"\nWould process: {filepath}")
            # Just show what would be done
            content = filepath.read_text()
            if 'asyncio_marks' in fix_types and 'async def test_' in content and '@pytest.mark.asyncio' not in content:
                click.echo("  - Would add @pytest.mark.asyncio decorators")
            if 'httpx_mocks' in fix_types and 'httpx' in content and ('mock' in content or 'Mock' in content):
                click.echo("  - Would convert httpx mocks to respx")
            if 'context_managers' in fix_types and 'httpx.AsyncClient()' in content:
                click.echo("  - Would add async context managers")
        else:
            fixes = process_file(filepath, fix_types)
            if fixes:
                total_fixes += len(fixes)
                click.echo(f"\n{filepath}:")
                for fix in fixes:
                    click.echo(f"  ✓ {fix}")
    
    if not dry_run:
        click.echo(f"\nTotal fixes applied: {total_fixes}")
        
        # Run formatters
        if total_fixes > 0:
            click.echo("\nRunning formatters...")
            subprocess.run(['ruff', 'check', 'tests', '--fix'], check=False)
            subprocess.run(['ruff', 'format', 'tests'], check=False)


if __name__ == '__main__':
    main()