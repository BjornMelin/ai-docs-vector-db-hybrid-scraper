#!/usr/bin/env python3
"""Fix final remaining async test issues."""

import re
from pathlib import Path


def fix_asyncio_run_patterns(content: str) -> str:
    """Replace asyncio.run() patterns with proper pytest-asyncio patterns."""
    # Pattern 1: Simple asyncio.run calls
    content = re.sub(
        r'(\s+)(result = )?asyncio\.run\(([^)]+)\)\)',
        r'\1\2await \3)',
        content
    )
    
    # Pattern 2: asyncio.run without assignment
    content = re.sub(
        r'(\s+)asyncio\.run\(([^)]+)\)\)',
        r'\1await \2)',
        content
    )
    
    # Pattern 3: Inside try blocks
    content = re.sub(
        r'(\s+)try:\n(\s+)asyncio\.run\(([^)]+)\)\)',
        r'\1await \3)',
        content
    )
    
    return content


def ensure_async_test_decorators(content: str) -> str:
    """Ensure test methods that use await have @pytest.mark.asyncio."""
    lines = content.split('\n')
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a test method definition
        if re.match(r'^\s*def test_', line):
            # Look ahead for await usage
            j = i + 1
            has_await = False
            indent_level = len(line) - len(line.lstrip())
            
            while j < len(lines) and (lines[j].strip() == '' or len(lines[j]) - len(lines[j].lstrip()) > indent_level):
                if 'await ' in lines[j]:
                    has_await = True
                    break
                j += 1
            
            if has_await:
                # Check if there's already a decorator
                k = i - 1
                has_async_decorator = False
                while k >= 0 and (lines[k].strip().startswith('@') or lines[k].strip() == ''):
                    if '@pytest.mark.asyncio' in lines[k]:
                        has_async_decorator = True
                        break
                    k -= 1
                
                if not has_async_decorator:
                    # Add the decorator
                    indent = ' ' * indent_level
                    new_lines.append(f'{indent}@pytest.mark.asyncio')
                
                # Make the function async
                if 'async def' not in line:
                    line = line.replace('def ', 'async def ')
        
        new_lines.append(line)
        i += 1
    
    return '\n'.join(new_lines)


def fix_observability_instrumentation(file_path: Path) -> None:
    """Fix async issues in test_observability_instrumentation.py."""
    content = file_path.read_text()
    
    # Fix asyncio.run patterns
    content = fix_asyncio_run_patterns(content)
    
    # Ensure async test decorators
    content = ensure_async_test_decorators(content)
    
    file_path.write_text(content)
    print(f"Fixed {file_path}")


def fix_firecrawl_provider(file_path: Path) -> None:
    """Fix async issues in test_firecrawl_provider.py."""
    content = file_path.read_text()
    
    # Check for asyncio patterns
    if 'asyncio.run' in content or 'loop.run_until_complete' in content:
        content = fix_asyncio_run_patterns(content)
        content = ensure_async_test_decorators(content)
        file_path.write_text(content)
        print(f"Fixed {file_path}")
    else:
        print(f"No fixes needed for {file_path}")


def fix_test_utils(file_path: Path) -> None:
    """Fix async issues in test_utils.py."""
    content = file_path.read_text()
    
    # Check for asyncio patterns
    if 'asyncio.run' in content or 'loop.run_until_complete' in content:
        content = fix_asyncio_run_patterns(content)
        content = ensure_async_test_decorators(content)
        file_path.write_text(content)
        print(f"Fixed {file_path}")
    else:
        print(f"No fixes needed for {file_path}")


def fix_remaining_test_files():
    """Fix remaining test files with async issues."""
    base_path = Path("/workspace/repos/ai-docs-vector-db-hybrid-scraper")
    
    # List of test files that need fixing
    files_to_fix = [
        ("tests/unit/services/observability/test_observability_instrumentation.py", fix_observability_instrumentation),
        ("tests/unit/services/crawling/test_firecrawl_provider.py", fix_firecrawl_provider),
        ("tests/unit/utils/test_utils.py", fix_test_utils),
    ]
    
    for file_path, fix_function in files_to_fix:
        full_path = base_path / file_path
        if full_path.exists():
            try:
                fix_function(full_path)
            except Exception as e:
                print(f"Error fixing {full_path}: {e}")
        else:
            print(f"File not found: {full_path}")


if __name__ == "__main__":
    fix_remaining_test_files()