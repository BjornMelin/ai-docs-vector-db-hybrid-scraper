#!/usr/bin/env python3
"""Fix remaining async test issues including asyncio.run() and event loop patterns."""

import re
from pathlib import Path
from typing import List, Tuple


def fix_asyncio_run_patterns(file_path: Path) -> bool:
    """Replace asyncio.run() patterns with pytest-asyncio compatible code."""
    try:
        content = file_path.read_text()
        original = content
        
        # Pattern 1: asyncio.run(some_async_func())
        # Replace with: await some_async_func()
        pattern1 = re.compile(r'asyncio\.run\(([^)]+)\)')
        
        # Find all matches and their context
        matches = list(pattern1.finditer(content))
        
        if not matches:
            return False
            
        # Process from end to start to maintain positions
        for match in reversed(matches):
            func_call = match.group(1)
            
            # Check if we're in a test function
            # Look backwards for the function definition
            before_match = content[:match.start()]
            lines_before = before_match.split('\n')
            
            # Find the enclosing function
            func_indent = None
            func_is_async = False
            for i in range(len(lines_before) - 1, -1, -1):
                line = lines_before[i]
                if re.match(r'^\s*def\s+test_', line):
                    func_indent = len(line) - len(line.lstrip())
                    func_is_async = 'async def' in line
                    break
            
            if func_indent is not None:
                # Replace asyncio.run with await
                if func_is_async:
                    content = content[:match.start()] + f'await {func_call}' + content[match.end():]
                else:
                    # Need to make the test async
                    # Find the test function and make it async
                    test_pattern = re.compile(r'^(\s*)def\s+(test_[^(]+)', re.MULTILINE)
                    test_match = None
                    for m in test_pattern.finditer(before_match):
                        test_match = m
                    
                    if test_match:
                        # Make function async
                        indent = test_match.group(1)
                        func_name = test_match.group(2)
                        content = content[:test_match.start()] + f'{indent}async def {func_name}' + content[test_match.end():]
                        
                        # Add pytest.mark.asyncio if not present
                        # Look for decorators above the function
                        func_start = test_match.start()
                        lines = content[:func_start].split('\n')
                        
                        # Check if @pytest.mark.asyncio is already there
                        has_asyncio_mark = False
                        for i in range(len(lines) - 1, max(0, len(lines) - 10), -1):
                            if '@pytest.mark.asyncio' in lines[i]:
                                has_asyncio_mark = True
                                break
                        
                        if not has_asyncio_mark:
                            # Add the decorator
                            lines.append(f'{indent}@pytest.mark.asyncio')
                            content = '\n'.join(lines) + '\n' + content[func_start:]
                        
                        # Now replace asyncio.run with await
                        pattern1 = re.compile(r'asyncio\.run\(([^)]+)\)')
                        content = pattern1.sub(r'await \1', content)
        
        # Ensure pytest is imported if we added marks
        if '@pytest.mark.asyncio' in content and 'import pytest' not in content:
            content = 'import pytest\n' + content
        
        if content != original:
            file_path.write_text(content)
            return True
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
    return False


def fix_event_loop_patterns(file_path: Path) -> bool:
    """Fix event loop anti-patterns."""
    try:
        content = file_path.read_text()
        original = content
        
        # Pattern 1: asyncio.get_event_loop() -> use pytest-asyncio's event_loop fixture
        content = re.sub(
            r'asyncio\.get_event_loop\(\)',
            'event_loop',
            content
        )
        
        # Pattern 2: asyncio.new_event_loop() -> use pytest-asyncio's event_loop fixture
        content = re.sub(
            r'asyncio\.new_event_loop\(\)',
            'event_loop',
            content
        )
        
        # Pattern 3: loop.run_until_complete(coro) -> await coro
        content = re.sub(
            r'[a-zA-Z_]\w*\.run_until_complete\(([^)]+)\)',
            r'await \1',
            content
        )
        
        # Add event_loop fixture parameter if we modified the content
        if content != original and 'event_loop' in content:
            # Find test functions that now use event_loop
            test_pattern = re.compile(r'^(\s*)async\s+def\s+(test_[^(]+)\(([^)]*)\)', re.MULTILINE)
            
            for match in test_pattern.finditer(content):
                params = match.group(3)
                if 'event_loop' in content[match.end():content.find('\n', match.end())] and 'event_loop' not in params:
                    # Add event_loop parameter
                    if params.strip():
                        new_params = params + ', event_loop'
                    else:
                        new_params = 'event_loop'
                    
                    content = content[:match.start(3)] + new_params + content[match.end(3):]
        
        if content != original:
            file_path.write_text(content)
            return True
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
    return False


def main():
    """Fix remaining async test issues."""
    # Files identified with asyncio.run patterns
    files_with_asyncio_run = [
        "tests/benchmarks/performance_suite.py",
        "tests/benchmarks/test_database_performance.py", 
        "tests/benchmarks/test_config_reload_performance.py",
        "tests/unit/services/query_processing/test_federated.py",
        "tests/performance/test_performance_targets.py",
        "tests/unit/services/observability/test_observability_integration.py",
        "tests/unit/services/observability/test_observability_performance.py",
        "tests/utils/performance_utils.py",
    ]
    
    # Files with event loop patterns
    files_with_event_loops = [
        "tests/conftest.py",
        "tests/unit/services/crawling/test_firecrawl_provider.py",
        "tests/unit/cli/conftest.py",
        "tests/unit/mcp_services/conftest.py",
        "tests/fixtures/async_fixtures.py",
        "tests/utils/performance_fixtures.py",
    ]
    
    project_root = Path.cwd()
    
    print("Fixing asyncio.run() patterns...")
    for file_path in files_with_asyncio_run:
        full_path = project_root / file_path
        if full_path.exists():
            if fix_asyncio_run_patterns(full_path):
                print(f"✓ Fixed asyncio.run patterns in {file_path}")
            else:
                print(f"  No changes needed in {file_path}")
    
    print("\nFixing event loop patterns...")
    for file_path in files_with_event_loops:
        full_path = project_root / file_path
        if full_path.exists():
            if fix_event_loop_patterns(full_path):
                print(f"✓ Fixed event loop patterns in {file_path}")
            else:
                print(f"  No changes needed in {file_path}")
    
    print("\nAsync test modernization fixes applied!")
    print("\nNext steps:")
    print("1. Run tests to verify the fixes work correctly")
    print("2. Check for any remaining async anti-patterns")
    print("3. Update any test utilities to use modern async patterns")


if __name__ == "__main__":
    main()