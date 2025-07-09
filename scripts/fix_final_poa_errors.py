#!/usr/bin/env python3
"""Fix final linting errors in POA files."""

import re
from pathlib import Path


def fix_redefined_loop_var(content: str) -> str:
    """Fix PLW2901: loop variable overwritten."""
    # Fix the chunk redefinition in api_optimizer.py
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        if 'async for chunk in data_generator:' in line:
            new_lines.append(line)
            # Process next few lines carefully
            j = i + 1
            while j < len(lines) and j < i + 10:
                if 'chunk = processed_chunk.encode' in lines[j]:
                    new_lines.append(lines[j].replace('chunk =', 'chunk_data ='))
                elif 'chunk = chunk.encode' in lines[j]:
                    new_lines.append(lines[j].replace('chunk = chunk.encode', 'chunk_data = chunk.encode'))
                elif 'compressed = gzip.compress(chunk)' in lines[j]:
                    new_lines.append(lines[j].replace('compress(chunk)', 'compress(chunk_data)'))
                else:
                    new_lines.append(lines[j])
                j += 1
            i = j - 1
        elif i >= len(new_lines):
            new_lines.append(line)
    
    return '\n'.join(new_lines)


def fix_logging_exception(content: str) -> str:
    """Fix TRY400: Use logging.exception instead of error."""
    # Replace logger.error with logger.exception in except blocks
    content = re.sub(
        r'except Exception as (\w+):\s*logger\.error\(f"([^"]+)"\)',
        r'except Exception as \1:\n                logger.exception(f"\2")',
        content
    )
    
    return content


def fix_unused_imports(content: str) -> str:
    """Remove specific unused imports."""
    # Remove unused imports
    content = re.sub(r'from collections import defaultdict\n', '', content)
    content = re.sub(r'from collections.abc import.*Callable.*\n', '', content)
    content = re.sub(r'from src.utils.async_utils import gather_with_taskgroup\n', '', content)
    
    return content


def fix_suppressible_exception(content: str) -> str:
    """Fix SIM105: Use contextlib.suppress."""
    # Add contextlib import if needed
    if 'contextlib.suppress' in content or 'try:' not in content:
        return content
        
    # Add import if not present
    if 'import contextlib' not in content and 'from contextlib import' not in content:
        # Find where to add import
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('import ') and i < 20:
                lines.insert(i, 'import contextlib')
                break
        content = '\n'.join(lines)
    
    # Replace try-except-pass with suppress
    content = re.sub(
        r'try:\s*type_sizes\[obj_type\] \+= sys\.getsizeof\(obj\)\s*except TypeError:\s*pass.*?Some objects',
        'with contextlib.suppress(TypeError):\n                type_sizes[obj_type] += sys.getsizeof(obj)\n            # Some objects',
        content,
        flags=re.DOTALL
    )
    
    return content


def add_slp001_noqa(content: str) -> str:
    """Add noqa comments for private member access."""
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if '_value' in line and 'semaphore._value' in line and '# noqa: SLF001' not in line:
            line = line.rstrip() + '  # noqa: SLF001'
        elif '_gc_disabled_contexts' in line and 'self.optimizer._gc_disabled_contexts' in line and '# noqa: SLF001' not in line:
            line = line.rstrip() + '  # noqa: SLF001'
        elif '_initial_value' in line and '# noqa: SLF001' not in line:
            line = line.rstrip() + '  # noqa: SLF001'
        new_lines.append(line)
    
    return '\n'.join(new_lines)


def add_arg002_noqa(content: str) -> str:
    """Add noqa for unused method arguments."""
    # Add noqa for specific methods with unused args
    patterns = [
        r'(async def _optimize_batch_queries.*?query_type: str)',
        r'(async def _optimize_filter_pushdown.*?query_type: str)',
        r'(async def _optimize_projection.*?query_type: str)',
    ]
    
    for pattern in patterns:
        content = re.sub(
            pattern,
            lambda m: m.group(0) + '  # noqa: ARG002' if '# noqa:' not in m.group(0) else m.group(0),
            content,
            flags=re.DOTALL
        )
    
    return content


def add_try_noqa(content: str) -> str:
    """Add noqa for TRY300 and TRY301."""
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        if 'raise HTTPException(' in line and i > 0 and 'if snapshot.get("status")' in lines[i-1]:
            # This is TRY301
            if '# noqa:' not in line:
                line = line.rstrip() + '  # noqa: TRY301'
        elif 'return {' in line and i > 2 and 'await poa.' in lines[i-2]:
            # This is TRY300
            if '# noqa:' not in line:
                line = line.rstrip() + '  # noqa: TRY300'
        new_lines.append(line)
    
    return '\n'.join(new_lines)


def fix_file(file_path: Path) -> None:
    """Apply all fixes to a file."""
    try:
        content = file_path.read_text()
        original_content = content
        
        # Apply fixes
        content = fix_redefined_loop_var(content)
        content = fix_logging_exception(content)
        content = fix_unused_imports(content)
        content = fix_suppressible_exception(content)
        content = add_slp001_noqa(content)
        content = add_arg002_noqa(content)
        content = add_try_noqa(content)
        
        if content != original_content:
            file_path.write_text(content)
            print(f"Fixed errors in: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    """Fix final linting errors."""
    files_to_fix = [
        "src/api/routes/optimization.py",
        "src/services/performance/api_optimizer.py",
        "src/services/performance/async_optimizer.py",
        "src/services/performance/memory_optimizer.py",
        "src/services/performance/performance_optimizer.py",
        "src/services/performance/benchmarks.py",
    ]
    
    for file_path in files_to_fix:
        path = Path(file_path)
        if path.exists():
            fix_file(path)


if __name__ == "__main__":
    main()