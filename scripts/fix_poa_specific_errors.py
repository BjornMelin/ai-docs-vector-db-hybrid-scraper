#!/usr/bin/env python3
"""Fix specific linting errors B904 and B008 in POA files."""

import re
from pathlib import Path


def fix_b904_errors(content: str) -> str:
    """Fix B904: raise exceptions with 'raise ... from err' in except blocks."""
    # Pattern to match except blocks with raise
    pattern = r'except\s+(\w+(?:\s*\|\s*\w+)*)\s+as\s+(\w+):(.*?)raise\s+([\w.]+)\((.*?)\)(?=\s*(?:except|finally|$))'
    
    def replacer(match):
        exception_type = match.group(1)
        exception_var = match.group(2)
        block_content = match.group(3)
        raise_exception = match.group(4)
        raise_args = match.group(5)
        
        # Check if this is already using 'from'
        if ' from ' in block_content:
            return match.group(0)
        
        # Add 'from err' to the raise statement
        return f'except {exception_type} as {exception_var}:{block_content}raise {raise_exception}({raise_args}) from {exception_var}'
    
    # Apply the fix
    content = re.sub(pattern, replacer, content, flags=re.DOTALL | re.MULTILINE)
    
    # Also handle simple patterns like HTTPException without intermediate processing
    pattern2 = r'except Exception as (\w+):\s*logger\.exception\((.*?)\)\s*raise HTTPException\((.*?)\)'
    
    def replacer2(match):
        var = match.group(1)
        log_msg = match.group(2)
        http_args = match.group(3)
        return f'except Exception as {var}:\n        logger.exception({log_msg})\n        raise HTTPException({http_args}) from {var}'
    
    content = re.sub(pattern2, replacer2, content, flags=re.DOTALL)
    
    return content


def fix_b008_errors_for_fastapi(content: str) -> str:
    """Fix B008 for FastAPI by using Annotated pattern."""
    # Don't fix B008 for FastAPI Depends - it's a false positive
    # FastAPI requires Depends() in default arguments
    return content


def fix_file(file_path: Path) -> None:
    """Fix linting errors in a file."""
    try:
        content = file_path.read_text()
        original_content = content
        
        # Apply fixes
        content = fix_b904_errors(content)
        
        # Only write if changed
        if content != original_content:
            file_path.write_text(content)
            print(f"Fixed B904 errors in: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    """Fix specific linting errors."""
    files_to_fix = [
        "src/api/routes/optimization.py",
        "src/services/performance/poa_service.py",
        "src/services/performance/benchmarks.py",
        "src/services/performance/api_optimizer.py",
        "src/services/performance/async_optimizer.py",
        "src/services/performance/database_optimizer.py",
        "src/services/performance/memory_optimizer.py",
        "src/services/performance/performance_optimizer.py",
    ]
    
    for file_path in files_to_fix:
        path = Path(file_path)
        if path.exists():
            fix_file(path)


if __name__ == "__main__":
    main()