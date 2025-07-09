#!/usr/bin/env python3
"""Fix the final 30 POA linting errors."""

import re
from pathlib import Path


def fix_file(filepath: Path) -> bool:
    """Fix errors in a single file."""
    try:
        content = filepath.read_text()
        original_content = content
        
        # Fix specific errors by file
        if filepath.name == "optimization.py":
            # Remove unused contextlib import
            content = content.replace("import contextlib\n", "")
            
        elif filepath.name == "async_optimizer.py":
            # Remove unused contextlib import
            content = content.replace("import contextlib\n", "")
            # Fix TRY401 - Remove e from logger.exception
            content = re.sub(
                r'logger\.exception\(f"Queue worker \{name\} error: \{e\}"\)',
                'logger.exception(f"Queue worker {name} error")',
                content
            )
            
        elif filepath.name == "benchmarks.py":
            # Remove unused imports
            content = re.sub(
                r'from opentelemetry\.trace import Status, StatusCode\n',
                '',
                content
            )
            content = re.sub(
                r'from src\.infrastructure\.clients\.openai_client import OpenAIClient\n',
                '',
                content
            )
            content = re.sub(
                r'from src\.infrastructure\.clients\.redis_client import RedisClientWrapper\n',
                '',
                content
            )
            # Fix open() calls to use Path.open()
            content = re.sub(
                r'with open\(filepath, "w"\) as f:',
                'with filepath.open("w") as f:',
                content
            )
            content = re.sub(
                r'with open\(filepath\) as f:',
                'with filepath.open() as f:',
                content
            )
            
        elif filepath.name == "database_optimizer.py":
            # Remove unused VectorParams import
            content = re.sub(
                r'    VectorParams,\n',
                '',
                content
            )
            # Fix logger.error to logger.exception
            content = re.sub(
                r'logger\.error\(f"(.+?)"\)',
                r'logger.exception(f"\1")',
                content
            )
            # Add noqa comments for unused arguments
            content = re.sub(
                r'        target_recall: float = 0\.95,',
                '        target_recall: float = 0.95,  # noqa: ARG002',
                content
            )
            
        elif filepath.name == "memory_optimizer.py":
            # Fix Type import
            content = re.sub(
                r'from typing import Any, Type',
                'from typing import Any',
                content
            )
            # Add required imports at top
            if 'from collections import defaultdict' not in content:
                content = re.sub(
                    r'(import asyncio\n)',
                    r'\1from collections import defaultdict\n',
                    content
                )
            
        elif filepath.name == "performance_optimizer.py":
            # Add noqa comments for unused query arguments
            content = re.sub(
                r'        query: str,\n        query_type: str,  # noqa: ARG002',
                '        query: str,  # noqa: ARG002\n        query_type: str,  # noqa: ARG002',
                content
            )
            # Add missing imports
            if 'from collections import defaultdict' not in content:
                content = re.sub(
                    r'(import asyncio\n)',
                    r'\1from collections import defaultdict\n',
                    content
                )
            if 'from collections.abc import Callable' not in content:
                content = re.sub(
                    r'(from typing import Any\n)',
                    r'\1from collections.abc import Callable\n',
                    content
                )
        
        if content != original_content:
            filepath.write_text(content)
            return True
        return False
        
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def main():
    """Fix all POA linting errors."""
    base_path = Path("/workspace/repos/ai-docs-vector-db-hybrid-scraper")
    
    files_to_fix = [
        base_path / "src/api/routes/optimization.py",
        base_path / "src/services/performance/async_optimizer.py",
        base_path / "src/services/performance/benchmarks.py",
        base_path / "src/services/performance/database_optimizer.py",
        base_path / "src/services/performance/memory_optimizer.py",
        base_path / "src/services/performance/performance_optimizer.py",
    ]
    
    fixed_count = 0
    for filepath in files_to_fix:
        if filepath.exists():
            if fix_file(filepath):
                fixed_count += 1
                print(f"Fixed: {filepath.name}")
        else:
            print(f"File not found: {filepath}")
    
    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()