#!/usr/bin/env python3
"""Add noqa: B008 comments for FastAPI Depends usage."""

import re
from pathlib import Path


def add_noqa_for_depends(content: str) -> str:
    """Add noqa: B008 for FastAPI Depends patterns."""
    # Pattern to match Depends() in function arguments
    pattern = r'(\s+\w+: [^=]+ = Depends\([^)]+\)),?$'
    
    def replacer(match):
        line = match.group(0)
        # Check if already has noqa
        if '# noqa' in line:
            return line
        # Add noqa: B008
        if line.endswith(','):
            return line[:-1] + ',  # noqa: B008'
        else:
            return line + '  # noqa: B008'
    
    # Apply line by line
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if ' = Depends(' in line and '# noqa' not in line:
            line = re.sub(r'(\s+\w+: [^=]+ = Depends\([^)]+\))(,?)$', r'\1\2  # noqa: B008', line)
        new_lines.append(line)
    
    return '\n'.join(new_lines)


def fix_file(file_path: Path) -> None:
    """Add noqa comments to a file."""
    try:
        content = file_path.read_text()
        original_content = content
        
        content = add_noqa_for_depends(content)
        
        if content != original_content:
            file_path.write_text(content)
            print(f"Added noqa comments to: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    """Add noqa comments to FastAPI routes."""
    files = [
        "src/api/routes/optimization.py",
    ]
    
    for file_path in files:
        path = Path(file_path)
        if path.exists():
            fix_file(path)


if __name__ == "__main__":
    main()