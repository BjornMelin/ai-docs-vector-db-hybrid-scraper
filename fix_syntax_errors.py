#!/usr/bin/env python3
"""Fix syntax errors in f-strings caused by nested quotes."""

import re
from pathlib import Path


def fix_fstring_syntax(file_path: Path) -> bool:
    """Fix f-string syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix nested quotes in f-strings by using single quotes inside
        def fix_nested_quotes(match):
            full_match = match.group(0)
            # Replace double quotes inside f-string with single quotes
            # Look for patterns like f"...{... "text" ...}..."
            fixed = full_match
            
            # Simple replacement: change internal double quotes to single quotes
            # Find content between { and }
            brace_pattern = r'\{([^}]*"[^}]*)\}'
            
            def replace_inner_quotes(inner_match):
                inner_content = inner_match.group(1)
                # Replace double quotes with single quotes
                fixed_inner = inner_content.replace('"', "'")
                return f'{{{fixed_inner}}}'
            
            fixed = re.sub(brace_pattern, replace_inner_quotes, fixed)
            return fixed
        
        # Find f-strings and fix them
        fstring_pattern = r'f"[^"]*"'
        content = re.sub(fstring_pattern, fix_nested_quotes, content)
        
        # Additional fixes for specific known issues
        content = content.replace('f"...{fallback_reason or "pydantic_ai_unavailable"}..."', 
                                  'f"...{fallback_reason or \'pydantic_ai_unavailable\'}..."')
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed syntax in {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Fix syntax errors in target directories."""
    target_dirs = [
        Path("src/services/agents"),
        Path("src/services/security")
    ]
    
    fixed_files = []
    
    for target_dir in target_dirs:
        if not target_dir.exists():
            print(f"Directory {target_dir} does not exist")
            continue
            
        for py_file in target_dir.rglob("*.py"):
            if fix_fstring_syntax(py_file):
                fixed_files.append(py_file)
    
    print(f"\nFixed syntax in {len(fixed_files)} files:")
    for file_path in fixed_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()