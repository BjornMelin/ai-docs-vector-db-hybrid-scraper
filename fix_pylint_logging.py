#!/usr/bin/env python3
"""Fix pylint E1205 logging format errors by converting to f-strings."""

import re
import ast
from pathlib import Path


def fix_logging_to_fstring(file_path: Path) -> bool:
    """Convert problematic logging calls to f-strings."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern to match logger calls with format strings and arguments
        # Look for: logger.method("format string %s %s", arg1, arg2)
        pattern = r'(logger\.\w+)\(\s*"([^"]*?)",\s*([^)]+)\)'
        
        def convert_to_fstring(match):
            method = match.group(1)
            format_str = match.group(2)
            args_str = match.group(3).strip()
            
            # Skip if no format specifiers
            if '%' not in format_str:
                return match.group(0)
            
            # Split arguments but handle nested calls
            args = []
            bracket_count = 0
            paren_count = 0
            current_arg = ""
            
            for char in args_str:
                if char == ',' and bracket_count == 0 and paren_count == 0:
                    args.append(current_arg.strip())
                    current_arg = ""
                else:
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                    elif char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                    current_arg += char
            
            if current_arg.strip():
                args.append(current_arg.strip())
            
            # Convert %s and %d to f-string format
            f_string = format_str
            arg_index = 0
            
            # Replace format specifiers with f-string syntax
            def replace_format(m):
                nonlocal arg_index
                if arg_index < len(args):
                    arg = args[arg_index]
                    arg_index += 1
                    spec = m.group(1)
                    if spec in ['%s', '%d', '%f']:
                        return f"{{{arg}}}"
                    elif spec.startswith('%.') and spec.endswith('f'):
                        # Handle precision formatting like %.2f
                        precision = spec[2:-1]
                        return f"{{{arg}:.{precision}f}}"
                    elif spec == '%%':
                        return '%'
                    else:
                        return f"{{{arg}}}"
                return m.group(0)
            
            # Find and replace format specifiers
            f_string = re.sub(r'(%\.?\d*[sdfg%]|%%)', replace_format, f_string)
            
            return f'{method}(f"{f_string}")'
        
        # Apply conversion
        content = re.sub(pattern, convert_to_fstring, content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Converted logging in {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Convert logging format strings to f-strings in target directories."""
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
            if fix_logging_to_fstring(py_file):
                fixed_files.append(py_file)
    
    print(f"\nConverted logging in {len(fixed_files)} files:")
    for file_path in fixed_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()