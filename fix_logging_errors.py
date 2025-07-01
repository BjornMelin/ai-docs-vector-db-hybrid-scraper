#!/usr/bin/env python3
"""Fix pylint E1205 logging format errors across the codebase."""

import re
import os
from pathlib import Path


def fix_logging_format_errors(file_path: Path) -> bool:
    """Fix logging format errors in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: logger.info("message %s with %s", arg1, arg2) -> logger.info("message %s with %s", arg1, arg2)
        # This pattern finds logging calls with too many format specifiers
        
        # Find logging calls with format strings
        pattern = r'(logger\.\w+)\(\s*"([^"]*?)",\s*(.*?)\)'
        
        def fix_logging_call(match):
            method = match.group(1)
            format_str = match.group(2)
            args_str = match.group(3).strip()
            
            # Count % format specifiers in the format string
            format_count = format_str.count('%s') + format_str.count('%d') + format_str.count('%f') + format_str.count('%.') 
            
            if not args_str:
                # No arguments, just return the logging call as-is
                return f'{method}("{format_str}")'
            
            # Split arguments but be careful with nested function calls
            args_list = []
            paren_count = 0
            current_arg = ""
            
            for char in args_str:
                if char == ',' and paren_count == 0:
                    args_list.append(current_arg.strip())
                    current_arg = ""
                else:
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                    current_arg += char
            
            if current_arg.strip():
                args_list.append(current_arg.strip())
            
            arg_count = len(args_list)
            
            # If format count matches arg count, no problem
            if format_count == arg_count:
                return match.group(0)
            
            # If we have more args than format specifiers, we have the E1205 error
            if arg_count > format_count:
                # Add more %s specifiers to match the number of arguments
                needed_specifiers = arg_count - format_count
                if format_str.endswith('.'):
                    # If format string ends with period, add before period
                    format_str = format_str[:-1] + ' ' + ' '.join(['%s'] * needed_specifiers) + '.'
                else:
                    # Add at the end
                    format_str += ' ' + ' '.join(['%s'] * needed_specifiers)
            
            return f'{method}("{format_str}", {args_str})'
        
        # Apply the fix
        content = re.sub(pattern, fix_logging_call, content)
        
        # If content changed, write it back
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed logging errors in {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Fix logging errors in agents and security directories."""
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
            if fix_logging_format_errors(py_file):
                fixed_files.append(py_file)
    
    print(f"\nFixed {len(fixed_files)} files:")
    for file_path in fixed_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()