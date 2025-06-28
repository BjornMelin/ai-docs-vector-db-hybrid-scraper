#!/usr/bin/env python3
"""Apply High-Confidence F821 Undefined Name Fixes - Phase 3 Type Safety Enhancement.

This script applies automatic fixes for high-confidence F821 undefined name violations
to achieve zero remaining undefined names in the codebase.
"""

import ast
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import json
from dataclasses import dataclass

# Color codes for output
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

@dataclass
class ImportFix:
    """Represents an import fix to be applied."""
    file_path: str
    import_line: str
    undefined_name: str
    fix_description: str

class F821Fixer:
    """Focused fixer for F821 undefined name violations."""
    
    def __init__(self):
        self.applied_fixes = 0
        self.failed_fixes = 0
        self.common_imports = {
            # Typing imports
            'List': 'from typing import List',
            'Dict': 'from typing import Dict', 
            'Set': 'from typing import Set',
            'Tuple': 'from typing import Tuple',
            'Optional': 'from typing import Optional',
            'Union': 'from typing import Union',
            'Any': 'from typing import Any',
            'Callable': 'from typing import Callable',
            'Generator': 'from typing import Generator',
            'Iterator': 'from typing import Iterator',
            'Type': 'from typing import Type',
            'TypeVar': 'from typing import TypeVar',
            'Generic': 'from typing import Generic',
            'Protocol': 'from typing import Protocol',
            'Literal': 'from typing import Literal',
            'Final': 'from typing import Final',
            'ClassVar': 'from typing import ClassVar',
            
            # Standard library imports
            'datetime': 'import datetime',
            'time': 'import time',
            'os': 'import os',
            'sys': 'import sys',
            'json': 'import json',
            'Path': 'from pathlib import Path',
            'defaultdict': 'from collections import defaultdict',
            'Counter': 'from collections import Counter',
            'deque': 'from collections import deque',
            'OrderedDict': 'from collections import OrderedDict',
            
            # Testing imports
            'pytest': 'import pytest',
            'Mock': 'from unittest.mock import Mock',
            'MagicMock': 'from unittest.mock import MagicMock',
            'AsyncMock': 'from unittest.mock import AsyncMock',
            'patch': 'from unittest.mock import patch',
            
            # Async imports
            'asyncio': 'import asyncio',
            'aiohttp': 'import aiohttp',
            
            # Data processing
            'pandas': 'import pandas as pd',
            'numpy': 'import numpy as np',
            
            # Web frameworks
            'fastapi': 'import fastapi',
            'FastAPI': 'from fastapi import FastAPI',
            'Request': 'from fastapi import Request',
            'Response': 'from fastapi import Response',
            'HTTPException': 'from fastapi import HTTPException',
            'Depends': 'from fastapi import Depends',
            
            # Database
            'SQLAlchemy': 'import sqlalchemy',
            'create_engine': 'from sqlalchemy import create_engine',
            'sessionmaker': 'from sqlalchemy.orm import sessionmaker',
        }
    
    def get_f821_violations(self, file_path: str) -> List[Dict]:
        """Get F821 violations for a specific file."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "ruff", "check", file_path,
                "--select", "F821", "--output-format", "json"
            ], capture_output=True, text=True, check=False)
            
            if result.stdout:
                return json.loads(result.stdout)
            return []
            
        except Exception as e:
            print(f"{Colors.RED}Error getting F821 violations for {file_path}: {e}{Colors.RESET}")
            return []
    
    def extract_undefined_name(self, message: str) -> Optional[str]:
        """Extract undefined name from ruff message."""
        match = re.search(r"Undefined name `([^`]+)`", message)
        return match.group(1) if match else None
    
    def can_fix_automatically(self, undefined_name: str) -> bool:
        """Check if we can fix this undefined name automatically."""
        return undefined_name in self.common_imports
    
    def find_import_insertion_point(self, lines: List[str]) -> int:
        """Find the best place to insert imports."""
        # Find the last import line
        last_import_line = -1
        in_docstring = False
        docstring_chars = ['"""', "'''"]
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Track docstring state
            for char in docstring_chars:
                if char in stripped:
                    in_docstring = not in_docstring
                    break
            
            if in_docstring:
                continue
                
            # Skip comments and empty lines at the top
            if stripped.startswith('#') or not stripped:
                continue
                
            # Check for import statements
            if (stripped.startswith('import ') or 
                stripped.startswith('from ') and ' import ' in stripped):
                last_import_line = i
                
            # If we hit non-import code, stop
            elif (stripped and 
                  not stripped.startswith('#') and 
                  not stripped.startswith('"""') and
                  not stripped.startswith("'''")):
                break
        
        # Insert after the last import, or at the beginning if no imports
        return last_import_line + 1 if last_import_line >= 0 else 0
    
    def import_already_exists(self, lines: List[str], import_line: str) -> bool:
        """Check if the import already exists in the file."""
        # Handle different import formats
        if import_line.startswith('from '):
            # Extract module and imported names for "from x import y" format
            match = re.match(r'from\s+(.+?)\s+import\s+(.+)', import_line)
            if match:
                module, names = match.groups()
                names_set = {name.strip() for name in names.split(',')}
                
                for line in lines:
                    line = line.strip()
                    if line.startswith(f'from {module} import'):
                        existing_match = re.match(r'from\s+.+?\s+import\s+(.+)', line)
                        if existing_match:
                            existing_names = {name.strip() for name in existing_match.group(1).split(',')}
                            if names_set.issubset(existing_names):
                                return True
        else:
            # Handle "import x" format
            for line in lines:
                if line.strip() == import_line:
                    return True
        
        return False
    
    def add_import_to_existing_line(self, lines: List[str], import_line: str) -> bool:
        """Try to add import to existing from import line."""
        if not import_line.startswith('from '):
            return False
        
        match = re.match(r'from\s+(.+?)\s+import\s+(.+)', import_line)
        if not match:
            return False
        
        module, new_names = match.groups()
        new_names_set = {name.strip() for name in new_names.split(',')}
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith(f'from {module} import'):
                existing_match = re.match(r'from\s+.+?\s+import\s+(.+)', line_stripped)
                if existing_match:
                    existing_names = [name.strip() for name in existing_match.group(1).split(',')]
                    # Add new names that don't exist
                    for new_name in new_names_set:
                        if new_name not in existing_names:
                            existing_names.append(new_name)
                    
                    # Sort the imports for consistency
                    existing_names.sort()
                    
                    # Reconstruct the line
                    indent = line[:len(line) - len(line.lstrip())]
                    new_line = f"{indent}from {module} import {', '.join(existing_names)}\n"
                    lines[i] = new_line
                    return True
        
        return False
    
    def apply_import_fix(self, file_path: str, undefined_name: str) -> bool:
        """Apply import fix to a file."""
        if not self.can_fix_automatically(undefined_name):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            import_line = self.common_imports[undefined_name]
            
            # Check if import already exists
            if self.import_already_exists(lines, import_line):
                print(f"  ✓ Import for {undefined_name} already exists in {file_path}")
                return True
            
            # Try to add to existing import line first
            if self.add_import_to_existing_line(lines, import_line):
                print(f"  ✓ Added {undefined_name} to existing import in {file_path}")
            else:
                # Add new import line
                insertion_point = self.find_import_insertion_point(lines)
                lines.insert(insertion_point, import_line + '\n')
                print(f"  ✓ Added new import for {undefined_name} in {file_path}")
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            self.applied_fixes += 1
            return True
            
        except Exception as e:
            print(f"  ❌ Error applying fix for {undefined_name} in {file_path}: {e}")
            self.failed_fixes += 1
            return False
    
    def process_file(self, file_path: str) -> int:
        """Process F821 violations in a single file."""
        violations = self.get_f821_violations(file_path)
        if not violations:
            return 0
        
        print(f"{Colors.CYAN}Processing: {file_path}{Colors.RESET}")
        
        fixed_count = 0
        processed_names = set()  # Avoid duplicate processing
        
        for violation in violations:
            undefined_name = self.extract_undefined_name(violation['message'])
            
            if not undefined_name or undefined_name in processed_names:
                continue
            
            processed_names.add(undefined_name)
            
            if self.can_fix_automatically(undefined_name):
                if self.apply_import_fix(file_path, undefined_name):
                    fixed_count += 1
            else:
                print(f"  ⚠️ Cannot auto-fix undefined name: {undefined_name}")
        
        return fixed_count

def main():
    """Main execution function."""
    print(f"{Colors.BOLD}{Colors.MAGENTA}🚀 B2 Code Quality Enhancement - Apply F821 Fixes{Colors.RESET}")
    print(f"{Colors.CYAN}Phase 3: Type Safety Enhancement - Eliminate Undefined Names{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*70}{Colors.RESET}")
    print()
    
    fixer = F821Fixer()
    
    # Get all Python files with F821 violations
    print(f"{Colors.CYAN}🔍 Finding Python files with F821 violations...{Colors.RESET}")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "ruff", "check", ".",
            "--select", "F821", "--output-format", "json"
        ], capture_output=True, text=True, check=False)
        
        if not result.stdout:
            print(f"{Colors.GREEN}✅ No F821 violations found!{Colors.RESET}")
            return 0
        
        violations = json.loads(result.stdout)
        
    except Exception as e:
        print(f"{Colors.RED}❌ Error finding F821 violations: {e}{Colors.RESET}")
        return 1
    
    # Group violations by file
    files_with_violations = defaultdict(list)
    for violation in violations:
        files_with_violations[violation['filename']].append(violation)
    
    print(f"{Colors.CYAN}📁 Found {len(files_with_violations)} files with F821 violations{Colors.RESET}")
    print(f"{Colors.CYAN}🎯 Total F821 violations: {len(violations)}{Colors.RESET}")
    print()
    
    # Process each file
    total_fixed = 0
    for file_path in sorted(files_with_violations.keys()):
        fixed_in_file = fixer.process_file(file_path)
        total_fixed += fixed_in_file
    
    print()
    print(f"{Colors.BOLD}📊 F821 Fix Results:{Colors.RESET}")
    print(f"• Files processed: {len(files_with_violations)}")
    print(f"• Total fixes applied: {fixer.applied_fixes}")
    print(f"• Failed fixes: {fixer.failed_fixes}")
    print()
    
    # Verify results
    print(f"{Colors.CYAN}🔍 Verifying F821 elimination...{Colors.RESET}")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "ruff", "check", ".",
            "--select", "F821", "--statistics"
        ], capture_output=True, text=True, check=False)
        
        if "F821" not in result.stderr:
            print(f"{Colors.GREEN}🎉 SUCCESS: All F821 undefined name violations eliminated!{Colors.RESET}")
        else:
            remaining_match = re.search(r'(\d+)\s+F821', result.stderr)
            remaining = remaining_match.group(1) if remaining_match else "unknown"
            print(f"{Colors.YELLOW}⚠️ {remaining} F821 violations still remain{Colors.RESET}")
            
    except Exception as e:
        print(f"{Colors.RED}❌ Error verifying results: {e}{Colors.RESET}")
    
    print()
    print(f"{Colors.BOLD}✅ Phase 3 F821 Fix Application Complete!{Colors.RESET}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())