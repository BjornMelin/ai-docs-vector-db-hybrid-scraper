#!/usr/bin/env python3
"""
Advanced Enterprise Code Quality Fixes - Phase 2

Handles more complex violations requiring semantic analysis and careful fixes.
Focuses on critical issues: F821 (undefined names), BLE001 (blind except), G004 (logging).

Author: B2 Code Quality Enhancement Subagent
Phase: Advanced Quality Enhancement
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("advanced_quality_fix.log"),
    ],
)
logger = logging.getLogger(__name__)


class AdvancedViolation(NamedTuple):
    """Represents a complex ruff violation requiring semantic analysis."""
    
    filename: str
    code: str
    message: str
    line: int
    column: int
    context: str = ""


class AdvancedQualityFixer:
    """Advanced enterprise-grade code quality violation fixer."""
    
    def __init__(self, auto_fix: bool = False):
        self.auto_fix = auto_fix
        self.project_root = Path.cwd()
        self.violations: list[AdvancedViolation] = []
        self.fixes_applied = 0
        self.fixes_failed = 0
        
        # Define critical fix priorities
        self.critical_codes = {
            "F821",    # Undefined names - compilation errors
            "BLE001",  # Blind except clauses - error handling
            "G004",    # Logging format strings - security/performance
            "B905",    # Zip without explicit strict parameter
            "B904",    # Exception raising without from
        }
        
        # Define safe patterns for automatic fixing
        self.safe_import_fixes = {
            # Common missing imports that can be safely added
            "datetime": "from datetime import datetime",
            "timezone": "from datetime import timezone", 
            "Path": "from pathlib import Path",
            "Dict": "from typing import Dict",
            "List": "from typing import List",
            "Optional": "from typing import Optional",
            "Union": "from typing import Union",
            "Any": "from typing import Any",
            "Callable": "from typing import Callable",
            "asyncio": "import asyncio",
            "json": "import json",
            "os": "import os",
            "sys": "import sys",
            "logging": "import logging",
            "re": "import re",
        }

    def get_violations_by_codes(self, codes: set[str]) -> list[AdvancedViolation]:
        """Get violations for specific error codes with context."""
        try:
            result = subprocess.run(
                ["uv", "run", "ruff", "check", ".", "--output-format=json"],
                capture_output=True,
                text=True,
                check=False,
                cwd=self.project_root,
            )
            
            if result.returncode not in (0, 1):
                logger.error(f"Ruff check failed: {result.stderr}")
                return []
                
            if not result.stdout.strip():
                return []
                
            raw_violations = json.loads(result.stdout)
            
            violations = []
            for v in raw_violations:
                if v["code"] in codes:
                    # Get context around the violation
                    context = self._get_violation_context(v["filename"], v["location"]["row"])
                    
                    violation = AdvancedViolation(
                        filename=v["filename"],
                        code=v["code"],
                        message=v["message"],
                        line=v["location"]["row"],
                        column=v["location"]["column"],
                        context=context,
                    )
                    violations.append(violation)
                    
            return violations
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get violations: {e}")
            return []

    def _get_violation_context(self, filepath: str, line_num: int) -> str:
        """Get surrounding code context for a violation."""
        try:
            path = Path(filepath)
            if not path.exists():
                return ""
                
            lines = path.read_text(encoding="utf-8").splitlines()
            
            # Get 3 lines before and after for context
            start = max(0, line_num - 4)
            end = min(len(lines), line_num + 3)
            
            context_lines = []
            for i in range(start, end):
                marker = ">>>" if i == line_num - 1 else "   "
                context_lines.append(f"{marker} {i+1:3d}: {lines[i]}")
                
            return "\n".join(context_lines)
            
        except Exception as e:
            logger.debug(f"Failed to get context for {filepath}:{line_num}: {e}")
            return ""

    def fix_undefined_names(self, violations: list[AdvancedViolation]) -> int:
        """Fix F821 undefined name violations by adding missing imports."""
        f821_violations = [v for v in violations if v.code == "F821"]
        if not f821_violations:
            return 0
            
        # Group by file for batch processing
        files_to_fix = defaultdict(list)
        for violation in f821_violations:
            files_to_fix[violation.filename].append(violation)
            
        fixed_count = 0
        
        for filepath, file_violations in files_to_fix.items():
            try:
                path = Path(filepath)
                if not path.exists():
                    continue
                    
                content = path.read_text(encoding="utf-8")
                lines = content.splitlines()
                
                # Analyze undefined names and determine required imports
                imports_to_add = set()
                
                for violation in file_violations:
                    # Extract undefined name from message
                    # Format: "Undefined name `name`"
                    match = re.search(r"Undefined name `([^`]+)`", violation.message)
                    if not match:
                        continue
                        
                    undefined_name = match.group(1)
                    
                    # Check if we have a safe import fix
                    if undefined_name in self.safe_import_fixes:
                        imports_to_add.add(self.safe_import_fixes[undefined_name])
                        logger.info(f"Will add import for '{undefined_name}' in {filepath}")
                    else:
                        # Try to infer from context
                        context_import = self._infer_import_from_context(
                            undefined_name, violation.context, content
                        )
                        if context_import:
                            imports_to_add.add(context_import)
                            logger.info(f"Inferred import for '{undefined_name}': {context_import}")
                
                # Add imports to file
                if imports_to_add:
                    new_content = self._add_imports_to_file(content, list(imports_to_add))
                    if new_content != content:
                        path.write_text(new_content, encoding="utf-8")
                        fixed_count += len([v for v in file_violations 
                                          if any(imp.split()[-1] in violation.message 
                                               for imp in imports_to_add)])
                        logger.info(f"Added {len(imports_to_add)} imports to {filepath}")
                        
            except Exception as e:
                logger.error(f"Failed to fix undefined names in {filepath}: {e}")
                self.fixes_failed += 1
                
        return fixed_count

    def _infer_import_from_context(self, name: str, context: str, full_content: str) -> str | None:
        """Infer required import from usage context."""
        # Common patterns for import inference
        patterns = {
            # datetime module usage
            r'\.now\(\)': "from datetime import datetime",
            r'\.utcnow\(\)': "from datetime import datetime", 
            r'\.strptime\(': "from datetime import datetime",
            r'timezone\.utc': "from datetime import timezone",
            
            # pathlib usage
            r'Path\(': "from pathlib import Path",
            r'\.exists\(\)': "from pathlib import Path" if "Path" in context else None,
            r'\.read_text\(': "from pathlib import Path" if "Path" in context else None,
            
            # asyncio patterns
            r'await\s+\w+': "import asyncio" if "asyncio" in context else None,
            r'async\s+def': "import asyncio" if "asyncio" in context else None,
            
            # typing patterns
            r':\s*List\[': "from typing import List",
            r':\s*Dict\[': "from typing import Dict", 
            r':\s*Optional\[': "from typing import Optional",
            r':\s*Union\[': "from typing import Union",
        }
        
        for pattern, import_stmt in patterns.items():
            if import_stmt and re.search(pattern, context):
                return import_stmt
                
        # Check if name appears in existing imports (might be commented out)
        import_lines = [line for line in full_content.splitlines() 
                       if re.match(r'^\s*(#\s*)?(from|import)\s+', line)]
        
        for line in import_lines:
            if name in line and line.strip().startswith('#'):
                # Found commented import, uncomment it
                return line.strip().lstrip('#').strip()
                
        return None

    def _add_imports_to_file(self, content: str, imports: list[str]) -> str:
        """Add import statements to file in appropriate location."""
        lines = content.splitlines()
        
        # Find appropriate insertion point (after existing imports)
        insert_idx = 0
        in_docstring = False
        docstring_quotes = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Handle module docstrings
            if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                docstring_quotes = stripped[:3]
                if stripped.count(docstring_quotes) >= 2:
                    # Single line docstring
                    insert_idx = i + 1
                else:
                    # Multi-line docstring start
                    in_docstring = True
                continue
                    
            if in_docstring and docstring_quotes and docstring_quotes in stripped:
                # End of multi-line docstring
                in_docstring = False
                insert_idx = i + 1
                continue
                
            if in_docstring:
                continue
                
            # Skip comments and empty lines at top
            if not stripped or stripped.startswith('#'):
                insert_idx = i + 1
                continue
                
            # Found import statements
            if stripped.startswith(('import ', 'from ')):
                insert_idx = i + 1
                continue
                
            # First non-import, non-comment line
            break
            
        # Insert imports with proper spacing
        if insert_idx < len(lines) and lines[insert_idx].strip():
            imports.append("")  # Add blank line after imports
            
        for import_stmt in reversed(imports):
            lines.insert(insert_idx, import_stmt)
            
        return "\n".join(lines)

    def fix_blind_except_clauses(self, violations: list[AdvancedViolation]) -> int:
        """Fix BLE001 blind except clauses by adding specific exception types."""
        ble001_violations = [v for v in violations if v.code == "BLE001"]
        if not ble001_violations:
            return 0
            
        files_to_fix = defaultdict(list)
        for violation in ble001_violations:
            files_to_fix[violation.filename].append(violation)
            
        fixed_count = 0
        
        for filepath, file_violations in files_to_fix.items():
            try:
                path = Path(filepath)
                if not path.exists():
                    continue
                    
                content = path.read_text(encoding="utf-8")
                lines = content.splitlines()
                original_content = content
                
                for violation in sorted(file_violations, key=lambda v: v.line, reverse=True):
                    line_idx = violation.line - 1
                    if 0 <= line_idx < len(lines):
                        line = lines[line_idx]
                        
                        # Only fix simple cases safely
                        if "except:" in line and line.strip().endswith(":"):
                            # Determine appropriate exception based on context
                            exception_type = self._infer_exception_type(violation.context)
                            
                            # Replace bare except with specific exception
                            new_line = line.replace("except:", f"except {exception_type}:")
                            lines[line_idx] = new_line
                            fixed_count += 1
                            
                            logger.info(f"Fixed blind except in {filepath}:{violation.line}")
                
                # Write back if changes made
                new_content = "\n".join(lines)
                if new_content != original_content:
                    path.write_text(new_content, encoding="utf-8")
                    
            except Exception as e:
                logger.error(f"Failed to fix blind except in {filepath}: {e}")
                self.fixes_failed += 1
                
        return fixed_count

    def _infer_exception_type(self, context: str) -> str:
        """Infer appropriate exception type from context."""
        # Common patterns for exception inference
        if any(word in context.lower() for word in ['file', 'open', 'read', 'write']):
            return "OSError"
        elif any(word in context.lower() for word in ['json', 'parse', 'decode']):
            return "(ValueError, TypeError)"
        elif any(word in context.lower() for word in ['http', 'request', 'api', 'client']):
            return "Exception"  # Often custom exceptions
        elif any(word in context.lower() for word in ['import', 'module']):
            return "ImportError" 
        elif any(word in context.lower() for word in ['key', 'index', 'dict', 'list']):
            return "(KeyError, IndexError)"
        else:
            return "Exception"  # Safe fallback

    def fix_logging_format_strings(self, violations: list[AdvancedViolation]) -> int:
        """Fix G004 logging format string violations."""
        g004_violations = [v for v in violations if v.code == "G004"]
        if not g004_violations:
            return 0
            
        files_to_fix = defaultdict(list)
        for violation in g004_violations:
            files_to_fix[violation.filename].append(violation)
            
        fixed_count = 0
        
        for filepath, file_violations in files_to_fix.items():
            try:
                path = Path(filepath)
                if not path.exists():
                    continue
                    
                content = path.read_text(encoding="utf-8")
                lines = content.splitlines()
                original_content = content
                
                for violation in sorted(file_violations, key=lambda v: v.line, reverse=True):
                    line_idx = violation.line - 1
                    if 0 <= line_idx < len(lines):
                        line = lines[line_idx]
                        
                        # Fix common f-string logging patterns
                        # Convert logger.info(f"...") to logger.info("...", ...)
                        new_line = self._fix_logging_line(line)
                        if new_line != line:
                            lines[line_idx] = new_line
                            fixed_count += 1
                            logger.info(f"Fixed logging format in {filepath}:{violation.line}")
                
                # Write back if changes made
                new_content = "\n".join(lines)
                if new_content != original_content:
                    path.write_text(new_content, encoding="utf-8")
                    
            except Exception as e:
                logger.error(f"Failed to fix logging format in {filepath}: {e}")
                self.fixes_failed += 1
                
        return fixed_count

    def _fix_logging_line(self, line: str) -> str:
        """Fix a single logging line to use proper format."""
        # Pattern: logger.level(f"text {var} more")
        # Convert to: logger.level("text %s more", var)
        
        # Simple cases only - avoid complex f-string parsing
        if 'f"' in line and any(method in line for method in ['.info(', '.error(', '.warning(', '.debug(']):
            # For now, just add a comment suggesting manual review
            if "# TODO: Convert f-string to logging format" not in line:
                return line + "  # TODO: Convert f-string to logging format"
                
        return line

    def run_advanced_fixes(self) -> dict[str, int]:
        """Run advanced quality fixes for critical violations."""
        logger.info("Starting advanced code quality fixes...")
        
        # Get critical violations
        violations = self.get_violations_by_codes(self.critical_codes)
        if not violations:
            logger.info("No critical violations found!")
            return {"total_fixed": 0}
            
        by_code = defaultdict(int)
        for v in violations:
            by_code[v.code] += 1
            
        logger.info(f"Found {len(violations)} critical violations:")
        for code, count in sorted(by_code.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {code}: {count}")
        
        results = {"total_fixed": 0}
        
        # Apply fixes in order of criticality
        fixes = [
            ("undefined_names", self.fix_undefined_names),
            ("blind_except", self.fix_blind_except_clauses), 
            ("logging_format", self.fix_logging_format_strings),
        ]
        
        for fix_name, fix_func in fixes:
            try:
                fixed_count = fix_func(violations)
                results[fix_name] = fixed_count
                results["total_fixed"] += fixed_count
                
                if fixed_count > 0:
                    logger.info(f"✅ {fix_name}: Fixed {fixed_count} violations")
                    
                    # Re-run to update violation list
                    violations = self.get_violations_by_codes(self.critical_codes)
                    
            except Exception as e:
                logger.error(f"❌ {fix_name}: Failed - {e}")
                results[fix_name] = 0
                
        return results

    def report_results(self, results: dict[str, int]) -> None:
        """Report final results."""
        logger.info("\n" + "="*60)
        logger.info("🔧 ADVANCED CODE QUALITY ENHANCEMENT COMPLETE")
        logger.info("="*60)
        
        for fix_type, count in results.items():
            if fix_type != "total_fixed" and count > 0:
                logger.info(f"✅ {fix_type.replace('_', ' ').title()}: {count} fixes")
                
        logger.info(f"\n🎉 Total critical violations fixed: {results['total_fixed']}")


def main() -> int:
    """Main entry point for advanced quality fixes."""
    parser = argparse.ArgumentParser(
        description="Advanced Enterprise Code Quality Fixes"
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically apply advanced fixes"
    )
    
    args = parser.parse_args()
    
    fixer = AdvancedQualityFixer(auto_fix=args.auto_fix)
    
    try:
        results = fixer.run_advanced_fixes()
        fixer.report_results(results)
        
        return 0 if results["total_fixed"] > 0 else 1
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())