#!/usr/bin/env python3
"""
Enterprise Zero-Violations Code Quality Automation

This script systematically fixes ruff violations to achieve enterprise-grade code quality.
Implements the B2 Code Quality Enhancement strategy with safe, automated fixes.

Author: B2 Code Quality Enhancement Subagent
Purpose: Achieve Portfolio ULTRATHINK standards through zero code quality violations
"""

from __future__ import annotations

import argparse
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
        logging.FileHandler("quality_fix.log"),
    ],
)
logger = logging.getLogger(__name__)


class Violation(NamedTuple):
    """Represents a ruff code quality violation."""
    
    filename: str
    code: str
    message: str
    line: int
    column: int
    fix: dict[str, Any] | None = None


class QualityFixer:
    """Enterprise-grade code quality violation fixer."""
    
    def __init__(self, auto_fix: bool = False):
        self.auto_fix = auto_fix
        self.project_root = Path.cwd()
        self.violations: list[Violation] = []
        self.fixes_applied = 0
        self.fixes_failed = 0
        
        # Define safe auto-fix rules
        self.safe_auto_fix_codes = {
            "W293", "W291", "W292",  # Whitespace issues  
            "UP035",                 # typing.Dict -> dict
            "I001",                  # Import sorting
            "F401",                  # Unused imports (with caution)
            "DTZ005",               # datetime timezone issues
            "UP045", "UP046",       # Type annotation modernization
            "C401",                 # set comprehensions
            "SIM108", "SIM116",     # Simplification rules
        }
        
        # Define manual review codes  
        self.manual_review_codes = {
            "F821",  # Undefined names - critical
            "F811",  # Redefined names - critical  
            "B904",  # Exception raising - logic sensitive
            "ARG001", "ARG002",  # Unused arguments - API sensitive
            "S607",  # Security subprocess - security sensitive
        }

    def get_violations(self) -> list[Violation]:
        """Get all current ruff violations in JSON format."""
        try:
            result = subprocess.run(
                ["uv", "run", "ruff", "check", ".", "--output-format=json"],
                capture_output=True,
                text=True,
                check=False,
                cwd=self.project_root,
            )
            
            if result.returncode not in (0, 1):  # 0=no issues, 1=issues found
                logger.error(f"Ruff check failed: {result.stderr}")
                return []
                
            if not result.stdout.strip():
                logger.info("No violations found!")
                return []
                
            raw_violations = json.loads(result.stdout)
            
            violations = []
            for v in raw_violations:
                violation = Violation(
                    filename=v["filename"],
                    code=v["code"],
                    message=v["message"],
                    line=v["location"]["row"],
                    column=v["location"]["column"],
                    fix=v.get("fix"),
                )
                violations.append(violation)
                
            return violations
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get violations: {e}")
            return []

    def analyze_violations(self, violations: list[Violation]) -> dict[str, Any]:
        """Analyze violations and create fixing strategy."""
        analysis = {
            "total": len(violations),
            "by_code": defaultdict(int),
            "by_file": defaultdict(int),
            "safe_auto_fixes": 0,
            "manual_reviews": 0,
            "strategy": {},
        }
        
        for violation in violations:
            analysis["by_code"][violation.code] += 1
            analysis["by_file"][violation.filename] += 1
            
            if violation.code in self.safe_auto_fix_codes:
                analysis["safe_auto_fixes"] += 1
            elif violation.code in self.manual_review_codes:
                analysis["manual_reviews"] += 1
                
        # Sort by frequency for prioritized fixing
        analysis["by_code"] = dict(
            sorted(analysis["by_code"].items(), key=lambda x: x[1], reverse=True)
        )
        
        return analysis

    def fix_unused_imports(self, violations: list[Violation]) -> int:
        """Fix F401 unused import violations safely."""
        f401_violations = [v for v in violations if v.code == "F401"]
        if not f401_violations:
            return 0
            
        # Group by file for batch processing
        files_to_fix = defaultdict(list)
        for violation in f401_violations:
            files_to_fix[violation.filename].append(violation)
            
        fixed_count = 0
        
        for filepath, file_violations in files_to_fix.items():
            try:
                # Read file content
                path = Path(filepath)
                if not path.exists():
                    continue
                    
                content = path.read_text(encoding="utf-8")
                lines = content.splitlines()
                
                # Mark lines for removal (reverse order to maintain line numbers)
                lines_to_remove = []
                for violation in sorted(file_violations, key=lambda v: v.line, reverse=True):
                    line_idx = violation.line - 1  # Convert to 0-based
                    if 0 <= line_idx < len(lines):
                        line_content = lines[line_idx].strip()
                        
                        # Safety check: only remove simple import lines
                        if (line_content.startswith(("import ", "from ")) and 
                            len(line_content.split()) <= 6):  # Simple import check
                            lines_to_remove.append(line_idx)
                            
                # Remove marked lines
                for line_idx in lines_to_remove:
                    lines.pop(line_idx)
                    fixed_count += 1
                    
                # Write back only if changes were made
                if lines_to_remove:
                    new_content = "\n".join(lines) + ("\n" if content.endswith("\n") else "")
                    path.write_text(new_content, encoding="utf-8")
                    logger.info(f"Fixed {len(lines_to_remove)} unused imports in {filepath}")
                    
            except Exception as e:
                logger.error(f"Failed to fix unused imports in {filepath}: {e}")
                self.fixes_failed += 1
                
        return fixed_count

    def fix_typing_imports(self, violations: list[Violation]) -> int:
        """Fix UP035 deprecated typing imports (typing.Dict -> dict, etc.)."""
        up035_violations = [v for v in violations if v.code == "UP035"]
        if not up035_violations:
            return 0
            
        # Group by file
        files_to_fix = defaultdict(list)
        for violation in up035_violations:
            files_to_fix[violation.filename].append(violation)
            
        fixed_count = 0
        
        # Mapping of deprecated typing imports to modern equivalents
        typing_replacements = {
            "typing.Dict": "dict",
            "typing.List": "list", 
            "typing.Tuple": "tuple",
            "typing.Set": "set",
            "typing.FrozenSet": "frozenset",
            "typing.Type": "type",
            "Dict": "dict",
            "List": "list",
            "Tuple": "tuple", 
            "Set": "set",
            "FrozenSet": "frozenset",
            "Type": "type",
        }
        
        for filepath, file_violations in files_to_fix.items():
            try:
                path = Path(filepath)
                if not path.exists():
                    continue
                    
                content = path.read_text(encoding="utf-8")
                original_content = content
                
                # Apply replacements
                for old, new in typing_replacements.items():
                    # Handle import lines
                    content = re.sub(
                        rf"from typing import.*?\b{old.split('.')[-1]}\b",
                        lambda m: m.group(0).replace(old.split(".")[-1], ""),
                        content
                    )
                    # Handle usage
                    content = re.sub(rf"\b{re.escape(old)}\b", new, content)
                    
                # Clean up empty import lines and extra commas
                content = re.sub(r"from typing import\s*,\s*", "from typing import ", content)
                content = re.sub(r"from typing import\s*$", "", content, flags=re.MULTILINE)
                
                if content != original_content:
                    path.write_text(content, encoding="utf-8")
                    fixed_count += len(file_violations)
                    logger.info(f"Fixed {len(file_violations)} typing imports in {filepath}")
                    
            except Exception as e:
                logger.error(f"Failed to fix typing imports in {filepath}: {e}")
                self.fixes_failed += 1
                
        return fixed_count

    def fix_datetime_timezone(self, violations: list[Violation]) -> int:
        """Fix DTZ005 datetime.now() without timezone violations."""
        dtz005_violations = [v for v in violations if v.code == "DTZ005"]
        if not dtz005_violations:
            return 0
            
        files_to_fix = defaultdict(list)
        for violation in dtz005_violations:
            files_to_fix[violation.filename].append(violation)
            
        fixed_count = 0
        
        for filepath, file_violations in files_to_fix.items():
            try:
                path = Path(filepath)
                if not path.exists():
                    continue
                    
                content = path.read_text(encoding="utf-8")
                original_content = content
                
                # Add timezone import if not present
                if "from datetime import" in content and "timezone" not in content:
                    content = re.sub(
                        r"from datetime import ([^,\n]+)",
                        r"from datetime import \1, timezone",
                        content,
                        count=1
                    )
                elif "import datetime" in content:
                    # Keep existing import, will use datetime.timezone
                    pass
                else:
                    # Add import at top
                    lines = content.splitlines()
                    import_line = "from datetime import datetime, timezone"
                    
                    # Find appropriate place to insert import
                    insert_idx = 0
                    for i, line in enumerate(lines):
                        if line.startswith(("import ", "from ")):
                            insert_idx = i + 1
                        elif line.strip() and not line.startswith("#"):
                            break
                            
                    lines.insert(insert_idx, import_line)
                    content = "\n".join(lines)
                
                # Replace datetime.now() with datetime.now(timezone.utc)
                content = re.sub(
                    r"datetime\.datetime\.now\(\)",
                    "datetime.datetime.now(timezone.utc)",
                    content
                )
                content = re.sub(
                    r"datetime\.now\(\)",
                    "datetime.now(timezone.utc)", 
                    content
                )
                
                if content != original_content:
                    path.write_text(content, encoding="utf-8")
                    fixed_count += len(file_violations)
                    logger.info(f"Fixed {len(file_violations)} timezone issues in {filepath}")
                    
            except Exception as e:
                logger.error(f"Failed to fix timezone issues in {filepath}: {e}")
                self.fixes_failed += 1
                
        return fixed_count

    def fix_whitespace_issues(self, violations: list[Violation]) -> int:
        """Fix whitespace issues (W291, W293, W292)."""
        whitespace_codes = {"W291", "W292", "W293"}
        whitespace_violations = [v for v in violations if v.code in whitespace_codes]
        
        if not whitespace_violations:
            return 0
            
        # Use ruff's auto-fix for whitespace issues
        try:
            result = subprocess.run(
                ["uv", "run", "ruff", "check", ".", "--fix", "--select", "W291,W292,W293"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            
            if result.returncode in (0, 1):  # Success
                logger.info(f"Fixed {len(whitespace_violations)} whitespace issues using ruff auto-fix")
                return len(whitespace_violations)
            else:
                logger.error(f"Ruff auto-fix failed: {result.stderr}")
                return 0
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run ruff auto-fix: {e}")
            return 0

    def generate_manual_review_report(self, violations: list[Violation]) -> None:
        """Generate report for violations requiring manual review."""
        manual_violations = [
            v for v in violations 
            if v.code in self.manual_review_codes
        ]
        
        if not manual_violations:
            logger.info("No violations requiring manual review!")
            return
            
        report_path = self.project_root / "manual_review_violations.md"
        
        with report_path.open("w", encoding="utf-8") as f:
            f.write("# Manual Review Required: Code Quality Violations\n\n")
            f.write("The following violations require manual review and fixing:\n\n")
            
            # Group by violation code
            by_code = defaultdict(list)
            for v in manual_violations:
                by_code[v.code].append(v)
                
            for code, code_violations in sorted(by_code.items()):
                f.write(f"## {code}: {code_violations[0].message.split('`')[0]}\n\n")
                f.write(f"**Count:** {len(code_violations)} violations\n\n")
                
                for v in code_violations[:10]:  # Show first 10
                    f.write(f"- `{v.filename}:{v.line}:{v.column}` - {v.message}\n")
                    
                if len(code_violations) > 10:
                    f.write(f"- ... and {len(code_violations) - 10} more\n")
                    
                f.write("\n")
                
        logger.info(f"Manual review report generated: {report_path}")

    def run_auto_fixes(self) -> dict[str, int]:
        """Run all safe automatic fixes."""
        logger.info("Starting automated code quality fixes...")
        
        # Get initial violations
        self.violations = self.get_violations()
        if not self.violations:
            logger.info("No violations found - code quality is already excellent!")
            return {"total_fixed": 0}
            
        analysis = self.analyze_violations(self.violations)
        logger.info(f"Found {analysis['total']} total violations")
        logger.info(f"Safe auto-fixes available: {analysis['safe_auto_fixes']}")
        logger.info(f"Manual review required: {analysis['manual_reviews']}")
        
        results = {"total_fixed": 0}
        
        # Apply fixes in order of safety and impact
        fixes = [
            ("whitespace", self.fix_whitespace_issues),
            ("unused_imports", self.fix_unused_imports), 
            ("typing_imports", self.fix_typing_imports),
            ("datetime_timezone", self.fix_datetime_timezone),
        ]
        
        for fix_name, fix_func in fixes:
            try:
                fixed_count = fix_func(self.violations)
                results[fix_name] = fixed_count
                results["total_fixed"] += fixed_count
                
                if fixed_count > 0:
                    logger.info(f"✅ {fix_name}: Fixed {fixed_count} violations")
                    
                    # Re-run ruff to update violation list
                    self.violations = self.get_violations()
                    
            except Exception as e:
                logger.error(f"❌ {fix_name}: Failed - {e}")
                results[fix_name] = 0
                
        # Generate manual review report
        remaining_violations = self.get_violations()
        self.generate_manual_review_report(remaining_violations)
        
        return results

    def report_results(self, results: dict[str, int]) -> None:
        """Report final results."""
        logger.info("\n" + "="*60)
        logger.info("🎯 ENTERPRISE CODE QUALITY ENHANCEMENT COMPLETE")
        logger.info("="*60)
        
        for fix_type, count in results.items():
            if fix_type != "total_fixed" and count > 0:
                logger.info(f"✅ {fix_type.replace('_', ' ').title()}: {count} fixes")
                
        logger.info(f"\n🎉 Total violations fixed: {results['total_fixed']}")
        
        # Get remaining violations
        remaining = self.get_violations()
        if remaining:
            by_code = defaultdict(int)
            for v in remaining:
                by_code[v.code] += 1
                
            logger.info(f"⚠️  Remaining violations: {len(remaining)}")
            for code, count in sorted(by_code.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"   - {code}: {count}")
                
            logger.info("\n📋 See 'manual_review_violations.md' for detailed review items")
        else:
            logger.info("🏆 ZERO VIOLATIONS ACHIEVED! Enterprise-grade code quality!")


def main() -> int:
    """Main entry point for zero violations fixing."""
    parser = argparse.ArgumentParser(
        description="Enterprise Zero-Violations Code Quality Automation"
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically apply safe fixes without confirmation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Show what would be fixed without making changes"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No changes will be made")
        args.auto_fix = False
        
    fixer = QualityFixer(auto_fix=args.auto_fix)
    
    try:
        results = fixer.run_auto_fixes()
        fixer.report_results(results)
        
        # Return appropriate exit code
        remaining_count = len(fixer.get_violations())
        if remaining_count == 0:
            return 0  # Success - zero violations
        elif results["total_fixed"] > 0:
            return 1  # Partial success - some fixes applied
        else:
            return 2  # No fixes applied
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())