#!/usr/bin/env python3
"""
Docker Configuration Validator for AI Docs Vector DB Hybrid Scraper
Validates Dockerfile configurations without requiring Docker build
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class DockerValidator:
    """Validates Docker configuration files for common issues."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    
    def validate_dockerfile(self, dockerfile_path: Path) -> Dict[str, List[str]]:
        """Validate a specific Dockerfile for common issues."""
        if not dockerfile_path.exists():
            self.errors.append(f"Dockerfile not found: {dockerfile_path}")
            return self._get_results()
        
        content = dockerfile_path.read_text()
        self._validate_dockerfile_content(content, dockerfile_path.name)
        return self._get_results()
    
    def _validate_dockerfile_content(self, content: str, filename: str) -> None:
        """Validate Dockerfile content for UV-specific issues."""
        lines = content.split('\n')
        
        # Check for UV installation
        if not any('uv' in line for line in lines):
            self.warnings.append(f"{filename}: No UV installation found")
        
        # Check for virtual environment creation
        venv_lines = [line for line in lines if 'uv venv' in line]
        if venv_lines:
            for line in venv_lines:
                if '--relocatable' not in line:
                    self.warnings.append(f"{filename}: UV venv should use --relocatable flag: {line.strip()}")
        
        # Check for proper UV environment variables
        uv_env_vars = ['UV_COMPILE_BYTECODE', 'UV_LINK_MODE', 'UV_PYTHON_INSTALL_DIR']
        for var in uv_env_vars:
            if var not in content:
                self.warnings.append(f"{filename}: Missing recommended UV environment variable: {var}")
        
        # Check for multi-stage build issues
        stages = re.findall(r'FROM .* AS (\w+)', content)
        if len(stages) > 1:
            self._validate_multistage_build(content, filename, stages)
        
        # Check for non-root user setup
        if 'useradd' in content:
            self._validate_user_setup(content, filename)
        
        # Check for proper file copying
        copy_lines = [line for line in lines if line.strip().startswith('COPY')]
        self._validate_copy_commands(copy_lines, filename)
        
        # Check for health check
        if 'HEALTHCHECK' not in content:
            self.warnings.append(f"{filename}: No health check defined")
        
        # Check for Python path setup
        if 'PYTHONPATH' not in content:
            self.warnings.append(f"{filename}: PYTHONPATH not set")
    
    def _validate_multistage_build(self, content: str, filename: str, stages: List[str]) -> None:
        """Validate multi-stage build specific issues."""
        # Check if Python interpreter is copied between stages
        if 'COPY --from=' in content:
            if '/root/.local/share/uv/python' not in content and '/opt/uv/python' not in content:
                self.warnings.append(f"{filename}: Multi-stage build may miss UV Python installation")
        
        # Check for proper virtual environment copying
        venv_copy = [line for line in content.split('\n') if 'COPY --from=' in line and 'venv' in line]
        if not venv_copy:
            self.errors.append(f"{filename}: Multi-stage build missing virtual environment copy")
    
    def _validate_user_setup(self, content: str, filename: str) -> None:
        """Validate non-root user setup."""
        lines = content.split('\n')
        
        # Check for proper chown operations
        chown_lines = [line for line in lines if 'chown' in line]
        required_paths = ['/app', 'venv']
        
        for path in required_paths:
            if not any(path in line for line in chown_lines):
                self.warnings.append(f"{filename}: Missing chown for {path}")
    
    def _validate_copy_commands(self, copy_lines: List[str], filename: str) -> None:
        """Validate COPY commands for proper ownership."""
        for line in copy_lines:
            if 'appuser' in line.lower() and '--chown=' not in line:
                self.warnings.append(f"{filename}: COPY command may need --chown flag: {line.strip()}")
    
    def validate_required_files(self) -> None:
        """Validate that required files exist for Docker build."""
        required_files = [
            'pyproject.toml',
            'uv.lock',
            'src/api/main.py',
            'src/__init__.py'
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                self.errors.append(f"Required file missing: {file_path}")
    
    def validate_dependencies(self) -> None:
        """Validate dependency configuration."""
        pyproject_path = self.project_root / 'pyproject.toml'
        uv_lock_path = self.project_root / 'uv.lock'
        
        if pyproject_path.exists():
            self.info.append("✓ pyproject.toml found")
        else:
            self.errors.append("pyproject.toml not found")
        
        if uv_lock_path.exists():
            self.info.append("✓ uv.lock found")
        else:
            self.warnings.append("uv.lock not found - run 'uv lock' to generate")
    
    def _get_results(self) -> Dict[str, List[str]]:
        """Get validation results."""
        return {
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy(),
            'info': self.info.copy()
        }
    
    def validate_all(self) -> Dict[str, Dict[str, List[str]]]:
        """Validate all Dockerfile configurations."""
        results = {}
        
        # Validate required files first
        self.validate_required_files()
        self.validate_dependencies()
        
        # Find and validate all Dockerfiles
        dockerfiles = [
            'Dockerfile',
            'Dockerfile.fixed',
            'Dockerfile.simple'
        ]
        
        for dockerfile in dockerfiles:
            dockerfile_path = self.project_root / dockerfile
            if dockerfile_path.exists():
                # Reset validation state for each file
                self.errors = []
                self.warnings = []
                self.info = []
                
                results[dockerfile] = self.validate_dockerfile(dockerfile_path)
        
        return results


def print_results(results: Dict[str, Dict[str, List[str]]]) -> None:
    """Print validation results in a readable format."""
    
    # Colors for terminal output
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color
    
    print(f"{BLUE}=== Docker Configuration Validation Results ==={NC}\n")
    
    total_errors = 0
    total_warnings = 0
    
    for dockerfile, result in results.items():
        print(f"{BLUE}📋 {dockerfile}{NC}")
        print("-" * 40)
        
        errors = result.get('errors', [])
        warnings = result.get('warnings', [])
        info = result.get('info', [])
        
        total_errors += len(errors)
        total_warnings += len(warnings)
        
        if errors:
            print(f"{RED}❌ Errors ({len(errors)}):{NC}")
            for error in errors:
                print(f"   {RED}• {error}{NC}")
        
        if warnings:
            print(f"{YELLOW}⚠️  Warnings ({len(warnings)}):{NC}")
            for warning in warnings:
                print(f"   {YELLOW}• {warning}{NC}")
        
        if info:
            print(f"{GREEN}✓ Info ({len(info)}):{NC}")
            for item in info:
                print(f"   {GREEN}• {item}{NC}")
        
        if not errors and not warnings:
            print(f"{GREEN}✅ No issues found{NC}")
        
        print()
    
    # Summary
    print(f"{BLUE}=== Summary ==={NC}")
    if total_errors == 0:
        print(f"{GREEN}✅ No critical errors found{NC}")
    else:
        print(f"{RED}❌ {total_errors} critical errors found{NC}")
    
    if total_warnings == 0:
        print(f"{GREEN}✅ No warnings{NC}")
    else:
        print(f"{YELLOW}⚠️  {total_warnings} warnings found{NC}")
    
    print(f"\n{BLUE}Next steps:{NC}")
    if total_errors > 0:
        print(f"{RED}• Fix critical errors before attempting Docker build{NC}")
    if total_warnings > 0:
        print(f"{YELLOW}• Review warnings for potential improvements{NC}")
    print(f"{GREEN}• Run ./scripts/test_docker_builds.sh to test actual builds{NC}")


def main():
    """Main validation function."""
    project_root = Path(__file__).parent.parent
    validator = DockerValidator(project_root)
    
    results = validator.validate_all()
    print_results(results)
    
    # Exit with error code if critical errors found
    total_errors = sum(len(result.get('errors', [])) for result in results.values())
    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()